"""Flask API server for the Indian Institutional Trading Agent.

Run:  python app.py
Open: http://localhost:5001
"""

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, date as dt_date
from zoneinfo import ZoneInfo

from dotenv import load_dotenv  # type: ignore[import-untyped]
load_dotenv()  # Load .env before anything else

from flask import Flask, jsonify, request, send_from_directory

from engine import INITIAL_CAPITAL, INDEX_TICKERS, DataFetcher, get_agent
from angelone_feed import get_feed

app = Flask(__name__, template_folder="templates", static_folder="static")

# ── In-memory agent log (ring buffer, last 500 lines) ────────────────────────
_agent_log: deque = deque(maxlen=500)
_log_lock   = threading.Lock()

# Level → colour tag used by the dashboard
_LEVEL_TAG = {
    "DEBUG":    "muted",
    "INFO":     "text",
    "WARNING":  "yellow",
    "ERROR":    "red",
    "CRITICAL": "red",
}

# Only capture logs from our own modules — silence library internals
_OUR_MODULES = {
    "engine", "app", "telegram_agent", "news_agent",
    "fundamental_analyzer", "angelone_feed", "__main__",
}

class _AgentLogHandler(logging.Handler):
    """Captures agent log records into the ring buffer, ignoring library noise."""
    def emit(self, record: logging.LogRecord):
        # Drop anything from yfinance, peewee, urllib, requests, etc.
        top = record.name.split(".")[0]
        if top not in _OUR_MODULES:
            return
        msg = self.format(record)
        entry = {
            "t":     datetime.now().strftime("%H:%M:%S"),
            "level": record.levelname,
            "tag":   _LEVEL_TAG.get(record.levelname, "text"),
            "msg":   msg,
        }
        with _log_lock:
            _agent_log.append(entry)

# Attach to root but only emit for our modules (handler filters internally)
_handler = _AgentLogHandler()
_handler.setFormatter(logging.Formatter("%(name)s — %(message)s"))
_handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(_handler)

# Silence noisy third-party loggers explicitly
for _noisy in ("yfinance", "peewee", "urllib3", "requests", "httpx",
               "asyncio", "telethon", "websocket", "charset_normalizer"):
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)


# ── Agent state ─────────────────────────────────────────────────────────────

# NSE market schedule (IST = UTC+5:30)
_IST          = ZoneInfo("Asia/Kolkata")
_MARKET_OPEN  = dt_time(9, 15)
_MARKET_CLOSE = dt_time(15, 30)

# ── NSE holiday calendar (via pandas_market_calendars, cached per process) ────
_nse_calendar = None
_nse_calendar_loaded = False  # True once we've attempted to load

def _get_nse_calendar():
    """Lazy-load the NSE exchange calendar using pandas_market_calendars."""
    global _nse_calendar, _nse_calendar_loaded  # noqa: PLW0603
    if not _nse_calendar_loaded:
        _nse_calendar_loaded = True
        try:
            import pandas_market_calendars as mcal  # type: ignore[import-untyped]
            _nse_calendar = mcal.get_calendar("NSE")
            logging.getLogger(__name__).info("[Scheduler] NSE holiday calendar loaded (pandas_market_calendars)")
        except Exception as exc:
            logging.getLogger(__name__).warning(
                f"[Scheduler] pandas_market_calendars unavailable — holiday check disabled: {exc}"
            )
    return _nse_calendar


def _is_nse_holiday(date_obj) -> bool:
    """Return True if date_obj is an NSE trading holiday (not a session day)."""
    cal = _get_nse_calendar()
    if cal is None:
        return False  # can't check, assume not a holiday
    try:
        date_str = date_obj.strftime("%Y-%m-%d") if hasattr(date_obj, "strftime") else str(date_obj)
        schedule = cal.schedule(start_date=date_str, end_date=date_str)
        return schedule.empty  # empty schedule = holiday or weekend
    except Exception:
        return False


def _ist_now() -> datetime:
    """Return the current time in IST."""
    return datetime.now(_IST)


def _market_open() -> bool:
    """Return True if NSE is currently open (weekday, 9:15–15:30 IST, non-holiday)."""
    now = _ist_now()
    if now.weekday() >= 5:
        return False  # weekend
    if not (_MARKET_OPEN <= now.time() <= _MARKET_CLOSE):
        return False  # outside trading hours
    if _is_nse_holiday(now.date()):
        return False  # NSE declared holiday
    return True


def _secs_until_next_open() -> float:
    """Seconds until the next NSE trading day opening bell."""
    from datetime import timedelta
    now = _ist_now()
    candidate = now.replace(hour=9, minute=15, second=0, microsecond=0)
    if now.time() >= _MARKET_OPEN:
        candidate += timedelta(days=1)
    # Skip weekends AND NSE holidays
    while candidate.weekday() >= 5 or _is_nse_holiday(candidate.date()):
        candidate += timedelta(days=1)
    return max(60.0, (candidate - now).total_seconds())


@dataclass
class _AgentState:
    """Mutable runtime state for the background agent loop."""

    running: bool = False
    last_cycle: dict = field(default_factory=dict)
    auto_interval: int = 900  # seconds between intra-day cycles
    closing_report_dates: set = field(default_factory=set)


_state = _AgentState()

# ── Dashboard cache — stale-while-revalidate pattern ─────────────────────────
# The API ALWAYS returns instantly. If cache is warm (even stale), return it
# immediately and kick off a background refresh. If cold, build synchronously
# once, then all future calls are instant.
_dashboard_cache: dict = {"ts": 0, "data": None, "refreshing": False}
_DASHBOARD_CACHE_TTL = 45   # seconds before background refresh kicks in


_dash_logger = logging.getLogger("app")

def _refresh_dashboard_cache():
    """Rebuild dashboard data in a background thread — never blocks requests."""
    if _dashboard_cache["refreshing"]:
        return
    _dashboard_cache["refreshing"] = True
    def _do():
        try:
            data = get_agent().get_dashboard_data()
            data["agent_running"]    = _state.running
            data["last_cycle"]       = _state.last_cycle
            data["auto_interval_s"]  = _state.auto_interval
            _dashboard_cache["data"] = data
            _dashboard_cache["ts"]   = time.time()
        except Exception as exc:
            _dash_logger.debug(f"Dashboard cache refresh error: {exc}")
        finally:
            _dashboard_cache["refreshing"] = False
    threading.Thread(target=_do, daemon=True, name="dash-cache-refresh").start()


# ── Dashboard ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the single-page dashboard."""
    return send_from_directory("templates", "dashboard.html")


# ── Data endpoints ───────────────────────────────────────────────────────────

@app.route("/api/dashboard")
def api_dashboard():
    """
    Stale-while-revalidate: always returns instantly.
    • Cache hit (fresh): return immediately
    • Cache hit (stale): return stale data + trigger background refresh
    • Cache cold (first request): build synchronously once, cache it
    """
    try:
        now  = time.time()
        data = _dashboard_cache["data"]

        if data is not None:
            # Always return cached data immediately
            resp = dict(data)
            resp["agent_running"]   = _state.running
            resp["last_cycle"]      = _state.last_cycle
            resp["auto_interval_s"] = _state.auto_interval
            resp["_cached"] = True
            # Trigger background refresh if stale
            if now - _dashboard_cache["ts"] > _DASHBOARD_CACHE_TTL:
                _refresh_dashboard_cache()
            return jsonify({"ok": True, "data": resp})

        # Cold start — build once synchronously, then all future calls are cached
        data = get_agent().get_dashboard_data()
        data["agent_running"]   = _state.running
        data["last_cycle"]      = _state.last_cycle
        data["auto_interval_s"] = _state.auto_interval
        _dashboard_cache["data"] = data
        _dashboard_cache["ts"]   = now
        return jsonify({"ok": True, "data": data})

    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/portfolio")
def api_portfolio():
    """Return current portfolio summary (value, cash, P&L)."""
    agent = get_agent()
    return jsonify({
        "ok": True,
        "portfolio": {
            "total_value":    round(agent.portfolio.get_total_value(), 2),
            "cash":           round(agent.portfolio.state["cash"], 2),
            "realised_pnl":   round(agent.portfolio.state.get("realised_pnl", 0), 2),
            "unrealised_pnl": round(agent.portfolio.get_unrealised_pnl(), 2),
        },
    })


@app.route("/api/positions")
def api_positions():
    """Return all open positions with live P&L."""
    return jsonify({"ok": True, "positions": get_agent().portfolio.get_positions_display()})


@app.route("/api/trades")
def api_trades():
    """Return full trade log, newest first."""
    from engine import TRADE_LOG_FILE  # pylint: disable=import-outside-toplevel

    trades = []
    if TRADE_LOG_FILE.exists():
        with open(TRADE_LOG_FILE, encoding="utf-8") as fh:
            trades = json.load(fh)
    return jsonify({"ok": True, "trades": list(reversed(trades))})


@app.route("/api/pnl_calendar")
def api_pnl_calendar():
    """
    Returns daily P&L aggregated by date for the calendar view.
    Response: { days: { "2026-05-27": { pnl, trades, wins, win_rate } } }
    """
    from engine import TRADE_LOG_FILE  # pylint: disable=import-outside-toplevel
    from collections import defaultdict

    trades = []
    if TRADE_LOG_FILE.exists():
        with open(TRADE_LOG_FILE, encoding="utf-8") as fh:
            trades = json.load(fh)

    # Only look at SELL trades (realised P&L)
    days: dict = defaultdict(lambda: {"pnl": 0.0, "trades": 0, "wins": 0})
    for t in trades:
        if t.get("action") != "SELL" or t.get("pnl") is None:
            continue
        day = (t.get("time") or "")[:10]   # "2026-05-27"
        if not day:
            continue
        pnl = float(t["pnl"])
        days[day]["pnl"]    = round(days[day]["pnl"] + pnl, 2)
        days[day]["trades"] += 1
        if pnl > 0:
            days[day]["wins"] += 1

    # Compute win rate per day
    result = {}
    for day, d in days.items():
        result[day] = {
            "pnl":      d["pnl"],
            "trades":   d["trades"],
            "wins":     d["wins"],
            "win_rate": round(d["wins"] / d["trades"] * 100) if d["trades"] else 0,
        }

    # Summary stats
    all_pnl = [v["pnl"] for v in result.values()]
    total_days = len(result)
    green_days = sum(1 for p in all_pnl if p > 0)
    red_days   = sum(1 for p in all_pnl if p < 0)

    return jsonify({
        "ok":         True,
        "days":       result,
        "summary": {
            "total_days":  total_days,
            "green_days":  green_days,
            "red_days":    red_days,
            "best_day":    max(all_pnl) if all_pnl else 0,
            "worst_day":   min(all_pnl) if all_pnl else 0,
            "total_pnl":   round(sum(all_pnl), 2),
        },
    })


@app.route("/api/signals")
def api_signals():
    """Return the latest generated signals from all strategies."""
    from engine import SIGNALS_FILE  # pylint: disable=import-outside-toplevel

    if SIGNALS_FILE.exists():
        with open(SIGNALS_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        return jsonify({"ok": True, **data})
    return jsonify({"ok": True, "signals": [], "updated_at": ""})


_market_overview_cache: dict = {}
_market_overview_ts: float = 0.0
_MARKET_OVERVIEW_TTL = 15  # seconds between yfinance refreshes (fast enough for ~live feel)


def _fetch_index_price(ticker: str) -> tuple[float, float]:
    """
    Return (current_price, pct_change_vs_prev_close) for an index ticker.
    Uses yf.Ticker.fast_info which gives the *live delayed* price (15-min lag)
    rather than the previous day's EOD close that daily OHLCV bars return.
    Falls back to 5d daily bars if fast_info fails.
    """
    import yfinance as yf  # pylint: disable=import-outside-toplevel
    try:
        fi = yf.Ticker(ticker).fast_info
        price = float(fi.last_price or 0)
        prev  = float(fi.previous_close or 0)
        if price > 0 and prev > 0:
            return price, (price / prev - 1) * 100
    except Exception:
        pass
    # Fallback to daily bars (gives previous close during market hours)
    try:
        df = yf.download(ticker, period="5d", interval="1d",
                         auto_adjust=True, progress=False)
        if not df.empty:
            price = float(df["Close"].iloc[-1])
            prev  = float(df["Close"].iloc[-2]) if len(df) > 1 else price
            return price, (price / prev - 1) * 100
    except Exception:
        pass
    return 0.0, 0.0


@app.route("/api/market_overview")
def api_market_overview():
    """Return latest price and % change for Nifty 50 and Bank Nifty.

    Priority:
      1. Angel One SmartAPI live WebSocket feed (if configured + connected)
      2. yfinance fast_info — delayed ~15 min but updates *intraday*
      3. Daily OHLCV bar fallback (gives previous EOD close)
    Result cached for 60 s to avoid hammering yfinance.
    """
    global _market_overview_cache, _market_overview_ts  # noqa: PLW0603

    feed = get_feed()
    now  = time.time()

    # ── 1. Angel One live feed (sub-second, no cache needed) ──────────────
    now_ist_str = _ist_now().strftime("%H:%M:%S IST")
    if feed.is_connected():
        live = feed.get_all_prices()
        results: dict = {}
        for name, ticker in INDEX_TICKERS.items():
            ltp = live.get(ticker)
            if ltp:
                results[name] = {
                    "price":       round(ltp, 2),
                    "chg_pct":     round(feed.get_change(ticker) or 0, 2),
                    "ticker":      ticker,
                    "source":      "live",
                    "last_updated": now_ist_str,
                }
        if results:
            return jsonify({"ok": True, "indices": results})

    # ── 2. yfinance fast_info (60-second TTL cache) ───────────────────────
    if now - _market_overview_ts < _MARKET_OVERVIEW_TTL and _market_overview_cache:
        return jsonify({"ok": True, "indices": _market_overview_cache, "source": "cache"})

    results = {}
    for name, ticker in INDEX_TICKERS.items():
        price, chg_pct = _fetch_index_price(ticker)
        if price > 0:
            results[name] = {
                "price":        round(price, 2),
                "chg_pct":      round(chg_pct, 2),
                "ticker":       ticker,
                "source":       "delayed",
                "last_updated": now_ist_str,
            }

    _market_overview_cache = results
    _market_overview_ts    = now
    return jsonify({"ok": True, "indices": results})


# ── Agent controls ───────────────────────────────────────────────────────────

@app.route("/api/run_cycle", methods=["POST"])
def api_run_cycle():
    """Trigger a full agent cycle (fetch → signal → execute → stop-check)."""
    if _state.running:
        return jsonify({"ok": False, "error": "Agent already running"})
    _state.running = True
    try:
        summary = get_agent().run_cycle()
        _state.last_cycle = summary
        return jsonify({"ok": True, "summary": summary})
    except (ValueError, RuntimeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    finally:
        _state.running = False


@app.route("/api/manual_buy", methods=["POST"])
def api_manual_buy():
    """Manually buy a stock at current LTP using standard position sizing."""
    body = request.get_json(force=True)
    ticker = body.get("ticker", "").upper()
    if not ticker.endswith(".NS"):
        ticker += ".NS"
    price = DataFetcher.get_current_price(ticker)
    if price <= 0:
        return jsonify({"ok": False, "error": f"Could not fetch price for {ticker}"}), 400
    trade = get_agent().portfolio.execute_buy(ticker, price, "MANUAL", "Manual buy")
    if trade:
        return jsonify({"ok": True, "trade": trade})
    return jsonify({
        "ok": False,
        "error": "Could not execute buy (check capital / position limits)",
    }), 400


@app.route("/api/manual_sell", methods=["POST"])
def api_manual_sell():
    """Manually exit an entire position at current LTP."""
    body = request.get_json(force=True)
    ticker = body.get("ticker", "")
    price = DataFetcher.get_current_price(ticker)
    trade = get_agent().portfolio.execute_sell(ticker, price, "MANUAL")
    if trade:
        return jsonify({"ok": True, "trade": trade})
    return jsonify({"ok": False, "error": f"No position in {ticker}"}), 400


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Reset portfolio to initial ₹10,00,000 and clear all trade history."""
    get_agent().portfolio.reset()
    DataFetcher.clear_cache()
    return jsonify({"ok": True, "message": "Portfolio reset to ₹10,00,000"})


@app.route("/api/set_interval", methods=["POST"])
def api_set_interval():
    """Set the auto-run interval in seconds (minimum 60)."""
    body = request.get_json(force=True)
    secs = int(body.get("seconds", 900))
    _state.auto_interval = max(60, secs)
    return jsonify({"ok": True, "interval": _state.auto_interval})




@app.route("/api/live_prices")
def api_live_prices():
    """Return live LTP and % change for all tracked instruments from Angel One feed."""
    feed = get_feed()
    prices  = feed.get_all_prices()
    changes = feed.get_all_changes()

    # Build a unified response
    instruments = {}
    all_tickers = list(prices.keys()) or []
    for ticker in all_tickers:
        instruments[ticker] = {
            "ltp":     prices.get(ticker, 0),
            "chg_pct": changes.get(ticker, 0),
        }

    return jsonify({
        "ok":        True,
        "connected": feed.is_connected(),
        "count":     len(instruments),
        "prices":    instruments,
        "ts":        datetime.now(_IST).isoformat(),
    })


@app.route("/api/feed_status")
def api_feed_status():
    """Return Angel One live feed connection status."""
    feed = get_feed()
    return jsonify({
        "ok":           True,
        "connected":    feed.is_connected(),
        "configured":   feed.is_configured(),
        "price_count":  len(feed.get_all_prices()),
    })


def _build_ohlcv_response(df, period: str, ticker: str = "") -> dict:
    """Shared helper: convert a yfinance DataFrame to an API response dict."""
    intraday = period == "1d"
    records = []
    for ts, row in df.iterrows():
        # For intraday use HH:MM label; for daily use YYYY-MM-DD
        label = str(ts)[11:16] if intraday else str(ts)[:10]
        records.append({
            "date":   label,
            "open":   round(float(row["Open"]),  2),
            "high":   round(float(row["High"]),  2),
            "low":    round(float(row["Low"]),   2),
            "close":  round(float(row["Close"]), 2),
            "volume": int(row["Volume"]) if "Volume" in df.columns else 0,
        })
    latest     = records[-1]["close"] if records else 0
    first      = records[0]["close"]  if records else 0
    change_pct = round((latest / first - 1) * 100, 2) if first else 0
    return {"ok": True, "ticker": ticker, "data": records,
            "change_pct": change_pct, "period": period, "intraday": intraday}


@app.route("/api/nifty_chart")
def api_nifty_chart():
    """Return Nifty 50 OHLCV history for the requested period."""
    period    = request.args.get("period", "1y")
    intraday  = period == "1d"
    valid_day = {"1w": "5d", "1m": "1mo", "3m": "3mo", "6m": "6mo", "1y": "1y", "5y": "5y"}
    yf_period = "1d"    if intraday else valid_day.get(period, "1y")
    yf_itvl   = "5m"   if intraday else "1d"
    if intraday:
        DataFetcher._cache.pop(f"^NSEI_{yf_period}_{yf_itvl}", None)
    df = DataFetcher.fetch("^NSEI", period=yf_period, interval=yf_itvl)
    if df.empty:
        return jsonify({"ok": False, "error": "No data"}), 500
    return jsonify(_build_ohlcv_response(df, period, "^NSEI"))


@app.route("/api/chart/<path:ticker>")
def api_chart(ticker: str):
    """Return OHLCV history for any ticker with optional period.

    period values: 1d (5-min intraday) | 1w | 1m | 3m | 6m | 1y | 5y
    """
    period    = request.args.get("period", "3m")
    intraday  = period == "1d"
    valid_day = {"1w": "5d", "1m": "1mo", "3m": "3mo", "6m": "6mo", "1y": "1y", "5y": "5y"}
    yf_period = "1d"  if intraday else valid_day.get(period, "3mo")
    yf_itvl   = "5m"  if intraday else "1d"

    # Always bypass cache for intraday so we get the freshest bars
    if intraday:
        DataFetcher._cache.pop(f"{ticker}_{yf_period}_{yf_itvl}", None)

    df = DataFetcher.fetch(ticker, period=yf_period, interval=yf_itvl)
    if df.empty:
        return jsonify({"ok": False, "error": f"No data for {ticker}"}), 500
    return jsonify(_build_ohlcv_response(df, period, ticker))

# ── Background loop ──────────────────────────────────────────────────────────



@app.route("/api/market_status")
def api_market_status():
    """Return whether NSE is currently open plus time to open/close."""
    now      = _ist_now()
    is_open  = _market_open()
    holiday  = _is_nse_holiday(now.date())
    if is_open:
        close_dt  = now.replace(hour=15, minute=30, second=0, microsecond=0)
        secs_left = max(0, (close_dt - now).total_seconds())
        label = f"Closes in {int(secs_left//3600):02d}:{int((secs_left%3600)//60):02d}"
    elif holiday:
        secs_left = _secs_until_next_open()
        label = f"Holiday — opens in {int(secs_left//3600):02d}:{int((secs_left%3600)//60):02d}"
    else:
        secs_left = _secs_until_next_open()
        label = f"Opens in {int(secs_left//3600):02d}:{int((secs_left%3600)//60):02d}"
    return jsonify({
        "ok":       True,
        "is_open":  is_open,
        "holiday":  holiday,
        "time_ist": now.strftime("%H:%M IST"),
        "label":    label,
        "weekday":  now.strftime("%A"),
    })


@app.route("/api/report")
def api_report():
    """Return the latest closing report."""
    from pathlib import Path  # pylint: disable=import-outside-toplevel
    p = Path("data/closing_report.json")
    if p.exists():
        with open(p, encoding="utf-8") as fh:
            import json as _j  # pylint: disable=import-outside-toplevel
            return jsonify({"ok": True, "report": _j.load(fh)})
    return jsonify({"ok": True, "report": None})

# ── Telegram Intelligence endpoints ─────────────────────────────────────────

@app.route("/api/telegram/status")
def api_telegram_status():
    """Return Telegram agent connection status and aggregate stats."""
    from telegram_agent import get_telegram_agent  # pylint: disable=import-outside-toplevel
    return jsonify({"ok": True, **get_telegram_agent().get_stats()})


@app.route("/api/telegram/groups")
def api_telegram_groups():
    """Return all tracked groups sorted by score (best first)."""
    from telegram_agent import get_telegram_agent  # pylint: disable=import-outside-toplevel
    agent  = get_telegram_agent()
    status = request.args.get("status")          # filter: active | probation | dropped
    groups = agent.get_groups()
    if status:
        groups = [g for g in groups if g.get("status") == status]
    return jsonify({"ok": True, "groups": groups, "total": len(groups)})


@app.route("/api/telegram/messages")
def api_telegram_messages():
    """Return recent raw messages from all monitored groups."""
    from telegram_agent import get_telegram_agent  # pylint: disable=import-outside-toplevel
    agent    = get_telegram_agent()
    limit    = int(request.args.get("limit", 50))
    group_id = request.args.get("group_id")
    messages = agent.get_messages(group_id=group_id, limit=limit)
    return jsonify({"ok": True, "messages": messages, "total": len(messages)})


@app.route("/api/telegram/signals")
def api_telegram_signals():
    """Return recent Telegram signals with parsed fields and outcomes."""
    from telegram_agent import get_telegram_agent  # pylint: disable=import-outside-toplevel
    limit   = int(request.args.get("limit", 50))
    status  = request.args.get("status")         # filter: pending | hit_target | hit_sl | expired
    signals = get_telegram_agent().get_signals(limit=max(limit, 200))
    if status:
        signals = [s for s in signals if s.get("status") == status]
    return jsonify({"ok": True, "signals": signals[:limit], "total": len(signals)})


@app.route("/api/telegram/reparse", methods=["POST"])
def api_telegram_reparse():
    """
    Re-parse all stored signals from their raw_text using the current parser.
    Call once after a parser upgrade to fix historical bad targets/entries.
    """
    from telegram_agent import get_telegram_agent, SignalParser  # pylint: disable=import-outside-toplevel
    agent  = get_telegram_agent()
    fixed  = 0
    kept   = 0
    parser = SignalParser()

    for i, sig in enumerate(agent._signals):
        raw = sig.get("raw_text", "")
        if not raw:
            kept += 1
            continue
        new_parsed = parser.parse(
            raw,
            sig.get("group_id", 0),
            sig.get("group_title", ""),
            sig.get("message_id", 0),
        )
        if new_parsed:
            # Preserve identity fields and outcome — only update the parsed block
            sig["parsed"] = new_parsed["parsed"]
            fixed += 1
        else:
            kept += 1

    agent._save_signals()
    return jsonify({"ok": True, "reparsed": fixed, "kept": kept,
                    "total": len(agent._signals)})


@app.route("/api/telegram/add", methods=["POST"])
def api_telegram_add():
    """Manually add a Telegram group/channel by @username or invite link."""
    from telegram_agent import get_telegram_agent  # pylint: disable=import-outside-toplevel
    body       = request.get_json(force=True)
    identifier = (body.get("identifier") or "").strip()
    if not identifier:
        return jsonify({"ok": False, "error": "identifier is required"}), 400
    result = get_telegram_agent().add_group_manual(identifier)
    return jsonify(result), (200 if result["ok"] else 400)


@app.route("/api/telegram/discover", methods=["POST"])
def api_telegram_discover():
    """Trigger a manual discovery cycle (searches Telegram for new signal groups)."""
    from telegram_agent import get_telegram_agent  # pylint: disable=import-outside-toplevel
    result = get_telegram_agent().trigger_discovery()
    return jsonify(result), (200 if result["ok"] else 400)


# ── News & Commodity Intelligence endpoints ──────────────────────────────────

@app.route("/api/news/headlines")
def api_news_headlines():
    """Return latest news headlines with sentiment and ticker mentions."""
    try:
        from news_agent import get_news_agent  # pylint: disable=import-outside-toplevel
        n = int(request.args.get("n", 30))
        headlines = get_news_agent().latest_headlines(n=n)
        return jsonify({"ok": True, "headlines": headlines})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/news/commodities")
def api_news_commodities():
    """Return current commodity prices and % moves."""
    try:
        from news_agent import get_news_agent  # pylint: disable=import-outside-toplevel
        data = get_news_agent().get_commodity_data()
        return jsonify({"ok": True, "commodities": data})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/news/signals")
def api_news_signals():
    """Return all news + commodity signals."""
    try:
        from news_agent import get_news_agent  # pylint: disable=import-outside-toplevel
        signals = get_news_agent().get_all_signals()
        return jsonify({"ok": True, "signals": signals, "count": len(signals)})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


# ── Fundamental Analysis endpoints ───────────────────────────────────────────

@app.route("/api/fundamentals/<path:ticker>")
def api_fundamentals_ticker(ticker: str):
    """Return fundamental score and metrics for a single ticker."""
    try:
        from fundamental_analyzer import get_analyzer  # pylint: disable=import-outside-toplevel
        result = get_analyzer().get_score(ticker)
        return jsonify({"ok": True, "ticker": ticker, **result})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/fundamentals")
def api_fundamentals_all():
    """Return all cached fundamental scores."""
    try:
        from fundamental_analyzer import get_analyzer  # pylint: disable=import-outside-toplevel
        scores = get_analyzer().get_all_scores()
        return jsonify({"ok": True, "scores": scores, "count": len(scores)})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/screener/<path:ticker>")
def api_screener_ticker(ticker: str):
    """Return screener.in data for a single ticker (live scrape, cached 24h)."""
    try:
        from fundamental_analyzer import get_screener  # pylint: disable=import-outside-toplevel
        data = get_screener().scrape(ticker)
        return jsonify({"ok": True, "ticker": ticker, "data": data})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/screener")
def api_screener_all():
    """Return all cached screener.in data."""
    try:
        from fundamental_analyzer import get_screener  # pylint: disable=import-outside-toplevel
        return jsonify({"ok": True, "data": get_screener().get_all_cached()})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/logs")
def api_logs():
    """Return recent agent log lines.  ?since=HH:MM:SS returns only newer entries."""
    since = request.args.get("since", "")
    n     = int(request.args.get("n", 200))
    with _log_lock:
        entries = list(_agent_log)
    if since:
        entries = [e for e in entries if e["t"] > since]
    return jsonify({"ok": True, "logs": entries[-n:], "total": len(entries)})


@app.route("/api/learning")
def api_learning():
    """Return the current learning engine state: strategy weights, threshold, adjustment log."""
    try:
        from learning_engine import get_learning_engine  # pylint: disable=import-outside-toplevel
        state = get_learning_engine().get_state_snapshot()
        return jsonify({"ok": True, **state})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/notifier/status")
def api_notifier_status():
    """Return notification channel configuration status."""
    try:
        from notifier import get_notifier  # pylint: disable=import-outside-toplevel
        return jsonify({"ok": True, **get_notifier().get_status()})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/notifier/test", methods=["POST"])
def api_notifier_test():
    """Send a test alert to all configured channels."""
    try:
        from notifier import get_notifier  # pylint: disable=import-outside-toplevel
        results = get_notifier().send_test()
        return jsonify({"ok": True, "results": results})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/risk")
def api_risk():
    """Return current dynamic risk snapshot: macro score, drawdown, event calendar."""
    try:
        from risk_manager import get_risk_manager  # pylint: disable=import-outside-toplevel
        from engine import get_engine              # pylint: disable=import-outside-toplevel
        eng = get_engine()
        port_val = eng.portfolio.get_total_value() if eng else 1_000_000
        status = get_risk_manager().full_status(port_val)
        return jsonify({"ok": True, **status})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/learning/reset", methods=["POST"])
def api_learning_reset():
    """Reset all learning state to defaults (use with caution)."""
    try:
        from learning_engine import get_learning_engine  # pylint: disable=import-outside-toplevel
        get_learning_engine().reset()
        return jsonify({"ok": True, "message": "Learning state reset to defaults"})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/universe")
def api_universe():
    """Return the current full NSE trading universe (or force-refresh it)."""
    try:
        from engine import load_nse_universe, PENNY_UNIVERSE  # pylint: disable=import-outside-toplevel
        force = request.args.get("refresh", "").lower() in ("1", "true", "yes")
        tickers = load_nse_universe(force_refresh=force)
        return jsonify({"ok": True, "count": len(tickers), "tickers": tickers})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/fundamentals/top")
def api_fundamentals_top():
    """Return top N stocks by fundamental quality from the full universe."""
    try:
        from fundamental_analyzer import get_analyzer  # pylint: disable=import-outside-toplevel
        from engine import load_nse_universe, PENNY_UNIVERSE  # pylint: disable=import-outside-toplevel
        n      = int(request.args.get("n", 15))
        all_t  = load_nse_universe() + PENNY_UNIVERSE
        result = get_analyzer().top_stocks(all_t, n=n)
        return jsonify({"ok": True, "top": result})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


# ── Penny Stock endpoints ─────────────────────────────────────────────────────

@app.route("/api/penny/universe")
def api_penny_universe():
    """Return the penny/small-cap universe with latest prices and fundamental scores."""
    try:
        from engine import PENNY_UNIVERSE, DataFetcher  # pylint: disable=import-outside-toplevel
        from fundamental_analyzer import get_analyzer   # pylint: disable=import-outside-toplevel

        analyzer = get_analyzer()
        rows = []
        for ticker in PENNY_UNIVERSE:
            price = DataFetcher.get_current_price(ticker)
            cached = analyzer._cache.get(ticker, {})
            rows.append({
                "ticker":     ticker,
                "price":      round(price, 2),
                "fund_score": cached.get("score"),
                "fund_grade": cached.get("grade", "-"),
                "metrics":    cached.get("metrics", {}),
            })
        # Sort by fundamental score desc, with unscored at bottom
        rows.sort(key=lambda r: r["fund_score"] or -1, reverse=True)
        return jsonify({"ok": True, "universe": rows, "count": len(rows)})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/penny/positions")
def api_penny_positions():
    """Return open positions that are in the penny universe."""
    try:
        from engine import PENNY_UNIVERSE  # pylint: disable=import-outside-toplevel
        penny_set  = set(PENNY_UNIVERSE)
        positions  = get_agent().portfolio.get_positions_display()
        penny_pos  = [p for p in positions if p["ticker"] in penny_set]
        return jsonify({"ok": True, "positions": penny_pos, "count": len(penny_pos)})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


def _run_cycle_safe() -> None:
    """Run one agent cycle, guarded against concurrent calls."""
    if _state.running:
        return
    _state.running = True
    try:
        summary = get_agent().run_cycle()
        _state.last_cycle = summary
    except (ValueError, RuntimeError) as exc:
        print(f"[BG] Error in cycle: {exc}")
    finally:
        _state.running = False


def _generate_closing_report() -> None:
    """Persist an end-of-day summary to data/closing_report.json."""
    import json as _json
    from pathlib import Path
    from engine import TRADE_LOG_FILE  # pylint: disable=import-outside-toplevel

    agent  = get_agent()
    today  = _ist_now().date().isoformat()
    trades = []
    if TRADE_LOG_FILE.exists():
        with open(TRADE_LOG_FILE, encoding="utf-8") as fh:
            trades = _json.load(fh)

    today_trades = [t for t in trades if t.get("time", "").startswith(today)]
    day_pnl      = sum(t.get("pnl") or 0 for t in today_trades if t["action"] == "SELL")
    port_value   = agent.portfolio.get_total_value()
    positions    = agent.portfolio.get_positions_display()

    report = {
        "date":           today,
        "generated_at":   _ist_now().isoformat(),
        "portfolio_value": round(port_value, 2),
        "day_pnl":        round(day_pnl, 2),
        "day_pnl_pct":    round(day_pnl / 1_000_000 * 100, 4),
        "trades_today":   today_trades,
        "trades_count":   len(today_trades),
        "open_positions": len(positions),
        "top_gainers":    sorted(positions, key=lambda p: p["pnl_pct"], reverse=True)[:3],
        "top_losers":     sorted(positions, key=lambda p: p["pnl_pct"])[:3],
        "last_cycle":     _state.last_cycle,
    }
    Path("data/closing_report.json").write_text(
        _json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"[CLOSE] Closing report saved for {today}  day_pnl=₹{day_pnl:,.0f}")

    # ── Full closing report notification ─────────────────────────────────────
    try:
        from notifier import get_notifier  # pylint: disable=import-outside-toplevel
        today_sells  = [t for t in today_trades if t.get("action") == "SELL"]
        wins         = [t for t in today_sells if (t.get("pnl") or 0) > 0]
        win_rate_val = (len(wins) / len(today_sells)) if today_sells else None
        get_notifier().send_closing_report(
            date=today,
            day_pnl=day_pnl,
            portfolio_value=port_value,
            initial_capital=INITIAL_CAPITAL,
            today_trades=today_trades,
            open_positions=positions,
            win_rate=win_rate_val,
        )
    except Exception as _ne:
        print(f"[CLOSE] Notifier closing report error: {_ne}")


# ═══════════════════════════════════════════════════════════════════════════
# FOREX API ROUTES  (/api/forex/*)
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/api/news/premarket", methods=["POST"])
def api_news_premarket():
    """Manually trigger the pre-market briefing (news + events + commodities)."""
    try:
        from news_agent import send_premarket_briefing  # pylint: disable=import-outside-toplevel
        send_premarket_briefing()
        return jsonify({"ok": True, "message": "Pre-market briefing triggered — check Telegram in ~15 seconds"})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/digest/send", methods=["POST"])
def api_digest_send():
    """Manually trigger the morning digest (for testing or on-demand use)."""
    try:
        from notifier import send_morning_digest  # pylint: disable=import-outside-toplevel
        send_morning_digest()
        return jsonify({"ok": True, "message": "Morning digest triggered — check your Telegram group in ~30 seconds"})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/forex/dashboard")
def api_forex_dashboard():
    """Portfolio summary for the forex agent."""
    try:
        from forex_engine import get_forex_agent, is_forex_market_open, current_session  # noqa
        agent   = get_forex_agent()
        summary = agent.portfolio.get_summary()
        session = current_session()
        return jsonify({"ok": True, "market_open": is_forex_market_open(),
                        "session": session, **summary})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/forex/positions")
def api_forex_positions():
    try:
        from forex_engine import get_forex_agent  # noqa
        positions = get_forex_agent().portfolio.get_positions()
        return jsonify({"ok": True, "positions": positions})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/forex/signals")
def api_forex_signals():
    limit = int(request.args.get("limit", 50))
    try:
        from forex_engine import get_forex_agent  # noqa
        signals = get_forex_agent().get_signals(limit=limit)
        return jsonify({"ok": True, "signals": signals})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/forex/trades")
def api_forex_trades():
    limit = int(request.args.get("limit", 100))
    try:
        from forex_engine import get_forex_agent  # noqa
        trades = list(reversed(get_forex_agent().portfolio._load_trades()[-limit:]))
        return jsonify({"ok": True, "trades": trades})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/forex/pairs")
def api_forex_pairs():
    """Live prices for all tracked forex pairs."""
    try:
        from forex_engine import FOREX_PAIRS, get_price  # noqa
        out = []
        for pair, meta in FOREX_PAIRS.items():
            price = get_price(pair)
            out.append({"pair": pair, "name": meta["name"],
                        "group": meta["group"], "price": price})
        return jsonify({"ok": True, "pairs": out})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/forex/run", methods=["POST"])
def api_forex_run():
    """Manually trigger a forex agent cycle."""
    try:
        from forex_engine import get_forex_agent  # noqa
        result = get_forex_agent().run_cycle()
        return jsonify({"ok": True, **result})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/forex/close", methods=["POST"])
def api_forex_close():
    """Manually close a forex position."""
    try:
        from forex_engine import get_forex_agent, get_price  # noqa
        data = request.get_json(force=True)
        pair = data.get("pair", "").strip()
        if not pair:
            return jsonify({"ok": False, "error": "pair required"}), 400
        price = get_price(pair)
        if not price:
            return jsonify({"ok": False, "error": "Could not fetch current price"}), 400
        trade = get_forex_agent().portfolio.execute_close(pair, price, reason="MANUAL")
        return jsonify({"ok": bool(trade), "trade": trade})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/forex/reset", methods=["POST"])
def api_forex_reset():
    """Reset forex portfolio to initial capital."""
    try:
        from forex_engine import FOREX_PORTFOLIO_FILE, FOREX_TRADES_FILE, FOREX_INITIAL_CAPITAL  # noqa
        import json as _j
        state = {"cash": FOREX_INITIAL_CAPITAL, "initial": FOREX_INITIAL_CAPITAL,
                 "positions": {}, "cooldowns": {}}
        with open(FOREX_PORTFOLIO_FILE, "w") as f:
            _j.dump(state, f)
        with open(FOREX_TRADES_FILE, "w") as f:
            _j.dump([], f)
        # Reset singleton
        import forex_engine as _fe  # noqa
        _fe._forex_agent = None
        return jsonify({"ok": True, "message": f"Forex portfolio reset to ${FOREX_INITIAL_CAPITAL:,.0f}"})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/forex/strategy_weights")
def api_forex_strategy_weights():
    try:
        from forex_engine import get_forex_agent  # noqa
        return jsonify({"ok": True, "weights": get_forex_agent().aggregator.get_weights()})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/forex/chart/<path:pair>")
def api_forex_chart(pair: str):
    """OHLCV chart data for a forex pair."""
    try:
        from forex_engine import _fetch  # noqa
        period   = request.args.get("period", "1w")
        interval_map = {"1d": ("2d", "15m"), "1w": ("7d", "1h"),
                        "1m": ("30d", "4h"), "3m": ("90d", "1d"), "1y": ("1y", "1d")}
        yf_period, yf_interval = interval_map.get(period, ("7d", "1h"))
        df = _fetch(pair, period=yf_period, interval=yf_interval)
        if df is None:
            return jsonify({"ok": False, "data": []})
        data = [{"date": str(idx), "open": round(float(r["Open"]), 6),
                 "high": round(float(r["High"]), 6), "low": round(float(r["Low"]), 6),
                 "close": round(float(r["Close"]), 6)}
                for idx, r in df.iterrows()]
        change = round((data[-1]["close"] - data[0]["close"]) / data[0]["close"] * 100, 3) if data else 0
        return jsonify({"ok": True, "data": data, "change_pct": change})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/master")
def api_master():
    """Combined overview of NSE stocks + Forex agents."""
    try:
        from engine import get_agent  # noqa
        from forex_engine import get_forex_agent, is_forex_market_open, current_session  # noqa

        # NSE stats
        nse_agent = get_agent()
        nse_state  = nse_agent.portfolio.state
        nse_total  = nse_agent.portfolio.get_total_value()
        nse_initial = nse_state.get("initial", 1_000_000)
        nse_pnl    = nse_total - nse_initial
        nse_trades = nse_agent.portfolio._load_trades() if hasattr(nse_agent.portfolio, '_load_trades') else []
        nse_closes = [t for t in nse_trades if t.get("action") == "SELL"]
        nse_wins   = [t for t in nse_closes if t.get("pnl", t.get("pnl_inr", 0)) > 0]

        # Forex stats
        fx_agent  = get_forex_agent()
        fx_summary = fx_agent.portfolio.get_summary()

        return jsonify({
            "ok": True,
            "nse": {
                "total_value": round(nse_total, 2),
                "initial":     nse_initial,
                "pnl":         round(nse_pnl, 2),
                "pnl_pct":     round(nse_pnl / nse_initial * 100, 2),
                "positions":   len(nse_state.get("positions", {})),
                "cash":        round(nse_state.get("cash", 0), 2),
                "win_rate":    round(len(nse_wins) / len(nse_closes) * 100, 1) if nse_closes else 0,
                "total_trades": len(nse_closes),
                "currency":    "INR",
                "market_open": _market_open(),
            },
            "forex": fx_summary | {
                "market_open": is_forex_market_open(),
                "session":     current_session(),
            },
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


# ═══════════════════════════════════════════════════════════════════════════
# FOREX BACKGROUND LOOP — independent scheduler, runs 24/5
# ═══════════════════════════════════════════════════════════════════════════

_FOREX_INTERVAL = int(os.environ.get("FOREX_CYCLE_INTERVAL", "3600"))  # default 1 hour

def _forex_background_loop() -> None:
    """
    Forex agent runs independently of the NSE scheduler.
    Waits 90s on startup so the main app finishes booting before we hit yfinance.
    """
    _bg_logger = logging.getLogger("app")
    _bg_logger.info("[Forex Scheduler] Forex agent loop starting in 90s…")
    time.sleep(90)   # let main app fully boot first

    from forex_engine import get_forex_agent, is_forex_market_open, current_session  # noqa
    _bg_logger.info("[Forex Scheduler] Forex agent loop started")

    while True:
        try:
            if is_forex_market_open():
                session = current_session()
                get_forex_agent().run_cycle()
                sleep_s = 900 if session["in_kill_zone"] else _FOREX_INTERVAL
            else:
                sleep_s = 3600
                _bg_logger.info("[Forex Scheduler] Forex market closed (weekend)")
        except Exception as exc:
            _bg_logger.warning(f"[Forex Scheduler] Cycle error: {exc}")
            sleep_s = 600

        time.sleep(sleep_s)


_bg_logger = logging.getLogger("app")

def _background_loop() -> None:
    """Market-aware scheduler: runs every 15 min during NSE hours (9:15–15:30 IST).

    Opening bell  → immediate first cycle
    Intra-day     → cycle every _state.auto_interval seconds
    Closing bell  → final cycle + closing report at 15:15–15:30 IST
    Holiday/weekend → sleep until next trading day open
    """
    _bg_logger.info("[Scheduler] Market-aware agent loop started")
    opening_done_dates:  set = set()
    _holiday_logged_dates: set = set()
    _digest_done_dates:  set = set()   # track 9 AM digest

    while True:
        now   = _ist_now()
        today = now.date()

        # ── 8:45 AM Pre-Market Briefing (news + events + commodities) ────────
        if now.hour == 8 and now.minute >= 44 and now.minute < 50 and today not in _digest_done_dates:
            _bg_logger.info("🌅 8:45 AM — sending pre-market briefing")
            try:
                from news_agent import send_premarket_briefing  # pylint: disable=import-outside-toplevel
                send_premarket_briefing()
            except Exception as _pe:
                _bg_logger.warning(f"Pre-market briefing error: {_pe}")

        # ── 9 AM Morning Digest — graded signals (runs regardless of market open/close) ──
        if now.hour == 9 and now.minute < 5 and today not in _digest_done_dates:
            _bg_logger.info("☀️ 9 AM — sending morning digest")
            try:
                from notifier import send_morning_digest  # pylint: disable=import-outside-toplevel
                send_morning_digest()
            except Exception as _de:
                _bg_logger.warning(f"Morning digest error: {_de}")
            _digest_done_dates.add(today)

        if _market_open():
            # ── Opening bell: run immediately on first entry ──────────────
            if today not in opening_done_dates:
                _bg_logger.info(f"🔔 Opening bell — {now.strftime('%H:%M IST')}")
                _run_cycle_safe()
                opening_done_dates.add(today)

            # ── Closing bell: generate report when ≥15:15 ─────────────────
            if now.time() >= dt_time(15, 15) and today not in _state.closing_report_dates:
                _run_cycle_safe()
                _generate_closing_report()
                _state.closing_report_dates.add(today)

            time.sleep(_state.auto_interval)

        else:
            secs = _secs_until_next_open()
            if _is_nse_holiday(today):
                if today not in _holiday_logged_dates:
                    _bg_logger.info(
                        f"🏖️ NSE Holiday today ({today}) — next trading session opens "
                        f"in {secs/3600:.1f}h"
                    )
                    _holiday_logged_dates.add(today)
            else:
                _bg_logger.info(
                    f"[Scheduler] Market closed. Next open in {secs/3600:.1f}h"
                )
            # Sleep at most 5 min at a time so we catch the open promptly
            time.sleep(min(300, secs))


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Start Angel One live feed (if credentials are present in .env) ────────
    feed = get_feed()
    if feed.is_configured():
        feed.start()
        print("  📡  Angel One SmartAPI feed starting…")
    else:
        print("  ⚠️   Angel One credentials not found in .env — using yfinance (delayed)")
        print("       Copy .env.example → .env and fill in your credentials for live data.")

    # ── Start Telegram intelligence agent ────────────────────────────────────
    from telegram_agent import get_telegram_agent  # pylint: disable=import-outside-toplevel
    tg = get_telegram_agent()
    tg.start()
    if tg.is_configured():
        print("  💬  Telegram agent starting (discovering signal groups…)")
    else:
        print("  ⚠️   Telegram credentials not set — add TELEGRAM_API_ID / HASH / PHONE to .env")

    # ── Start Telegram bot command listener (ticker analysis on demand) ───────
    try:
        from bot_listener import start_bot_listener  # pylint: disable=import-outside-toplevel
        start_bot_listener()
        print("  🤖  Bot command listener started (type any NSE ticker in your group)")
    except Exception as _ble:
        print(f"  ⚠️   Bot listener failed to start: {_ble}")

    # ── Start NSE market-aware background agent loop ─────────────────────────
    threading.Thread(target=_background_loop, daemon=True, name="nse-scheduler").start()

    # ── Start Forex 24/5 background agent loop ────────────────────────────────
    threading.Thread(target=_forex_background_loop, daemon=True, name="forex-scheduler").start()

    print("\n" + "=" * 60)
    print("  🌐  Multi-Market Trading Agent  —  Paper Mode")
    print("=" * 60)
    print("  Dashboard → http://localhost:5001")
    print(f"  NSE Capital  → ₹{INITIAL_CAPITAL:,.0f}")
    from forex_engine import FOREX_INITIAL_CAPITAL  # noqa
    print(f"  Forex Capital → ${FOREX_INITIAL_CAPITAL:,.0f}")
    print("  Strategies: Momentum | Mean Rev | Multi-Factor | Sector Rot | SMA | Fibonacci")
    print("              RSI Divergence | Bollinger Squeeze | Volume Breakout | Telegram")
    print("              News Sentiment | Commodity | Fundamental Rescoring")
    print("=" * 60 + "\n")

    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
