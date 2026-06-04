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


# ── Dashboard ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the single-page dashboard."""
    return send_from_directory("templates", "dashboard.html")


# ── Data endpoints ───────────────────────────────────────────────────────────

@app.route("/api/dashboard")
def api_dashboard():
    """Return all data needed to render the dashboard in one call."""
    try:
        data = get_agent().get_dashboard_data()
        data["agent_running"] = _state.running
        data["last_cycle"] = _state.last_cycle
        data["auto_interval_s"] = _state.auto_interval
        return jsonify({"ok": True, "data": data})
    except (ValueError, RuntimeError, KeyError) as exc:
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


@app.route("/api/pnl_calendar")
def api_pnl_calendar():
    """Daily P&L aggregated across Equity + F&O + Commodity for the calendar view."""
    from pathlib import Path as _P  # pylint: disable=import-outside-toplevel
    from engine import TRADE_LOG_FILE  # pylint: disable=import-outside-toplevel

    days: dict = {}

    def _add(date_str, pnl, is_win):
        if date_str not in days:
            days[date_str] = {"pnl": 0.0, "trades": 0, "wins": 0}
        days[date_str]["pnl"]    += pnl
        days[date_str]["trades"] += 1
        if is_win:
            days[date_str]["wins"] += 1

    # ── Equity trades ─────────────────────────────────────────────────────── #
    try:
        if TRADE_LOG_FILE.exists():
            for t in json.loads(TRADE_LOG_FILE.read_text(encoding="utf-8")):
                if t.get("action") != "SELL":
                    continue
                ts  = (t.get("timestamp") or t.get("date") or "")[:10]
                pnl = float(t.get("pnl") or 0)
                if ts:
                    _add(ts, pnl, pnl > 0)
    except Exception:
        pass

    # ── F&O trades ────────────────────────────────────────────────────────── #
    try:
        fno_file = _P("data/fno_trades.json")
        if fno_file.exists():
            for t in json.loads(fno_file.read_text()):
                action = t.get("action", "")
                if "CLOSE" not in action and action != "SELL":
                    continue
                ts  = (t.get("time") or t.get("timestamp") or t.get("date") or "")[:10]
                pnl = float(t.get("pnl") or 0)
                if ts:
                    _add(ts, pnl, pnl > 0)
    except Exception:
        pass

    # ── Commodity trades ──────────────────────────────────────────────────── #
    try:
        comm_file = _P("data/commodity_trades.json")
        if comm_file.exists():
            for t in json.loads(comm_file.read_text()):
                action = t.get("action", "")
                if "CLOSE" not in action:
                    continue
                ts  = (t.get("time") or t.get("timestamp") or "")[:10]
                pnl = float(t.get("pnl") or 0)
                if ts:
                    _add(ts, pnl, pnl > 0)
    except Exception:
        pass

    # Compute win rate per day
    result = {}
    for ds, v in days.items():
        result[ds] = {
            "pnl":      round(v["pnl"], 2),
            "trades":   v["trades"],
            "wins":     v["wins"],
            "win_rate": round(v["wins"] / v["trades"] * 100, 0) if v["trades"] else 0,
        }

    return jsonify({"ok": True, "days": result})


@app.route("/api/trades")
def api_trades():
    """Return full trade log, newest first."""
    from engine import TRADE_LOG_FILE  # pylint: disable=import-outside-toplevel

    trades = []
    if TRADE_LOG_FILE.exists():
        with open(TRADE_LOG_FILE, encoding="utf-8") as fh:
            trades = json.load(fh)
    return jsonify({"ok": True, "trades": list(reversed(trades))})


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
    """
    Trigger a full agent cycle in the background — returns immediately.
    The cycle downloads data for 500+ stocks and takes 3-5 minutes.
    Poll /api/dashboard to see when signals update.
    """
    if _state.running:
        return jsonify({"ok": False, "error": "Agent cycle already running — check back in a few minutes"})
    threading.Thread(target=_run_cycle_safe, daemon=True, name="manual-cycle").start()
    return jsonify({"ok": True, "message": "Cycle started in background — signals will update in 3-5 minutes"})


@app.route("/api/refresh_signals", methods=["POST"])
def api_refresh_signals():
    """
    Fast signal refresh — skips trade execution, just regenerates signals.
    Completes in ~60-90 seconds (much faster than full cycle).
    Returns immediately; poll /api/dashboard for updated signals.
    """
    if _state.running:
        return jsonify({"ok": False, "error": "Agent already running"})

    def _fast_refresh():
        _state.running = True
        try:
            from engine import SignalAggregator  # pylint: disable=import-outside-toplevel
            agg     = SignalAggregator()
            signals = agg.run()
            logging.getLogger(__name__).info(f"[FastRefresh] {len(signals)} signals generated")
        except Exception as exc:
            logging.getLogger(__name__).error(f"[FastRefresh] Error: {exc}", exc_info=True)
        finally:
            _state.running = False

    threading.Thread(target=_fast_refresh, daemon=True, name="fast-refresh").start()
    return jsonify({"ok": True, "message": "Signal refresh started — takes ~60-90s, then refresh the page"})


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
    """Reset equity portfolio. Optional body: {"capital": 1300000}"""
    body    = request.get_json(force=True, silent=True) or {}
    capital = int(body.get("capital", int(os.getenv("NSE_CAPITAL", 1_300_000))))
    get_agent().portfolio.reset(capital=capital)
    DataFetcher.clear_cache()
    return jsonify({"ok": True, "message": f"Portfolio reset to ₹{capital:,.0f}", "capital": capital})


@app.route("/api/full_reset", methods=["POST"])
def api_full_reset():
    """
    Full system reset — clears ALL desks (equity, F&O, commodity).
    Optional body: {"capital": 1300000}
    """
    import json as _json
    from pathlib import Path as _Path

    body    = request.get_json(force=True, silent=True) or {}
    capital = int(body.get("capital", int(os.getenv("NSE_CAPITAL", 1_300_000))))

    # ── 1. Equity reset ───────────────────────────────────────────────────── #
    get_agent().portfolio.reset(capital=capital)
    DataFetcher.clear_cache()

    # ── 2. F&O reset (returns any open premiums first, then clears) ───────── #
    try:
        from fno_engine import get_fno_agent  # pylint: disable=import-outside-toplevel
        fno = get_fno_agent()
        fno.portfolio.reset()  # returns premiums to equity cash first
        # Clear F&O trade log
        fno_log = _Path("data/fno_trades.json")
        if fno_log.exists():
            fno_log.write_text("[]")
    except Exception as _e:
        logging.getLogger(__name__).warning(f"F&O reset error: {_e}")

    # ── 3. Commodity reset ────────────────────────────────────────────────── #
    try:
        from commodity_engine import get_commodity_agent  # pylint: disable=import-outside-toplevel
        comm = get_commodity_agent()
        comm.portfolio.reset()
        comm_log = _Path("data/commodity_trades.json")
        if comm_log.exists():
            comm_log.write_text("[]")
    except Exception as _e:
        logging.getLogger(__name__).warning(f"Commodity reset error: {_e}")

    # ── 4. Clear signals and live orders ─────────────────────────────────── #
    _log = logging.getLogger(__name__)
    for fname in ["data/signals.json", "data/live_orders.json"]:
        p = _Path(fname)
        if p.exists():
            p.write_text("{}" if "signals" in fname else "[]")

    _log.info(f"🔄 Full system reset — ₹{capital:,.0f} fresh start")
    return jsonify({
        "ok":      True,
        "message": f"All desks reset. Fresh start with ₹{capital:,.0f}.",
        "capital": capital,
    })


# ── Live trading controls ─────────────────────────────────────────────────────

@app.route("/api/trader/status")
def api_trader_status():
    """Return live trading status, halt state, today's orders."""
    from angelone_trader import get_trader  # pylint: disable=import-outside-toplevel
    t = get_trader()
    from angelone_trader import _halt_reason  # pylint: disable=import-outside-toplevel
    orders = t.get_order_book()
    return jsonify({
        "ok":          True,
        "is_live":     t.is_live,
        "halted":      t.halted,
        "halt_reason": _halt_reason,
        "mode":        "LIVE" if t.is_live else "PAPER",
        "orders_today": orders,
    })

@app.route("/api/trader/halt", methods=["POST"])
def api_trader_halt():
    """Emergency halt — blocks all new orders immediately."""
    from angelone_trader import get_trader  # pylint: disable=import-outside-toplevel
    body = request.get_json(force=True) or {}
    get_trader().halt(body.get("reason", "Manual halt from dashboard"))
    return jsonify({"ok": True, "message": "Trading halted"})

@app.route("/api/trader/resume", methods=["POST"])
def api_trader_resume():
    """Lift trading halt."""
    from angelone_trader import get_trader  # pylint: disable=import-outside-toplevel
    get_trader().resume()
    return jsonify({"ok": True, "message": "Trading resumed"})

@app.route("/api/trader/sync", methods=["POST"])
def api_trader_sync():
    """Reconcile agent positions with actual Angel One account."""
    from angelone_trader import get_trader  # pylint: disable=import-outside-toplevel
    result = get_trader().sync_portfolio(get_agent().portfolio)
    return jsonify(result)

@app.route("/api/trader/orders")
def api_trader_orders():
    """Return today's live order log."""
    from angelone_trader import get_trader  # pylint: disable=import-outside-toplevel
    return jsonify({"ok": True, "orders": get_trader().get_order_book()})


@app.route("/api/add_capital", methods=["POST"])
def api_add_capital():
    """Add capital to the portfolio cash balance. Body: {amount: 300000}"""
    body    = request.get_json(force=True) or {}
    amount  = float(body.get("amount", 0))
    if amount <= 0:
        return jsonify({"ok": False, "error": "amount must be > 0"}), 400
    port = get_agent().portfolio
    port.state["cash"]          += amount
    port.state["initial"]        = port.state.get("initial", INITIAL_CAPITAL)
    # Reset today's baseline so the capital addition doesn't show as profit
    port.state["day_start_value"] = round(port.get_total_value(), 2)
    port.state["day_start_date"]  = _ist_now().strftime("%Y-%m-%d")
    port._save()
    return jsonify({
        "ok":      True,
        "added":   amount,
        "cash":    round(port.state["cash"], 2),
        "initial": port.state["initial"],
        "message": f"Added ₹{amount:,.0f} — cash now ₹{port.state['cash']:,.0f}",
    })


@app.route("/api/fix_today_baseline", methods=["POST"])
def api_fix_today_baseline():
    """Reset today's P&L baseline to current portfolio value. Call once after any capital addition."""
    port = get_agent().portfolio
    current = round(port.get_total_value(), 2)
    port.state["day_start_value"] = current
    port.state["day_start_date"]  = _ist_now().strftime("%Y-%m-%d")
    port._save()
    return jsonify({"ok": True, "day_start_value": current, "message": "Today baseline reset"})


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


# ── Unified portfolio summary ────────────────────────────────────────────────

@app.route("/api/portfolio_summary")
def api_portfolio_summary():
    """Aggregated view: Equity + F&O + Commodity in one response."""
    import traceback  # pylint: disable=import-outside-toplevel

    # ── Equity ───────────────────────────────────────────────────────────────
    try:
        agent  = get_agent()
        eq_pos = agent.portfolio.get_positions_display()
        eq_cash   = round(agent.portfolio.state["cash"], 2)
        eq_equity = round(sum(p.get("value", 0) for p in eq_pos), 2)
        eq_unreal = round(sum(p.get("pnl", 0) for p in eq_pos), 2)
        eq_total  = round(eq_cash + eq_equity, 2)
        eq_cost   = round(sum(p.get("avg_price", 0) * p.get("qty", 0) for p in eq_pos), 2)
        # Use the actual initial capital recorded in portfolio state (handles top-ups correctly)
        eq_initial = agent.portfolio.state.get("initial", INITIAL_CAPITAL)
    except Exception:
        app.logger.error("portfolio_summary equity error:\n" + traceback.format_exc())
        eq_pos, eq_cash, eq_equity, eq_unreal, eq_total, eq_cost, eq_initial = [], 0, 0, 0, 0, 0, INITIAL_CAPITAL

    # ── F&O ──────────────────────────────────────────────────────────────────
    fno_positions, fno_unreal, fno_realised, fno_greeks = [], 0.0, 0.0, {}
    try:
        from fno_engine import get_fno_agent as _get_fno  # pylint: disable=import-outside-toplevel
        fno_d        = _get_fno().get_dashboard_data()
        fno_positions = fno_d.get("positions", [])
        fno_unreal    = float(fno_d.get("unrealised_pnl", 0) or 0)
        fno_realised  = float(fno_d.get("realised_pnl", 0) or 0)
        fno_greeks    = fno_d.get("greeks", {})
    except Exception:
        app.logger.error("portfolio_summary F&O error:\n" + traceback.format_exc())

    # ── Commodity ────────────────────────────────────────────────────────────
    comm_positions, comm_unreal, comm_realised, comm_prices = [], 0.0, 0.0, {}
    try:
        from commodity_engine import get_commodity_agent as _get_comm  # pylint: disable=import-outside-toplevel
        comm_d         = _get_comm().get_dashboard_data()
        comm_positions = comm_d.get("positions", [])
        comm_unreal    = float(comm_d.get("unrealised_pnl", 0) or 0)
        comm_realised  = float(comm_d.get("realised_pnl", 0) or 0)
        comm_prices    = comm_d.get("prices", {})
    except Exception:
        app.logger.error("portfolio_summary Commodity error:\n" + traceback.format_exc())

    # F&O deployed capital = entry premiums still in open positions
    # (these were deducted from equity cash when opened, so we add them back to get true AUM)
    fno_margin = round(sum(
        p.get("entry_premium", 0) * p.get("qty_lots", 1) for p in fno_positions
    ), 2)
    comm_margin = round(sum(p.get("margin", 0) for p in comm_positions), 2)

    # True AUM = cash (after premiums deducted) + equity stock positions
    #          + F&O deployed premiums + F&O unrealised P&L
    #          + commodity margin + commodity unrealised
    # = original_capital + all_unrealised_pnl  (simplifies to this)
    total_aum = round(eq_total + fno_margin + fno_unreal + comm_margin + comm_unreal, 2)

    return jsonify({
        "ok": True,
        "summary": {
            "total_aum":        total_aum,
            "initial_capital":  eq_initial,
            "total_unrealised": round(eq_unreal + fno_unreal + comm_unreal, 2),
            "total_realised":   round(fno_realised + comm_realised, 2),
            "free_cash":        eq_cash,
        },
        "equity": {
            "cash": eq_cash, "equity_val": eq_equity, "cost_basis": eq_cost,
            "unrealised": eq_unreal, "total": eq_total,
            "positions": eq_pos, "count": len(eq_pos),
            "initial": eq_initial,
        },
        "fno": {
            "margin_used": fno_margin,
            "unrealised":  round(fno_unreal, 2),
            "realised":    round(fno_realised, 2),
            "positions":   fno_positions,
            "count":       len(fno_positions),
            "greeks":      fno_greeks,
        },
        "commodity": {
            "margin_used": comm_margin,
            "unrealised":  round(comm_unreal, 2),
            "realised":    round(comm_realised, 2),
            "positions":   comm_positions,
            "count":       len(comm_positions),
            "prices":      comm_prices,
        },
    })


# ── F&O API endpoints ────────────────────────────────────────────────────────

@app.route("/api/fno/dashboard")
def api_fno_dashboard():
    """Full F&O dashboard data."""
    try:
        from fno_engine import get_fno_agent  # pylint: disable=import-outside-toplevel
        return jsonify({"ok": True, "data": get_fno_agent().get_dashboard_data()})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

@app.route("/api/fno/positions")
def api_fno_positions():
    try:
        from fno_engine import get_fno_agent  # pylint: disable=import-outside-toplevel
        return jsonify({"ok": True, "positions": get_fno_agent().portfolio.get_positions_display()})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

@app.route("/api/fno/greeks")
def api_fno_greeks():
    try:
        from fno_engine import get_fno_agent  # pylint: disable=import-outside-toplevel
        return jsonify({"ok": True, "greeks": get_fno_agent().portfolio.portfolio_greeks()})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

@app.route("/api/fno/trades")
def api_fno_trades():
    try:
        from fno_engine import FNO_TRADE_FILE  # pylint: disable=import-outside-toplevel
        import json as _j
        trades = []
        if FNO_TRADE_FILE.exists():
            with open(FNO_TRADE_FILE) as _f:
                trades = _j.load(_f)
        return jsonify({"ok": True, "trades": list(reversed(trades[-100:]))})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

@app.route("/api/fno/close_position", methods=["POST"])
def api_fno_close_position():
    data = request.get_json() or {}
    pid  = data.get("position_id", "").strip()
    if not pid:
        return jsonify({"ok": False, "error": "position_id required"}), 400
    try:
        from fno_engine import get_fno_agent  # pylint: disable=import-outside-toplevel
        fno = get_fno_agent()
        pos = fno.portfolio.state["positions"].get(pid)
        if not pos:
            return jsonify({"ok": False, "error": "Position not found"}), 404
        if pos["instrument_type"] == "OPTION":
            t = fno.portfolio.close_option(pid, reason="MANUAL_CLOSE")
        else:
            t = fno.portfolio.close_future(pid, reason="MANUAL_CLOSE")
        return jsonify({"ok": True, "trade": t})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

@app.route("/api/events")
def api_events():
    """Return recent detected geopolitical/macro events."""
    try:
        from event_engine import get_recent_events  # pylint: disable=import-outside-toplevel
        hours = float(request.args.get("hours", 24))
        return jsonify({"ok": True, "events": get_recent_events(hours)})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/commodity/dashboard")
def api_commodity_dashboard():
    try:
        from commodity_engine import get_commodity_agent  # pylint: disable=import-outside-toplevel
        return jsonify({"ok": True, **get_commodity_agent().get_dashboard_data()})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/commodity/close", methods=["POST"])
def api_commodity_close():
    symbol = request.json.get("symbol", "").upper()
    try:
        from commodity_engine import get_commodity_agent  # pylint: disable=import-outside-toplevel
        trade = get_commodity_agent().portfolio.close_position(symbol, reason="MANUAL")
        if trade:
            return jsonify({"ok": True, "trade": trade})
        return jsonify({"ok": False, "error": "No open position for that symbol"})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/commodity/reset", methods=["POST"])
def api_commodity_reset():
    try:
        from commodity_engine import get_commodity_agent  # pylint: disable=import-outside-toplevel
        get_commodity_agent().portfolio.reset()
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/fno/hourly_signals")
def api_fno_hourly_signals():
    try:
        from fno_engine import FNO_HOURLY_SIGNALS_FILE  # pylint: disable=import-outside-toplevel
        import json as _j
        if FNO_HOURLY_SIGNALS_FILE.exists():
            with open(FNO_HOURLY_SIGNALS_FILE) as _f:
                return jsonify({"ok": True, **_j.load(_f)})
        return jsonify({"ok": True, "signals": [], "generated_at": None})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

@app.route("/api/fno/reset", methods=["POST"])
def api_fno_reset():
    from fno_engine import get_fno_agent  # pylint: disable=import-outside-toplevel
    get_fno_agent().portfolio.reset()
    return jsonify({"ok": True, "message": "F&O portfolio reset to ₹2,00,000"})

@app.route("/api/fno/option_price")
def api_fno_option_price():
    """Compute theoretical option price for any input."""
    try:
        from fno_engine import BlackScholes, historical_vol, days_to_expiry, get_expiry  # pylint: disable=import-outside-toplevel
        import datetime as _dt
        ticker  = request.args.get("ticker", "^NSEI")
        strike  = float(request.args.get("strike", 0))
        opt_type= request.args.get("type", "call").lower()
        monthly = request.args.get("monthly", "false").lower() == "true"

        if strike <= 0:
            return jsonify({"ok": False, "error": "strike required"}), 400

        expiry = get_expiry(ticker, monthly=monthly)
        T      = days_to_expiry(expiry)
        iv     = historical_vol(ticker)

        import yfinance as _yf
        spot_data = _yf.download(ticker, period="2d", interval="1d", progress=False)
        if isinstance(spot_data.columns, __import__("pandas").MultiIndex):
            spot_data.columns = spot_data.columns.get_level_values(0)
        spot = float(spot_data["Close"].iloc[-1]) if not spot_data.empty else 0

        from fno_engine import RISK_FREE_RATE  # pylint: disable=import-outside-toplevel
        price  = BlackScholes.price(spot, strike, T, RISK_FREE_RATE, iv, opt_type)
        greeks = BlackScholes.greeks(spot, strike, T, RISK_FREE_RATE, iv, opt_type)

        return jsonify({
            "ok": True,
            "spot": round(spot, 2), "strike": strike,
            "expiry": expiry.isoformat(), "days_to_expiry": int(T * 365),
            "iv": iv, "price": round(price, 2),
            "greeks": greeks,
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


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
    """Run one agent cycle (equity + F&O), guarded against concurrent calls."""
    if _state.running:
        return
    _state.running = True
    try:
        agent   = get_agent()
        summary = agent.run_cycle()
        _state.last_cycle = summary

        # ── Daily loss circuit breaker (live mode only) ─────────────────── #
        try:
            from angelone_trader import get_trader  # pylint: disable=import-outside-toplevel
            get_trader().check_daily_loss(agent.portfolio.get_total_value())
        except Exception:
            pass

        # ── F&O cycle: pass equity signals + positions ─────────────────── #
        try:
            from fno_engine import get_fno_agent  # pylint: disable=import-outside-toplevel
            fno     = get_fno_agent()
            eq_pos  = agent.portfolio.get_positions_display()
            eq_val  = agent.portfolio.get_total_value()
            initial = agent.portfolio.state.get("initial", 1_000_000)
            eq_drawdown = max(0.0, (initial - eq_val) / initial * 100)

            # Pull latest signals from file
            from engine import SIGNALS_FILE  # pylint: disable=import-outside-toplevel
            import json as _json
            eq_signals = []
            if SIGNALS_FILE.exists():
                with open(SIGNALS_FILE) as _f:
                    eq_signals = _json.load(_f).get("signals", [])

            fno_summary = fno.run_cycle(
                equity_signals=eq_signals,
                equity_positions=eq_pos,
                equity_drawdown_pct=eq_drawdown,
            )
            summary["fno"] = fno_summary
        except Exception as _fe:
            import logging as _logging
            _logging.getLogger(__name__).warning(f"F&O cycle error: {_fe}")

    except Exception as exc:
        import traceback as _tb
        import logging as _logging
        _logging.getLogger("engine").error(f"❌ Agent cycle crashed: {exc}\n{_tb.format_exc()}")
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
        "day_pnl_pct":    round(day_pnl / INITIAL_CAPITAL * 100, 4),
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


_bg_logger = logging.getLogger("app")

def _background_loop() -> None:
    """Market-aware scheduler: runs every 15 min during NSE hours (9:15–15:30 IST).

    Opening bell  → immediate first cycle
    Intra-day     → cycle every _state.auto_interval seconds
    Closing bell  → final cycle + closing report at 15:15–15:30 IST
    Holiday/weekend → sleep until next trading day open
    """
    _bg_logger.info("[Scheduler] Market-aware agent loop started")
    opening_done_dates: set = set()
    _holiday_logged_dates: set = set()

    while True:
        now   = _ist_now()
        today = now.date()

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
    # ── Self-initialise on Railway volume if data files don't exist ───────────
    import json as _j
    from pathlib import Path as _P
    from datetime import datetime as _dt

    _data_dir = _P("data")
    _data_dir.mkdir(exist_ok=True)

    _STARTUP_CAPITAL = int(os.getenv("NSE_CAPITAL", 1_300_000))

    _port_file = _data_dir / "portfolio.json"
    if not _port_file.exists():
        _port_file.write_text(_j.dumps({
            "cash":            _STARTUP_CAPITAL,
            "initial":         _STARTUP_CAPITAL,
            "positions":       {},
            "realised_pnl":    0.0,
            "created_at":      _dt.now().isoformat(),
            "last_updated":    _dt.now().isoformat(),
            "day_start_value": _STARTUP_CAPITAL,
            "day_start_date":  _dt.now().date().isoformat(),
        }, indent=2))
        print(f"  🆕  Created portfolio.json with ₹{_STARTUP_CAPITAL:,.0f} capital")

    for _fname, _empty in [
        ("trade_log.json", "[]"),
        ("signals.json", "{}"),
        ("fno_portfolio.json", None),
        ("fno_trades.json", "[]"),
        ("commodity_trades.json", "[]"),
        ("live_orders.json", "[]"),
    ]:
        _fp = _data_dir / _fname
        if not _fp.exists() and _empty is not None:
            _fp.write_text(_empty)
            print(f"  🆕  Created {_fname}")

    # ── Ensure initial capital is correct (handles restarts after capital change)
    try:
        from engine import get_agent as _ga  # pylint: disable=import-outside-toplevel
        _port = _ga().portfolio
        if _port.state.get("initial", 0) != _STARTUP_CAPITAL:
            _port.state["initial"] = _STARTUP_CAPITAL
            _port._save()
        print(f"  ✅  Portfolio: cash=₹{_port.state['cash']:,.0f}  initial=₹{_port.state['initial']:,.0f}")
    except Exception as _te:
        print(f"  ⚠️  Capital check failed: {_te}")

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

    # ── Start Telegram bot command listener (/analyze, /grade, /help) ────────
    try:
        from bot_listener import start_bot_listener  # pylint: disable=import-outside-toplevel
        start_bot_listener()
        print("  🤖  Bot command listener started (/analyze, /grade, /help)")
    except Exception as _ble:
        print(f"  ⚠️   Bot listener failed to start: {_ble}")

    # ── Start market-aware background agent loop ──────────────────────────────
    threading.Thread(target=_background_loop, daemon=True).start()

    # ── Hourly F&O signal generator ───────────────────────────────────────
    def _hourly_fno_loop():
        import time as _time
        while True:
            try:
                from fno_engine import run_hourly_fno_signals  # pylint: disable=import-outside-toplevel
                run_hourly_fno_signals()
            except Exception as _e:
                print(f"[HourlyFNO] Error: {_e}")
            _time.sleep(3600)   # every 60 minutes

    threading.Thread(target=_hourly_fno_loop, daemon=True, name="hourly-fno").start()
    print("  ⚡  Hourly F&O signal generator started")

    # ── Real-time F&O position monitor (60s loop) ─────────────────────────
    # Uses Angel One live prices to recompute Greeks every minute and trigger
    # stops/alerts faster than the 15-min equity cycle.
    def _fno_realtime_monitor():
        import time as _time
        from zoneinfo import ZoneInfo
        _IST = ZoneInfo("Asia/Kolkata")

        while True:
            try:
                now = datetime.now(_IST).time()
                # Only run during market hours
                from datetime import time as _dt_time
                if not (_dt_time(9, 15) <= now <= _dt_time(15, 0)):
                    _time.sleep(60)
                    continue

                from fno_engine import get_fno_agent   # noqa: PLC0415
                from angelone_feed import get_feed     # noqa: PLC0415
                from fno_engine import BlackScholes, days_to_expiry, RISK_FREE_RATE  # noqa: PLC0415
                from datetime import date as _date    # noqa: PLC0415

                fno      = get_fno_agent()
                feed     = get_feed()
                portfolio = fno.portfolio
                positions = dict(portfolio.state.get("positions", {}))

                for pid, pos in positions.items():
                    if pos.get("instrument_type") != "OPTION":
                        continue
                    if pos.get("position") != "LONG":
                        continue   # only manage long options in real-time

                    underlying = pos["underlying"]
                    # Try live price first, fall back to last known
                    spot = feed.get_price(underlying) if feed.is_connected() else None
                    if not spot or spot <= 0:
                        continue

                    try:
                        # Minimum hold: don't trigger realtime stops within 2h of opening
                        # BS model prices can diverge from entry by 30-50% on tiny spot moves
                        # for OTM options — this is model noise, not a real loss
                        entry_date = pos.get("entry_date", "")
                        if entry_date:
                            from datetime import datetime as _dt  # noqa: PLC0415
                            entry_dt = _dt.fromisoformat(entry_date)
                            if entry_dt.tzinfo is None:
                                entry_dt = entry_dt.replace(tzinfo=_IST)
                            age_hours = (datetime.now(_IST) - entry_dt).total_seconds() / 3600
                            if age_hours < 2.0:
                                continue  # too young — skip stop check

                        # Try Angel One option LTP first (actual traded price)
                        curr_prem = None
                        try:
                            if feed.is_connected() and feed._smart:
                                ltp = feed.get_option_ltp(
                                    underlying, pos["strike"],
                                    _date.fromisoformat(pos["expiry"]), pos["option_type"]
                                )
                                if ltp and ltp > 0:
                                    curr_prem = ltp
                        except Exception:
                            pass

                        # Fall back to Black-Scholes if Angel One unavailable
                        if curr_prem is None:
                            T  = days_to_expiry(_date.fromisoformat(pos["expiry"]))
                            iv = pos.get("iv", 0.25)
                            curr_prem = BlackScholes.price(
                                spot, pos["strike"], T, RISK_FREE_RATE, iv, pos["option_type"]
                            )

                        entry_prem = pos["entry_premium"]
                        pnl_pct = (curr_prem - entry_prem) / entry_prem if entry_prem else 0

                        # Intraday stop: exit if premium falls 50% from entry (after 2h hold)
                        if pnl_pct <= -0.50:
                            portfolio.close_option(pid, reason="REALTIME_STOP_50PCT")
                            logger.info(
                                f"[FNO-RT] 🛑 Stop hit {underlying} {pos['strike']}{pos['option_type'][0].upper()} "
                                f"prem={curr_prem:.2f} ({pnl_pct:.0%}) — closed"
                            )

                        # Intraday target: exit if premium gains 80% (after 2h hold)
                        elif pnl_pct >= 0.80:
                            portfolio.close_option(pid, reason="REALTIME_TARGET_80PCT")
                            logger.info(
                                f"[FNO-RT] ✅ Target hit {underlying} {pos['strike']}{pos['option_type'][0].upper()} "
                                f"prem={curr_prem:.2f} ({pnl_pct:.0%}) — closed"
                            )

                    except Exception as _pos_err:
                        logger.debug(f"[FNO-RT] Error checking {pid}: {_pos_err}")

            except Exception as _e:
                logger.warning(f"[FNO-RT] Monitor error: {_e}")

            _time.sleep(60)   # check every 60 seconds

    threading.Thread(target=_fno_realtime_monitor, daemon=True, name="fno-rt-monitor").start()
    print("  📡  Real-time F&O position monitor started (60s loop, Angel One prices)")

    # ── Geopolitical / macro event monitor (5-min loop) ───────────────────────
    def _event_monitor_loop():
        import time as _time
        while True:
            try:
                from event_engine import run_event_scan  # noqa: PLC0415
                from fno_engine import get_fno_agent     # noqa: PLC0415
                agent_obj  = get_agent()
                fno_obj    = get_fno_agent()
                results    = run_event_scan(agent_obj.portfolio, fno_obj.portfolio)

                for r in results:
                    event   = r["event"]
                    actions = r["actions"]
                    if not actions and event["severity"] == "MEDIUM":
                        continue  # don't spam for medium events with no actions

                    # Telegram alert
                    try:
                        from notifier import get_notifier  # noqa: PLC0415
                        _send_event_alert(get_notifier(), event, actions)
                    except Exception as _ne:
                        logger.warning(f"[EventMonitor] Notifier error: {_ne}")

            except Exception as _e:
                logger.warning(f"[EventMonitor] Error: {_e}")

            _time.sleep(300)  # check every 5 minutes

    def _send_event_alert(notifier, event: dict, actions: list):
        severity_emoji = {"CRITICAL": "🚨", "HIGH": "⚠️", "MEDIUM": "📋"}.get(event["severity"], "📋")
        impact_lines = "\n".join(
            f"  {'🟢' if i['direction']=='BUY' else '🔴'} {i['sector']}: {i['direction']} — {i['reason']}"
            for i in event["impact"][:6]
        )
        actions_text = "\n".join(f"  {a}" for a in actions) if actions else "  (monitoring — no positions affected)"
        msg = (
            f"<b>{severity_emoji} Market Event Detected</b>\n\n"
            f"<b>{event['event_id'].replace('_', ' ').title()}</b>\n"
            f"Severity: {event['severity']}\n\n"
            f"📰 Trigger: <i>{event['trigger_headline'][:200]}</i>\n\n"
            f"<b>Sector Impact:</b>\n{impact_lines}\n\n"
            f"<b>Actions Taken:</b>\n{actions_text}\n\n"
            f"🕐 {datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%d %b %Y, %H:%M IST')}"
        )
        notifier._tg.send_async(msg)

    threading.Thread(target=_event_monitor_loop, daemon=True, name="event-monitor").start()
    print("  🌐  Geopolitical/macro event monitor started (5-min loop)")

    # ── Commodity desk scheduler (15-min loop) ────────────────────────────────
    def _commodity_loop():
        import time as _time
        while True:
            try:
                from commodity_engine import get_commodity_agent  # noqa: PLC0415
                from event_engine import get_recent_events         # noqa: PLC0415
                recent_events = get_recent_events(hours=6)
                get_commodity_agent().run_cycle(active_events=recent_events)
            except Exception as _e:
                logger.warning(f"[CommodityLoop] Error: {_e}")
            _time.sleep(900)  # every 15 minutes

    threading.Thread(target=_commodity_loop, daemon=True, name="commodity-loop").start()
    print("  🥇  Commodity desk started (15-min loop, MCX paper trading)")

    print("\n" + "=" * 60)
    print("  🇮🇳  Indian Institutional Trading Agent  —  Paper Mode")
    print("=" * 60)
    print("  Dashboard → http://localhost:5001")
    print(f"  Capital   → ₹{INITIAL_CAPITAL:,.0f}")
    print("  Strategies: Momentum | Mean Rev | Multi-Factor | Sector Rot | SMA | Fibonacci")
    print("              RSI Divergence | Bollinger Squeeze | Volume Breakout | Telegram")
    print("              News Sentiment | Commodity | Fundamental Rescoring")
    print("=" * 60 + "\n")

    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
