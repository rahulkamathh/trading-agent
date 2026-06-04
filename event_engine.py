"""
event_engine.py — Geopolitical & Macro Event Detection + Portfolio Response

Monitors news in real-time and maps events to sector impacts.
Bypasses the 15-min cycle to act immediately when markets are moving.

Event hierarchy:
  CRITICAL  → immediate portfolio action (close hedges, open protection)
  HIGH      → next available execution slot (within 5 min)
  MEDIUM    → flag for next regular cycle

Sector impact format:
  {"sector": "Energy", "direction": "BUY", "tickers": [...], "reason": "..."}
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

logger = logging.getLogger("event_engine")
_IST = ZoneInfo("Asia/Kolkata")

DATA_DIR          = Path(__file__).parent / "data"
EVENT_LOG_FILE    = DATA_DIR / "event_log.json"
EVENT_STATE_FILE  = DATA_DIR / "event_state.json"

# ── Event keyword taxonomy ────────────────────────────────────────────────────

EVENT_PATTERNS: list[dict] = [
    # ── Geopolitical ─────────────────────────────────────────────────────────
    {
        "id":       "MIDDLE_EAST_CONFLICT",
        "severity": "CRITICAL",
        "keywords": ["iran", "israel", "strike", "missile", "attack", "hezbollah",
                     "hamas", "gaza", "middle east war", "strait of hormuz"],
        "impact": [
            {"sector": "Energy",     "direction": "BUY",  "reason": "Crude supply disruption"},
            {"sector": "Defence",    "direction": "BUY",  "reason": "Defence spending surge"},
            {"sector": "Aviation",   "direction": "SELL", "reason": "Fuel cost spike + route disruptions"},
            {"sector": "Paints",     "direction": "SELL", "reason": "Crude-linked raw material cost surge"},
            {"sector": "Auto",       "direction": "SELL", "reason": "Fuel cost dampens demand"},
            {"sector": "Broad",      "direction": "SELL", "reason": "Risk-off sentiment"},
        ],
        "vix_action": "HEDGE",   # buy Nifty PUT if VIX spikes
        "crude_action": "BUY",
    },
    {
        "id":       "US_MILITARY_ACTION",
        "severity": "CRITICAL",
        "keywords": ["us military", "american strike", "pentagon", "us navy", "us attack",
                     "us forces", "american forces", "us launched", "iran strike"],
        "impact": [
            {"sector": "Energy",     "direction": "BUY",  "reason": "Crude spike on supply fears"},
            {"sector": "Gold",       "direction": "BUY",  "reason": "Safe-haven demand"},
            {"sector": "IT",         "direction": "SELL", "reason": "US client uncertainty"},
            {"sector": "Broad",      "direction": "SELL", "reason": "Global risk-off"},
        ],
        "vix_action": "HEDGE",
        "crude_action": "BUY",
    },
    {
        "id":       "RUSSIA_UKRAINE",
        "severity": "HIGH",
        "keywords": ["russia ukraine", "nato", "zelensky", "putin", "sanctions russia",
                     "kyiv", "kharkiv", "russian missile"],
        "impact": [
            {"sector": "Fertilizer", "direction": "SELL", "reason": "Supply chain disruption"},
            {"sector": "Metals",     "direction": "BUY",  "reason": "Steel/aluminium shortage"},
            {"sector": "Energy",     "direction": "BUY",  "reason": "Gas supply fears"},
            {"sector": "Pharma",     "direction": "BUY",  "reason": "Defensive sector"},
        ],
        "vix_action": "MONITOR",
        "crude_action": "BUY",
    },
    {
        "id":       "CHINA_TAIWAN",
        "severity": "CRITICAL",
        "keywords": ["china taiwan", "taiwan strait", "pla exercises", "taipei",
                     "chinese military taiwan", "taiwan invasion"],
        "impact": [
            {"sector": "IT",         "direction": "SELL", "reason": "Global supply chain collapse risk"},
            {"sector": "Semicon",    "direction": "SELL", "reason": "TSMC disruption"},
            {"sector": "Pharma",     "direction": "BUY",  "reason": "Defensive"},
            {"sector": "Gold",       "direction": "BUY",  "reason": "Safe-haven"},
            {"sector": "Broad",      "direction": "SELL", "reason": "Extreme risk-off"},
        ],
        "vix_action": "HEDGE",
        "crude_action": "NEUTRAL",
    },

    # ── Indian macro ──────────────────────────────────────────────────────────
    {
        "id":       "RBI_RATE_HIKE",
        "severity": "HIGH",
        "keywords": ["rbi rate hike", "rbi raises", "repo rate hike", "rbi tightening",
                     "monetary policy tightening", "rbi hawkish"],
        "impact": [
            {"sector": "Banking",    "direction": "SELL", "reason": "NIM compression risk"},
            {"sector": "NBFC",       "direction": "SELL", "reason": "Higher borrowing costs"},
            {"sector": "Realty",     "direction": "SELL", "reason": "EMI burden rises"},
            {"sector": "Auto",       "direction": "SELL", "reason": "Loan cost rises"},
            {"sector": "FMCG",       "direction": "BUY",  "reason": "Defensive, rate-insensitive"},
            {"sector": "IT",         "direction": "BUY",  "reason": "Dollar strengthens on rate diff"},
        ],
        "vix_action": "MONITOR",
        "crude_action": "NEUTRAL",
    },
    {
        "id":       "RBI_RATE_CUT",
        "severity": "HIGH",
        "keywords": ["rbi rate cut", "rbi cuts", "repo rate cut", "rbi easing",
                     "monetary easing", "rbi dovish", "rbi reduces"],
        "impact": [
            {"sector": "Banking",    "direction": "BUY",  "reason": "Loan growth, NIM support"},
            {"sector": "Realty",     "direction": "BUY",  "reason": "EMI falls, demand rises"},
            {"sector": "Auto",       "direction": "BUY",  "reason": "Cheaper loans boost sales"},
            {"sector": "NBFC",       "direction": "BUY",  "reason": "Lower borrowing costs"},
        ],
        "vix_action": "MONITOR",
        "crude_action": "NEUTRAL",
    },
    {
        "id":       "INDIA_GDP_MISS",
        "severity": "MEDIUM",
        "keywords": ["india gdp miss", "india growth slows", "gdp below estimate",
                     "economic slowdown india", "india recession"],
        "impact": [
            {"sector": "Cyclical",   "direction": "SELL", "reason": "Growth disappointment"},
            {"sector": "FMCG",       "direction": "BUY",  "reason": "Defensive rotation"},
            {"sector": "Pharma",     "direction": "BUY",  "reason": "Defensive rotation"},
        ],
        "vix_action": "MONITOR",
        "crude_action": "NEUTRAL",
    },
    {
        "id":       "FII_SELLOFF",
        "severity": "HIGH",
        "keywords": ["fii selling", "fii outflow", "foreign selling india",
                     "fpi outflow", "foreign portfolio selloff"],
        "impact": [
            {"sector": "Banking",    "direction": "SELL", "reason": "FII-heavy sector"},
            {"sector": "IT",         "direction": "SELL", "reason": "FII-heavy sector"},
            {"sector": "Broad",      "direction": "SELL", "reason": "Liquidity withdrawal"},
            {"sector": "Gold",       "direction": "BUY",  "reason": "Safe haven"},
        ],
        "vix_action": "MONITOR",
        "crude_action": "NEUTRAL",
    },

    # ── Crude oil ─────────────────────────────────────────────────────────────
    {
        "id":       "CRUDE_SPIKE",
        "severity": "HIGH",
        "keywords": ["crude oil surges", "oil price spike", "brent above 100",
                     "crude above 90", "opec cut", "oil supply cut"],
        "impact": [
            {"sector": "Energy",     "direction": "BUY",  "reason": "Direct beneficiary"},
            {"sector": "Aviation",   "direction": "SELL", "reason": "ATF cost surge"},
            {"sector": "Paints",     "direction": "SELL", "reason": "Crude-linked RM costs"},
            {"sector": "Tyre",       "direction": "SELL", "reason": "Rubber + crude cost rise"},
            {"sector": "OMC",        "direction": "SELL", "reason": "Marketing margin compression"},
        ],
        "vix_action": "MONITOR",
        "crude_action": "BUY",
    },
    {
        "id":       "CRUDE_CRASH",
        "severity": "HIGH",
        "keywords": ["crude oil crash", "oil price falls", "brent below 60",
                     "opec production increase", "oil glut"],
        "impact": [
            {"sector": "Aviation",   "direction": "BUY",  "reason": "ATF cost falls"},
            {"sector": "Paints",     "direction": "BUY",  "reason": "RM cost relief"},
            {"sector": "Tyre",       "direction": "BUY",  "reason": "Input cost relief"},
            {"sector": "OMC",        "direction": "BUY",  "reason": "Marketing margin improves"},
            {"sector": "Energy",     "direction": "SELL", "reason": "Revenue falls"},
        ],
        "vix_action": "MONITOR",
        "crude_action": "SELL",
    },

    # ── US macro ──────────────────────────────────────────────────────────────
    {
        "id":       "FED_RATE_HIKE",
        "severity": "HIGH",
        "keywords": ["fed rate hike", "federal reserve hikes", "powell hike",
                     "fed raises rates", "fomc hike", "us rate hike"],
        "impact": [
            {"sector": "IT",         "direction": "SELL", "reason": "USD strengthens, valuation hit"},
            {"sector": "Metals",     "direction": "SELL", "reason": "Dollar up, commodity down"},
            {"sector": "Gold",       "direction": "SELL", "reason": "Higher US yields"},
            {"sector": "NBFC",       "direction": "SELL", "reason": "FII outflow risk"},
        ],
        "vix_action": "MONITOR",
        "crude_action": "NEUTRAL",
    },
    {
        "id":       "US_RECESSION_FEAR",
        "severity": "HIGH",
        "keywords": ["us recession", "american recession", "us gdp contraction",
                     "us slowdown", "yield curve inversion us"],
        "impact": [
            {"sector": "IT",         "direction": "SELL", "reason": "US client spending cuts"},
            {"sector": "Metals",     "direction": "SELL", "reason": "Demand destruction"},
            {"sector": "Pharma",     "direction": "BUY",  "reason": "Defensive, US-export driven"},
            {"sector": "FMCG",       "direction": "BUY",  "reason": "Domestic defensive"},
        ],
        "vix_action": "MONITOR",
        "crude_action": "NEUTRAL",
    },
]

# Sector → NSE tickers mapping (F&O eligible where possible)
SECTOR_TICKERS: dict[str, list[str]] = {
    "Energy":     ["ONGC.NS", "RELIANCE.NS", "OIL.NS", "CAIRN.NS"],
    "Defence":    ["HAL.NS", "BEL.NS", "BHEL.NS", "MAZDOCK.NS"],
    "Aviation":   ["INDIGO.NS", "SPICEJET.NS"],
    "Paints":     ["ASIANPAINT.NS", "BERGEPAINT.NS", "KANSAINER.NS"],
    "Auto":       ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS"],
    "Banking":    ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS"],
    "NBFC":       ["BAJFINANCE.NS", "BAJAJFINSV.NS", "MUTHOOTFIN.NS"],
    "Realty":     ["DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS"],
    "FMCG":       ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS"],
    "Pharma":     ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS"],
    "IT":         ["INFY.NS", "TCS.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "Metals":     ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "COALINDIA.NS"],
    "Fertilizer": ["CHAMBLFERT.NS", "COROMANDEL.NS", "GNFC.NS"],
    "Tyre":       ["MRF.NS", "APOLLOTYRE.NS", "CEATLTD.NS"],
    "OMC":        ["BPCL.NS", "HPCL.NS", "IOC.NS"],
    "Gold":       ["GOLDBEES.NS"],
    "Broad":      ["NIFTY_INDEX"],   # sentinel for index-level action
    "Semicon":    ["DIXON.NS", "KAYNES.NS"],
    "Cyclical":   ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS"],
}


# ── Event log helpers ─────────────────────────────────────────────────────────

def _load_event_log() -> list:
    if EVENT_LOG_FILE.exists():
        try:
            return json.loads(EVENT_LOG_FILE.read_text())
        except Exception:
            pass
    return []


def _save_event_log(log: list):
    DATA_DIR.mkdir(exist_ok=True)
    EVENT_LOG_FILE.write_text(json.dumps(log, indent=2))


def _already_processed(event_id: str, window_hours: float = 6.0) -> bool:
    """True if this event was already processed within `window_hours`."""
    log = _load_event_log()
    cutoff = (datetime.now(_IST) - timedelta(hours=window_hours)).isoformat()
    for entry in reversed(log):
        if entry.get("event_id") == event_id and entry.get("time", "") >= cutoff:
            return True
    return False


def _log_event(event_id: str, severity: str, headline: str, actions: list):
    log = _load_event_log()
    log.append({
        "event_id":  event_id,
        "severity":  severity,
        "headline":  headline[:300],
        "actions":   actions,
        "time":      datetime.now(_IST).isoformat(),
    })
    # Keep last 200 events
    _save_event_log(log[-200:])


# ── News fetcher ──────────────────────────────────────────────────────────────

_NEWS_SOURCES = [
    "https://news.google.com/rss/search?q=india+stock+market+geopolitical&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=iran+strike+oil+india&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=global+macro+india+market&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=nifty+sensex+news&hl=en-IN&gl=IN&ceid=IN:en",
]


def fetch_recent_headlines(max_age_minutes: int = 30) -> list[str]:
    """Fetch headlines from Google News RSS. Returns list of recent headline strings."""
    headlines = []
    cutoff = datetime.now(_IST) - timedelta(minutes=max_age_minutes)

    for url in _NEWS_SOURCES:
        try:
            resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                continue
            # Parse RSS items
            items = re.findall(r"<item>(.*?)</item>", resp.text, re.DOTALL)
            for item in items:
                title_m = re.search(r"<title><!\[CDATA\[(.*?)\]\]></title>", item)
                if not title_m:
                    title_m = re.search(r"<title>(.*?)</title>", item)
                if not title_m:
                    continue
                title = title_m.group(1).strip()

                # Check pubDate
                date_m = re.search(r"<pubDate>(.*?)</pubDate>", item)
                if date_m:
                    try:
                        from email.utils import parsedate_to_datetime  # noqa: PLC0415
                        pub_dt = parsedate_to_datetime(date_m.group(1))
                        if pub_dt.tzinfo is None:
                            pub_dt = pub_dt.replace(tzinfo=_IST)
                        pub_ist = pub_dt.astimezone(_IST)
                        if pub_ist < cutoff.replace(tzinfo=_IST):
                            continue
                    except Exception:
                        pass

                headlines.append(title.lower())
        except Exception as exc:
            logger.debug(f"[EventEngine] RSS fetch failed for {url}: {exc}")

    return list(set(headlines))


# ── Event detector ────────────────────────────────────────────────────────────

class EventDetector:
    """
    Scans recent headlines for known event patterns.
    Returns list of matched events with their impact profiles.
    """

    def scan(self, headlines: list[str]) -> list[dict]:
        """Match headlines against all event patterns. Returns matched events."""
        full_text = " ".join(headlines)
        matched = []

        for pattern in EVENT_PATTERNS:
            score = sum(1 for kw in pattern["keywords"] if kw in full_text)
            if score == 0:
                continue

            # Deduplicate — don't re-fire within 6 hours
            if _already_processed(pattern["id"]):
                logger.debug(f"[EventEngine] {pattern['id']} already processed within 6h — skipping")
                continue

            # Find triggering headline
            trigger_headline = ""
            for kw in pattern["keywords"]:
                for h in headlines:
                    if kw in h:
                        trigger_headline = h
                        break
                if trigger_headline:
                    break

            matched.append({
                "event_id":        pattern["id"],
                "severity":        pattern["severity"],
                "impact":          pattern["impact"],
                "vix_action":      pattern.get("vix_action", "MONITOR"),
                "crude_action":    pattern.get("crude_action", "NEUTRAL"),
                "trigger_headline": trigger_headline,
                "keyword_hits":    score,
            })
            logger.info(
                f"[EventEngine] 🚨 EVENT DETECTED: {pattern['id']} "
                f"(severity={pattern['severity']}, hits={score})"
            )

        # Sort by severity
        order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
        matched.sort(key=lambda e: order.get(e["severity"], 9))
        return matched


# ── Portfolio action executor ─────────────────────────────────────────────────

class EventActionExecutor:
    """
    Translates event impacts into concrete portfolio actions:
    - Closes equity positions in sectors being sold
    - Buys Nifty PUT hedge for HEDGE vix_action events
    - Logs all actions for Telegram notification
    """

    def execute(self, event: dict, portfolio, fno_portfolio=None) -> list[str]:
        """
        Execute portfolio actions for a detected event.
        Returns list of action strings for Telegram.
        """
        actions = []
        positions = dict(portfolio.state.get("positions", {}))
        impact_list = event.get("impact", [])

        # Build set of SELL sectors for quick lookup
        sell_sectors = {
            imp["sector"] for imp in impact_list if imp["direction"] == "SELL"
        }
        buy_sectors = {
            imp["sector"] for imp in impact_list if imp["direction"] == "BUY"
        }

        # ── 1. Close equity positions in SELL sectors ─────────────────────
        from engine import DataFetcher  # noqa: PLC0415
        for ticker, pos in list(positions.items()):
            # Find which sector this ticker belongs to
            ticker_sector = None
            base = ticker.replace(".NS", "")
            for sector, tickers in SECTOR_TICKERS.items():
                if ticker in tickers or base + ".NS" in tickers:
                    ticker_sector = sector
                    break

            if ticker_sector in sell_sectors:
                price = DataFetcher.get_current_price(ticker) or pos.get("avg_price", 0)
                if price > 0:
                    trade = portfolio.execute_sell(
                        ticker, price,
                        reason=f"EVENT_{event['event_id']}"
                    )
                    if trade:
                        pnl = trade.get("pnl", 0)
                        actions.append(
                            f"📤 SOLD {ticker.replace('.NS','')} @ ₹{price:.0f} "
                            f"({'+'if pnl>=0 else ''}₹{pnl:.0f}) — {ticker_sector} sector sell"
                        )

        # ── 2. Nifty PUT hedge for CRITICAL/HEDGE events ──────────────────
        if event.get("vix_action") == "HEDGE" and fno_portfolio is not None:
            try:
                from fno_engine import (  # noqa: PLC0415
                    get_expiry, BlackScholes, days_to_expiry,
                    select_strike, historical_vol, RISK_FREE_RATE
                )
                from datetime import date as _date  # noqa: PLC0415
                import yfinance as _yf  # noqa: PLC0415

                nifty_df = DataFetcher.fetch("^NSEI", period="5d", interval="1d")
                if nifty_df is not None and not nifty_df.empty:
                    nifty_spot = float(nifty_df["Close"].iloc[-1])
                    expiry = get_expiry("^NSEI", monthly=False)  # weekly
                    T      = days_to_expiry(expiry)
                    iv     = historical_vol("^NSEI")
                    strike = select_strike(nifty_spot, "put", moneyness="OTM1")
                    prem   = BlackScholes.price(nifty_spot, strike, T, RISK_FREE_RATE, iv, "put")

                    if prem >= 10 and not fno_portfolio.has_open_position("^NSEI", "put"):
                        trade = fno_portfolio.open_option(
                            underlying="^NSEI", strike=strike, expiry=expiry,
                            option_type="put", position="LONG", qty_lots=1,
                            strategy="EVENT_HEDGE",
                            reason=f"Hedge: {event['event_id']}"
                        )
                        if trade:
                            actions.append(
                                f"🛡️ HEDGE: Bought Nifty {strike}PE @ ₹{prem:.0f} "
                                f"(1 lot, event hedge, expiry {expiry})"
                            )
            except Exception as _he:
                logger.warning(f"[EventEngine] Hedge execution failed: {_he}")

        # ── 3. Log the event ──────────────────────────────────────────────
        _log_event(
            event_id  = event["event_id"],
            severity  = event["severity"],
            headline  = event["trigger_headline"],
            actions   = actions,
        )

        logger.info(
            f"[EventEngine] {event['event_id']} — executed {len(actions)} action(s)"
        )
        return actions


# ── Main entry point called by the app.py event loop ─────────────────────────

def run_event_scan(portfolio, fno_portfolio=None) -> list[dict]:
    """
    Fetch recent headlines, detect events, execute actions.
    Returns list of {event, actions} dicts for Telegram notification.
    Called every 5 minutes from app.py during market hours.
    """
    from zoneinfo import ZoneInfo as _ZI  # noqa: PLC0415
    from datetime import time as _dtt    # noqa: PLC0415
    now_t = datetime.now(ZoneInfo("Asia/Kolkata")).time()

    # Only run during market hours + 30 min buffer either side
    if not (_dtt(8, 45) <= now_t <= _dtt(16, 0)):
        return []

    headlines = fetch_recent_headlines(max_age_minutes=30)
    if not headlines:
        return []

    detector = EventDetector()
    executor = EventActionExecutor()
    results  = []

    for event in detector.scan(headlines):
        actions = executor.execute(event, portfolio, fno_portfolio)
        results.append({"event": event, "actions": actions})

    return results


def get_recent_events(hours: float = 24.0) -> list[dict]:
    """Return events from the log within the last `hours` hours."""
    log    = _load_event_log()
    cutoff = (datetime.now(_IST) - timedelta(hours=hours)).isoformat()
    return [e for e in log if e.get("time", "") >= cutoff]
