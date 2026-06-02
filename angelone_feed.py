"""
Angel One SmartAPI — Live WebSocket Price Feed
===============================================
Streams real-time NSE tick data for the full NSE equity universe.
On startup the Angel One scrip master JSON is downloaded to build a
dynamic symbol→token map covering all ~2000 NSE EQ stocks.
Subscribes in batches of 50 tokens (SmartStream limit per call).

Fallback: if the scrip master download fails the original 55-stock
hardcoded map is used so the feed still works.

Usage:
    from angelone_feed import LiveFeed
    feed = LiveFeed()          # reads credentials from .env
    feed.start()               # connects and subscribes in background thread
    price = feed.get_price("RELIANCE.NS")   # returns live LTP or None
    feed.stop()
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ── Scrip master URL ──────────────────────────────────────────────────────────
_SCRIP_MASTER_URL = (
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
)

# ── Hardcoded fallback token map (used if scrip master download fails) ────────
# Nifty 50 constituents + sector ETFs — stable tokens sourced from Angel One.
_FALLBACK_CM_TOKENS: dict[str, str] = {
    "RELIANCE.NS":    "2885",
    "TCS.NS":         "11536",
    "HDFCBANK.NS":    "1333",
    "INFY.NS":        "1594",
    "ICICIBANK.NS":   "4963",
    "HINDUNILVR.NS":  "1394",
    "SBIN.NS":        "3045",
    "BHARTIARTL.NS":  "10604",
    "BAJFINANCE.NS":  "317",
    "KOTAKBANK.NS":   "1922",
    "LT.NS":          "11483",
    "AXISBANK.NS":    "5900",
    "ASIANPAINT.NS":  "236",
    "MARUTI.NS":      "10999",
    "SUNPHARMA.NS":   "3351",
    "TITAN.NS":       "3506",
    "WIPRO.NS":       "3787",
    "ULTRACEMCO.NS":  "11532",
    "NESTLEIND.NS":   "17963",
    "POWERGRID.NS":   "14977",
    "NTPC.NS":        "11630",
    "M&M.NS":         "2031",
    "HCLTECH.NS":     "7229",
    "ONGC.NS":        "2475",
    "JSWSTEEL.NS":    "11723",
    "TATAMOTORS.NS":  "3456",
    "ADANIENT.NS":    "25",
    "COALINDIA.NS":   "20374",
    "BAJAJFINSV.NS":  "16675",
    "GRASIM.NS":      "1232",
    "TECHM.NS":       "13538",
    "BPCL.NS":        "526",
    "CIPLA.NS":       "694",
    "DRREDDY.NS":     "881",
    "EICHERMOT.NS":   "910",
    "APOLLOHOSP.NS":  "157",
    "DIVISLAB.NS":    "10940",
    "TATACONSUM.NS":  "3432",
    "INDUSINDBK.NS":  "5258",
    "SBILIFE.NS":     "21808",
    "HDFCLIFE.NS":    "467",
    "ADANIPORTS.NS":  "15083",
    "UPL.NS":         "11287",
    "HEROMOTOCO.NS":  "1348",
    "BRITANNIA.NS":   "547",
    "TATASTEEL.NS":   "3499",
    "ITC.NS":         "1660",
    "BAJAJ-AUTO.NS":  "16669",
    "HINDALCO.NS":    "1363",
    "VEDL.NS":        "3063",
    "NIFTYBEES.NS":   "2714",
    "BANKBEES.NS":    "13269",
    "ITBEES.NS":      "15141",
    "PHARMABEES.NS":  "28413",
    "GOLDBEES.NS":    "8179",
}

# Nifty 50 & Bank Nifty indices — NSE_INDEX segment (exchange type 13)
NSE_INDEX_TOKENS: dict[str, str] = {
    "^NSEI":    "99926000",   # Nifty 50
    "^NSEBANK": "99926009",   # Bank Nifty
}

# ── Dynamic maps (populated at startup from scrip master) ─────────────────────
# NSE_CM_TOKENS  :  { "RELIANCE.NS": "2885", ... }  — ALL NSE EQ stocks
# _TOKEN_TO_TICKER: { "2885": "RELIANCE.NS", ... }  — reverse lookup
NSE_CM_TOKENS:    dict[str, str] = {}
_TOKEN_TO_TICKER: dict[str, str] = {}

_BATCH_SIZE = 50          # SmartStream max tokens per subscribe call
_MAX_STOCKS = 9999        # no practical cap — subscribe to full NSE EQ universe


def _load_scrip_master() -> dict[str, str]:
    """
    Download the Angel One scrip master JSON and return a dict mapping
    yfinance-style tickers ("RELIANCE.NS") to Angel One numeric tokens.

    Only NSE cash-market equities are included (exch_seg=NSE, symbol ends -EQ).
    Falls back to the hardcoded 55-stock map on any error.
    """
    try:
        import requests as req  # type: ignore[import-untyped]
        logger.info("[LiveFeed] Downloading scrip master…")
        resp = req.get(_SCRIP_MASTER_URL, timeout=30)
        resp.raise_for_status()
        records: list[dict] = resp.json()

        result: dict[str, str] = {}
        for item in records:
            if (
                item.get("exch_seg") == "NSE"
                and str(item.get("symbol", "")).endswith("-EQ")
            ):
                raw_symbol = item["symbol"][:-3]        # strip "-EQ"
                token      = str(item["token"]).strip()
                if raw_symbol and token:
                    result[raw_symbol + ".NS"] = token

        count = len(result)
        if count < 10:
            raise ValueError(f"Scrip master returned only {count} records — suspicious")

        logger.info(f"[LiveFeed] Scrip master loaded: {count} NSE EQ stocks")
        return result

    except Exception as exc:
        logger.warning(
            f"[LiveFeed] Scrip master download failed ({exc}) — using fallback {len(_FALLBACK_CM_TOKENS)}-stock map"
        )
        return dict(_FALLBACK_CM_TOKENS)


# ── LiveFeed ──────────────────────────────────────────────────────────────────

class LiveFeed:
    """
    Manages the Angel One SmartStream WebSocket connection.

    Thread-safe price cache:  LiveFeed._prices  { "RELIANCE.NS": 2450.35, ... }
    Change map:               LiveFeed._changes { "RELIANCE.NS": +1.23,   ... }
    """

    _prices:  dict[str, float] = {}
    _changes: dict[str, float] = {}
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._api_key        = os.environ.get("ANGELONE_API_KEY", "")
        self._client_id      = os.environ.get("ANGELONE_CLIENT_ID", "")
        self._password       = os.environ.get("ANGELONE_PASSWORD", "")
        self._totp_secret    = os.environ.get("ANGELONE_TOTP_SECRET", "")
        self._sws            = None
        self._smart          = None
        self._thread         = None
        self._stop_event     = threading.Event()
        self._connected      = False
        self._auth_token     = ""
        self._feed_token     = ""
        self._correlation_id = "trading_agent_live"

    # ── Public API ────────────────────────────────────────────────────────────

    def is_configured(self) -> bool:
        return all([self._api_key, self._client_id, self._password, self._totp_secret])

    def is_connected(self) -> bool:
        return self._connected

    def get_price(self, ticker: str) -> Optional[float]:
        with self._lock:
            return LiveFeed._prices.get(ticker)

    def get_change(self, ticker: str) -> Optional[float]:
        with self._lock:
            return LiveFeed._changes.get(ticker)

    def get_all_prices(self) -> dict[str, float]:
        with self._lock:
            return dict(LiveFeed._prices)

    def get_all_changes(self) -> dict[str, float]:
        with self._lock:
            return dict(LiveFeed._changes)

    def start(self) -> bool:
        """Authenticate and start the WebSocket feed in a daemon thread."""
        if not self.is_configured():
            logger.warning("[LiveFeed] Credentials not set — running in yfinance-only mode")
            return False
        self._thread = threading.Thread(target=self._run, daemon=True, name="angelone-feed")
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._sws:
            try:
                self._sws.close_connection()
            except Exception:
                pass

    # ── Internal ──────────────────────────────────────────────────────────────

    def _generate_totp(self) -> str:
        import pyotp  # type: ignore[import-untyped]
        return pyotp.TOTP(self._totp_secret).now()

    def _authenticate(self) -> bool:
        """Log in to SmartAPI, cache tokens, and load the scrip master."""
        global NSE_CM_TOKENS, _TOKEN_TO_TICKER  # noqa: PLW0603
        try:
            from SmartApi import SmartConnect  # type: ignore[import-untyped]
            self._smart = SmartConnect(api_key=self._api_key)
            totp = self._generate_totp()
            data = self._smart.generateSession(self._client_id, self._password, totp)
            if not data or data.get("status") is False:
                logger.error(f"[LiveFeed] Auth failed: {data}")
                return False
            self._auth_token = data["data"]["jwtToken"]
            self._feed_token = self._smart.getfeedToken()
            logger.info("[LiveFeed] Authenticated with Angel One SmartAPI")

            # Build dynamic token maps from scrip master
            NSE_CM_TOKENS = _load_scrip_master()
            _TOKEN_TO_TICKER = {
                **{v: k for k, v in NSE_CM_TOKENS.items()},
                **{v: k for k, v in NSE_INDEX_TOKENS.items()},
            }
            return True
        except Exception as exc:
            logger.error(f"[LiveFeed] Authentication error: {exc}")
            return False

    def _on_data(self, wsapp, message: str) -> None:  # noqa: ARG002
        try:
            if isinstance(message, bytes):
                message = message.decode("utf-8")
            tick = json.loads(message) if isinstance(message, str) else message
            self._process_tick(tick)
        except Exception as exc:
            logger.debug(f"[LiveFeed] Tick parse error: {exc}")

    def _process_tick(self, tick: dict) -> None:
        try:
            token   = str(tick.get("token", ""))
            ltp     = tick.get("last_traded_price", 0) or tick.get("ltp", 0)
            open_px = tick.get("open_price_of_the_day", 0) or tick.get("open", 0)

            # Angel One sends prices ×100 for CM segment
            ltp     = ltp / 100.0
            open_px = open_px / 100.0 if open_px else ltp

            ticker = _TOKEN_TO_TICKER.get(token)
            if not ticker or ltp <= 0:
                return

            chg_pct = ((ltp / open_px) - 1) * 100 if open_px > 0 else 0.0
            with self._lock:
                LiveFeed._prices[ticker]  = round(ltp, 2)
                LiveFeed._changes[ticker] = round(chg_pct, 4)
        except Exception as exc:
            logger.debug(f"[LiveFeed] Tick process error: {exc}")

    def _on_open(self, wsapp) -> None:  # noqa: ARG002
        self._connected = True
        logger.info("[LiveFeed] WebSocket connected — subscribing to tickers")
        try:
            # Subscribe NSE CM equities in batches of _BATCH_SIZE
            cm_tokens  = list(NSE_CM_TOKENS.values())[:_MAX_STOCKS]
            batches    = [cm_tokens[i:i + _BATCH_SIZE] for i in range(0, len(cm_tokens), _BATCH_SIZE)]
            subscribed = 0
            for batch in batches:
                self._sws.subscribe(
                    correlation_id=self._correlation_id,
                    mode=1,      # LTP only — lowest bandwidth
                    token_list=[{"exchangeType": 1, "tokens": batch}],
                )
                subscribed += len(batch)
                time.sleep(0.05)   # small pause to avoid flooding the socket

            # Subscribe NSE indices
            self._sws.subscribe(
                correlation_id=self._correlation_id,
                mode=1,
                token_list=[{"exchangeType": 13, "tokens": list(NSE_INDEX_TOKENS.values())}],
            )
            logger.info(
                f"[LiveFeed] Subscribed to {subscribed} stocks + {len(NSE_INDEX_TOKENS)} indices"
                f" ({len(batches)} batches)"
            )
        except Exception as exc:
            logger.error(f"[LiveFeed] Subscribe error: {exc}")

    def _on_error(self, wsapp, error) -> None:  # noqa: ARG002
        logger.error(f"[LiveFeed] WebSocket error: {error}")
        self._connected = False

    def _on_close(self, wsapp) -> None:  # noqa: ARG002
        self._connected = False
        logger.warning("[LiveFeed] WebSocket closed")

    def get_historical(self, ticker: str, period: str = "1y", interval: str = "1d") -> "Optional[pd.DataFrame]":
        """
        Fetch OHLCV history from Angel One SmartAPI as a fallback to yfinance.
        Returns a DataFrame with columns [Open, High, Low, Close, Volume] or None on failure.
        """
        if not self._smart or not self._connected:
            return None
        try:
            import pandas as pd  # noqa: PLC0415
            from datetime import datetime, timedelta  # noqa: PLC0415

            # Map ticker to Angel One token
            sym = ticker.replace(".NS", "").replace(".BO", "")
            token = NSE_CM_TOKENS.get(sym) or NSE_CM_TOKENS.get(ticker)
            if not token:
                return None

            # Map period → fromdate
            period_days = {"5d": 5, "1mo": 30, "3mo": 90, "6mo": 180,
                           "1y": 365, "2y": 730, "5y": 1825}.get(period, 365)
            to_dt   = datetime.now()
            from_dt = to_dt - timedelta(days=period_days)

            # Map interval to Angel One resolution
            res_map = {"1m": "ONE_MINUTE", "5m": "FIVE_MINUTE", "15m": "FIFTEEN_MINUTE",
                       "1h": "ONE_HOUR", "1d": "ONE_DAY", "1wk": "ONE_WEEK"}
            resolution = res_map.get(interval, "ONE_DAY")

            params = {
                "exchange": "NSE",
                "symboltoken": token,
                "interval": resolution,
                "fromdate": from_dt.strftime("%Y-%m-%d %H:%M"),
                "todate": to_dt.strftime("%Y-%m-%d %H:%M"),
            }
            resp = self._smart.getCandleData(params)
            if not resp or not resp.get("data"):
                return None

            df = pd.DataFrame(resp["data"], columns=["Date", "Open", "High", "Low", "Close", "Volume"])
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df.dropna()
        except Exception as exc:
            logger.debug(f"[LiveFeed] Historical fetch failed for {ticker}: {exc}")
            return None

    def _run(self) -> None:
        """Main feed loop with exponential-backoff reconnect."""
        backoff = 5
        while not self._stop_event.is_set():
            try:
                if not self._authenticate():
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 300)
                    continue

                from SmartApi.smartWebSocketV2 import SmartWebSocketV2  # type: ignore[import-untyped]
                self._sws = SmartWebSocketV2(
                    auth_token=self._auth_token,
                    api_key=self._api_key,
                    client_code=self._client_id,
                    feed_token=self._feed_token,
                    max_retry_attempt=5,
                )
                self._sws.on_open  = self._on_open
                self._sws.on_data  = self._on_data
                self._sws.on_error = self._on_error
                self._sws.on_close = self._on_close

                logger.info("[LiveFeed] Connecting to SmartStream WebSocket…")
                self._sws.connect()   # blocks until closed

                backoff = 5
            except Exception as exc:
                logger.error(f"[LiveFeed] Feed error: {exc}")
                self._connected = False
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)


# ── Module-level singleton ────────────────────────────────────────────────────

_feed: Optional[LiveFeed] = None


def get_feed() -> LiveFeed:
    """Return the module-level LiveFeed singleton (creates it on first call)."""
    global _feed  # noqa: PLW0603
    if _feed is None:
        _feed = LiveFeed()
    return _feed
