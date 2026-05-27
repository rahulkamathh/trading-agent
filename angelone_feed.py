"""
Angel One SmartAPI — Live WebSocket Price Feed
===============================================
Streams real-time NSE tick data for all Nifty 50 stocks + sector ETFs + indices.
Maintains a thread-safe in-memory price cache that the rest of the app reads.

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

# ── Instrument token map ──────────────────────────────────────────────────────
# Angel One uses numeric tokens to identify instruments on the exchange.
# Exchange type 1 = NSE CM (cash market equities + ETFs)
# These tokens are stable; sourced from Angel One's instrument master.
# Indices use a special token on NSE_INDEX segment (exchange type 13).

NSE_CM_TOKENS: dict[str, str] = {
    # Nifty 50 constituents
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
    # Sector ETFs
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

# Reverse map: token → ticker (for decoding WebSocket messages)
_TOKEN_TO_TICKER: dict[str, str] = {
    **{v: k for k, v in NSE_CM_TOKENS.items()},
    **{v: k for k, v in NSE_INDEX_TOKENS.items()},
}

# ── LiveFeed ──────────────────────────────────────────────────────────────────

class LiveFeed:
    """
    Manages the Angel One SmartStream WebSocket connection.

    Thread-safe price cache:  LiveFeed._prices  { "RELIANCE.NS": 2450.35, ... }
    Change map:               LiveFeed._changes { "RELIANCE.NS": +1.23,   ... }  (% change vs day open)
    """

    _prices:  dict[str, float] = {}
    _changes: dict[str, float] = {}
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._api_key    = os.environ.get("ANGELONE_API_KEY", "")
        self._client_id  = os.environ.get("ANGELONE_CLIENT_ID", "")
        self._password   = os.environ.get("ANGELONE_PASSWORD", "")
        self._totp_secret = os.environ.get("ANGELONE_TOTP_SECRET", "")
        self._sws        = None
        self._smart      = None
        self._thread     = None
        self._stop_event = threading.Event()
        self._connected  = False
        self._auth_token = ""
        self._feed_token = ""
        self._correlation_id = "trading_agent_live"

    # ── Public API ────────────────────────────────────────────────────────────

    def is_configured(self) -> bool:
        """Return True if all 4 credentials are present in env."""
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
        """Generate current TOTP from secret using pyotp."""
        import pyotp  # type: ignore[import-untyped]
        return pyotp.TOTP(self._totp_secret).now()

    def _authenticate(self) -> bool:
        """Log in to SmartAPI and cache auth + feed tokens."""
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
            return True
        except Exception as exc:
            logger.error(f"[LiveFeed] Authentication error: {exc}")
            return False

    def _build_subscription(self) -> list[dict]:
        """Build the subscribe payload for SmartWebSocketV2."""
        return [
            {
                "exchangeType": 1,   # NSE CM
                "tokens": list(NSE_CM_TOKENS.values()),
            },
            {
                "exchangeType": 13,  # NSE Index
                "tokens": list(NSE_INDEX_TOKENS.values()),
            },
        ]

    def _on_data(self, wsapp, message: str) -> None:  # noqa: ARG002
        """Handle incoming tick messages from SmartStream."""
        try:
            if isinstance(message, bytes):
                message = message.decode("utf-8")
            tick = json.loads(message) if isinstance(message, str) else message
            # SmartStreamV2 sends dicts directly when parsed
            self._process_tick(tick)
        except Exception as exc:
            logger.debug(f"[LiveFeed] Tick parse error: {exc}")

    def _process_tick(self, tick: dict) -> None:
        """Update the in-memory price cache from a decoded tick."""
        try:
            token   = str(tick.get("token", ""))
            ltp     = tick.get("last_traded_price", 0) or tick.get("ltp", 0)
            open_px = tick.get("open_price_of_the_day", 0) or tick.get("open", 0)

            # Angel One sends prices multiplied by 100 for CM segment
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
            token_list = self._build_subscription()
            self._sws.subscribe(
                correlation_id=self._correlation_id,
                mode=1,          # Mode 1 = LTP only (fastest)
                token_list=token_list,
            )
            logger.info(f"[LiveFeed] Subscribed to {len(NSE_CM_TOKENS)} stocks + 2 indices")
        except Exception as exc:
            logger.error(f"[LiveFeed] Subscribe error: {exc}")

    def _on_error(self, wsapp, error) -> None:  # noqa: ARG002
        logger.error(f"[LiveFeed] WebSocket error: {error}")
        self._connected = False

    def _on_close(self, wsapp) -> None:  # noqa: ARG002
        self._connected = False
        logger.warning("[LiveFeed] WebSocket closed")

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
                self._sws.on_open    = self._on_open
                self._sws.on_data    = self._on_data
                self._sws.on_error   = self._on_error
                self._sws.on_close   = self._on_close

                logger.info("[LiveFeed] Connecting to SmartStream WebSocket…")
                self._sws.connect()       # blocks until closed

                backoff = 5  # reset on clean disconnect
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
