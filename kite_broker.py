"""
Kite Connect Broker Integration
================================
Wraps the Zerodha Kite Connect API for live order placement.

PAPER_MODE (default: True)
--------------------------
When PAPER_MODE=True, all orders are logged but NOT sent to Zerodha.
This lets you verify the system is generating correct signals before
flipping to live. Switch to live by setting env var:
  KITE_PAPER_MODE=false

Architecture
------------
- get_kite()           — returns authenticated KiteConnect instance (or None)
- KiteBroker           — main class, singleton via get_broker()
  - get_price(ticker)  — live LTP from Kite, falls back to yfinance
  - place_order(...)   — places real or paper order
  - get_positions()    — live Zerodha positions
  - get_margins()      — available funds
  - cancel_order(...)  — cancel a pending order
  - is_connected()     — True if Kite session is active

All order calls check PAPER_MODE before executing.
"""

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

KITE_API_KEY    = os.environ.get("KITE_API_KEY", "")
KITE_PAPER_MODE = os.environ.get("KITE_PAPER_MODE", "true").lower() == "true"

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
PAPER_ORDERS_FILE = DATA_DIR / "kite_paper_orders.json"

# ---------------------------------------------------------------------------
# Singleton KiteConnect session
# ---------------------------------------------------------------------------

_kite_instance = None
_kite_lock = threading.Lock()


def get_kite():
    """
    Return an authenticated KiteConnect instance, or None if not configured.
    Lazy-initialises on first call.
    """
    global _kite_instance
    if _kite_instance is not None:
        return _kite_instance

    if not KITE_API_KEY:
        return None

    try:
        from kiteconnect import KiteConnect
        from kite_auth import get_access_token
    except ImportError:
        logger.warning("[KiteBroker] kiteconnect package not installed. Run: pip install kiteconnect")
        return None

    access_token = get_access_token()
    if not access_token:
        logger.warning("[KiteBroker] No valid access token — Kite not connected")
        return None

    with _kite_lock:
        if _kite_instance is None:
            kite = KiteConnect(api_key=KITE_API_KEY)
            kite.set_access_token(access_token)
            _kite_instance = kite
            logger.info("[KiteBroker] KiteConnect session initialised")

    return _kite_instance


def reset_kite_session() -> None:
    """Force re-initialisation of the Kite session (call after token refresh)."""
    global _kite_instance
    with _kite_lock:
        _kite_instance = None
    logger.info("[KiteBroker] Kite session reset")


# ---------------------------------------------------------------------------
# NSE ticker conversion
# ---------------------------------------------------------------------------

def to_kite_symbol(ticker: str) -> str:
    """Convert engine-style ticker (RELIANCE.NS) to Kite symbol (RELIANCE)."""
    return ticker.replace(".NS", "").replace(".BO", "").upper()


def to_engine_ticker(symbol: str) -> str:
    """Convert Kite symbol (RELIANCE) to engine-style (RELIANCE.NS)."""
    if "." not in symbol:
        return symbol + ".NS"
    return symbol


# ---------------------------------------------------------------------------
# KiteBroker
# ---------------------------------------------------------------------------

class KiteBroker:
    """
    Main broker interface. All methods are safe to call even when Kite
    is not configured — they fall back to yfinance / paper trading.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._price_cache: dict = {}
        self._price_cache_ts: dict = {}
        self._PRICE_CACHE_SEC = 10  # cache LTP for 10 seconds

    # ------------------------------------------------------------------
    # Connection status
    # ------------------------------------------------------------------

    def is_connected(self) -> bool:
        """True if Kite session is active and token is valid."""
        if not KITE_API_KEY:
            return False
        try:
            from kite_auth import is_token_valid
            if not is_token_valid():
                return False
            return get_kite() is not None
        except Exception:
            return False

    def connection_status(self) -> dict:
        """Detailed connection info for the dashboard."""
        try:
            from kite_auth import token_info
            info = token_info()
        except Exception:
            info = {"valid": False}

        connected = self.is_connected()
        return {
            "connected": connected,
            "paper_mode": KITE_PAPER_MODE,
            "mode_label": "PAPER" if KITE_PAPER_MODE else "🔴 LIVE",
            "api_key_set": bool(KITE_API_KEY),
            "token_valid": info.get("valid", False),
            "token_masked": info.get("access_token"),
            "token_generated_at": info.get("generated_at"),
            "token_expires_at": info.get("expires_at"),
            "credentials_configured": info.get("credentials_configured", False),
            "status": "CONNECTED" if connected else ("CONFIGURED_NO_TOKEN" if KITE_API_KEY else "NOT_CONFIGURED"),
        }

    # ------------------------------------------------------------------
    # Live price (with 10s cache + yfinance fallback)
    # ------------------------------------------------------------------

    def get_price(self, ticker: str) -> float | None:
        """
        Get live LTP for a ticker.
        Priority: Kite LTP → yfinance fallback
        """
        kite_sym = to_kite_symbol(ticker)
        now_ts = datetime.now().timestamp()

        # Check cache
        if kite_sym in self._price_cache:
            if now_ts - self._price_cache_ts.get(kite_sym, 0) < self._PRICE_CACHE_SEC:
                return self._price_cache[kite_sym]

        # Try Kite
        kite = get_kite()
        if kite:
            try:
                quote = kite.ltp([f"NSE:{kite_sym}"])
                ltp = quote[f"NSE:{kite_sym}"]["last_price"]
                if ltp and ltp > 0:
                    with self._lock:
                        self._price_cache[kite_sym] = float(ltp)
                        self._price_cache_ts[kite_sym] = now_ts
                    return float(ltp)
            except Exception as e:
                logger.debug(f"[KiteBroker] Kite price fetch failed for {kite_sym}: {e}")

        # Fallback to yfinance
        try:
            from engine import DataFetcher
            price = DataFetcher.get_current_price(ticker)
            if price and price > 0:
                return float(price)
        except Exception:
            pass

        return None

    def get_prices_bulk(self, tickers: list) -> dict:
        """Get LTP for multiple tickers in one Kite API call."""
        kite = get_kite()
        if kite and tickers:
            try:
                symbols = [f"NSE:{to_kite_symbol(t)}" for t in tickers]
                quotes = kite.ltp(symbols)
                result = {}
                now_ts = datetime.now().timestamp()
                for t in tickers:
                    sym = f"NSE:{to_kite_symbol(t)}"
                    if sym in quotes:
                        ltp = float(quotes[sym]["last_price"])
                        result[t] = ltp
                        with self._lock:
                            self._price_cache[to_kite_symbol(t)] = ltp
                            self._price_cache_ts[to_kite_symbol(t)] = now_ts
                return result
            except Exception as e:
                logger.debug(f"[KiteBroker] Bulk price fetch failed: {e}")

        # Fallback: individual yfinance calls
        result = {}
        for t in tickers:
            try:
                from engine import DataFetcher
                price = DataFetcher.get_current_price(t)
                if price:
                    result[t] = price
            except Exception:
                pass
        return result

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_order(
        self,
        ticker: str,
        qty: int,
        action: str,          # "BUY" or "SELL"
        order_type: str = "MARKET",   # "MARKET" or "LIMIT"
        price: float = 0.0,
        product: str = "CNC",          # "CNC" (delivery) or "MIS" (intraday)
        tag: str = "",
    ) -> dict:
        """
        Place a real or paper order.

        In PAPER_MODE: logs the order to kite_paper_orders.json.
        In LIVE mode:  sends the order to Zerodha via Kite API.

        Returns order dict with order_id, status, mode.
        """
        kite_sym = to_kite_symbol(ticker)
        order_rec = {
            "timestamp": datetime.now(IST).isoformat(),
            "ticker": ticker,
            "symbol": kite_sym,
            "action": action.upper(),
            "qty": qty,
            "order_type": order_type,
            "price": price,
            "product": product,
            "tag": tag,
            "mode": "PAPER" if KITE_PAPER_MODE else "LIVE",
            "status": "PENDING",
            "order_id": None,
            "error": None,
        }

        if KITE_PAPER_MODE:
            # Paper mode: log only
            order_rec["status"] = "PAPER_EXECUTED"
            order_rec["order_id"] = f"PAPER_{int(datetime.now().timestamp())}"
            self._log_paper_order(order_rec)
            logger.info(
                f"[KiteBroker] 📝 PAPER ORDER: {action} {qty} × {kite_sym} "
                f"@ {'MARKET' if order_type == 'MARKET' else f'₹{price:.2f}'}"
            )
        else:
            # LIVE mode: send to Zerodha
            kite = get_kite()
            if not kite:
                order_rec["status"] = "FAILED"
                order_rec["error"] = "Kite not connected"
                logger.error("[KiteBroker] Cannot place live order — Kite not connected")
                return order_rec

            try:
                from kiteconnect import KiteConnect
                kite_action = KiteConnect.TRANSACTION_TYPE_BUY if action.upper() == "BUY" else KiteConnect.TRANSACTION_TYPE_SELL
                kite_order_type = KiteConnect.ORDER_TYPE_MARKET if order_type == "MARKET" else KiteConnect.ORDER_TYPE_LIMIT
                kite_product = KiteConnect.PRODUCT_CNC if product == "CNC" else KiteConnect.PRODUCT_MIS

                order_id = kite.place_order(
                    tradingsymbol=kite_sym,
                    exchange=KiteConnect.EXCHANGE_NSE,
                    transaction_type=kite_action,
                    quantity=qty,
                    order_type=kite_order_type,
                    product=kite_product,
                    price=price if order_type == "LIMIT" else None,
                    tag=tag[:20] if tag else None,
                    variety=KiteConnect.VARIETY_REGULAR,
                )
                order_rec["order_id"] = order_id
                order_rec["status"] = "PLACED"
                logger.info(f"[KiteBroker] ✅ LIVE ORDER PLACED: {action} {qty} × {kite_sym} | order_id={order_id}")

            except Exception as e:
                order_rec["status"] = "FAILED"
                order_rec["error"] = str(e)
                logger.error(f"[KiteBroker] ❌ Order failed: {e}")

        return order_rec

    def cancel_order(self, order_id: str, variety: str = "regular") -> bool:
        """Cancel a pending order."""
        if KITE_PAPER_MODE:
            logger.info(f"[KiteBroker] PAPER: cancel order {order_id}")
            return True
        kite = get_kite()
        if not kite:
            return False
        try:
            kite.cancel_order(variety=variety, order_id=order_id)
            logger.info(f"[KiteBroker] Cancelled order {order_id}")
            return True
        except Exception as e:
            logger.error(f"[KiteBroker] Cancel failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Portfolio / positions
    # ------------------------------------------------------------------

    def get_positions(self) -> dict:
        """
        Get current Zerodha positions.
        Returns {"net": [...], "day": [...]} or empty on failure.
        """
        kite = get_kite()
        if not kite:
            return {"net": [], "day": [], "source": "none"}
        try:
            positions = kite.positions()
            return {
                "net": positions.get("net", []),
                "day": positions.get("day", []),
                "source": "kite",
            }
        except Exception as e:
            logger.warning(f"[KiteBroker] get_positions failed: {e}")
            return {"net": [], "day": [], "source": "error", "error": str(e)}

    def get_holdings(self) -> list:
        """Get long-term holdings from Zerodha demat."""
        kite = get_kite()
        if not kite:
            return []
        try:
            return kite.holdings()
        except Exception as e:
            logger.warning(f"[KiteBroker] get_holdings failed: {e}")
            return []

    def get_margins(self) -> dict:
        """Get available margin / cash from Zerodha."""
        kite = get_kite()
        if not kite:
            return {}
        try:
            margins = kite.margins()
            equity = margins.get("equity", {})
            return {
                "available_cash": equity.get("available", {}).get("cash", 0),
                "used_margin": equity.get("utilised", {}).get("debits", 0),
                "net": equity.get("net", 0),
                "source": "kite",
            }
        except Exception as e:
            logger.warning(f"[KiteBroker] get_margins failed: {e}")
            return {"source": "error", "error": str(e)}

    def get_orders(self) -> list:
        """Get today's orders from Zerodha."""
        kite = get_kite()
        if not kite:
            return []
        try:
            return kite.orders()
        except Exception as e:
            logger.warning(f"[KiteBroker] get_orders failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Paper order log
    # ------------------------------------------------------------------

    def _log_paper_order(self, order: dict) -> None:
        """Append paper order to the paper orders log file."""
        try:
            orders = []
            if PAPER_ORDERS_FILE.exists():
                with open(PAPER_ORDERS_FILE) as f:
                    orders = json.load(f)
            orders.append(order)
            # Keep last 500
            if len(orders) > 500:
                orders = orders[-500:]
            with open(PAPER_ORDERS_FILE, "w") as f:
                json.dump(orders, f, indent=2)
        except Exception as e:
            logger.warning(f"[KiteBroker] Could not log paper order: {e}")

    def get_paper_orders(self, limit: int = 50) -> list:
        """Return recent paper orders."""
        try:
            if PAPER_ORDERS_FILE.exists():
                with open(PAPER_ORDERS_FILE) as f:
                    orders = json.load(f)
                return list(reversed(orders[-limit:]))
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------
    # Dashboard data
    # ------------------------------------------------------------------

    def get_dashboard_data(self) -> dict:
        """All broker data for the dashboard page."""
        status = self.connection_status()
        paper_orders = self.get_paper_orders(20)

        margins = {}
        positions = {"net": [], "day": []}
        orders = []

        if status["connected"]:
            try:
                margins = self.get_margins()
                positions = self.get_positions()
                orders = self.get_orders()
            except Exception as e:
                logger.warning(f"[KiteBroker] Dashboard data fetch error: {e}")

        return {
            "status": status,
            "margins": margins,
            "positions": positions,
            "orders": orders,
            "paper_orders": paper_orders,
            "last_updated": datetime.now(IST).isoformat(),
        }


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_BROKER_INSTANCE: KiteBroker | None = None
_BROKER_LOCK = threading.Lock()


def get_broker() -> KiteBroker:
    global _BROKER_INSTANCE
    if _BROKER_INSTANCE is None:
        with _BROKER_LOCK:
            if _BROKER_INSTANCE is None:
                _BROKER_INSTANCE = KiteBroker()
    return _BROKER_INSTANCE
