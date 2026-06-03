"""
angelone_trader.py — Live Order Execution via Angel One SmartAPI

Gate: only fires when LIVE_TRADING=true in environment variables.
When LIVE_TRADING is false/missing, every call is a no-op and returns a
simulated response — paper trading continues exactly as before.

Usage:
    from angelone_trader import get_trader
    trader = get_trader()

    # Place a buy order (equity)
    result = trader.buy("RELIANCE.NS", qty=10, price=2850.0)

    # Place a sell order
    result = trader.sell("RELIANCE.NS", qty=10, price=2900.0)

    # Check if live trading is enabled
    if trader.is_live:
        print("LIVE TRADING ACTIVE")

Angel One env vars (same as angelone_feed.py):
    ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_MPIN, ANGEL_TOTP_SECRET
    LIVE_TRADING=true   ← flip this when ready to go live
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger("angelone_trader")
_IST  = ZoneInfo("Asia/Kolkata")

# ── Exchange / variety constants ──────────────────────────────────────────────
_NSE   = "NSE"
_BSE   = "BSE"
_NFO   = "NFO"       # F&O
_MCX   = "MCX"       # Commodities
_CNC   = "CNC"       # Cash-and-carry (delivery equity)
_NRML  = "NRML"     # Normal (F&O / commodity)
_LIMIT = "LIMIT"
_BUY   = "BUY"
_SELL  = "SELL"

# ── Circuit-breaker state ─────────────────────────────────────────────────────
_halt_reason: str | None = None   # non-None → all orders blocked
_day_open_value: float    = 0.0
_orders_today: list       = []
_ORDER_LOG = Path("data/live_orders.json")


def _now_ist() -> datetime:
    return datetime.now(_IST)


def _market_open() -> bool:
    t = _now_ist().time()
    import datetime as _dt
    return _dt.time(9, 15) <= t <= _dt.time(15, 20)


class AngelOneTrader:
    """
    Wraps SmartAPI order placement behind the LIVE_TRADING gate.
    All public methods are safe to call in paper mode — they log and return
    a simulated result dict without touching Angel One.
    """

    def __init__(self):
        self._live: bool = os.getenv("LIVE_TRADING", "false").lower() == "true"
        self._smart = None   # SmartConnect instance, wired in from angelone_feed
        self._daily_loss_limit_pct = 0.03   # halt if portfolio drops >3% from day open

        if self._live:
            logger.warning("⚡ LIVE TRADING MODE ACTIVE — real orders will be placed!")
        else:
            logger.info("📋 Paper trading mode — LIVE_TRADING not set")

    # ── Public gate ───────────────────────────────────────────────────────────

    @property
    def is_live(self) -> bool:
        return self._live

    @property
    def halted(self) -> bool:
        return _halt_reason is not None

    def halt(self, reason: str):
        global _halt_reason
        _halt_reason = reason
        logger.error(f"🛑 TRADING HALTED: {reason}")
        try:
            from notifier import get_notifier
            get_notifier()._tg.send_async(
                f"🛑 <b>TRADING HALTED</b>\n\n{reason}\n\n"
                f"All new orders blocked. Use /resume to lift halt."
            )
        except Exception:
            pass

    def resume(self):
        global _halt_reason
        _halt_reason = None
        logger.info("✅ Trading halt lifted")

    # ── Order placement ───────────────────────────────────────────────────────

    def buy(self, ticker: str, qty: int, price: float,
            segment: str = "equity") -> dict:
        """Place a BUY limit order. Returns order result dict."""
        return self._order(_BUY, ticker, qty, price, segment)

    def sell(self, ticker: str, qty: int, price: float,
             segment: str = "equity") -> dict:
        """Place a SELL limit order. Returns order result dict."""
        return self._order(_SELL, ticker, qty, price, segment)

    def cancel(self, order_id: str, variety: str = "NORMAL") -> dict:
        """Cancel a pending order by order_id."""
        if not self._live:
            return {"ok": True, "paper": True, "order_id": order_id, "status": "CANCELLED"}
        try:
            smart = self._get_smart()
            if not smart:
                return {"ok": False, "error": "SmartAPI not connected"}
            resp = smart.cancelOrder(order_id, variety)
            logger.info(f"[Trader] CANCEL {order_id}: {resp}")
            return {"ok": True, "order_id": order_id, "response": resp}
        except Exception as e:
            logger.error(f"[Trader] Cancel error: {e}")
            return {"ok": False, "error": str(e)}

    def get_order_book(self) -> list:
        """Return today's orders from Angel One."""
        if not self._live:
            return _orders_today
        try:
            smart = self._get_smart()
            if not smart:
                return []
            resp = smart.orderBook()
            return resp.get("data") or []
        except Exception as e:
            logger.error(f"[Trader] orderBook error: {e}")
            return []

    def get_positions(self) -> list:
        """Return open positions from Angel One (net positions)."""
        if not self._live:
            return []
        try:
            smart = self._get_smart()
            if not smart:
                return []
            resp = smart.position()
            return resp.get("data") or []
        except Exception as e:
            logger.error(f"[Trader] positions error: {e}")
            return []

    def get_holdings(self) -> list:
        """Return equity holdings (demat) from Angel One."""
        if not self._live:
            return []
        try:
            smart = self._get_smart()
            if not smart:
                return []
            resp = smart.holding()
            return resp.get("data") or []
        except Exception as e:
            logger.error(f"[Trader] holdings error: {e}")
            return []

    def sync_portfolio(self, agent_portfolio) -> dict:
        """
        Reconcile agent's internal state with actual Angel One positions.
        In paper mode: no-op.
        In live mode: fetches real positions and flags mismatches.
        Returns dict with any discrepancies found.
        """
        if not self._live:
            return {"ok": True, "paper": True, "message": "No sync needed in paper mode"}

        discrepancies = []
        try:
            real_positions = {
                p["tradingsymbol"]: p
                for p in (self.get_positions() or [])
                if int(p.get("netqty", 0)) != 0
            }
            agent_positions = agent_portfolio.state.get("positions", {})

            # Check: in agent but not in Angel One
            for ticker, pos in agent_positions.items():
                symbol = ticker.replace(".NS", "-EQ")
                if symbol not in real_positions:
                    discrepancies.append({
                        "type": "MISSING_IN_BROKER",
                        "ticker": ticker,
                        "agent_qty": pos["qty"],
                    })

            # Check: in Angel One but agent doesn't know about it
            for symbol, pos in real_positions.items():
                ticker = symbol.replace("-EQ", ".NS")
                if ticker not in agent_positions:
                    discrepancies.append({
                        "type": "MISSING_IN_AGENT",
                        "symbol": symbol,
                        "broker_qty": pos["netqty"],
                    })

            if discrepancies:
                logger.warning(f"[Trader] Portfolio sync found {len(discrepancies)} discrepancies")
                for d in discrepancies:
                    logger.warning(f"  {d}")
            else:
                logger.info("[Trader] Portfolio sync: agent matches broker ✓")

        except Exception as e:
            logger.error(f"[Trader] sync_portfolio error: {e}")

        return {"ok": True, "discrepancies": discrepancies}

    # ── Daily loss circuit breaker ────────────────────────────────────────────

    def check_daily_loss(self, current_value: float):
        """
        Call this each agent cycle. Halts trading if daily loss exceeds limit.
        Only active in live mode.
        """
        global _day_open_value
        if not self._live:
            return
        if _day_open_value == 0:
            _day_open_value = current_value
            return
        loss_pct = (_day_open_value - current_value) / _day_open_value
        if loss_pct >= self._daily_loss_limit_pct:
            self.halt(
                f"Daily loss circuit breaker: down {loss_pct*100:.1f}% from open "
                f"(₹{_day_open_value:,.0f} → ₹{current_value:,.0f})"
            )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _order(self, side: str, ticker: str, qty: int, price: float,
               segment: str) -> dict:
        """Core order logic — paper or live."""
        ts   = _now_ist().isoformat()
        tag  = f"{'LIVE' if self._live else 'PAPER'} {side} {ticker} qty={qty} px={price:.2f}"

        # ── Safety gates ──────────────────────────────────────────────────── #
        if _halt_reason:
            logger.warning(f"[Trader] ORDER BLOCKED (halt): {tag}")
            return {"ok": False, "blocked": True, "reason": _halt_reason}

        if not _market_open():
            logger.warning(f"[Trader] ORDER BLOCKED (market closed): {tag}")
            return {"ok": False, "blocked": True, "reason": "market_closed"}

        if qty < 1:
            return {"ok": False, "blocked": True, "reason": "qty<1"}

        # ── Paper mode ────────────────────────────────────────────────────── #
        if not self._live:
            result = {
                "ok": True, "paper": True,
                "side": side, "ticker": ticker, "qty": qty, "price": price,
                "order_id": f"PAPER-{int(time.time()*1000)}",
                "timestamp": ts,
            }
            _orders_today.append(result)
            logger.info(f"[Trader] PAPER {tag}")
            return result

        # ── Live mode ─────────────────────────────────────────────────────── #
        try:
            smart = self._get_smart()
            if not smart:
                return {"ok": False, "error": "SmartAPI not connected"}

            exchange, trading_symbol, product, variety = self._resolve(ticker, segment)

            # Use LIMIT order with 0.1% slippage buffer
            limit_price = round(price * 1.001, 2) if side == _BUY else round(price * 0.999, 2)

            order_params = {
                "variety":         variety,
                "tradingsymbol":   trading_symbol,
                "symboltoken":     self._get_token(trading_symbol, exchange),
                "transactiontype": side,
                "exchange":        exchange,
                "ordertype":       _LIMIT,
                "producttype":     product,
                "duration":        "DAY",
                "price":           str(limit_price),
                "squareoff":       "0",
                "stoploss":        "0",
                "quantity":        str(qty),
            }

            resp = smart.placeOrder(order_params)
            order_id = resp.get("data", {}).get("orderid") if isinstance(resp.get("data"), dict) else None

            result = {
                "ok":       True,
                "live":     True,
                "side":     side,
                "ticker":   ticker,
                "qty":      qty,
                "price":    limit_price,
                "order_id": order_id,
                "response": resp,
                "timestamp": ts,
            }
            _orders_today.append(result)
            self._log_order(result)
            logger.info(f"[Trader] LIVE ORDER PLACED: {tag} → order_id={order_id}")

            # Telegram alert
            try:
                from notifier import get_notifier
                emoji = "📈" if side == _BUY else "📉"
                get_notifier()._tg.send_async(
                    f"{emoji} <b>Live Order Placed</b>\n\n"
                    f"{side} {ticker}\n"
                    f"Qty: {qty} @ ₹{limit_price:,.2f}\n"
                    f"Order ID: {order_id}\n"
                    f"Segment: {segment}\n"
                    f"🕐 {_now_ist().strftime('%H:%M IST')}"
                )
            except Exception:
                pass

            return result

        except Exception as e:
            logger.error(f"[Trader] Live order error for {tag}: {e}")
            return {"ok": False, "error": str(e)}

    def _resolve(self, ticker: str, segment: str) -> tuple:
        """Return (exchange, trading_symbol, product, variety) for Angel One."""
        symbol = ticker.replace(".NS", "").replace(".BO", "")
        if segment == "equity":
            return _NSE, f"{symbol}-EQ", _CNC, "NORMAL"
        elif segment == "fno":
            return _NFO, symbol, _NRML, "NORMAL"
        elif segment == "commodity":
            return _MCX, symbol, _NRML, "NORMAL"
        return _NSE, f"{symbol}-EQ", _CNC, "NORMAL"

    def _get_token(self, trading_symbol: str, exchange: str) -> str:
        """Look up Angel One token from scrip master."""
        try:
            from angelone_feed import NSE_CM_TOKENS
            return str(NSE_CM_TOKENS.get(trading_symbol.replace("-EQ", ""), ""))
        except Exception:
            return ""

    def _get_smart(self):
        """Get the authenticated SmartConnect object from the live feed."""
        if self._smart:
            return self._smart
        try:
            from angelone_feed import get_feed
            feed = get_feed()
            if hasattr(feed, "_smart") and feed._smart:
                self._smart = feed._smart
                return self._smart
        except Exception:
            pass
        logger.error("[Trader] SmartAPI not available — is Angel One feed connected?")
        return None

    def _log_order(self, order: dict):
        """Persist live orders to disk for audit trail."""
        try:
            _ORDER_LOG.parent.mkdir(exist_ok=True)
            log = []
            if _ORDER_LOG.exists():
                import json
                log = json.loads(_ORDER_LOG.read_text())
            log.append(order)
            import json
            _ORDER_LOG.write_text(json.dumps(log[-1000:], indent=2))
        except Exception:
            pass


# ── Singleton ─────────────────────────────────────────────────────────────────
_trader: AngelOneTrader | None = None


def get_trader() -> AngelOneTrader:
    global _trader
    if _trader is None:
        _trader = AngelOneTrader()
    return _trader
