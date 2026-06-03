"""
commodity_engine.py — MCX Commodity Paper Trading Desk

Trades Gold, Silver, Crude Oil, Natural Gas, Copper on MCX (paper).
Prices via yfinance international futures, converted to INR.
Event-driven signals from event_engine + technical signals.

MCX lot sizes (standard):
  Gold        1 kg      (~₹90,000 contract value)
  Silver      30 kg     (~₹1,05,000 contract value)
  Crude Oil   100 bbl   (~₹6,50,000 contract value)
  Natural Gas 1250 mmBtu (~₹2,00,000 contract value)
  Copper      2500 kg   (~₹4,50,000 contract value)
  Zinc        5000 kg   (~₹1,50,000 contract value)

Margin: ~5–10% of contract value (MCX SPAN margin approximation)
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("commodity_engine")
_IST = ZoneInfo("Asia/Kolkata")

DATA_DIR              = Path(__file__).parent / "data"
COMMODITY_PORT_FILE   = DATA_DIR / "commodity_portfolio.json"
COMMODITY_TRADE_FILE  = DATA_DIR / "commodity_trades.json"
COMMODITY_SIGNAL_FILE = DATA_DIR / "commodity_signals.json"
DATA_DIR.mkdir(exist_ok=True)

COMMODITY_CAPITAL = 2_00_000   # ₹2 lakhs starting capital
MAX_COMMODITY_POSITIONS = 6    # max concurrent commodity positions
MAX_RISK_PER_TRADE = 0.10      # max 10% of capital per trade (margin)

# ── Commodity definitions ─────────────────────────────────────────────────────

COMMODITIES = {
    "GOLD": {
        "name":         "Gold",
        "yf_ticker":    "GC=F",       # COMEX Gold futures
        "mcx_symbol":   "GOLDM",      # MCX Mini Gold
        "unit":         "per 10g",
        "lot_size":     10,           # grams (mini lot)
        "lot_value_usd": 0.32,        # 10g in troy oz (~0.32 oz)
        "margin_pct":   0.05,         # ~5% margin
        "currency":     "USD",
        "usd_multiplier": 0.0321507,  # troy oz per gram
        "emoji":        "🥇",
    },
    "SILVER": {
        "name":         "Silver",
        "yf_ticker":    "SI=F",
        "mcx_symbol":   "SILVERMIC",
        "unit":         "per kg",
        "lot_size":     1,            # kg (mini lot)
        "lot_value_usd": 0.0321507 * 1000,  # 1kg in troy oz
        "margin_pct":   0.05,
        "currency":     "USD",
        "usd_multiplier": 0.0321507 * 1000,
        "emoji":        "🥈",
    },
    "CRUDE": {
        "name":         "Crude Oil",
        "yf_ticker":    "CL=F",       # WTI Crude futures
        "mcx_symbol":   "CRUDEOIL",
        "unit":         "per bbl",
        "lot_size":     10,           # barrels (mini lot)
        "lot_value_usd": 10,          # 10 bbl
        "margin_pct":   0.08,
        "currency":     "USD",
        "usd_multiplier": 1,
        "emoji":        "🛢️",
    },
    "NATGAS": {
        "name":         "Natural Gas",
        "yf_ticker":    "NG=F",
        "mcx_symbol":   "NATURALGAS",
        "unit":         "per mmBtu",
        "lot_size":     250,          # mmBtu (mini lot)
        "lot_value_usd": 250,
        "margin_pct":   0.10,
        "currency":     "USD",
        "usd_multiplier": 1,
        "emoji":        "🔥",
    },
    "COPPER": {
        "name":         "Copper",
        "yf_ticker":    "HG=F",       # COMEX Copper futures (USD per lb)
        "mcx_symbol":   "COPPER",
        "unit":         "per kg",
        "lot_size":     250,          # kg (mini lot)
        "lot_value_usd": 0.453592 * 250,  # kg to lb conversion * 250
        "margin_pct":   0.06,
        "currency":     "USD",
        "usd_multiplier": 0.453592 * 250,
        "emoji":        "🔶",
    },
    "ZINC": {
        "name":         "Zinc",
        "yf_ticker":    "ZNC=F",      # Zinc futures
        "mcx_symbol":   "ZINC",
        "unit":         "per kg",
        "lot_size":     1000,         # kg (mini lot)
        "lot_value_usd": 0.453592 * 1000,
        "margin_pct":   0.06,
        "currency":     "USD",
        "usd_multiplier": 0.453592 * 1000,
        "emoji":        "⚙️",
    },
}

# Event → commodity impact (mirrors event_engine.py patterns)
EVENT_COMMODITY_MAP = {
    "MIDDLE_EAST_CONFLICT": {
        "CRUDE":  "BUY",
        "GOLD":   "BUY",
        "SILVER": "BUY",
        "NATGAS": "BUY",
    },
    "US_MILITARY_ACTION": {
        "CRUDE":  "BUY",
        "GOLD":   "BUY",
        "NATGAS": "BUY",
    },
    "RUSSIA_UKRAINE": {
        "CRUDE":  "BUY",
        "NATGAS": "BUY",
        "GOLD":   "BUY",
        "COPPER": "SELL",
    },
    "CHINA_TAIWAN": {
        "GOLD":   "BUY",
        "COPPER": "SELL",
        "CRUDE":  "NEUTRAL",
    },
    "FII_SELLOFF": {
        "GOLD":   "BUY",
        "SILVER": "BUY",
        "CRUDE":  "SELL",
    },
    "CRUDE_SPIKE": {
        "CRUDE":  "BUY",
        "NATGAS": "BUY",
        "GOLD":   "BUY",
    },
    "CRUDE_CRASH": {
        "CRUDE":  "SELL",
        "NATGAS": "SELL",
        "COPPER": "BUY",
    },
    "FED_RATE_HIKE": {
        "GOLD":   "SELL",
        "SILVER": "SELL",
        "COPPER": "SELL",
    },
    "RBI_RATE_CUT": {
        "GOLD":   "BUY",
        "SILVER": "BUY",
    },
    "US_RECESSION_FEAR": {
        "GOLD":   "BUY",
        "CRUDE":  "SELL",
        "COPPER": "SELL",
    },
}


# ── Price helpers ─────────────────────────────────────────────────────────────

_price_cache: dict[str, tuple[float, float]] = {}  # ticker → (price, timestamp)
_PRICE_TTL = 300  # 5 minutes


def _get_usd_inr() -> float:
    """Fetch current USD/INR rate."""
    key = "USDINR"
    if key in _price_cache:
        price, ts = _price_cache[key]
        if time.time() - ts < _PRICE_TTL:
            return price
    try:
        df = yf.download("INR=X", period="2d", interval="1d",
                         auto_adjust=True, progress=False)
        if not df.empty:
            rate = float(df["Close"].iloc[-1])
            _price_cache[key] = (rate, time.time())
            return rate
    except Exception:
        pass
    return 84.0  # fallback


def get_commodity_price_inr(symbol: str) -> Optional[float]:
    """
    Fetch commodity price in INR per standard unit.
    Returns None on failure.
    """
    if symbol not in COMMODITIES:
        return None
    defn = COMMODITIES[symbol]
    key  = defn["yf_ticker"]

    if key in _price_cache:
        price, ts = _price_cache[key]
        if time.time() - ts < _PRICE_TTL:
            return price

    try:
        df = yf.download(key, period="5d", interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            return None
        usd_price = float(df["Close"].iloc[-1])
        usd_inr   = _get_usd_inr()
        inr_price = usd_price * usd_inr

        # For gold/silver: convert from troy oz price to per gram/kg
        if symbol == "GOLD":
            # GC=F is USD per troy oz → convert to INR per 10g
            inr_price = usd_price * usd_inr * 0.321507   # per 10g

        _price_cache[key] = (inr_price, time.time())
        logger.debug(f"[Commodity] {symbol} @ ₹{inr_price:,.0f} (USD {usd_price:.2f})")
        return round(inr_price, 2)

    except Exception as exc:
        logger.warning(f"[Commodity] Price fetch failed for {symbol}: {exc}")
        return None


def get_all_prices() -> dict[str, float]:
    """Fetch prices for all commodities. Returns {symbol: inr_price}."""
    prices = {}
    for sym in COMMODITIES:
        p = get_commodity_price_inr(sym)
        if p:
            prices[sym] = p
    return prices


# ── Technical signal generator ────────────────────────────────────────────────

def _fetch_ohlcv(yf_ticker: str, days: int = 60) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(yf_ticker, period=f"{days}d", interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 10:
            return None
        return df.dropna()
    except Exception:
        return None


def generate_technical_signals() -> list[dict]:
    """
    Run momentum + RSI + trend signals on all commodities.
    Returns list of signal dicts.
    """
    signals = []
    usd_inr = _get_usd_inr()

    for symbol, defn in COMMODITIES.items():
        df = _fetch_ohlcv(defn["yf_ticker"], days=90)
        if df is None or len(df) < 20:
            continue

        close = df["Close"]
        price_inr = get_commodity_price_inr(symbol)
        if not price_inr:
            continue

        # Indicators
        ma20  = float(close.rolling(20).mean().iloc[-1])
        ma50  = float(close.rolling(50).mean().iloc[-1]) if len(df) >= 50 else ma20
        curr  = float(close.iloc[-1])

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = float((100 - 100 / (1 + rs)).iloc[-1])

        # 20-day return
        ret20 = (curr - float(close.iloc[-21])) / float(close.iloc[-21]) if len(df) >= 21 else 0

        # Signal logic
        signal    = "NEUTRAL"
        strength  = 50
        reason    = ""

        if curr > ma20 > ma50 and rsi < 70 and ret20 > 0.01:
            signal   = "BUY"
            strength = min(90, 55 + int(ret20 * 300) + (5 if rsi < 55 else 0))
            reason   = f"Price above MA20/MA50, RSI {rsi:.0f}, 20d return {ret20:.1%}"
        elif curr < ma20 < ma50 and rsi > 30 and ret20 < -0.01:
            signal   = "SELL"
            strength = min(90, 55 + int(-ret20 * 300) + (5 if rsi > 65 else 0))
            reason   = f"Price below MA20/MA50, RSI {rsi:.0f}, 20d return {ret20:.1%}"

        if signal != "NEUTRAL":
            signals.append({
                "symbol":    symbol,
                "name":      defn["name"],
                "signal":    signal,
                "strength":  strength,
                "price_inr": price_inr,
                "rsi":       round(rsi, 1),
                "ma20_inr":  round(ma20 * usd_inr, 0) if symbol not in ("GOLD", "SILVER") else round(ma20 * usd_inr * 0.321507, 0),
                "ret20":     round(ret20 * 100, 2),
                "reason":    reason,
                "source":    "TECHNICAL",
                "generated_at": datetime.now(_IST).isoformat(),
            })

    return signals


def generate_event_signals(active_events: list[dict]) -> list[dict]:
    """Generate commodity signals from detected geopolitical/macro events."""
    signals = []
    for event in active_events:
        event_id = event.get("event_id", "")
        impact   = EVENT_COMMODITY_MAP.get(event_id, {})
        for symbol, direction in impact.items():
            if direction == "NEUTRAL":
                continue
            price_inr = get_commodity_price_inr(symbol)
            if not price_inr:
                continue
            signals.append({
                "symbol":    symbol,
                "name":      COMMODITIES[symbol]["name"],
                "signal":    direction,
                "strength":  85 if event.get("severity") == "CRITICAL" else 75,
                "price_inr": price_inr,
                "rsi":       None,
                "reason":    f"Event: {event_id.replace('_',' ').title()} ({event.get('severity','')})",
                "source":    "EVENT",
                "event_id":  event_id,
                "generated_at": datetime.now(_IST).isoformat(),
            })
    return signals


# ── Commodity portfolio ───────────────────────────────────────────────────────

class CommodityPortfolio:
    """
    Paper-trading commodity portfolio.
    Tracks positions in terms of lots, with INR P&L.
    """

    def __init__(self):
        self.state = self._load()

    def _default_state(self) -> dict:
        return {
            "cash":         COMMODITY_CAPITAL,
            "initial":      COMMODITY_CAPITAL,
            "positions":    {},
            "realised_pnl": 0.0,
            "created_at":   datetime.now(_IST).isoformat(),
        }

    def _load(self) -> dict:
        if COMMODITY_PORT_FILE.exists():
            try:
                return json.loads(COMMODITY_PORT_FILE.read_text())
            except Exception:
                pass
        state = self._default_state()
        self._save(state)
        return state

    def _save(self, state: dict = None):
        if state:
            self.state = state
        self.state["last_updated"] = datetime.now(_IST).isoformat()
        COMMODITY_PORT_FILE.write_text(json.dumps(self.state, indent=2))

    def _log_trade(self, trade: dict):
        log = []
        if COMMODITY_TRADE_FILE.exists():
            try:
                log = json.loads(COMMODITY_TRADE_FILE.read_text())
            except Exception:
                pass
        trade["id"]   = len(log) + 1
        trade["time"] = datetime.now(_IST).isoformat()
        log.append(trade)
        COMMODITY_TRADE_FILE.write_text(json.dumps(log[-500:], indent=2))

    def get_unrealised_pnl(self) -> float:
        total = 0.0
        for sym, pos in self.state["positions"].items():
            curr = get_commodity_price_inr(sym)
            if curr:
                pnl = (curr - pos["entry_price"]) * pos["qty_lots"] * COMMODITIES[sym]["lot_size"]
                if pos["direction"] == "SELL":
                    pnl = -pnl
                total += pnl
        return round(total, 2)

    def get_total_value(self) -> float:
        return round(self.state["cash"] + self.get_unrealised_pnl(), 2)

    def has_position(self, symbol: str) -> bool:
        return symbol in self.state["positions"]

    def at_max_positions(self) -> bool:
        return len(self.state["positions"]) >= MAX_COMMODITY_POSITIONS

    def open_position(self, symbol: str, direction: str, price_inr: float,
                      strategy: str, reason: str = "", qty_lots: int = 1) -> Optional[dict]:
        """Open a commodity position. direction: BUY or SELL."""
        if symbol not in COMMODITIES:
            return None
        defn   = COMMODITIES[symbol]
        margin = price_inr * qty_lots * defn["lot_size"] * defn["margin_pct"]

        if margin > self.state["cash"]:
            logger.info(f"[Commodity] SKIP {symbol} — insufficient margin (need ₹{margin:,.0f})")
            return None
        if margin > self.state["initial"] * MAX_RISK_PER_TRADE:
            qty_lots = max(1, int(self.state["initial"] * MAX_RISK_PER_TRADE /
                                  (price_inr * defn["lot_size"] * defn["margin_pct"])))
            margin = price_inr * qty_lots * defn["lot_size"] * defn["margin_pct"]

        sl_pct  = 0.03    # 3% stop loss
        tgt_pct = 0.06    # 6% target (2× risk)
        sl  = round(price_inr * (1 - sl_pct if direction == "BUY" else 1 + sl_pct), 2)
        tgt = round(price_inr * (1 + tgt_pct if direction == "BUY" else 1 - tgt_pct), 2)

        pos = {
            "symbol":      symbol,
            "name":        defn["name"],
            "direction":   direction,
            "qty_lots":    qty_lots,
            "lot_size":    defn["lot_size"],
            "entry_price": round(price_inr, 2),
            "stop_loss":   sl,
            "target":      tgt,
            "margin":      round(margin, 2),
            "strategy":    strategy,
            "reason":      reason,
            "entry_date":  datetime.now(_IST).isoformat(),
            "emoji":       defn["emoji"],
        }
        self.state["positions"][symbol] = pos
        self.state["cash"] -= margin
        self._save()

        trade = {
            "action":      f"OPEN_{direction}",
            "symbol":      symbol,
            "name":        defn["name"],
            "qty_lots":    qty_lots,
            "price_inr":   round(price_inr, 2),
            "margin":      round(margin, 2),
            "stop_loss":   sl,
            "target":      tgt,
            "strategy":    strategy,
            "reason":      reason,
        }
        self._log_trade(trade)
        logger.info(f"[Commodity] OPEN {direction} {symbol} @ ₹{price_inr:,.0f} "
                    f"({qty_lots}L, margin ₹{margin:,.0f}) [{strategy}]")
        return trade

    def close_position(self, symbol: str, reason: str = "") -> Optional[dict]:
        pos = self.state["positions"].get(symbol)
        if not pos:
            return None

        curr = get_commodity_price_inr(symbol)
        if not curr:
            return None

        defn   = COMMODITIES[symbol]
        pnl    = (curr - pos["entry_price"]) * pos["qty_lots"] * defn["lot_size"]
        if pos["direction"] == "SELL":
            pnl = -pnl

        self.state["cash"]         += pos["margin"] + pnl
        self.state["realised_pnl"] += pnl
        del self.state["positions"][symbol]
        self._save()

        trade = {
            "action":      f"CLOSE_{pos['direction']}",
            "symbol":      symbol,
            "name":        defn["name"],
            "qty_lots":    pos["qty_lots"],
            "entry_price": pos["entry_price"],
            "exit_price":  round(curr, 2),
            "pnl":         round(pnl, 2),
            "reason":      reason,
            "strategy":    pos.get("strategy", ""),
        }
        self._log_trade(trade)
        logger.info(f"[Commodity] CLOSE {symbol} @ ₹{curr:,.0f}  P&L=₹{pnl:+,.0f}  ({reason})")

        # Telegram alert
        try:
            from notifier import get_notifier  # noqa: PLC0415
            notif = get_notifier()
            emoji  = "✅" if pnl >= 0 else "❌"
            pnl_pct = pnl / (pos["entry_price"] * pos["qty_lots"] * defn["lot_size"]) * 100
            msg = (
                f"<b>{emoji} Commodity Position Closed</b>\n\n"
                f"{defn['emoji']} <b>{defn['name']}</b>\n\n"
                f"Entry: ₹{pos['entry_price']:,.0f}  →  Exit: ₹{curr:,.0f}\n"
                f"P&L: <b>{'+'if pnl>=0 else ''}₹{pnl:,.0f}</b> ({pnl_pct:+.1f}%)\n"
                f"Lots: {pos['qty_lots']} × {defn['lot_size']} {defn['unit']}\n"
                f"Exit reason: {reason}\n\n"
                f"🕐 {datetime.now(_IST).strftime('%d %b %Y, %H:%M IST')}\n"
                f"<i>Paper trade — not real money</i>"
            )
            notif._tg.send_async(msg)
        except Exception:
            pass

        return trade

    def check_stops(self) -> list:
        """Check stop-loss and target for all open positions."""
        closed = []
        for symbol in list(self.state["positions"].keys()):
            pos  = self.state["positions"].get(symbol)
            if not pos:
                continue
            curr = get_commodity_price_inr(symbol)
            if not curr:
                continue

            hit_sl  = (pos["direction"] == "BUY"  and curr <= pos["stop_loss"]) or \
                      (pos["direction"] == "SELL" and curr >= pos["stop_loss"])
            hit_tgt = (pos["direction"] == "BUY"  and curr >= pos["target"]) or \
                      (pos["direction"] == "SELL" and curr <= pos["target"])

            if hit_sl:
                t = self.close_position(symbol, reason="STOP_LOSS")
                if t:
                    closed.append(t)
            elif hit_tgt:
                t = self.close_position(symbol, reason="TAKE_PROFIT")
                if t:
                    closed.append(t)

        return closed

    def reset(self):
        state = self._default_state()
        self._save(state)
        logger.info("[Commodity] Portfolio reset")


# ── Main orchestrator ─────────────────────────────────────────────────────────

class CommodityAgent:
    def __init__(self):
        self.portfolio = CommodityPortfolio()

    def run_cycle(self, active_events: list = None) -> dict:
        """
        Full commodity cycle:
        1. Check stops on open positions
        2. Generate technical + event signals
        3. Execute high-conviction BUY/SELL signals
        """
        from datetime import time as _dtt  # noqa: PLC0415
        now_t = datetime.now(_IST).time()

        # Only trade during market hours (MCX hours: 9:00 AM – 11:30 PM IST)
        # For safety, align with NSE hours
        if not (_dtt(9, 15) <= now_t <= _dtt(23, 30)):
            return {"status": "market_closed"}

        stops = self.portfolio.check_stops()

        # Generate signals
        tech_signals  = generate_technical_signals()
        event_signals = generate_event_signals(active_events or [])

        # Merge: event signals take priority, dedup by symbol
        all_signals: dict[str, dict] = {}
        for sig in tech_signals:
            all_signals[sig["symbol"]] = sig
        for sig in event_signals:
            # Event signals override technical if stronger
            existing = all_signals.get(sig["symbol"])
            if not existing or sig["strength"] > existing["strength"]:
                all_signals[sig["symbol"]] = sig

        # Save signals file
        COMMODITY_SIGNAL_FILE.write_text(json.dumps({
            "signals":    list(all_signals.values()),
            "updated_at": datetime.now(_IST).isoformat(),
        }, indent=2))

        # Execute signals
        executed = []
        for symbol, sig in all_signals.items():
            if sig["strength"] < 70:
                continue
            if self.portfolio.has_position(symbol):
                # Check if signal flipped — close if opposite direction
                pos = self.portfolio.state["positions"][symbol]
                if pos["direction"] != sig["signal"]:
                    self.portfolio.close_position(symbol, reason="SIGNAL_FLIP")
                continue
            if self.portfolio.at_max_positions():
                break

            trade = self.portfolio.open_position(
                symbol    = symbol,
                direction = sig["signal"],
                price_inr = sig["price_inr"],
                strategy  = sig["source"],
                reason    = sig["reason"],
            )
            if trade:
                executed.append(trade)
                # Telegram alert for event-driven trades
                if sig["source"] == "EVENT":
                    try:
                        from notifier import get_notifier  # noqa: PLC0415
                        defn = COMMODITIES[symbol]
                        msg = (
                            f"<b>{'📈' if sig['signal']=='BUY' else '📉'} Commodity Trade — {sig['signal']}</b>\n\n"
                            f"{defn['emoji']} <b>{defn['name']}</b> @ ₹{sig['price_inr']:,.0f}\n"
                            f"Lots: 1 × {defn['lot_size']} {defn['unit']}\n"
                            f"SL: ₹{trade['stop_loss']:,.0f}  Target: ₹{trade['target']:,.0f}\n\n"
                            f"Reason: {sig['reason']}\n"
                            f"Strength: {sig['strength']}\n\n"
                            f"🕐 {datetime.now(_IST).strftime('%d %b %Y, %H:%M IST')}\n"
                            f"<i>Paper trade — not real money</i>"
                        )
                        get_notifier()._tg.send_async(msg)
                    except Exception:
                        pass

        return {
            "stops_closed": len(stops),
            "signals":      len(all_signals),
            "executed":     len(executed),
            "open_positions": len(self.portfolio.state["positions"]),
            "portfolio_value": self.portfolio.get_total_value(),
        }

    def get_dashboard_data(self) -> dict:
        prices = get_all_prices()
        positions_display = []
        for sym, pos in self.portfolio.state["positions"].items():
            curr = prices.get(sym, pos["entry_price"])
            pnl  = (curr - pos["entry_price"]) * pos["qty_lots"] * pos["lot_size"]
            if pos["direction"] == "SELL":
                pnl = -pnl
            pnl_pct = pnl / (pos["entry_price"] * pos["qty_lots"] * pos["lot_size"]) * 100
            positions_display.append({
                **pos,
                "current_price": round(curr, 2),
                "pnl":           round(pnl, 2),
                "pnl_pct":       round(pnl_pct, 2),
            })

        signals = []
        if COMMODITY_SIGNAL_FILE.exists():
            try:
                signals = json.loads(COMMODITY_SIGNAL_FILE.read_text()).get("signals", [])
            except Exception:
                pass

        trades = []
        if COMMODITY_TRADE_FILE.exists():
            try:
                trades = json.loads(COMMODITY_TRADE_FILE.read_text())[-50:]
            except Exception:
                pass

        return {
            "portfolio_value": self.portfolio.get_total_value(),
            "cash":            round(self.portfolio.state["cash"], 2),
            "initial":         self.portfolio.state["initial"],
            "realised_pnl":    round(self.portfolio.state["realised_pnl"], 2),
            "unrealised_pnl":  self.portfolio.get_unrealised_pnl(),
            "open_positions":  positions_display,
            "live_prices":     {sym: {"price": p, **{k: v for k, v in COMMODITIES[sym].items()
                                                     if k in ("name","emoji","unit","lot_size")}}
                                for sym, p in prices.items()},
            "signals":         signals,
            "trades":          trades,
        }


_commodity_agent: Optional[CommodityAgent] = None


def get_commodity_agent() -> CommodityAgent:
    global _commodity_agent
    if _commodity_agent is None:
        _commodity_agent = CommodityAgent()
    return _commodity_agent
