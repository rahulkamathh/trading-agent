"""
commodity_engine.py — MCX Commodity Paper Trading

Gold, Silver, Crude Oil, Natural Gas, Copper via yfinance futures.
Unified cash pool with equity portfolio.
"""
from __future__ import annotations
import json, logging, time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import yfinance as yf

logger = logging.getLogger("commodity_engine")
_IST = ZoneInfo("Asia/Kolkata")
DATA_DIR = Path(__file__).parent / "data"
COMMODITY_FILE = DATA_DIR / "commodity_portfolio.json"
COMMODITY_TRADES_FILE = DATA_DIR / "commodity_trades.json"
DATA_DIR.mkdir(exist_ok=True)

COMMODITIES = {
    "GOLD":       {"name":"Gold",          "yf":"GC=F",  "emoji":"🥇", "mult":32.1507,  "unit":"$/oz",    "lot_unit":"1 kg",      "margin_pct":0.05},
    "SILVER":     {"name":"Silver",        "yf":"SI=F",  "emoji":"🥈", "mult":964.507,  "unit":"$/oz",    "lot_unit":"30 kg",     "margin_pct":0.05},
    "CRUDEOIL":   {"name":"Crude Oil",     "yf":"CL=F",  "emoji":"🛢️", "mult":100,      "unit":"$/bbl",   "lot_unit":"100 bbl",   "margin_pct":0.08},
    "NATURALGAS": {"name":"Natural Gas",   "yf":"NG=F",  "emoji":"⛽", "mult":1250,     "unit":"$/mmBtu", "lot_unit":"1250 mmBtu","margin_pct":0.10},
    "COPPER":     {"name":"Copper",        "yf":"HG=F",  "emoji":"🔶", "mult":5511.56,  "unit":"$/lb",    "lot_unit":"2500 kg",   "margin_pct":0.05},
}

_USD_INR = 84.0
_USD_INR_TS = 0.0
_PRICE_CACHE: dict = {}

def get_usd_inr() -> float:
    global _USD_INR, _USD_INR_TS
    if time.time() - _USD_INR_TS < 3600:
        return _USD_INR
    try:
        df = yf.download("USDINR=X", period="2d", interval="1h", auto_adjust=True, progress=False)
        if not df.empty:
            r = float(df["Close"].iloc[-1])
            if 70 < r < 120:
                _USD_INR, _USD_INR_TS = r, time.time()
                return r
    except Exception:
        pass
    return _USD_INR

def get_price(symbol: str) -> float | None:
    cached = _PRICE_CACHE.get(symbol)
    if cached and time.time() - cached[1] < 300:
        return cached[0]
    try:
        df = yf.download(COMMODITIES[symbol]["yf"], period="5d", interval="1h", auto_adjust=True, progress=False)
        if df is not None and not df.empty:
            p = float(df["Close"].iloc[-1])
            _PRICE_CACHE[symbol] = (p, time.time())
            return p
    except Exception:
        pass
    return cached[0] if cached else None

def lot_value_inr(symbol: str, price_usd: float) -> float:
    return price_usd * COMMODITIES[symbol]["mult"] * get_usd_inr()

def margin_inr(symbol: str, price_usd: float) -> float:
    return lot_value_inr(symbol, price_usd) * COMMODITIES[symbol]["margin_pct"]

def get_all_prices() -> dict:
    usd_inr = get_usd_inr()
    result = {}
    for sym, meta in COMMODITIES.items():
        p = get_price(sym)
        result[sym] = {
            "symbol": sym, "name": meta["name"], "emoji": meta["emoji"],
            "price_usd": round(p, 4) if p else None,
            "price_inr": round(p * usd_inr, 2) if p else None,
            "lot_value": round(lot_value_inr(sym, p), 0) if p else None,
            "margin":    round(margin_inr(sym, p), 0) if p else None,
            "unit": meta["unit"], "lot_unit": meta["lot_unit"], "usd_inr": round(usd_inr, 2),
        }
    return result

def generate_signals() -> list:
    signals = []
    usd_inr = get_usd_inr()
    for sym, meta in COMMODITIES.items():
        try:
            df = yf.download(meta["yf"], period="6mo", interval="1d", auto_adjust=True, progress=False)
            if df is None or df.empty or len(df) < 55:
                continue
            close = df["Close"].squeeze()
            ema20 = close.ewm(span=20).mean()
            ema50 = close.ewm(span=50).mean()
            delta = close.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rsi   = 100 - 100 / (1 + gain / loss.replace(0, float("nan")))
            curr  = float(close.iloc[-1])
            e20   = float(ema20.iloc[-1]); e50 = float(ema50.iloc[-1])
            pe20  = float(ema20.iloc[-2]); pe50= float(ema50.iloc[-2])
            r     = float(rsi.iloc[-1])
            golden= pe20 < pe50 and e20 > e50
            death = pe20 > pe50 and e20 < e50
            if golden or (e20 > e50 and r < 65):
                direction, strength = "BUY",  80 if golden else 65
                reason = "Golden cross" if golden else f"EMA bullish, RSI={r:.0f}"
            elif death or (e20 < e50 and r > 55):
                direction, strength = "SELL", 80 if death else 60
                reason = "Death cross" if death else f"EMA bearish, RSI={r:.0f}"
            else:
                direction, strength, reason = "NEUTRAL", 50, "No clear trend"
            signals.append({
                "symbol": sym, "name": meta["name"], "emoji": meta["emoji"],
                "direction": direction, "strength": strength,
                "price_usd": round(curr, 4), "price_inr": round(curr * usd_inr, 2),
                "rsi": round(r, 1), "ema20": round(e20, 4), "ema50": round(e50, 4),
                "reason": reason,
                "lot_value": round(lot_value_inr(sym, curr), 0),
                "margin":    round(margin_inr(sym, curr), 0),
                "generated_at": datetime.now(_IST).isoformat(),
            })
        except Exception as e:
            logger.warning(f"[Commodity] Signal failed {sym}: {e}")
    return signals

class CommodityPortfolio:
    def __init__(self, equity_portfolio=None):
        self._equity = equity_portfolio
        self.state   = self._load()

    def _default_state(self):
        return {"positions": {}, "realised_pnl": 0.0,
                "created_at": datetime.now(_IST).isoformat(),
                "last_updated": datetime.now(_IST).isoformat()}

    def _load(self):
        if COMMODITY_FILE.exists():
            try: return json.loads(COMMODITY_FILE.read_text())
            except Exception: pass
        s = self._default_state(); self._save(s); return s

    def _save(self, s=None):
        if s: self.state = s
        self.state["last_updated"] = datetime.now(_IST).isoformat()
        COMMODITY_FILE.write_text(json.dumps(self.state, indent=2))

    def _log(self, trade):
        log = []
        if COMMODITY_TRADES_FILE.exists():
            try: log = json.loads(COMMODITY_TRADES_FILE.read_text())
            except Exception: pass
        trade["id"] = len(log) + 1; trade["time"] = datetime.now(_IST).isoformat()
        log.append(trade)
        COMMODITY_TRADES_FILE.write_text(json.dumps(log[-500:], indent=2))

    def _cash(self): return self._equity.state["cash"] if self._equity else 0.0
    def _deduct(self, a):
        if self._equity: self._equity.state["cash"] -= a; self._equity._save()
    def _return(self, a):
        if self._equity: self._equity.state["cash"] += a; self._equity._save()

    def has_position(self, sym): return sym in self.state["positions"]

    def open_position(self, symbol, direction, qty_lots, price_usd, reason="", strategy=""):
        from datetime import time as _dtt
        now_t = datetime.now(_IST).time()
        if not (_dtt(9, 0) <= now_t <= _dtt(23, 30)):
            logger.info(f"[Commodity] MCX closed — blocked {symbol}"); return None
        if self.has_position(symbol):
            logger.info(f"[Commodity] Already in {symbol}"); return None
        mgn = margin_inr(symbol, price_usd) * qty_lots
        if mgn > self._cash():
            logger.warning(f"[Commodity] Insufficient margin ₹{mgn:.0f} for {symbol}"); return None
        usd_inr = get_usd_inr()
        pos = {"symbol": symbol, "name": COMMODITIES[symbol]["name"],
               "direction": direction, "qty_lots": qty_lots,
               "entry_usd": round(price_usd, 4), "entry_inr": round(price_usd * usd_inr, 2),
               "entry_usd_inr": round(usd_inr, 2), "margin_blocked": round(mgn, 2),
               "strategy": strategy, "reason": reason,
               "entry_date": datetime.now(_IST).isoformat()}
        self.state["positions"][symbol] = pos
        self._deduct(mgn); self._save()
        trade = {"action": f"OPEN_{direction}", "symbol": symbol,
                 "name": COMMODITIES[symbol]["name"], "qty_lots": qty_lots,
                 "price_usd": round(price_usd, 4), "price_inr": round(price_usd * usd_inr, 2),
                 "margin": round(mgn, 2), "strategy": strategy, "reason": reason}
        self._log(trade)
        logger.info(f"[Commodity] OPEN {direction} {symbol} @ ${price_usd:.2f} margin=₹{mgn:.0f}")
        try:
            from notifier import get_notifier
            c = COMMODITIES[symbol]
            msg = (f"<b>{'📈' if direction=='LONG' else '📉'} Commodity Trade</b>\n\n"
                   f"{c['emoji']} <b>{c['name']}</b> — {direction}\n\n"
                   f"Price: ${price_usd:.2f} / ₹{price_usd*usd_inr:,.0f}\n"
                   f"Margin: ₹{mgn:,.0f} ({c['lot_unit']})\n"
                   f"Reason: {reason}\n\n"
                   f"🕐 {datetime.now(_IST).strftime('%d %b %Y, %H:%M IST')}")
            get_notifier()._tg.send_async(msg)
        except Exception: pass
        return trade

    def close_position(self, symbol, reason=""):
        pos = self.state["positions"].get(symbol)
        if not pos: return None
        curr = get_price(symbol)
        if not curr: logger.warning(f"[Commodity] No price for {symbol}"); return None
        usd_inr = get_usd_inr()
        c = COMMODITIES[symbol]
        entry_inr = pos["entry_usd"] * pos["entry_usd_inr"]
        curr_inr  = curr * usd_inr
        pnl_lot   = (curr_inr - entry_inr) * c["mult"]
        if pos["direction"] == "SHORT": pnl_lot = -pnl_lot
        pnl = round(pnl_lot * pos["qty_lots"], 2)
        self._return(pos["margin_blocked"] + pnl)
        self.state["realised_pnl"] += pnl
        del self.state["positions"][symbol]; self._save()
        trade = {"action": f"CLOSE_{pos['direction']}", "symbol": symbol,
                 "name": c["name"], "qty_lots": pos["qty_lots"],
                 "entry_usd": pos["entry_usd"], "exit_usd": round(curr, 4),
                 "pnl": pnl, "reason": reason}
        self._log(trade)
        logger.info(f"[Commodity] CLOSE {symbol} P&L=₹{pnl:+,.0f} ({reason})")
        try:
            from notifier import get_notifier
            emoji = "✅" if pnl >= 0 else "❌"
            msg = (f"<b>{emoji} Commodity Closed</b>\n\n"
                   f"{c['emoji']} <b>{c['name']}</b>\n\n"
                   f"Entry: ${pos['entry_usd']:.2f}  →  Exit: ${curr:.2f}\n"
                   f"P&L: <b>{'+'if pnl>=0 else ''}₹{pnl:,.0f}</b>\n"
                   f"Exit reason: {reason}\n\n"
                   f"🕐 {datetime.now(_IST).strftime('%d %b %Y, %H:%M IST')}")
            get_notifier()._tg.send_async(msg)
        except Exception: pass
        return trade

    def get_positions_display(self):
        result = []
        usd_inr = get_usd_inr()
        for sym, pos in self.state["positions"].items():
            curr = get_price(sym) or pos["entry_usd"]
            c = COMMODITIES.get(sym, {})
            entry_inr = pos["entry_usd"] * pos["entry_usd_inr"]
            curr_inr  = curr * usd_inr
            pnl_lot   = (curr_inr - entry_inr) * c.get("mult", 1)
            if pos["direction"] == "SHORT": pnl_lot = -pnl_lot
            pnl = round(pnl_lot * pos["qty_lots"], 2)
            pnl_pct = pnl / (pos["margin_blocked"]) * 100 if pos["margin_blocked"] else 0
            result.append({"symbol": sym, "name": pos["name"], "emoji": c.get("emoji","📦"),
                           "direction": pos["direction"], "qty_lots": pos["qty_lots"],
                           "entry_usd": pos["entry_usd"], "curr_usd": round(curr, 4),
                           "entry_inr": round(entry_inr, 2), "curr_inr": round(curr_inr, 2),
                           "pnl": pnl, "pnl_pct": round(pnl_pct, 2),
                           "margin": pos["margin_blocked"], "strategy": pos.get("strategy",""),
                           "entry_date": pos.get("entry_date","")})
        return result

    def reset(self):
        # Return all margin to equity before resetting
        for sym, pos in self.state["positions"].items():
            self._return(pos.get("margin_blocked", 0))
        s = self._default_state()
        self._save(s)
        logger.info("[Commodity] Portfolio reset")

    def check_stops(self):
        from datetime import time as _dtt
        if not (_dtt(9, 0) <= datetime.now(_IST).time() <= _dtt(23, 0)): return []
        closed = []
        for sym in list(self.state["positions"].keys()):
            pos = self.state["positions"].get(sym)
            if not pos: continue
            curr = get_price(sym)
            if not curr: continue
            c = COMMODITIES.get(sym, {})
            entry_inr = pos["entry_usd"] * pos["entry_usd_inr"]
            pnl_lot   = (curr * get_usd_inr() - entry_inr) * c.get("mult", 1)
            if pos["direction"] == "SHORT": pnl_lot = -pnl_lot
            mgn = pos["margin_blocked"] / pos["qty_lots"]
            ratio = pnl_lot / mgn if mgn else 0
            if ratio <= -0.50:
                t = self.close_position(sym, reason="STOP_LOSS"); 
                if t: closed.append(t)
            elif ratio >= 1.50:
                t = self.close_position(sym, reason="TAKE_PROFIT")
                if t: closed.append(t)
        return closed


def _hedge_signals(equity_portfolio) -> list:
    """
    Generate commodity hedge signals by analysing the equity portfolio.

    Hedge rules (institutional logic):
    1. GOLD hedge  — buy Gold when equity drawdown > 4% OR India VIX > 20
    2. CRUDE hedge — buy Crude when >20% of equity is energy stocks AND crude trending up
    3. SILVER hedge— buy Silver when Gold already held as hedge (diversify safe haven)
    4. NATGAS hedge— sell NatGas (go short) when energy positions are oversized and gas trending down
    5. Cash-raise  — if hedge needed but cash < margin, liquidate the weakest equity position first
    """
    if not equity_portfolio:
        return []

    signals = []
    positions = equity_portfolio.state.get("positions", {})
    total_val = equity_portfolio.get_total_value()
    initial   = equity_portfolio.state.get("initial", 1_300_000)
    drawdown  = (total_val - initial) / initial   # negative = loss

    # ── Rule 1: Gold hedge on drawdown or high VIX ──────────────────────── #
    try:
        vix_df = yf.download("^INDIAVIX", period="3d", interval="1d", progress=False, auto_adjust=True)
        vix = float(vix_df["Close"].iloc[-1]) if not vix_df.empty else 0
    except Exception:
        vix = 0

    if drawdown < -0.04 or vix > 20:
        reason = (f"Equity drawdown {drawdown*100:.1f}%" if drawdown < -0.04
                  else f"India VIX elevated at {vix:.1f}")
        signals.append({"symbol": "GOLD", "direction": "BUY", "strength": 90,
                         "reason": f"HEDGE: {reason}", "strategy": "HEDGE"})

    # ── Rule 2: Crude hedge when energy-heavy ───────────────────────────── #
    energy_tickers = {"ONGC.NS", "BPCL.NS", "IOC.NS", "HINDPETRO.NS",
                      "RELIANCE.NS", "OIL.NS", "MRPL.NS", "GAIL.NS"}
    energy_val = sum(
        equity_portfolio.get_position_value(t)
        for t in positions if t in energy_tickers
    )
    energy_pct = energy_val / total_val if total_val else 0

    if energy_pct > 0.20:
        # Only buy crude if it's in uptrend
        crude_p = get_price("CRUDEOIL")
        try:
            crude_df = yf.download("CL=F", period="30d", interval="1d", progress=False, auto_adjust=True)
            crude_trend = "UP" if float(crude_df["Close"].iloc[-1]) > float(crude_df["Close"].ewm(span=20).mean().iloc[-1]) else "DOWN"
        except Exception:
            crude_trend = "UNKNOWN"
        if crude_trend == "UP":
            signals.append({"symbol": "CRUDEOIL", "direction": "BUY", "strength": 80,
                             "reason": f"HEDGE: Energy exposure {energy_pct*100:.0f}% of portfolio, crude trending up",
                             "strategy": "HEDGE"})

    # ── Rule 3: Silver as secondary safe haven when Gold hedge is active ── #
    # (detected by checking if GOLD hedge was just added)
    if any(s["symbol"] == "GOLD" for s in signals):
        signals.append({"symbol": "SILVER", "direction": "BUY", "strength": 75,
                         "reason": "HEDGE: Silver diversification alongside Gold safe-haven",
                         "strategy": "HEDGE"})

    return signals


def _liquidate_weakest_equity(equity_portfolio, needed_cash: float) -> bool:
    """
    If free cash is insufficient for a hedge, sell the weakest-performing
    equity position to raise the required capital.
    Returns True if a position was liquidated.
    """
    positions = equity_portfolio.state.get("positions", {})
    if not positions:
        return False

    # Rank positions by unrealised P&L % (sell the biggest loser first)
    ranked = []
    for ticker, pos in positions.items():
        try:
            from engine import DataFetcher  # pylint: disable=import-outside-toplevel
            price = DataFetcher.get_current_price(ticker)
            if not price: continue
            pnl_pct = (price - pos["avg_price"]) / pos["avg_price"]
            ranked.append((pnl_pct, ticker, price, pos))
        except Exception:
            continue

    if not ranked:
        return False

    ranked.sort(key=lambda x: x[0])  # worst first
    pnl_pct, ticker, price, pos = ranked[0]

    # Only liquidate if it's a loser or has been held < 3% gain (dead money)
    if pnl_pct > 0.05:
        logger.info(f"[Commodity Hedge] Weakest equity {ticker} is +{pnl_pct*100:.1f}% — not liquidating")
        return False

    proceeds = price * pos["qty"]
    logger.info(f"[Commodity Hedge] Liquidating {ticker} ({pnl_pct*100:+.1f}%) to raise ₹{proceeds:,.0f} for hedge")
    equity_portfolio.execute_sell(ticker, price, reason="HEDGE_LIQUIDATION")
    return True


class CommodityAgent:
    def __init__(self, equity_portfolio=None):
        self.portfolio = CommodityPortfolio(equity_portfolio=equity_portfolio)

    def run_cycle(self, active_events=None):
        # ── 1. Event-driven signals ──────────────────────────────────────── #
        if active_events:
            for ev in (active_events if isinstance(active_events, list) else [active_events]):
                for action in ev.get("actions", []):
                    sym = action.get("commodity")
                    direction = action.get("direction")
                    if sym and direction and sym in COMMODITIES:
                        self.execute_event_signal(sym, direction, ev.get("headline", "Event-driven"))

        # ── 2. Stop losses / take profits on existing positions ─────────── #
        stops = self.portfolio.check_stops()

        # ── 3. Hedge signals from equity portfolio analysis ──────────────── #
        hedge_executed = 0
        eq = self.portfolio._equity
        if eq:
            hedge_sigs = _hedge_signals(eq)
            for sig in hedge_sigs:
                sym = sig["symbol"]
                if self.portfolio.has_position(sym):
                    continue
                p = get_price(sym)
                if not p:
                    continue
                mgn = margin_inr(sym, p)
                # If not enough cash, try to liquidate a weak equity position
                if self.portfolio._cash() < mgn:
                    logger.info(f"[Commodity Hedge] Insufficient cash (₹{self.portfolio._cash():,.0f}) for {sym} margin ₹{mgn:,.0f} — attempting equity liquidation")
                    freed = _liquidate_weakest_equity(eq, mgn)
                    if not freed:
                        logger.info(f"[Commodity Hedge] Could not free cash for {sym} — skipping")
                        continue
                t = self.portfolio.open_position(
                    symbol=sym, direction="LONG" if sig["direction"] == "BUY" else "SHORT",
                    qty_lots=1, price_usd=p, reason=sig["reason"], strategy="HEDGE")
                if t:
                    hedge_executed += 1
                    logger.info(f"[Commodity Hedge] Opened {sym} hedge — {sig['reason']}")

        # ── 4. Momentum signals ──────────────────────────────────────────── #
        signals  = generate_signals()
        executed = 0
        for sig in signals:
            if sig["direction"] == "NEUTRAL" or sig["strength"] < 70:
                continue
            if self.portfolio.has_position(sig["symbol"]):
                continue
            t = self.portfolio.open_position(
                symbol=sig["symbol"],
                direction="LONG" if sig["direction"] == "BUY" else "SHORT",
                qty_lots=1, price_usd=sig["price_usd"],
                reason=sig["reason"], strategy="COMMODITY_MOMENTUM")
            if t:
                executed += 1

        return {"stops_closed": len(stops), "signals": signals,
                "executed": executed, "hedge_executed": hedge_executed}

    def execute_event_signal(self, symbol, direction, reason):
        p = get_price(symbol)
        if not p: return None
        return self.portfolio.open_position(symbol=symbol, direction=direction,
                                            qty_lots=1, price_usd=p,
                                            reason=reason, strategy="EVENT_DRIVEN")

    def get_dashboard_data(self):
        signals  = generate_signals()
        trades   = []
        if COMMODITY_TRADES_FILE.exists():
            try: trades = json.loads(COMMODITY_TRADES_FILE.read_text())[-50:]
            except Exception: pass
        positions   = self.portfolio.get_positions_display()
        unrealised  = sum(p["pnl"] for p in positions)
        return {
            "positions":      positions,
            "signals":        signals,
            "trades":         trades,
            "realised_pnl":   round(self.portfolio.state["realised_pnl"], 2),
            "unrealised_pnl": round(unrealised, 2),
            "total_pnl":      round(self.portfolio.state["realised_pnl"] + unrealised, 2),
            "open_count":     len(positions),
            "prices":         get_all_prices(),
            "usd_inr":        round(get_usd_inr(), 2),
        }


_commodity_agent: CommodityAgent | None = None

def get_commodity_agent(equity_portfolio=None) -> CommodityAgent:
    global _commodity_agent
    if _commodity_agent is None:
        if equity_portfolio is None:
            try:
                from engine import get_agent as _ge
                equity_portfolio = _ge().portfolio
            except Exception: pass
        _commodity_agent = CommodityAgent(equity_portfolio=equity_portfolio)
    elif equity_portfolio is not None and _commodity_agent.portfolio._equity is None:
        _commodity_agent.portfolio._equity = equity_portfolio
    return _commodity_agent
