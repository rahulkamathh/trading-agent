"""
F&O Engine — Institutional-Grade Derivatives Trading
======================================================
Supports:
  • Index Options  — Nifty / BankNifty weekly & monthly CE/PE
  • Index Futures  — Nifty / BankNifty continuous front-month
  • Stock Futures  — NSE F&O stocks, monthly expiry
  • Stock Options  — NSE F&O stocks, monthly CE/PE

Pricing:
  • Live option chain via yfinance (when available)
  • Black-Scholes fallback with HV(30) as IV proxy

Strategies:
  1. DirectionalOptions  — equity signal → buy ATM/OTM call or put
  2. SpreadStrategy      — bull call spread / bear put spread (defined risk)
  3. IronCondorStrategy  — weekly premium selling on index options
  4. HedgeStrategy       — protective puts on open equity positions

Integration:
  • Equity BUY signals with strength ≥ 80 → also route to call options
  • Portfolio-level Greeks displayed on dashboard
  • Separate F&O capital book (default ₹2,00,000)

Paper trading only — no real orders.
"""

import json
import logging
import math
import time
from collections import defaultdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DATA_DIR         = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
FNO_PORTFOLIO_FILE = DATA_DIR / "fno_portfolio.json"
FNO_TRADE_FILE     = DATA_DIR / "fno_trades.json"

FNO_INITIAL_CAPITAL = 2_00_000   # ₹2 lakhs allocated to F&O desk
RISK_FREE_RATE      = 0.0675     # RBI repo rate approximation
MAX_RISK_PER_TRADE  = 0.05       # max 5% of F&O capital at risk per trade


# ---------------------------------------------------------------------------
# Lot sizes (NSE standard — verify at nseindia.com/market-data/lot-size)
# ---------------------------------------------------------------------------

LOT_SIZES: dict[str, int] = {
    # Indices
    "^NSEI":       75,    # Nifty 50
    "^NSEBANK":    30,    # Bank Nifty
    "NIFTY":       75,
    "BANKNIFTY":   30,
    "FINNIFTY":    40,
    "MIDCPNIFTY":  75,
    # Large caps (common F&O stocks)
    "RELIANCE.NS": 500,   "TCS.NS":      150,   "HDFCBANK.NS": 550,
    "ICICIBANK.NS":700,   "INFY.NS":     400,   "SBIN.NS":    1500,
    "KOTAKBANK.NS":400,   "AXISBANK.NS": 1200,  "WIPRO.NS":   1500,
    "LT.NS":       300,   "BAJFINANCE.NS":125,  "MARUTI.NS":  100,
    "TATAMOTORS.NS":2850, "SUNPHARMA.NS":700,   "HCLTECH.NS": 700,
    "TITAN.NS":    375,   "BHARTIARTL.NS":1851, "NTPC.NS":    3000,
    "POWERGRID.NS":2700,  "ONGC.NS":    3850,   "BPCL.NS":    1800,
    "IOC.NS":      3500,  "COALINDIA.NS":4200,  "HINDALCO.NS":2150,
    "TATASTEEL.NS":5500,  "JSWSTEEL.NS": 675,   "VEDL.NS":    2750,
    "ADANIPORTS.NS":1250, "ULTRACEMCO.NS":100,  "GRASIM.NS":  475,
    "INDUSINDBK.NS":900,  "ITC.NS":      3200,  "HINDUNILVR.NS":300,
    "DRREDDY.NS":  125,   "CIPLA.NS":    650,   "DIVISLAB.NS":  100,
    "EICHERMOT.NS":150,   "HEROMOTOCO.NS":300,  "BAJAJ-AUTO.NS":250,
    "M&M.NS":      900,   "TECHM.NS":    600,   "LTIM.NS":    150,
    "APOLLOHOSP.NS":125,
}

def get_lot_size(ticker: str) -> int:
    """Return lot size for a given ticker, defaulting to 1."""
    return LOT_SIZES.get(ticker, LOT_SIZES.get(ticker.replace(".NS", ""), 1))


# ---------------------------------------------------------------------------
# Expiry calendar helpers
# ---------------------------------------------------------------------------

def _last_thursday(year: int, month: int) -> date:
    """Last Thursday of the given month."""
    d = date(year, month % 12 + 1, 1) - timedelta(days=1)   # last day of month
    while d.weekday() != 3:   # 3 = Thursday
        d -= timedelta(days=1)
    return d

def _next_thursday(from_date: date = None) -> date:
    """Next Thursday on or after from_date (for weekly Nifty expiry)."""
    d = from_date or date.today()
    days_ahead = (3 - d.weekday()) % 7  # 3 = Thursday
    return d + timedelta(days=days_ahead or 7)

def _next_wednesday(from_date: date = None) -> date:
    """Next Wednesday on or after from_date (for weekly BankNifty expiry)."""
    d = from_date or date.today()
    days_ahead = (2 - d.weekday()) % 7
    return d + timedelta(days=days_ahead or 7)

def get_expiry(ticker: str, monthly: bool = False) -> date:
    """
    Get the nearest expiry date for a given ticker.
    - Index options: weekly (Nifty=Thu, BankNifty=Wed)
    - Stock options/futures: monthly (last Thursday)
    """
    today = date.today()
    if not monthly and ticker in ("^NSEI", "NIFTY"):
        exp = _next_thursday(today)
        # If expiry is today and market likely closed, use next week's
        if exp == today:
            exp = _next_thursday(today + timedelta(days=1))
        return exp
    if not monthly and ticker in ("^NSEBANK", "BANKNIFTY"):
        exp = _next_wednesday(today)
        if exp == today:
            exp = _next_wednesday(today + timedelta(days=1))
        return exp
    # Monthly: last Thursday of current month, or next month if within 3 days
    exp = _last_thursday(today.year, today.month)
    if (exp - today).days < 3:
        # Roll to next month
        next_month = today.month % 12 + 1
        next_year  = today.year + (1 if today.month == 12 else 0)
        exp = _last_thursday(next_year, next_month)
    return exp

def days_to_expiry(expiry: date) -> float:
    """Calendar days to expiry as a fraction of a year."""
    delta = (expiry - date.today()).days
    return max(delta, 0) / 365.0


# ---------------------------------------------------------------------------
# Black-Scholes pricing (no scipy dependency)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF using Abramowitz & Stegun approximation (error < 1.5e-7)."""
    if x < 0:
        return 1.0 - _norm_cdf(-x)
    a1, a2, a3, a4, a5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    t = 1.0 / (1.0 + 0.2316419 * x)
    poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
    return 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x) * poly

def _norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)


class BlackScholes:
    """
    European option pricing using Black-Scholes.
    NSE index options are European; stock options are technically American
    but BS is a good approximation for non-dividend stocks.
    """

    @staticmethod
    def _d1d2(S: float, K: float, T: float, r: float, sigma: float):
        if T <= 0 or sigma <= 0:
            return 0.0, 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2

    @classmethod
    def price(cls, S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = "call") -> float:
        """
        S: spot price, K: strike, T: time to expiry (years),
        r: risk-free rate, sigma: implied/historical vol, option_type: call|put
        """
        if T <= 0:
            return max(S - K, 0) if option_type == "call" else max(K - S, 0)
        d1, d2 = cls._d1d2(S, K, T, r, sigma)
        disc = math.exp(-r * T)
        if option_type == "call":
            return S * _norm_cdf(d1) - K * disc * _norm_cdf(d2)
        else:
            return K * disc * _norm_cdf(-d2) - S * _norm_cdf(-d1)

    @classmethod
    def greeks(cls, S: float, K: float, T: float, r: float, sigma: float,
               option_type: str = "call") -> dict:
        """
        Returns delta, gamma, theta (per day), vega (per 1% IV move).
        """
        if T <= 0 or sigma <= 0:
            iv = 1.0 if (option_type == "call" and S > K) else 0.0
            return {"delta": iv, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

        d1, d2 = cls._d1d2(S, K, T, r, sigma)
        disc   = math.exp(-r * T)
        nd1    = _norm_pdf(d1)
        sqrt_T = math.sqrt(T)

        delta  = _norm_cdf(d1) if option_type == "call" else _norm_cdf(d1) - 1
        gamma  = nd1 / (S * sigma * sqrt_T)
        # Theta in ₹ per day
        theta_base = -(S * nd1 * sigma) / (2 * sqrt_T)
        if option_type == "call":
            theta = (theta_base - r * K * disc * _norm_cdf(d2)) / 365
        else:
            theta = (theta_base + r * K * disc * _norm_cdf(-d2)) / 365
        vega   = S * nd1 * sqrt_T / 100   # per 1% vol change

        return {
            "delta": round(delta, 4),
            "gamma": round(gamma, 6),
            "theta": round(theta, 4),
            "vega":  round(vega, 4),
        }

    @classmethod
    def implied_vol(cls, market_price: float, S: float, K: float, T: float,
                    r: float, option_type: str = "call",
                    tol: float = 1e-5, max_iter: int = 100) -> float:
        """
        Compute implied volatility via Newton-Raphson.
        Returns HV fallback (0.25) if it doesn't converge.
        """
        if T <= 0 or market_price <= 0:
            return 0.25
        sigma = 0.25   # initial guess
        for _ in range(max_iter):
            price = cls.price(S, K, T, r, sigma, option_type)
            vega  = cls.greeks(S, K, T, r, sigma, option_type)["vega"] * 100
            if abs(vega) < 1e-10:
                break
            diff  = price - market_price
            if abs(diff) < tol:
                break
            sigma -= diff / vega
            sigma  = max(0.01, min(sigma, 5.0))   # clamp to [1%, 500%]
        return round(sigma, 4)


def historical_vol(ticker: str, window: int = 30) -> float:
    """
    Compute annualised historical volatility from the last `window` daily closes.
    Used as IV proxy when live option chain is unavailable.
    """
    try:
        df = yf.download(ticker, period=f"{window+5}d", interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 5:
            return 0.25
        log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        hv = float(log_ret.std() * math.sqrt(252))
        return round(max(0.05, min(hv, 2.0)), 4)
    except Exception:
        return 0.25


# ---------------------------------------------------------------------------
# Strike selection helpers
# ---------------------------------------------------------------------------

def select_strike(spot: float, option_type: str, moneyness: str = "ATM",
                  step: float = None) -> float:
    """
    Select a strike given spot price.
    moneyness: ATM | OTM1 | OTM2 | ITM1 | ITM2
    step: rounding interval (auto-detected if None)
    """
    if step is None:
        if spot < 200:      step = 5
        elif spot < 500:    step = 10
        elif spot < 1000:   step = 20
        elif spot < 5000:   step = 50
        elif spot < 10000:  step = 100
        elif spot < 25000:  step = 200
        else:               step = 500

    atm = round(spot / step) * step
    offsets = {
        "ATM":  0,
        "OTM1": 1 if option_type == "call" else -1,
        "OTM2": 2 if option_type == "call" else -2,
        "ITM1": -1 if option_type == "call" else 1,
        "ITM2": -2 if option_type == "call" else 2,
    }
    return atm + offsets.get(moneyness, 0) * step


# ---------------------------------------------------------------------------
# F&O Portfolio
# ---------------------------------------------------------------------------

class FNOPortfolio:
    """
    Paper-trading F&O portfolio.
    Tracks options and futures positions separately from equity.
    Computes portfolio-level Greeks in real time.
    """

    def __init__(self):
        self.state = self._load()

    def _default_state(self) -> dict:
        return {
            "cash":          FNO_INITIAL_CAPITAL,
            "initial":       FNO_INITIAL_CAPITAL,
            "positions":     {},    # key: position_id → position dict
            "realised_pnl":  0.0,
            "created_at":    datetime.now().isoformat(),
            "last_updated":  datetime.now().isoformat(),
        }

    def _load(self) -> dict:
        if FNO_PORTFOLIO_FILE.exists():
            try:
                with open(FNO_PORTFOLIO_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        state = self._default_state()
        self._save(state)
        return state

    def _save(self, state: dict = None):
        if state:
            self.state = state
        self.state["last_updated"] = datetime.now().isoformat()
        with open(FNO_PORTFOLIO_FILE, "w") as f:
            json.dump(self.state, f, indent=2)

    def _log_trade(self, trade: dict):
        log = []
        if FNO_TRADE_FILE.exists():
            try:
                with open(FNO_TRADE_FILE) as f:
                    log = json.load(f)
            except Exception:
                pass
        trade["id"] = len(log) + 1
        trade["time"] = datetime.now().isoformat()
        log.append(trade)
        with open(FNO_TRADE_FILE, "w") as f:
            json.dump(log, f, indent=2)
        return trade

    def get_total_value(self) -> float:
        """Cash + mark-to-market value of all open positions."""
        total = self.state["cash"]
        for pid, pos in self.state["positions"].items():
            total += self._mtm_value(pos)
        return round(total, 2)

    def _mtm_value(self, pos: dict) -> float:
        """Current mark-to-market value of one position."""
        try:
            spot   = self._get_spot(pos["underlying"])
            if pos["instrument_type"] == "FUTURE":
                entry  = pos["entry_price"]
                qty    = pos["qty"] * get_lot_size(pos["underlying"])
                direction = 1 if pos["position"] == "LONG" else -1
                return round(direction * (spot - entry) * qty, 2)
            else:
                # Option
                T      = days_to_expiry(date.fromisoformat(pos["expiry"]))
                iv     = pos.get("iv", 0.25)
                curr_premium = BlackScholes.price(
                    spot, pos["strike"], T,
                    RISK_FREE_RATE, iv, pos["option_type"]
                )
                qty = pos["qty"] * get_lot_size(pos["underlying"])
                direction = 1 if pos["position"] == "LONG" else -1
                return round(direction * curr_premium * qty, 2)
        except Exception:
            return 0.0

    def _get_spot(self, ticker: str) -> float:
        try:
            data = yf.download(ticker, period="2d", interval="1d",
                               auto_adjust=True, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if not data.empty:
                return float(data["Close"].iloc[-1])
        except Exception:
            pass
        return 0.0

    def portfolio_greeks(self) -> dict:
        """Compute net portfolio-level Greeks across all option positions."""
        net = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
        for pos in self.state["positions"].values():
            if pos["instrument_type"] != "OPTION":
                # Futures contribute delta only
                lot = get_lot_size(pos["underlying"])
                d   = pos["qty"] * lot * (1 if pos["position"] == "LONG" else -1)
                net["delta"] += d
                continue
            try:
                spot = self._get_spot(pos["underlying"])
                T    = days_to_expiry(date.fromisoformat(pos["expiry"]))
                iv   = pos.get("iv", 0.25)
                g    = BlackScholes.greeks(
                    spot, pos["strike"], T,
                    RISK_FREE_RATE, iv, pos["option_type"]
                )
                lot  = get_lot_size(pos["underlying"])
                qty  = pos["qty"] * lot
                sign = 1 if pos["position"] == "LONG" else -1
                for k in net:
                    net[k] += sign * qty * g[k]
            except Exception:
                pass
        return {k: round(v, 4) for k, v in net.items()}

    # ── Entry / Exit ────────────────────────────────────────────────────── #

    def open_option(self, underlying: str, strike: float, expiry: date,
                    option_type: str, position: str, qty_lots: int,
                    strategy: str, reason: str = "") -> dict | None:
        """
        Open an option position.
        position: LONG (buy) | SHORT (sell/write)
        """
        spot = self._get_spot(underlying)
        if spot <= 0:
            logger.warning(f"[FNO] Cannot fetch spot for {underlying}")
            return None

        T  = days_to_expiry(expiry)
        iv = historical_vol(underlying)
        premium = BlackScholes.price(spot, strike, T, RISK_FREE_RATE, iv, option_type)
        lot     = get_lot_size(underlying)
        total_qty = qty_lots * lot
        cost    = premium * total_qty   # premium paid (LONG) or received (SHORT)

        if position == "LONG" and cost > self.state["cash"]:
            logger.warning(f"[FNO] Insufficient cash for {underlying} {option_type} — need ₹{cost:.0f}")
            return None

        pid = f"{underlying}_{strike}_{expiry}_{option_type}_{position}_{int(time.time())}"

        pos = {
            "position_id":    pid,
            "instrument_type":"OPTION",
            "underlying":     underlying,
            "strike":         strike,
            "expiry":         expiry.isoformat(),
            "option_type":    option_type,   # call | put
            "position":       position,       # LONG | SHORT
            "qty":            qty_lots,
            "entry_premium":  round(premium, 2),
            "entry_spot":     round(spot, 2),
            "iv":             iv,
            "strategy":       strategy,
            "reason":         reason,
            "entry_date":     datetime.now().isoformat(),
        }
        self.state["positions"][pid] = pos

        if position == "LONG":
            self.state["cash"] -= cost
        else:
            # Short option: collect premium, reserve margin (approx 20% of notional)
            margin = spot * lot * qty_lots * 0.20
            self.state["cash"] -= margin
            pos["margin_blocked"] = round(margin, 2)
            self.state["cash"] += cost   # premium received

        self._save()
        trade = {
            "action":       f"OPEN_{position}_{option_type.upper()}",
            "underlying":   underlying,
            "strike":       strike,
            "expiry":       expiry.isoformat(),
            "option_type":  option_type,
            "qty_lots":     qty_lots,
            "premium":      round(premium, 2),
            "total_value":  round(cost, 2),
            "strategy":     strategy,
            "reason":       reason,
            "iv":           iv,
            "spot_at_entry":round(spot, 2),
        }
        self._log_trade(trade)
        logger.info(
            f"[FNO] OPEN {position} {qty_lots}L {underlying} {strike}{option_type[0].upper()} "
            f"@ ₹{premium:.2f}  IV={iv:.1%}  T={T*365:.0f}d  [{strategy}]"
        )
        return trade

    def close_option(self, position_id: str, reason: str = "") -> dict | None:
        pos = self.state["positions"].get(position_id)
        if not pos or pos["instrument_type"] != "OPTION":
            return None

        spot = self._get_spot(pos["underlying"])
        T    = days_to_expiry(date.fromisoformat(pos["expiry"]))
        iv   = pos.get("iv", 0.25)
        curr_premium = BlackScholes.price(
            spot, pos["strike"], T, RISK_FREE_RATE, iv, pos["option_type"]
        )
        lot   = get_lot_size(pos["underlying"])
        qty   = pos["qty"] * lot
        entry = pos["entry_premium"]

        if pos["position"] == "LONG":
            pnl = (curr_premium - entry) * qty
            self.state["cash"] += curr_premium * qty
        else:
            pnl = (entry - curr_premium) * qty
            self.state["cash"] += pos.get("margin_blocked", 0)
            self.state["cash"] -= curr_premium * qty   # buy back cost

        self.state["realised_pnl"] += pnl
        del self.state["positions"][position_id]
        self._save()

        trade = {
            "action":       f"CLOSE_{pos['position']}_{pos['option_type'].upper()}",
            "underlying":   pos["underlying"],
            "strike":       pos["strike"],
            "expiry":       pos["expiry"],
            "option_type":  pos["option_type"],
            "qty_lots":     pos["qty"],
            "exit_premium": round(curr_premium, 2),
            "entry_premium":entry,
            "pnl":          round(pnl, 2),
            "reason":       reason,
        }
        self._log_trade(trade)
        logger.info(
            f"[FNO] CLOSE {pos['position']} {pos['underlying']} "
            f"{pos['strike']}{pos['option_type'][0].upper()} "
            f"P&L=₹{pnl:+.0f}  ({reason})"
        )
        return trade

    def open_future(self, underlying: str, expiry: date, position: str,
                    qty_lots: int, strategy: str, reason: str = "") -> dict | None:
        spot   = self._get_spot(underlying)
        if spot <= 0:
            return None
        lot    = get_lot_size(underlying)
        margin = spot * lot * qty_lots * 0.15   # ~15% SPAN margin
        if margin > self.state["cash"]:
            logger.warning(f"[FNO] Insufficient margin for {underlying} futures")
            return None

        pid = f"{underlying}_FUT_{expiry}_{position}_{int(time.time())}"
        pos = {
            "position_id":    pid,
            "instrument_type":"FUTURE",
            "underlying":     underlying,
            "expiry":         expiry.isoformat(),
            "position":       position,
            "qty":            qty_lots,
            "entry_price":    round(spot, 2),
            "margin_blocked": round(margin, 2),
            "strategy":       strategy,
            "reason":         reason,
            "entry_date":     datetime.now().isoformat(),
        }
        self.state["positions"][pid] = pos
        self.state["cash"] -= margin
        self._save()

        trade = {
            "action":       f"OPEN_{position}_FUTURE",
            "underlying":   underlying,
            "expiry":       expiry.isoformat(),
            "qty_lots":     qty_lots,
            "entry_price":  round(spot, 2),
            "margin":       round(margin, 2),
            "strategy":     strategy,
            "reason":       reason,
        }
        self._log_trade(trade)
        logger.info(
            f"[FNO] OPEN {position} {qty_lots}L {underlying} FUT @ ₹{spot:.2f}  "
            f"Margin=₹{margin:.0f}  [{strategy}]"
        )
        return trade

    def close_future(self, position_id: str, reason: str = "") -> dict | None:
        pos = self.state["positions"].get(position_id)
        if not pos or pos["instrument_type"] != "FUTURE":
            return None

        spot  = self._get_spot(pos["underlying"])
        lot   = get_lot_size(pos["underlying"])
        qty   = pos["qty"] * lot
        sign  = 1 if pos["position"] == "LONG" else -1
        pnl   = sign * (spot - pos["entry_price"]) * qty

        self.state["cash"]        += pos.get("margin_blocked", 0) + pnl
        self.state["realised_pnl"] += pnl
        del self.state["positions"][position_id]
        self._save()

        trade = {
            "action":      f"CLOSE_{pos['position']}_FUTURE",
            "underlying":  pos["underlying"],
            "expiry":      pos["expiry"],
            "qty_lots":    pos["qty"],
            "exit_price":  round(spot, 2),
            "entry_price": pos["entry_price"],
            "pnl":         round(pnl, 2),
            "reason":      reason,
        }
        self._log_trade(trade)
        logger.info(f"[FNO] CLOSE {pos['position']} {pos['underlying']} FUT  P&L=₹{pnl:+.0f}")
        return trade

    def check_expiry_and_stops(self) -> list:
        """
        Close positions approaching expiry (≤1 day) and enforce P&L stops:
        - Single position loss > 50% of premium paid → exit
        - Single position gain > 80% of max profit   → take profit
        """
        closed = []
        for pid, pos in list(self.state["positions"].items()):
            exp = date.fromisoformat(pos["expiry"])
            days_left = (exp - date.today()).days

            # Auto-close 1 day before expiry to avoid exercise risk
            if days_left <= 1:
                logger.info(f"[FNO] EXPIRY-CLOSE {pid} ({days_left}d left)")
                if pos["instrument_type"] == "OPTION":
                    t = self.close_option(pid, reason="EXPIRY_CLOSE")
                else:
                    t = self.close_future(pid, reason="EXPIRY_CLOSE")
                if t:
                    closed.append(t)
                continue

            # Option P&L stops
            if pos["instrument_type"] == "OPTION" and pos["position"] == "LONG":
                mtm = self._mtm_value(pos)
                cost = pos["entry_premium"] * pos["qty"] * get_lot_size(pos["underlying"])
                pnl_pct = mtm / cost if cost != 0 else 0

                if pnl_pct <= -0.50:   # Stop: lost 50% of premium
                    logger.warning(f"[FNO] OPTION STOP {pid}  lost {pnl_pct:.0%}")
                    t = self.close_option(pid, reason="OPTION_STOP_50PCT")
                    if t:
                        closed.append(t)
                elif pnl_pct >= 0.80:  # Target: up 80%
                    logger.info(f"[FNO] OPTION TARGET {pid}  gained {pnl_pct:.0%}")
                    t = self.close_option(pid, reason="OPTION_TARGET_80PCT")
                    if t:
                        closed.append(t)

        return closed

    def get_positions_display(self) -> list:
        result = []
        for pid, pos in self.state["positions"].items():
            spot = self._get_spot(pos["underlying"])
            mtm  = self._mtm_value(pos)
            T    = days_to_expiry(date.fromisoformat(pos["expiry"]))

            if pos["instrument_type"] == "OPTION":
                curr_prem = BlackScholes.price(
                    spot, pos["strike"], T,
                    RISK_FREE_RATE, pos.get("iv", 0.25), pos["option_type"]
                )
                greeks = BlackScholes.greeks(
                    spot, pos["strike"], T,
                    RISK_FREE_RATE, pos.get("iv", 0.25), pos["option_type"]
                )
                lot  = get_lot_size(pos["underlying"])
                cost = pos["entry_premium"] * pos["qty"] * lot
                result.append({
                    "position_id":   pid,
                    "instrument":    f"{pos['underlying']} {pos['strike']}{pos['option_type'][0].upper()}",
                    "type":          f"{pos['position']} {pos['option_type'].upper()}",
                    "expiry":        pos["expiry"],
                    "days_left":     int(T * 365),
                    "qty_lots":      pos["qty"],
                    "entry_premium": pos["entry_premium"],
                    "curr_premium":  round(curr_prem, 2),
                    "mtm":           round(mtm, 2),
                    "pnl_pct":       round(mtm / cost * 100, 1) if cost else 0,
                    "delta":         greeks["delta"],
                    "theta":         greeks["theta"],
                    "vega":          greeks["vega"],
                    "strategy":      pos["strategy"],
                    "spot":          round(spot, 2),
                    "iv":            pos.get("iv", 0.25),
                })
            else:
                lot = get_lot_size(pos["underlying"])
                qty = pos["qty"] * lot
                result.append({
                    "position_id": pid,
                    "instrument":  f"{pos['underlying']} FUT",
                    "type":        f"{pos['position']} FUTURE",
                    "expiry":      pos["expiry"],
                    "days_left":   int(T * 365),
                    "qty_lots":    pos["qty"],
                    "entry_price": pos["entry_price"],
                    "curr_price":  round(spot, 2),
                    "mtm":         round(mtm, 2),
                    "pnl_pct":     round((spot / pos["entry_price"] - 1) * 100, 2),
                    "strategy":    pos["strategy"],
                })
        return result

    def reset(self):
        self.state = self._default_state()
        self._save()


# ---------------------------------------------------------------------------
# F&O Strategies
# ---------------------------------------------------------------------------

class DirectionalOptionsStrategy:
    """
    Route strong equity signals to call/put options for leverage.

    Signal strength ≥ 80 BUY  → buy ATM call (1 lot)
    Signal strength ≥ 80 SELL → buy ATM put  (1 lot)
    Uses the nearest weekly expiry for indices, monthly for stocks.
    """
    name       = "Directional Options"
    short_name = "DIR_OPT"

    def run(self, signals: list, portfolio: FNOPortfolio) -> list:
        executed = []
        for sig in signals:
            if sig.get("strength", 0) < 80:
                continue
            ticker = sig["ticker"]
            action = sig.get("signal", "")
            if action not in ("BUY", "SELL"):
                continue

            opt_type = "call" if action == "BUY" else "put"
            is_index = ticker in ("^NSEI", "^NSEBANK", "NIFTY", "BANKNIFTY")
            monthly  = not is_index

            try:
                expiry = get_expiry(ticker, monthly=monthly)
                spot   = portfolio._get_spot(ticker)
                if spot <= 0:
                    continue

                strike = select_strike(spot, opt_type, moneyness="ATM")
                T      = days_to_expiry(expiry)
                iv     = historical_vol(ticker)
                prem   = BlackScholes.price(spot, strike, T, RISK_FREE_RATE, iv, opt_type)
                lot    = get_lot_size(ticker)
                cost   = prem * lot * 1   # 1 lot

                if cost > portfolio.state["cash"] * 0.20:   # max 20% of F&O capital
                    continue
                if cost < 100:   # nonsensical premium
                    continue

                result = portfolio.open_option(
                    underlying=ticker, strike=strike, expiry=expiry,
                    option_type=opt_type, position="LONG", qty_lots=1,
                    strategy=self.short_name,
                    reason=f"Equity signal {action} strength={sig['strength']:.0f}"
                )
                if result:
                    executed.append(result)
            except Exception as e:
                logger.warning(f"[DirOpt] Error on {ticker}: {e}")

        return executed


class SpreadStrategy:
    """
    Bull Call Spread (strong BUY) or Bear Put Spread (strong SELL).
    Defined risk, defined reward. Cheaper than outright options.

    Bull Call Spread: buy ATM call + sell OTM1 call
    Bear Put Spread:  buy ATM put  + sell OTM1 put (lower strike)
    """
    name       = "Spreads"
    short_name = "SPREAD"

    def run(self, signals: list, portfolio: FNOPortfolio) -> list:
        executed = []
        for sig in signals:
            if sig.get("strength", 0) < 75:
                continue
            ticker = sig["ticker"]
            action = sig.get("signal", "")
            if action not in ("BUY", "SELL"):
                continue

            try:
                expiry = get_expiry(ticker, monthly=True)
                spot   = portfolio._get_spot(ticker)
                if spot <= 0:
                    continue

                if action == "BUY":
                    # Bull call spread
                    k_long  = select_strike(spot, "call", "ATM")
                    k_short = select_strike(spot, "call", "OTM1")
                    T       = days_to_expiry(expiry)
                    iv      = historical_vol(ticker)
                    prem_l  = BlackScholes.price(spot, k_long,  T, RISK_FREE_RATE, iv, "call")
                    prem_s  = BlackScholes.price(spot, k_short, T, RISK_FREE_RATE, iv, "call")
                    net_cost = (prem_l - prem_s) * get_lot_size(ticker)

                    if net_cost > portfolio.state["cash"] * 0.15:
                        continue

                    r1 = portfolio.open_option(ticker, k_long,  expiry, "call", "LONG",  1, self.short_name, "BullCallSpread-BuyLeg")
                    r2 = portfolio.open_option(ticker, k_short, expiry, "call", "SHORT", 1, self.short_name, "BullCallSpread-SellLeg")
                    if r1 and r2:
                        executed.extend([r1, r2])

                else:
                    # Bear put spread
                    k_long  = select_strike(spot, "put", "ATM")
                    k_short = select_strike(spot, "put", "OTM1")
                    T       = days_to_expiry(expiry)
                    iv      = historical_vol(ticker)
                    prem_l  = BlackScholes.price(spot, k_long,  T, RISK_FREE_RATE, iv, "put")
                    prem_s  = BlackScholes.price(spot, k_short, T, RISK_FREE_RATE, iv, "put")
                    net_cost = (prem_l - prem_s) * get_lot_size(ticker)

                    if net_cost > portfolio.state["cash"] * 0.15:
                        continue

                    r1 = portfolio.open_option(ticker, k_long,  expiry, "put", "LONG",  1, self.short_name, "BearPutSpread-BuyLeg")
                    r2 = portfolio.open_option(ticker, k_short, expiry, "put", "SHORT", 1, self.short_name, "BearPutSpread-SellLeg")
                    if r1 and r2:
                        executed.extend([r1, r2])

            except Exception as e:
                logger.warning(f"[Spread] Error on {ticker}: {e}")

        return executed


class IronCondorStrategy:
    """
    Weekly Iron Condor on Nifty / BankNifty.
    Sell OTM call + buy further OTM call (call spread wing)
    Sell OTM put  + buy further OTM put  (put spread wing)
    Collect premium when index stays rangebound.

    Runs on Monday/Tuesday, targeting weekly expiry.
    Premium target: collect ≥ 0.5% of notional.
    """
    name       = "Iron Condor"
    short_name = "IRON_CONDOR"

    INDICES = ["^NSEI", "^NSEBANK"]

    def run(self, portfolio: FNOPortfolio) -> list:
        from datetime import datetime as _dt
        today   = date.today()
        weekday = today.weekday()   # 0=Mon .. 4=Fri
        # Only initiate on Monday or Tuesday
        if weekday not in (0, 1):
            return []

        executed = []
        for ticker in self.INDICES:
            try:
                spot   = portfolio._get_spot(ticker)
                if spot <= 0:
                    continue
                expiry = get_expiry(ticker, monthly=False)
                T      = days_to_expiry(expiry)
                if T * 365 < 3:   # too close to expiry
                    continue

                iv = historical_vol(ticker)
                # Wing strikes: ±1.5 standard deviations (~85% probability of expiring OTM)
                std_move = spot * iv * math.sqrt(T)
                call_sell = select_strike(spot, "call", step=None)
                put_sell  = select_strike(spot, "put",  step=None)

                # Short strikes: ±1.5σ from spot
                step = 100 if spot > 10000 else 50
                call_sell = round((spot + 1.5 * std_move) / step) * step
                put_sell  = round((spot - 1.5 * std_move) / step) * step
                call_buy  = call_sell + 2 * step
                put_buy   = put_sell  - 2 * step

                lot = get_lot_size(ticker)
                # Net premium estimate
                cs_prem = BlackScholes.price(spot, call_sell, T, RISK_FREE_RATE, iv, "call")
                ps_prem = BlackScholes.price(spot, put_sell,  T, RISK_FREE_RATE, iv, "put")
                cb_prem = BlackScholes.price(spot, call_buy,  T, RISK_FREE_RATE, iv, "call")
                pb_prem = BlackScholes.price(spot, put_buy,   T, RISK_FREE_RATE, iv, "put")
                net_credit = (cs_prem + ps_prem - cb_prem - pb_prem) * lot

                if net_credit < 50:   # minimum ₹50 credit per condor
                    continue

                r1 = portfolio.open_option(ticker, call_sell, expiry, "call", "SHORT", 1, self.short_name, "IronCondor-ShortCall")
                r2 = portfolio.open_option(ticker, call_buy,  expiry, "call", "LONG",  1, self.short_name, "IronCondor-LongCall")
                r3 = portfolio.open_option(ticker, put_sell,  expiry, "put",  "SHORT", 1, self.short_name, "IronCondor-ShortPut")
                r4 = portfolio.open_option(ticker, put_buy,   expiry, "put",  "LONG",  1, self.short_name, "IronCondor-LongPut")

                for r in [r1, r2, r3, r4]:
                    if r:
                        executed.append(r)

                logger.info(
                    f"[IronCondor] {ticker} @ {spot:.0f}  "
                    f"Calls: {put_sell}/{put_buy}  Puts: {call_sell}/{call_buy}  "
                    f"Net credit: ₹{net_credit:.0f}"
                )
            except Exception as e:
                logger.warning(f"[IronCondor] Error on {ticker}: {e}")

        return executed


class HedgeStrategy:
    """
    Protective puts on equity positions.
    When equity portfolio drawdown exceeds 3%, buy ATM puts on held stocks
    to cap further downside.
    """
    name       = "Portfolio Hedge"
    short_name = "HEDGE"

    def run(self, equity_positions: list, portfolio: FNOPortfolio,
            equity_drawdown_pct: float) -> list:
        if equity_drawdown_pct < 3.0:
            return []

        executed = []
        # Already have a hedge on for this position?
        hedged = {
            pos["underlying"]
            for pos in portfolio.state["positions"].values()
            if pos.get("strategy") == self.short_name
        }

        for eq_pos in equity_positions:
            ticker = eq_pos.get("ticker", "")
            if ticker in hedged:
                continue
            if ticker not in LOT_SIZES:
                continue

            try:
                spot   = portfolio._get_spot(ticker)
                if spot <= 0:
                    continue
                expiry = get_expiry(ticker, monthly=True)
                strike = select_strike(spot, "put", moneyness="ATM")
                T      = days_to_expiry(expiry)
                iv     = historical_vol(ticker)
                prem   = BlackScholes.price(spot, strike, T, RISK_FREE_RATE, iv, "put")
                lot    = get_lot_size(ticker)
                cost   = prem * lot

                # Hedge cost must be < 1% of equity position value
                eq_value = eq_pos.get("qty", 1) * spot
                if cost > eq_value * 0.01:
                    continue
                if cost > portfolio.state["cash"] * 0.10:
                    continue

                result = portfolio.open_option(
                    underlying=ticker, strike=strike, expiry=expiry,
                    option_type="put", position="LONG", qty_lots=1,
                    strategy=self.short_name,
                    reason=f"Portfolio hedge — equity drawdown {equity_drawdown_pct:.1f}%"
                )
                if result:
                    executed.append(result)
                    hedged.add(ticker)
            except Exception as e:
                logger.warning(f"[Hedge] Error on {ticker}: {e}")

        return executed


# ---------------------------------------------------------------------------
# F&O Agent — orchestrates all strategies
# ---------------------------------------------------------------------------

class FNOAgent:
    """
    Runs all F&O strategies in sequence each cycle.
    Called from the main TradingAgent.run_cycle().
    """

    def __init__(self):
        self.portfolio     = FNOPortfolio()
        self.directional   = DirectionalOptionsStrategy()
        self.spreads       = SpreadStrategy()
        self.iron_condor   = IronCondorStrategy()
        self.hedge         = HedgeStrategy()

    def run_cycle(self, equity_signals: list = None,
                  equity_positions: list = None,
                  equity_drawdown_pct: float = 0.0) -> dict:
        """
        Full F&O cycle:
        1. Check expiry and stops on existing positions
        2. Run Iron Condor (index weekly)
        3. Run Directional Options on strong equity signals
        4. Run Spread strategy
        5. Run Hedge if equity is in drawdown
        """
        equity_signals   = equity_signals or []
        equity_positions = equity_positions or []

        logger.info("[FNO] === F&O Cycle Start ===")
        stops    = self.portfolio.check_expiry_and_stops()
        condors  = self.iron_condor.run(self.portfolio)
        directs  = self.directional.run(equity_signals, self.portfolio)
        spreads  = self.spreads.run(equity_signals, self.portfolio)
        hedges   = self.hedge.run(equity_positions, self.portfolio, equity_drawdown_pct)

        total_value = self.portfolio.get_total_value()
        greeks      = self.portfolio.portfolio_greeks()
        pnl         = total_value - self.portfolio.state["initial"] + self.portfolio.state["realised_pnl"]

        summary = {
            "fno_value":    total_value,
            "fno_pnl":      round(pnl, 2),
            "fno_cash":     round(self.portfolio.state["cash"], 2),
            "stops_closed": len(stops),
            "new_condors":  len(condors) // 4,
            "new_directs":  len(directs),
            "new_spreads":  len(spreads) // 2,
            "new_hedges":   len(hedges),
            "open_positions": len(self.portfolio.state["positions"]),
            "portfolio_greeks": greeks,
        }
        logger.info(f"[FNO] Cycle done — value=₹{total_value:,.0f}  Greeks={greeks}")
        return summary

    def get_dashboard_data(self) -> dict:
        return {
            "portfolio_value": self.portfolio.get_total_value(),
            "cash":            self.portfolio.state["cash"],
            "initial":         self.portfolio.state["initial"],
            "realised_pnl":    self.portfolio.state["realised_pnl"],
            "unrealised_pnl":  self.portfolio.get_total_value() - self.portfolio.state["cash"] - self.portfolio.state["realised_pnl"],
            "positions":       self.portfolio.get_positions_display(),
            "greeks":          self.portfolio.portfolio_greeks(),
            "open_count":      len(self.portfolio.state["positions"]),
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_fno_agent: FNOAgent | None = None

def get_fno_agent() -> FNOAgent:
    global _fno_agent
    if _fno_agent is None:
        _fno_agent = FNOAgent()
    return _fno_agent
