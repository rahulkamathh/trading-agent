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
from zoneinfo import ZoneInfo
_IST = ZoneInfo("Asia/Kolkata")
def _now_ist() -> datetime:
    return datetime.now(_IST)
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
FNO_MAX_POSITIONS   = 10         # hard cap on open F&O positions
FNO_MAX_RISK_PER_TRADE = 0.10   # max 10% of F&O capital per trade
RISK_FREE_RATE      = 0.0675     # RBI repo rate approximation
MAX_RISK_PER_TRADE  = 0.05       # max 5% of F&O capital at risk per trade


# ---------------------------------------------------------------------------
# Lot sizes — fetched live from NSE, cached locally
# ---------------------------------------------------------------------------

_LOT_CACHE_FILE = DATA_DIR / "nse_lot_sizes.json"
_LOT_CACHE: dict[str, int] = {}
_LOT_CACHE_DATE: str = ""   # "YYYY-MM-DD" of last successful fetch
# NSE lot sizes change quarterly (per SEBI circular) — refresh once per day is plenty.
# On first call each calendar day, fetch fresh from NSE. Otherwise use local cache.

def _fetch_lot_sizes_from_nse() -> dict[str, int]:
    """
    Fetch current F&O lot sizes from NSE's public API.
    Returns dict: {ticker_with_NS_suffix: lot_size}
    Falls back to cached file if NSE is unreachable.
    """
    from urllib.request import urlopen, Request
    from urllib.error import URLError
    import re as _re

    url = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,*/*",
        "Referer": "https://www.nseindia.com/",
    }
    try:
        req  = Request(url, headers=headers)
        resp = urlopen(req, timeout=10)
        raw  = resp.read().decode("utf-8", errors="replace")
        lots: dict[str, int] = {}
        for line in raw.splitlines()[1:]:   # skip header
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            sym = parts[1].strip().upper()
            try:
                lot = int(parts[2].strip())
            except ValueError:
                continue
            if not sym or lot <= 0:
                continue
            # Index symbols
            if sym in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"):
                index_map = {
                    "NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK",
                    "FINNIFTY": "FINNIFTY", "MIDCPNIFTY": "MIDCPNIFTY",
                }
                lots[index_map.get(sym, sym)] = lot
                lots[sym] = lot
            else:
                lots[sym + ".NS"] = lot
                lots[sym] = lot
        logger.info(f"[FNO] Fetched {len(lots)} lot sizes from NSE")
        return lots
    except Exception as exc:
        logger.warning(f"[FNO] Could not fetch lot sizes from NSE: {exc}")
        return {}

def _load_lot_cache() -> dict[str, int]:
    """Load lot sizes from local cache file."""
    if _LOT_CACHE_FILE.exists():
        try:
            with open(_LOT_CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_lot_cache(lots: dict):
    try:
        with open(_LOT_CACHE_FILE, "w") as f:
            json.dump(lots, f)
    except Exception:
        pass

def get_lot_size(ticker: str) -> int:
    """
    Return NSE F&O lot size for a given ticker.

    Refresh policy: once per calendar day (lot sizes are set quarterly by SEBI,
    never change intraday). Falls back to local cache file if NSE unreachable.
    Returns 1 for unknown instruments to avoid division errors.
    """
    global _LOT_CACHE, _LOT_CACHE_DATE

    today = date.today().isoformat()
    if _LOT_CACHE_DATE != today:
        # New day — attempt a fresh fetch from NSE
        fresh = _fetch_lot_sizes_from_nse()
        if fresh:
            _LOT_CACHE      = fresh
            _LOT_CACHE_DATE = today
            _save_lot_cache(fresh)
        else:
            # NSE unreachable — load from disk cache and stay on that
            if not _LOT_CACHE:
                _LOT_CACHE = _load_lot_cache()
            _LOT_CACHE_DATE = today   # don't retry until tomorrow

    # Try exact match, then without .NS suffix, then bare symbol
    for key in (ticker, ticker.replace(".NS", ""), ticker.replace(".NS", "") + ".NS"):
        if key in _LOT_CACHE:
            return _LOT_CACHE[key]
    return 1   # unknown instrument — return 1 to avoid division errors


def is_fno_eligible(ticker: str) -> bool:
    """
    Return True if the ticker has an active F&O contract on NSE.
    Uses the lot size cache — if a lot size exists for this ticker,
    NSE recognises it as an F&O instrument.
    Returns False for stocks with lot size == 1 (our sentinel for 'unknown').
    """
    # Ensure cache is loaded
    get_lot_size(ticker)   # triggers cache refresh if needed
    for key in (ticker, ticker.replace(".NS", ""), ticker.replace(".NS", "") + ".NS"):
        if key in _LOT_CACHE:
            return True
    return False


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
            "created_at":    _now_ist().isoformat(),
            "last_updated":  _now_ist().isoformat(),
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
        self.state["last_updated"] = _now_ist().isoformat()
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
        trade["time"] = _now_ist().isoformat()
        log.append(trade)
        with open(FNO_TRADE_FILE, "w") as f:
            json.dump(log, f, indent=2)
        return trade

    def in_cooldown(self, underlying: str, hours: float = 4.0) -> bool:
        """True if this underlying was closed within the last `hours` hours."""
        cutoff = (_now_ist() - timedelta(hours=hours)).isoformat()
        try:
            if FNO_TRADE_FILE.exists():
                with open(FNO_TRADE_FILE) as f:
                    log = json.load(f)
                for t in reversed(log):
                    if t.get("underlying") == underlying and "CLOSE" in t.get("action", ""):
                        if t.get("time", "") >= cutoff:
                            return True
                        break  # log is chronological, stop once past cutoff
        except Exception:
            pass
        return False

    def has_open_position(self, underlying: str, option_type: str = None) -> bool:
        """True if we already have an open position on this underlying (optionally filtered by type)."""
        for pos in self.state["positions"].values():
            if pos["underlying"] == underlying:
                if option_type is None or pos.get("option_type") == option_type:
                    return True
        return False

    def at_max_positions(self) -> bool:
        """True if we've hit the F&O position cap."""
        return len(self.state["positions"]) >= FNO_MAX_POSITIONS

    def can_afford(self, cost: float) -> bool:
        """True if we have enough cash and cost is within single-trade risk limit."""
        if self.state["cash"] < cost:
            return False
        if cost > self.state.get("initial", FNO_INITIAL_CAPITAL) * FNO_MAX_RISK_PER_TRADE:
            return False
        return True

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
        # Hard gate: no new F&O positions in last 30 min of trading (after 3:00 PM IST)
        # Positions opened near close get immediately closed by end-of-day cleanup at ₹0
        now_time = _now_ist().time()
        from datetime import time as _dtt  # noqa: PLC0415
        if not (_dtt(9, 15) <= now_time <= _dtt(15, 0)):
            logger.info(f"[FNO] BLOCKED open_option {underlying} — outside window (after 3PM or pre-market)")
            return None

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
            "entry_date":     _now_ist().isoformat(),
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
        # Telegram alert
        if position == "LONG":
            try:
                from notifier import get_notifier  # noqa: PLC0415
                get_notifier().notify_fno_open(
                    underlying=underlying, option_type=option_type,
                    strike=strike, expiry=expiry.isoformat(),
                    premium=premium, lot_size=lot, qty_lots=qty_lots,
                    strategy=strategy, reason=reason, strength=0,
                )
            except Exception:
                pass
        return trade

    def execute_sell_signal_as_put(
        self,
        ticker: str,
        spot: float,
        strength: float,
        strategy: str,
        reason: str = "",
    ) -> dict | None:
        """
        Convert an equity SELL signal into a PUT option buy.
        Called when the equity desk fires a bearish signal on an F&O-eligible stock.

        Strike selection:
          - Strength ≥ 85  → ATM put (high conviction — maximum delta)
          - Strength 70-85 → OTM1 put (moderate — cheaper, more leverage)
          - Below 70       → skip (not enough conviction for options)

        Expiry: nearest monthly (more liquidity than weeklies for stock options).
        Position size: capped at 2% of F&O portfolio per trade.
        """
        if strength < 70:
            logger.debug(f"[FNO] SELL signal on {ticker} too weak ({strength:.0f}) for PUT — skipping")
            return None

        if not is_fno_eligible(ticker):
            logger.info(f"[FNO] {ticker} not F&O eligible — cannot express bearish view as PUT")
            return None

        if self.at_max_positions():
            logger.info(f"[FNO] PUT {ticker} skipped — at max {FNO_MAX_POSITIONS} positions")
            return None

        if self.has_open_position(ticker, "put"):
            logger.debug(f"[FNO] PUT {ticker} skipped — already have open PUT position")
            return None

        if self.in_cooldown(ticker):
            logger.info(f"[FNO] PUT {ticker} skipped — in 4h cooldown after recent close")
            return None

        moneyness = "ATM" if strength >= 85 else "OTM1"
        expiry    = get_expiry(ticker, monthly=True)   # monthly for liquidity
        T         = days_to_expiry(expiry)

        if T < 7:
            # Too close to expiry — force next month
            today = date.today()
            next_m = today.month % 12 + 1
            next_y = today.year + (1 if today.month == 12 else 0)
            expiry = _last_thursday(next_y, next_m)
            T      = days_to_expiry(expiry)

        iv      = historical_vol(ticker)
        strike  = select_strike(spot, "put", moneyness=moneyness)
        premium = BlackScholes.price(spot, strike, T, RISK_FREE_RATE, iv, "put")
        greeks  = BlackScholes.greeks(spot, strike, T, RISK_FREE_RATE, iv, "put")
        lot     = get_lot_size(ticker)

        if premium < 10:
            logger.info(f"[FNO] PUT {ticker} skipped — premium ₹{premium:.2f} below ₹10 minimum")
            return None

        # Position sizing: max 10% of F&O capital per PUT trade, minimum 1 lot
        max_spend = self.state.get("initial", FNO_INITIAL_CAPITAL) * FNO_MAX_RISK_PER_TRADE
        qty_lots  = max(1, int(max_spend / (premium * lot)))
        cost      = premium * lot * qty_lots
        if not self.can_afford(cost):
            # Try 1 lot minimum
            if not self.can_afford(premium * lot):
                logger.info(f"[FNO] PUT {ticker} skipped — insufficient cash (need ₹{premium*lot:.0f})")
                return None
            qty_lots = 1

        full_reason = (
            f"Equity SELL→PUT | {reason} | "
            f"moneyness={moneyness} IV={iv:.1%} delta={greeks['delta']:.2f}"
        )

        trade = self.open_option(
            underlying  = ticker,
            strike      = strike,
            expiry      = expiry,
            option_type = "put",
            position    = "LONG",
            qty_lots    = qty_lots,
            strategy    = f"EQ_SELL_PUT|{strategy}",
            reason      = full_reason,
        )
        if trade:
            logger.info(
                f"[FNO] 🔴 Expressed SELL {ticker} as PUT: "
                f"{qty_lots}L {strike}PE {expiry} @ ₹{premium:.2f}  "
                f"total=₹{premium*lot*qty_lots:.0f}  strength={strength:.0f}"
            )
        return trade

    def close_option(self, position_id: str, reason: str = "") -> dict | None:
        pos = self.state["positions"].get(position_id)
        if not pos or pos["instrument_type"] != "OPTION":
            return None

        spot = self._get_spot(pos["underlying"])
        T    = days_to_expiry(date.fromisoformat(pos["expiry"]))
        iv   = pos.get("iv", 0.25)

        # Try Angel One live option LTP first (actual market price)
        # Falls back to Black-Scholes if Angel One unavailable
        curr_premium = None
        try:
            from angelone_feed import get_feed as _get_feed  # noqa: PLC0415
            _feed = _get_feed()
            if _feed.is_connected() and _feed._smart:
                live_ltp = _feed.get_option_ltp(
                    pos["underlying"], pos["strike"],
                    date.fromisoformat(pos["expiry"]), pos["option_type"]
                )
                if live_ltp and live_ltp > 0:
                    curr_premium = live_ltp
                    logger.debug(f"[FNO] Using Angel One LTP ₹{live_ltp:.2f} for {pos['underlying']} {pos['strike']}{pos['option_type'][0].upper()}")
        except Exception as _ltp_err:
            logger.debug(f"[FNO] Angel One LTP failed, using BS: {_ltp_err}")

        if curr_premium is None:
            curr_premium = BlackScholes.price(spot, pos["strike"], T, RISK_FREE_RATE, iv, pos["option_type"])

        # Safety: don't close at ₹0 — market is likely closed or data is bad
        if curr_premium <= 0:
            logger.warning(f"[FNO] SKIP close {position_id} — premium computed as ₹0 (market closed or bad data)")
            return None
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
        try:
            from notifier import get_notifier  # noqa: PLC0415
            get_notifier().notify_fno_close(
                underlying=pos["underlying"], option_type=pos["option_type"],
                strike=pos["strike"], entry_premium=pos["entry_premium"],
                exit_premium=round(curr_premium, 2),
                qty_lots=pos["qty"], lot_size=get_lot_size(pos["underlying"]),
                pnl=round(pnl, 2), reason=reason,
            )
        except Exception:
            pass
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
            "entry_date":     _now_ist().isoformat(),
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

            # Option P&L stops — only during market hours 9:15–15:00
            # After 3PM, BS prices become unreliable as liquidity dries up
            now_t = _now_ist().time()
            from datetime import time as _dtt2  # noqa: PLC0415
            stops_active = _dtt2(9, 15) <= now_t <= _dtt2(15, 0)

            if pos["instrument_type"] == "OPTION" and pos["position"] == "LONG" and stops_active:
                mtm = self._mtm_value(pos)
                cost = pos["entry_premium"] * pos["qty"] * get_lot_size(pos["underlying"])
                pnl_pct = mtm / cost if cost != 0 else 0

                if pnl_pct <= -0.50:   # Stop: lost 50% of premium
                    logger.warning(f"[FNO] OPTION STOP {pid}  lost {pnl_pct:.0%}")
                    t = self.close_option(pid, reason="STOP_LOSS_50PCT")
                    if t:
                        # Override reason based on actual P&L (model MTM can lie)
                        actual_pnl = t.get("pnl", 0)
                        t["reason"] = "STOP_LOSS" if actual_pnl < 0 else "STOP_LOSS_50PCT"
                        closed.append(t)
                elif pnl_pct >= 0.80:  # Target: up 80%
                    logger.info(f"[FNO] OPTION TARGET {pid}  gained {pnl_pct:.0%}")
                    t = self.close_option(pid, reason="TAKE_PROFIT_80PCT")
                    if t:
                        # If actual P&L is negative despite model saying +80%, label correctly
                        actual_pnl = t.get("pnl", 0)
                        t["reason"] = "TAKE_PROFIT" if actual_pnl >= 0 else "MODEL_EXIT_LOSS"
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
            if portfolio.at_max_positions():
                break
            if sig.get("strength", 0) < 80:
                continue
            ticker = sig["ticker"]
            action = sig.get("signal", "")
            if action not in ("BUY", "SELL"):
                continue

            opt_type = "call" if action == "BUY" else "put"

            # Skip if we already have this underlying open or recently closed
            if portfolio.has_open_position(ticker, opt_type):
                continue
            if portfolio.in_cooldown(ticker):
                continue

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
                cost   = prem * lot

                if prem < 10:    # minimum ₹10 premium — below this spreads are too wide
                    logger.info(f"[DirOpt] SKIP {ticker} — premium ₹{prem:.2f} below ₹10 minimum")
                    continue
                if cost < 500:   # minimum total outlay
                    continue
                if not portfolio.can_afford(cost):
                    logger.info(f"[DirOpt] SKIP {ticker} — cannot afford ₹{cost:.0f} (cash=₹{portfolio.state['cash']:.0f})")
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
# Hourly F&O Signal Generator
# ---------------------------------------------------------------------------

FNO_HOURLY_SIGNALS_FILE = DATA_DIR / "fno_hourly_signals.json"

class HourlyFNOStrategy:
    """
    Generates intraday F&O signals every hour using 1h candle data.

    Signals cover:
      1. Index momentum   — Nifty / BankNifty EMA crossover + MACD on 1h chart
      2. IV environment   — compare current HV to 30d avg; low IV → buy options,
                            high IV → sell premium
      3. PCR proxy        — put/call premium ratio from yfinance chain data
      4. Strike targeting — which specific strike + expiry to trade
      5. Intraday S/R     — VWAP deviation for entry timing

    Each signal includes: direction, instrument, strike, expiry, premium estimate,
    entry/SL/target, conviction score, and plain-English reason.
    """

    INDICES = {
        "^NSEI":    "Nifty 50",
        "^NSEBANK": "Bank Nifty",
    }

    def generate(self) -> list[dict]:
        signals = []
        for ticker, name in self.INDICES.items():
            try:
                sigs = self._analyse_index(ticker, name)
                signals.extend(sigs)
            except Exception as exc:
                logger.warning(f"[HourlyFNO] Error on {name}: {exc}")
        # Also scan top F&O stocks for directional opportunity
        for stock in ["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
                      "TCS.NS", "SBIN.NS", "AXISBANK.NS", "BAJFINANCE.NS"]:
            try:
                sigs = self._analyse_stock(stock)
                signals.extend(sigs)
            except Exception:
                pass
        return signals

    def _fetch_hourly(self, ticker: str, days: int = 10) -> pd.DataFrame:
        df = yf.download(ticker, period=f"{days}d", interval="1h",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    def _ema(self, series: pd.Series, n: int) -> pd.Series:
        return series.ewm(span=n, adjust=False).mean()

    def _macd(self, close: pd.Series):
        fast = self._ema(close, 12)
        slow = self._ema(close, 26)
        macd = fast - slow
        signal = self._ema(macd, 9)
        return macd, signal

    def _vwap(self, df: pd.DataFrame) -> float:
        """Today's VWAP."""
        from datetime import datetime as _dt
        today = _dt.now().date()
        today_df = df[df.index.date == today] if hasattr(df.index, 'date') else df.iloc[-8:]
        if today_df.empty:
            return float(df["Close"].iloc[-1])
        tp = (today_df["High"] + today_df["Low"] + today_df["Close"]) / 3
        return float((tp * today_df["Volume"]).sum() / today_df["Volume"].sum())

    def _iv_environment(self, ticker: str) -> dict:
        """
        Compare current 10-day HV to 30-day HV.
        Returns: {'iv_regime': 'LOW'|'NORMAL'|'HIGH', 'hv10': x, 'hv30': x}
        """
        hv10 = historical_vol(ticker, window=10)
        hv30 = historical_vol(ticker, window=30)
        ratio = hv10 / hv30 if hv30 > 0 else 1.0
        if ratio < 0.80:
            regime = "LOW"      # IV compressed → buy options (cheap)
        elif ratio > 1.20:
            regime = "HIGH"     # IV elevated  → sell premium
        else:
            regime = "NORMAL"
        return {"iv_regime": regime, "hv10": round(hv10, 4), "hv30": round(hv30, 4), "ratio": round(ratio, 2)}

    def _pcr_proxy(self, ticker: str, spot: float) -> dict:
        """
        Estimate PCR using yfinance option chain.
        Returns put_vol / call_vol ratio and sentiment.
        """
        try:
            import yfinance as _yf
            tk  = _yf.Ticker(ticker)
            exp = tk.options
            if not exp:
                return {"pcr": 1.0, "sentiment": "NEUTRAL"}
            chain = tk.option_chain(exp[0])
            call_oi = float(chain.calls["openInterest"].sum())
            put_oi  = float(chain.puts["openInterest"].sum())
            pcr = put_oi / call_oi if call_oi > 0 else 1.0
            if pcr > 1.3:
                sentiment = "BEARISH"   # heavy put buying
            elif pcr < 0.7:
                sentiment = "BULLISH"   # heavy call buying
            else:
                sentiment = "NEUTRAL"
            return {"pcr": round(pcr, 2), "sentiment": sentiment}
        except Exception:
            return {"pcr": 1.0, "sentiment": "NEUTRAL"}

    def _analyse_index(self, ticker: str, name: str) -> list[dict]:
        df = self._fetch_hourly(ticker, days=15)
        if df.empty or len(df) < 30:
            return []

        close  = df["Close"]
        spot   = float(close.iloc[-1])
        ema9   = float(self._ema(close, 9).iloc[-1])
        ema21  = float(self._ema(close, 21).iloc[-1])
        ema9_1 = float(self._ema(close, 9).iloc[-2])
        ema21_1= float(self._ema(close, 21).iloc[-2])
        macd, macd_sig = self._macd(close)
        macd_v  = float(macd.iloc[-1])
        macd_s  = float(macd_sig.iloc[-1])
        macd_v1 = float(macd.iloc[-2])
        macd_s1 = float(macd_sig.iloc[-2])

        # VWAP position
        vwap = self._vwap(df)
        above_vwap = spot > vwap
        vwap_dev   = (spot - vwap) / vwap * 100

        # IV environment
        iv_env = self._iv_environment(ticker)

        # PCR
        pcr    = self._pcr_proxy(ticker, spot)

        signals = []

        # ── Signal 1: EMA crossover (golden/death cross on 1h) ──────────── #
        golden_cross = ema9_1 < ema21_1 and ema9 > ema21
        death_cross  = ema9_1 > ema21_1 and ema9 < ema21

        # ── Signal 2: MACD crossover on 1h ──────────────────────────────── #
        macd_bull = macd_v1 < macd_s1 and macd_v > macd_s
        macd_bear = macd_v1 > macd_s1 and macd_v < macd_s

        # ── Determine direction ──────────────────────────────────────────── #
        bullish_count = sum([golden_cross, macd_bull, above_vwap, pcr["sentiment"] == "BULLISH"])
        bearish_count = sum([death_cross,  macd_bear, not above_vwap, pcr["sentiment"] == "BEARISH"])

        if bullish_count >= 2:
            direction   = "BUY"
            option_type = "call"
            conviction  = min(95, 55 + bullish_count * 10)
        elif bearish_count >= 2:
            direction   = "SELL"
            option_type = "put"
            conviction  = min(95, 55 + bearish_count * 10)
        else:
            return []   # no clear signal

        # ── Strike & expiry selection ───────────────────────────────────── #
        expiry  = get_expiry(ticker, monthly=False)
        T       = days_to_expiry(expiry)
        iv      = iv_env["hv30"]

        # IV regime → moneyness preference
        # Low IV → buy OTM (cheap, more leverage)
        # High IV → buy ATM or ITM (less vega exposure)
        if iv_env["iv_regime"] == "LOW":
            moneyness = "OTM1"
            trade_note = "Low IV — buying OTM for leverage"
        elif iv_env["iv_regime"] == "HIGH":
            moneyness = "ATM"
            trade_note = "High IV — staying ATM to limit vega exposure"
        else:
            moneyness = "ATM"
            trade_note = "Normal IV environment"

        strike  = select_strike(spot, option_type, moneyness=moneyness)
        premium = BlackScholes.price(spot, strike, T, RISK_FREE_RATE, iv, option_type)
        greeks  = BlackScholes.greeks(spot, strike, T, RISK_FREE_RATE, iv, option_type)
        lot     = get_lot_size(ticker)

        # Risk/reward: SL at 40% of premium, target at 80%
        sl_prem     = round(premium * 0.60, 2)   # exit if falls to 60% of entry
        target_prem = round(premium * 1.80, 2)   # exit at 180% (80% gain)

        reason_parts = []
        if golden_cross: reason_parts.append("1h EMA golden cross")
        if death_cross:  reason_parts.append("1h EMA death cross")
        if macd_bull:    reason_parts.append("MACD bullish crossover (1h)")
        if macd_bear:    reason_parts.append("MACD bearish crossover (1h)")
        reason_parts.append(f"VWAP {vwap_dev:+.2f}%")
        reason_parts.append(f"PCR {pcr['pcr']:.2f} ({pcr['sentiment']})")
        reason_parts.append(trade_note)

        signals.append({
            "type":           "HOURLY_FNO",
            "underlying":     ticker,
            "name":           name,
            "direction":      direction,
            "option_type":    option_type,
            "strike":         strike,
            "expiry":         expiry.isoformat(),
            "days_to_expiry": int(T * 365),
            "moneyness":      moneyness,
            "entry_premium":  round(premium, 2),
            "sl_premium":     sl_prem,
            "target_premium": target_prem,
            "lot_size":       lot,
            "cost_1lot":      round(premium * lot, 0),
            "rr_ratio":       round((target_prem - premium) / (premium - sl_prem), 2) if premium > sl_prem else 0,
            "iv_regime":      iv_env["iv_regime"],
            "hv10":           iv_env["hv10"],
            "hv30":           iv_env["hv30"],
            "pcr":            pcr["pcr"],
            "pcr_sentiment":  pcr["sentiment"],
            "spot":           round(spot, 2),
            "vwap":           round(vwap, 2),
            "vwap_dev_pct":   round(vwap_dev, 2),
            "ema9":           round(ema9, 2),
            "ema21":          round(ema21, 2),
            "conviction":     conviction,
            "delta":          greeks["delta"],
            "theta":          greeks["theta"],
            "reason":         " | ".join(reason_parts),
            "generated_at":   _now_ist().isoformat(),
        })

        return signals

    def _analyse_stock(self, ticker: str) -> list[dict]:
        """Lighter analysis for individual F&O stocks — EMA + VWAP only."""
        df = self._fetch_hourly(ticker, days=5)
        if df.empty or len(df) < 10:
            return []

        close  = df["Close"]
        spot   = float(close.iloc[-1])
        ema9   = float(self._ema(close, 9).iloc[-1])
        ema21  = float(self._ema(close, 21).iloc[-1])
        vwap   = self._vwap(df)
        vwap_dev = (spot - vwap) / vwap * 100

        # Require strong trend alignment
        if spot > ema9 > ema21 and vwap_dev > 0.3:
            direction, option_type, conviction = "BUY",  "call", 68
        elif spot < ema9 < ema21 and vwap_dev < -0.3:
            direction, option_type, conviction = "SELL", "put",  68
        else:
            return []

        iv_env  = self._iv_environment(ticker)
        expiry  = get_expiry(ticker, monthly=True)
        T       = days_to_expiry(expiry)
        iv      = iv_env["hv30"]
        strike  = select_strike(spot, option_type, moneyness="ATM")
        premium = BlackScholes.price(spot, strike, T, RISK_FREE_RATE, iv, option_type)
        greeks  = BlackScholes.greeks(spot, strike, T, RISK_FREE_RATE, iv, option_type)
        lot     = get_lot_size(ticker)

        return [{
            "type":           "HOURLY_FNO",
            "underlying":     ticker,
            "name":           ticker.replace(".NS", ""),
            "direction":      direction,
            "option_type":    option_type,
            "strike":         strike,
            "expiry":         expiry.isoformat(),
            "days_to_expiry": int(T * 365),
            "moneyness":      "ATM",
            "entry_premium":  round(premium, 2),
            "sl_premium":     round(premium * 0.60, 2),
            "target_premium": round(premium * 1.80, 2),
            "lot_size":       lot,
            "cost_1lot":      round(premium * lot, 0),
            "rr_ratio":       2.0,
            "iv_regime":      iv_env["iv_regime"],
            "hv30":           iv_env["hv30"],
            "spot":           round(spot, 2),
            "vwap":           round(vwap, 2),
            "vwap_dev_pct":   round(vwap_dev, 2),
            "conviction":     conviction,
            "delta":          greeks["delta"],
            "theta":          greeks["theta"],
            "reason":         f"1h trend aligned | VWAP {vwap_dev:+.2f}% | IV {iv_env['iv_regime']}",
            "generated_at":   _now_ist().isoformat(),
        }]


def run_hourly_fno_signals() -> list:
    """
    Generate and persist hourly F&O signals.
    Called by the hourly scheduler in app.py.
    """
    try:
        strategy = HourlyFNOStrategy()
        signals  = strategy.generate()
        with open(FNO_HOURLY_SIGNALS_FILE, "w") as f:
            json.dump({"signals": signals, "generated_at": _now_ist().isoformat()}, f, indent=2)
        logger.info(f"[HourlyFNO] Generated {len(signals)} signals")
        return signals
    except Exception as exc:
        logger.error(f"[HourlyFNO] Error: {exc}")
        return []


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
        # SpreadStrategy and IronCondorStrategy DISABLED:
        # Both write (sell) options which require margin accounts and create
        # theoretically unlimited loss exposure — not appropriate for paper
        # trading without real broker margin controls.
        # Only LONG options (buying calls/puts) are permitted.
        condors  = []   # self.iron_condor.run(self.portfolio)
        spreads  = []   # self.spreads.run(equity_signals, self.portfolio)
        directs  = self.directional.run(equity_signals, self.portfolio)
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
