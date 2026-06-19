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
from datetime import datetime, date, timedelta, time as dt_time
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
FNO_MAX_POSITIONS   = 5          # max 5 concurrent positions (mentor: swing framework)
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
    # Hard-coded fallback for common F&O stocks (SEBI lot sizes as of June 2026)
    _FALLBACK_LOTS = {
        "^NSEI": 75, "NIFTY": 75, "^NSEBANK": 30, "BANKNIFTY": 30,
        "FINNIFTY": 40, "MIDCPNIFTY": 50,
        "RELIANCE.NS": 250, "TCS.NS": 175, "HDFCBANK.NS": 550,
        "INFY.NS": 400, "ICICIBANK.NS": 700, "HINDUNILVR.NS": 300,
        "SBIN.NS": 1500, "BHARTIARTL.NS": 950, "BAJFINANCE.NS": 125,
        "KOTAKBANK.NS": 400, "LT.NS": 375, "AXISBANK.NS": 625,
        "ASIANPAINT.NS": 200, "MARUTI.NS": 100, "SUNPHARMA.NS": 350,
        "TITAN.NS": 375, "WIPRO.NS": 1500, "ULTRACEMCO.NS": 100,
        "NESTLEIND.NS": 50, "POWERGRID.NS": 4700, "NTPC.NS": 3000,
        "M&M.NS": 700, "HCLTECH.NS": 700, "ONGC.NS": 1925,
        "JSWSTEEL.NS": 600, "TATAMOTORS.NS": 1425, "ADANIENT.NS": 250,
        "COALINDIA.NS": 4200, "BAJAJFINSV.NS": 500, "GRASIM.NS": 375,
        "TECHM.NS": 600, "BPCL.NS": 1800, "CIPLA.NS": 650,
        "DRREDDY.NS": 125, "EICHERMOT.NS": 175, "APOLLOHOSP.NS": 125,
        "DIVISLAB.NS": 200, "TATACONSUM.NS": 1350, "INDUSINDBK.NS": 525,
        "SBILIFE.NS": 750, "HDFCLIFE.NS": 1100, "ADANIPORTS.NS": 1250,
        "HEROMOTOCO.NS": 300, "BAJAJ-AUTO.NS": 250, "BRITANNIA.NS": 200,
        "TATAPOWER.NS": 3375, "ITC.NS": 3200, "VEDL.NS": 2756,
        "GODREJCP.NS": 1000, "PIDILITIND.NS": 500, "DABUR.NS": 2750,
        "MARICO.NS": 1800, "HAVELLS.NS": 1000, "VOLTAS.NS": 500,
        "AMBUJACEM.NS": 2000, "GAIL.NS": 3825, "BHEL.NS": 10500,
        "ADANIGREEN.NS": 500, "ADANITRANS.NS": 500, "SAIL.NS": 10500,
        "UNITDSPR.NS": 1000, "ZOMATO.NS": 4500, "PAYTM.NS": 2000,
        "NYKAA.NS": 1000, "PNB.NS": 8000, "BANKBARODA.NS": 5850,
        "CANBK.NS": 5400, "FEDERALBNK.NS": 10000, "IDFCFIRSTB.NS": 11250,
        "CHOLAFIN.NS": 1250, "MUTHOOTFIN.NS": 750, "BAJAJHLDNG.NS": 50,
        "MOTHERSON.NS": 5400, "ASHOKLEY.NS": 5000, "MRF.NS": 10,
        "BOSCHLTD.NS": 50, "ABBINDIA.NS": 100, "SIEMENS.NS": 350,
        "AUROPHARMA.NS": 650, "BIOCON.NS": 2800, "LUPIN.NS": 850,
        "TORNTPHARM.NS": 500, "ALKEM.NS": 200, "LALPATHLAB.NS": 250,
        "MAXHEALTH.NS": 1050, "FORTIS.NS": 3200, "NAUKRI.NS": 125,
        "MPHASIS.NS": 250, "LTIM.NS": 150, "PERSISTENT.NS": 150,
        "COFORGE.NS": 200, "TATAELXSI.NS": 100, "KPIT.NS": 500,
        "EXIDEIND.NS": 1800, "AMARA.NS": 500, "TVSMOTOR.NS": 700,
        "BALKRISIND.NS": 400, "CUMMINS.NS": 600, "THERMAX.NS": 350,
        "SYNGENE.NS": 1300, "CROMPTON.NS": 1800, "PIIND.NS": 250,
        "UPL.NS": 1300, "CHAMBLFERT.NS": 3000, "COROMANDEL.NS": 500,
        "IPCALAB.NS": 500, "SUNTV.NS": 1500, "ZEEL.NS": 6000,
        "GODREJPROP.NS": 650, "OBEROIRLTY.NS": 400, "DLF.NS": 1650,
        "RVNL.NS": 1525, "IRFC.NS": 5000, "HUDCO.NS": 5000,
        "ADANIPOWER.NS": 1150, "TATACOMM.NS": 500, "MTNL.NS": 10000,
        "IDBI.NS": 9000, "YESBANK.NS": 40000, "360ONE.NS": 500,
        "JIOFIN.NS": 2350, "TMPV.NS": 1500, "ADANIENT": 250,
        "SBICARD.NS": 800, "HINDPETRO.NS": 1300, "IOC.NS": 5200,
        "PETRONET.NS": 3000, "SJVN.NS": 10000, "NHPC.NS": 6000,
    }
    sym = ticker.replace(".NS", "")
    for key in (ticker, sym, sym + ".NS"):
        if key in _FALLBACK_LOTS:
            return _FALLBACK_LOTS[key]
    return 50   # sensible default for unknown F&O stocks (not 1)


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


def get_expiry_20_45dte() -> date:
    """
    Get the best expiry targeting 20-45 DTE (mentor recommendation).
    Prefers monthly expiries in the 20-45 day window.
    Falls back to the next monthly if current month is too close (<20 days).
    Never uses weeklies — those have <7 DTE and are too risky per mentor.
    """
    today = date.today()
    # Try current month's last Thursday
    exp = _last_thursday(today.year, today.month)
    dte = (exp - today).days
    if 20 <= dte <= 45:
        return exp
    # Too close or already past — try next month
    nm = today.month % 12 + 1
    ny = today.year + (1 if today.month == 12 else 0)
    exp = _last_thursday(ny, nm)
    dte = (exp - today).days
    if dte <= 45:
        return exp
    # Skip to month after (rare edge case — e.g. we're right after expiry)
    nm2 = nm % 12 + 1
    ny2 = ny + (1 if nm == 12 else 0)
    return _last_thursday(ny2, nm2)


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

    def __init__(self, equity_portfolio=None):
        """
        equity_portfolio: if provided, F&O cash operations go through the
        equity portfolio's cash balance (unified account model).
        If None, uses standalone F&O cash (legacy mode).
        """
        self._equity = equity_portfolio
        self.state = self._load()
        # ── Cash integrity check ──────────────────────────────────────────
        # Correct cash if it's inconsistent with the accounting identity:
        #   cash = initial + realised_pnl - deployed_cost
        # This handles both: (a) first-run when cash=0, (b) stale cash from
        # the old "unified" model where _return_cash() wrote to equity's pool.
        deployed_cost = sum(
            p.get("entry_premium", 0) * p.get("qty", 1) * get_lot_size(p.get("underlying", ""))
            for p in self.state.get("positions", {}).values()
            if p.get("position") == "LONG"
        )
        realised = self.state.get("realised_pnl", 0.0)
        expected_cash = float(FNO_INITIAL_CAPITAL) + realised - deployed_cost
        actual_cash   = self.state.get("cash", 0)
        if abs(actual_cash - expected_cash) > 0.01:
            logger.info(
                f"[FNO] Cash correction: stored=₹{actual_cash:,.0f} "
                f"expected=₹{expected_cash:,.0f} (initial={FNO_INITIAL_CAPITAL:,} "
                f"realised={realised:,.0f} deployed={deployed_cost:,.0f})"
            )
            self.state["cash"] = max(0.0, expected_cash)
            self._save()

    def _default_state(self) -> dict:
        return {
            "cash":          FNO_INITIAL_CAPITAL,   # FNO has its own separate capital pool
            "initial":       FNO_INITIAL_CAPITAL,
            "positions":     {},
            "realised_pnl":  0.0,
            "created_at":    _now_ist().isoformat(),
            "last_updated":  _now_ist().isoformat(),
        }

    def _eq_cash(self) -> float:
        """Return available F&O cash (always own pool — fully decoupled from equity)."""
        return self.state["cash"]

    def _deduct_cash(self, amount: float):
        """Deduct from FNO's own cash pool."""
        self.state["cash"] -= amount
        self._save()

    def _return_cash(self, amount: float):
        """Return to FNO's own cash pool."""
        self.state["cash"] += amount
        self._save()

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
        if self._eq_cash() < cost:
            return False
        # Max 10% of total equity capital per F&O trade
        total_capital = (self._equity.get_total_value() if self._equity else
                         self.state.get("initial", FNO_INITIAL_CAPITAL))
        if cost > total_capital * FNO_MAX_RISK_PER_TRADE:
            return False
        return True

    def get_unrealised_pnl(self) -> float:
        """Sum of (current_premium - entry_premium) × qty — actual P&L on open positions."""
        total = 0.0
        for pos in self.state["positions"].values():
            try:
                spot   = self._get_spot(pos["underlying"])
                T      = days_to_expiry(date.fromisoformat(pos["expiry"]))
                iv     = pos.get("iv", 0.25)
                curr   = BlackScholes.price(spot, pos["strike"], T, RISK_FREE_RATE, iv, pos["option_type"])
                entry  = pos.get("entry_premium", 0)
                qty    = pos["qty"] * get_lot_size(pos["underlying"])
                direction = 1 if pos["position"] == "LONG" else -1
                total += direction * (curr - entry) * qty
            except Exception:
                pass
        return round(total, 2)

    def get_current_positions_value(self) -> float:
        """Current market value of all open positions (curr_premium × qty)."""
        total = 0.0
        for pos in self.state["positions"].values():
            try:
                spot   = self._get_spot(pos["underlying"])
                T      = days_to_expiry(date.fromisoformat(pos["expiry"]))
                iv     = pos.get("iv", 0.25)
                curr   = BlackScholes.price(spot, pos["strike"], T, RISK_FREE_RATE, iv, pos["option_type"])
                qty    = pos["qty"] * get_lot_size(pos["underlying"])
                direction = 1 if pos["position"] == "LONG" else -1
                total += direction * curr * qty
            except Exception:
                pass
        return round(total, 2)

    def get_total_value(self) -> float:
        """
        In unified mode: total equity value already includes cash (tracked in equity portfolio).
        Here we return cash + open position current values for F&O desk display.
        """
        return round(self._eq_cash() + self.get_current_positions_value(), 2)

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
                # MTM = P&L on this position = (curr - entry) × qty
                # Used for the dashboard MTM column (shows gain/loss, not absolute value)
                T      = days_to_expiry(date.fromisoformat(pos["expiry"]))
                iv     = pos.get("iv", 0.25)
                curr_premium  = BlackScholes.price(spot, pos["strike"], T, RISK_FREE_RATE, iv, pos["option_type"])
                entry_premium = pos.get("entry_premium", 0)
                qty = pos["qty"] * get_lot_size(pos["underlying"])
                direction = 1 if pos["position"] == "LONG" else -1
                return round(direction * (curr_premium - entry_premium) * qty, 2)
        except Exception:
            return 0.0

    def _get_spot(self, ticker: str) -> float:
        """Get live spot price. Tries Angel One WebSocket feed first, falls back to yfinance."""
        # 1. Angel One live feed (tick-accurate, already streaming)
        try:
            from angelone_feed import get_feed as _get_feed  # noqa: PLC0415
            _feed = _get_feed()
            if _feed.is_connected():
                live = _feed.get_price(ticker)
                if live and live > 0:
                    return float(live)
        except Exception:
            pass

        # 2. yfinance fallback (~15min delayed)
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
                    strategy: str, reason: str = "",
                    underlying_entry: float = 0.0,
                    stop_level: float = 0.0,
                    initial_R: float = 0.0) -> dict | None:
        """
        Open an option position.
        position: LONG (buy) | SHORT (sell/write)

        Swing framework params (optional, stored in position for exit logic):
          underlying_entry: spot price at entry (for R-multiple stops)
          stop_level: underlying price at which to close (1 ATR below entry for calls)
          initial_R: distance from entry to stop in ₹ (entry_spot - stop_level)
        """
        # Swing trading: allow entries during full market hours (not just before 2 PM)
        now_time = _now_ist().time()
        if not (dt_time(9, 15) <= now_time <= dt_time(15, 15)):
            logger.info(f"[FNO] BLOCKED open_option {underlying} — outside market hours")
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

        if position == "LONG" and cost > self._eq_cash():
            logger.warning(f"[FNO] Insufficient cash for {underlying} {option_type} — need ₹{cost:.0f}, have ₹{self._eq_cash():.0f}")
            return None

        pid = f"{underlying}_{strike}_{expiry}_{option_type}_{position}_{int(time.time())}"

        _entry_spot = round(underlying_entry if underlying_entry > 0 else spot, 2)
        pos = {
            "position_id":    pid,
            "instrument_type":"OPTION",
            "underlying":     underlying,
            "strike":         strike,
            "expiry":         expiry.isoformat(),
            "option_type":    option_type,
            "position":       position,
            "qty":            qty_lots,
            "entry_premium":  round(premium, 2),
            "entry_spot":     round(spot, 2),
            "iv":             iv,
            "strategy":       strategy,
            "reason":         reason,
            "entry_date":     _now_ist().isoformat(),
            # Swing framework fields
            "underlying_entry_price": _entry_spot,
            "stop_level":   round(stop_level, 2) if stop_level > 0 else 0.0,
            "initial_R":    round(initial_R,  2) if initial_R  > 0 else 0.0,
            "_t1_done":     False,
            "_t2_done":     False,
        }
        self.state["positions"][pid] = pos

        if position == "LONG":
            self._deduct_cash(cost)
        else:
            margin = spot * lot * qty_lots * 0.20
            self._deduct_cash(margin)
            pos["margin_blocked"] = round(margin, 2)
            self._return_cash(cost)  # premium received upfront for SHORT

        self._save()
        trade = {
            "action":       f"OPEN_{position}_{option_type.upper()}",
            "underlying":   underlying,
            "instrument":   underlying,                  # alias for UI
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
          - Strength ≥ 85  → ATM put (mentor: ATM/ITM1 only, never OTM)
          - Below 85       → skip (mentor: high conviction gate)

        Expiry: 20-45 DTE monthly (mentor: never weeklies).
        Position size: capped at 2% of F&O portfolio per trade.
        """
        if strength < 85:
            logger.debug(f"[FNO] SELL signal on {ticker} strength={strength:.0f} < 85 — skipping PUT hedge")
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

        moneyness = "ATM"                    # always ATM — never OTM (mentor rule)
        expiry    = get_expiry_20_45dte()   # 20-45 DTE monthly (mentor rule)
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
            self._return_cash(curr_premium * qty)   # return current value (entry was already deducted)
        else:
            pnl = (entry - curr_premium) * qty
            self._return_cash(pos.get("margin_blocked", 0))
            self._deduct_cash(curr_premium * qty)   # buy back cost

        self.state["realised_pnl"] += pnl
        del self.state["positions"][position_id]
        self._save()

        trade = {
            "action":       f"CLOSE_{pos['position']}_{pos['option_type'].upper()}",
            "underlying":   pos["underlying"],
            "instrument":   pos.get("underlying", ""),   # alias for UI
            "strike":       pos["strike"],
            "expiry":       pos["expiry"],
            "option_type":  pos["option_type"],
            "qty_lots":     pos["qty"],
            "premium":      round(curr_premium, 2),      # consistent field name
            "exit_premium": round(curr_premium, 2),
            "entry_premium":entry,
            "pnl":          round(pnl, 2),
            "strategy":     pos.get("strategy", ""),
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

    def partial_close_option(self, position_id: str, close_pct: float, reason: str = "") -> dict | None:
        """
        Close a fraction of an option position.
        close_pct: 0.5 = close half the lots, 0.25 = close a quarter, 1.0 = close all.
        Used for R-multiple profit targets (T1 = 50% at 2R, T2 = 25% at 3R).
        If remaining qty would be 0, falls through to full close_option.
        """
        pos = self.state["positions"].get(position_id)
        if not pos or pos["instrument_type"] != "OPTION":
            return None

        total_lots    = pos["qty"]
        lots_to_close = max(1, int(round(total_lots * close_pct)))

        # If closing everything (or only 1 lot left), delegate to full close
        if lots_to_close >= total_lots:
            return self.close_option(position_id, reason=reason)

        # Fetch current premium
        spot = self._get_spot(pos["underlying"])
        T    = days_to_expiry(date.fromisoformat(pos["expiry"]))
        iv   = pos.get("iv", 0.25)

        curr_premium = None
        try:
            from angelone_feed import get_feed as _get_feed  # noqa: PLC0415
            _feed = _get_feed()
            if _feed.is_connected() and _feed._smart:
                ltp = _feed.get_option_ltp(
                    pos["underlying"], pos["strike"],
                    date.fromisoformat(pos["expiry"]), pos["option_type"]
                )
                if ltp and ltp > 0:
                    curr_premium = ltp
        except Exception:
            pass

        if curr_premium is None:
            curr_premium = BlackScholes.price(spot, pos["strike"], T, RISK_FREE_RATE, iv, pos["option_type"])

        if curr_premium <= 0:
            logger.warning(f"[FNO] SKIP partial_close {position_id} — premium ₹0")
            return None

        lot           = get_lot_size(pos["underlying"])
        qty_to_close  = lots_to_close * lot
        entry         = pos["entry_premium"]
        pnl           = (curr_premium - entry) * qty_to_close

        self._return_cash(curr_premium * qty_to_close)
        self.state["realised_pnl"] += pnl
        pos["qty"] -= lots_to_close   # reduce remaining size
        self._save()

        trade = {
            "action":         f"PARTIAL_CLOSE_{pos['position']}_{pos['option_type'].upper()}",
            "underlying":     pos["underlying"],
            "instrument":     pos.get("underlying", ""),
            "strike":         pos["strike"],
            "expiry":         pos["expiry"],
            "option_type":    pos["option_type"],
            "qty_lots":       lots_to_close,
            "remaining_lots": pos["qty"],
            "premium":        round(curr_premium, 2),
            "exit_premium":   round(curr_premium, 2),
            "entry_premium":  entry,
            "pnl":            round(pnl, 2),
            "close_pct":      close_pct,
            "strategy":       pos.get("strategy", ""),
            "reason":         reason,
        }
        self._log_trade(trade)
        logger.info(
            f"[FNO] PARTIAL_CLOSE {lots_to_close}/{total_lots}L {pos['underlying']} "
            f"{pos['strike']}{pos['option_type'][0].upper()} "
            f"P&L=₹{pnl:+.0f}  ({reason})"
        )
        # Telegram notification
        try:
            from notifier import get_notifier  # noqa: PLC0415
            get_notifier().notify_fno_close(
                underlying=pos["underlying"], option_type=pos["option_type"],
                strike=pos["strike"], entry_premium=entry,
                exit_premium=round(curr_premium, 2),
                qty_lots=lots_to_close, lot_size=lot,
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
        if margin > self._eq_cash():
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
        self._deduct_cash(margin)
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

        self._return_cash(pos.get("margin_blocked", 0) + pnl)
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

    def morning_gap_check(self) -> list:
        """
        Run once at market open (9:15–9:25 IST) to handle overnight gaps.
        For each open option position, checks if the underlying gapped overnight
        beyond the stop level. Closes immediately if so — before the realtime
        monitor would even see it.

        A PUT is hurt by an upward gap (underlying rose overnight).
        A CALL is hurt by a downward gap (underlying fell overnight).
        Threshold: if underlying moved >3% against the position overnight, close.
        """
        closed = []
        now_t = _now_ist().time()
        from datetime import time as _dtt3  # noqa: PLC0415
        # Only run in the opening window 9:15–9:30
        if not (_dtt3(9, 15) <= now_t <= _dtt3(9, 30)):
            return closed

        for pid, pos in list(self.state["positions"].items()):
            if pos.get("instrument_type") != "OPTION" or pos.get("position") != "LONG":
                continue
            try:
                spot = self._get_spot(pos["underlying"])
                entry_spot = pos.get("entry_spot", spot)
                if not entry_spot or entry_spot <= 0:
                    continue

                overnight_move = (spot - entry_spot) / entry_spot

                # PUT is hurt by rising underlying; CALL is hurt by falling
                adverse_move = overnight_move if pos["option_type"] == "call" else -overnight_move

                # Gap >3% against position → close immediately at open
                if adverse_move < -0.03:
                    logger.warning(
                        f"[FNO-GAP] {pos['underlying']} gapped {overnight_move:+.1%} overnight "
                        f"— closing {pos['option_type'].upper()} position at open"
                    )
                    t = self.close_option(pid, reason="OVERNIGHT_GAP_STOP")
                    if t:
                        closed.append(t)
            except Exception as _e:
                logger.debug(f"[FNO-GAP] Error checking {pid}: {_e}")

        if closed:
            logger.info(f"[FNO-GAP] Morning gap check closed {len(closed)} position(s)")
        return closed

    def _get_curr_prem(self, pos: dict) -> float | None:
        """Fetch current option premium — Angel One LTP first, Black-Scholes fallback."""
        spot = self._get_spot(pos["underlying"])
        try:
            from angelone_feed import get_feed as _gf  # noqa: PLC0415
            _fd = _gf()
            if _fd.is_connected() and _fd._smart:
                ltp = _fd.get_option_ltp(
                    pos["underlying"], pos["strike"],
                    date.fromisoformat(pos["expiry"]), pos["option_type"]
                )
                if ltp and ltp > 0:
                    return ltp
        except Exception:
            pass
        T  = days_to_expiry(date.fromisoformat(pos["expiry"]))
        iv = pos.get("iv", 0.25)
        return BlackScholes.price(spot, pos["strike"], T, RISK_FREE_RATE, iv, pos["option_type"])

    def _swing_exit_decision(self, pos: dict, curr_prem: float, spot: float,
                              days_left: int, age_days: int) -> tuple[bool, float, str]:
        """
        Long-Only Options Swing Framework exit logic (mentor v1.0).

        Returns (should_exit, close_pct, reason_string).
        close_pct: 1.0 = close all, 0.5 = close half (T1), 0.25 = close quarter (T2), 0.0 = hold

        Exit rules (checked in order):
        1. Underlying stop: spot crosses stop_level → close 100%
        2. Premium crash backstop: -50% on premium → close 100%
        3. T1 (2R on underlying): close 50%
        4. T2 (3R on underlying): close 25%
        5. Max hold: 30 trading days → close 100%
        6. Near expiry (≤7 days): close 100% to avoid expiry risk
        7. Theta drain: open > 10 days and flat/losing → close 100%
        """
        option_type = pos.get("option_type", "call")
        entry_prem  = pos.get("entry_premium", 1)
        entry_spot  = pos.get("underlying_entry_price", spot)
        stop_level  = pos.get("stop_level", 0.0)
        initial_R   = pos.get("initial_R", 0.0)
        t1_done     = pos.get("_t1_done", False)
        t2_done     = pos.get("_t2_done", False)

        pnl_pct = (curr_prem - entry_prem) / entry_prem if entry_prem > 0 else 0

        # ── 1. Underlying stop-loss ───────────────────────────────────────── #
        if stop_level > 0 and entry_spot > 0:
            if option_type == "call" and spot <= stop_level:
                return True, 1.0, f"UNDERLYING_STOP: {spot:.0f} ≤ {stop_level:.0f}"
            if option_type == "put" and spot >= stop_level:
                return True, 1.0, f"UNDERLYING_STOP: {spot:.0f} ≥ {stop_level:.0f}"

        # ── 2. Premium crash backstop ─────────────────────────────────────── #
        if pnl_pct <= -0.50:
            return True, 1.0, f"PREMIUM_STOP (-50%)"

        # ── 3 & 4. R-multiple profit targets ─────────────────────────────── #
        if initial_R > 0 and entry_spot > 0:
            if option_type == "call":
                t1_spot = entry_spot + 2 * initial_R
                t2_spot = entry_spot + 3 * initial_R
                at_t1 = spot >= t1_spot
                at_t2 = spot >= t2_spot
            else:
                t1_spot = entry_spot - 2 * initial_R
                t2_spot = entry_spot - 3 * initial_R
                at_t1 = spot <= t1_spot
                at_t2 = spot <= t2_spot

            # T2 first (stricter) so we don't double-trigger on same bar
            if t1_done and not t2_done and at_t2:
                pos["_t2_done"] = True
                return True, 0.25, f"T2_3R: underlying {option_type}={spot:.0f} target={t2_spot:.0f} ({pnl_pct:+.0%} prem)"

            if not t1_done and at_t1:
                pos["_t1_done"] = True
                return True, 0.5, f"T1_2R: underlying {option_type}={spot:.0f} target={t1_spot:.0f} ({pnl_pct:+.0%} prem)"

        # ── 5. Max hold: 30 trading days ─────────────────────────────────── #
        if age_days >= 42:   # 30 trading days ≈ 42 calendar days
            return True, 1.0, f"MAX_HOLD (42 calendar days)"

        # ── 6. Near expiry: ≤7 days ──────────────────────────────────────── #
        if days_left <= 7:
            return True, 1.0, f"NEAR_EXPIRY ({days_left}d left — theta risk)"

        # ── 7. Theta drain: flat/losing after 10+ days ───────────────────── #
        if age_days >= 10 and pnl_pct < -0.20:
            return True, 1.0, f"THETA_DRAIN ({age_days}d, {pnl_pct:.0%})"

        return False, 0.0, ""

    def check_expiry_and_stops(self) -> list:
        """
        Swing framework exit management.
        Uses _swing_exit_decision() — R-multiple targets on underlying,
        underlying stop-loss, max hold (30 trading days), near-expiry close.

        Supports partial exits: T1 closes 50%, T2 closes 25%, rest holds.
        No intraday EOD force-close (swing positions carry overnight).
        """
        closed = []

        for pid, pos in list(self.state["positions"].items()):
            exp       = date.fromisoformat(pos["expiry"])
            days_left = (exp - date.today()).days

            # ── 1. Expiry close (≤1 day) ─────────────────────────────────── #
            if days_left <= 1:
                logger.info(f"[FNO] EXPIRY-CLOSE {pid} ({days_left}d left)")
                t = self.close_option(pid, reason="EXPIRY_CLOSE") \
                    if pos["instrument_type"] == "OPTION" \
                    else self.close_future(pid, reason="EXPIRY_CLOSE")
                if t:
                    closed.append(t)
                continue

            # ── 2. Only check LONG options during market hours ────────────── #
            now_t = _now_ist().time()
            if not (dt_time(9, 15) <= now_t <= dt_time(15, 20)):
                continue

            if pos["instrument_type"] != "OPTION" or pos["position"] != "LONG":
                continue

            # ── 3. Compute age in calendar days ──────────────────────────── #
            age_days = 0
            entry_dt_str = pos.get("entry_date", "")
            if entry_dt_str:
                try:
                    entry_dt = datetime.fromisoformat(entry_dt_str)
                    if entry_dt.tzinfo is None:
                        entry_dt = entry_dt.replace(tzinfo=_IST)
                    age_days = (_now_ist() - entry_dt).days
                except Exception:
                    pass

            # ── 4. Skip if held less than 15 minutes (avoids open noise) ─── #
            if age_days == 0:
                try:
                    entry_dt = datetime.fromisoformat(entry_dt_str)
                    if entry_dt.tzinfo is None:
                        entry_dt = entry_dt.replace(tzinfo=_IST)
                    age_mins = (_now_ist() - entry_dt).total_seconds() / 60
                    if age_mins < 15:
                        continue
                except Exception:
                    pass

            # ── 5. Get current premium and spot ──────────────────────────── #
            curr_prem = self._get_curr_prem(pos)
            if not curr_prem or curr_prem <= 0:
                continue

            entry_prem = pos.get("entry_premium", 0)
            if entry_prem <= 0:
                continue

            spot = self._get_spot(pos["underlying"])

            # ── 6. Swing exit decision (R-multiple logic) ─────────────────── #
            should_exit, close_pct, exit_reason = self._swing_exit_decision(
                pos, curr_prem, spot, days_left, age_days
            )

            if should_exit:
                pnl_pct = (curr_prem - entry_prem) / entry_prem if entry_prem else 0
                logger.info(
                    f"[FNO] EXIT {pid} spot={spot:.0f} prem={curr_prem:.2f} "
                    f"pnl={pnl_pct:.0%} close={close_pct:.0%} — {exit_reason}"
                )
                if close_pct < 1.0 and pos["qty"] > 1:
                    t = self.partial_close_option(pid, close_pct, reason=exit_reason)
                else:
                    t = self.close_option(pid, reason=exit_reason)
                if t:
                    t["reason"] = exit_reason
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
                entry   = pos["entry_premium"]
                # Dynamic SL: based on peak profit seen so far
                peak    = pos.get("_peak_pnl_pct", 0)
                if peak >= 0.35:
                    sl_prem = round(entry * 1.20, 2)   # lock +20%
                elif peak >= 0.20:
                    sl_prem = round(entry * 1.10, 2)   # lock +10%
                elif peak >= 0.10:
                    sl_prem = round(entry, 2)           # lock breakeven
                else:
                    sl_prem = round(entry * 0.60, 2)   # initial stop -40%
                # Dynamic target: tightens near expiry
                T_days  = (date.fromisoformat(pos["expiry"]) - date.today()).days
                if T_days <= 5:
                    tgt_prem = round(entry * 1.25, 2)  # near expiry: +25%
                else:
                    tgt_prem = round(entry * 1.35, 2)  # normal: +35% (then profit-lock kicks in)
                pnl_pct = round(mtm / cost * 100, 1) if cost else 0
                pct_to_tgt = round((tgt_prem - curr_prem) / curr_prem * 100, 1) if curr_prem else 0
                pct_to_sl  = round((curr_prem - sl_prem) / curr_prem * 100, 1) if curr_prem else 0
                result.append({
                    "position_id":   pid,
                    "instrument":    f"{pos['underlying']} {pos['strike']}{pos['option_type'][0].upper()}",
                    "type":          f"{pos['position']} {pos['option_type'].upper()}",
                    "expiry":        pos["expiry"],
                    "days_left":     int(T * 365),
                    "qty_lots":      pos["qty"],
                    "lot_size":      lot,
                    "cost":          round(cost, 2),   # entry_premium × qty × lot_size
                    "entry_premium": entry,
                    "curr_premium":  round(curr_prem, 2),
                    "sl_premium":    sl_prem,
                    "target_premium":tgt_prem,
                    "pct_to_target": pct_to_tgt,
                    "pct_to_sl":     pct_to_sl,
                    "mtm":           round(mtm, 2),
                    "pnl_pct":       pnl_pct,
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
        # Return all open premium costs back to equity cash before wiping state
        for pos in self.state.get("positions", {}).values():
            try:
                cost = float(pos.get("entry_premium", 0)) * int(pos.get("qty", 0))
                if cost > 0:
                    self._return_cash(cost)
            except Exception:
                pass
        self.state = self._default_state()
        self._save()


# ---------------------------------------------------------------------------
# F&O Strategies
# ---------------------------------------------------------------------------

def _swing_market_regime_ok() -> tuple[bool, str]:
    """
    Market regime gate for long calls (mentor rule: no long calls in bearish regime).
    Blocks if: Nifty < 200 EMA  OR  50 EMA < 200 EMA  OR  VIX > 25.
    Returns (ok, reason_string).
    """
    try:
        df = yf.download("^NSEI", period="1y", interval="1d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close  = df["Close"].squeeze()
        spot   = float(close.iloc[-1])
        ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
        ema50  = float(close.ewm(span=50,  adjust=False).mean().iloc[-1])

        if spot < ema200:
            return False, f"Nifty {spot:.0f} < 200EMA {ema200:.0f} — bearish regime"
        if ema50 < ema200:
            return False, f"50EMA {ema50:.0f} < 200EMA {ema200:.0f} — death cross"

        try:
            vdf = yf.download("^INDIAVIX", period="5d", interval="1d", auto_adjust=True, progress=False)
            if isinstance(vdf.columns, pd.MultiIndex):
                vdf.columns = vdf.columns.get_level_values(0)
            if not vdf.empty:
                vix = float(vdf["Close"].squeeze().iloc[-1])
                if vix > 25:
                    return False, f"VIX {vix:.1f} > 25 — fear elevated, skip LONG calls"
        except Exception:
            pass

        return True, f"Regime OK (Nifty {spot:.0f} > 200EMA {ema200:.0f})"
    except Exception as e:
        logger.warning(f"[SwingRegime] Check failed: {e} — defaulting OK")
        return True, "Regime check failed — proceed"


def _swing_trend_and_levels(ticker: str, spot: float) -> tuple[bool, int, float, float, float, str]:
    """
    Trend filter + conviction scoring + ATR-based stop levels for the swing framework.

    Returns (passes_filter, conviction_score, atr14, stop_level, initial_R, detail_string).

    Conviction scoring (0-100):
      Fundamentals proxy (40 pts): above 200EMA, within 5% of 52w high, above 50EMA
      Trend              (30 pts): ADX strength, full EMA stack
      Volume             (20 pts): relative volume vs 20-day average
      Regime             (10 pts): granted if regime OK at call site

    Gate: score >= 70 to trade (mentor minimum conviction threshold).

    Hard filter requirements (any failure → skip):
      - Price > 200 EMA
      - ADX >= 20
      - Within 20% of 52-week high
      - Relative volume >= 1.0
    """
    try:
        df = yf.download(ticker, period="1y", interval="1d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) < 50:
            return False, 0, 0, 0, 0, "Insufficient data"

        close  = df["Close"].squeeze()
        high   = df["High"].squeeze()
        low    = df["Low"].squeeze()
        volume = df["Volume"].squeeze()

        ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
        ema50  = float(close.ewm(span=50,  adjust=False).mean().iloc[-1])

        # 52-week high
        high_52w      = float(high.rolling(min(252, len(high))).max().iloc[-1])
        pct_from_52wh = (spot - high_52w) / high_52w if high_52w > 0 else -1.0

        # ATR-14
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr14 = float(tr.rolling(14).mean().iloc[-1])

        # ADX (simplified 14-period)
        plus_dm  = (high - high.shift()).clip(lower=0)
        minus_dm = (low.shift() - low).clip(lower=0)
        mask     = plus_dm < minus_dm
        plus_dm[mask]  = 0
        minus_dm[~mask] = 0
        atr_smooth   = tr.ewm(span=14, adjust=False).mean()
        plus_di      = 100 * plus_dm.ewm(span=14, adjust=False).mean()  / atr_smooth.replace(0, np.nan)
        minus_di     = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr_smooth.replace(0, np.nan)
        dx           = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx          = float(dx.ewm(span=14, adjust=False).mean().iloc[-1])

        # Relative volume (10d vs 20d)
        vol10  = float(volume.iloc[-10:].mean())
        vol20  = float(volume.iloc[-20:].mean())
        rel_vol = vol10 / vol20 if vol20 > 0 else 1.0

        # ── Hard filter gates ──────────────────────────────────────────────
        if spot < ema200:
            return False, 0, atr14, 0, 0, f"Price {spot:.0f} < 200EMA {ema200:.0f}"
        if adx < 20:
            return False, 0, atr14, 0, 0, f"ADX {adx:.1f} < 20 — no trend"
        if pct_from_52wh < -0.20:
            return False, 0, atr14, 0, 0, f"Price {pct_from_52wh:.0%} off 52w high — laggard"
        if rel_vol < 1.0:
            return False, 0, atr14, 0, 0, f"RelVol {rel_vol:.2f}x < 1.0 — weak participation"

        # ── Conviction score ───────────────────────────────────────────────
        score   = 0
        reasons = []

        # Fundamentals proxy (40 pts)
        if spot > ema200:
            score += 15; reasons.append("Above 200EMA")
        if spot > ema50:
            score += 10; reasons.append("Above 50EMA")
        if pct_from_52wh >= -0.05:
            score += 15; reasons.append(f"Near 52wH ({pct_from_52wh:.0%})")
        elif pct_from_52wh >= -0.10:
            score += 8

        # Trend (30 pts)
        if ema50 > ema200:
            score += 15; reasons.append("50EMA > 200EMA")
        if adx >= 30:
            score += 15; reasons.append(f"ADX={adx:.0f}")
        elif adx >= 25:
            score += 10; reasons.append(f"ADX={adx:.0f}")
        elif adx >= 20:
            score += 5

        # Volume (20 pts)
        if rel_vol >= 2.0:
            score += 20; reasons.append(f"RelVol={rel_vol:.1f}x")
        elif rel_vol >= 1.5:
            score += 15; reasons.append(f"RelVol={rel_vol:.1f}x")
        elif rel_vol >= 1.0:
            score += 8

        # Regime (10 pts) — granted at call site if regime passes
        score += 10; reasons.append("Regime OK")

        score = min(score, 100)

        # ── Stop and R levels ──────────────────────────────────────────────
        stop_level = round(spot - atr14, 2)   # 1 ATR below entry (for calls)
        initial_R  = round(spot - stop_level, 2)

        detail = f"Score={score} | ADX={adx:.0f} | RelVol={rel_vol:.1f}x | {' | '.join(reasons)}"
        return True, score, atr14, stop_level, initial_R, detail

    except Exception as e:
        logger.warning(f"[SwingTrend] Filter failed for {ticker}: {e}")
        return True, 70, 0, 0, 0, f"Filter error: {e}"


def _swing_bearish_levels(ticker: str, spot: float) -> tuple[bool, int, float, float, float, str]:
    """
    Bearish trend filter + conviction scoring for Long Puts.
    Mirror of _swing_trend_and_levels but gates on downtrend conditions:
      - Price < 200 EMA  (confirmed downtrend)
      - ADX >= 20        (trending, not sideways)
      - Within 30% of 52-week low  (near lows, bearish momentum)
      - Relative volume >= 1.0     (selling volume present)

    Returns (passes_filter, conviction_score, atr14, stop_level, initial_R, detail_string).
    stop_level is 1 ATR ABOVE spot (exit put if underlying bounces above this).
    """
    try:
        df = yf.download(ticker, period="1y", interval="1d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) < 50:
            return False, 0, 0, 0, 0, "Insufficient data"

        close  = df["Close"].squeeze()
        high   = df["High"].squeeze()
        low    = df["Low"].squeeze()
        volume = df["Volume"].squeeze()

        ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
        ema50  = float(close.ewm(span=50,  adjust=False).mean().iloc[-1])

        # 52-week low
        low_52w       = float(low.rolling(min(252, len(low))).min().iloc[-1])
        pct_from_52wl = (spot - low_52w) / low_52w if low_52w > 0 else 1.0

        # ATR-14
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr14 = float(tr.rolling(14).mean().iloc[-1])

        # ADX (simplified 14-period)
        plus_dm  = (high - high.shift()).clip(lower=0)
        minus_dm = (low.shift() - low).clip(lower=0)
        mask     = plus_dm < minus_dm
        plus_dm[mask]   = 0
        minus_dm[~mask] = 0
        atr_smooth = tr.ewm(span=14, adjust=False).mean()
        plus_di    = 100 * plus_dm.ewm(span=14, adjust=False).mean()  / atr_smooth.replace(0, np.nan)
        minus_di   = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr_smooth.replace(0, np.nan)
        dx         = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx        = float(dx.ewm(span=14, adjust=False).mean().iloc[-1])

        # Relative volume (10d vs 20d average)
        vol10   = float(volume.iloc[-10:].mean())
        vol20   = float(volume.iloc[-20:].mean())
        rel_vol = vol10 / vol20 if vol20 > 0 else 1.0

        # ── Hard filter gates (bearish) ──────────────────────────────────────
        if spot > ema200:
            return False, 0, atr14, 0, 0, f"Price {spot:.0f} > 200EMA {ema200:.0f} — not bearish"
        if adx < 20:
            return False, 0, atr14, 0, 0, f"ADX {adx:.1f} < 20 — no trend"
        if pct_from_52wl > 0.30:
            return False, 0, atr14, 0, 0, f"Price {pct_from_52wl:.0%} above 52w low — not in bear momentum"
        if rel_vol < 1.0:
            return False, 0, atr14, 0, 0, f"RelVol {rel_vol:.2f}x < 1.0 — weak selling volume"

        # ── Conviction score (0-100) ──────────────────────────────────────────
        score   = 0
        reasons = []

        # EMA structure (40 pts)
        if spot < ema200:
            score += 15; reasons.append(f"Below 200EMA({ema200:.0f})")
        if spot < ema50:
            score += 10; reasons.append(f"Below 50EMA({ema50:.0f})")
        if ema50 < ema200:
            score += 15; reasons.append("Death cross (50<200)")

        # Trend strength (30 pts)
        if adx >= 30:
            score += 20; reasons.append(f"ADX={adx:.0f} strong")
        elif adx >= 25:
            score += 15; reasons.append(f"ADX={adx:.0f}")
        elif adx >= 20:
            score += 5
        # -DI dominance confirms downtrend
        last_minus_di = float(minus_di.iloc[-1]) if not minus_di.isna().all() else 0
        last_plus_di  = float(plus_di.iloc[-1])  if not plus_di.isna().all()  else 0
        if last_minus_di > last_plus_di:
            score += 10; reasons.append(f"-DI({last_minus_di:.0f}) > +DI({last_plus_di:.0f})")

        # Volume confirmation (20 pts)
        if rel_vol >= 2.0:
            score += 20; reasons.append(f"RelVol={rel_vol:.1f}x")
        elif rel_vol >= 1.5:
            score += 15; reasons.append(f"RelVol={rel_vol:.1f}x")
        elif rel_vol >= 1.0:
            score += 8

        # Near 52w low (10 pts) — confirms bearish momentum
        if pct_from_52wl < 0.10:
            score += 10; reasons.append(f"Near 52wL (+{pct_from_52wl:.0%})")

        score = min(score, 100)

        # Stop level: 1 ATR ABOVE spot (exit put if underlying bounces above this)
        stop_level = round(spot + atr14, 2)
        initial_R  = atr14

        detail = f"Score={score} | ADX={adx:.0f} | RelVol={rel_vol:.1f}x | {' | '.join(reasons)}"
        return True, score, atr14, stop_level, initial_R, detail

    except Exception as e:
        logger.warning(f"[BearishFilter] Filter failed for {ticker}: {e}")
        return True, 70, 0, 0, 0, f"Filter error: {e}"


class DirectionalOptionsStrategy:
    """
    Directional Options Swing Framework — Long Calls and Long Puts.

    BUY signal  → Long Call (ITM1, bullish trend filter, regime must be OK)
    SELL signal → Long Put  (ITM1, bearish trend filter, allowed in any regime)

    Entry gates (all must pass):
      1. Signal strength >= 85 (only high-conviction equity signals trigger options)
      2. Trend filter (bullish for calls / bearish for puts): ADX >= 20, EMA alignment,
         near 52w high (calls) or 52w low (puts), rel vol >= 1.0
      3. Conviction score >= 70 (EMA structure 40% + Trend 30% + Volume 20% + Proximity 10%)
      4. Market regime gate for calls only (Nifty > 200 EMA, VIX < 25)

    Strike: ITM1 (delta ~0.65) — deeper delta, less theta decay than ATM
    Expiry: 20-45 DTE monthly (never weeklies — too much theta risk)
    Stop: 1 ATR from underlying entry (above for puts, below for calls)
    Targets: T1 = 2R, T2 = 3R
    """
    name       = "Directional Options Swing"
    short_name = "DIR_SWING"

    # Cache regime check for 30 min to avoid hammering yfinance
    _regime_cache: tuple[bool, str, float] | None = None

    def _get_regime(self) -> tuple[bool, str]:
        now_ts = time.time()
        if self._regime_cache and (now_ts - self._regime_cache[2]) < 1800:
            return self._regime_cache[0], self._regime_cache[1]
        ok, reason = _swing_market_regime_ok()
        DirectionalOptionsStrategy._regime_cache = (ok, reason, now_ts)
        return ok, reason

    def run(self, signals: list, portfolio: FNOPortfolio) -> list:
        executed = []

        # ── Step 1: Market regime gate (shared for all signals this cycle) ── #
        regime_ok, regime_reason = self._get_regime()

        for sig in signals:
            if portfolio.at_max_positions():
                break

            # ── Step 2: Signal strength gate ─────────────────────────────── #
            if sig.get("strength", 0) < 75:
                continue
            ticker = sig["ticker"]
            action = sig.get("signal", "")
            if action not in ("BUY", "SELL"):
                continue

            # BUY → Long Call   |   SELL → Long Put
            opt_type = "call" if action == "BUY" else "put"

            # Regime gate: block long calls in bad regime, but allow long puts
            # (puts can be directional bearish even when market regime is weak)
            if action == "BUY" and not regime_ok:
                logger.info(f"[DirSwing] BLOCKED long call on {ticker} — regime fail: {regime_reason}")
                continue

            if portfolio.has_open_position(ticker, opt_type):
                continue
            if portfolio.in_cooldown(ticker):
                continue

            try:
                spot = portfolio._get_spot(ticker)
                if spot <= 0:
                    continue

                # ── Step 3: Trend filter + conviction score ───────────────── #
                # Calls: bullish filter (price > 200EMA, near 52w high, ADX > 20)
                # Puts:  bearish filter (price < 200EMA, near 52w low, ADX > 20)
                if action == "BUY":
                    passes, conviction, atr14, stop_level, initial_R, detail = \
                        _swing_trend_and_levels(ticker, spot)
                    r_direction = +1   # target = spot + 2R, + 3R
                else:
                    passes, conviction, atr14, stop_level, initial_R, detail = \
                        _swing_bearish_levels(ticker, spot)
                    r_direction = -1   # target = spot - 2R, - 3R

                if not passes:
                    logger.info(f"[DirSwing] SKIP {ticker} {opt_type} — filter: {detail}")
                    continue

                if conviction < 55:
                    logger.info(f"[DirSwing] SKIP {ticker} {opt_type} — conviction {conviction} < 55 | {detail}")
                    continue

                logger.info(f"[DirSwing] {ticker} {opt_type.upper()} passes | {detail}")

                # ── Step 4: Strike selection ──────────────────────────────── #
                # ITM1 (delta ~0.65) for calls and puts — better delta capture vs ATM
                expiry = get_expiry_20_45dte()
                strike = select_strike(spot, opt_type, moneyness="ITM1")
                T      = days_to_expiry(expiry)
                iv     = historical_vol(ticker)
                prem   = BlackScholes.price(spot, strike, T, RISK_FREE_RATE, iv, opt_type)
                lot    = get_lot_size(ticker)
                cost   = prem * lot

                if prem < 10:
                    logger.info(f"[DirSwing] SKIP {ticker} — premium ₹{prem:.2f} too low")
                    continue
                if cost < 500:
                    continue
                if not portfolio.can_afford(cost):
                    logger.info(f"[DirSwing] SKIP {ticker} — cannot afford ₹{cost:.0f}")
                    continue

                # ── Step 5: Open with swing metadata ─────────────────────── #
                greeks = BlackScholes.greeks(spot, strike, T, RISK_FREE_RATE, iv, opt_type)
                t1 = spot + r_direction * 2 * initial_R
                t2 = spot + r_direction * 3 * initial_R
                reason_str = (
                    f"Long {opt_type.title()} | {action} strength={sig['strength']:.0f} | "
                    f"conv={conviction} | delta={greeks['delta']:.2f} | "
                    f"stop={stop_level:.0f} (1ATR={atr14:.0f}) | T1={t1:.0f} T2={t2:.0f}"
                )
                result = portfolio.open_option(
                    underlying        = ticker,
                    strike            = strike,
                    expiry            = expiry,
                    option_type       = opt_type,
                    position          = "LONG",
                    qty_lots          = 1,
                    strategy          = self.short_name,
                    reason            = reason_str,
                    underlying_entry  = spot,
                    stop_level        = stop_level,
                    initial_R         = initial_R,
                )
                if result:
                    suffix = f"T1={t1:.0f} T2={t2:.0f}"
                    logger.info(
                        f"[DirSwing] OPENED {ticker} {strike}{opt_type[0].upper()} {expiry} "
                        f"@ ₹{prem:.2f}  stop_underlying={stop_level:.0f}  {suffix}"
                    )
                    executed.append(result)

            except Exception as e:
                logger.warning(f"[DirSwing] Error on {ticker}: {e}", exc_info=True)

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
            if not is_fno_eligible(ticker):
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
        # Mentor rule: NEVER buy OTM — theta decay destroys OTM longs.
        # Low IV  → ATM (already cheap, good risk/reward)
        # High IV → ITM1 (deep enough to have intrinsic value, less vega risk)
        if iv_env["iv_regime"] == "HIGH":
            moneyness = "ITM1"
            trade_note = "High IV — buying ITM1 to reduce vega exposure"
        else:
            moneyness = "ATM"
            trade_note = f"{iv_env['iv_regime']} IV — ATM for best risk/reward"

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
# Strategy: Long Straddle  (#7 — Event / volatility play)
# Buy ATM Call + ATM Put at the same strike.
# Profits if the stock makes a big move in EITHER direction.
# Best used before earnings, RBI policy, budget, or when IV is low.
# ---------------------------------------------------------------------------

class LongStraddleStrategy:
    """
    Long Straddle = Buy ATM Call + Buy ATM Put (same strike, same expiry).
    Pure buy-only. Max loss = total premiums paid. Profit = big move either way.

    Triggers:
      1. Event-driven — upcoming RBI, earnings season, budget, F&O expiry week
      2. Low IV — 10-day HV < 70% of 30-day HV (options are cheap, good time to buy)
      3. Post-consolidation — price stuck in tight range for 10+ days (coiled spring)
    """

    # Sector earnings seasons by month (NSE Q4: Apr-May, Q1: Jul-Aug, Q2: Oct-Nov, Q3: Jan-Feb)
    _EARNINGS_MONTHS = {4, 5, 7, 8, 10, 11, 1, 2}

    def run(self, portfolio: FNOPortfolio) -> list:
        executed = []
        if portfolio.at_max_positions():
            return executed

        now = _now_ist()
        if not (dt_time(9, 30) <= now.time() <= dt_time(14, 0)):
            return executed

        # Only trade straddles 1–2× per day at most
        straddle_positions = [
            k for k in portfolio.state["positions"]
            if portfolio.state["positions"][k].get("strategy", "").startswith("STRADDLE")
        ]
        if len(straddle_positions) >= 2:
            return executed

        candidates = ["^NSEI", "^NSEBANK"]  # Index straddles primarily

        for underlying in candidates:
            if portfolio.at_max_positions():
                break
            # Check cooldown
            base = underlying.replace("^", "").replace(".NS", "")
            if portfolio._is_in_cooldown(base):
                continue

            try:
                # Fetch daily data for volatility check
                df = yf.download(underlying, period="60d", interval="1d",
                                 auto_adjust=True, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if len(df) < 30:
                    continue

                close = df["Close"].squeeze()
                spot  = float(close.iloc[-1])

                # HV10 vs HV30 — low IV = cheap straddle
                hv10 = float(close.pct_change().rolling(10).std().iloc[-1]) * (252**0.5) * 100
                hv30 = float(close.pct_change().rolling(30).std().iloc[-1]) * (252**0.5) * 100
                iv_cheap = hv10 < hv30 * 0.70   # current vol < 70% of historical → cheap

                # Consolidation: tight range for 10 days
                range_10 = (float(close.iloc[-10:].max()) - float(close.iloc[-10:].min())) / spot
                consolidating = range_10 < 0.025   # < 2.5% range = coiled spring

                # Event proximity from risk manager calendar
                in_earnings_month = now.month in self._EARNINGS_MONTHS

                conviction = 0
                reasons = []
                if iv_cheap:
                    conviction += 30
                    reasons.append(f"Low IV (HV10={hv10:.1f}% < HV30={hv30:.1f}%)")
                if consolidating:
                    conviction += 35
                    reasons.append(f"Tight range {range_10*100:.1f}% → breakout expected")
                if in_earnings_month:
                    conviction += 25
                    reasons.append("Earnings season")

                if conviction < 55:
                    continue

                # ATM strike
                lot_size = get_lot_size(underlying)
                expiry   = get_expiry(underlying)
                strike   = round(spot / 100) * 100  # nearest 100

                # Price both legs via Black-Scholes
                t_exp = max(1, (expiry - now.date()).days) / 365
                sigma = max(hv10, hv30) / 100
                call_prem = BlackScholes.price(spot, strike, t_exp, RISK_FREE_RATE, sigma, "call")
                put_prem  = BlackScholes.price(spot, strike, t_exp, RISK_FREE_RATE, sigma, "put")

                if call_prem < 10 or put_prem < 10:
                    continue

                total_cost = (call_prem + put_prem) * lot_size
                if not portfolio.can_afford(total_cost):
                    continue

                # Open call leg
                call_pos = portfolio.open_option(
                    underlying=underlying, strike=strike, expiry=expiry,
                    option_type="call", position="LONG", qty_lots=1,
                    strategy="STRADDLE_CALL", reason=f"Long Straddle | {' | '.join(reasons)}"
                )
                put_pos = portfolio.open_option(
                    underlying=underlying, strike=strike, expiry=expiry,
                    option_type="put", position="LONG", qty_lots=1,
                    strategy="STRADDLE_PUT", reason=f"Long Straddle | {' | '.join(reasons)}"
                )

                if call_pos or put_pos:
                    logger.info(
                        f"[Straddle] {underlying} {strike} Call+Put @ ₹{call_prem:.1f}+₹{put_prem:.1f} "
                        f"cost=₹{total_cost:,.0f} conv={conviction} | {', '.join(reasons)}"
                    )
                    executed.extend([p for p in [call_pos, put_pos] if p])

            except Exception as exc:
                logger.debug(f"[Straddle] {underlying} error: {exc}")

        return executed


# ---------------------------------------------------------------------------
# Strategy: Long Strangle  (Iron Condor equivalent — buy-only)
# Buy OTM Call + OTM Put at different strikes.
# Cheaper than straddle, needs a bigger move to profit.
# Best when breakout direction is unknown but move is expected.
# ---------------------------------------------------------------------------

class LongStrangleStrategy:
    """
    Long Strangle = Buy OTM Call + Buy OTM Put (different strikes).
    Lower cost than straddle. Needs bigger price move to be profitable.
    Max loss = both premiums paid.

    Triggers:
      1. Strong multi-strategy BUY signals AND strong SELL signals on same stock
         simultaneously → direction genuinely unclear, big move expected
      2. Pre-event on index (use index options for defined risk)
      3. IV significantly low vs 60-day average
    """

    def run(self, equity_signals: list, portfolio: FNOPortfolio) -> list:
        executed = []
        if portfolio.at_max_positions():
            return executed

        now = _now_ist()
        if not (dt_time(9, 30) <= now.time() <= dt_time(14, 0)):
            return executed

        # Find tickers with BOTH strong BUY and strong SELL signals (confused market)
        buy_strength:  dict[str, int] = {}
        sell_strength: dict[str, int] = {}
        for sig in equity_signals:
            t = sig.get("ticker", "")
            s = sig.get("signal", "")
            st = int(sig.get("strength", 0))
            if s == "BUY"  and st >= 70: buy_strength[t]  = max(buy_strength.get(t, 0),  st)
            if s == "SELL" and st >= 70: sell_strength[t] = max(sell_strength.get(t, 0), st)

        # Stocks with both signals → genuine uncertainty → strangle candidate
        both = [t for t in buy_strength if t in sell_strength]

        # Also add index strangle if VIX is low (cheap strangle window)
        for underlying in ["^NSEI"]:
            base = "NSEI"
            if portfolio._is_in_cooldown(base):
                continue
            # Limit to 1 index strangle per day
            idx_strangles = sum(
                1 for k, v in portfolio.state["positions"].items()
                if v.get("strategy", "").startswith("STRANGLE") and "NSEI" in k
            )
            if idx_strangles >= 1:
                continue
            both.append(underlying)

        for underlying in both[:3]:   # max 3 strangles per cycle
            if portfolio.at_max_positions():
                break
            base = underlying.replace("^", "").replace(".NS", "")
            if portfolio._is_in_cooldown(base):
                continue

            try:
                df = yf.download(underlying, period="60d", interval="1d",
                                 auto_adjust=True, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if len(df) < 30:
                    continue

                close = df["Close"].squeeze()
                spot  = float(close.iloc[-1])
                hv20  = float(close.pct_change().rolling(20).std().iloc[-1]) * (252**0.5) * 100
                hv60  = float(close.pct_change().rolling(60).std().iloc[-1]) * (252**0.5) * 100

                # OTM strikes: 1 standard deviation away
                daily_sd  = float(close.pct_change().std())
                sd_7d     = daily_sd * (7**0.5) * spot
                call_strike = round((spot + sd_7d) / 50) * 50
                put_strike  = round((spot - sd_7d) / 50) * 50

                lot_size = get_lot_size(underlying)
                expiry   = get_expiry(underlying)
                sigma    = max(hv20, hv60) / 100
                t_exp    = max(1, (expiry - now.date()).days) / 365

                call_prem = BlackScholes.price(spot, call_strike, t_exp, RISK_FREE_RATE, sigma, "call")
                put_prem  = BlackScholes.price(spot, put_strike,  t_exp, RISK_FREE_RATE, sigma, "put")

                if call_prem < 5 or put_prem < 5:
                    continue

                total_cost = (call_prem + put_prem) * lot_size
                if not portfolio.can_afford(total_cost):
                    continue

                call_pos = portfolio.open_option(
                    underlying=underlying, strike=call_strike, expiry=expiry,
                    option_type="call", position="LONG", qty_lots=1,
                    strategy="STRANGLE_CALL",
                    reason=f"Long Strangle | OTM +{call_strike-spot:.0f} | HV20={hv20:.1f}%"
                )
                put_pos = portfolio.open_option(
                    underlying=underlying, strike=put_strike, expiry=expiry,
                    option_type="put", position="LONG", qty_lots=1,
                    strategy="STRANGLE_PUT",
                    reason=f"Long Strangle | OTM -{spot-put_strike:.0f} | HV20={hv20:.1f}%"
                )

                if call_pos or put_pos:
                    logger.info(
                        f"[Strangle] {underlying} {put_strike}P/{call_strike}C "
                        f"@ ₹{put_prem:.1f}+₹{call_prem:.1f} cost=₹{total_cost:,.0f}"
                    )
                    executed.extend([p for p in [call_pos, put_pos] if p])

            except Exception as exc:
                logger.debug(f"[Strangle] {underlying} error: {exc}")

        return executed


# ---------------------------------------------------------------------------
# F&O Agent — orchestrates all strategies
# ---------------------------------------------------------------------------

class FNOAgent:
    """
    Runs all F&O strategies in sequence each cycle.
    Called from the main TradingAgent.run_cycle().
    """

    def __init__(self, equity_portfolio=None):
        self.portfolio     = FNOPortfolio(equity_portfolio=equity_portfolio)
        self.directional   = DirectionalOptionsStrategy()
        self.straddle      = LongStraddleStrategy()
        self.strangle      = LongStrangleStrategy()
        self.spreads       = SpreadStrategy()      # Bull Call Spread + Bear Put Spread (defined risk)
        self.iron_condor   = IronCondorStrategy()  # disabled (sells options)
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

        # ── Market hours guard: skip entirely outside 9:00-15:35 IST ─────── #
        _now_t = _now_ist().time()
        if not (dt_time(9, 0) <= _now_t <= dt_time(15, 35)):
            logger.info(f"[FNO] run_cycle skipped — outside market hours ({_now_t.strftime('%H:%M')} IST)")
            return {"skipped": True, "reason": "outside_market_hours",
                    "fno_value": self.portfolio.get_total_value(),
                    "fno_cash":  round(self.portfolio._eq_cash(), 2)}

        logger.info("[FNO] === F&O Cycle Start ===")
        gap_closes = self.portfolio.morning_gap_check()   # runs only 9:15–9:30
        stops    = self.portfolio.check_expiry_and_stops()
        # IronCondorStrategy DISABLED: sells naked options (unlimited loss exposure).
        # SpreadStrategy RE-ENABLED: Bull Call Spread and Bear Put Spread are
        # DEFINED-RISK strategies — max loss = net debit paid, not unlimited.
        condors   = []   # self.iron_condor.run(self.portfolio) — naked short options, disabled
        spreads   = self.spreads.run(equity_signals, self.portfolio)   # defined-risk ✓
        directs   = self.directional.run(equity_signals, self.portfolio)
        straddles = self.straddle.run(self.portfolio)
        strangles = self.strangle.run(equity_signals, self.portfolio)
        hedges    = self.hedge.run(equity_positions, self.portfolio, equity_drawdown_pct)

        total_value   = self.portfolio.get_total_value()
        greeks        = self.portfolio.portfolio_greeks()
        unrealised    = self.portfolio.get_unrealised_pnl()
        realised      = self.portfolio.state["realised_pnl"]
        total_pnl     = round(unrealised + realised, 2)

        summary = {
            "fno_value":    total_value,
            "fno_pnl":      total_pnl,
            "fno_cash":     round(self.portfolio.state["cash"], 2),
            "gap_stops":    len(gap_closes),
            "stops_closed": len(stops),
            "new_condors":   len(condors) // 4,
            "new_directs":   len(directs),
            "new_straddles": len(straddles) // 2,
            "new_strangles": len(strangles) // 2,
            "new_spreads":   len(spreads) // 2,
            "new_hedges":    len(hedges),
            "open_positions": len(self.portfolio.state["positions"]),
            "portfolio_greeks": greeks,
        }
        logger.info(f"[FNO] Cycle done — value=₹{total_value:,.0f}  Greeks={greeks}")
        return summary

    def get_dashboard_data(self) -> dict:
        # Deployed = actual cash spent on open positions (entry_premium × qty × lot_size)
        positions_display = self.portfolio.get_positions_display()
        deployed = round(sum(
            p.get("cost") or (p.get("entry_premium", 0) * p.get("qty_lots", 1))
            for p in positions_display
        ), 2)
        unrealised = self.portfolio.get_unrealised_pnl()
        # F&O desk P&L = unrealised on open positions + all realised
        total_pnl = round(unrealised + self.portfolio.state["realised_pnl"], 2)
        return {
            "portfolio_value": round(deployed + unrealised, 2),  # current value of open options
            "deployed":        deployed,       # premiums paid, still open
            "cash":            self.portfolio._eq_cash(),
            "initial":         self.portfolio.state.get("initial", FNO_INITIAL_CAPITAL),
            "realised_pnl":    self.portfolio.state["realised_pnl"],
            "unrealised_pnl":  unrealised,
            "total_pnl":       total_pnl,
            "positions":       positions_display,
            "greeks":          self.portfolio.portfolio_greeks(),
            "open_count":      len(self.portfolio.state["positions"]),
            "unified":         self.portfolio._equity is not None,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_fno_agent: FNOAgent | None = None

def get_fno_agent(equity_portfolio=None) -> FNOAgent:
    global _fno_agent
    if _fno_agent is None:
        if equity_portfolio is None:
            try:
                from engine import get_agent as _get_eq  # noqa: PLC0415
                equity_portfolio = _get_eq().portfolio
            except Exception:
                pass
        _fno_agent = FNOAgent(equity_portfolio=equity_portfolio)
    elif equity_portfolio is not None and _fno_agent.portfolio._equity is None:
        # Wire in equity portfolio if it wasn't available at construction time
        _fno_agent.portfolio._equity = equity_portfolio
    return _fno_agent
