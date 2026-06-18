"""
Smart Execution Agent
=====================
Optimises order execution timing, sizing, and limit price suggestions for
the Indian paper-trading terminal (NSE equities).

Key capabilities
----------------
- optimal_entry_window   — best IST hours to buy/sell based on intraday volume profile
- optimal_position_size  — ATR-based Kelly position sizing capped at 8% of portfolio
- suggest_limit_price    — VWAP-anchored limit price for BUY / SELL orders
- get_pre_trade_analysis — GO / WAIT / NO decision combining the above three
- get_dashboard_data     — pre-trade analysis for the top 5 BUY signals (cached 15 min)
"""

import json
import logging
import threading
from datetime import datetime, time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from engine import DataFetcher

logger = logging.getLogger(__name__)

_IST_TZ = ZoneInfo("Asia/Kolkata")
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SIGNALS_PATH = DATA_DIR / "signals.json"

# ---------------------------------------------------------------------------
# Constants (mirrors engine.py)
# ---------------------------------------------------------------------------
MAX_POSITION_PCT = 0.08   # 8% max per position
STOP_LOSS_PCT    = 0.07   # 7% stop-loss
TAKE_PROFIT_PCT  = 0.20   # 20% take-profit

# ATR-stop multiplier: stop_distance = ATR_MULT * ATR(14)
ATR_MULT = 2.0

# Limit order offset from reference price
LIMIT_OFFSET_BUY  = 0.001   # 0.1% below VWAP/current
LIMIT_OFFSET_SELL = 0.001   # 0.1% above VWAP/current

# Dashboard cache TTL
_DASH_CACHE_TTL_SEC = 900  # 15 minutes

# Best / avoid windows (IST)
_BEST_BUY_WINDOWS  = [("09:15", "09:45"), ("14:30", "15:00")]
_AVOID_WINDOWS     = [("11:30", "13:00")]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_ist() -> datetime:
    return datetime.now(_IST_TZ)


def _market_open() -> bool:
    now = _now_ist().time()
    return dt_time(9, 15) <= now <= dt_time(15, 30)


def _in_best_window() -> bool:
    now = _now_ist().time()
    return (
        (dt_time(9, 15) <= now <= dt_time(9, 45))
        or (dt_time(14, 30) <= now <= dt_time(15, 0))
    )


def _in_avoid_window() -> bool:
    now = _now_ist().time()
    return dt_time(11, 30) <= now <= dt_time(13, 0)


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """True-range ATR for a DataFrame with High, Low, Close columns."""
    high = df["High"].astype(float)
    low  = df["Low"].astype(float)
    close_prev = df["Close"].astype(float).shift(1)

    tr = pd.concat(
        [high - low, (high - close_prev).abs(), (low - close_prev).abs()], axis=1
    ).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def _compute_vwap(df: pd.DataFrame) -> float:
    """
    VWAP from a 5-min intraday DataFrame with columns Close, High, Low, Volume.
    Uses typical price = (H + L + C) / 3.
    """
    typical = (df["High"].astype(float) + df["Low"].astype(float) + df["Close"].astype(float)) / 3
    vol = df["Volume"].astype(float).replace(0, np.nan).fillna(1)
    vwap = (typical * vol).sum() / vol.sum()
    return float(vwap)


def _load_signals() -> list:
    """Load the latest signals from data/signals.json."""
    try:
        if SIGNALS_PATH.exists():
            with open(SIGNALS_PATH, "r") as f:
                data = json.load(f)
            # signals.json may be a plain list or a dict {"signals": [...]}
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("signals", []) or data.get("data", [])
    except Exception as exc:
        logger.warning("Could not load signals.json: %s", exc)
    return []


# ---------------------------------------------------------------------------
# Singleton guard
# ---------------------------------------------------------------------------
_INSTANCE: "SmartExecutionAgent | None" = None
_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SmartExecutionAgent:
    """
    Execution intelligence layer.
    All public methods are safe to call from Flask request threads concurrently.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._dash_cache: dict | None = None
        self._dash_cache_ts: datetime | None = None

    # ------------------------------------------------------------------
    # 1. Optimal entry window
    # ------------------------------------------------------------------

    def optimal_entry_window(self, ticker: str) -> dict:
        """
        Fetch 5-day 5-min intraday data for *ticker* and return the best
        buy windows, windows to avoid, average bid-ask spread proxy, and
        a plain-English recommendation.

        Returns
        -------
        ::

            {
              "ticker":            "RELIANCE.NS",
              "best_buy_windows":  ["09:15-09:45 IST", "14:30-15:00 IST"],
              "avoid_windows":     ["11:30-13:00 IST"],
              "avg_spread_pct":    0.12,
              "hourly_volume":     {"09": 1_200_000, "10": 800_000, ...},
              "recommendation":    "..."
            }
        """
        ticker = ticker.upper()
        try:
            df = yf.download(
                ticker,
                period="5d",
                interval="5m",
                auto_adjust=True,
                progress=False,
            )
            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
        except Exception as exc:
            logger.warning("optimal_entry_window download failed %s: %s", ticker, exc)
            df = pd.DataFrame()

        best_buy_windows = ["09:15-09:45 IST", "14:30-15:00 IST"]
        avoid_windows = ["11:30-13:00 IST"]
        avg_spread_pct = 0.10
        hourly_volume: dict[str, int] = {}

        if not df.empty and "Volume" in df.columns and len(df) > 5:
            df.index = pd.to_datetime(df.index)
            # Localise or convert to IST
            try:
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC").tz_convert(_IST_TZ)
                else:
                    df.index = df.index.tz_convert(_IST_TZ)
            except Exception:
                pass

            # Hourly volume profile
            df["hour"] = df.index.hour
            hourly_vol = df.groupby("hour")["Volume"].mean().astype(int).to_dict()
            hourly_volume = {str(h).zfill(2): int(v) for h, v in hourly_vol.items()}

            # Average spread proxy: (High - Low) / Close as % (5-min candle spread)
            spread_series = (df["High"].astype(float) - df["Low"].astype(float)) / df["Close"].astype(float) * 100
            avg_spread_pct = round(float(spread_series.mean()), 3)

            # Identify high-volume hours to refine windows
            if hourly_vol:
                max_hour = max(hourly_vol, key=hourly_vol.get)
                max_vol = hourly_vol[max_hour]
                high_vol_hours = [h for h, v in hourly_vol.items() if v >= 0.6 * max_vol]
                best_buy_windows = [
                    f"{str(h).zfill(2)}:00-{str(h).zfill(2)}:59 IST"
                    for h in sorted(high_vol_hours)
                    if h in (9, 14, 15)
                ] or best_buy_windows

        now = _now_ist()
        if _in_best_window():
            recommendation = (
                f"Market is currently in a high-liquidity window for {ticker}. "
                "Spreads are tight — good time to place a limit order."
            )
        elif _in_avoid_window():
            recommendation = (
                f"Market is in the lunch lull (11:30–13:00 IST) for {ticker}. "
                "Volumes are thin and spreads widen. Wait for 14:30 IST."
            )
        elif not _market_open():
            recommendation = (
                f"Market is closed. Best windows for {ticker} are 09:15–09:45 IST "
                "(opening momentum) and 14:30–15:00 IST (closing push)."
            )
        else:
            recommendation = (
                f"Acceptable window for {ticker}. Optimal windows are "
                "09:15–09:45 IST and 14:30–15:00 IST. Avoid 11:30–13:00 IST."
            )

        return {
            "ticker": ticker,
            "best_buy_windows": best_buy_windows,
            "avoid_windows": avoid_windows,
            "avg_spread_pct": avg_spread_pct,
            "hourly_volume": hourly_volume,
            "recommendation": recommendation,
        }

    # ------------------------------------------------------------------
    # 2. ATR-based position sizing
    # ------------------------------------------------------------------

    def optimal_position_size(
        self,
        ticker: str,
        portfolio_value: float,
        risk_per_trade_pct: float = 0.01,
    ) -> dict:
        """
        ATR(14)-based position sizing.

        stop_distance = ATR_MULT * ATR(14)
        shares        = (portfolio_value * risk_per_trade_pct) / stop_distance
        Capped at MAX_POSITION_PCT (8%) of portfolio.

        Returns
        -------
        ::

            {
              "ticker":             "RELIANCE.NS",
              "current_price":      2850.50,
              "atr_14":             45.30,
              "stop_price":         2759.90,
              "recommended_shares": 22,
              "position_value":     62711.00,
              "position_pct":       6.27,
              "risk_amount":        9966.00,
              "limit_price":        2841.95
            }
        """
        ticker = ticker.upper()

        # Current price
        current_price = 0.0
        try:
            current_price = DataFetcher.get_current_price(ticker) or 0.0
        except Exception as exc:
            logger.warning("get_current_price failed for %s: %s", ticker, exc)

        # 60-day daily OHLCV for ATR
        atr_14 = 0.0
        try:
            df = yf.download(ticker, period="60d", interval="1d", auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty and len(df) >= 15:
                atr_14 = _compute_atr(df, period=14)
                if current_price <= 0:
                    current_price = float(df["Close"].iloc[-1])
        except Exception as exc:
            logger.warning("ATR fetch failed for %s: %s", ticker, exc)

        if current_price <= 0 or portfolio_value <= 0:
            return {
                "ticker": ticker,
                "current_price": current_price,
                "atr_14": atr_14,
                "stop_price": 0.0,
                "recommended_shares": 0,
                "position_value": 0.0,
                "position_pct": 0.0,
                "risk_amount": 0.0,
                "limit_price": 0.0,
                "error": "Insufficient price data",
            }

        stop_distance = ATR_MULT * atr_14 if atr_14 > 0 else current_price * STOP_LOSS_PCT
        stop_price = round(current_price - stop_distance, 2)

        # Risk-based shares
        risk_rupees = portfolio_value * risk_per_trade_pct
        shares_risk = int(risk_rupees / stop_distance) if stop_distance > 0 else 0

        # Cap at 8% of portfolio
        max_value = portfolio_value * MAX_POSITION_PCT
        shares_cap = int(max_value / current_price)

        recommended_shares = min(shares_risk, shares_cap)
        recommended_shares = max(0, recommended_shares)

        position_value = round(recommended_shares * current_price, 2)
        position_pct = round(position_value / portfolio_value * 100, 2) if portfolio_value > 0 else 0.0
        risk_amount = round(recommended_shares * stop_distance, 2)
        limit_price = round(current_price * (1 - LIMIT_OFFSET_BUY), 2)

        return {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "atr_14": round(atr_14, 2),
            "stop_price": stop_price,
            "recommended_shares": recommended_shares,
            "position_value": position_value,
            "position_pct": position_pct,
            "risk_amount": risk_amount,
            "limit_price": limit_price,
        }

    # ------------------------------------------------------------------
    # 3. VWAP-anchored limit price
    # ------------------------------------------------------------------

    def suggest_limit_price(self, ticker: str, action: str = "BUY") -> dict:
        """
        Compute a VWAP-anchored limit price for the given action.

        BUY  limit = min(current_price, vwap) - 0.1%
        SELL limit = max(current_price, vwap) + 0.1%

        Returns
        -------
        ::

            {
              "ticker":        "RELIANCE.NS",
              "action":        "BUY",
              "current_price": 2850.50,
              "vwap":          2845.20,
              "limit_price":   2842.35,
              "rationale":     "..."
            }
        """
        ticker = ticker.upper()
        action = action.upper()

        current_price = 0.0
        try:
            current_price = DataFetcher.get_current_price(ticker) or 0.0
        except Exception:
            pass

        vwap = 0.0
        try:
            df = yf.download(ticker, period="5d", interval="5m", auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty and "Volume" in df.columns and len(df) > 2:
                vwap = _compute_vwap(df)
                if current_price <= 0:
                    current_price = float(df["Close"].iloc[-1])
        except Exception as exc:
            logger.warning("suggest_limit_price VWAP fetch failed %s: %s", ticker, exc)

        if current_price <= 0:
            return {
                "ticker": ticker,
                "action": action,
                "current_price": 0.0,
                "vwap": 0.0,
                "limit_price": 0.0,
                "rationale": "Could not fetch price data",
            }

        if vwap <= 0:
            vwap = current_price  # fallback: treat VWAP = current

        if action == "BUY":
            reference = min(current_price, vwap)
            limit_price = round(reference * (1 - LIMIT_OFFSET_BUY), 2)
            rationale = (
                f"BUY limit set 0.1% below the lower of current price (₹{current_price:,.2f}) "
                f"and VWAP (₹{vwap:,.2f}). Places order at ₹{limit_price:,.2f} — "
                "likely to fill on any small dip while saving on slippage."
            )
        else:  # SELL
            reference = max(current_price, vwap)
            limit_price = round(reference * (1 + LIMIT_OFFSET_SELL), 2)
            rationale = (
                f"SELL limit set 0.1% above the higher of current price (₹{current_price:,.2f}) "
                f"and VWAP (₹{vwap:,.2f}). Places order at ₹{limit_price:,.2f} — "
                "squeezes extra value on exit."
            )

        return {
            "ticker": ticker,
            "action": action,
            "current_price": round(current_price, 2),
            "vwap": round(vwap, 2),
            "limit_price": limit_price,
            "rationale": rationale,
        }

    # ------------------------------------------------------------------
    # 4. Pre-trade GO / WAIT / NO decision
    # ------------------------------------------------------------------

    def get_pre_trade_analysis(self, ticker: str, portfolio_value: float) -> dict:
        """
        Combine optimal_entry_window, optimal_position_size, and
        suggest_limit_price into a single pre-trade decision.

        GO   — good timing window AND risk within limits
        WAIT — outside best window (but risk is acceptable)
        NO   — ATR-stop risk > 2% of portfolio per trade OR price unavailable

        Returns
        -------
        ::

            {
              "ticker":          "RELIANCE.NS",
              "go_no_go":        "GO",         # GO | WAIT | NO
              "reason":          "...",
              "entry_window":    {...},
              "position_size":   {...},
              "limit_price":     {...},
              "portfolio_value": 1000000.0
            }
        """
        ticker = ticker.upper()

        entry_window = self.optimal_entry_window(ticker)
        position_size = self.optimal_position_size(ticker, portfolio_value)
        limit_price_data = self.suggest_limit_price(ticker, action="BUY")

        current_price = position_size.get("current_price", 0.0)
        risk_amount = position_size.get("risk_amount", 0.0)
        recommended_shares = position_size.get("recommended_shares", 0)

        # Risk as % of portfolio
        risk_pct_of_portfolio = (risk_amount / portfolio_value * 100) if portfolio_value > 0 else 0.0

        # NO: no price or catastrophic risk
        if current_price <= 0 or recommended_shares == 0:
            go_no_go = "NO"
            reason = (
                f"Cannot determine a valid entry for {ticker}. "
                "Price data is unavailable or position size would be zero."
            )
        elif risk_pct_of_portfolio > 2.0:
            go_no_go = "NO"
            reason = (
                f"Risk per trade ({risk_pct_of_portfolio:.2f}% of portfolio) exceeds the 2% hard limit. "
                f"ATR-14 is ₹{position_size.get('atr_14', 0):.2f} — too volatile to size safely."
            )
        elif not _market_open():
            go_no_go = "WAIT"
            reason = "Market is currently closed. Submit the order at 09:15 IST tomorrow."
        elif _in_best_window():
            go_no_go = "GO"
            reason = (
                f"Market is in a high-volume, tight-spread window for {ticker}. "
                f"Risk per trade: {risk_pct_of_portfolio:.2f}% of portfolio. "
                f"Suggested limit: ₹{limit_price_data.get('limit_price', 0):,.2f}."
            )
        elif _in_avoid_window():
            go_no_go = "WAIT"
            reason = (
                f"Market is in the lunch lull for {ticker} (11:30–13:00 IST). "
                "Liquidity is reduced and spreads are wider. Wait for 14:30 IST."
            )
        else:
            go_no_go = "WAIT"
            reason = (
                f"Outside the optimal windows for {ticker}. Best windows are "
                "09:15–09:45 IST and 14:30–15:00 IST."
            )

        return {
            "ticker": ticker,
            "go_no_go": go_no_go,
            "reason": reason,
            "risk_pct_of_portfolio": round(risk_pct_of_portfolio, 3),
            "entry_window": entry_window,
            "position_size": position_size,
            "limit_price_data": limit_price_data,
            "portfolio_value": portfolio_value,
        }

    # ------------------------------------------------------------------
    # 5. Dashboard data (top-5 BUY signals, cached 15 min)
    # ------------------------------------------------------------------

    def get_dashboard_data(self) -> dict:
        """
        Run get_pre_trade_analysis for the top 5 BUY signals from
        data/signals.json. Result is cached for 15 minutes.

        Returns
        -------
        ::

            {
              "analyses":      [...],   # list of pre_trade_analysis dicts
              "market_open":   bool,
              "current_window": str,
              "last_updated":  "2026-06-18T15:30:00"
            }
        """
        with self._lock:
            if self._dash_cache and self._dash_cache_ts:
                age = (_now_ist() - self._dash_cache_ts).total_seconds()
                if age < _DASH_CACHE_TTL_SEC:
                    return self._dash_cache

        # Determine portfolio value
        portfolio_value = 1_000_000.0
        try:
            from engine import get_agent  # local import to avoid circular at module load
            agent = get_agent()
            portfolio_value = float(agent.portfolio.get_total_value())
        except Exception as exc:
            logger.debug("Could not fetch portfolio value: %s", exc)

        # Load latest signals
        signals = _load_signals()
        buy_signals = [s for s in signals if str(s.get("signal", "")).upper() == "BUY"]
        # Sort by score descending
        buy_signals.sort(key=lambda s: float(s.get("score", 0) or 0), reverse=True)
        top5 = buy_signals[:5]

        analyses = []
        for sig in top5:
            ticker = sig.get("ticker", "")
            if not ticker:
                continue
            try:
                analysis = self.get_pre_trade_analysis(ticker, portfolio_value)
                analysis["signal_score"] = sig.get("score", 0)
                analysis["signal_strategy"] = sig.get("strategy", "")
                analyses.append(analysis)
            except Exception as exc:
                logger.warning("Pre-trade analysis failed for %s: %s", ticker, exc)

        # Describe current window
        now = _now_ist()
        now_t = now.time()
        if now_t < dt_time(9, 15):
            window_label = "Pre-market"
        elif now_t <= dt_time(9, 45):
            window_label = "09:15-09:45 IST (opening push)"
        elif now_t <= dt_time(11, 30):
            window_label = "09:45-11:30 IST (optimal)"
        elif now_t <= dt_time(13, 0):
            window_label = "11:30-13:00 IST (lunch lull — avoid)"
        elif now_t <= dt_time(14, 30):
            window_label = "13:00-14:30 IST (afternoon)"
        elif now_t <= dt_time(15, 30):
            window_label = "14:30-15:30 IST (closing push)"
        else:
            window_label = "After-market"

        result = {
            "analyses": analyses,
            "market_open": _market_open(),
            "current_window": window_label,
            "last_updated": now.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        with self._lock:
            self._dash_cache = result
            self._dash_cache_ts = _now_ist()

        return result


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

def get_execution_agent() -> SmartExecutionAgent:
    """Return the global SmartExecutionAgent singleton."""
    global _INSTANCE
    if _INSTANCE is None:
        with _LOCK:
            if _INSTANCE is None:
                _INSTANCE = SmartExecutionAgent()
    return _INSTANCE
