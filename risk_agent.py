"""
Portfolio Risk Agent
====================
Computes institutional-grade risk metrics for the paper trading portfolio:
  - Value at Risk (VaR 95%, 1-day)
  - Portfolio Beta to Nifty 50
  - Annualised Sharpe Ratio
  - Maximum Drawdown
  - Correlation Matrix (top-10 positions)
  - Concentration Risk

Singleton: get_risk_agent()
Cache: 10 minutes (avoids hammering yfinance on every API call)
"""

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
TRADE_LOG_FILE = DATA_DIR / "trade_log.json"

_CACHE_TTL = 600  # 10 minutes in seconds

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_trade_log() -> list:
    try:
        if TRADE_LOG_FILE.exists():
            with open(TRADE_LOG_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _fetch_returns(ticker: str, period: str = "1y") -> pd.Series:
    """Fetch daily close-to-close returns for a ticker. Returns empty Series on failure."""
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or "Close" not in df.columns:
            return pd.Series(dtype=float)
        closes = df["Close"].dropna()
        return closes.pct_change().dropna()
    except Exception as exc:
        logger.warning(f"[RiskAgent] fetch_returns failed for {ticker}: {exc}")
        return pd.Series(dtype=float)


def _equity_curve_from_trades(trade_log: list, initial_capital: float) -> pd.Series:
    """
    Reconstruct a daily equity curve from the trade log.
    We accumulate realised P&L day by day.
    Returns a pd.Series indexed by date string, values = cumulative portfolio value.
    """
    if not trade_log:
        return pd.Series(dtype=float)

    daily_pnl: dict = {}
    for trade in trade_log:
        raw_time = trade.get("time", "")
        try:
            date_str = str(raw_time)[:10]  # "YYYY-MM-DD"
            pnl = float(trade.get("pnl", 0) or 0)
            daily_pnl[date_str] = daily_pnl.get(date_str, 0.0) + pnl
        except Exception:
            continue

    if not daily_pnl:
        return pd.Series(dtype=float)

    dates = sorted(daily_pnl.keys())
    cumulative = initial_capital
    curve = {}
    for d in dates:
        cumulative += daily_pnl[d]
        curve[d] = cumulative

    return pd.Series(curve)


# ---------------------------------------------------------------------------
# PortfolioRiskAgent
# ---------------------------------------------------------------------------

class PortfolioRiskAgent:
    """Computes and caches portfolio risk metrics."""

    def __init__(self):
        self._lock = threading.Lock()
        self._last_run: float = 0.0
        self._cached_result: dict = self._empty_result()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, positions: list, cash: float, initial_capital: float) -> dict:
        """
        Compute all risk metrics.

        Parameters
        ----------
        positions : list of dicts, each containing:
            ticker, qty, avg_price, ltp, pnl_pct
        cash : float
            Current available cash.
        initial_capital : float
            Starting capital (used for equity curve baseline).

        Returns
        -------
        dict with all risk metrics.
        """
        with self._lock:
            now = time.time()
            if now - self._last_run < _CACHE_TTL and self._last_run > 0:
                return self._cached_result

            result = self._compute(positions, cash, initial_capital)
            self._cached_result = result
            self._last_run = now
            return result

    def get_dashboard_data(self) -> dict:
        """Return cached result from last run(). Returns zeros if never run."""
        with self._lock:
            return self._cached_result

    def get_risk_summary(self) -> dict:
        """Quick headline metrics."""
        with self._lock:
            d = self._cached_result
            return {
                "var_95":               d.get("var_95", 0.0),
                "beta":                 d.get("beta", 0.0),
                "sharpe":               d.get("sharpe", 0.0),
                "max_drawdown_pct":     d.get("max_drawdown_pct", 0.0),
                "top_concentration_pct":d.get("concentration", {}).get("top_position_pct", 0.0),
            }

    # ------------------------------------------------------------------
    # Internal computation
    # ------------------------------------------------------------------

    def _compute(self, positions: list, cash: float, initial_capital: float) -> dict:
        """Run all risk calculations. Called under the lock."""
        result = self._empty_result()

        if not positions:
            return result

        # ── Position values ───────────────────────────────────────────────
        pos_values = []
        for p in positions:
            ltp = float(p.get("ltp") or p.get("avg_price") or 0)
            qty = int(p.get("qty", 0))
            val = ltp * qty
            pos_values.append({
                "ticker": p["ticker"],
                "value":  val,
                "ltp":    ltp,
                "qty":    qty,
                "strategy": p.get("strategy", "UNKNOWN"),
            })

        total_equity = sum(pv["value"] for pv in pos_values) + cash
        if total_equity <= 0:
            return result

        # ── Fetch Nifty returns ───────────────────────────────────────────
        nifty_rets = _fetch_returns("^NSEI", period="1y")

        # ── Fetch individual stock returns ─────────────────────────────────
        stock_rets: dict[str, pd.Series] = {}
        for pv in pos_values:
            ticker = pv["ticker"]
            if not ticker.endswith(".NS"):
                ticker = ticker + ".NS"
            rets = _fetch_returns(ticker, period="1y")
            if not rets.empty:
                stock_rets[ticker] = rets

        # ── VaR (95%, 1-day) ──────────────────────────────────────────────
        result["var_95"] = self._compute_var(pos_values, stock_rets, total_equity)

        # ── Portfolio Beta ─────────────────────────────────────────────────
        result["beta"] = self._compute_beta(pos_values, stock_rets, nifty_rets, total_equity)

        # ── Sharpe Ratio ──────────────────────────────────────────────────
        trade_log = _load_trade_log()
        result["sharpe"] = self._compute_sharpe(trade_log, initial_capital)

        # ── Max Drawdown ──────────────────────────────────────────────────
        result["max_drawdown_pct"] = self._compute_max_drawdown(trade_log, initial_capital)

        # ── Correlation Matrix (top 10) ───────────────────────────────────
        result["correlation_matrix"] = self._compute_correlation(pos_values, stock_rets)

        # ── Concentration Risk ────────────────────────────────────────────
        result["concentration"] = self._compute_concentration(pos_values, total_equity)

        result["computed_at"] = datetime.now().isoformat()
        result["total_portfolio_value"] = total_equity
        result["num_positions"] = len(positions)

        return result

    def _compute_var(
        self,
        pos_values: list,
        stock_rets: dict,
        total_portfolio_value: float,
    ) -> float:
        """VaR 95% confidence, 1-day, historical simulation."""
        if total_portfolio_value <= 0:
            return 0.0

        # Build weighted portfolio return series
        # Align all series to a common date index
        series_list = []
        weights = []
        for pv in pos_values:
            ticker = pv["ticker"]
            if not ticker.endswith(".NS"):
                ticker = ticker + ".NS"
            rets = stock_rets.get(ticker)
            if rets is None or rets.empty:
                continue
            w = pv["value"] / total_portfolio_value
            series_list.append(rets.rename(ticker))
            weights.append(w)

        if not series_list:
            return 0.0

        df = pd.concat(series_list, axis=1).dropna()
        if df.empty:
            return 0.0

        weights_arr = np.array(weights[:len(df.columns)])
        # Normalise weights in case some tickers had no data
        w_sum = weights_arr.sum()
        if w_sum <= 0:
            return 0.0
        weights_arr = weights_arr / w_sum

        portfolio_daily_rets = df.values @ weights_arr
        var_pct = np.percentile(portfolio_daily_rets, 5)
        var_inr = abs(var_pct) * total_portfolio_value  # positive number
        return round(var_inr, 2)

    def _compute_beta(
        self,
        pos_values: list,
        stock_rets: dict,
        nifty_rets: pd.Series,
        total_portfolio_value: float,
    ) -> float:
        """Weighted average portfolio beta to Nifty 50."""
        if nifty_rets.empty or total_portfolio_value <= 0:
            return 0.0

        portfolio_beta = 0.0
        total_weight = 0.0

        nifty_var = nifty_rets.var()
        if nifty_var == 0:
            return 0.0

        for pv in pos_values:
            ticker = pv["ticker"]
            if not ticker.endswith(".NS"):
                ticker = ticker + ".NS"
            rets = stock_rets.get(ticker)
            if rets is None or rets.empty:
                continue

            # Align to common dates
            aligned = pd.concat([rets, nifty_rets], axis=1).dropna()
            if aligned.shape[0] < 20:
                continue

            stock_r = aligned.iloc[:, 0].values
            nifty_r = aligned.iloc[:, 1].values
            cov = np.cov(stock_r, nifty_r)[0, 1]
            beta_i = cov / nifty_var

            w = pv["value"] / total_portfolio_value
            portfolio_beta += w * beta_i
            total_weight += w

        if total_weight > 0:
            portfolio_beta = portfolio_beta / total_weight

        return round(portfolio_beta, 4)

    def _compute_sharpe(self, trade_log: list, initial_capital: float) -> float:
        """Annualised Sharpe ratio from trade log P&L series. Risk-free = 6.5%."""
        RF_DAILY = 0.065 / 252

        equity_curve = _equity_curve_from_trades(trade_log, initial_capital)
        if len(equity_curve) < 2:
            return 0.0

        daily_rets = equity_curve.pct_change().dropna()
        if daily_rets.empty or daily_rets.std() == 0:
            return 0.0

        mean_ret = daily_rets.mean()
        std_ret = daily_rets.std()
        sharpe = (mean_ret - RF_DAILY) / std_ret * np.sqrt(252)
        return round(sharpe, 4)

    def _compute_max_drawdown(self, trade_log: list, initial_capital: float) -> float:
        """Max drawdown % from equity curve reconstructed from trade log."""
        equity_curve = _equity_curve_from_trades(trade_log, initial_capital)
        if len(equity_curve) < 2:
            return 0.0

        values = equity_curve.values
        rolling_max = np.maximum.accumulate(values)
        drawdowns = (values - rolling_max) / rolling_max
        max_dd = float(drawdowns.min())  # most negative
        return round(abs(max_dd) * 100, 4)  # return as positive %

    def _compute_correlation(
        self,
        pos_values: list,
        stock_rets: dict,
    ) -> list:
        """Pairwise correlation for top-10 positions by value. Returns sorted list of dicts."""
        # Sort by value descending, take top 10
        top10 = sorted(pos_values, key=lambda x: x["value"], reverse=True)[:10]

        # Build return matrix
        series_dict = {}
        for pv in top10:
            ticker = pv["ticker"]
            key = ticker if ticker.endswith(".NS") else ticker + ".NS"
            rets = stock_rets.get(key)
            if rets is not None and not rets.empty:
                series_dict[pv["ticker"]] = rets

        if len(series_dict) < 2:
            return []

        df = pd.concat(series_dict.values(), axis=1, keys=series_dict.keys()).dropna()
        if df.shape[0] < 5:
            return []

        corr_matrix = df.corr()
        tickers = list(corr_matrix.columns)

        pairs = []
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                corr_val = corr_matrix.iloc[i, j]
                if pd.notna(corr_val):
                    pairs.append({
                        "ticker_a":    tickers[i],
                        "ticker_b":    tickers[j],
                        "correlation": round(float(corr_val), 4),
                    })

        pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return pairs

    def _compute_concentration(self, pos_values: list, total_portfolio_value: float) -> dict:
        """Sector / strategy exposure, top position weight, positions > 5%."""
        if total_portfolio_value <= 0:
            return {"sector_exposure": {}, "top_position_pct": 0.0, "positions_over_5pct": 0}

        # Strategy-level grouping (first word of strategy field)
        sector_exposure: dict[str, float] = {}
        for pv in pos_values:
            strategy = pv.get("strategy", "UNKNOWN")
            group = strategy.split("_")[0] if strategy else "UNKNOWN"
            pct = pv["value"] / total_portfolio_value * 100
            sector_exposure[group] = sector_exposure.get(group, 0.0) + pct

        # Top single position
        if pos_values:
            top_val = max(pv["value"] for pv in pos_values)
            top_pct = top_val / total_portfolio_value * 100
        else:
            top_pct = 0.0

        over_5 = sum(
            1 for pv in pos_values
            if (pv["value"] / total_portfolio_value * 100) > 5
        )

        return {
            "sector_exposure":       {k: round(v, 2) for k, v in sector_exposure.items()},
            "top_position_pct":      round(top_pct, 2),
            "positions_over_5pct":   over_5,
        }

    @staticmethod
    def _empty_result() -> dict:
        return {
            "var_95":               0.0,
            "beta":                 0.0,
            "sharpe":               0.0,
            "max_drawdown_pct":     0.0,
            "correlation_matrix":   [],
            "concentration": {
                "sector_exposure":     {},
                "top_position_pct":    0.0,
                "positions_over_5pct": 0,
            },
            "total_portfolio_value": 0.0,
            "num_positions":         0,
            "computed_at":           None,
        }


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_INSTANCE: PortfolioRiskAgent | None = None
_INSTANCE_LOCK = threading.Lock()


def get_risk_agent() -> PortfolioRiskAgent:
    """Return the singleton PortfolioRiskAgent, creating it on first call."""
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = PortfolioRiskAgent()
    return _INSTANCE
