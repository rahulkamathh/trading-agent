"""
Backtest Agent
==============
Runs a simplified momentum strategy backtest on historical NSE data.
Uses the same DataFetcher + add_indicators pipeline as the live engine.
"""

import json
import logging
import math
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from engine import DataFetcher, add_indicators, NIFTY50_TICKERS

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

BACKTEST_CACHE_FILE = DATA_DIR / "backtest_results.json"
CACHE_MAX_AGE_DAYS = 7

TRADE_CAPITAL = 100_000       # ₹1 lakh per trade
STOP_LOSS_PCT = 0.07          # 7%
TAKE_PROFIT_PCT = 0.20        # 20%
MAX_HOLD_DAYS = 60


# ---------------------------------------------------------------------------
# BacktestAgent
# ---------------------------------------------------------------------------

class BacktestAgent:
    """Singleton backtest agent."""

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_backtest(self, ticker: str, period_years: int = 3) -> dict:
        """
        Backtest a simplified momentum strategy on *ticker* for the given
        number of years.  Returns a performance-metrics dict.
        """
        days_needed = period_years * 252 + 250  # extra 250 for indicator warmup
        period_str = f"{days_needed}d"

        df = DataFetcher.fetch(ticker, period=period_str, interval="1d")
        if df is None or df.empty or len(df) < 250:
            return self._empty_result(ticker, period_years)

        df = add_indicators(df.copy())
        df = df.dropna(subset=["ema_200", "ema_50", "rsi", "adx"]).reset_index()

        # Rename the datetime index column if needed
        if "Date" not in df.columns and "Datetime" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Date"})
        date_col = "Date" if "Date" in df.columns else "Datetime"

        trades = self._simulate_trades(df, date_col)
        metrics = self._compute_metrics(ticker, period_years, trades)
        return metrics

    def run_portfolio_backtest(
        self, tickers: list = None, period_years: int = 3
    ) -> dict:
        """
        Run run_backtest() across multiple tickers (default: first 20
        NIFTY50_TICKERS).  Caches result to data/backtest_results.json.
        """
        # Return cached result if fresh
        cached = self._load_cache()
        if cached is not None:
            return cached

        if tickers is None:
            tickers = NIFTY50_TICKERS[:20]

        results = []
        for ticker in tickers:
            logger.info(f"Backtesting {ticker} …")
            try:
                result = self.run_backtest(ticker, period_years)
                results.append(result)
            except Exception as exc:
                logger.warning(f"Backtest failed for {ticker}: {exc}")
            time.sleep(0.3)

        results.sort(key=lambda r: r.get("total_return_pct", 0), reverse=True)

        # Aggregate metrics (ignore tickers with zero trades)
        valid = [r for r in results if r.get("total_trades", 0) > 0]
        all_win_rates = [r["win_rate"] for r in valid]
        all_pf = [r["profit_factor"] for r in valid if r.get("profit_factor") is not None]
        all_sharpe = [r["sharpe_ratio"] for r in valid if r.get("sharpe_ratio") is not None]

        best = max(results, key=lambda r: r.get("total_return_pct", -999)) if results else {}
        worst = min(results, key=lambda r: r.get("total_return_pct", 999)) if results else {}

        output = {
            "overall_win_rate": round(float(np.mean(all_win_rates)), 2) if all_win_rates else 0.0,
            "overall_profit_factor": round(float(np.mean(all_pf)), 2) if all_pf else 0.0,
            "best_performer": {
                "ticker": best.get("ticker", ""),
                "return_pct": best.get("total_return_pct", 0.0),
            } if best else {},
            "worst_performer": {
                "ticker": worst.get("ticker", ""),
                "return_pct": worst.get("total_return_pct", 0.0),
            } if worst else {},
            "avg_sharpe": round(float(np.mean(all_sharpe)), 2) if all_sharpe else 0.0,
            "results": results,
            "run_at": datetime.now().isoformat(),
        }

        self._save_cache(output)
        return output

    def get_dashboard_data(self) -> dict:
        """
        Returns cached portfolio backtest result.
        Loads from data/backtest_results.json if present, else empty structure.
        """
        cached = self._load_cache(ignore_age=True)
        if cached is not None:
            return cached
        return {
            "overall_win_rate": 0.0,
            "overall_profit_factor": 0.0,
            "best_performer": {},
            "worst_performer": {},
            "avg_sharpe": 0.0,
            "results": [],
            "run_at": None,
            "message": "No backtest data yet. Call /api/run_backtest to generate.",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _simulate_trades(self, df: pd.DataFrame, date_col: str) -> list:
        """Simulate momentum strategy trades on the indicator-enriched DataFrame."""
        trades = []
        in_position = False
        entry_idx = None
        entry_price = None
        entry_date = None
        stop_price = None
        take_price = None

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]

            if in_position:
                # Use today's close to check exit conditions
                close = float(row["Close"])
                exit_reason = None
                exit_price = close

                if close <= stop_price:
                    exit_reason = "STOP_LOSS"
                    exit_price = stop_price  # fill at stop
                elif close >= take_price:
                    exit_reason = "TAKE_PROFIT"
                    exit_price = take_price
                elif close < float(row["ema_200"]):
                    exit_reason = "TREND_BREAK"
                else:
                    hold_days = (i - entry_idx)
                    if hold_days >= MAX_HOLD_DAYS:
                        exit_reason = "MAX_HOLD"

                if exit_reason:
                    qty = TRADE_CAPITAL / entry_price
                    pnl = (exit_price - entry_price) * qty
                    hold_days = i - entry_idx
                    trades.append({
                        "entry_date": str(df.iloc[entry_idx][date_col])[:10],
                        "exit_date": str(row[date_col])[:10],
                        "entry_price": round(entry_price, 2),
                        "exit_price": round(exit_price, 2),
                        "pnl": round(pnl, 2),
                        "hold_days": hold_days,
                        "exit_reason": exit_reason,
                    })
                    in_position = False
                    entry_idx = entry_price = entry_date = None
                    stop_price = take_price = None
                continue  # don't check entry while in position

            # Entry conditions (evaluated on previous bar's indicators,
            # enter at today's open to simulate realistic fill)
            try:
                price_gt_ema200 = float(prev["Close"]) > float(prev["ema_200"])
                ema50_gt_ema200 = float(prev["ema_50"]) > float(prev["ema_200"])
                rsi_ok = 45 <= float(prev["rsi"]) <= 70
                adx_ok = float(prev["adx"]) > 20
            except (ValueError, TypeError):
                continue

            if price_gt_ema200 and ema50_gt_ema200 and rsi_ok and adx_ok:
                open_price = float(row.get("Open", row["Close"]))
                if open_price <= 0:
                    continue
                in_position = True
                entry_idx = i
                entry_price = open_price
                entry_date = str(row[date_col])[:10]
                stop_price = entry_price * (1 - STOP_LOSS_PCT)
                take_price = entry_price * (1 + TAKE_PROFIT_PCT)

        return trades

    def _compute_metrics(self, ticker: str, period_years: int, trades: list) -> dict:
        """Compute performance metrics from a list of simulated trades."""
        if not trades:
            return self._empty_result(ticker, period_years)

        pnls = [t["pnl"] for t in trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        total_trades = len(trades)
        win_rate = round(len(winners) / total_trades * 100, 2)
        avg_winner = round(float(np.mean(winners)), 2) if winners else 0.0
        avg_loser = round(float(np.mean(losers)), 2) if losers else 0.0

        gross_profit = sum(winners) if winners else 0.0
        gross_loss = abs(sum(losers)) if losers else 0.0
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else None

        total_return_pct = round(sum(pnls) / TRADE_CAPITAL * 100, 2)
        expectancy = round(float(np.mean(pnls)), 2)

        # Sharpe — use trade-level returns as daily proxy
        trade_returns = [p / TRADE_CAPITAL for p in pnls]
        if len(trade_returns) > 1:
            mean_r = np.mean(trade_returns)
            std_r = np.std(trade_returns, ddof=1)
            sharpe = round(float(mean_r / std_r * math.sqrt(252)), 2) if std_r > 0 else 0.0
        else:
            sharpe = 0.0

        # Max drawdown on cumulative PnL series
        cum_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cum_pnl)
        drawdowns = (running_max - cum_pnl) / TRADE_CAPITAL * 100
        max_drawdown_pct = round(float(np.max(drawdowns)), 2) if len(drawdowns) > 0 else 0.0

        hold_days_list = [t["hold_days"] for t in trades]

        return {
            "ticker": ticker,
            "period": f"{period_years} years",
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_winner": avg_winner,
            "avg_loser": avg_loser,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_drawdown_pct,
            "total_return_pct": total_return_pct,
            "expectancy": expectancy,
            "best_trade": round(max(pnls), 2),
            "worst_trade": round(min(pnls), 2),
            "avg_hold_days": round(float(np.mean(hold_days_list)), 1),
            "trades": trades,
        }

    def _empty_result(self, ticker: str, period_years: int) -> dict:
        return {
            "ticker": ticker,
            "period": f"{period_years} years",
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_winner": 0.0,
            "avg_loser": 0.0,
            "profit_factor": None,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "total_return_pct": 0.0,
            "expectancy": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "avg_hold_days": 0.0,
            "trades": [],
        }

    def _save_cache(self, data: dict) -> None:
        try:
            with open(BACKTEST_CACHE_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            logger.warning(f"Could not save backtest cache: {exc}")

    def _load_cache(self, ignore_age: bool = False) -> dict | None:
        if not BACKTEST_CACHE_FILE.exists():
            return None
        try:
            age_days = (time.time() - BACKTEST_CACHE_FILE.stat().st_mtime) / 86400
            if not ignore_age and age_days > CACHE_MAX_AGE_DAYS:
                return None
            with open(BACKTEST_CACHE_FILE) as f:
                return json.load(f)
        except Exception as exc:
            logger.warning(f"Could not load backtest cache: {exc}")
            return None


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_BACKTEST_INSTANCE: BacktestAgent | None = None
_BACKTEST_LOCK = threading.Lock()


def get_backtest_agent() -> BacktestAgent:
    """Return the singleton BacktestAgent instance."""
    global _BACKTEST_INSTANCE
    if _BACKTEST_INSTANCE is None:
        with _BACKTEST_LOCK:
            if _BACKTEST_INSTANCE is None:
                _BACKTEST_INSTANCE = BacktestAgent()
    return _BACKTEST_INSTANCE
