"""
Slippage Tracker
================
Tracks execution quality — the difference between signal price and actual
fill price. In paper trading slippage is always zero; the framework is
ready to be wired into a live broker feed that supplies actual fill prices.
"""

import json
import threading
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TRADE_LOG_PATH = DATA_DIR / "trade_log.json"


def _load_trade_log() -> list:
    try:
        if TRADE_LOG_PATH.exists():
            with open(TRADE_LOG_PATH, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load trade_log.json: {e}")
    return []


class SlippageTracker:
    """
    Analyse execution quality from the trade log.

    In paper trading every fill is at the signal price, so slippage is 0.
    In live trading, pass a `fills` dict {trade_id: actual_fill_price} to
    override the signal price with the real fill price.
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------
    def analyse(self, trades: list = None, fills: dict = None) -> dict:
        """
        Analyse slippage for BUY trades that have a `signal_score` field
        (or for all BUY trades when signal_score is absent).

        Parameters
        ----------
        trades : list, optional
            Trade records. Defaults to loading trade_log.json.
        fills : dict, optional
            Map of trade id -> actual fill price for live trading.
            In paper trading leave as None (slippage will be 0).

        Returns
        -------
        dict  matching the documented schema.
        """
        if trades is None:
            trades = _load_trade_log()
        if fills is None:
            fills = {}

        by_strategy: dict = {}
        by_ticker: dict = {}
        worst_trades: list = []

        total_slippage_cost = 0.0
        total_slippage_pct_sum = 0.0
        trade_count = 0

        buy_trades = [t for t in trades if t.get("action", "").upper() == "BUY"]

        for t in buy_trades:
            trade_id = t.get("id")
            signal_price = float(t.get("price", 0))
            if signal_price == 0:
                continue

            # Actual fill: from fills map or same as signal (paper trading)
            actual_price = float(fills.get(trade_id, signal_price))

            slippage = actual_price - signal_price  # positive = worse fill
            slippage_pct = (slippage / signal_price * 100) if signal_price else 0.0
            qty = int(t.get("qty", 0))
            slippage_cost = slippage * qty

            ticker = t.get("ticker", "UNKNOWN")
            strat = t.get("strategy", "UNKNOWN")

            total_slippage_cost += slippage_cost
            total_slippage_pct_sum += slippage_pct
            trade_count += 1

            # By strategy
            if strat not in by_strategy:
                by_strategy[strat] = {"slippage_pct_sum": 0.0, "trade_count": 0}
            by_strategy[strat]["slippage_pct_sum"] += slippage_pct
            by_strategy[strat]["trade_count"] += 1

            # By ticker
            if ticker not in by_ticker:
                by_ticker[ticker] = {"slippage_pct_sum": 0.0, "trade_count": 0}
            by_ticker[ticker]["slippage_pct_sum"] += slippage_pct
            by_ticker[ticker]["trade_count"] += 1

            # Collect worst slippage trades (top candidates — filtered later)
            worst_trades.append({
                "id": trade_id,
                "ticker": ticker,
                "strategy": strat,
                "signal_price": signal_price,
                "actual_price": actual_price,
                "slippage_pct": round(slippage_pct, 4),
                "slippage_cost": round(slippage_cost, 2),
                "time": t.get("time", ""),
            })

        avg_slippage_pct = (total_slippage_pct_sum / trade_count) if trade_count else 0.0

        # Summarise by_strategy
        strategy_summary = {
            strat: {
                "avg_slippage_pct": round(v["slippage_pct_sum"] / v["trade_count"], 4),
                "trade_count": v["trade_count"],
            }
            for strat, v in by_strategy.items()
        }

        # Summarise by_ticker
        ticker_summary = {
            ticker: {
                "avg_slippage_pct": round(v["slippage_pct_sum"] / v["trade_count"], 4),
                "trade_count": v["trade_count"],
            }
            for ticker, v in by_ticker.items()
        }

        # Worst slippage trades: sort by slippage_pct descending, keep top 5
        worst_trades_sorted = sorted(worst_trades, key=lambda x: x["slippage_pct"], reverse=True)[:5]

        # Execution quality score: 100 - (avg_slippage_pct * 20), clamped [0, 100]
        quality_score = max(0, min(100, round(100 - abs(avg_slippage_pct) * 20)))

        is_paper = not fills  # if no fills map provided, assume paper trading

        return {
            "avg_slippage_pct": round(avg_slippage_pct, 4),
            "total_slippage_cost": round(total_slippage_cost, 2),
            "by_strategy": strategy_summary,
            "by_ticker": ticker_summary,
            "worst_slippage_trades": worst_trades_sorted,
            "execution_quality_score": quality_score,
            "trade_count_analysed": trade_count,
            "note": (
                "Paper trading: slippage tracking ready for live execution"
                if is_paper
                else "Live trading: slippage from actual fill prices"
            ),
        }

    # ------------------------------------------------------------------
    # Quality label
    # ------------------------------------------------------------------
    def get_execution_quality(self, avg_slippage_pct: float = None) -> str:
        """
        Return a quality label based on average slippage percentage.

        Thresholds (absolute slippage %):
            EXCELLENT: < 0.05%
            GOOD:      0.05% – 0.20%
            FAIR:      0.20% – 0.50%
            POOR:      > 0.50%
        """
        if avg_slippage_pct is None:
            result = self.analyse()
            avg_slippage_pct = result["avg_slippage_pct"]

        abs_slip = abs(avg_slippage_pct)
        if abs_slip < 0.05:
            return "EXCELLENT"
        elif abs_slip < 0.20:
            return "GOOD"
        elif abs_slip < 0.50:
            return "FAIR"
        else:
            return "POOR"

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------
    def get_dashboard_data(self) -> dict:
        """Run analyse() on current trade_log.json and add quality label."""
        result = self.analyse()
        result["execution_quality"] = self.get_execution_quality(result["avg_slippage_pct"])
        return result


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_SLIPPAGE_AGENT_INSTANCE = None
_SLIPPAGE_AGENT_LOCK = threading.Lock()


def get_slippage_agent() -> SlippageTracker:
    global _SLIPPAGE_AGENT_INSTANCE
    if _SLIPPAGE_AGENT_INSTANCE is None:
        with _SLIPPAGE_AGENT_LOCK:
            if _SLIPPAGE_AGENT_INSTANCE is None:
                _SLIPPAGE_AGENT_INSTANCE = SlippageTracker()
    return _SLIPPAGE_AGENT_INSTANCE
