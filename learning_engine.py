"""
Learning Engine — Self-Improving Strategy Weights
==================================================
Tracks per-strategy win rates and adjusts weight multipliers so that
strategies that consistently produce profitable trades get more influence
and strategies that produce losing trades get less.

State is persisted to  data/learning_state.json  so it survives restarts.

Usage (from engine.py):
    from learning_engine import get_learning_engine

    learning = get_learning_engine()

    # Apply weights when building composite strength:
    weighted = learning.weighted_strength(agg["strengths"], agg["strategies"])
    threshold = learning.get_threshold()

    # At end of cycle:
    changes = learning.learn_from_trades()
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_IST_TZ = ZoneInfo("Asia/Kolkata")

# ── Constants ─────────────────────────────────────────────────────────────────

_STATE_FILE   = Path("data/learning_state.json")
_TRADE_LOG    = Path("data/trade_log.json")

# Strategy weight bounds
_MIN_WEIGHT   = 0.3
_MAX_WEIGHT   = 2.0
_DEFAULT_WEIGHT = 1.0

# Smoothing factor for exponential weight update (0 = never change, 1 = jump instantly)
_ALPHA = 0.15

# Buy-threshold bounds and default
_MIN_THRESHOLD = 55
_MAX_THRESHOLD = 85
_DEFAULT_THRESHOLD = 65

# Minimum trades per strategy before we trust the win rate
_MIN_TRADES_TO_LEARN = 3

# How many trades to look back when computing recent performance
_LOOKBACK_TRADES = 200


def _now_ist() -> datetime:
    return datetime.now(_IST_TZ)


def _empty_strategy_rec() -> dict:
    """Default record for a strategy that has never been seen before."""
    return {
        "weight":           _DEFAULT_WEIGHT,
        "wins":             0,
        "losses":           0,
        "tp_hits":          0,
        "sl_hits":          0,
        "total_pnl":        0.0,
        "win_rate":         0.0,
        "sum_rr_wins":      0.0,
        "sum_rr_losses":    0.0,
        "sum_planned_rr":   0.0,
        "planned_rr_count": 0,
        "avg_win_rr":       0.0,
        "avg_loss_rr":      0.0,
        "avg_planned_rr":   0.0,
        "avg_actual_rr":    0.0,
        "expected_value":   0.0,
    }


# ── LearningEngine ────────────────────────────────────────────────────────────

class LearningEngine:
    """
    Maintains per-strategy statistics and adapts signal weights + buy threshold.

    State schema (data/learning_state.json):
    {
        "threshold": 65,
        "overall_win_rate": 0.0,
        "total_closed_trades": 0,
        "strategies": {
            "SMA": {
                "weight": 1.0,
                "wins": 0,
                "losses": 0,
                "total_pnl": 0.0,
                "win_rate": 0.0
            },
            ...
        },
        "adjustment_log": [
            {"time": "...", "strategy": "SMA", "old_weight": 1.0, "new_weight": 1.1,
             "win_rate": 0.62, "reason": "..."},
            ...
        ],
        "last_processed_trade_index": 0
    }
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict = self._default_state()
        self._load()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_weight(self, strategy: str) -> float:
        """Return the current weight multiplier for a strategy (default 1.0)."""
        with self._lock:
            return self._state["strategies"].get(strategy, {}).get("weight", _DEFAULT_WEIGHT)

    def get_threshold(self) -> int:
        """Return the current dynamic buy-signal threshold."""
        with self._lock:
            return int(self._state.get("threshold", _DEFAULT_THRESHOLD))

    def weighted_strength(self, strengths: list[float], strategies: list[str]) -> float:
        """
        Given parallel lists of raw signal strengths and strategy names,
        return a weighted composite strength (0–100).

        If a strategy appears multiple times (different signals from same strategy),
        its weight is applied to each occurrence.
        """
        if not strengths:
            return 0.0

        with self._lock:
            total_w = 0.0
            total_ws = 0.0
            for s, strat in zip(strengths, strategies):
                w = self._state["strategies"].get(strat, {}).get("weight", _DEFAULT_WEIGHT)
                total_ws += s * w
                total_w  += w

        if total_w == 0:
            return 0.0

        raw = total_ws / total_w
        # Bonus: +5 per additional unique confirming strategy (same as before)
        unique_strats = len(set(strategies))
        boosted = min(raw + (unique_strats - 1) * 5, 100.0)
        return round(boosted, 2)

    def learn_from_trades(self) -> dict:
        """
        Read the trade log, process new closed trades, update per-strategy stats
        and weights, adjust the buy threshold.

        Weight formula is based on Expected Value (EV), not just win rate:
            EV = win_rate × avg_RR_on_wins  −  loss_rate × avg_RR_on_losses
        This means a strategy with 40% win rate but 1:3 RR (EV = 0.4×3 − 0.6×1 = 0.6)
        scores BETTER than one with 55% win rate but only 1:1 RR (EV = 0.55−0.45 = 0.1).
        The goal is to keep pushing actual RR toward 1:2 minimum.

        Returns a summary dict with weight_changes, threshold_change, overall stats.
        """
        summary: dict = {
            "new_trades_processed": 0,
            "weight_changes":       [],
            "threshold_change":     None,
            "overall_win_rate":     0.0,
            "overall_avg_rr":       0.0,
        }

        trades = self._read_trade_log()
        if not trades:
            return summary

        with self._lock:
            last_idx  = self._state.get("last_processed_trade_index", 0)
            closed    = [t for t in trades if t.get("pnl") is not None]
            new_closed = closed[last_idx:]

            if not new_closed:
                summary["overall_win_rate"] = self._state.get("overall_win_rate", 0.0)
                summary["overall_avg_rr"]   = self._state.get("overall_avg_rr",   0.0)
                return summary

            # ── Update per-strategy stats ──────────────────────────────────
            for trade in new_closed:
                pnl        = float(trade.get("pnl", 0))
                strategy   = trade.get("strategy", "UNKNOWN")
                actual_rr  = trade.get("actual_rr")   # float or None
                planned_rr = trade.get("planned_rr")  # float or None
                reason     = trade.get("reason", "")
                is_win     = pnl > 0
                is_tp      = reason == "TAKE_PROFIT"
                is_sl      = reason == "STOP_LOSS"

                strats = [s.strip() for s in strategy.split("+") if s.strip()]
                if not strats:
                    strats = ["UNKNOWN"]

                for strat in strats:
                    rec = self._state["strategies"].setdefault(strat, _empty_strategy_rec())

                    # Win / loss counts
                    if is_win:
                        rec["wins"] += 1
                    else:
                        rec["losses"] += 1

                    # TP / SL hit counts
                    if is_tp:
                        rec["tp_hits"] += 1
                    elif is_sl:
                        rec["sl_hits"] += 1

                    # Accumulate actual RR for wins and losses separately
                    if actual_rr is not None:
                        rr = float(actual_rr)
                        if is_win:
                            rec["sum_rr_wins"]   = round(rec.get("sum_rr_wins", 0.0)   + rr, 4)
                        else:
                            rec["sum_rr_losses"] = round(rec.get("sum_rr_losses", 0.0) + abs(rr), 4)

                    # Planned RR tracking
                    if planned_rr is not None:
                        rec["sum_planned_rr"]    = round(rec.get("sum_planned_rr", 0.0) + float(planned_rr), 4)
                        rec["planned_rr_count"]  = rec.get("planned_rr_count", 0) + 1

                    rec["total_pnl"] = round(rec.get("total_pnl", 0.0) + pnl, 2)
                    total            = rec["wins"] + rec["losses"]
                    rec["win_rate"]  = round(rec["wins"] / total, 4) if total else 0.0

                    # Derived averages
                    wins_cnt   = max(rec["wins"], 1)
                    losses_cnt = max(rec["losses"], 1)
                    total_cnt  = max(total, 1)
                    avg_win_rr    = round(rec.get("sum_rr_wins",   0.0) / wins_cnt,   3)
                    avg_loss_rr   = round(rec.get("sum_rr_losses", 0.0) / losses_cnt, 3)
                    avg_planned   = round(rec.get("sum_planned_rr", 0.0) / max(rec.get("planned_rr_count", 1), 1), 3)
                    # Overall avg actual RR across all trades
                    sum_all_rr    = rec.get("sum_rr_wins", 0.0) - rec.get("sum_rr_losses", 0.0)
                    avg_actual_rr = round(sum_all_rr / total_cnt, 3)

                    rec["avg_win_rr"]    = avg_win_rr
                    rec["avg_loss_rr"]   = avg_loss_rr
                    rec["avg_planned_rr"] = avg_planned
                    rec["avg_actual_rr"] = avg_actual_rr

                    # Expected Value per trade in units of risk
                    # EV > 0 means the strategy makes money in the long run
                    ev = round(rec["win_rate"] * avg_win_rr - (1 - rec["win_rate"]) * avg_loss_rr, 4)
                    rec["expected_value"] = ev

            summary["new_trades_processed"] = len(new_closed)
            self._state["last_processed_trade_index"] = len(closed)
            self._state["total_closed_trades"]         = len(closed)

            # ── Overall stats from the last LOOKBACK window ────────────────
            recent = closed[-_LOOKBACK_TRADES:]
            wins   = sum(1 for t in recent if float(t.get("pnl", 0)) > 0)
            overall_wr = round(wins / len(recent), 4) if recent else 0.0

            # Overall avg actual RR across recent closed trades
            rr_vals = [float(t["actual_rr"]) for t in recent if t.get("actual_rr") is not None]
            overall_avg_rr = round(sum(rr_vals) / len(rr_vals), 3) if rr_vals else 0.0

            self._state["overall_win_rate"] = overall_wr
            self._state["overall_avg_rr"]   = overall_avg_rr
            summary["overall_win_rate"]     = overall_wr
            summary["overall_avg_rr"]       = overall_avg_rr

            # ── Update strategy weights using EV ───────────────────────────
            for strat, rec in self._state["strategies"].items():
                total = rec["wins"] + rec["losses"]
                if total < _MIN_TRADES_TO_LEARN:
                    continue

                ev       = rec.get("expected_value", 0.0)
                win_rate = rec["win_rate"]
                avg_win_rr = rec.get("avg_win_rr", 0.0)

                # Map EV → target weight
                # EV ≤ -0.5 → 0.3  (consistently losing money per risk unit)
                # EV = 0    → 0.65 (breakeven — slight discount for uncertainty)
                # EV = 1.0  → 1.0  (making 1R per trade on average)
                # EV = 2.0  → 1.5  (making 2R per trade — 1:2 RR target met)
                # EV ≥ 3.0  → 2.0  (exceptional)
                if ev <= -0.5:
                    target = _MIN_WEIGHT
                elif ev <= 0:
                    target = 0.65 + (ev + 0.5) * (0.35 / 0.5)   # 0.3 → 0.65
                elif ev <= 1.0:
                    target = 0.65 + ev * (0.35 / 1.0)             # 0.65 → 1.0
                elif ev <= 2.0:
                    target = 1.0  + (ev - 1.0) * (0.5 / 1.0)     # 1.0  → 1.5
                else:
                    target = min(1.5 + (ev - 2.0) * (0.5 / 1.0), _MAX_WEIGHT)  # 1.5 → 2.0

                target = round(max(_MIN_WEIGHT, min(_MAX_WEIGHT, target)), 4)

                old_weight = rec["weight"]
                new_weight = round(old_weight + _ALPHA * (target - old_weight), 4)
                new_weight = max(_MIN_WEIGHT, min(_MAX_WEIGHT, new_weight))

                if abs(new_weight - old_weight) >= 0.01:
                    rec["weight"] = new_weight
                    rr_note = f"avg_win_RR={avg_win_rr:.2f}" if avg_win_rr else ""
                    reason  = (
                        f"EV={ev:.3f} (WR={win_rate:.1%}, {rr_note}) "
                        f"→ target_w={target:.2f}, smoothed to {new_weight:.2f}"
                    )
                    log_entry = {
                        "time":       _now_ist().strftime("%Y-%m-%d %H:%M:%S IST"),
                        "strategy":   strat,
                        "old_weight": old_weight,
                        "new_weight": new_weight,
                        "win_rate":   win_rate,
                        "ev":         ev,
                        "avg_win_rr": avg_win_rr,
                        "trades":     total,
                        "reason":     reason,
                    }
                    adj_log = self._state.setdefault("adjustment_log", [])
                    adj_log.append(log_entry)
                    if len(adj_log) > 500:
                        self._state["adjustment_log"] = adj_log[-500:]

                    summary["weight_changes"].append({
                        "strategy":   strat,
                        "old":        old_weight,
                        "new":        new_weight,
                        "win_rate":   win_rate,
                        "ev":         ev,
                        "avg_win_rr": avg_win_rr,
                    })
                    logger.info(
                        f"[Learning] {strat} weight {old_weight:.2f}→{new_weight:.2f} | {reason}"
                    )

            # ── Adjust buy threshold (uses EV + win rate + RR) ─────────────
            old_threshold = int(self._state.get("threshold", _DEFAULT_THRESHOLD))
            new_threshold = old_threshold
            reason        = None

            # Primary driver: overall EV
            # Secondary: win rate to catch cases with few RR data points yet
            if overall_avg_rr > 0 and overall_avg_rr < 0.8:
                # RR is terrible — we're getting stopped out for small gains/losses
                new_threshold = min(old_threshold + 3, _MAX_THRESHOLD)
                reason = f"avg_RR={overall_avg_rr:.2f} < 0.8 — raising threshold aggressively"
            elif overall_avg_rr > 0 and overall_avg_rr < 1.5:
                new_threshold = min(old_threshold + 2, _MAX_THRESHOLD)
                reason = f"avg_RR={overall_avg_rr:.2f} < 1.5 (below 1:2 target) — raising threshold"
            elif overall_wr < 0.40:
                new_threshold = min(old_threshold + 2, _MAX_THRESHOLD)
                reason = f"win_rate={overall_wr:.1%} < 40% — raising threshold"
            elif overall_wr < 0.50:
                new_threshold = min(old_threshold + 1, _MAX_THRESHOLD)
                reason = f"win_rate={overall_wr:.1%} < 50% — raising threshold"
            elif overall_avg_rr >= 2.0 and overall_wr >= 0.55:
                # Consistently hitting 1:2+ and winning — allow more trades
                new_threshold = max(old_threshold - 1, _MIN_THRESHOLD)
                reason = f"avg_RR={overall_avg_rr:.2f} ≥ 2.0 + win_rate={overall_wr:.1%} ≥ 55% — lowering threshold"
            elif overall_wr > 0.65:
                new_threshold = max(old_threshold - 1, _MIN_THRESHOLD)
                reason = f"win_rate={overall_wr:.1%} > 65% — lowering threshold slightly"

            if new_threshold != old_threshold and reason:
                self._state["threshold"] = new_threshold
                summary["threshold_change"] = {"old": old_threshold, "new": new_threshold}
                logger.info(f"[Learning] Buy threshold {old_threshold}→{new_threshold} | {reason}")

        self._save()
        return summary

    def get_state_snapshot(self) -> dict:
        """Return a serialisable snapshot of the current learning state for the API."""
        with self._lock:
            import copy
            snap = copy.deepcopy(self._state)
        # Make adjustment_log human-readable (most recent 50)
        snap["adjustment_log"] = snap.get("adjustment_log", [])[-50:]
        return snap

    def reset(self) -> None:
        """Reset all learning state to defaults (useful for testing)."""
        with self._lock:
            self._state = self._default_state()
        self._save()
        logger.info("[Learning] State reset to defaults")

    # ── Internal ───────────────────────────────────────────────────────────────

    @staticmethod
    def _default_state() -> dict:
        return {
            "threshold":               _DEFAULT_THRESHOLD,
            "overall_win_rate":        0.0,
            "overall_avg_rr":          0.0,
            "total_closed_trades":     0,
            "strategies":              {},
            "adjustment_log":          [],
            "last_processed_trade_index": 0,
        }

    def _load(self) -> None:
        try:
            if _STATE_FILE.exists():
                with open(_STATE_FILE) as f:
                    loaded = json.load(f)
                # Merge loaded state over defaults (handles missing keys gracefully)
                for k, v in loaded.items():
                    self._state[k] = v
                logger.info(
                    f"[Learning] State loaded from {_STATE_FILE} "
                    f"({len(self._state['strategies'])} strategies, "
                    f"threshold={self._state['threshold']})"
                )
        except Exception as exc:
            logger.warning(f"[Learning] Could not load state ({exc}) — starting fresh")

    def _save(self) -> None:
        try:
            _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = self._state.copy()
            tmp = str(_STATE_FILE) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, _STATE_FILE)
        except Exception as exc:
            logger.error(f"[Learning] Could not save state: {exc}")

    @staticmethod
    def _read_trade_log() -> list[dict]:
        try:
            if not _TRADE_LOG.exists():
                return []
            with open(_TRADE_LOG) as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception as exc:
            logger.debug(f"[Learning] Could not read trade log: {exc}")
            return []


# ── Module-level singleton ─────────────────────────────────────────────────────

_engine: Optional[LearningEngine] = None
_engine_lock = threading.Lock()


def get_learning_engine() -> LearningEngine:
    """Return the module-level LearningEngine singleton."""
    global _engine  # noqa: PLW0603
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = LearningEngine()
    return _engine
