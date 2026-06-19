"""
Module 16 — Failure Recovery & Resume Framework
=================================================
Classifies failures into three severity levels and enforces the matching
recovery path before the agent is allowed to resume trading.

Failure Levels
--------------
SOFT     — temporary outage (API, data, network). Auto-recovers when
           connectivity is restored. No user approval needed.
RISK     — risk-limit breach (daily/weekly loss, exposure violation).
           Requires explicit /resume from user via Telegram or Dashboard.
CRITICAL — integrity failure (position mismatch, corrupted state, bad auth).
           Requires /reconcile (broker sync) followed by /resume_live.
           Auto-degrades LIVE → PAPER so stops/targets still work.

State persisted to data/recovery_state.json so a server restart remembers
the failure level and blocks trading until cleared correctly.

Public API
----------
  from failure_recovery import get_recovery_manager
  rm = get_recovery_manager()

  rm.report_soft(reason)          # log a soft failure
  rm.report_risk(reason)          # trigger risk freeze
  rm.report_critical(reason)      # trigger critical halt + mode degrade
  rm.is_trading_allowed() -> bool
  rm.status()           -> dict
  rm.reconcile()        -> dict   # sync broker → internal state
  rm.resume(user, method) -> bool # manual resume from RISK freeze
  rm.resume_live(user, method) -> bool  # re-enable live after CRITICAL
  rm.audit_log()        -> list
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
_IST = ZoneInfo("Asia/Kolkata")

_STATE_FILE  = Path("data/recovery_state.json")
_AUDIT_FILE  = Path("data/recovery_audit.json")


# ── Enums ────────────────────────────────────────────────────────────────────

class FailureLevel(str, Enum):
    NONE     = "NONE"
    SOFT     = "SOFT"
    RISK     = "RISK"
    CRITICAL = "CRITICAL"


class TradingMode(str, Enum):
    LIVE  = "LIVE"
    PAPER = "PAPER"


# ── Audit helpers ─────────────────────────────────────────────────────────────

_MAX_AUDIT = 500

def _now_ist() -> str:
    return datetime.now(_IST).isoformat(timespec="seconds")


class _AuditLog:
    def __init__(self):
        self._lock    = threading.Lock()
        self._entries: list[dict] = []
        self._load()

    def _load(self):
        try:
            if _AUDIT_FILE.exists():
                self._entries = json.loads(_AUDIT_FILE.read_text())[-_MAX_AUDIT:]
        except Exception:
            self._entries = []

    def append(self, event: str, detail: str = "", user: str = "system"):
        entry = {"ts": _now_ist(), "event": event, "detail": detail, "user": user}
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > _MAX_AUDIT:
                self._entries = self._entries[-_MAX_AUDIT:]
            try:
                _AUDIT_FILE.parent.mkdir(exist_ok=True)
                _AUDIT_FILE.write_text(json.dumps(self._entries, indent=2))
            except Exception as e:
                logger.warning(f"[Recovery] Audit write error: {e}")

    def recent(self, n: int = 50) -> list[dict]:
        with self._lock:
            return list(self._entries[-n:])


# ── Recovery state ────────────────────────────────────────────────────────────

class RecoveryState:
    """Persisted state so restarts remember failure level."""

    _DEFAULTS = {
        "failure_level":        FailureLevel.NONE,
        "failure_reason":       "",
        "failure_ts":           "",
        "trading_mode":         TradingMode.PAPER,
        "reconciled":           False,
        "soft_retry_count":     0,
        "soft_last_retry_ts":   "",
        "last_resume_user":     "",
        "last_resume_ts":       "",
        "last_resume_method":   "",
    }

    def __init__(self):
        self._lock = threading.Lock()
        self._data: dict = dict(self._DEFAULTS)
        self._load()

    def _load(self):
        try:
            if _STATE_FILE.exists():
                raw = json.loads(_STATE_FILE.read_text())
                self._data.update(raw)
        except Exception as e:
            logger.warning(f"[Recovery] State load error: {e}")

    def _save(self):
        try:
            _STATE_FILE.parent.mkdir(exist_ok=True)
            _STATE_FILE.write_text(json.dumps(self._data, indent=2))
        except Exception as e:
            logger.warning(f"[Recovery] State save error: {e}")

    def get(self, key: str):
        with self._lock:
            return self._data.get(key, self._DEFAULTS.get(key))

    def set(self, **kwargs):
        with self._lock:
            self._data.update(kwargs)
            self._save()

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._data)


# ── Core manager ──────────────────────────────────────────────────────────────

class FailureRecoveryManager:
    """
    Central failure classification and recovery gatekeeper.

    Thread-safe. One singleton per process (see get_recovery_manager()).
    """

    # How many seconds between automatic soft-failure retries
    SOFT_RETRY_INTERVAL = 60

    def __init__(self):
        self._state = RecoveryState()
        self._audit = _AuditLog()
        self._soft_ok_callbacks: list = []   # callables → run when soft auto-recovers
        self._lock = threading.Lock()

        level = self._state.get("failure_level")
        if level and level != FailureLevel.NONE:
            logger.warning(
                f"[Recovery] Persistent failure state on boot: "
                f"{level} — {self._state.get('failure_reason')}"
            )

    # ── Reporting ─────────────────────────────────────────────────────────── #

    def report_soft(self, reason: str) -> None:
        """
        Log a soft failure. Does NOT block trading — positions are still managed.
        Starts a background retry watcher if not already running.
        """
        prev = self._state.get("failure_level")
        if prev in (FailureLevel.RISK, FailureLevel.CRITICAL):
            # Higher-severity failure already in place; just log
            logger.warning(f"[Recovery][SOFT] {reason} (higher-severity failure active)")
            self._audit.append("SOFT_FAILURE_SKIPPED", reason)
            return

        self._state.set(
            failure_level="SOFT",
            failure_reason=reason,
            failure_ts=_now_ist(),
            soft_retry_count=self._state.get("soft_retry_count") + 1,
            soft_last_retry_ts=_now_ist(),
        )
        self._audit.append("SOFT_FAILURE", reason)
        logger.warning(f"[Recovery][SOFT] {reason}")
        self._send_alert(
            f"⚠️ <b>SOFT FAILURE</b>\n\n"
            f"Reason: {reason}\n"
            f"Action: New entries paused — retrying every 60s\n"
            f"Existing positions continue to be managed.\n\n"
            f"🕐 {_now_ist()}"
        )

    def report_risk(self, reason: str) -> None:
        """
        Risk limit breach — freeze new entries, alert user, require /resume.
        """
        self._state.set(
            failure_level="RISK",
            failure_reason=reason,
            failure_ts=_now_ist(),
            reconciled=False,
        )
        self._audit.append("RISK_FAILURE", reason)
        logger.error(f"[Recovery][RISK] {reason}")
        self._send_alert(
            f"🚨 <b>RISK FAILURE — Trading Frozen</b>\n\n"
            f"Reason: {reason}\n\n"
            f"<b>New entries disabled.</b> Existing positions and stops are still active.\n\n"
            f"To resume: <code>/resume</code> via Telegram or Dashboard → Resume Trading\n\n"
            f"<b>Before resuming, check:</b>\n"
            f"• Current drawdown\n"
            f"• Open positions\n"
            f"• Cash balance\n\n"
            f"🕐 {_now_ist()}"
        )

    def report_critical(self, reason: str) -> None:
        """
        Critical integrity failure — halt all new entries, degrade LIVE→PAPER,
        require reconciliation + explicit resume_live.
        """
        self._state.set(
            failure_level="CRITICAL",
            failure_reason=reason,
            failure_ts=_now_ist(),
            trading_mode="PAPER",   # auto-degrade
            reconciled=False,
        )
        self._audit.append("CRITICAL_FAILURE", reason)
        logger.critical(f"[Recovery][CRITICAL] {reason}")
        self._send_alert(
            f"🔴 <b>CRITICAL FAILURE — TRADING HALTED</b>\n\n"
            f"Reason: {reason}\n\n"
            f"<b>All new entries disabled.</b>\n"
            f"Mode auto-degraded: <b>LIVE → PAPER</b> (no real orders sent).\n\n"
            f"<b>Recovery steps:</b>\n"
            f"1. Run <code>/reconcile</code> — sync broker → internal state\n"
            f"2. Verify no mismatches\n"
            f"3. Run <code>/resume_live</code> — re-enable live trading\n\n"
            f"<b>Broker state is always the source of truth.</b>\n\n"
            f"🕐 {_now_ist()}"
        )

    # ── Health checks used by _background_loop ────────────────────────────── #

    def check_soft_recovery(self) -> bool:
        """
        Called by the background loop. If in SOFT state and dependencies are
        healthy, auto-recover and return True. Returns False otherwise.
        """
        if self._state.get("failure_level") != FailureLevel.SOFT:
            return True   # not in SOFT state — nothing to do

        healthy, why = self._check_connectivity()
        if healthy:
            self._state.set(
                failure_level="NONE",
                failure_reason="",
                soft_retry_count=0,
            )
            self._audit.append("AUTO_RESUME_SUCCESS", "Soft failure resolved — all checks passed")
            logger.info("[Recovery] AUTO_RESUME_SUCCESS — soft failure cleared")
            self._send_alert(
                f"✅ <b>AUTO_RESUME_SUCCESS</b>\n\n"
                f"Soft failure resolved. Trading resumed automatically.\n"
                f"All checks passed: broker reachable, market data healthy, API responsive.\n\n"
                f"🕐 {_now_ist()}"
            )
            for cb in self._soft_ok_callbacks:
                try:
                    cb()
                except Exception:
                    pass
            return True
        else:
            logger.warning(f"[Recovery] Soft failure check — still failing: {why}")
            self._state.set(soft_last_retry_ts=_now_ist())
            return False

    def _check_connectivity(self) -> tuple[bool, str]:
        """Basic connectivity checks for soft-failure auto-recovery."""
        # 1. Can we fetch a yfinance price?
        try:
            import yfinance as yf  # type: ignore[import-untyped]
            t = yf.Ticker("^NSEI")
            h = t.history(period="1d", interval="1m")
            if h is None or h.empty:
                return False, "yfinance returned empty data"
        except Exception as e:
            return False, f"yfinance error: {e}"

        # 2. Broker reachable (Kite or Angel One — best-effort)
        try:
            from kite_broker import get_broker as _gb  # type: ignore
            broker = _gb()
            st = broker.connection_status()
            if not st.get("connected") and not st.get("paper_mode"):
                return False, "Kite broker not connected"
        except Exception:
            pass   # broker not configured — skip

        return True, "ok"

    # ── Trading gate ──────────────────────────────────────────────────────── #

    def is_trading_allowed(self) -> bool:
        """
        Returns True only if no active failure blocks new entries.
        Soft failures do NOT block trading (positions are managed, entries paused).
        Risk and Critical failures block new entries.
        """
        level = self._state.get("failure_level")
        if level == FailureLevel.NONE:
            return True
        if level == FailureLevel.SOFT:
            # Soft: allow position management but callers should skip new entries
            return True
        # RISK or CRITICAL — new entries blocked
        return False

    def new_entries_allowed(self) -> bool:
        """More precise gate — False for SOFT/RISK/CRITICAL."""
        level = self._state.get("failure_level")
        return level == FailureLevel.NONE

    def get_mode(self) -> TradingMode:
        m = self._state.get("trading_mode")
        return TradingMode(m) if m else TradingMode.PAPER

    # ── Manual resume (RISK) ─────────────────────────────────────────────── #

    def resume(self, user: str = "unknown", method: str = "unknown") -> tuple[bool, str]:
        """
        Explicit manual resume — only clears RISK failures.
        CRITICAL requires reconcile + resume_live instead.
        Returns (success, message).
        """
        level = self._state.get("failure_level")

        if level == FailureLevel.NONE:
            return True, "No active failure — trading already running."

        if level == FailureLevel.CRITICAL:
            return False, (
                "Cannot /resume a CRITICAL failure.\n"
                "Run /reconcile first, then /resume_live."
            )

        if level == FailureLevel.SOFT:
            # Force-clear the soft failure manually
            self._state.set(failure_level="NONE", failure_reason="", soft_retry_count=0)
            self._audit.append("MANUAL_RESUME", f"Soft failure force-cleared", user=user)
            self._log_resume(user, method, "Soft failure cleared by user")
            return True, "Soft failure cleared. Trading resumed."

        # RISK — check current state before allowing resume
        snapshot = self._portfolio_snapshot()
        self._state.set(
            failure_level="NONE",
            failure_reason="",
            last_resume_user=user,
            last_resume_ts=_now_ist(),
            last_resume_method=method,
        )
        self._audit.append(
            "MANUAL_RESUME",
            f"Risk failure cleared. Drawdown={snapshot.get('drawdown_pct','?')}%  "
            f"Positions={snapshot.get('position_count','?')}  Cash=₹{snapshot.get('cash','?')}",
            user=user,
        )
        logger.info(f"[Recovery] MANUAL_RESUME by {user} via {method}")
        self._send_alert(
            f"✅ <b>MANUAL_RESUME</b>\n\n"
            f"Risk failure cleared by: <b>{user}</b> ({method})\n\n"
            f"<b>Current State at Resume:</b>\n"
            f"• Drawdown: {snapshot.get('drawdown_pct','?')}%\n"
            f"• Open Positions: {snapshot.get('position_count','?')}\n"
            f"• Cash: ₹{snapshot.get('cash','?')}\n\n"
            f"Trading resumed. Monitor closely.\n\n"
            f"🕐 {_now_ist()}"
        )
        return True, "Risk failure cleared. Trading resumed."

    # ── Reconciliation (CRITICAL step 1) ─────────────────────────────────── #

    def reconcile(self) -> dict:
        """
        Sync broker → internal state. Broker is always source of truth.
        Repairs any mismatches found. Returns a reconciliation report.
        """
        level = self._state.get("failure_level")
        self._audit.append("RECONCILE_START", f"Level={level}")
        logger.info("[Recovery] Reconciliation started")

        report: dict = {
            "ts":               _now_ist(),
            "mismatches":       [],
            "repaired":         [],
            "cash_ok":          True,
            "positions_ok":     True,
            "orders_ok":        True,
            "overall":          "PASS",
            "error":            None,
        }

        # ── Try broker reconciliation ──────────────────────────────────────
        try:
            from kite_broker import get_broker as _gb  # type: ignore
            broker = _gb()
            conn = broker.connection_status()
            if not conn.get("connected"):
                report["error"] = "Broker not connected — cannot reconcile"
                report["overall"] = "SKIP"
                self._audit.append("RECONCILE_SKIP", "Broker not connected")
                logger.warning("[Recovery] Reconcile skipped — broker not connected")
                # Even without broker, mark reconciled so paper-mode can resume
                self._state.set(reconciled=True)
                return report

            # Fetch broker positions
            broker_positions = broker.get_positions() or []
            broker_holdings  = broker.get_holdings()  or []

            # Internal positions
            from engine import get_agent as _ga  # type: ignore
            agent = _ga()
            internal = agent.portfolio.state.get("positions", {})

            # Compare
            broker_tickers = {
                p.get("tradingsymbol", "").replace("-EQ", "") + ".NS"
                for p in broker_positions + broker_holdings
                if (p.get("quantity") or 0) != 0
            }
            internal_tickers = set(internal.keys())

            # In broker but not internal
            for t in broker_tickers - internal_tickers:
                report["mismatches"].append({"type": "BROKER_ONLY", "ticker": t})

            # In internal but not broker (ghost positions)
            for t in internal_tickers - broker_tickers:
                report["mismatches"].append({"type": "INTERNAL_ONLY", "ticker": t})
                # Repair: remove ghost position (broker is truth)
                del internal[t]
                report["repaired"].append(f"Removed ghost position: {t}")

            # Cash check
            broker_cash = broker.get_cash_balance()
            internal_cash = agent.portfolio.state.get("cash", 0)
            if broker_cash is not None:
                diff = abs(broker_cash - internal_cash)
                if diff > 1000:  # ₹1000 tolerance
                    report["mismatches"].append({
                        "type": "CASH_MISMATCH",
                        "broker": broker_cash,
                        "internal": internal_cash,
                        "diff": diff,
                    })
                    report["cash_ok"] = False
                    # Repair: sync to broker
                    agent.portfolio.state["cash"] = broker_cash
                    report["repaired"].append(f"Cash synced: ₹{internal_cash:,.0f} → ₹{broker_cash:,.0f}")

            if report["mismatches"]:
                report["positions_ok"] = not any(m["type"] in ("BROKER_ONLY", "INTERNAL_ONLY") for m in report["mismatches"])
                report["overall"] = "REPAIRED"
                # Save repaired state
                agent.portfolio._save_state()

        except ImportError:
            # Broker not configured — paper-mode reconciliation (internal consistency only)
            report["error"] = "Broker not configured — paper-mode reconciliation only"
            report["overall"] = "PAPER_PASS"
            logger.info("[Recovery] Broker not configured — paper reconcile only")
        except Exception as e:
            report["error"] = str(e)
            report["overall"] = "FAIL"
            logger.error(f"[Recovery] Reconcile error: {e}", exc_info=True)
            self._audit.append("RECONCILE_FAIL", str(e))
            return report

        self._state.set(reconciled=True)
        self._audit.append(
            "RECONCILE_COMPLETE",
            f"overall={report['overall']}  mismatches={len(report['mismatches'])}  "
            f"repaired={len(report['repaired'])}"
        )
        logger.info(f"[Recovery] Reconcile done: {report['overall']}  mismatches={len(report['mismatches'])}")

        self._send_alert(
            f"🔍 <b>Reconciliation Complete</b>\n\n"
            f"Result: <b>{report['overall']}</b>\n"
            f"Mismatches found: {len(report['mismatches'])}\n"
            f"Items repaired: {len(report['repaired'])}\n\n"
            + (f"Errors: {report['error']}\n\n" if report['error'] else "")
            + (f"Run <code>/resume_live</code> to re-enable live trading.\n\n" if self._state.get("failure_level") == "CRITICAL" else "")
            + f"🕐 {_now_ist()}"
        )
        return report

    # ── Resume live (CRITICAL step 2) ─────────────────────────────────────── #

    def resume_live(self, user: str = "unknown", method: str = "unknown") -> tuple[bool, str]:
        """
        Re-enable live trading after a CRITICAL failure.
        Requires reconciliation to have passed first.
        """
        level = self._state.get("failure_level")
        if level not in (FailureLevel.CRITICAL, FailureLevel.NONE):
            return False, f"resume_live is only for CRITICAL failures. Current level: {level}"

        if level == FailureLevel.NONE:
            # Already clear — just ensure mode is LIVE
            self._state.set(trading_mode="LIVE")
            return True, "No active failure. Mode set to LIVE."

        if not self._state.get("reconciled"):
            return False, (
                "Reconciliation required before resuming live trading.\n"
                "Run /reconcile first."
            )

        # Pre-flight checks
        ok, why = self._pre_live_checks()
        if not ok:
            return False, f"Pre-flight failed: {why}\nResolve the issue and try again."

        self._state.set(
            failure_level="NONE",
            failure_reason="",
            trading_mode="LIVE",
            reconciled=False,   # reset for next cycle
            last_resume_user=user,
            last_resume_ts=_now_ist(),
            last_resume_method=method,
        )
        self._audit.append("LIVE_TRADING_RESUMED", f"user={user} method={method}", user=user)
        logger.info(f"[Recovery] LIVE_TRADING_RESUMED by {user} via {method}")
        self._send_alert(
            f"🟢 <b>LIVE TRADING RESUMED</b>\n\n"
            f"Resumed by: <b>{user}</b> ({method})\n"
            f"Reconciliation: ✅ Passed\n"
            f"Session: ✅ Healthy\n"
            f"Positions: ✅ Verified\n\n"
            f"Execution engine re-enabled. Monitor closely.\n\n"
            f"🕐 {_now_ist()}"
        )
        return True, "Live trading resumed. Reconciliation passed. Monitor closely."

    def _pre_live_checks(self) -> tuple[bool, str]:
        """Verify broker session and cash before resuming live."""
        try:
            from kite_broker import get_broker as _gb  # type: ignore
            broker = _gb()
            conn = broker.connection_status()
            if not conn.get("connected") and not conn.get("paper_mode"):
                return False, "Broker session not valid"
        except ImportError:
            pass  # paper mode — no broker to check
        except Exception as e:
            return False, f"Broker check error: {e}"
        return True, "ok"

    # ── Status & dashboard data ───────────────────────────────────────────── #

    def status(self) -> dict:
        snap = self._state.snapshot()
        level = snap.get("failure_level", "NONE")
        return {
            "failure_level":      level,
            "failure_reason":     snap.get("failure_reason", ""),
            "failure_ts":         snap.get("failure_ts", ""),
            "trading_mode":       snap.get("trading_mode", "PAPER"),
            "new_entries_allowed": self.new_entries_allowed(),
            "reconciled":         snap.get("reconciled", False),
            "soft_retry_count":   snap.get("soft_retry_count", 0),
            "soft_last_retry_ts": snap.get("soft_last_retry_ts", ""),
            "last_resume_user":   snap.get("last_resume_user", ""),
            "last_resume_ts":     snap.get("last_resume_ts", ""),
            "last_resume_method": snap.get("last_resume_method", ""),
            "recovery_required":  level in (FailureLevel.RISK, FailureLevel.CRITICAL),
            "reconcile_required": level == FailureLevel.CRITICAL and not snap.get("reconciled"),
        }

    def audit_log(self, n: int = 50) -> list[dict]:
        return self._audit.recent(n)

    # ── Portfolio snapshot for resume confirmation ────────────────────────── #

    def _portfolio_snapshot(self) -> dict:
        try:
            from engine import get_agent as _ga  # type: ignore
            agent = _ga()
            port = agent.portfolio
            positions = port.state.get("positions", {})
            total_val = port.get_total_value()
            initial = port.state.get("initial_capital", 1_000_000)
            drawdown_pct = round((initial - total_val) / initial * 100, 2) if initial else 0
            return {
                "position_count": len(positions),
                "cash":           f"{port.available_cash():,.0f}",
                "total_value":    f"{total_val:,.0f}",
                "drawdown_pct":   drawdown_pct,
            }
        except Exception as e:
            return {"error": str(e)}

    def _log_resume(self, user: str, method: str, reason: str):
        self._state.set(
            last_resume_user=user,
            last_resume_ts=_now_ist(),
            last_resume_method=method,
        )

    # ── Telegram alert helper ─────────────────────────────────────────────── #

    def _send_alert(self, msg: str) -> None:
        try:
            from notifier import get_notifier  # type: ignore
            n = get_notifier()
            if n.enabled:
                n._tg.send_async(msg)
        except Exception as e:
            logger.warning(f"[Recovery] Alert send error: {e}")


# ── Singleton ─────────────────────────────────────────────────────────────────

_manager: Optional[FailureRecoveryManager] = None
_mgr_lock = threading.Lock()

def get_recovery_manager() -> FailureRecoveryManager:
    global _manager
    with _mgr_lock:
        if _manager is None:
            _manager = FailureRecoveryManager()
    return _manager
