"""
Trading Configuration Manager
==============================
Centralized store for trading rules that the engine enforces at runtime.

Capital limits are HARD STOPS — the engine will not place any trade that
would cause deployed capital to exceed these limits, regardless of what
signals the strategies generate.

Paper vs live mode is per-desk (equity / F&O) and independent.

Persisted to: data/trading_config.json
Singleton: get_trading_config() / update_trading_config(**kwargs)
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
CONFIG_FILE = DATA_DIR / "trading_config.json"

# ── Defaults ─────────────────────────────────────────────────────────────────
# These match the capital allocation set by the user in the Controls page.
# Overridden on first load if capital_config.json exists.

_DEFAULTS: dict = {
    # Paper vs live execution (via Kite broker).
    # True = paper only (orders logged but NOT sent to Zerodha).
    # False = live orders via Kite Connect API.
    "equity_paper_mode": True,
    "fno_paper_mode":    True,

    # Hard capital limits (INR).
    # The engine cannot deploy more than this across all open positions.
    "equity_capital_limit": 1_100_000,   # ₹11L
    "fno_capital_limit":      200_000,   # ₹2L

    # Guardrails — these pause new buys when triggered, but do NOT liquidate.
    # equity_daily_loss_pct: halt new equity buys if today's unrealised+realised
    #   P&L falls below -(this % * equity_capital_limit).
    "equity_daily_loss_pct":   2.0,  # 2% of equity capital limit
    "equity_max_drawdown_pct": 12.0, # 12% peak-to-trough drawdown → halt buys
    "fno_daily_loss_pct":      1.5,  # 1.5% of F&O limit
    "fno_max_drawdown_pct":    10.0, # 10% F&O drawdown → halt

    # Emergency global kill switch. When True, the engine does NOT execute any
    # new buy or sell during run_cycle(). Existing positions are left open.
    "trading_halted": False,
    "halt_reason":    "",

    "last_updated": None,
}


# ── Singleton state ────────────────────────────────────────────────────────────

_config: dict | None = None
_lock = threading.Lock()


def _load_from_disk() -> dict:
    """Load config from disk, merging missing keys from defaults."""
    cfg = dict(_DEFAULTS)

    # Try to sync capital limits from existing capital_config.json on first run
    cap_file = DATA_DIR / "capital_config.json"
    if cap_file.exists() and not CONFIG_FILE.exists():
        try:
            cap = json.loads(cap_file.read_text())
            cfg["equity_capital_limit"] = int(cap.get("equity_allocation", cfg["equity_capital_limit"]))
            cfg["fno_capital_limit"]    = int(cap.get("fno_allocation",    cfg["fno_capital_limit"]))
        except Exception:
            pass

    if CONFIG_FILE.exists():
        try:
            saved = json.loads(CONFIG_FILE.read_text())
            for k, v in saved.items():
                cfg[k] = v
        except Exception as e:
            logger.warning(f"[TradingConfig] Could not read config file: {e}")

    return cfg


def _save_to_disk(cfg: dict) -> None:
    cfg["last_updated"] = datetime.now(IST).isoformat()
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


# ── Public API ─────────────────────────────────────────────────────────────────

def get_trading_config() -> dict:
    """Return the current trading config (cached in memory)."""
    global _config
    if _config is None:
        with _lock:
            if _config is None:
                _config = _load_from_disk()
    return _config


def update_trading_config(**kwargs) -> dict:
    """
    Update one or more config keys and persist to disk.

    Example:
        update_trading_config(equity_paper_mode=False, equity_capital_limit=500_000)
    """
    global _config
    with _lock:
        cfg = get_trading_config()
        for k, v in kwargs.items():
            if k in _DEFAULTS:
                # Type-coerce to match default type
                default_val = _DEFAULTS[k]
                try:
                    if isinstance(default_val, bool):
                        cfg[k] = bool(v)
                    elif isinstance(default_val, int):
                        cfg[k] = int(v)
                    elif isinstance(default_val, float):
                        cfg[k] = float(v)
                    else:
                        cfg[k] = v
                except (TypeError, ValueError):
                    cfg[k] = v
            else:
                logger.warning(f"[TradingConfig] Unknown key '{k}' — ignoring")
        _save_to_disk(cfg)
        logger.info(f"[TradingConfig] Updated: {kwargs}")
    return cfg


def reload_config() -> dict:
    """Force reload from disk (useful after external edits)."""
    global _config
    with _lock:
        _config = _load_from_disk()
    return _config


# ── Enforcement helpers ────────────────────────────────────────────────────────

def check_equity_trade_allowed(deployed_value: float) -> tuple[bool, str, float]:
    """
    Check whether a new equity buy is allowed given current deployed capital.

    Parameters
    ----------
    deployed_value : float
        Sum of (qty × current_price) for all open equity positions.

    Returns
    -------
    (allowed, reason, remaining_capital)
        allowed           — True if a new buy can proceed
        reason            — human-readable explanation (empty if allowed)
        remaining_capital — how much capital is still available under the limit
    """
    cfg = get_trading_config()

    # 1. Emergency halt
    if cfg.get("trading_halted"):
        reason = cfg.get("halt_reason") or "Emergency trading halt is active"
        return False, f"🛑 HALTED: {reason}", 0.0

    limit = float(cfg.get("equity_capital_limit", float("inf")))
    remaining = limit - deployed_value

    if remaining <= 0:
        return (
            False,
            f"Capital limit reached: deployed ₹{deployed_value:,.0f} ≥ limit ₹{limit:,.0f}",
            0.0,
        )

    return True, "", remaining


def check_fno_trade_allowed(deployed_margin: float) -> tuple[bool, str, float]:
    """Same as check_equity_trade_allowed but for the F&O desk."""
    cfg = get_trading_config()

    if cfg.get("trading_halted"):
        reason = cfg.get("halt_reason") or "Emergency trading halt is active"
        return False, f"🛑 HALTED: {reason}", 0.0

    limit = float(cfg.get("fno_capital_limit", float("inf")))
    remaining = limit - deployed_margin

    if remaining <= 0:
        return (
            False,
            f"F&O capital limit reached: deployed ₹{deployed_margin:,.0f} ≥ limit ₹{limit:,.0f}",
            0.0,
        )

    return True, "", remaining


def get_dashboard_data() -> dict:
    """Return config plus descriptive status for the dashboard."""
    cfg = get_trading_config()
    return {
        **cfg,
        "equity_capital_limit_lakh": round(cfg.get("equity_capital_limit", 0) / 1e5, 2),
        "fno_capital_limit_lakh":    round(cfg.get("fno_capital_limit", 0) / 1e5, 2),
    }
