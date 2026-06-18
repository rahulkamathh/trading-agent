"""
Market Regime Detection Agent
==============================
Classifies the current market into one of 5 regimes based on:
  - Nifty 50 price vs EMA200, EMA50 vs EMA200
  - ADX (14-period manual calculation: +DI, -DI, TR)
  - India VIX

Regimes:
  STRONG_BULL  — Nifty > EMA200, EMA50 > EMA200, ADX > 25, VIX < 15
  BULL         — Nifty > EMA200, EMA50 > EMA200, VIX < 20
  NEUTRAL      — Mixed signals
  BEAR         — Nifty < EMA200 OR EMA50 < EMA200, VIX > 20
  CRISIS       — VIX > 25, Nifty < EMA200, EMA50 < EMA200

Strategy weights are adjusted per regime.

Singleton: get_regime_agent()
Cache: 30 minutes
"""

import logging
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_CACHE_TTL = 1800  # 30 minutes in seconds

# ---------------------------------------------------------------------------
# Regime constants
# ---------------------------------------------------------------------------

REGIMES = ["STRONG_BULL", "BULL", "NEUTRAL", "BEAR", "CRISIS"]

STRATEGY_WEIGHTS = {
    "STRONG_BULL": {
        "MOMENTUM":        0.40,
        "MULTIFACTOR":     0.35,
        "SECTOR_ROTATION": 0.20,
        "MEAN_REVERSION":  0.05,
    },
    "BULL": {
        "MOMENTUM":        0.35,
        "MULTIFACTOR":     0.30,
        "SECTOR_ROTATION": 0.25,
        "MEAN_REVERSION":  0.10,
    },
    "NEUTRAL": {
        "MOMENTUM":        0.25,
        "MULTIFACTOR":     0.25,
        "SECTOR_ROTATION": 0.25,
        "MEAN_REVERSION":  0.25,
    },
    "BEAR": {
        "MOMENTUM":        0.10,
        "MULTIFACTOR":     0.15,
        "SECTOR_ROTATION": 0.30,
        "MEAN_REVERSION":  0.45,
    },
    "CRISIS": {
        "MOMENTUM":        0.05,
        "MULTIFACTOR":     0.10,
        "SECTOR_ROTATION": 0.20,
        "MEAN_REVERSION":  0.65,
    },
}

# Signals allowed per regime
SIGNALS_ALLOWED = {
    "STRONG_BULL": ["BUY"],
    "BULL":        ["BUY"],
    "NEUTRAL":     ["BUY", "SELL"],
    "BEAR":        ["SELL"],
    "CRISIS":      [],
}

# Regime score mapping (100 = strongest bull, 0 = crisis)
REGIME_SCORES = {
    "STRONG_BULL": 90,
    "BULL":        70,
    "NEUTRAL":     50,
    "BEAR":        25,
    "CRISIS":      5,
}

# ---------------------------------------------------------------------------
# Indicator helpers (manual ADX calculation — avoids ta library dependency)
# ---------------------------------------------------------------------------

def _compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute ADX (Average Directional Index) manually.

    Uses: High, Low, Close columns in df.
    Returns a pd.Series of ADX values.
    """
    high = df["High"]
    low  = df["Low"]
    close = df["Close"]

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional Movement
    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s  = pd.Series(plus_dm,  index=df.index)
    minus_dm_s = pd.Series(minus_dm, index=df.index)

    # Wilder smoothing (EMA with alpha = 1/period)
    alpha = 1.0 / period

    def wilder_smooth(s: pd.Series) -> pd.Series:
        result = s.ewm(alpha=alpha, adjust=False).mean()
        return result

    atr_s     = wilder_smooth(tr)
    plus_di   = 100 * wilder_smooth(plus_dm_s) / atr_s.replace(0, np.nan)
    minus_di  = 100 * wilder_smooth(minus_dm_s) / atr_s.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = wilder_smooth(dx.fillna(0))
    return adx


# ---------------------------------------------------------------------------
# RegimeAgent
# ---------------------------------------------------------------------------

class RegimeAgent:
    """Detects the current market regime and recommends strategy weights."""

    def __init__(self):
        self._lock = threading.Lock()
        self._last_run: float = 0.0
        self._cached_result: dict = self._empty_result()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self) -> dict:
        """
        Detect current market regime.

        Returns
        -------
        dict with keys:
            regime, regime_score, nifty_price, ema_200, ema_50, adx, vix,
            trend_direction, signals_allowed, strategy_weights
        """
        with self._lock:
            now = time.time()
            if now - self._last_run < _CACHE_TTL and self._last_run > 0:
                return self._cached_result

            result = self._compute()
            self._cached_result = result
            self._last_run = now
            return result

    def get_dashboard_data(self) -> dict:
        """Return cached detect() result. Returns empty structure if never run."""
        with self._lock:
            return self._cached_result

    # ------------------------------------------------------------------
    # Internal computation
    # ------------------------------------------------------------------

    def _compute(self) -> dict:
        """Fetch data and classify regime."""
        result = self._empty_result()

        # ── Fetch Nifty 50 (300 days) ────────────────────────────────────
        nifty_df = self._fetch_ohlcv("^NSEI", period="2y")
        if nifty_df is None or len(nifty_df) < 210:
            logger.warning("[RegimeAgent] Insufficient Nifty data for regime detection")
            return result

        # ── Compute EMAs ──────────────────────────────────────────────────
        close = nifty_df["Close"]
        ema_200_series = _compute_ema(close, 200)
        ema_50_series  = _compute_ema(close, 50)
        adx_series     = _compute_adx(nifty_df, period=14)

        # Latest values
        nifty_price = float(close.iloc[-1])
        ema_200     = float(ema_200_series.iloc[-1])
        ema_50      = float(ema_50_series.iloc[-1])
        adx         = float(adx_series.iloc[-1]) if not adx_series.empty else 0.0

        # ── Fetch India VIX (100 days) ────────────────────────────────────
        vix = self._fetch_vix()

        # ── Trend direction ───────────────────────────────────────────────
        if nifty_price > ema_200 and ema_50 > ema_200:
            trend_direction = "UPTREND"
        elif nifty_price < ema_200 and ema_50 < ema_200:
            trend_direction = "DOWNTREND"
        else:
            trend_direction = "MIXED"

        # ── Classify regime ───────────────────────────────────────────────
        regime = self._classify(nifty_price, ema_200, ema_50, adx, vix)

        # Compute a more precise regime_score (0-100) using interpolation
        regime_score = self._compute_score(
            regime, nifty_price, ema_200, ema_50, adx, vix
        )

        result.update({
            "regime":           regime,
            "regime_score":     regime_score,
            "nifty_price":      round(nifty_price, 2),
            "ema_200":          round(ema_200, 2),
            "ema_50":           round(ema_50, 2),
            "adx":              round(adx, 2),
            "vix":              round(vix, 2),
            "trend_direction":  trend_direction,
            "signals_allowed":  SIGNALS_ALLOWED[regime],
            "strategy_weights": STRATEGY_WEIGHTS[regime],
            "detected_at":      datetime.now().isoformat(),
        })

        logger.info(
            f"[RegimeAgent] Regime={regime} score={regime_score} "
            f"Nifty={nifty_price:.0f} EMA200={ema_200:.0f} "
            f"EMA50={ema_50:.0f} ADX={adx:.1f} VIX={vix:.2f}"
        )
        return result

    def _classify(
        self,
        price: float,
        ema_200: float,
        ema_50: float,
        adx: float,
        vix: float,
    ) -> str:
        """Apply regime classification rules in priority order."""

        above_ema200 = price > ema_200
        ema50_above_200 = ema_50 > ema_200

        # CRISIS: VIX > 25, price below EMA200, EMA50 below EMA200
        if vix > 25 and not above_ema200 and not ema50_above_200:
            return "CRISIS"

        # STRONG_BULL: all bullish + strong trend + low VIX
        if above_ema200 and ema50_above_200 and adx > 25 and vix < 15:
            return "STRONG_BULL"

        # BULL: price & EMA50 above EMA200, moderate VIX
        if above_ema200 and ema50_above_200 and vix < 20:
            return "BULL"

        # BEAR: below EMA200 or EMA50 crossed below, VIX elevated
        if (not above_ema200 or not ema50_above_200) and vix > 20:
            return "BEAR"

        # NEUTRAL: everything else (mixed signals)
        return "NEUTRAL"

    def _compute_score(
        self,
        regime: str,
        price: float,
        ema_200: float,
        ema_50: float,
        adx: float,
        vix: float,
    ) -> int:
        """
        Compute a continuous regime_score 0-100.
        Base from regime bucket, then nudge based on sub-signals.
        """
        base = REGIME_SCORES[regime]

        # Nudge: price vs EMA200 distance (positive = bullish)
        if ema_200 > 0:
            dist_pct = (price - ema_200) / ema_200 * 100
            nudge_ema = max(-10, min(10, dist_pct * 2))
        else:
            nudge_ema = 0

        # Nudge: ADX strength bonus in bull regimes, penalty in bear
        if regime in ("STRONG_BULL", "BULL"):
            nudge_adx = max(-5, min(5, (adx - 20) / 4))
        else:
            nudge_adx = max(-5, min(5, -(adx - 20) / 4))

        # Nudge: VIX (low VIX = bullish)
        nudge_vix = max(-10, min(10, (20 - vix) * 0.5))

        raw_score = base + nudge_ema + nudge_adx + nudge_vix
        return int(max(0, min(100, round(raw_score))))

    def _fetch_ohlcv(self, ticker: str, period: str = "2y") -> pd.DataFrame | None:
        """Download OHLCV data. Returns None on failure."""
        try:
            df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.dropna(inplace=True)
            return df if not df.empty else None
        except Exception as exc:
            logger.warning(f"[RegimeAgent] OHLCV fetch failed for {ticker}: {exc}")
            return None

    def _fetch_vix(self) -> float:
        """Fetch latest India VIX value. Returns 20.0 (neutral) on failure."""
        try:
            df = yf.download("^INDIAVIX", period="5d", interval="1d", auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty and "Close" in df.columns:
                val = float(df["Close"].dropna().iloc[-1])
                if val > 0:
                    return val
        except Exception as exc:
            logger.warning(f"[RegimeAgent] VIX fetch failed: {exc}")
        return 20.0  # neutral fallback

    @staticmethod
    def _empty_result() -> dict:
        return {
            "regime":           "NEUTRAL",
            "regime_score":     50,
            "nifty_price":      0.0,
            "ema_200":          0.0,
            "ema_50":           0.0,
            "adx":              0.0,
            "vix":              20.0,
            "trend_direction":  "MIXED",
            "signals_allowed":  ["BUY", "SELL"],
            "strategy_weights": STRATEGY_WEIGHTS["NEUTRAL"],
            "detected_at":      None,
        }


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_INSTANCE: RegimeAgent | None = None
_INSTANCE_LOCK = threading.Lock()


def get_regime_agent() -> RegimeAgent:
    """Return the singleton RegimeAgent, creating it on first call."""
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = RegimeAgent()
    return _INSTANCE
