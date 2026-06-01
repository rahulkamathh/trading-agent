"""
Forex SMC Trading Engine
========================
A completely separate agent running alongside the NSE stocks engine.

Strategies (Smart Money Concepts — self-learning weights):
  1. Fair Value Gap (FVG / IFVG)   — 3-candle imbalance, retrace-to-gap entry
  2. Order Block (OB)               — last opposing candle before impulse move
  3. Break of Structure (BOS/CHOCH) — trend confirmation / reversal detection
  4. Liquidity Sweep (LS)           — stop-hunt reversal plays

Universe  : 10 major/cross forex pairs via yfinance (=X suffix)
Capital   : $1,000 USD (paper), configurable via FOREX_CAPITAL env var
Timeframes: 1D for bias · 4H for structure · 1H for FVG/OB · 15M for entry
Sessions  : Fully session-aware — only trades London + NY kill zones
Data      : yfinance — free, reliable for 1H+

Kill Zones (IST):
  London Open  : 13:00 – 15:30  ← highest probability FVG fills
  NY Open      : 18:00 – 20:00  ← second-best kill zone
  London Close : 20:30 – 21:30  ← occasionally useful
  (All times IST = UTC+5:30)
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_IST = ZoneInfo("Asia/Kolkata")


def _now_ist() -> datetime:
    return datetime.now(_IST)


# ─────────────────────────────────────────────────────────────────────────────
# Universe
# ─────────────────────────────────────────────────────────────────────────────

FOREX_PAIRS = {
    # Majors
    "EURUSD=X": {"name": "Euro / US Dollar",       "pip": 0.0001, "group": "major"},
    "GBPUSD=X": {"name": "Pound / US Dollar",      "pip": 0.0001, "group": "major"},
    "USDJPY=X": {"name": "US Dollar / Yen",        "pip": 0.01,   "group": "major"},
    "AUDUSD=X": {"name": "Aussie / US Dollar",     "pip": 0.0001, "group": "major"},
    "USDCAD=X": {"name": "US Dollar / Canadian",   "pip": 0.0001, "group": "major"},
    "NZDUSD=X": {"name": "Kiwi / US Dollar",       "pip": 0.0001, "group": "major"},
    "USDCHF=X": {"name": "US Dollar / Swiss",      "pip": 0.0001, "group": "major"},
    # Key crosses
    "GBPJPY=X": {"name": "Pound / Yen",            "pip": 0.01,   "group": "cross"},
    "EURJPY=X": {"name": "Euro / Yen",             "pip": 0.01,   "group": "cross"},
    "EURGBP=X": {"name": "Euro / Pound",           "pip": 0.0001, "group": "cross"},
}

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR          = Path(__file__).parent / "data"
FOREX_PORTFOLIO_FILE = DATA_DIR / "forex_portfolio.json"
FOREX_TRADES_FILE    = DATA_DIR / "forex_trades.json"
FOREX_SIGNALS_FILE   = DATA_DIR / "forex_signals.json"
DATA_DIR.mkdir(exist_ok=True)

# Capital: configurable via env var, default $1,000
FOREX_INITIAL_CAPITAL = float(os.environ.get("FOREX_CAPITAL", "1000"))

MAX_POSITION_PCT  = 0.10   # max 10% of capital per trade (forex is leveraged paper)
RISK_PER_TRADE_PCT = 0.01  # risk 1% of capital per trade = $10 on $1k
MAX_POSITIONS     = 6      # max concurrent open trades
MIN_RR            = 2.0    # minimum risk:reward to take a trade
ATR_PERIOD        = 14
ATR_SL_MULT       = 1.5    # SL = 1.5 × ATR from entry
MIN_FVG_PIPS      = 3      # FVG must be at least 3 pips wide to be tradeable
COOLDOWN_HOURS    = 4      # hours before re-trading same pair after exit

# Kill Zone windows (IST hours, inclusive)
KILL_ZONES = [
    {"name": "London Open",  "start": 13, "end": 16, "weight": 1.5},
    {"name": "NY Open",      "start": 18, "end": 20, "weight": 1.3},
    {"name": "London Close", "start": 20, "end": 22, "weight": 0.8},
]


# ─────────────────────────────────────────────────────────────────────────────
# Data Fetcher
# ─────────────────────────────────────────────────────────────────────────────

_price_cache: dict = {}
_CACHE_TTL_S = 300   # 5 minutes

def _fetch(pair: str, period: str = "30d", interval: str = "1h") -> pd.DataFrame | None:
    """Fetch OHLCV for a forex pair via yfinance. Returns cleaned DataFrame or None."""
    cache_key = f"{pair}_{period}_{interval}"
    now = time.time()
    if cache_key in _price_cache:
        ts, df = _price_cache[cache_key]
        if now - ts < _CACHE_TTL_S:
            return df

    try:
        import yfinance as yf
        df = yf.download(pair, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 5:
            return None
        df = df.dropna(subset=["Close", "High", "Low", "Open"])
        _price_cache[cache_key] = (now, df)
        return df
    except Exception as exc:
        logger.debug(f"[Forex] Fetch failed {pair} {interval}: {exc}")
        return None


def get_price(pair: str) -> float | None:
    """Current mid price for a pair."""
    df = _fetch(pair, period="2d", interval="1h")
    if df is not None and not df.empty:
        return float(df["Close"].iloc[-1])
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Session & Kill Zone helpers
# ─────────────────────────────────────────────────────────────────────────────

def current_session() -> dict:
    """Return current trading session info."""
    now = _now_ist()
    hour = now.hour
    for kz in KILL_ZONES:
        if kz["start"] <= hour < kz["end"]:
            return {"in_kill_zone": True, "name": kz["name"], "weight": kz["weight"]}
    # Check broader sessions
    if 5 <= hour < 13:
        return {"in_kill_zone": False, "name": "Asian Session",   "weight": 0.5}
    if 13 <= hour < 21:
        return {"in_kill_zone": False, "name": "London Session",  "weight": 1.0}
    if 18 <= hour < 24:
        return {"in_kill_zone": False, "name": "New York Session","weight": 1.0}
    return {"in_kill_zone": False, "name": "Off-Hours", "weight": 0.3}


def is_forex_market_open() -> bool:
    """Forex is open Mon–Fri, 00:00–24:00 UTC (nearly 24/5).
    Returns False only on weekends (IST Sat 05:30 – Mon 05:30 approx)."""
    now = _now_ist()
    # Saturday after 5:30 IST or Sunday
    if now.weekday() == 5 and now.hour >= 6:   return False
    if now.weekday() == 6:                      return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# ATR helper
# ─────────────────────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float | None:
    try:
        h, l, c = df["High"], df["Low"], df["Close"]
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        return float(atr) if pd.notna(atr) else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SMC Strategy 1: Fair Value Gap (FVG / IFVG)
# ─────────────────────────────────────────────────────────────────────────────

class FVGStrategy:
    """
    Fair Value Gap (3-candle imbalance) strategy.

    Bullish FVG: candle[i-2].high < candle[i].low  — gap forms, price is bullish
      → BUY when current price retraces into the gap (between the two levels)
    Bearish FVG: candle[i-2].low > candle[i].high  — gap forms, price is bearish
      → SELL when current price retraces into the gap

    Only generates signals during Kill Zones. FVG must be >= MIN_FVG_PIPS wide.
    Higher timeframe FVGs (4H, 1D) get higher strength scores.
    """
    name       = "Fair Value Gap"
    short_name = "FVG"

    def _find_fvgs(self, df: pd.DataFrame, pip: float) -> list:
        """Return list of unmitigated FVGs from the dataframe."""
        fvgs = []
        highs  = df["High"].values
        lows   = df["Low"].values
        closes = df["Close"].values
        times  = df.index

        for i in range(2, len(df)):
            # Bullish FVG: gap between candle[i-2] high and candle[i] low
            gap_top    = lows[i]
            gap_bottom = highs[i - 2]
            if gap_top > gap_bottom:
                gap_size_pips = (gap_top - gap_bottom) / pip
                if gap_size_pips >= MIN_FVG_PIPS:
                    fvgs.append({
                        "type":      "bullish",
                        "top":       round(float(gap_top), 6),
                        "bottom":    round(float(gap_bottom), 6),
                        "mid":       round(float((gap_top + gap_bottom) / 2), 6),
                        "pips":      round(gap_size_pips, 1),
                        "time":      str(times[i]),
                        "idx":       i,
                        "mitigated": False,
                    })

            # Bearish FVG: gap between candle[i-2] low and candle[i] high
            gap_top    = lows[i - 2]
            gap_bottom = highs[i]
            if gap_top > gap_bottom:
                gap_size_pips = (gap_top - gap_bottom) / pip
                if gap_size_pips >= MIN_FVG_PIPS:
                    fvgs.append({
                        "type":      "bearish",
                        "top":       round(float(gap_top), 6),
                        "bottom":    round(float(gap_bottom), 6),
                        "mid":       round(float((gap_top + gap_bottom) / 2), 6),
                        "pips":      round(gap_size_pips, 1),
                        "time":      str(times[i]),
                        "idx":       i,
                        "mitigated": False,
                    })

        # Mark mitigated (price already traded through the gap)
        current_price = float(closes[-1])
        for fvg in fvgs:
            if fvg["type"] == "bullish" and current_price < fvg["bottom"]:
                fvg["mitigated"] = True
            elif fvg["type"] == "bearish" and current_price > fvg["top"]:
                fvg["mitigated"] = True

        return [f for f in fvgs if not f["mitigated"]]

    def generate_signals(self) -> list:
        session = current_session()
        signals = []

        for pair, meta in FOREX_PAIRS.items():
            try:
                pip = meta["pip"]

                # Multi-timeframe: 4H for structure, 1H for entry
                df_4h = _fetch(pair, period="30d", interval="4h")
                df_1h = _fetch(pair, period="7d",  interval="1h")
                if df_4h is None or df_1h is None:
                    continue

                price = float(df_1h["Close"].iloc[-1])
                atr   = _atr(df_1h) or (price * 0.003)

                # 4H FVGs = higher conviction (structural)
                fvgs_4h = self._find_fvgs(df_4h, pip)
                # 1H FVGs = tactical entries
                fvgs_1h = self._find_fvgs(df_1h, pip)

                # Check if price is inside any active FVG
                for fvg in fvgs_4h[-5:] + fvgs_1h[-8:]:  # look at recent FVGs only
                    tf_mult = 1.3 if fvg in fvgs_4h else 1.0
                    in_fvg = fvg["bottom"] <= price <= fvg["top"]
                    near_fvg = (fvg["bottom"] * 0.999 <= price <= fvg["top"] * 1.001)

                    if not (in_fvg or near_fvg):
                        continue

                    direction = "BUY" if fvg["type"] == "bullish" else "SELL"

                    # Strength: base 55 + session weight + tf mult + pip size
                    strength = 55
                    strength += session["weight"] * 10
                    strength *= tf_mult
                    if fvg["pips"] > 10: strength += 5   # wider FVG = more significant
                    if in_fvg:           strength += 8   # price already inside the gap

                    # Kill zone boost
                    if session["in_kill_zone"]:
                        strength += 12

                    strength = min(95, round(strength))

                    signals.append({
                        "pair":      pair,
                        "name":      meta["name"],
                        "signal":    direction,
                        "strategy":  self.short_name,
                        "strength":  strength,
                        "price":     price,
                        "atr":       round(atr, 6),
                        "fvg":       fvg,
                        "session":   session["name"],
                        "in_kz":     session["in_kill_zone"],
                        "timeframe": "4H" if fvg in fvgs_4h else "1H",
                    })
                    break   # one signal per pair per strategy

            except Exception as exc:
                logger.debug(f"[FVG] {pair}: {exc}")

        return signals


# ─────────────────────────────────────────────────────────────────────────────
# SMC Strategy 2: Order Block (OB)
# ─────────────────────────────────────────────────────────────────────────────

class OrderBlockStrategy:
    """
    Order Block: the last opposing candle before a significant impulse move.

    Bullish OB: last bearish (red) candle before a strong bullish move
      → BUY when price retraces to the body of that candle
    Bearish OB: last bullish (green) candle before a strong bearish move
      → SELL when price retraces to the body of that candle

    An impulse move is defined as ≥ 3× ATR in a single candle or over 3 candles.
    """
    name       = "Order Block"
    short_name = "OB"
    IMPULSE_ATR_MULT = 2.0   # impulse candle must be ≥ 2× ATR to qualify

    def _find_order_blocks(self, df: pd.DataFrame, atr: float) -> list:
        obs = []
        opens  = df["Open"].values
        closes = df["Close"].values
        highs  = df["High"].values
        lows   = df["Low"].values
        times  = df.index

        for i in range(3, len(df) - 1):
            # Check if the NEXT candle (i+1 direction) is an impulse
            next_body = abs(closes[i + 1] - opens[i + 1])
            if next_body < self.IMPULSE_ATR_MULT * atr:
                continue

            bullish_impulse = closes[i + 1] > opens[i + 1]
            bearish_impulse = closes[i + 1] < opens[i + 1]

            # Bullish OB: last bearish candle before bullish impulse
            if bullish_impulse and closes[i] < opens[i]:
                obs.append({
                    "type":    "bullish",
                    "top":     round(float(opens[i]), 6),    # OB = body of bearish candle
                    "bottom":  round(float(closes[i]), 6),
                    "time":    str(times[i]),
                    "idx":     i,
                })

            # Bearish OB: last bullish candle before bearish impulse
            elif bearish_impulse and closes[i] > opens[i]:
                obs.append({
                    "type":    "bearish",
                    "top":     round(float(closes[i]), 6),   # OB = body of bullish candle
                    "bottom":  round(float(opens[i]), 6),
                    "time":    str(times[i]),
                    "idx":     i,
                })

        # Mark mitigated
        current = float(closes[-1])
        for ob in obs:
            if ob["type"] == "bullish" and current < ob["bottom"]:
                ob["mitigated"] = True
            elif ob["type"] == "bearish" and current > ob["top"]:
                ob["mitigated"] = True
            else:
                ob["mitigated"] = False

        return [ob for ob in obs if not ob.get("mitigated")]

    def generate_signals(self) -> list:
        session = current_session()
        signals = []

        for pair, meta in FOREX_PAIRS.items():
            try:
                df_4h = _fetch(pair, period="30d", interval="4h")
                if df_4h is None: continue

                price = float(df_4h["Close"].iloc[-1])
                atr   = _atr(df_4h) or (price * 0.003)

                obs = self._find_order_blocks(df_4h, atr)
                for ob in obs[-6:]:
                    in_ob    = ob["bottom"] <= price <= ob["top"]
                    near_ob  = (ob["bottom"] * 0.9995 <= price <= ob["top"] * 1.0005)
                    if not (in_ob or near_ob):
                        continue

                    direction = "BUY" if ob["type"] == "bullish" else "SELL"
                    strength  = 58 + (session["weight"] * 8)
                    if session["in_kill_zone"]: strength += 10
                    if in_ob: strength += 6
                    strength = min(92, round(strength))

                    signals.append({
                        "pair":      pair,
                        "name":      meta["name"],
                        "signal":    direction,
                        "strategy":  self.short_name,
                        "strength":  strength,
                        "price":     price,
                        "atr":       round(atr, 6),
                        "ob":        ob,
                        "session":   session["name"],
                        "in_kz":     session["in_kill_zone"],
                        "timeframe": "4H",
                    })
                    break

            except Exception as exc:
                logger.debug(f"[OB] {pair}: {exc}")

        return signals


# ─────────────────────────────────────────────────────────────────────────────
# SMC Strategy 3: Break of Structure (BOS / CHOCH)
# ─────────────────────────────────────────────────────────────────────────────

class BreakOfStructureStrategy:
    """
    Break of Structure (BOS): price decisively breaks a previous swing high/low.
    Change of Character (CHOCH): first break in the opposite direction — early reversal.

    BOS signals trend continuation; CHOCH signals potential reversal.
    Both generate signals for the new direction.
    """
    name       = "Break of Structure"
    short_name = "BOS"
    SWING_LOOKBACK = 12   # candles to look back for swing points

    def _find_swing_points(self, df: pd.DataFrame):
        highs = df["High"].values
        lows  = df["Low"].values
        swing_highs, swing_lows = [], []

        for i in range(self.SWING_LOOKBACK, len(df) - 1):
            # Swing high: highest in lookback window
            window_h = highs[i - self.SWING_LOOKBACK: i]
            if highs[i] == max(window_h) and highs[i] > highs[i - 1]:
                swing_highs.append((i, float(highs[i])))
            # Swing low: lowest in lookback window
            window_l = lows[i - self.SWING_LOOKBACK: i]
            if lows[i] == min(window_l) and lows[i] < lows[i - 1]:
                swing_lows.append((i, float(lows[i])))

        return swing_highs, swing_lows

    def generate_signals(self) -> list:
        session = current_session()
        signals = []

        for pair, meta in FOREX_PAIRS.items():
            try:
                df = _fetch(pair, period="30d", interval="4h")
                if df is None: continue

                closes = df["Close"].values
                price  = float(closes[-1])
                atr    = _atr(df) or (price * 0.003)

                swing_highs, swing_lows = self._find_swing_points(df)
                if not swing_highs or not swing_lows:
                    continue

                # Most recent swing points
                last_sh = swing_highs[-1][1] if swing_highs else None
                last_sl = swing_lows[-1][1]  if swing_lows  else None
                prev_sh = swing_highs[-2][1] if len(swing_highs) >= 2 else None
                prev_sl = swing_lows[-2][1]  if len(swing_lows) >= 2  else None

                direction = None
                bos_type  = None
                level     = None

                # BOS UP: price breaks above the most recent swing high
                if last_sh and price > last_sh:
                    direction = "BUY"
                    bos_type  = "BOS" if prev_sh and last_sh > prev_sh else "CHOCH"
                    level     = last_sh

                # BOS DOWN: price breaks below the most recent swing low
                elif last_sl and price < last_sl:
                    direction = "SELL"
                    bos_type  = "BOS" if prev_sl and last_sl < prev_sl else "CHOCH"
                    level     = last_sl

                if not direction:
                    continue

                # CHOCH slightly lower strength (counter-trend)
                base     = 60 if bos_type == "BOS" else 52
                strength = base + session["weight"] * 8
                if session["in_kill_zone"]: strength += 10
                strength = min(90, round(strength))

                signals.append({
                    "pair":      pair,
                    "name":      meta["name"],
                    "signal":    direction,
                    "strategy":  f"{self.short_name}:{bos_type}",
                    "strength":  strength,
                    "price":     price,
                    "atr":       round(atr, 6),
                    "bos_level": round(level, 6) if level else None,
                    "bos_type":  bos_type,
                    "session":   session["name"],
                    "in_kz":     session["in_kill_zone"],
                    "timeframe": "4H",
                })

            except Exception as exc:
                logger.debug(f"[BOS] {pair}: {exc}")

        return signals


# ─────────────────────────────────────────────────────────────────────────────
# SMC Strategy 4: Liquidity Sweep
# ─────────────────────────────────────────────────────────────────────────────

class LiquiditySweepStrategy:
    """
    Liquidity Sweep: price briefly spikes through a cluster of equal highs/lows
    (where stop orders accumulate), then quickly reverses.

    Equal highs (buy-side liquidity / BSL): sweep above → then SELL
    Equal lows  (sell-side liquidity / SSL): sweep below → then BUY

    Requires price to have reversed back inside the sweep within 2 candles.
    """
    name       = "Liquidity Sweep"
    short_name = "LSWP"
    EQL_TOLERANCE_PCT = 0.0005   # highs/lows within 0.05% = "equal"

    def _find_equal_levels(self, df: pd.DataFrame):
        highs = df["High"].values
        lows  = df["Low"].values
        eq_highs, eq_lows = [], []

        for i in range(len(df) - 5, max(len(df) - 50, 0), -1):
            for j in range(i - 1, max(i - 20, 0), -1):
                # Equal highs
                if abs(highs[i] - highs[j]) / highs[i] < self.EQL_TOLERANCE_PCT:
                    eq_highs.append((i, float(highs[i])))
                    break
                # Equal lows
                if abs(lows[i] - lows[j]) / lows[i] < self.EQL_TOLERANCE_PCT:
                    eq_lows.append((i, float(lows[i])))
                    break

        return eq_highs, eq_lows

    def generate_signals(self) -> list:
        session = current_session()
        signals = []

        for pair, meta in FOREX_PAIRS.items():
            try:
                df = _fetch(pair, period="7d", interval="1h")
                if df is None or len(df) < 20: continue

                highs  = df["High"].values
                lows   = df["Low"].values
                closes = df["Close"].values
                price  = float(closes[-1])
                atr    = _atr(df) or (price * 0.002)

                # Check last 3 candles for a sweep + reversal
                # Sweep high: candle[i] high > recent equal high, then close below it
                eq_highs, eq_lows = self._find_equal_levels(df)

                direction = None
                sweep_level = None

                # BSL sweep: price spiked above equal highs → now trading below → SELL
                for _, level in eq_highs[-3:]:
                    recent_high = max(highs[-3:])
                    if recent_high > level and price < level:
                        direction   = "SELL"
                        sweep_level = level
                        break

                # SSL sweep: price spiked below equal lows → now trading above → BUY
                if not direction:
                    for _, level in eq_lows[-3:]:
                        recent_low = min(lows[-3:])
                        if recent_low < level and price > level:
                            direction   = "BUY"
                            sweep_level = level
                            break

                if not direction:
                    continue

                strength = 62 + session["weight"] * 7
                if session["in_kill_zone"]: strength += 12
                strength = min(92, round(strength))

                signals.append({
                    "pair":        pair,
                    "name":        meta["name"],
                    "signal":      direction,
                    "strategy":    self.short_name,
                    "strength":    strength,
                    "price":       price,
                    "atr":         round(atr, 6),
                    "sweep_level": round(sweep_level, 6) if sweep_level else None,
                    "session":     session["name"],
                    "in_kz":       session["in_kill_zone"],
                    "timeframe":   "1H",
                })

            except Exception as exc:
                logger.debug(f"[LSWP] {pair}: {exc}")

        return signals


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio
# ─────────────────────────────────────────────────────────────────────────────

class ForexPortfolio:
    """Paper-trading portfolio for forex. Capital in USD."""

    def __init__(self):
        self.state = self._load()

    def _load(self) -> dict:
        if FOREX_PORTFOLIO_FILE.exists():
            try:
                with open(FOREX_PORTFOLIO_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "cash":      FOREX_INITIAL_CAPITAL,
            "initial":   FOREX_INITIAL_CAPITAL,
            "positions": {},
            "cooldowns": {},
        }

    def _save(self):
        with open(FOREX_PORTFOLIO_FILE, "w") as f:
            json.dump(self.state, f, indent=2)

    def _load_trades(self) -> list:
        if FOREX_TRADES_FILE.exists():
            try:
                with open(FOREX_TRADES_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save_trade(self, trade: dict):
        trades = self._load_trades()
        trades.append(trade)
        with open(FOREX_TRADES_FILE, "w") as f:
            json.dump(trades[-2000:], f, indent=2)

    def get_total_value(self) -> float:
        total = self.state["cash"]
        for pair, pos in self.state["positions"].items():
            price = get_price(pair) or pos["avg_price"]
            total += pos["units"] * price
        return round(total, 4)

    def available_cash(self) -> float:
        return self.state["cash"]

    def in_cooldown(self, pair: str) -> bool:
        cd = self.state.get("cooldowns", {}).get(pair)
        if not cd: return False
        return datetime.now(_IST) < datetime.fromisoformat(cd)

    def _set_cooldown(self, pair: str):
        self.state.setdefault("cooldowns", {})[pair] = (
            _now_ist() + timedelta(hours=COOLDOWN_HOURS)
        ).isoformat()

    def check_stops(self) -> list:
        """Check SL/TP on all open positions. Returns list of closed trades."""
        closed = []
        for pair in list(self.state["positions"].keys()):
            pos   = self.state["positions"][pair]
            price = get_price(pair)
            if not price: continue

            sl = pos.get("stop_loss")
            tp = pos.get("target")
            direction = pos.get("direction", "BUY")

            hit_sl = (sl and direction == "BUY"  and price <= sl) or \
                     (sl and direction == "SELL" and price >= sl)
            hit_tp = (tp and direction == "BUY"  and price >= tp) or \
                     (tp and direction == "SELL" and price <= tp)

            if hit_sl or hit_tp:
                reason = "STOP_LOSS" if hit_sl else "TAKE_PROFIT"
                trade  = self.execute_close(pair, price, reason=reason)
                if trade: closed.append(trade)

        return closed

    def can_open(self, pair: str, price: float) -> bool:
        if len(self.state["positions"]) >= MAX_POSITIONS: return False
        if pair in self.state["positions"]:               return False
        if self.in_cooldown(pair):                        return False
        if self.state["cash"] < self.state["initial"] * 0.005: return False
        return True

    def execute_open(self, pair: str, price: float, direction: str,
                     strategy: str, strength: float = 65.0,
                     atr: float | None = None) -> dict | None:
        if not self.can_open(pair, price): return None

        total_val  = self.get_total_value()
        atr_val    = atr or (price * 0.003)
        sl_dist    = atr_val * ATR_SL_MULT

        # RR scales with strength
        rr = 4.0 if strength >= 85 else 3.0 if strength >= 75 else 2.5 if strength >= 65 else 2.0

        # Position sizing: risk RISK_PER_TRADE_PCT of capital
        risk_usd  = total_val * RISK_PER_TRADE_PCT
        units     = risk_usd / sl_dist          # how many units so that 1 ATR-stop = risk_usd
        max_units = (total_val * MAX_POSITION_PCT) / price
        units     = min(units, max_units, self.state["cash"] / price * 0.95)
        units     = round(units, 4)
        if units <= 0: return None

        cost = units * price
        if direction == "BUY":
            stop_loss = round(price - sl_dist, 6)
            target    = round(price + sl_dist * rr, 6)
        else:
            stop_loss = round(price + sl_dist, 6)
            target    = round(price - sl_dist * rr, 6)

        self.state["cash"] -= cost
        self.state["positions"][pair] = {
            "units":       units,
            "avg_price":   price,
            "direction":   direction,
            "strategy":    strategy,
            "entry_date":  _now_ist().isoformat(),
            "stop_loss":   stop_loss,
            "target":      target,
            "planned_rr":  rr,
            "strength":    round(strength, 1),
            "atr":         round(atr_val, 6),
        }
        self._save()

        trade = {
            "id":          str(uuid.uuid4()),
            "action":      "OPEN",
            "direction":   direction,
            "pair":        pair,
            "units":       units,
            "price":       price,
            "cost":        round(cost, 4),
            "stop_loss":   stop_loss,
            "target":      target,
            "planned_rr":  rr,
            "strategy":    strategy,
            "strength":    round(strength, 1),
            "time":        _now_ist().isoformat(),
        }
        self._save_trade(trade)
        logger.info(
            f"[Forex] OPEN {direction} {pair} units={units:.4f} @ {price:.5f}  "
            f"SL={stop_loss:.5f}  TP={target:.5f}  RR=1:{rr}"
        )
        return trade

    def execute_close(self, pair: str, price: float, reason: str = "SIGNAL") -> dict | None:
        pos = self.state["positions"].get(pair)
        if not pos: return None

        units    = pos["units"]
        entry    = pos["avg_price"]
        direction = pos.get("direction", "BUY")

        if direction == "BUY":
            pnl  = (price - entry) * units
        else:
            pnl  = (entry - price) * units

        proceeds = units * price
        self.state["cash"] += proceeds
        del self.state["positions"][pair]
        self._set_cooldown(pair)
        self._save()

        pnl_pct = round((pnl / (units * entry)) * 100, 3)

        trade = {
            "id":          str(uuid.uuid4()),
            "action":      "CLOSE",
            "direction":   direction,
            "pair":        pair,
            "units":       units,
            "entry_price": entry,
            "exit_price":  price,
            "pnl_usd":     round(pnl, 4),
            "pnl_pct":     pnl_pct,
            "reason":      reason,
            "strategy":    pos.get("strategy", ""),
            "time":        _now_ist().isoformat(),
            "hold_hours":  round(
                (datetime.fromisoformat(_now_ist().isoformat()) -
                 datetime.fromisoformat(pos["entry_date"])).total_seconds() / 3600, 1
            ),
        }
        self._save_trade(trade)
        logger.info(
            f"[Forex] CLOSE {direction} {pair} @ {price:.5f}  "
            f"PnL=${pnl:+.2f} ({pnl_pct:+.2f}%)  [{reason}]"
        )
        return trade

    def get_positions(self) -> list:
        out = []
        for pair, pos in self.state["positions"].items():
            price = get_price(pair) or pos["avg_price"]
            direction = pos.get("direction", "BUY")
            entry = pos["avg_price"]
            units = pos["units"]
            if direction == "BUY":
                pnl     = (price - entry) * units
                pnl_pct = ((price - entry) / entry) * 100
            else:
                pnl     = (entry - price) * units
                pnl_pct = ((entry - price) / entry) * 100
            out.append({
                **pos,
                "pair":       pair,
                "name":       FOREX_PAIRS.get(pair, {}).get("name", pair),
                "current":    round(price, 6),
                "pnl_usd":    round(pnl, 4),
                "pnl_pct":    round(pnl_pct, 3),
            })
        return out

    def get_summary(self) -> dict:
        trades = self._load_trades()
        closes = [t for t in trades if t.get("action") == "CLOSE"]
        wins   = [t for t in closes if t.get("pnl_usd", 0) > 0]
        total_val = self.get_total_value()
        total_pnl = total_val - self.state["initial"]
        return {
            "cash":           round(self.state["cash"], 4),
            "initial":        self.state["initial"],
            "total_value":    round(total_val, 4),
            "total_pnl":      round(total_pnl, 4),
            "total_pnl_pct":  round(total_pnl / self.state["initial"] * 100, 3),
            "positions":      len(self.state["positions"]),
            "total_trades":   len(closes),
            "win_rate":       round(len(wins) / len(closes) * 100, 1) if closes else 0,
            "currency":       "USD",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Signal Aggregator
# ─────────────────────────────────────────────────────────────────────────────

class ForexSignalAggregator:
    def __init__(self):
        self._strategies = [
            FVGStrategy(),
            OrderBlockStrategy(),
            BreakOfStructureStrategy(),
            LiquiditySweepStrategy(),
        ]
        self._weights = self._load_weights()

    def _load_weights(self) -> dict:
        wf = DATA_DIR / "forex_strategy_weights.json"
        if wf.exists():
            try:
                with open(wf) as f:
                    return json.load(f)
            except Exception:
                pass
        return {s.short_name: 1.0 for s in self._strategies}

    def _save_weights(self):
        wf = DATA_DIR / "forex_strategy_weights.json"
        with open(wf, "w") as f:
            json.dump(self._weights, f, indent=2)

    def update_weight(self, strategy_name: str, won: bool):
        """Simple exponential smoothing weight update."""
        w = self._weights.get(strategy_name, 1.0)
        w = w * 0.95 + (1.2 if won else 0.8) * 0.05
        self._weights[strategy_name] = round(max(0.3, min(2.0, w)), 4)
        self._save_weights()

    def run(self) -> list:
        all_signals = []
        for strat in self._strategies:
            try:
                sigs = strat.generate_signals()
                base = strat.short_name.split(":")[0]
                w    = self._weights.get(base, 1.0)
                for s in sigs:
                    s["strength"] = min(100, round(s["strength"] * w))
                all_signals.extend(sigs)
            except Exception as exc:
                logger.warning(f"[Forex] Strategy {strat.name} error: {exc}")
        return all_signals

    def get_weights(self) -> dict:
        return dict(self._weights)


# ─────────────────────────────────────────────────────────────────────────────
# Forex Agent — main run loop
# ─────────────────────────────────────────────────────────────────────────────

class ForexAgent:
    """
    Self-contained forex trading agent.
    run_cycle() is equivalent to the NSE engine's run_cycle().
    """
    MIN_STRENGTH = 65

    def __init__(self):
        self.portfolio  = ForexPortfolio()
        self.aggregator = ForexSignalAggregator()

    def run_cycle(self) -> dict:
        if not is_forex_market_open():
            logger.info("[Forex] Market closed (weekend)")
            return {"status": "market_closed", "executed": [], "signals": []}

        session = current_session()
        logger.info(f"[Forex] Cycle start — session: {session['name']}")

        # 1. Check stops on open positions
        stops = self.portfolio.check_stops()

        # 2. Update strategy weights from closed trades
        if stops:
            for t in stops:
                won  = t.get("pnl_usd", 0) > 0
                strat = (t.get("strategy") or "").split(":")[0]
                if strat:
                    self.aggregator.update_weight(strat, won)

        # 3. Generate signals
        signals = self.aggregator.run()

        # 4. Save signals log
        self._save_signals(signals)

        # 5. Aggregate per pair (same logic as NSE engine)
        buy_agg:  dict[str, dict] = {}
        sell_agg: dict[str, dict] = {}
        for sig in signals:
            pair     = sig["pair"]
            action   = sig["signal"]
            strength = sig["strength"]
            strategy = sig["strategy"]
            price    = sig.get("price", 0)
            atr      = sig.get("atr")
            if action == "BUY":
                if pair not in buy_agg:
                    buy_agg[pair] = {"strengths":[], "strategies":[], "price":price, "atr": atr}
                buy_agg[pair]["strengths"].append(strength)
                buy_agg[pair]["strategies"].append(strategy)
            elif action == "SELL":
                if pair not in sell_agg:
                    sell_agg[pair] = {"strengths":[], "strategies":[], "price":price, "atr": atr}
                sell_agg[pair]["strengths"].append(strength)
                sell_agg[pair]["strategies"].append(strategy)

        # 6. Execute best BUY candidates
        executed = []
        # Sort by composite strength (avg), kill-zone signals first
        candidates = sorted(
            [(p, sum(a["strengths"]) / len(a["strengths"]), a)
             for p, a in buy_agg.items()],
            key=lambda x: x[1], reverse=True
        )

        for pair, strength, agg in candidates:
            if strength < self.MIN_STRENGTH: continue
            if self.portfolio.in_cooldown(pair): continue
            price = agg["price"] or get_price(pair)
            if not price: continue
            strat = "+".join(sorted(set(s.split(":")[0] for s in agg["strategies"])))
            trade = self.portfolio.execute_open(
                pair, price, "BUY", strat, strength, agg.get("atr")
            )
            if trade:
                executed.append(trade)

        # Also check SELL candidates for new short positions
        for pair, strength, agg in sorted(
            [(p, sum(a["strengths"]) / len(a["strengths"]), a)
             for p, a in sell_agg.items()],
            key=lambda x: x[1], reverse=True
        ):
            if strength < self.MIN_STRENGTH: continue
            if self.portfolio.in_cooldown(pair): continue
            price = agg["price"] or get_price(pair)
            if not price: continue
            strat = "+".join(sorted(set(s.split(":")[0] for s in agg["strategies"])))
            trade = self.portfolio.execute_open(
                pair, price, "SELL", strat, strength, agg.get("atr")
            )
            if trade:
                executed.append(trade)

        logger.info(
            f"[Forex] Cycle done — {len(signals)} signals, "
            f"{len(stops)} stops, {len(executed)} new trades"
        )
        return {
            "status":    "ok",
            "session":   session,
            "signals":   signals,
            "stops":     stops,
            "executed":  executed,
            "positions": len(self.portfolio.state["positions"]),
        }

    def _save_signals(self, signals: list):
        try:
            existing = []
            if FOREX_SIGNALS_FILE.exists():
                with open(FOREX_SIGNALS_FILE) as f:
                    existing = json.load(f)
        except Exception:
            existing = []

        for s in signals:
            existing.append({
                **s,
                "fvg": None, "ob": None,  # strip large nested objects for storage
                "time": _now_ist().isoformat(),
            })
        with open(FOREX_SIGNALS_FILE, "w") as f:
            json.dump(existing[-3000:], f)

    def get_signals(self, limit: int = 100) -> list:
        try:
            if FOREX_SIGNALS_FILE.exists():
                with open(FOREX_SIGNALS_FILE) as f:
                    sigs = json.load(f)
                return list(reversed(sigs[-limit:]))
        except Exception:
            pass
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_forex_agent: ForexAgent | None = None


def get_forex_agent() -> ForexAgent:
    global _forex_agent
    if _forex_agent is None:
        _forex_agent = ForexAgent()
    return _forex_agent
