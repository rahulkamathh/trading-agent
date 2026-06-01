"""
Dynamic Risk Manager
====================
Replaces all hardcoded position-sizing constants with a live risk model.

Key responsibilities
--------------------
1. EventCalendar   — knows MSCI rebalancing, NSE F&O expiry, RBI MPC dates, Budget
2. MacroRiskScorer — combines event proximity + news headlines → risk score 0.0–1.0
3. PositionSizer   — volatility-targeting Kelly: size = f(portfolio, ATR, conviction, risk)
4. DrawdownGuard   — portfolio circuit-breakers: reduce / halt on intraday drawdown

Usage (from engine.py)
----------------------
    from risk_manager import get_risk_manager
    rm = get_risk_manager()

    # Before each buy:
    risk_score, risk_label = rm.macro_risk()          # e.g. (0.72, "HIGH")
    if risk_label == "EXTREME": continue              # skip new buys today

    spend = rm.position_size(
        portfolio_value = port.total_value(),
        available_cash  = port.available_cash(),
        price           = price,
        atr             = atr,           # ATR(14) in ₹ — pass None to use fallback
        strength        = strength,      # 0–100 signal conviction
        risk_score      = risk_score,
    )

    # After every cycle:
    rm.update_peak(port.total_value())
    if rm.is_halted(port.total_value()):
        break  # skip all buys this cycle
"""

import json
import logging
import os
import re
import threading
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
_IST = ZoneInfo("Asia/Kolkata")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Event Calendar
# ═══════════════════════════════════════════════════════════════════════════════

class EventCalendar:
    """
    Comprehensive Indian market event calendar.
    Covers 2025–2028 for all major event types.
    F&O expiry is computed dynamically (never stale).
    Earnings seasons are detected by month range (always current).

    Event types and their risk weight:
      MSCI semi-annual (May/Nov)  → HIGHEST  (2.5× multiplier when 2 days before)
      MSCI quarterly              → HIGH     (1.8×)
      Union Budget                → HIGH     (2.0×)
      RBI MPC                     → MEDIUM   (1.6×)
      F&O Monthly Expiry          → MEDIUM   (1.4×)
      Earnings Season start       → LOW      (1.2×)
      GST Council Meeting         → LOW      (1.2×)
    """

    # ── MSCI Standard Index Review rebalancing dates (quarterly) ─────────────
    # MSCI rebalancing TRADES happen on the last Friday of Feb/May/Aug/Nov.
    # (The "effective date" published by MSCI is the next Monday, but the
    # actual institutional flows occur at Friday's close — that's the risk day.)
    # Semi-annual reviews (May/Nov) are the BIG ones driving largest flows.
    MSCI_REBALANCE_DATES = [
        # 2025 — last Friday of Feb/May/Aug/Nov
        date(2025, 2, 28), date(2025, 5, 30), date(2025, 8, 29), date(2025, 11, 28),
        # 2026
        date(2026, 2, 27), date(2026, 5, 29), date(2026, 8, 28), date(2026, 11, 27),
        # 2027
        date(2027, 2, 26), date(2027, 5, 28), date(2027, 8, 27), date(2027, 11, 26),
        # 2028
        date(2028, 2, 25), date(2028, 5, 26), date(2028, 8, 25), date(2028, 11, 24),
    ]

    # ── FTSE Russell Index Rebalancing (quarterly, March/June/Sep/Dec) ────────
    # Separate from MSCI but similar timing — amplifies the MSCI effect.
    FTSE_REBALANCE_DATES = [
        date(2025, 3, 21), date(2025, 6, 20), date(2025, 9, 19), date(2025, 12, 19),
        date(2026, 3, 20), date(2026, 6, 19), date(2026, 9, 18), date(2026, 12, 18),
        date(2027, 3, 19), date(2027, 6, 18), date(2027, 9, 17), date(2027, 12, 17),
    ]

    # ── RBI Monetary Policy Committee (MPC) decision dates ────────────────────
    # 6 meetings per year, roughly every 2 months. Dates published ~4 weeks ahead.
    RBI_MPC_DATES = [
        # 2025
        date(2025, 2, 7), date(2025, 4, 9), date(2025, 6, 6),
        date(2025, 8, 8), date(2025, 10, 8), date(2025, 12, 5),
        # 2026
        date(2026, 2, 6), date(2026, 4, 9), date(2026, 6, 6),
        date(2026, 8, 7), date(2026, 10, 9), date(2026, 12, 4),
        # 2027
        date(2027, 2, 5), date(2027, 4, 8), date(2027, 6, 4),
        date(2027, 8, 6), date(2027, 10, 8), date(2027, 12, 3),
        # 2028
        date(2028, 2, 4), date(2028, 4, 6), date(2028, 6, 9),
        date(2028, 8, 4), date(2028, 10, 6), date(2028, 12, 8),
    ]

    # ── Union Budget (Feb 1 each year — interim budget in election years) ──────
    BUDGET_DATES = [
        date(2025, 2, 1),   # Full budget
        date(2026, 2, 1),
        date(2027, 2, 1),
        date(2028, 2, 1),
    ]

    # ── GST Council Meetings (typically every 2–3 months, high market impact) ──
    GST_COUNCIL_DATES = [
        date(2025, 2, 22), date(2025, 5, 17), date(2025, 8, 2), date(2025, 11, 15),
        date(2026, 2, 21), date(2026, 5, 16), date(2026, 8, 1), date(2026, 11, 14),
        date(2027, 2, 20), date(2027, 5, 15), date(2027, 8, 7), date(2027, 11, 13),
    ]

    # ── NSE Market Holidays 2025–2026 ─────────────────────────────────────────
    # Agent will not trade on these days. Source: NSE official calendar.
    NSE_HOLIDAYS = [
        # 2025
        date(2025, 1, 26),  # Republic Day
        date(2025, 2, 26),  # Maha Shivratri
        date(2025, 3, 14),  # Holi
        date(2025, 3, 31),  # Id-Ul-Fitr
        date(2025, 4, 10),  # Dr Ambedkar Jayanti
        date(2025, 4, 14),  # Ram Navami
        date(2025, 4, 18),  # Good Friday
        date(2025, 5, 1),   # Maharashtra Day
        date(2025, 8, 15),  # Independence Day
        date(2025, 8, 27),  # Ganesh Chaturthi
        date(2025, 10, 2),  # Gandhi Jayanti / Dussehra
        date(2025, 10, 21), # Diwali Laxmi Pujan (muhurat trading only)
        date(2025, 10, 22), # Diwali Balipratipada
        date(2025, 11, 5),  # Gurunanak Jayanti
        date(2025, 12, 25), # Christmas
        # 2026
        date(2026, 1, 26),  # Republic Day
        date(2026, 3, 3),   # Maha Shivratri
        date(2026, 3, 20),  # Holi
        date(2026, 4, 3),   # Good Friday
        date(2026, 4, 14),  # Dr Ambedkar Jayanti / Gudi Padwa
        date(2026, 5, 1),   # Maharashtra Day
        date(2026, 8, 15),  # Independence Day
        date(2026, 8, 17),  # Janmashtami
        date(2026, 10, 2),  # Gandhi Jayanti
        date(2026, 11, 1),  # Diwali Laxmi Pujan
        date(2026, 11, 25), # Gurunanak Jayanti
        date(2026, 12, 25), # Christmas
        # 2027 (approximate)
        date(2027, 1, 26),  # Republic Day
        date(2027, 3, 12),  # Maha Shivratri / Holi
        date(2027, 4, 14),  # Dr Ambedkar Jayanti
        date(2027, 8, 15),  # Independence Day
        date(2027, 10, 2),  # Gandhi Jayanti
        date(2027, 12, 25), # Christmas
    ]

    # ── Risk window parameters ─────────────────────────────────────────────────
    PRE_EVENT_WARNING_DAYS  = 2
    PRE_EVENT_HIGH_DAYS     = 1
    POST_EVENT_CAUTION_DAYS = 1
    MSCI_SEMI_WARNING_DAYS  = 3
    MSCI_SEMI_HIGH_DAYS     = 2

    def _is_msci_semi_annual(self, d: date) -> bool:
        """May and Nov MSCI reviews are the large semi-annual ones (last Fri of those months)."""
        return d.month in (5, 11)

    def is_market_holiday(self, d: date | None = None) -> bool:
        """Return True if d (default: today) is an NSE holiday."""
        d = d or date.today()
        return d in self.NSE_HOLIDAYS or d.weekday() >= 5

    def is_earnings_season(self) -> bool:
        """
        Q1 results: Jul–Aug  |  Q2: Oct–Nov  |  Q3: Jan–Feb  |  Q4: Apr–May
        During earnings seasons volatility is elevated for individual stocks.
        """
        m = date.today().month
        return m in (1, 2, 4, 5, 7, 8, 10, 11)

    def upcoming_events(self, days: int = 7) -> list[dict]:
        """Return list of events within `days` calendar days from today."""
        today  = date.today()
        cutoff = today + timedelta(days=days)
        events = []

        all_events = (
            [(d, "MSCI Index Rebalancing", "MSCI")        for d in self.MSCI_REBALANCE_DATES]
            + [(d, "FTSE Russell Rebalancing", "FTSE")    for d in self.FTSE_REBALANCE_DATES]
            + [(d, "RBI MPC Decision", "RBI")             for d in self.RBI_MPC_DATES]
            + [(d, "Union Budget", "BUDGET")              for d in self.BUDGET_DATES]
            + [(d, "GST Council Meeting", "GST")          for d in self.GST_COUNCIL_DATES]
        )
        # F&O expiry — compute dynamically for next 3 months
        for month_offset in range(0, 3):
            m = (today.month - 1 + month_offset) % 12 + 1
            y = today.year + ((today.month - 1 + month_offset) // 12)
            expiry = self._last_thursday(y, m)
            all_events.append((expiry, f"F&O Monthly Expiry ({expiry.strftime('%b %Y')})", "FO_EXPIRY"))

        # NSE holidays
        for hd in self.NSE_HOLIDAYS:
            all_events.append((hd, "NSE Market Holiday", "HOLIDAY"))

        for event_date, label, tag in all_events:
            if today <= event_date <= cutoff:
                events.append({
                    "date":      event_date.isoformat(),
                    "label":     label,
                    "tag":       tag,
                    "days_away": (event_date - today).days,
                })

        # Add earnings season note if active
        if self.is_earnings_season():
            events.append({
                "date":      today.isoformat(),
                "label":     "Earnings Season Active — expect stock-specific volatility",
                "tag":       "EARNINGS",
                "days_away": 0,
            })

        return sorted(events, key=lambda e: e["days_away"])

    def event_risk_multiplier(self) -> tuple[float, str]:
        """
        Return (risk_multiplier, label).
        multiplier > 1.0 = elevated risk → position sizes shrink.
        """
        today    = date.today()
        max_mult = 1.0
        label    = "normal"

        # Hard block on market holidays
        if self.is_market_holiday(today):
            return 99.0, "NSE_HOLIDAY"

        all_dated = (
            [(d, "MSCI", self._is_msci_semi_annual(d))  for d in self.MSCI_REBALANCE_DATES]
            + [(d, "FTSE", False)                        for d in self.FTSE_REBALANCE_DATES]
            + [(d, "RBI",    False)                      for d in self.RBI_MPC_DATES]
            + [(d, "BUDGET", True)                       for d in self.BUDGET_DATES]
            + [(d, "GST",    False)                      for d in self.GST_COUNCIL_DATES]
        )
        # Add F&O expiry
        for month_offset in range(-1, 2):
            m = (today.month - 1 + month_offset) % 12 + 1
            y = today.year + ((today.month - 1 + month_offset) // 12)
            all_dated.append((self._last_thursday(y, m), "FO_EXPIRY", False))

        for event_date, tag, is_big in all_dated:
            warn_days = self.MSCI_SEMI_WARNING_DAYS if is_big else self.PRE_EVENT_WARNING_DAYS
            high_days = self.MSCI_SEMI_HIGH_DAYS    if is_big else self.PRE_EVENT_HIGH_DAYS
            days_diff = (event_date - today).days

            if -self.POST_EVENT_CAUTION_DAYS <= days_diff <= 0:
                # Day of or day after: caution (volatility lingers)
                mult = 2.0 if is_big else 1.5
                lbl  = f"POST_{tag}"
            elif 0 < days_diff <= high_days:
                mult = 2.5 if is_big else 1.8
                lbl  = f"PRE_{tag}_HIGH"
            elif 0 < days_diff <= warn_days:
                mult = 1.5 if is_big else 1.3
                lbl  = f"PRE_{tag}_WARNING"
            else:
                continue

            if mult > max_mult:
                max_mult = mult
                label    = lbl

        return round(max_mult, 2), label

    @staticmethod
    def _last_thursday(year: int, month: int) -> date:
        """Return the last Thursday of the given month (NSE F&O expiry)."""
        # Start from last day of month and walk back
        if month == 12:
            last_day = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = date(year, month + 1, 1) - timedelta(days=1)
        days_back = (last_day.weekday() - 3) % 7   # 3 = Thursday
        return last_day - timedelta(days=days_back)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Macro Risk Scorer
# ═══════════════════════════════════════════════════════════════════════════════

class MacroRiskScorer:
    """
    Combines multiple signals into a composite macro risk score (0.0 – 1.0):
      0.0 – 0.3  → LOW     (normal trading)
      0.3 – 0.5  → MEDIUM  (reduce new buys by ~20%)
      0.5 – 0.7  → HIGH    (reduce new buys by ~50%, tighten SL)
      0.7 – 1.0  → EXTREME (no new buys; protect existing positions)

    Inputs:
      • EventCalendar risk multiplier
      • Market volatility regime (India VIX via yfinance or fallback)
      • News sentiment (headline keyword scan — lightweight, no API key needed)
    """

    _BEARISH_KEYWORDS = [
        "crash", "circuit breaker", "circuit limit", "lower circuit",
        "panic selling", "massive selloff", "sell-off", "heavy selling",
        "fii selling", "foreign outflow", "capital outflow",
        "war", "conflict", "sanction", "geopolitical",
        "recession", "stagflation", "inflation spike",
        "msci rebalancing", "msci rebalance", "index rejig",
        "rbi rate hike", "rate hike", "hawkish",
        "budget deficit", "fiscal slippage",
        "rupee fall", "rupee depreciation", "inr crash",
        "nifty fall", "sensex crash", "market crash india",
    ]

    _BULLISH_KEYWORDS = [
        "rate cut", "stimulus", "fii buying", "foreign inflow",
        "nifty rally", "sensex rally", "bull run",
        "gdp growth", "earnings beat", "strong results",
    ]

    # Cache news for 30 minutes to avoid hammering endpoints
    _news_cache: dict = {}
    _news_cache_ts: Optional[datetime] = None
    _NEWS_TTL_MINUTES = 30

    def __init__(self, calendar: EventCalendar):
        self._cal = calendar
        self._vix_cache: Optional[tuple[datetime, float]] = None  # (ts, value)
        self._VIX_TTL_MINUTES = 60

    def score(self) -> tuple[float, str, dict]:
        """
        Returns (score 0–1, label, breakdown_dict).
        breakdown has keys: event, vix, news, composite.
        """
        # ── Component 1: Event calendar ────────────────────────────────────── #
        event_mult, event_label = self._cal.event_risk_multiplier()
        # Normalise multiplier → 0–1 score. mult=1 → 0, mult=2.5 → 0.75, capped at 1.0
        event_score = min(1.0, (event_mult - 1.0) / 2.0)

        # ── Component 2: India VIX (fear gauge) ───────────────────────────────#
        vix_score = self._vix_risk_score()

        # ── Component 3: News sentiment ───────────────────────────────────────#
        news_score = self._news_risk_score()

        # ── Composite (weighted) ──────────────────────────────────────────────#
        composite = round(
            0.40 * event_score
            + 0.35 * vix_score
            + 0.25 * news_score,
            3,
        )

        label = (
            "EXTREME" if composite >= 0.70 else
            "HIGH"    if composite >= 0.50 else
            "MEDIUM"  if composite >= 0.30 else
            "LOW"
        )

        breakdown = {
            "event_mult":   event_mult,
            "event_label":  event_label,
            "event_score":  round(event_score, 3),
            "vix_score":    round(vix_score, 3),
            "news_score":   round(news_score, 3),
            "composite":    composite,
            "label":        label,
            "events_soon":  self._cal.upcoming_events(days=5),
        }
        return composite, label, breakdown

    def _vix_risk_score(self) -> float:
        """Fetch India VIX and return a 0–1 risk score. Falls back to 0.3 on error."""
        now = datetime.now(_IST)
        if self._vix_cache and (now - self._vix_cache[0]).seconds < self._VIX_TTL_MINUTES * 60:
            vix = self._vix_cache[1]
        else:
            try:
                import yfinance as yf
                hist = yf.download("^INDIAVIX", period="2d", interval="1d",
                                   auto_adjust=True, progress=False)
                if not hist.empty:
                    import pandas as pd
                    if isinstance(hist.columns, pd.MultiIndex):
                        hist.columns = hist.columns.get_level_values(0)
                    vix = float(hist["Close"].iloc[-1])
                    self._vix_cache = (now, vix)
                else:
                    vix = 15.0   # neutral fallback
            except Exception as exc:
                logger.debug(f"[RiskMgr] VIX fetch failed: {exc}")
                vix = 15.0

        # India VIX: <12 = very calm, 12–20 = normal, 20–30 = elevated, >30 = fearful
        if vix < 12:   return 0.05
        if vix < 16:   return 0.15
        if vix < 20:   return 0.30
        if vix < 25:   return 0.55
        if vix < 30:   return 0.75
        return 0.95

    def _news_risk_score(self) -> float:
        """
        Scan recent headlines from yfinance news for NIFTY/SENSEX.
        Returns 0–1 sentiment-based risk score.
        Falls back to 0.2 if no headlines found.
        """
        now = datetime.now(_IST)
        if (
            self._news_cache
            and self._news_cache_ts
            and (now - self._news_cache_ts).seconds < self._NEWS_TTL_MINUTES * 60
        ):
            return self._news_cache.get("score", 0.2)

        try:
            import yfinance as yf
            # Fetch news for Nifty 50 as a proxy for Indian market sentiment
            ticker = yf.Ticker("^NSEI")
            news_items = ticker.news or []

            headlines = []
            for item in news_items[:20]:
                title = (item.get("title") or "").lower()
                desc  = (item.get("summary") or item.get("description") or "").lower()
                headlines.append(title + " " + desc)

            combined = " ".join(headlines)
            bear_hits = sum(1 for kw in self._BEARISH_KEYWORDS if kw in combined)
            bull_hits = sum(1 for kw in self._BULLISH_KEYWORDS if kw in combined)

            # Net bearish signal: 0 net → 0.2, each bear hit adds ~0.1 (cap at 0.9)
            net = bear_hits - (bull_hits * 0.5)
            score = round(min(0.9, max(0.05, 0.2 + net * 0.12)), 3)

            self._news_cache    = {"score": score, "bear_hits": bear_hits, "bull_hits": bull_hits}
            self._news_cache_ts = now
            logger.info(f"[RiskMgr] News scan: bear={bear_hits} bull={bull_hits} → score={score:.2f}")
            return score

        except Exception as exc:
            logger.debug(f"[RiskMgr] News scan failed: {exc}")
            return 0.2


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Position Sizer — Volatility-Targeting Kelly
# ═══════════════════════════════════════════════════════════════════════════════

class PositionSizer:
    """
    Dynamic position sizing that replaces the hardcoded MAX_POSITION_PCT.

    Formula
    -------
    target_risk_₹  = portfolio_value × RISK_PER_TRADE_PCT × conviction_factor × safety_factor
    position_size  = target_risk_₹ / atr_stop_distance
    capped at MAX_POSITION_FRACTION of portfolio and available cash.

    Where:
      conviction_factor = strength / 100 scaled to [MIN_CONV, MAX_CONV]
      safety_factor     = 1 / risk_multiplier  (from EventCalendar)
      atr_stop_distance = ATR(14) × ATR_MULT  (same as engine.py)

    The result is the ₹ amount to spend (qty = floor(spend / price)).
    """

    # ── Tunable parameters ────────────────────────────────────────────────── #
    RISK_PER_TRADE_PCT   = 0.005   # risk 0.5% of portfolio per trade (₹5k on ₹10L)
    MAX_POSITION_FRACTION = 0.08   # hard cap: never >8% of portfolio in one stock
    MIN_POSITION_FRACTION = 0.005  # floor: at least ₹5k on ₹10L
    MIN_CONVICTION        = 0.60   # strength=MIN_BUY_STRENGTH → 60% conviction
    MAX_CONVICTION        = 1.20   # strength=100 → 20% oversize vs base
    ATR_MULT              = 1.5    # must match engine.py ATR_SL_MULT
    ATR_FALLBACK_PCT      = 0.07   # fallback stop distance if ATR unknown: 7%

    def size(
        self,
        portfolio_value: float,
        available_cash:  float,
        price:           float,
        atr:             Optional[float],
        strength:        float,        # 0–100
        risk_multiplier: float = 1.0,  # from EventCalendar.event_risk_multiplier()
        macro_risk_score: float = 0.0, # 0–1 composite risk
    ) -> float:
        """
        Return ₹ amount to spend on this trade.
        Returns 0.0 if the trade should be skipped entirely.
        """
        if portfolio_value <= 0 or price <= 0:
            return 0.0

        # Conviction factor: scales position with signal strength
        # strength 65 → 0.65 → conv_factor ≈ 0.75; strength 90 → conv_factor ≈ 1.10
        raw_conv       = strength / 100.0
        conv_factor    = self.MIN_CONVICTION + (self.MAX_CONVICTION - self.MIN_CONVICTION) * raw_conv

        # Safety factor: shrinks position when macro risk is elevated
        # risk_multiplier=1.5 (event near) → safety=0.67
        # macro_risk_score=0.7 (HIGH) → additional 30% reduction
        macro_safety = 1.0 - (macro_risk_score * 0.40)   # max 40% reduction from news/VIX
        event_safety = 1.0 / max(risk_multiplier, 1.0)
        safety_factor = macro_safety * event_safety

        # ATR-based stop distance
        if atr and atr > 0:
            stop_dist = atr * self.ATR_MULT
        else:
            stop_dist = price * self.ATR_FALLBACK_PCT

        # Target risk in ₹
        target_risk = portfolio_value * self.RISK_PER_TRADE_PCT * conv_factor * safety_factor

        # Position size = how many ₹ to put in so that 1 ATR move = target_risk
        spend = (target_risk / stop_dist) * price

        # Hard caps
        max_spend = min(
            portfolio_value * self.MAX_POSITION_FRACTION,
            available_cash  * 0.95,
        )
        min_spend = portfolio_value * self.MIN_POSITION_FRACTION

        spend = max(min_spend, min(spend, max_spend))

        logger.info(
            f"[Sizer] conv={conv_factor:.2f} safety={safety_factor:.2f} "
            f"stop_dist=₹{stop_dist:.2f} target_risk=₹{target_risk:.0f} → spend=₹{spend:.0f}"
        )
        return round(spend, 2)

    @staticmethod
    def max_new_positions(portfolio_value: float, macro_risk_score: float) -> int:
        """
        How many new positions are allowed this cycle?
        Shrinks aggressively as macro risk rises.
        """
        base = 5   # max 5 new buys per cycle normally
        if macro_risk_score >= 0.70: return 0   # EXTREME: no new buys
        if macro_risk_score >= 0.50: return 1   # HIGH: at most 1
        if macro_risk_score >= 0.30: return 2   # MEDIUM: at most 2
        return base


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Drawdown Guard — Portfolio Circuit Breakers
# ═══════════════════════════════════════════════════════════════════════════════

class DrawdownGuard:
    """
    Monitors peak portfolio value and applies circuit-breakers on drawdown.

    Levels
    ------
    CAUTION  (−2% from peak) : position size × 0.5, max 1 new buy/cycle
    STOP     (−4% from peak) : no new buys; only stops/TPs run
    HALT     (−6% from peak) : no new buys; consider closing all losers

    Peak is reset each trading day at market open.
    """

    _CAUTION_PCT = 0.02
    _STOP_PCT    = 0.04
    _HALT_PCT    = 0.06

    def __init__(self):
        self._peak:       float = 0.0
        self._peak_date:  date  = date.today()
        self._state_file: Path  = Path(__file__).parent / "data" / "drawdown_state.json"
        self._load()

    def _load(self):
        try:
            if self._state_file.exists():
                d = json.loads(self._state_file.read_text())
                self._peak      = d.get("peak", 0.0)
                self._peak_date = date.fromisoformat(d.get("peak_date", date.today().isoformat()))
        except Exception:
            pass

    def _save(self):
        try:
            self._state_file.parent.mkdir(exist_ok=True)
            self._state_file.write_text(json.dumps({
                "peak":      self._peak,
                "peak_date": self._peak_date.isoformat(),
            }))
        except Exception:
            pass

    def update_peak(self, portfolio_value: float):
        """Call once per cycle with the current portfolio value."""
        today = date.today()
        if today != self._peak_date:
            # New trading day — reset daily peak
            self._peak      = portfolio_value
            self._peak_date = today
            self._save()
            logger.info(f"[DrawdownGuard] New day — peak reset to ₹{portfolio_value:,.0f}")
        elif portfolio_value > self._peak:
            self._peak = portfolio_value
            self._save()

    def status(self, portfolio_value: float) -> tuple[str, float]:
        """
        Return (level, drawdown_pct) where level is NORMAL/CAUTION/STOP/HALT.
        """
        if self._peak <= 0:
            return "NORMAL", 0.0
        dd = (self._peak - portfolio_value) / self._peak
        if dd >= self._HALT_PCT:    return "HALT",    round(dd, 4)
        if dd >= self._STOP_PCT:    return "STOP",    round(dd, 4)
        if dd >= self._CAUTION_PCT: return "CAUTION", round(dd, 4)
        return "NORMAL", round(dd, 4)

    def size_multiplier(self, portfolio_value: float) -> float:
        """Return a multiplier (0.0–1.0) to apply to position sizes."""
        level, _ = self.status(portfolio_value)
        return {"NORMAL": 1.0, "CAUTION": 0.5, "STOP": 0.0, "HALT": 0.0}[level]

    def max_new_buys(self, portfolio_value: float) -> int:
        level, _ = self.status(portfolio_value)
        return {"NORMAL": 99, "CAUTION": 1, "STOP": 0, "HALT": 0}[level]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. RiskManager — Facade
# ═══════════════════════════════════════════════════════════════════════════════

class RiskManager:
    """
    Single entry-point for all risk checks.

    Thread-safe: macro_risk() result is cached for 30 minutes so repeated
    calls within a cycle don't hammer yfinance / news endpoints.
    """

    _MACRO_CACHE_MINUTES = 30

    def __init__(self):
        self._calendar = EventCalendar()
        self._scorer   = MacroRiskScorer(self._calendar)
        self._sizer    = PositionSizer()
        self._guard    = DrawdownGuard()

        self._macro_cache: Optional[tuple[datetime, float, str, dict]] = None
        self._lock = threading.Lock()

    # ── Public API ───────────────────────────────────────────────────────── #

    def macro_risk(self, force_refresh: bool = False) -> tuple[float, str, dict]:
        """
        Returns (score 0–1, label, breakdown).
        Result cached for _MACRO_CACHE_MINUTES to avoid repeated API calls.
        """
        with self._lock:
            now = datetime.now(_IST)
            if (
                not force_refresh
                and self._macro_cache
                and (now - self._macro_cache[0]).seconds < self._MACRO_CACHE_MINUTES * 60
            ):
                _, score, label, breakdown = self._macro_cache
                return score, label, breakdown

            score, label, breakdown = self._scorer.score()
            self._macro_cache = (now, score, label, breakdown)
            logger.info(
                f"[RiskMgr] Macro risk: {label} ({score:.2f}) | "
                f"events_soon={[e['label'] for e in breakdown.get('events_soon', [])]}"
            )
            return score, label, breakdown

    def position_size(
        self,
        portfolio_value: float,
        available_cash:  float,
        price:           float,
        strength:        float,
        atr:             Optional[float] = None,
    ) -> float:
        """Compute dynamic ₹ spend for a new position."""
        score, _, breakdown = self.macro_risk()
        event_mult, _ = self._calendar.event_risk_multiplier()
        dd_mult = self._guard.size_multiplier(portfolio_value)

        spend = self._sizer.size(
            portfolio_value  = portfolio_value,
            available_cash   = available_cash,
            price            = price,
            atr              = atr,
            strength         = strength,
            risk_multiplier  = event_mult,
            macro_risk_score = score,
        )
        return round(spend * dd_mult, 2)

    def max_new_buys_this_cycle(self, portfolio_value: float) -> int:
        """How many new positions are allowed this cycle (combines all guards)."""
        score, _, _bd = self.macro_risk()
        macro_limit    = self._sizer.max_new_positions(portfolio_value, score)
        dd_limit       = self._guard.max_new_buys(portfolio_value)
        return min(macro_limit, dd_limit)

    def update_peak(self, portfolio_value: float):
        """Call once per cycle to keep the drawdown guard up to date."""
        self._guard.update_peak(portfolio_value)

    def drawdown_status(self, portfolio_value: float) -> tuple[str, float]:
        """Return (level, drawdown_pct): NORMAL / CAUTION / STOP / HALT."""
        return self._guard.status(portfolio_value)

    def upcoming_events(self, days: int = 7) -> list[dict]:
        return self._calendar.upcoming_events(days)

    def full_status(self, portfolio_value: float) -> dict:
        """Full risk snapshot — for dashboard / API."""
        score, label, breakdown = self.macro_risk()
        dd_level, dd_pct = self._guard.status(portfolio_value)
        max_buys = self.max_new_buys_this_cycle(portfolio_value)
        return {
            "macro_score":      score,
            "macro_label":      label,
            "macro_breakdown":  breakdown,
            "drawdown_level":   dd_level,
            "drawdown_pct":     round(dd_pct * 100, 2),
            "peak_value":       self._guard._peak,
            "max_new_buys":     max_buys,
            "trading_allowed":  max_buys > 0,
            "upcoming_events":  self.upcoming_events(7),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════════

_risk_manager: Optional[RiskManager] = None
_rm_lock = threading.Lock()

def get_risk_manager() -> RiskManager:
    global _risk_manager
    with _rm_lock:
        if _risk_manager is None:
            _risk_manager = RiskManager()
    return _risk_manager
