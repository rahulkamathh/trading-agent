"""
Fundamental Analyzer
====================
Fetches key fundamental metrics from yfinance for NSE stocks,
caches them daily, computes a composite quality score (0–1),
and re-scores / filters technical signals based on fundamentals.

Score components
----------------
  P/E ratio      : lower is better (vs sector median)
  P/B ratio      : lower is better (< 3 = good)
  ROE            : higher is better (> 15% = good)
  D/E ratio      : lower is better (< 1 = good)
  Revenue growth : higher is better (> 10% YoY = good)
  Profit margin  : higher is better (> 10% = good)
  EPS growth     : higher is better
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

FUNDAMENTALS_FILE = DATA_DIR / "fundamentals.json"
CACHE_TTL_HOURS   = 24          # re-fetch once a day
MIN_SCORE_BUY     = 0.35        # below this → veto BUY signals
BOOST_THRESHOLD   = 0.65        # above this → boost signal strength
BOOST_FACTOR      = 1.25        # multiply strength by this on good fundamentals
PENALTY_FACTOR    = 0.70        # multiply strength by this on poor fundamentals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _safe(d: dict, *keys, default=None):
    """Safely navigate nested dict / None."""
    val = d
    for k in keys:
        if not isinstance(val, dict):
            return default
        val = val.get(k, default)
    return val


# ---------------------------------------------------------------------------
# Fundamental Analyzer
# ---------------------------------------------------------------------------

class FundamentalAnalyzer:

    def __init__(self):
        self._cache: dict = {}          # ticker → {score, metrics, fetched_at}
        self._load_cache()

    # ------------------------------------------------------------------ cache

    def _load_cache(self):
        if FUNDAMENTALS_FILE.exists():
            try:
                with open(FUNDAMENTALS_FILE) as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

    def _save_cache(self):
        try:
            with open(FUNDAMENTALS_FILE, "w") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save fundamentals cache: {e}")

    def _is_stale(self, ticker: str) -> bool:
        entry = self._cache.get(ticker)
        if not entry:
            return True
        fetched = entry.get("fetched_at", "")
        if not fetched:
            return True
        try:
            age = datetime.now() - datetime.fromisoformat(fetched)
            return age > timedelta(hours=CACHE_TTL_HOURS)
        except Exception:
            return True

    # ------------------------------------------------------------------ fetch

    def _fetch_ticker_data(self, ticker: str) -> dict:
        """Pull fundamental metrics from yfinance info dict."""
        try:
            info = yf.Ticker(ticker).info
        except Exception as e:
            logger.debug(f"yfinance info failed for {ticker}: {e}")
            return {}

        # ── P/E ──
        pe = info.get("trailingPE") or info.get("forwardPE")

        # ── P/B ──
        pb = info.get("priceToBook")

        # ── ROE ──
        roe = info.get("returnOnEquity")
        if roe is not None:
            roe = roe * 100  # convert to %

        # ── D/E ──
        de = info.get("debtToEquity")
        if de is not None:
            de = de / 100  # yfinance returns as %, normalise to ratio

        # ── Revenue growth (trailing YoY from financials) ──
        rev_growth = info.get("revenueGrowth")
        if rev_growth is not None:
            rev_growth = rev_growth * 100  # to %

        # ── Profit margin ──
        margin = info.get("profitMargins")
        if margin is not None:
            margin = margin * 100  # to %

        # ── EPS growth (trailing) ──
        eps_growth = info.get("earningsGrowth")
        if eps_growth is not None:
            eps_growth = eps_growth * 100  # to %

        # ── Market cap ──
        mcap = info.get("marketCap")

        # ── Sector / industry ──
        sector   = info.get("sector", "")
        industry = info.get("industry", "")

        return {
            "pe":         pe,
            "pb":         pb,
            "roe":        roe,
            "de":         de,
            "rev_growth": rev_growth,
            "margin":     margin,
            "eps_growth": eps_growth,
            "mcap":       mcap,
            "sector":     sector,
            "industry":   industry,
        }

    # ------------------------------------------------------------------ score

    def _compute_score(self, m: dict) -> float:
        """
        Composite quality score 0–1.
        Each component contributes equally (1/6 weight).
        """
        components = []

        # P/E score: ideal < 15, neutral at 30, bad > 60
        pe = m.get("pe")
        if pe and pe > 0:
            pe_score = _clamp(1 - (pe - 5) / 55)
        else:
            pe_score = 0.5  # no data → neutral
        components.append(pe_score)

        # P/B score: ideal < 1.5, bad > 5
        pb = m.get("pb")
        if pb and pb > 0:
            pb_score = _clamp(1 - (pb - 0.5) / 5.5)
        else:
            pb_score = 0.5
        components.append(pb_score)

        # ROE score: ideal > 20%, bad < 5%
        roe = m.get("roe")
        if roe is not None:
            roe_score = _clamp((roe - 5) / 25)
        else:
            roe_score = 0.5
        components.append(roe_score)

        # D/E score: ideal < 0.3, bad > 2
        de = m.get("de")
        if de is not None and de >= 0:
            de_score = _clamp(1 - de / 2.0)
        else:
            de_score = 0.5
        components.append(de_score)

        # Revenue growth score: ideal > 20%, bad < 0%
        rg = m.get("rev_growth")
        if rg is not None:
            rg_score = _clamp((rg + 5) / 30)
        else:
            rg_score = 0.5
        components.append(rg_score)

        # Profit margin score: ideal > 20%, bad < 0%
        mg = m.get("margin")
        if mg is not None:
            mg_score = _clamp((mg + 2) / 25)
        else:
            mg_score = 0.5
        components.append(mg_score)

        return round(float(np.mean(components)), 3)

    # ------------------------------------------------------------------ public

    def get_score(self, ticker: str) -> dict:
        """
        Return {score, metrics, grade} for *ticker*.
        Uses cache; fetches fresh data if stale.
        """
        if self._is_stale(ticker):
            m = self._fetch_ticker_data(ticker)
            score = self._compute_score(m)
            grade = self._grade(score)
            self._cache[ticker] = {
                "score":      score,
                "grade":      grade,
                "metrics":    m,
                "fetched_at": datetime.now().isoformat(),
            }
            self._save_cache()

        entry = self._cache.get(ticker, {})
        return {
            "score":   entry.get("score", 0.5),
            "grade":   entry.get("grade", "C"),
            "metrics": entry.get("metrics", {}),
        }

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 0.75:   return "A"
        elif score >= 0.60: return "B"
        elif score >= 0.45: return "C"
        elif score >= 0.30: return "D"
        else:               return "F"

    def score_stock(self, ticker: str) -> float:
        """Convenience: return just the numeric score."""
        return self.get_score(ticker)["score"]

    def rescore_signals(self, signals: list) -> list:
        """
        Given a list of signal dicts (from engine.py strategies),
        adjust each signal's 'strength' field based on fundamentals.

        Rules
        -----
        • BUY  + good fundamentals (≥ BOOST_THRESHOLD)  → boost strength ×1.25
        • BUY  + poor fundamentals (< MIN_SCORE_BUY)    → mark vetoed, strength ×0.7
        • SELL + poor fundamentals                       → boost (confirms short thesis)
        • SELL + good fundamentals                       → slight penalty (don't short quality)
        """
        rescored = []
        for sig in signals:
            ticker    = sig.get("ticker", "")
            action    = sig.get("action", "BUY")
            strength  = float(sig.get("strength", 0.5))

            # Only score actual equities — skip pure ETF/index signals
            if not ticker.endswith(".NS") or ticker in {"^NSEI", "^NSEBANK"}:
                sig["fund_score"] = None
                sig["fund_grade"] = "-"
                rescored.append(sig)
                continue

            try:
                result     = self.get_score(ticker)
                fund_score = result["score"]
                fund_grade = result["grade"]
            except Exception as e:
                logger.debug(f"Fundamental score failed for {ticker}: {e}")
                fund_score = 0.5
                fund_grade = "C"

            if action == "BUY":
                if fund_score >= BOOST_THRESHOLD:
                    strength = min(1.0, strength * BOOST_FACTOR)
                elif fund_score < MIN_SCORE_BUY:
                    strength = strength * PENALTY_FACTOR
                    sig["vetoed_by_fundamentals"] = True

            elif action == "SELL":
                if fund_score < MIN_SCORE_BUY:
                    strength = min(1.0, strength * BOOST_FACTOR)   # confirms short
                elif fund_score >= BOOST_THRESHOLD:
                    strength = strength * PENALTY_FACTOR            # don't short quality

            sig["strength"]   = round(strength, 3)
            sig["fund_score"] = round(fund_score, 3)
            sig["fund_grade"] = fund_grade
            rescored.append(sig)

        logger.info(f"Fundamental rescoring: {len(rescored)} signals processed")
        return rescored

    def bulk_prefetch(self, tickers: list, delay: float = 0.3):
        """
        Pre-warm the cache for a list of tickers.
        Skips tickers that are already fresh.
        Useful to call once at startup.
        """
        stale = [t for t in tickers if self._is_stale(t)]
        logger.info(f"Fundamental bulk prefetch: {len(stale)} stale of {len(tickers)}")
        for ticker in stale:
            try:
                self.get_score(ticker)
                time.sleep(delay)
            except Exception as e:
                logger.debug(f"Prefetch failed {ticker}: {e}")

    def get_all_scores(self) -> dict:
        """Return all cached scores as {ticker: {score, grade, metrics}}."""
        return {
            t: {
                "score":   v.get("score"),
                "grade":   v.get("grade"),
                "metrics": v.get("metrics", {}),
            }
            for t, v in self._cache.items()
        }

    def top_stocks(self, tickers: list, n: int = 10) -> list:
        """
        Return top-N tickers sorted by fundamental score descending.
        Fetches any that are stale.
        """
        scored = []
        for t in tickers:
            try:
                r = self.get_score(t)
                scored.append((t, r["score"], r["grade"]))
            except Exception:
                scored.append((t, 0.0, "F"))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [{"ticker": t, "score": s, "grade": g} for t, s, g in scored[:n]]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_analyzer_instance: FundamentalAnalyzer | None = None


def get_analyzer() -> FundamentalAnalyzer:
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = FundamentalAnalyzer()
    return _analyzer_instance
