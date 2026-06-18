"""
Fundamental Screener Agent
==========================
Weekly fundamental scoring of the Nifty 50 universe using yfinance.
Scores are cached in data/fundamental_scores.json (24-hour TTL).
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta

import yfinance as yf

from engine import NIFTY50_TICKERS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_FILE = os.path.join(_BASE_DIR, "data", "fundamental_scores.json")
_CACHE_TTL_HOURS = 24

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_FUND_INSTANCE = None
_FUND_LOCK = threading.Lock()


def get_fundamental_agent() -> "FundamentalScreener":
    global _FUND_INSTANCE
    if _FUND_INSTANCE is None:
        with _FUND_LOCK:
            if _FUND_INSTANCE is None:
                _FUND_INSTANCE = FundamentalScreener()
    return _FUND_INSTANCE


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_revenue_growth(val) -> int:
    if val is None:
        return 0
    if val > 0.15:
        return 20
    if val > 0.10:
        return 12
    if val > 0.05:
        return 6
    return 0


def _score_earnings_growth(val) -> int:
    if val is None:
        return 0
    if val > 0.20:
        return 20
    if val > 0.10:
        return 12
    if val > 0.00:
        return 6
    return 0


def _score_roe(val) -> int:
    if val is None:
        return 0
    if val > 0.20:
        return 20
    if val > 0.15:
        return 12
    if val > 0.10:
        return 6
    return 0


def _score_debt_to_equity(val) -> int:
    if val is None:
        return 0
    if val < 0.3:
        return 15
    if val < 0.7:
        return 10
    if val < 1.5:
        return 5
    return 0


def _score_profit_margins(val) -> int:
    if val is None:
        return 0
    if val > 0.15:
        return 15
    if val > 0.10:
        return 10
    if val > 0.05:
        return 5
    return 0


def _score_current_ratio(val) -> int:
    if val is None:
        return 0
    if val > 2.0:
        return 10
    if val > 1.5:
        return 7
    if val > 1.0:
        return 4
    return 0


def _derive_grade(score: int) -> str:
    if score >= 80:
        return "A"
    if score >= 65:
        return "B"
    if score >= 50:
        return "C"
    return "D"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FundamentalScreener:

    def score_stock(self, ticker: str) -> dict:
        """
        Fetch yfinance info for `ticker` and compute a fundamental score (0-100).
        Returns a scored dict with grade and key metrics.
        """
        try:
            info = yf.Ticker(ticker).info
        except Exception as exc:
            logger.warning("yfinance info fetch failed for %s: %s", ticker, exc)
            info = {}

        market_cap      = info.get("marketCap",       0) or 0
        revenue_growth  = info.get("revenueGrowth",   None)
        earnings_growth = info.get("earningsGrowth",  None)
        roe             = info.get("returnOnEquity",   None)
        d2e             = info.get("debtToEquity",     None)
        pe_ratio        = info.get("trailingPE",       None)
        pb_ratio        = info.get("priceToBook",      None)
        dividend_yield  = info.get("dividendYield",    0) or 0
        profit_margins  = info.get("profitMargins",    None)
        current_ratio   = info.get("currentRatio",     None)

        # Debt/Equity from yfinance is sometimes expressed as a raw number
        # (e.g. 45 meaning 0.45); normalise values clearly > 10 by /100.
        if d2e is not None and d2e > 10:
            d2e = d2e / 100.0

        score = (
            _score_revenue_growth(revenue_growth)
            + _score_earnings_growth(earnings_growth)
            + _score_roe(roe)
            + _score_debt_to_equity(d2e)
            + _score_profit_margins(profit_margins)
            + _score_current_ratio(current_ratio)
        )
        score = max(0, min(100, score))

        def _pct(val):
            """Convert ratio to percentage, guarding None."""
            return round(val * 100, 2) if val is not None else None

        return {
            "ticker":               ticker,
            "fundamental_score":    score,
            "grade":                _derive_grade(score),
            "market_cap_cr":        round(market_cap / 1e7, 2) if market_cap else 0,
            "revenue_growth_pct":   _pct(revenue_growth),
            "earnings_growth_pct":  _pct(earnings_growth),
            "roe_pct":              _pct(roe),
            "debt_to_equity":       round(d2e, 4) if d2e is not None else None,
            "pe_ratio":             round(pe_ratio, 2) if pe_ratio is not None else None,
            "pb_ratio":             round(pb_ratio, 2) if pb_ratio is not None else None,
            "dividend_yield_pct":   _pct(dividend_yield),
            "profit_margin_pct":    _pct(profit_margins),
            "current_ratio":        round(current_ratio, 2) if current_ratio is not None else None,
            "screened_at":          datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def run_screen(self, tickers: list = None) -> list:
        """
        Screen a list of tickers fundamentally.
        Default: first 30 tickers from NIFTY50_TICKERS.
        Results cached to data/fundamental_scores.json (24-hour TTL).
        Returns list sorted by fundamental_score descending.
        """
        if tickers is None:
            tickers = NIFTY50_TICKERS[:30]

        # Serve from cache if fresh enough
        cached = self._load_cache()
        if cached is not None:
            logger.info("Returning fundamental scores from cache (%d entries).", len(cached))
            return cached

        results = []
        for i, ticker in enumerate(tickers):
            logger.info("Scoring %s (%d/%d)…", ticker, i + 1, len(tickers))
            try:
                scored = self.score_stock(ticker)
                results.append(scored)
            except Exception as exc:
                logger.warning("Failed to score %s: %s", ticker, exc)
            if i < len(tickers) - 1:
                time.sleep(0.5)

        results.sort(key=lambda r: r["fundamental_score"], reverse=True)
        self._save_cache(results)
        return results

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_cache(self) -> list | None:
        """Load cached scores if the file exists and is < 24 hours old."""
        if not os.path.exists(_CACHE_FILE):
            return None
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(_CACHE_FILE))
            if datetime.now() - mtime > timedelta(hours=_CACHE_TTL_HOURS):
                return None
            with open(_CACHE_FILE, "r") as fh:
                data = json.load(fh)
            if isinstance(data, list) and data:
                return data
        except Exception as exc:
            logger.warning("Could not load fundamental cache: %s", exc)
        return None

    def _save_cache(self, results: list) -> None:
        """Persist scored results to the cache file."""
        try:
            os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
            with open(_CACHE_FILE, "w") as fh:
                json.dump(results, fh, indent=2)
        except Exception as exc:
            logger.warning("Could not save fundamental cache: %s", exc)

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def get_dashboard_data(self) -> dict:
        """
        Return a dashboard-ready summary of the latest fundamental screen.
        Triggers run_screen() if no fresh cache exists.
        """
        results = self.run_screen()

        grade_distribution: dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0}
        for r in results:
            grade = r.get("grade", "D")
            grade_distribution[grade] = grade_distribution.get(grade, 0) + 1

        last_screened = results[0].get("screened_at") if results else None

        return {
            "top_10":            results[:10],
            "bottom_5":          results[-5:] if len(results) >= 5 else results,
            "grade_distribution": grade_distribution,
            "last_screened":     last_screened,
            "total_screened":    len(results),
        }
