"""
FII/DII Flow Agent
==================
Fetches daily FII/DII institutional flow data from NSE.
Falls back to mock-but-realistic data if NSE is unreachable.
Cache TTL: 60 minutes (NSE updates once per day after market close).
"""

import json
import logging
import random
import threading
from datetime import datetime, timedelta

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_FII_INSTANCE = None
_FII_LOCK = threading.Lock()


def get_fii_agent() -> "FIIDIIAgent":
    global _FII_INSTANCE
    if _FII_INSTANCE is None:
        with _FII_LOCK:
            if _FII_INSTANCE is None:
                _FII_INSTANCE = FIIDIIAgent()
    return _FII_INSTANCE


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FIIDIIAgent:
    NSE_HOME = "https://www.nseindia.com"
    NSE_API  = "https://www.nseindia.com/api/fiidiiTradeReact"
    HEADERS  = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com/",
        "Accept-Language": "en-US,en;q=0.9",
    }
    CACHE_TTL_MINUTES = 60

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update(self.HEADERS)
        self._cache: dict | None = None
        self._cache_time: datetime | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_cache_valid(self) -> bool:
        if self._cache is None or self._cache_time is None:
            return False
        return (datetime.now() - self._cache_time).total_seconds() < self.CACHE_TTL_MINUTES * 60

    def _init_session(self) -> bool:
        """GET the NSE homepage to acquire session cookies."""
        try:
            resp = self._session.get(self.NSE_HOME, timeout=10)
            return resp.status_code == 200
        except Exception as exc:
            logger.warning("NSE homepage request failed: %s", exc)
            return False

    def _fetch_from_nse(self) -> list:
        """
        Attempt to download FII/DII data from NSE API.
        Returns a list of parsed row dicts, or raises on failure.
        """
        self._init_session()
        resp = self._session.get(self.NSE_API, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        rows = []
        # NSE returns a list of objects; each has fields like:
        # date, buyValue, sellValue (for FII and DII separately)
        # The exact schema varies; we try common field names.
        for entry in data:
            try:
                # Prefer "date" field; fall back to "tradeDate"
                raw_date = entry.get("date") or entry.get("tradeDate", "")
                # Parse to a normalised YYYY-MM-DD string
                for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y"):
                    try:
                        parsed_date = datetime.strptime(raw_date.strip(), fmt).strftime("%Y-%m-%d")
                        break
                    except ValueError:
                        continue
                else:
                    parsed_date = raw_date  # keep as-is if unparseable

                fii_buy  = float(entry.get("fiiBuyValue")  or entry.get("buyValue", 0))
                fii_sell = float(entry.get("fiiSellValue") or entry.get("sellValue", 0))
                fii_net  = round(fii_buy - fii_sell, 2)

                dii_buy  = float(entry.get("diiBuyValue",  0))
                dii_sell = float(entry.get("diiSellValue", 0))
                dii_net  = round(dii_buy - dii_sell, 2)

                rows.append({
                    "date":         parsed_date,
                    "fii_buy":      round(fii_buy,  2),
                    "fii_sell":     round(fii_sell, 2),
                    "fii_net":      fii_net,
                    "dii_buy":      round(dii_buy,  2),
                    "dii_sell":     round(dii_sell, 2),
                    "dii_net":      dii_net,
                    "combined_net": round(fii_net + dii_net, 2),
                })
            except Exception as exc:
                logger.debug("Skipping NSE row due to parse error: %s | row=%s", exc, entry)
                continue

        if not rows:
            raise ValueError("NSE API returned zero parseable rows")

        # Sort descending by date (most-recent first)
        rows.sort(key=lambda r: r["date"], reverse=True)
        return rows

    @staticmethod
    def _generate_mock_data(days: int) -> list:
        """
        Generate plausible mock FII/DII data for the last `days` trading days.
        Used as a fallback when NSE is unreachable.
        """
        random.seed(42)  # reproducible mock
        rows = []
        current = datetime.now()
        count = 0
        offset = 0
        while count < days:
            day = current - timedelta(days=offset)
            offset += 1
            if day.weekday() >= 5:  # skip weekends
                continue
            fii_buy  = round(random.uniform(8000, 18000), 2)
            fii_net  = round(random.uniform(-3000, 3000), 2)
            fii_sell = round(fii_buy - fii_net, 2)
            dii_buy  = round(random.uniform(5000, 12000), 2)
            dii_net  = round(random.uniform(-1000, 2000), 2)
            dii_sell = round(dii_buy - dii_net, 2)
            rows.append({
                "date":         day.strftime("%Y-%m-%d"),
                "fii_buy":      fii_buy,
                "fii_sell":     fii_sell,
                "fii_net":      fii_net,
                "dii_buy":      dii_buy,
                "dii_sell":     dii_sell,
                "dii_net":      dii_net,
                "combined_net": round(fii_net + dii_net, 2),
            })
            count += 1
        return rows  # already newest-first

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, days: int = 30) -> list:
        """
        Return list of dicts, one per trading day, newest first.
        Tries NSE first; falls back to mock data on any failure.
        """
        try:
            all_rows = self._fetch_from_nse()
            return all_rows[:days]
        except Exception as exc:
            logger.warning("NSE FII/DII fetch failed (%s); using mock data.", exc)
            return self._generate_mock_data(days)

    def get_flow_signal(self) -> str:
        """
        Analyses last 5 days of FII net flows.
        BULLISH  if avg FII net > +500 Cr
        BEARISH  if avg FII net < -500 Cr
        NEUTRAL  otherwise
        """
        recent = self.fetch(days=30)[:5]
        if not recent:
            return "NEUTRAL"
        avg_fii_net = sum(r["fii_net"] for r in recent) / len(recent)
        if avg_fii_net > 500:
            return "BULLISH"
        if avg_fii_net < -500:
            return "BEARISH"
        return "NEUTRAL"

    def get_dashboard_data(self) -> dict:
        """
        Returns aggregated dashboard payload.
        Cached for CACHE_TTL_MINUTES minutes to avoid hammering NSE.
        """
        if self._is_cache_valid():
            return self._cache  # type: ignore[return-value]

        data_30 = self.fetch(30)
        recent_5 = data_30[:5]

        fii_5day_avg = (
            round(sum(r["fii_net"] for r in recent_5) / len(recent_5), 2)
            if recent_5 else 0.0
        )
        dii_5day_avg = (
            round(sum(r["dii_net"] for r in recent_5) / len(recent_5), 2)
            if recent_5 else 0.0
        )

        result = {
            "flow_signal":    self.get_flow_signal(),
            "last_30_days":   data_30,
            "fii_5day_avg":   fii_5day_avg,
            "dii_5day_avg":   dii_5day_avg,
            "last_updated":   datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        }

        self._cache = result
        self._cache_time = datetime.now()
        return result
