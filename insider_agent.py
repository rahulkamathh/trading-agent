"""
Insider Activity Agent
======================
Tracks insider buying/selling signals via NSE bulk/block deal data and
unusual volume spikes as a proxy for informed trading activity.

Data sources (in priority order):
  1. NSE Bulk Deals API  — session-cookie approach
  2. yfinance volume-spike fallback (volume > 3x 20-day average)

Cache TTL: 60 minutes.
"""

import logging
import threading
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import yfinance as yf

from engine import NIFTY50_TICKERS

logger = logging.getLogger(__name__)

_IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# NSE HTTP session constants
# ---------------------------------------------------------------------------
_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}

_NSE_HOME = "https://www.nseindia.com"
_NSE_BULK_URL = (
    "https://www.nseindia.com/api/bulk-deal-archives"
    "?number=10&series=EQ&startDate={start}&endDate={end}"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_ist() -> datetime:
    return datetime.now(_IST)


def _ticker_to_symbol(ticker: str) -> str:
    """Strip .NS suffix for NSE symbol matching."""
    return ticker.replace(".NS", "").replace(".BO", "")


def _symbol_to_ticker(symbol: str) -> str:
    return symbol.upper() + ".NS" if not symbol.upper().endswith(".NS") else symbol.upper()


def _date_str(d: date) -> str:
    return d.strftime("%d-%m-%Y")


def _make_nse_session() -> requests.Session:
    """Create a requests Session pre-loaded with NSE cookies."""
    s = requests.Session()
    s.headers.update(_NSE_HEADERS)
    try:
        s.get(_NSE_HOME, timeout=10)
    except Exception as exc:
        logger.warning("NSE cookie init failed: %s", exc)
    return s


# ---------------------------------------------------------------------------
# Main agent class
# ---------------------------------------------------------------------------

class InsiderActivityAgent:
    """
    Tracks insider / smart-money activity via:
      - NSE bulk deal archives (session-cookie approach)
      - yfinance volume spike detection (fallback when NSE is unreachable)
    """

    _CACHE_TTL_SEC = 3600  # 60 minutes

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict = {}
        self._cache_ts: datetime | None = None
        self._session: requests.Session = _make_nse_session()
        # Separate cache for raw bulk deals (reused across methods)
        self._deals_cache: list = []
        self._deals_cache_ts: datetime | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_bulk_deals(self, days: int = 30) -> list:
        """
        Return bulk deals for the last *days* calendar days.

        Each record::

            {
              "date":        "2026-06-18",
              "ticker":      "RELIANCE.NS",
              "client_name": "Some Fund",
              "deal_type":   "BUY",        # BUY | SELL
              "quantity":    500000,
              "price":       2850.50,
              "value_cr":    142.5,
              "source":      "NSE_BULK"    # or "VOLUME_SPIKE"
            }
        """
        with self._lock:
            if self._deals_cache and self._deals_cache_ts:
                age = (_now_ist() - self._deals_cache_ts).total_seconds()
                if age < self._CACHE_TTL_SEC:
                    return self._deals_cache

        end_dt = _now_ist().date()
        start_dt = end_dt - timedelta(days=days)

        deals = self._fetch_nse_bulk_deals(start_dt, end_dt)
        if not deals:
            logger.info("NSE bulk deal fetch returned 0 records — using volume spike fallback")
            deals = self._volume_spike_deals()

        with self._lock:
            self._deals_cache = deals
            self._deals_cache_ts = _now_ist()

        return deals

    def get_insider_signals(self, positions: list = None) -> list:
        """
        Return insider activity signals for the given tickers.

        Parameters
        ----------
        positions : list of ticker strings, or list of position dicts with "ticker" key.
                    If None, scans all NIFTY50_TICKERS.

        Returns
        -------
        List of signal dicts::

            {
              "ticker":         "RELIANCE.NS",
              "signal":         "ACCUMULATION",   # ACCUMULATION | DISTRIBUTION | NEUTRAL
              "strength":       75,               # 0-100
              "evidence":       ["Bulk buy 5L shares @ 2850", "Volume 4.2x avg"],
              "recommendation": "Bullish signal — insiders accumulating"
            }
        """
        if positions is None:
            tickers = list(NIFTY50_TICKERS)
        else:
            tickers = []
            for p in positions:
                if isinstance(p, dict):
                    tickers.append(p.get("ticker", ""))
                elif isinstance(p, str):
                    tickers.append(p)
            tickers = [t for t in tickers if t]

        # Normalise to .NS suffix
        tickers = [_symbol_to_ticker(_ticker_to_symbol(t)) for t in tickers]

        recent_deals = self.fetch_bulk_deals(days=5)
        volume_spikes = self._get_volume_spikes(tickers)

        signals = []
        for ticker in tickers:
            evidence: list[str] = []
            buy_weight = 0.0
            sell_weight = 0.0

            # --- bulk deal evidence (last 5 days) ---
            ticker_deals = [d for d in recent_deals if d["ticker"] == ticker]
            for deal in ticker_deals:
                qty_l = deal["quantity"] / 1e5
                price = deal["price"]
                val_cr = deal["value_cr"]
                if deal["deal_type"] == "BUY":
                    buy_weight += min(30.0, val_cr)
                    evidence.append(
                        f"Bulk BUY {qty_l:.1f}L shares @ ₹{price:,.2f}"
                        f" ({deal['date']}, ₹{val_cr:.1f} Cr)"
                    )
                else:
                    sell_weight += min(30.0, val_cr)
                    evidence.append(
                        f"Bulk SELL {qty_l:.1f}L shares @ ₹{price:,.2f}"
                        f" ({deal['date']}, ₹{val_cr:.1f} Cr)"
                    )

            # --- volume spike evidence ---
            spike = volume_spikes.get(ticker)
            if spike:
                ratio = spike["ratio"]
                if ratio >= 3.0:
                    buy_weight += 20.0
                    evidence.append(f"Volume {ratio:.1f}x 20-day avg (unusual activity)")
                elif ratio >= 2.0:
                    buy_weight += 10.0
                    evidence.append(f"Volume {ratio:.1f}x 20-day avg (elevated)")

            # --- classify ---
            net = buy_weight - sell_weight
            if net >= 15:
                signal = "ACCUMULATION"
                strength = min(100, 40 + int(net))
                recommendation = "Bullish signal — insiders accumulating"
            elif net <= -15:
                signal = "DISTRIBUTION"
                strength = min(100, 40 + int(abs(net)))
                recommendation = "Bearish signal — insiders distributing"
            else:
                signal = "NEUTRAL"
                strength = max(0, 50 - int(abs(net)))
                recommendation = "No clear insider activity detected"

            signals.append(
                {
                    "ticker": ticker,
                    "signal": signal,
                    "strength": strength,
                    "evidence": evidence,
                    "recommendation": recommendation,
                }
            )

        return signals

    def get_dashboard_data(self) -> dict:
        """
        Aggregate data for the dashboard (cached 60 min).

        Returns
        -------
        ::

            {
              "signals":         [...],
              "bulk_deals":      [...],   # last 30 days
              "unusual_volume":  [...],   # tickers with volume > 2x avg today
              "summary": {
                "accumulation_count": int,
                "distribution_count": int,
                "neutral_count":      int
              },
              "last_updated": "2026-06-18T15:30:00"
            }
        """
        with self._lock:
            if self._cache and self._cache_ts:
                age = (_now_ist() - self._cache_ts).total_seconds()
                if age < self._CACHE_TTL_SEC:
                    return self._cache

        bulk_deals = self.fetch_bulk_deals(days=30)
        signals = self.get_insider_signals()
        unusual_volume = self._unusual_volume_today()

        summary = {
            "accumulation_count": sum(1 for s in signals if s["signal"] == "ACCUMULATION"),
            "distribution_count": sum(1 for s in signals if s["signal"] == "DISTRIBUTION"),
            "neutral_count": sum(1 for s in signals if s["signal"] == "NEUTRAL"),
        }

        result = {
            "signals": signals,
            "bulk_deals": bulk_deals,
            "unusual_volume": unusual_volume,
            "summary": summary,
            "last_updated": _now_ist().strftime("%Y-%m-%dT%H:%M:%S"),
        }

        with self._lock:
            self._cache = result
            self._cache_ts = _now_ist()

        return result

    # ------------------------------------------------------------------
    # NSE fetching
    # ------------------------------------------------------------------

    def _fetch_nse_bulk_deals(self, start_dt: date, end_dt: date) -> list:
        """Hit the NSE bulk-deal-archives API. Returns empty list on failure."""
        url = _NSE_BULK_URL.format(start=_date_str(start_dt), end=_date_str(end_dt))
        for attempt in range(2):
            try:
                if attempt == 1:
                    self._session = _make_nse_session()
                resp = self._session.get(url, timeout=12)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as exc:
                logger.warning("NSE bulk deals attempt %d failed: %s", attempt + 1, exc)
        else:
            return []

        records = data if isinstance(data, list) else data.get("data", [])
        if not records:
            return []

        deals = []
        for row in records:
            try:
                symbol = str(
                    row.get("symbol") or row.get("Symbol") or row.get("SYMBOL") or ""
                ).strip()
                if not symbol:
                    continue

                qty_raw = row.get("quantity") or row.get("Quantity") or row.get("QTY_TRADED") or 0
                price_raw = row.get("price") or row.get("Price") or row.get("TRADE_PRICE") or 0
                client = str(
                    row.get("clientName") or row.get("ClientName") or row.get("CLIENT_NAME") or "Unknown"
                ).strip()
                buy_sell = str(
                    row.get("buySell") or row.get("BuySell") or row.get("BUY_SELL") or "B"
                ).strip().upper()
                raw_date = str(
                    row.get("date") or row.get("Date") or row.get("DealDate") or row.get("dealDate") or ""
                ).strip()

                qty = int(str(qty_raw).replace(",", "").split(".")[0])
                price = float(str(price_raw).replace(",", ""))
                deal_type = "BUY" if buy_sell.startswith("B") else "SELL"
                value_cr = round(qty * price / 1e7, 2)

                # Normalise date
                for fmt in ("%d-%b-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
                    try:
                        raw_date = datetime.strptime(raw_date, fmt).strftime("%Y-%m-%d")
                        break
                    except ValueError:
                        pass

                deals.append({
                    "date": raw_date,
                    "ticker": _symbol_to_ticker(symbol),
                    "client_name": client,
                    "deal_type": deal_type,
                    "quantity": qty,
                    "price": price,
                    "value_cr": value_cr,
                    "source": "NSE_BULK",
                })
            except Exception as parse_err:
                logger.debug("Skipping bulk deal row: %s", parse_err)

        logger.info("NSE bulk deals fetched: %d records", len(deals))
        return deals

    # ------------------------------------------------------------------
    # Volume spike detection
    # ------------------------------------------------------------------

    def _volume_spike_deals(self) -> list:
        """
        For each NIFTY50 ticker, flag today's volume > 3x 20-day avg.
        Returns synthetic deal records with source="VOLUME_SPIKE".
        """
        today_str = _now_ist().strftime("%Y-%m-%d")
        deals = []

        try:
            batch = yf.download(
                NIFTY50_TICKERS,
                period="25d",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as exc:
            logger.warning("yfinance batch download failed: %s", exc)
            return []

        for ticker in NIFTY50_TICKERS:
            try:
                df = (
                    batch[ticker].dropna()
                    if isinstance(batch.columns, pd.MultiIndex)
                    else batch.dropna()
                )
                if df.empty or len(df) < 5 or "Volume" not in df.columns:
                    continue

                avg_20 = float(df["Volume"].iloc[:-1].tail(20).mean())
                today_vol = float(df["Volume"].iloc[-1])
                if avg_20 <= 0 or today_vol < 3 * avg_20:
                    continue

                close = float(df["Close"].iloc[-1])
                value_cr = round(int(today_vol) * close / 1e7, 2)

                deals.append({
                    "date": today_str,
                    "ticker": ticker,
                    "client_name": "Volume Spike (proxy)",
                    "deal_type": "BUY",
                    "quantity": int(today_vol),
                    "price": round(close, 2),
                    "value_cr": value_cr,
                    "source": "VOLUME_SPIKE",
                })
            except Exception as e:
                logger.debug("Volume spike check failed %s: %s", ticker, e)

        logger.info("Volume spike fallback: %d spikes found", len(deals))
        return deals

    def _get_volume_spikes(self, tickers: list) -> dict:
        """
        Return {ticker: {"ratio": float}} for tickers with today's volume > 2x 20-day avg.
        """
        spikes: dict = {}
        if not tickers:
            return spikes
        try:
            batch = yf.download(
                tickers,
                period="25d",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            for ticker in tickers:
                try:
                    df = (
                        batch[ticker].dropna()
                        if isinstance(batch.columns, pd.MultiIndex)
                        else batch.dropna()
                    )
                    if df.empty or len(df) < 5 or "Volume" not in df.columns:
                        continue
                    avg_20 = float(df["Volume"].iloc[:-1].tail(20).mean())
                    today_vol = float(df["Volume"].iloc[-1])
                    if avg_20 > 0 and today_vol >= 2 * avg_20:
                        spikes[ticker] = {"ratio": round(today_vol / avg_20, 2)}
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("Batch volume spike fetch failed: %s", exc)
        return spikes

    def _unusual_volume_today(self) -> list:
        """Return list of tickers where today's volume > 2x 20-day average."""
        spikes = self._get_volume_spikes(list(NIFTY50_TICKERS))
        result = [
            {
                "ticker": ticker,
                "volume_ratio": info["ratio"],
                "note": f"Volume {info['ratio']:.1f}x 20-day avg",
            }
            for ticker, info in spikes.items()
        ]
        result.sort(key=lambda x: x["volume_ratio"], reverse=True)
        return result


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_INSTANCE: InsiderActivityAgent | None = None
_LOCK = threading.Lock()


def get_insider_agent() -> InsiderActivityAgent:
    """Return the global InsiderActivityAgent singleton."""
    global _INSTANCE
    if _INSTANCE is None:
        with _LOCK:
            if _INSTANCE is None:
                _INSTANCE = InsiderActivityAgent()
    return _INSTANCE
