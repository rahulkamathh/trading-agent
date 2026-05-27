"""
News & Commodity Intelligence Agent
=====================================
Layers:
  1. RSS Feed Scraping  – Economic Times, Moneycontrol, Business Standard
  2. Stock Mention Extraction + Sentiment scoring
  3. Commodity Price Tracking – copper, crude oil, gold, silver, steel
  4. Commodity → Sector mapping → BUY/SELL signals
  5. NewsSignalStrategy – surface for engine.py integration

Signals are cached for 2 hours to avoid hammering RSS endpoints.
"""

import json
import logging
import re
import time
import html
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

DATA_DIR      = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
NEWS_CACHE    = DATA_DIR / "news_cache.json"
COMMOD_CACHE  = DATA_DIR / "commodity_cache.json"
NEWS_TTL_MIN  = 30          # RSS re-fetch interval
COMMOD_TTL_MIN = 60         # Commodity price re-fetch interval

# ---------------------------------------------------------------------------
# Stock universe lookup  (company name / partial → NSE ticker)
# ---------------------------------------------------------------------------

COMPANY_TO_TICKER: dict[str, str] = {
    # Large caps
    "reliance": "RELIANCE.NS", "ril": "RELIANCE.NS",
    "tcs": "TCS.NS", "tata consultancy": "TCS.NS",
    "infosys": "INFY.NS", "infy": "INFY.NS",
    "hdfc bank": "HDFCBANK.NS", "hdfcbank": "HDFCBANK.NS",
    "icici bank": "ICICIBANK.NS", "icicibank": "ICICIBANK.NS",
    "wipro": "WIPRO.NS",
    "hcl tech": "HCLTECH.NS", "hcltech": "HCLTECH.NS",
    "bajaj finance": "BAJFINANCE.NS",
    "asian paints": "ASIANPAINT.NS",
    "maruti": "MARUTI.NS", "maruti suzuki": "MARUTI.NS",
    "sun pharma": "SUNPHARMA.NS", "sunpharma": "SUNPHARMA.NS",
    "dr reddy": "DRREDDY.NS", "dr. reddy": "DRREDDY.NS",
    "cipla": "CIPLA.NS",
    "divis": "DIVISLAB.NS", "divi": "DIVISLAB.NS",
    "titan": "TITAN.NS",
    "ltimindtree": "LTIM.NS", "lti": "LTIM.NS",
    "sbi": "SBIN.NS", "state bank": "SBIN.NS",
    "axis bank": "AXISBANK.NS", "axisbank": "AXISBANK.NS",
    "kotak": "KOTAKBANK.NS", "kotak bank": "KOTAKBANK.NS",
    "ongc": "ONGC.NS",
    "bpcl": "BPCL.NS",
    "ioc": "IOC.NS", "indian oil": "IOC.NS",
    "ntpc": "NTPC.NS",
    "power grid": "POWERGRID.NS", "powergrid": "POWERGRID.NS",
    "coal india": "COALINDIA.NS", "coalindia": "COALINDIA.NS",
    "hindalco": "HINDALCO.NS",
    "vedanta": "VEDL.NS", "vedl": "VEDL.NS",
    "tata steel": "TATASTEEL.NS", "tatasteel": "TATASTEEL.NS",
    "jsw steel": "JSWSTEEL.NS", "jswsteel": "JSWSTEEL.NS",
    "sail": "SAIL.NS", "steel authority": "SAIL.NS",
    "nmdc": "NMDC.NS",
    "l&t": "LT.NS", "larsen": "LT.NS",
    "adani ports": "ADANIPORTS.NS",
    "adani green": "ADANIGREEN.NS",
    "adani enterprises": "ADANIENT.NS",
    "itc": "ITC.NS",
    "britannia": "BRITANNIA.NS",
    "nestle": "NESTLEIND.NS",
    "tata motors": "TATAMOTORS.NS",
    "m&m": "M&M.NS", "mahindra": "M&M.NS",
    "bajaj auto": "BAJAJ-AUTO.NS",
    "hero motocorp": "HEROMOTOCO.NS", "hero": "HEROMOTOCO.NS",
    "eicher motors": "EICHERMOT.NS", "royal enfield": "EICHERMOT.NS",
    "upl": "UPL.NS",
    "grasim": "GRASIM.NS",
    "ultracemco": "ULTRACEMCO.NS", "ultratech": "ULTRACEMCO.NS",
    "bharti airtel": "BHARTIARTL.NS", "airtel": "BHARTIARTL.NS",
    "indusind": "INDUSINDBK.NS", "indusind bank": "INDUSINDBK.NS",
    # Mid caps / penny universe
    "suzlon": "SUZLON.NS",
    "nhpc": "NHPC.NS",
    "sjvn": "SJVN.NS",
    "tata power": "TATAPOWER.NS",
    "rvnl": "RVNL.NS",
    "irfc": "IRFC.NS",
    "ircon": "IRCON.NS",
    "hudco": "HUDCO.NS",
    "rec": "RECLTD.NS", "recltd": "RECLTD.NS",
    "pfc": "PFC.NS",
    "beml": "BEML.NS",
    "bel": "BEL.NS",
    "hal": "HAL.NS", "hindustan aeronautics": "HAL.NS",
    "cochin shipyard": "COCHINSHIP.NS",
    "mazagon dock": "MAZAGON.NS",
    "yes bank": "YESBANK.NS", "yesbank": "YESBANK.NS",
    "idfc first": "IDFCFIRSTB.NS",
    "federal bank": "FEDERALBNK.NS",
    "bandhan": "BANDHANBNK.NS",
    "laurus": "LAURUS.NS", "laurus labs": "LAURUS.NS",
    "granules": "GRANULES.NS",
    "syngene": "SYNGENE.NS",
    "nationalum": "NATIONALUM.NS", "national aluminium": "NATIONALUM.NS",
    "moil": "MOIL.NS",
    "hind zinc": "HINDZINC.NS", "hindustan zinc": "HINDZINC.NS",
    "gmr": "GMRINFRA.NS",
}

# ---------------------------------------------------------------------------
# Commodity → sector / stock mapping
# ---------------------------------------------------------------------------

COMMODITY_FUTURES = {
    "copper":    "HG=F",    # USD/lb
    "crude_oil": "CL=F",    # USD/bbl
    "gold":      "GC=F",    # USD/oz
    "silver":    "SI=F",    # USD/oz
    "natural_gas": "NG=F",  # USD/mmBtu
    "aluminium": "ALI=F",   # USD/ton (CME)
}

COMMODITY_SECTOR_MAP: dict[str, dict] = {
    "copper": {
        "description": "Copper price rising → benefit base metal companies",
        "tickers":     ["HINDALCO.NS", "VEDL.NS", "NATIONALUM.NS", "MOIL.NS"],
        "etf":         None,
        "threshold_pct": 2.0,    # % daily move to trigger a signal
        "strength":      0.65,
    },
    "crude_oil": {
        "description": "Crude rising → benefit upstream; hurts OMCs",
        "tickers":     ["ONGC.NS", "RELIANCE.NS"],
        "adverse":     ["BPCL.NS", "IOC.NS"],   # these get SELL signals when crude spikes
        "etf":         None,
        "threshold_pct": 2.5,
        "strength":      0.60,
    },
    "gold": {
        "description": "Gold rising → benefit gold ETF / gold financiers",
        "tickers":     ["GOLDBEES.NS"],
        "etf":         "GOLDBEES.NS",
        "threshold_pct": 1.5,
        "strength":      0.55,
    },
    "silver": {
        "description": "Silver rising → benefit silver / industrial metals",
        "tickers":     ["HINDALCO.NS", "VEDL.NS"],
        "etf":         None,
        "threshold_pct": 2.0,
        "strength":      0.50,
    },
    "natural_gas": {
        "description": "Natural gas rising → benefit gas distribution companies",
        "tickers":     ["GAIL.NS", "IGL.NS", "MGL.NS"],
        "etf":         None,
        "threshold_pct": 3.0,
        "strength":      0.55,
    },
    "aluminium": {
        "description": "Aluminium rising → benefit aluminium producers",
        "tickers":     ["NATIONALUM.NS", "HINDALCO.NS", "VEDL.NS"],
        "etf":         None,
        "threshold_pct": 2.0,
        "strength":      0.55,
    },
}

# ---------------------------------------------------------------------------
# RSS Feeds
# ---------------------------------------------------------------------------

RSS_FEEDS = [
    # Economic Times – Markets
    "https://economictimes.indiatimes.com/markets/stocks/rss.cms",
    # Economic Times – Economy
    "https://economictimes.indiatimes.com/news/economy/rss.cms",
    # Moneycontrol – Markets
    "https://www.moneycontrol.com/rss/marketsnews.xml",
    # Business Standard – Markets
    "https://www.business-standard.com/rss/markets-106.rss",
    # Livemint – Markets
    "https://www.livemint.com/rss/markets",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml,application/xml;q=0.9,*/*;q=0.8",
}

POSITIVE_WORDS = [
    "surge", "rally", "gain", "rise", "jump", "soar", "record", "high",
    "strong", "growth", "profit", "buy", "upgrade", "target", "bullish",
    "outperform", "beat", "positive", "upside", "momentum", "breakout",
    "recovery", "boom", "robust", "optimistic",
]

NEGATIVE_WORDS = [
    "fall", "drop", "decline", "crash", "loss", "down", "weak", "sell",
    "downgrade", "cut", "miss", "bearish", "underperform", "concern",
    "risk", "warning", "negative", "plunge", "slump", "worry", "caution",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_rss(url: str, timeout: int = 8) -> str:
    """Fetch raw RSS XML string."""
    try:
        req = Request(url, headers=HEADERS)
        with urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except URLError as e:
        logger.debug(f"RSS fetch failed {url}: {e}")
        return ""
    except Exception as e:
        logger.debug(f"RSS error {url}: {e}")
        return ""


def _parse_rss_items(xml: str) -> list[dict]:
    """Very simple RSS parser (no lxml dependency)."""
    items = []
    for item_match in re.finditer(r"<item>(.*?)</item>", xml, re.DOTALL):
        raw = item_match.group(1)

        def _tag(tag: str) -> str:
            m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", raw, re.DOTALL | re.I)
            if m:
                text = re.sub(r"<[^>]+>", "", m.group(1))
                return html.unescape(text).strip()
            return ""

        pub_raw = _tag("pubDate")
        try:
            from email.utils import parsedate_to_datetime
            pub_dt = parsedate_to_datetime(pub_raw)
            pub_iso = pub_dt.isoformat()
        except Exception:
            pub_iso = datetime.now().isoformat()

        items.append({
            "title":       _tag("title"),
            "description": _tag("description"),
            "link":        _tag("link"),
            "published":   pub_iso,
        })
    return items


def _sentiment_score(text: str) -> float:
    """
    Simple word-count sentiment: returns -1 (very negative) to +1 (very positive).
    """
    low = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in low)
    neg = sum(1 for w in NEGATIVE_WORDS if w in low)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def _extract_tickers(text: str) -> list[str]:
    """
    Find NSE tickers mentioned in a news headline / description.
    Match company names → NSE ticker via lookup table.
    """
    low = text.lower()
    found = set()
    for name, ticker in COMPANY_TO_TICKER.items():
        # Word-boundary-ish match
        pattern = r"(?<![a-z])" + re.escape(name) + r"(?![a-z])"
        if re.search(pattern, low):
            found.add(ticker)
    return list(found)


# ---------------------------------------------------------------------------
# News Agent
# ---------------------------------------------------------------------------

class NewsAgent:

    def __init__(self):
        self._news_cache:  list  = []
        self._news_ts:     float = 0
        self._commod_cache: dict = {}
        self._commod_ts:   float = 0
        self._load_persisted()

    # ------------------------------------------------------------ persistence

    def _load_persisted(self):
        if NEWS_CACHE.exists():
            try:
                with open(NEWS_CACHE) as f:
                    d = json.load(f)
                self._news_cache = d.get("items", [])
                self._news_ts    = d.get("ts", 0)
            except Exception:
                pass
        if COMMOD_CACHE.exists():
            try:
                with open(COMMOD_CACHE) as f:
                    d = json.load(f)
                self._commod_cache = d.get("data", {})
                self._commod_ts    = d.get("ts", 0)
            except Exception:
                pass

    def _save_news(self):
        try:
            with open(NEWS_CACHE, "w") as f:
                json.dump({"items": self._news_cache[:500], "ts": self._news_ts}, f)
        except Exception:
            pass

    def _save_commod(self):
        try:
            with open(COMMOD_CACHE, "w") as f:
                json.dump({"data": self._commod_cache, "ts": self._commod_ts}, f)
        except Exception:
            pass

    # ------------------------------------------------------------ RSS layer

    def _refresh_news(self):
        age_min = (time.time() - self._news_ts) / 60
        if age_min < NEWS_TTL_MIN:
            return

        all_items = []
        for feed_url in RSS_FEEDS:
            xml = _fetch_rss(feed_url)
            if xml:
                items = _parse_rss_items(xml)
                all_items.extend(items)
            time.sleep(0.2)

        if all_items:
            # De-duplicate by title
            seen  = set()
            dedup = []
            for it in all_items:
                key = it["title"].lower()[:80]
                if key not in seen:
                    seen.add(key)
                    dedup.append(it)
            self._news_cache = dedup
            self._news_ts    = time.time()
            self._save_news()
            logger.info(f"NewsAgent: fetched {len(dedup)} unique news items")

    def get_news_items(self) -> list[dict]:
        """Return cached news items (refreshes if stale)."""
        self._refresh_news()
        return list(self._news_cache)

    # ------------------------------------------------------------ Commodity layer

    def _refresh_commodities(self):
        age_min = (time.time() - self._commod_ts) / 60
        if age_min < COMMOD_TTL_MIN:
            return

        data = {}
        for name, ticker in COMMODITY_FUTURES.items():
            try:
                df = yf.download(ticker, period="5d", interval="1d", progress=False)
                if df.empty or len(df) < 2:
                    continue
                closes = df["Close"].dropna()
                latest = float(closes.iloc[-1])
                prev   = float(closes.iloc[-2])
                pct    = (latest - prev) / prev * 100 if prev != 0 else 0.0
                data[name] = {
                    "ticker":  ticker,
                    "price":   round(latest, 4),
                    "prev":    round(prev, 4),
                    "pct_chg": round(pct, 3),
                    "ts":      datetime.now().isoformat(),
                }
            except Exception as e:
                logger.debug(f"Commodity fetch failed {name} ({ticker}): {e}")

        if data:
            self._commod_cache = data
            self._commod_ts    = time.time()
            self._save_commod()
            logger.info(f"NewsAgent commodities refreshed: {list(data.keys())}")

    def get_commodity_data(self) -> dict:
        """Return {name: {price, pct_chg, …}} for all tracked commodities."""
        self._refresh_commodities()
        return dict(self._commod_cache)

    # ------------------------------------------------------------ Signal generation

    def get_news_signals(self) -> list[dict]:
        """
        Generate signals from news sentiment.
        Each signal:  {ticker, action, strength, strategy, reason}
        """
        self._refresh_news()
        signals = []
        cutoff  = datetime.now() - timedelta(hours=6)

        # Aggregate sentiment per ticker from recent news
        ticker_sentiment: dict[str, list[float]] = {}
        ticker_items:     dict[str, list[str]]   = {}

        for item in self._news_cache:
            # Only consider recent news
            try:
                pub_dt = datetime.fromisoformat(item["published"][:19])
            except Exception:
                continue
            if pub_dt < cutoff:
                continue

            text     = item["title"] + " " + item.get("description", "")
            tickers  = _extract_tickers(text)
            sentiment = _sentiment_score(text)

            for t in tickers:
                ticker_sentiment.setdefault(t, []).append(sentiment)
                ticker_items.setdefault(t, []).append(item["title"][:80])

        for ticker, scores in ticker_sentiment.items():
            avg_sentiment = float(np.mean(scores))
            mention_count = len(scores)
            # Only signal if mentioned ≥2 times or sentiment is strong
            if mention_count < 2 and abs(avg_sentiment) < 0.5:
                continue

            strength = _clamp(abs(avg_sentiment) * 0.7 + min(mention_count, 5) * 0.05)
            action   = "BUY" if avg_sentiment > 0 else "SELL"
            reason   = "; ".join(ticker_items[ticker][:2])

            signals.append({
                "ticker":   ticker,
                "action":   action,
                "strength": round(strength, 3),
                "strategy": "News Sentiment",
                "reason":   f"[{mention_count} articles] {reason}",
                "time":     datetime.now().isoformat(),
                "is_penny": False,
            })

        logger.info(f"NewsAgent: {len(signals)} news-based signals")
        return signals

    def get_commodity_signals(self) -> list[dict]:
        """
        Generate BUY/SELL signals from commodity price moves.
        """
        self._refresh_commodities()
        signals = []

        for commod_name, mapping in COMMODITY_SECTOR_MAP.items():
            data = self._commod_cache.get(commod_name)
            if not data:
                continue

            pct_chg    = data.get("pct_chg", 0)
            threshold  = mapping["threshold_pct"]
            base_str   = mapping["strength"]
            description = mapping["description"]

            # Scale strength by how much the commodity moved
            scale    = min(abs(pct_chg) / threshold, 2.5)
            strength = round(_clamp(base_str * scale), 3)

            if abs(pct_chg) < threshold * 0.5:
                continue  # move too small

            move_dir = "up" if pct_chg > 0 else "down"
            reason   = (
                f"{commod_name.replace('_', ' ').title()} {move_dir} "
                f"{abs(pct_chg):.1f}% — {description}"
            )

            # Primary beneficiary tickers
            for ticker in mapping.get("tickers", []):
                if ticker == mapping.get("etf"):
                    continue  # ETFs handled separately
                action = "BUY" if pct_chg > 0 else "SELL"
                signals.append({
                    "ticker":   ticker,
                    "action":   action,
                    "strength": strength,
                    "strategy": "Commodity",
                    "reason":   reason,
                    "time":     datetime.now().isoformat(),
                    "is_penny": False,
                })

            # Adverse effect tickers (e.g. OMCs when crude spikes)
            for ticker in mapping.get("adverse", []):
                action = "SELL" if pct_chg > 0 else "BUY"
                signals.append({
                    "ticker":   ticker,
                    "action":   action,
                    "strength": round(strength * 0.8, 3),
                    "strategy": "Commodity",
                    "reason":   f"[Adverse] {reason}",
                    "time":     datetime.now().isoformat(),
                    "is_penny": False,
                })

            # ETF signal if applicable
            if mapping.get("etf"):
                action = "BUY" if pct_chg > 0 else "SELL"
                signals.append({
                    "ticker":   mapping["etf"],
                    "action":   action,
                    "strength": strength,
                    "strategy": "Commodity",
                    "reason":   reason,
                    "time":     datetime.now().isoformat(),
                    "is_penny": False,
                })

        logger.info(f"NewsAgent: {len(signals)} commodity-based signals")
        return signals

    def get_all_signals(self) -> list[dict]:
        """Combined list of news + commodity signals."""
        news  = self.get_news_signals()
        commd = self.get_commodity_signals()
        return news + commd

    # ------------------------------------------------------------ Public API helpers

    def latest_headlines(self, n: int = 20) -> list[dict]:
        """Return latest N news items with sentiment pre-computed."""
        self._refresh_news()
        result = []
        for item in self._news_cache[:n]:
            text = item["title"] + " " + item.get("description", "")
            result.append({
                "title":     item["title"],
                "link":      item.get("link", ""),
                "published": item.get("published", ""),
                "sentiment": round(_sentiment_score(text), 3),
                "tickers":   _extract_tickers(text),
            })
        return result


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_news_agent_instance: NewsAgent | None = None


def get_news_agent() -> NewsAgent:
    global _news_agent_instance
    if _news_agent_instance is None:
        _news_agent_instance = NewsAgent()
    return _news_agent_instance


# ---------------------------------------------------------------------------
# Quick clamp helper (module-level, used in get_commodity_signals)
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))
