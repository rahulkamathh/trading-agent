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
    # ── Google News RSS (no auth, reliable from any server) ───────────────
    # Indian stock market general
    "https://news.google.com/rss/search?q=india+stock+market+NSE+BSE&hl=en-IN&gl=IN&ceid=IN:en",
    # Nifty specific
    "https://news.google.com/rss/search?q=nifty+sensex+today&hl=en-IN&gl=IN&ceid=IN:en",
    # FII/DII and macro
    "https://news.google.com/rss/search?q=FII+DII+india+market+inflow+outflow&hl=en-IN&gl=IN&ceid=IN:en",
    # RBI and economy
    "https://news.google.com/rss/search?q=RBI+india+economy+inflation+rate&hl=en-IN&gl=IN&ceid=IN:en",
    # Earnings and corporate results
    "https://news.google.com/rss/search?q=india+quarterly+results+earnings+NSE&hl=en-IN&gl=IN&ceid=IN:en",
    # Sector specific: banking, IT, pharma
    "https://news.google.com/rss/search?q=india+banking+IT+pharma+sector+stocks&hl=en-IN&gl=IN&ceid=IN:en",
    # Global macro that impacts India
    "https://news.google.com/rss/search?q=US+fed+dollar+crude+oil+india+market&hl=en-IN&gl=IN&ceid=IN:en",
    # NSE official announcements via Google News
    "https://news.google.com/rss/search?q=NSE+BSE+SEBI+announcement+circular&hl=en-IN&gl=IN&ceid=IN:en",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
    ),
    "Accept": "application/rss+xml,application/xml,text/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
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
            # De-duplicate by title; score sentiment + extract tickers for each item
            seen  = set()
            dedup = []
            for it in all_items:
                key = it["title"].lower()[:80]
                if key not in seen:
                    seen.add(key)
                    combined = it["title"] + " " + it.get("description", "")
                    it["sentiment"] = round(_sentiment_score(combined), 3)
                    it["tickers"]   = _extract_tickers(combined)
                    it.setdefault("source", "RSS")
                    dedup.append(it)
            self._news_cache = dedup
            self._news_ts    = time.time()
            self._save_news()
            logger.info(f"NewsAgent: fetched {len(dedup)} unique items, scored sentiment")

    def get_news_items(self) -> list[dict]:
        """Return cached news items (refreshes if stale)."""
        self._refresh_news()
        # Merge yfinance ticker news for held positions
        try:
            yf_items = self._fetch_yfinance_news()
            # Prepend yfinance news (most relevant to holdings) before RSS items
            combined = yf_items + list(self._news_cache)
            # Deduplicate by title
            seen, out = set(), []
            for it in combined:
                key = (it.get("title") or "")[:80].lower()
                if key and key not in seen:
                    seen.add(key)
                    out.append(it)
            return out
        except Exception:
            return list(self._news_cache)

    def _fetch_yfinance_news(self) -> list[dict]:
        """Fetch news from yfinance for Nifty + key tickers. Free, always works."""
        items = []
        # Always fetch index + top blue chips
        watchlist = ["^NSEI", "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS",
                     "SBIN.NS", "ICICIBANK.NS", "BAJFINANCE.NS", "MARUTI.NS"]
        # Also add any tickers we currently hold
        try:
            from engine import get_agent  # pylint: disable=import-outside-toplevel
            agent = get_agent()
            watchlist += list(agent.portfolio.state.get("positions", {}).keys())
        except Exception:
            pass

        seen_titles: set = set()
        for ticker in watchlist[:15]:   # cap to avoid slow startup
            try:
                news = yf.Ticker(ticker).news or []
                for n in news[:5]:
                    title = (n.get("title") or "").strip()
                    if not title or title.lower() in seen_titles:
                        continue
                    seen_titles.add(title.lower())
                    # Sentiment
                    desc = n.get("summary") or n.get("description") or ""
                    sentiment = _sentiment_score(title + " " + desc)
                    tickers_mentioned = _extract_tickers(title + " " + desc)
                    items.append({
                        "title":       title,
                        "description": desc[:300],
                        "link":        n.get("link") or n.get("url") or "",
                        "published":   datetime.fromtimestamp(
                            n.get("providerPublishTime", time.time())
                        ).isoformat(),
                        "source":      n.get("publisher") or "yfinance",
                        "sentiment":   round(sentiment, 3),
                        "tickers":     tickers_mentioned,
                    })
            except Exception:
                pass
        return items

    # ------------------------------------------------------------ Commodity layer

    def _refresh_commodities(self):
        age_min = (time.time() - self._commod_ts) / 60
        if age_min < COMMOD_TTL_MIN:
            return

        # Friendly display names for the briefing
        DISPLAY_NAMES = {
            "copper":      "Copper",
            "crude_oil":   "Crude Oil WTI",
            "gold":        "Gold",
            "silver":      "Silver",
            "natural_gas": "Natural Gas",
            "aluminium":   "Aluminium",
        }

        data = {}
        for key, ticker in COMMODITY_FUTURES.items():
            name = DISPLAY_NAMES.get(key, key)
            try:
                # Try 1h interval first (more current pre-market)
                df = yf.download(ticker, period="2d", interval="1h",
                                 auto_adjust=True, progress=False)
                if isinstance(df.columns, __import__("pandas").MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if df.empty or len(df) < 2:
                    raise ValueError("empty 1h data")
                closes = df["Close"].dropna()
                latest = float(closes.iloc[-1])
                prev   = float(closes.iloc[-2])
            except Exception:
                try:
                    # Fallback: daily data
                    df = yf.download(ticker, period="5d", interval="1d",
                                     auto_adjust=True, progress=False)
                    if isinstance(df.columns, __import__("pandas").MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    if df.empty or len(df) < 2:
                        continue
                    closes = df["Close"].dropna()
                    latest = float(closes.iloc[-1])
                    prev   = float(closes.iloc[-2])
                except Exception as e:
                    logger.debug(f"Commodity fetch failed {name} ({ticker}): {e}")
                    continue

            pct = (latest - prev) / prev * 100 if prev != 0 else 0.0
            data[name] = {
                "ticker":  ticker,
                "price":   round(latest, 4),
                "prev":    round(prev, 4),
                "pct_chg": round(pct, 3),
                "ts":      datetime.now().isoformat(),
            }

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

    # ------------------------------------------------------------ Spec-compatible API

    def fetch_news(self, max_articles: int = 50) -> list:
        """
        Return a normalised list of article dicts (spec-compatible).
        Each dict: title, source, url, published, sentiment_score, sentiment, tickers_mentioned.
        Results are cached; underlying cache is shared with get_news_items().
        """
        raw = self.get_news_items()
        out = []
        for item in raw[:max_articles]:
            text = item.get("title", "") + " " + item.get("description", "")
            score = round(_sentiment_score(text), 3)
            label = "BULLISH" if score > 0.1 else ("BEARISH" if score < -0.1 else "NEUTRAL")
            out.append({
                "title":             item.get("title", ""),
                "source":            item.get("source", "RSS"),
                "url":               item.get("link", ""),
                "published":         item.get("published", ""),
                "sentiment_score":   score,
                "sentiment":         label,
                "tickers_mentioned": item.get("tickers", []),
            })
        return out

    def get_ticker_sentiment(self, ticker: str, news: list = None) -> dict:
        """
        Filter news mentioning the given ticker and return a sentiment summary.
        Matches by ticker base name (without .NS suffix), case-insensitive.
        """
        if news is None:
            news = self.fetch_news()

        base = ticker.replace(".NS", "").upper()
        relevant = []
        for a in news:
            title_lower = a.get("title", "").lower()
            mentioned_bases = [t.replace(".NS", "") for t in a.get("tickers_mentioned", [])]
            if base.lower() in title_lower or base in mentioned_bases:
                relevant.append(a)

        full_ticker = ticker if ".NS" in ticker else f"{ticker}.NS"
        if not relevant:
            return {
                "ticker": full_ticker,
                "sentiment": "NEUTRAL",
                "score": 0.0,
                "article_count": 0,
                "recent_headlines": [],
            }

        avg_score = round(sum(a["sentiment_score"] for a in relevant) / len(relevant), 3)
        label = "BULLISH" if avg_score > 0.1 else ("BEARISH" if avg_score < -0.1 else "NEUTRAL")
        headlines = [a["title"] for a in relevant[:5]]

        return {
            "ticker": full_ticker,
            "sentiment": label,
            "score": avg_score,
            "article_count": len(relevant),
            "recent_headlines": headlines,
        }

    def get_market_sentiment(self, news: list = None) -> dict:
        """
        Compute overall market sentiment from all fetched articles.
        """
        if news is None:
            news = self.fetch_news()

        if not news:
            return {
                "overall": "NEUTRAL",
                "score": 0.0,
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0,
                "top_bullish": [],
                "top_bearish": [],
                "last_fetched": datetime.now().isoformat(timespec="seconds"),
            }

        bullish = [a for a in news if a["sentiment"] == "BULLISH"]
        bearish = [a for a in news if a["sentiment"] == "BEARISH"]
        neutral = [a for a in news if a["sentiment"] == "NEUTRAL"]

        avg_score = round(sum(a["sentiment_score"] for a in news) / len(news), 3)
        label = "BULLISH" if avg_score > 0.1 else ("BEARISH" if avg_score < -0.1 else "NEUTRAL")

        top_bullish = sorted(bullish, key=lambda x: x["sentiment_score"], reverse=True)[:3]
        top_bearish = sorted(bearish, key=lambda x: x["sentiment_score"])[:3]

        return {
            "overall": label,
            "score": avg_score,
            "bullish_count": len(bullish),
            "bearish_count": len(bearish),
            "neutral_count": len(neutral),
            "top_bullish": [a["title"] for a in top_bullish],
            "top_bearish": [a["title"] for a in top_bearish],
            "last_fetched": datetime.now().isoformat(timespec="seconds"),
        }

    def get_dashboard_data(self) -> dict:
        """
        Return data bundle for the dashboard:
          - market_sentiment
          - recent_news (last 20 articles)
          - last_fetched timestamp
        """
        news = self.fetch_news()
        market_sentiment = self.get_market_sentiment(news)
        recent_news = news[:20]
        last_fetched = datetime.now().isoformat(timespec="seconds")

        return {
            "market_sentiment": market_sentiment,
            "recent_news": recent_news,
            "last_fetched": last_fetched,
        }


# ---------------------------------------------------------------------------
# Pre-market Briefing  (sent to Telegram at 8:45 AM IST before market open)
# ---------------------------------------------------------------------------

def send_premarket_briefing() -> None:
    """
    Runs in a daemon thread. Fetches fresh news + event calendar + commodities
    and sends a structured pre-market briefing to TELEGRAM_NOTIFY_CHAT_ID.

    Covers:
      • Upcoming high-risk events today / this week
      • Top market headlines (last 12 hours) with sentiment
      • Key commodity moves (crude, gold, SGX Nifty)
      • FII/DII sentiment from headlines
      • Agent recommendation: cautious / normal / aggressive
    """
    import threading as _t
    _t.Thread(target=_do_premarket_briefing, daemon=True, name="premarket-briefing").start()


def _do_premarket_briefing() -> None:
    import os as _os, requests as _req
    logger.info("[PreMarket] Building pre-market briefing…")

    bot_token = _os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id   = _os.environ.get("TELEGRAM_NOTIFY_CHAT_ID", "").strip()
    if not bot_token or not chat_id:
        logger.warning("[PreMarket] Bot credentials not set — skipping briefing")
        return

    try:
        agent = get_news_agent()
        # Force-refresh news right now
        agent._news_ts = 0
        agent._refresh_news()
        news  = agent.get_news_items()
        comms = agent.get_commodity_data()

        from datetime import date as _date
        today_str = _date.today().strftime("%d %b %Y, %a")

        # ── Section 1: Event calendar ──────────────────────────────────────
        try:
            from risk_manager import get_risk_manager
            rm = get_risk_manager()
            events_today = rm.upcoming_events(days=2)
            macro_score, macro_label, _ = rm.macro_risk(force_refresh=True)
        except Exception:
            events_today, macro_score, macro_label = [], 0, "LOW"

        event_lines = ""
        if events_today:
            event_lines = "\n".join(
                f"  ⚠️ <b>{e['label']}</b> — {e['days_away'] == 0 and 'TODAY' or 'tomorrow'}"
                for e in events_today
            )
        else:
            event_lines = "  ✅ No major market events today"

        risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "EXTREME": "🚨"}.get(macro_label, "🟢")

        # ── Section 2: Top headlines (last 24 hours) ───────────────────────
        from datetime import datetime as _dt
        cutoff = _dt.now().timestamp() - 24 * 3600
        recent_news = []
        for item in news[:60]:
            try:
                pub = _dt.fromisoformat(item.get("published", "")[:19]).timestamp()
            except Exception:
                pub = cutoff + 1
            if pub >= cutoff:
                recent_news.append(item)

        # Sort by absolute sentiment (most impactful first)
        recent_news.sort(key=lambda x: abs(x.get("sentiment", 0)), reverse=True)

        bullish_headlines = [n for n in recent_news if n.get("sentiment", 0) > 0.1][:4]
        bearish_headlines = [n for n in recent_news if n.get("sentiment", 0) < -0.1][:4]
        neutral_headlines = [n for n in recent_news if abs(n.get("sentiment", 0)) <= 0.1][:2]

        def _fmt_headline(n: dict) -> str:
            title = n.get("title", "")[:90]
            src   = n.get("source") or n.get("publisher") or ""
            link  = n.get("link", "")
            if link:
                return f'  • <a href="{link}">{title}</a>'
            return f"  • {title}"

        bull_section = "\n".join(_fmt_headline(n) for n in bullish_headlines) or "  • No clearly bullish headlines"
        bear_section = "\n".join(_fmt_headline(n) for n in bearish_headlines) or "  • No clearly bearish headlines"

        # FII sentiment detection from headlines
        fii_text = " ".join(n.get("title","") for n in recent_news).lower()
        if "fii buying" in fii_text or "foreign inflow" in fii_text or "fii net buy" in fii_text:
            fii_line = "📥 FII: Buying (bullish signal)"
        elif "fii selling" in fii_text or "foreign outflow" in fii_text or "fii net sell" in fii_text:
            fii_line = "📤 FII: Selling (bearish signal)"
        else:
            fii_line = "📊 FII: No strong signal in headlines"

        # ── Section 3: Commodities ─────────────────────────────────────────
        comm_lines = []
        priority = ["Crude Oil WTI", "Gold", "Silver", "SGX Nifty", "Copper"]
        for name in priority:
            c = comms.get(name)
            if not c:
                continue
            pct = c.get("pct_chg", 0)
            arrow = "▲" if pct > 0 else "▼" if pct < 0 else "—"
            color_note = " ⚠️" if abs(pct) > 1.5 else ""
            comm_lines.append(f"  {arrow} {name}: {pct:+.2f}%{color_note}")

        comm_section = "\n".join(comm_lines) or "  • Commodity data unavailable"

        # ── Section 4: Agent recommendation ────────────────────────────────
        if macro_label == "EXTREME":
            rec = "🚨 <b>EXTREME CAUTION</b> — high-risk event. Agent will limit new trades."
        elif macro_label == "HIGH":
            rec = "🔴 <b>CAUTIOUS DAY</b> — risk elevated. Position sizes reduced."
        elif len(bearish_headlines) > len(bullish_headlines) + 1:
            rec = "🟡 <b>DEFENSIVE</b> — more bearish than bullish headlines. Watch carefully."
        elif len(bullish_headlines) > len(bearish_headlines) + 1:
            rec = "🟢 <b>CONSTRUCTIVE</b> — sentiment leaning positive. Normal position sizing."
        else:
            rec = "⚪ <b>NEUTRAL</b> — mixed signals. Standard risk management applies."

        # ── Assemble message ───────────────────────────────────────────────
        msg = (
            f"🌅 <b>Pre-Market Briefing — {today_str}</b>\n"
            f"Market opens in ~30 minutes\n"
            f"─────────────────────────\n\n"
            f"📅 <b>Events &amp; Risk</b>  {risk_emoji} {macro_label} ({macro_score*100:.0f}/100)\n"
            f"{event_lines}\n\n"
            f"📰 <b>Bullish Headlines</b>\n{bull_section}\n\n"
            f"📰 <b>Bearish Headlines</b>\n{bear_section}\n\n"
            f"🛢 <b>Commodities</b>\n{comm_section}\n\n"
            f"{fii_line}\n\n"
            f"🤖 <b>Agent Today:</b> {rec}"
        )

        _req.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": msg,
                  "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=15,
        )
        logger.info("[PreMarket] Briefing sent successfully")

    except Exception as exc:
        logger.error(f"[PreMarket] Briefing failed: {exc}", exc_info=True)


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
