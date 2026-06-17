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
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests
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
# Screener.in Scraper
# ---------------------------------------------------------------------------

SCREENER_CACHE_FILE = DATA_DIR / "screener_cache.json"
SCREENER_TTL_HOURS  = 24

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _parse_number(text: str) -> float | None:
    """Parse screener.in numbers like '1,23,456.78' or '12.3%' → float."""
    if not text:
        return None
    text = text.strip().replace(",", "").replace("%", "").replace("₹", "").replace("Cr.", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


class ScreenerScraper:
    """Scrapes screener.in for India-specific fundamental metrics."""

    def __init__(self):
        self._cache: dict = {}
        self._load_cache()

    def _load_cache(self):
        if SCREENER_CACHE_FILE.exists():
            try:
                with open(SCREENER_CACHE_FILE) as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

    def _save_cache(self):
        try:
            with open(SCREENER_CACHE_FILE, "w") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save screener cache: {e}")

    def _is_stale(self, symbol: str) -> bool:
        entry = self._cache.get(symbol)
        if not entry:
            return True
        try:
            age = datetime.now() - datetime.fromisoformat(entry.get("fetched_at", ""))
            return age > timedelta(hours=SCREENER_TTL_HOURS)
        except Exception:
            return True

    def _nse_to_screener(self, ticker: str) -> str:
        """Convert 'RELIANCE.NS' → 'RELIANCE'."""
        return ticker.replace(".NS", "").replace(".BO", "").upper()

    def scrape(self, ticker: str) -> dict:
        """Return screener.in data for ticker (cached 24h)."""
        symbol = self._nse_to_screener(ticker)
        if not self._is_stale(symbol):
            return self._cache.get(symbol, {}).get("data", {})
        data = self._fetch(symbol)
        self._cache[symbol] = {"data": data, "fetched_at": datetime.now().isoformat()}
        self._save_cache()
        return data

    def _fetch(self, symbol: str) -> dict:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning("beautifulsoup4 not installed — screener.in scraping disabled")
            return {}

        # Try consolidated first, fall back to standalone
        for url in [
            f"https://www.screener.in/company/{symbol}/consolidated/",
            f"https://www.screener.in/company/{symbol}/",
        ]:
            try:
                resp = requests.get(url, headers=_HEADERS, timeout=15)
                if resp.status_code == 200:
                    break
            except Exception as e:
                logger.debug(f"Screener.in fetch failed for {symbol}: {e}")
                return {}
        else:
            logger.debug(f"Screener.in {symbol}: not found")
            return {}

        soup = BeautifulSoup(resp.text, "html.parser")
        result: dict = {"symbol": symbol, "_url": url}

        # ── Company name & about ────────────────────────────────────────────
        name_tag = soup.find("h1")
        if name_tag:
            result["company_name"] = name_tag.get_text(strip=True)

        about_sec = soup.find(id="company-info") or soup.find(class_="company-profile")
        if not about_sec:
            about_sec = soup.find("div", class_=re.compile(r"about", re.I))
        if about_sec:
            about_p = about_sec.find("p")
            if about_p:
                result["about"] = about_p.get_text(strip=True)

        # ── Top ratios (header KPIs) ─────────────────────────────────────────
        ratios_section = soup.find(id="top-ratios")
        if ratios_section:
            for li in ratios_section.find_all("li"):
                name_tag  = li.find("span", class_="name")
                value_tag = li.find("span", class_="number")
                if not name_tag or not value_tag:
                    continue
                name  = name_tag.get_text(strip=True).lower()
                raw   = value_tag.get_text(strip=True)
                value = _parse_number(raw)
                result.setdefault("_raw_ratios", {})[name] = raw
                if "market cap"  in name: result["market_cap_cr"]      = value
                elif "current price" in name or "curr. price" in name:  result["price"]     = value
                elif "high / low"in name or "52 week"  in name:
                    # "1,608 / 1,114" format
                    parts = raw.replace(",", "").split("/")
                    if len(parts) == 2:
                        result["high_52w"] = _parse_number(parts[0])
                        result["low_52w"]  = _parse_number(parts[1])
                elif "book value"  in name: result["book_value"]        = value
                elif "dividend yld"in name or "div yld" in name: result["dividend_yield"] = value
                elif "roce"        in name: result["roce"]              = value
                elif "roe"         in name: result["roe"]               = value
                elif "face value"  in name: result["face_value"]        = value
                elif "p/e"         in name: result["pe"]                = value
                elif "p/b"         in name: result["pb"]                = value
                elif "eps"         in name: result["eps"]               = value
                elif "debt"        in name and "equity" not in name: result["debt_cr"] = value
                elif "sales gr"    in name: result["sales_growth_3yr"]  = value
                elif "profit gr"   in name: result["profit_growth_3yr"] = value
                elif "ind pe"      in name or "industry pe" in name: result["industry_pe"] = value

        # ── Helper: parse any screener.in financial table ────────────────────
        def _parse_table(section_id: str) -> dict:
            sec = soup.find(id=section_id)
            if not sec:
                return {}
            table = sec.find("table")
            if not table:
                return {}
            rows = table.find_all("tr")
            if not rows:
                return {}
            # Header row → period labels
            header_row = rows[0]
            th_cells = header_row.find_all(["th", "td"])
            periods = [c.get_text(strip=True) for c in th_cells]  # first cell = row label

            out: dict = {"periods": periods[1:], "rows": {}}
            for row in rows[1:]:
                cells = row.find_all(["td", "th"])
                if not cells:
                    continue
                label = cells[0].get_text(strip=True)
                if not label:
                    continue
                vals = []
                for c in cells[1:]:
                    txt = c.get_text(strip=True)
                    vals.append(txt)
                out["rows"][label] = vals
            return out

        # ── Quarterly results ────────────────────────────────────────────────
        result["quarterly"] = _parse_table("quarters")

        # Latest quarter single values (backwards compat)
        q = result["quarterly"]
        if q.get("rows"):
            def _latest(label_kws):
                for lbl, vals in q["rows"].items():
                    if any(k in lbl.lower() for k in label_kws):
                        nums = [_parse_number(v) for v in vals]
                        nums = [n for n in nums if n is not None]
                        return nums[-1] if nums else None
                return None
            result["q_sales_cr"]  = _latest(["sales", "revenue"])
            result["q_profit_cr"] = _latest(["net profit", "profit after"])
            result["q_opm_pct"]   = _latest(["opm"])

        # ── Annual Profit & Loss ─────────────────────────────────────────────
        result["annual_pl"] = _parse_table("profit-loss")

        # ── Balance Sheet ────────────────────────────────────────────────────
        result["balance_sheet"] = _parse_table("balance-sheet")

        # ── Cash Flows ───────────────────────────────────────────────────────
        result["cash_flows"] = _parse_table("cash-flow")

        # ── Ratios (efficiency) ───────────────────────────────────────────────
        result["ratios_table"] = _parse_table("ratios")

        # ── Shareholding pattern (multi-quarter trend) ───────────────────────
        holding_section = soup.find(id="shareholding")
        if holding_section:
            table = holding_section.find("table")
            if table:
                rows = table.find_all("tr")
                periods = []
                if rows:
                    hcells = rows[0].find_all(["th", "td"])
                    periods = [c.get_text(strip=True) for c in hcells[1:]]
                holding: dict = {"periods": periods, "rows": {}}
                for row in rows[1:]:
                    cells = row.find_all(["td", "th"])
                    if not cells:
                        continue
                    label = cells[0].get_text(strip=True)
                    vals  = [_parse_number(c.get_text(strip=True)) for c in cells[1:]]
                    holding["rows"][label] = vals
                result["shareholding"] = holding

                # Latest values for quick display
                def _hld(kws):
                    for lbl, vals in holding["rows"].items():
                        if any(k in lbl.lower() for k in kws):
                            v = [x for x in vals if x is not None]
                            return v[-1] if v else None
                    return None
                result["promoter_holding"] = _hld(["promoter"])
                result["fii_holding"]      = _hld(["fii", "foreign"])
                result["dii_holding"]      = _hld(["dii", "domestic inst"])
                result["public_holding"]   = _hld(["public"])

        # ── Peers ────────────────────────────────────────────────────────────
        peers_sec = soup.find(id="peers") or soup.find(id="peer-comparison")
        if peers_sec:
            table = peers_sec.find("table")
            if table:
                rows = table.find_all("tr")
                if rows:
                    headers = [c.get_text(strip=True) for c in rows[0].find_all(["th","td"])]
                    peers = []
                    for row in rows[1:]:
                        cells = row.find_all(["td","th"])
                        if not cells:
                            continue
                        peer = {}
                        for i, c in enumerate(cells):
                            key = headers[i] if i < len(headers) else str(i)
                            peer[key] = c.get_text(strip=True)
                        if peer:
                            peers.append(peer)
                    result["peers"] = peers

        # ── Pros / Cons ───────────────────────────────────────────────────────
        pros, cons = [], []
        for tag in soup.find_all("li", class_=re.compile(r"pro|positive", re.I)):
            pros.append(tag.get_text(strip=True))
        for tag in soup.find_all("li", class_=re.compile(r"con|negative|warn", re.I)):
            cons.append(tag.get_text(strip=True))
        # Fallback: find ul with class pros/cons
        for ul in soup.find_all("ul"):
            c = ul.get("class", [])
            cls_str = " ".join(c).lower()
            if "pros" in cls_str or "positive" in cls_str:
                pros += [li.get_text(strip=True) for li in ul.find_all("li")]
            elif "cons" in cls_str or "negative" in cls_str:
                cons += [li.get_text(strip=True) for li in ul.find_all("li")]
        result["pros"] = list(dict.fromkeys(pros))[:8]   # dedupe, max 8
        result["cons"] = list(dict.fromkeys(cons))[:8]

        # ── Piotroski score ───────────────────────────────────────────────────
        piotro_tag = soup.find(string=re.compile(r"Piotroski", re.I))
        if piotro_tag:
            parent = piotro_tag.find_parent()
            if parent:
                nums = re.findall(r"\b([0-9])\b", parent.get_text())
                if nums:
                    result["piotroski"] = int(nums[0])

        # ── Augment with yfinance (52W H/L, volume, EPS TTM) ─────────────────
        try:
            import yfinance as yf
            ticker_sym = f"{symbol}.NS"
            tk = yf.Ticker(ticker_sym)
            fi = tk.fast_info
            if not result.get("price"):
                result["price"] = round(float(fi.last_price or 0), 2)
            if not result.get("high_52w"):
                result["high_52w"] = round(float(fi.year_high or 0), 2)
            if not result.get("low_52w"):
                result["low_52w"]  = round(float(fi.year_low or 0), 2)
            result["volume"]    = int(fi.three_month_average_volume or 0)
            result["prev_close"] = round(float(fi.previous_close or 0), 2)
            if result["price"] and result["prev_close"]:
                result["price_chg"] = round(result["price"] - result["prev_close"], 2)
                result["price_chg_pct"] = round((result["price"] / result["prev_close"] - 1) * 100, 2)
        except Exception:
            pass

        logger.info(f"Screener.in scraped {symbol}: {len(result)} fields")
        return result

    def get_all_cached(self) -> dict:
        return {
            sym: entry.get("data", {})
            for sym, entry in self._cache.items()
        }


_screener_instance: ScreenerScraper | None = None


def get_screener() -> ScreenerScraper:
    global _screener_instance
    if _screener_instance is None:
        _screener_instance = ScreenerScraper()
    return _screener_instance


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

    def _compute_score(self, m: dict, sc: dict | None = None) -> float:
        """
        Composite quality score 0–1.
        Components: P/E, P/B, ROE, D/E, revenue growth, profit margin,
                    ROCE (screener.in), promoter holding (screener.in).
        """
        sc = sc or {}
        components = []

        # P/E score: ideal < 15, neutral at 30, bad > 60
        pe = m.get("pe")
        if pe and pe > 0:
            components.append(_clamp(1 - (pe - 5) / 55))
        else:
            components.append(0.5)

        # P/B score: ideal < 1.5, bad > 5
        pb = m.get("pb")
        if pb and pb > 0:
            components.append(_clamp(1 - (pb - 0.5) / 5.5))
        else:
            components.append(0.5)

        # ROE score: ideal > 20%, bad < 5%
        roe = m.get("roe")
        if roe is not None:
            components.append(_clamp((roe - 5) / 25))
        else:
            components.append(0.5)

        # D/E score: ideal < 0.3, bad > 2
        de = m.get("de")
        if de is not None and de >= 0:
            components.append(_clamp(1 - de / 2.0))
        else:
            components.append(0.5)

        # Revenue growth score: ideal > 20%, bad < 0%
        rg = m.get("rev_growth")
        if rg is not None:
            components.append(_clamp((rg + 5) / 30))
        else:
            components.append(0.5)

        # Profit margin score: ideal > 20%, bad < 0%
        mg = m.get("margin")
        if mg is not None:
            components.append(_clamp((mg + 2) / 25))
        else:
            components.append(0.5)

        # ROCE score (screener.in): ideal > 20%, bad < 8%
        roce = sc.get("roce")
        if roce is not None:
            components.append(_clamp((roce - 8) / 22))
        else:
            components.append(0.5)

        # Promoter holding (screener.in): ideal > 60%, bad < 25%
        promoter = sc.get("promoter_holding")
        if promoter is not None:
            components.append(_clamp((promoter - 25) / 50))
        else:
            components.append(0.5)

        # Piotroski F-score (screener.in): 0–9, ideal ≥ 7
        piotro = sc.get("piotroski")
        if piotro is not None:
            components.append(_clamp(piotro / 9))
        # (no neutral fallback — only include if available)

        return round(float(np.mean(components)), 3)

    # ------------------------------------------------------------------ public

    def get_score(self, ticker: str) -> dict:
        """
        Return {score, metrics, screener, grade} for *ticker*.
        Uses cache; fetches fresh data if stale.
        Merges yfinance + screener.in data.
        """
        if self._is_stale(ticker):
            m = self._fetch_ticker_data(ticker)

            # Enrich with screener.in data (non-blocking — failures return {})
            try:
                sc = get_screener().scrape(ticker)
            except Exception:
                sc = {}

            # Prefer screener.in P/E and ROE if yfinance is missing them
            if sc.get("pe_screener") and not m.get("pe"):
                m["pe"] = sc["pe_screener"]
            if sc.get("roe_screener") and not m.get("roe"):
                m["roe"] = sc["roe_screener"]

            score = self._compute_score(m, sc)
            grade = self._grade(score)
            self._cache[ticker] = {
                "score":      score,
                "grade":      grade,
                "metrics":    m,
                "screener":   sc,
                "fetched_at": datetime.now().isoformat(),
            }
            self._save_cache()

        entry = self._cache.get(ticker, {})
        return {
            "score":    entry.get("score", 0.5),
            "grade":    entry.get("grade", "C"),
            "metrics":  entry.get("metrics", {}),
            "screener": entry.get("screener", {}),
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
