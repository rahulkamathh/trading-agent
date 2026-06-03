"""
Indian Market Institutional Trading Engine
==========================================
Strategies: Cross-Sectional Momentum | Mean Reversion | Multi-Factor | Sector Rotation
Universe  : NSE Equities, Nifty 50 Index, F&O (paper), Sectoral ETFs
Capital   : ₹10,00,000 (paper trading)
Data      : yfinance — NSE historical data (max available, ~25 yrs for most stocks)
"""

import json
import os
import time
import logging
import requests
import io
from datetime import datetime, timedelta, time as dt_time
import time
from zoneinfo import ZoneInfo
from learning_engine import get_learning_engine
from notifier import get_notifier
from risk_manager import get_risk_manager

_IST_TZ = ZoneInfo("Asia/Kolkata")


def _now_ist() -> datetime:
    """Return current datetime in IST (UTC+5:30)."""
    return datetime.now(_IST_TZ)


def _market_open() -> bool:
    """True if current IST time is within NSE trading hours 9:15–15:30."""
    now = _now_ist().time()
    return dt_time(9, 15) <= now <= dt_time(15, 30)
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import ta

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Universe Definition
# ---------------------------------------------------------------------------

# ─────────────────────────────────────────────────────────────────────────────
# Full NSE/BSE Trading Universe — organised by sector
# ~260 liquid stocks covering every major industry on the Indian market.
# yfinance batch-fetches these; stocks with no data are silently skipped.
# ─────────────────────────────────────────────────────────────────────────────
NIFTY50_TICKERS = [

    # ══ BANKING — Private ═════════════════════════════════════════════════
    "HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","AXISBANK.NS","INDUSINDBK.NS",
    "BANDHANBNK.NS","FEDERALBNK.NS","RBLBANK.NS","YESBANK.NS","IDFCFIRSTB.NS",
    "KARURVYSYA.NS","DCBBANK.NS","SOUTHBANK.NS","CSBBANK.NS",
    "UJJIVANSFB.NS","EQUITASBNK.NS","AUBANK.NS",

    # ══ BANKING — PSU ════════════════════════════════════════════════════
    "SBIN.NS","CANBK.NS","PNB.NS","BANKBARODA.NS","UNIONBANK.NS",
    "INDIANB.NS","MAHABANK.NS","IOB.NS","UCOBANK.NS","CENTRALBK.NS","BANKINDIA.NS",

    # ══ NBFC / HOUSING FINANCE ════════════════════════════════════════════
    "BAJFINANCE.NS","BAJAJFINSV.NS","MUTHOOTFIN.NS","MANAPPURAM.NS",
    "CHOLAFIN.NS","SHRIRAMFIN.NS","LICHSGFIN.NS","RECLTD.NS","PFC.NS",
    "AAVAS.NS","HOMEFIRST.NS","APTUS.NS","CANFINHOME.NS","PNBHOUSING.NS",
    "BAJAJHLDNG.NS","PIRAMAL.NS","IDFC.NS","M&MFIN.NS",

    # ══ INSURANCE ════════════════════════════════════════════════════════
    "SBILIFE.NS","HDFCLIFE.NS","ICICIPRULI.NS","LICI.NS",
    "ICICIGI.NS","NIACL.NS","STARHEALTH.NS","POLICYBZR.NS",

    # ══ INFORMATION TECHNOLOGY ════════════════════════════════════════════
    "TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS",
    "LTIM.NS","LTTS.NS","TATAELXSI.NS","MPHASIS.NS","COFORGE.NS",
    "PERSISTENT.NS","KPIT.NS","NAUKRI.NS","ZOMATO.NS","PAYTM.NS","NYKAA.NS",

    # ══ PHARMACEUTICALS ══════════════════════════════════════════════════
    "SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","BIOCON.NS",
    "AUROPHARMA.NS","LUPIN.NS","TORNTPHARM.NS","ALKEM.NS","IPCALAB.NS",
    "GLENMARK.NS","NATCOPHARM.NS","ZYDUSLIFE.NS","ABBOTINDIA.NS",
    "PFIZER.NS","SANOFI.NS","LAURUSLABS.NS","GRANULES.NS","SYNGENE.NS",
    "AJANTPHARM.NS","MARKSANS.NS","SUVENPHAR.NS","NEULANDLAB.NS",

    # ══ HEALTHCARE / DIAGNOSTICS ══════════════════════════════════════════
    "APOLLOHOSP.NS","MAXHEALTH.NS","FORTIS.NS","NARAYANA.NS","ASTER.NS",
    "METROPOLIS.NS","LALPATHLAB.NS","THYROCARE.NS","KRSNAA.NS","VIJAYADIAG.NS",

    # ══ FMCG / CONSUMER STAPLES ════════════════════════════════════════════
    "HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","BRITANNIA.NS","DABUR.NS",
    "MARICO.NS","GODREJCP.NS","EMAMILTD.NS","COLPAL.NS","TATACONSUM.NS",
    "VBL.NS","RADICO.NS","UNITDSPR.NS","UBL.NS","PGHH.NS","GILLETTE.NS",

    # ══ CONSUMER DISCRETIONARY / RETAIL ════════════════════════════════════
    "TITAN.NS","DMART.NS","TRENT.NS","JUBLFOOD.NS","DEVYANI.NS",
    "WESTLIFE.NS","SAPPHIRE.NS","PAGEIND.NS","MANYAVAR.NS",
    "RELAXO.NS","BATA.NS","SHOPERSTOP.NS","VMART.NS","VSTIL.NS",
    "WHIRLPOOL.NS","VOLTAS.NS","HAVELLS.NS","CROMPTON.NS","ORIENT.NS",

    # ══ AUTOMOBILES & AUTO COMPONENTS ═════════════════════════════════════
    "MARUTI.NS","TATAMOTORS.NS","M&M.NS","EICHERMOT.NS","BAJAJ-AUTO.NS",
    "HEROMOTOCO.NS","TVSMOTOR.NS","ASHOKLEY.NS","ESCORTS.NS","OLECTRA.NS",
    "BALKRISIND.NS","APOLLOTYRE.NS","CEATLTD.NS","MOTHERSON.NS",
    "BOSCHLTD.NS","BHARATFORG.NS","SUNDRMFAST.NS","ENDURANCE.NS",
    "EXIDEIND.NS","AMARAJABAT.NS","SUBROS.NS","SUPRAJIT.NS",

    # ══ OIL & GAS / ENERGY ════════════════════════════════════════════════
    "RELIANCE.NS","ONGC.NS","IOC.NS","BPCL.NS","GAIL.NS",
    "HINDPETRO.NS","PETRONET.NS","MGL.NS","IGL.NS","GSPL.NS","ATGL.NS",
    "CASTROLIND.NS","MRPL.NS","CHENNPETRO.NS",

    # ══ POWER & RENEWABLES ════════════════════════════════════════════════
    "NTPC.NS","POWERGRID.NS","ADANIGREEN.NS","TATAPOWER.NS","CESC.NS",
    "TORNTPOWER.NS","SJVN.NS","NHPC.NS","JPPOWER.NS","JSPL.NS",
    "SUZLON.NS","GREENKO.NS","RPOWER.NS","INOXWIND.NS",

    # ══ METALS & MINING ════════════════════════════════════════════════════
    "TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","VEDL.NS","SAIL.NS",
    "COALINDIA.NS","NMDC.NS","NATIONALUM.NS","MOIL.NS","HINDZINC.NS",
    "APLAPOLLO.NS","RATNAMANI.NS","JINDALSTEL.NS","WELCORP.NS",

    # ══ CEMENT ════════════════════════════════════════════════════════════
    "ULTRACEMCO.NS","GRASIM.NS","AMBUJACEM.NS","ACC.NS","SHREECEM.NS",
    "RAMCOCEM.NS","JKCEMENT.NS","DALMIA.NS","JKLAKSHMI.NS","PRISM.NS",
    "HEIDELBERG.NS","BIRLACORPN.NS","NUVOCO.NS",

    # ══ CHEMICALS ═════════════════════════════════════════════════════════
    "PIDILITIND.NS","DEEPAKNTR.NS","ALKYLAMINE.NS","AARTI.NS","VINATI.NS",
    "SRF.NS","GALAXYSURF.NS","PCBL.NS","FINEORG.NS","NAVINFLUOR.NS",
    "TATACHEM.NS","GHCL.NS","GUJFLUORO.NS","CLEAN.NS","ATUL.NS",
    "NOCIL.NS","BALCHEMIND.NS","THIRUMALCHM.NS",

    # ══ PAINTS & COATINGS ═════════════════════════════════════════════════
    "ASIANPAINT.NS","BERGEPAINT.NS","KANSAINER.NS","AKZONOBEL.NS",
    "INDIGO.NS","SHALPAINTS.NS",

    # ══ CAPITAL GOODS / INDUSTRIALS ════════════════════════════════════════
    "LT.NS","SIEMENS.NS","ABB.NS","BHEL.NS","THERMAX.NS",
    "CUMMINSIND.NS","GRINDWELL.NS","TIMKEN.NS","SKF.NS","SCHAEFFLER.NS",
    "BEL.NS","BEML.NS","ELGIEQUIP.NS","KENNAMETAL.NS","AZIMUT.NS",

    # ══ ELECTRICAL / WIRES & CABLES ════════════════════════════════════════
    "POLYCAB.NS","KEIINDS.NS","FINOLEX.NS","HAVELLS.NS","VGUARD.NS",

    # ══ DEFENCE & AEROSPACE ════════════════════════════════════════════════
    "HAL.NS","COCHINSHIP.NS","MAZDOCK.NS","GRSE.NS","MTAR.NS",
    "DATAPATTNS.NS","IDEAFORGE.NS",

    # ══ INFRASTRUCTURE / CONSTRUCTION ══════════════════════════════════════
    "ADANIPORTS.NS","ADANIENT.NS","GMRAIRPORT.NS","IRB.NS",
    "NCC.NS","KPITL.NS","PNCINFRA.NS","HGINFRA.NS","RVNL.NS","IRCON.NS",

    # ══ REAL ESTATE ════════════════════════════════════════════════════════
    "DLF.NS","GODREJPROP.NS","OBEROIRLTY.NS","PRESTIGE.NS","SOBHA.NS",
    "BRIGADE.NS","PHOENIXLTD.NS","MACROTECH.NS","SUNTECK.NS","KOLTEPATIL.NS",

    # ══ LOGISTICS & AVIATION ════════════════════════════════════════════════
    "IRCTC.NS","INDIGO.NS","BLUEDART.NS","DELHIVERY.NS","CONCOR.NS",
    "GATI.NS","TCI.NS","MAHLOG.NS","SPICEJET.NS",

    # ══ TELECOM ════════════════════════════════════════════════════════════
    "BHARTIARTL.NS","IDEA.NS","TATACOMM.NS","HFCL.NS","TEJAS.NS",

    # ══ MEDIA & ENTERTAINMENT ══════════════════════════════════════════════
    "ZEEL.NS","SUNTV.NS","PVRINOX.NS","SAREGAMA.NS","TIPS.NS","NAVNETEDUL.NS",

    # ══ HOTELS & HOSPITALITY ════════════════════════════════════════════════
    "EIHOTEL.NS","INDHOTEL.NS","CHALET.NS","LEMON.NS","MHRIL.NS",

    # ══ TEXTILES & APPAREL ══════════════════════════════════════════════════
    "RAYMOND.NS","ARVIND.NS","WELSPUN.NS","TRIDENT.NS","VARDHMAN.NS",
    "KTEX.NS","NITIN.NS","GRASIM.NS",

    # ══ AGRICULTURE & FOOD PROCESSING ══════════════════════════════════════
    "KRBL.NS","LTFOODS.NS","AVANTIFEED.NS","WATERBASE.NS","GODREJAGRO.NS",
    "PATANJALIFOODS.NS","DHANUKA.NS","PIIND.NS",

    # ══ PSU / DIVERSIFIED ════════════════════════════════════════════════
    "HUDCO.NS","IRFC.NS","RECLTD.NS","IDBI.NS","IFCI.NS",
    "MMTC.NS","MSTC.NS","NBCC.NS","RAILTEL.NS","RITES.NS",
]


# ---------------------------------------------------------------------------
# Dynamic NSE Universe Loader
# ---------------------------------------------------------------------------
_NSE_EQUITY_CSV = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
_NSE_UNIVERSE_CACHE = None   # in-memory cache for the current process

def load_nse_universe(force_refresh: bool = False) -> list:
    """
    Download the full NSE equity list and return every EQ-series stock as a
    list of .NS ticker strings.  Results are cached to data/nse_universe.json
    for 24 h so we don't hammer NSE on every restart.

    Falls back to the hardcoded NIFTY50_TICKERS if the download fails.
    """
    global _NSE_UNIVERSE_CACHE
    if _NSE_UNIVERSE_CACHE and not force_refresh:
        return _NSE_UNIVERSE_CACHE

    cache_path = DATA_DIR / "nse_universe.json"

    # Return file-cached list if fresh (< 24 h)
    if not force_refresh and cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < 24:
            try:
                with open(cache_path) as f:
                    tickers = json.load(f)
                if tickers:
                    _NSE_UNIVERSE_CACHE = tickers
                    logger.info(f"NSE universe loaded from cache: {len(tickers)} stocks")
                    return tickers
            except Exception:
                pass

    logger.info("Downloading full NSE equity list…")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept-Encoding": "gzip, deflate",
            "Accept": "*/*",
            "Referer": "https://www.nseindia.com/",
        }
        resp = requests.get(_NSE_EQUITY_CSV, headers=headers, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))

        # NSE CSV columns: SYMBOL, NAME OF COMPANY, SERIES, DATE OF LISTING, …
        # Keep only regular equity (EQ series) — excludes ETFs, SME, debt, etc.
        if "SERIES" in df.columns:
            df = df[df["SERIES"].str.strip() == "EQ"]

        symbols = df["SYMBOL"].dropna().str.strip().unique().tolist()
        tickers = [s + ".NS" for s in symbols if s]

        with open(cache_path, "w") as f:
            json.dump(tickers, f)
        _NSE_UNIVERSE_CACHE = tickers
        logger.info(f"NSE universe refreshed: {len(tickers)} EQ stocks")
        return tickers

    except Exception as exc:
        logger.warning(f"NSE universe download failed ({exc}), using hardcoded fallback.")
        _NSE_UNIVERSE_CACHE = NIFTY50_TICKERS
        return NIFTY50_TICKERS


SECTOR_ETFS = {
    "Nifty50":   "NIFTYBEES.NS",
    "Banking":   "BANKBEES.NS",
    "IT":        "ITBEES.NS",
    "Pharma":    "PHARMABEES.NS",
    "Gold":      "GOLDBEES.NS",
}

# Small/micro cap universe — high upside potential, lower market cap
PENNY_UNIVERSE = [
    # Renewables & Power
    "SUZLON.NS", "NHPC.NS", "SJVN.NS", "TATAPOWER.NS", "JPPOWER.NS",
    "CESC.NS", "NTPC.NS", "POWERGRID.NS",
    # PSU / Infrastructure
    "RVNL.NS", "IRCON.NS", "IRFC.NS", "HUDCO.NS", "RECLTD.NS",
    "PFC.NS", "BEML.NS", "BEL.NS",
    # Metals & Mining
    "SAIL.NS", "NMDC.NS", "NATIONALUM.NS", "MOIL.NS", "HINDZINC.NS",
    # Defence & Aerospace
    "HAL.NS", "COCHINSHIP.NS", "MAZDOCK.NS", "GRSE.NS",
    # Banking & Finance (turnaround plays)
    "YESBANK.NS", "IDFCFIRSTB.NS", "FEDERALBNK.NS", "BANDHANBNK.NS",
    # Chemicals & Pharma
    "LAURUSLABS.NS", "GRANULES.NS", "SYNGENE.NS", "PCBL.NS",
    # Telecom
    "IDEA.NS",
    # Others
    "GMRAIRPORT.NS", "ADANIGREEN.NS",
]

INDEX_TICKERS = {
    "Nifty50":   "^NSEI",
    "BankNifty": "^NSEBANK",
}

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
TRADE_LOG_FILE = DATA_DIR / "trade_log.json"
SIGNALS_FILE   = DATA_DIR / "signals.json"

INITIAL_CAPITAL  = 1_300_000   # ₹13 lakhs (₹10L equity + ₹2L F&O + ₹1L commodity)
FNO_RESERVE      = 2_00_000   # ₹2L ring-fenced for F&O premiums — equity cannot touch this
COMMODITY_RESERVE= 1_00_000   # ₹1L ring-fenced for commodity margins — equity cannot touch this
EQUITY_CASH_FLOOR= FNO_RESERVE + COMMODITY_RESERVE  # ₹3L minimum cash equity must leave intact
MAX_POSITION_PCT = 0.08        # max 8% per position
STOP_LOSS_PCT    = 0.07        # 7% stop loss
TAKE_PROFIT_PCT  = 0.20        # 20% take profit
MAX_POSITIONS    = 17          # locked at current 17 — cash reserved for F&O/commodity

# ── Execution guard-rails ──────────────────────────────────────────────────
MIN_BUY_STRENGTH = 65   # signal strength threshold to open a position (0–100)
COOLDOWN_DAYS    = 3    # days before re-buying a ticker that was stopped out / took profit
MIN_HOLD_DAYS    = 1    # min days before a position can be exited (even by stop-loss signal flip)

# ── Dynamic SL / TP via ATR ────────────────────────────────────────────────
# Stop = entry − ATR_SL_MULT × ATR(14)
# TP   = entry + stop_distance × planned_RR
# planned_RR scales with signal strength (see _compute_sl_tp)
ATR_PERIOD      = 14    # bars used for ATR
ATR_SL_MULT     = 1.5   # stop distance = 1.5 × ATR
#
# EXIT POLICY: positions close ONLY via stop-loss (−7%) or take-profit (+20%).
# Strategy SELL signals are generated and displayed on the dashboard but
# never trigger an actual exit — they're informational only.

# ---------------------------------------------------------------------------
# Data Layer
# ---------------------------------------------------------------------------

class DataFetcher:
    """Fetches and caches NSE market data via yfinance."""

    _cache: dict = {}

    # ── Bad-ticker skip cache ─────────────────────────────────────────────────
    # Tickers that return empty data are tracked. After _FAIL_THRESHOLD consecutive
    # failures the ticker is added to _skip_set and silently skipped on future
    # calls (avoiding wasted HTTP round-trips for dead / delisted symbols).
    # The skip set is cleared every _SKIP_RESET_CALLS to retry periodically.
    _fail_counts:      dict[str, int] = {}
    _skip_set:         set[str]       = set()
    _fetch_call_count: int            = 0
    _FAIL_THRESHOLD   = 3
    _SKIP_RESET_CALLS = 500   # ~once per trading day at normal cycle frequency

    @classmethod
    def fetch(cls, ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
        # Periodically reset the skip set so tickers get a second chance
        cls._fetch_call_count += 1
        if cls._fetch_call_count % cls._SKIP_RESET_CALLS == 0:
            if cls._skip_set:
                logger.info(f"[DataFetcher] Clearing skip set ({len(cls._skip_set)} tickers) — will retry")
            cls._skip_set.clear()
            cls._fail_counts.clear()

        # Skip tickers that have consistently failed this session
        if ticker in cls._skip_set:
            return pd.DataFrame()

        key = f"{ticker}_{period}_{interval}"
        if key in cls._cache:
            return cls._cache[key]
        try:
            df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
            if df.empty:
                # Fallback: try Angel One historical data API
                try:
                    from angelone_feed import get_feed as _get_feed  # noqa: PLC0415
                    _feed = _get_feed()
                    if _feed.is_connected():
                        df_ao = _feed.get_historical(ticker, period=period, interval=interval)
                        if df_ao is not None and not df_ao.empty:
                            logger.info(f"[DataFetcher] yfinance empty for {ticker} — using Angel One historical data")
                            cls._fail_counts.pop(ticker, None)
                            cls._cache[key] = df_ao
                            return df_ao
                except Exception as _ao_err:
                    logger.debug(f"[DataFetcher] Angel One fallback failed for {ticker}: {_ao_err}")

                count = cls._fail_counts.get(ticker, 0) + 1
                cls._fail_counts[ticker] = count
                if count >= cls._FAIL_THRESHOLD:
                    cls._skip_set.add(ticker)
                    logger.info(f"[DataFetcher] Skipping {ticker} — no data after {count} attempts")
                else:
                    logger.warning(f"No data for {ticker} (attempt {count}/{cls._FAIL_THRESHOLD})")
                return pd.DataFrame()
            # Success — reset fail count and cache
            cls._fail_counts.pop(ticker, None)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.dropna(inplace=True)
            cls._cache[key] = df
            return df
        except Exception as e:
            logger.error(f"Fetch error for {ticker}: {e}")
            return pd.DataFrame()

    @classmethod
    def fetch_max(cls, ticker: str) -> pd.DataFrame:
        return cls.fetch(ticker, period="max", interval="1d")

    @classmethod
    def fetch_multi(cls, tickers: list, period: str = "2y", batch_size: int = 200) -> dict:
        """
        Batch-download historical data for a list of tickers.
        Uses yfinance's multi-ticker download (single HTTP call per batch)
        so even 1 000+ tickers are fetched in seconds rather than minutes.
        Results are cached per (ticker, period) in cls._cache.
        """
        if not tickers:
            return {}

        # Check which tickers are already cached
        missing = [t for t in tickers if f"{t}_{period}_1d" not in cls._cache]
        result  = {t: cls._cache[f"{t}_{period}_1d"] for t in tickers if f"{t}_{period}_1d" in cls._cache}

        # Batch-fetch missing ones in chunks of batch_size
        for i in range(0, len(missing), batch_size):
            chunk = missing[i:i + batch_size]
            try:
                raw = yf.download(
                    " ".join(chunk),
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    group_by="ticker",
                    threads=True,
                )
                if raw.empty:
                    continue

                # Single ticker: yfinance returns flat columns, not grouped
                if len(chunk) == 1:
                    t = chunk[0]
                    if not raw.empty:
                        raw.columns = [c if isinstance(c, str) else c[0] for c in raw.columns]
                        raw = raw.dropna(how="all")
                        if not raw.empty:
                            cls._cache[f"{t}_{period}_1d"] = raw
                            result[t] = raw
                    continue

                # Multi-ticker: top-level columns are tickers
                for t in chunk:
                    try:
                        df = raw[t].copy() if t in raw.columns.get_level_values(0) else pd.DataFrame()
                        df = df.dropna(how="all")
                        if not df.empty:
                            cls._cache[f"{t}_{period}_1d"] = df
                            result[t] = df
                    except Exception:
                        pass
            except Exception as exc:
                logger.debug(f"Batch download error (chunk {i}–{i+batch_size}): {exc}")
                # fall back to individual fetches for this chunk
                for t in chunk:
                    df = cls.fetch(t, period=period)
                    if not df.empty:
                        result[t] = df

        return result

    # Per-ticker price cache with a short TTL so stop-loss checks stay current
    _price_cache: dict = {}   # ticker → (price, timestamp)
    _PRICE_TTL   = 120        # seconds before re-fetching via fast_info

    @classmethod
    def get_current_price(cls, ticker: str) -> float:
        """
        Return the best available current price for a ticker.
        Priority:
          1. Angel One SmartAPI live feed (sub-second, if configured)
          2. yfinance fast_info.last_price (~15 min delayed, but intraday)
          3. Cached OHLCV last close (EOD, stale during market hours)
        Result is cached for _PRICE_TTL seconds to avoid hammering yfinance
        on every stop-loss check across hundreds of positions.
        """
        # 1. Angel One live feed
        try:
            from angelone_feed import get_feed  # pylint: disable=import-outside-toplevel
            live_price = get_feed().get_price(ticker)
            if live_price and live_price > 0:
                return live_price
        except ImportError:
            pass

        # 2. Short-TTL intraday price via fast_info
        now = time.time()
        cached = cls._price_cache.get(ticker)
        if cached and now - cached[1] < cls._PRICE_TTL:
            return cached[0]

        try:
            fi    = yf.Ticker(ticker).fast_info
            price = float(fi.last_price or 0)
            if price > 0:
                cls._price_cache[ticker] = (price, now)
                return price
        except Exception:
            pass

        # 3. Last close from historical data (EOD fallback)
        df = cls.fetch(ticker, period="5d")
        if not df.empty:
            price = float(df["Close"].iloc[-1])
            cls._price_cache[ticker] = (price, now)
            return price
        return 0.0

    @classmethod
    def clear_cache(cls):
        cls._cache = {}


# ---------------------------------------------------------------------------
# Technical Indicators Helper
# ---------------------------------------------------------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add a rich set of TA indicators to a OHLCV dataframe."""
    if df.empty or len(df) < 50:
        return df
    c = df["Close"].squeeze()
    h = df["High"].squeeze()
    l = df["Low"].squeeze()
    v = df["Volume"].squeeze() if "Volume" in df.columns else None

    # Trend
    df["ema_20"]  = ta.trend.ema_indicator(c, window=20)
    df["ema_50"]  = ta.trend.ema_indicator(c, window=50)
    df["ema_200"] = ta.trend.ema_indicator(c, window=200)
    df["adx"]     = ta.trend.adx(h, l, c, window=14)

    # Momentum
    df["rsi"]      = ta.momentum.rsi(c, window=14)
    df["roc_1m"]   = c.pct_change(21)   # 1-month return
    df["roc_3m"]   = c.pct_change(63)   # 3-month return
    df["roc_6m"]   = c.pct_change(126)  # 6-month return
    df["roc_12m"]  = c.pct_change(252)  # 12-month return

    # Volatility
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"]   = bb.bollinger_mavg()
    df["bb_pct"]   = bb.bollinger_pband()
    df["atr"]      = ta.volatility.average_true_range(h, l, c, window=14)

    # Volume
    if v is not None:
        df["vol_sma20"] = v.rolling(20).mean()
        df["vol_ratio"] = v / df["vol_sma20"]

    return df


# ---------------------------------------------------------------------------
# Strategy 1: Cross-Sectional Momentum (Quant/CTA Style)
# ---------------------------------------------------------------------------

class MomentumStrategy:
    """
    12-1 Month Cross-Sectional Momentum (Jegadeesh & Titman style).
    Universe : Nifty 50 stocks
    Signal   : 12m return minus most recent 1m (avoids reversal)
    Filter   : Stock above 200 EMA, ADX > 20 (trending)
    Rebalance: Monthly
    """
    name = "Cross-Sectional Momentum"
    short_name = "MOMENTUM"

    def generate_signals(self, data: dict) -> list:
        scores = []
        for ticker, df in data.items():
            df = add_indicators(df)
            if len(df) < 260:
                continue
            row = df.iloc[-1]
            # Momentum score: skip last month
            mom_score = df["Close"].iloc[-252] if len(df) >= 252 else np.nan
            if pd.isna(mom_score) or mom_score == 0:
                continue
            ret_12_1 = (df["Close"].iloc[-22] / df["Close"].iloc[-252]) - 1

            # Regime filters
            above_200 = row["Close"] > row.get("ema_200", 0)
            adx_ok    = row.get("adx", 0) > 20
            if not above_200:
                continue

            scores.append({
                "ticker":    ticker,
                "score":     ret_12_1,
                "price":     float(row["Close"]),
                "rsi":       float(row.get("rsi", 50)),
                "adx":       float(row.get("adx", 0)),
                "strategy":  self.short_name,
                "signal":    "BUY" if ret_12_1 > 0.05 else "NEUTRAL",
            })

        scores.sort(key=lambda x: x["score"], reverse=True)
        # Top quintile → BUY, bottom quintile → SELL
        n = max(1, len(scores) // 5)
        for i, s in enumerate(scores):
            if i < n:
                s["signal"] = "BUY"
                s["strength"] = min(100, int(50 + s["score"] * 200))
            elif i >= len(scores) - n:
                s["signal"] = "SELL"
                s["strength"] = max(0, int(50 - abs(s["score"]) * 200))
            else:
                s["strength"] = 50
        return scores


# ---------------------------------------------------------------------------
# Strategy 2: Mean Reversion (Statistical Arb Style)
# ---------------------------------------------------------------------------

class MeanReversionStrategy:
    """
    RSI + Bollinger Band mean reversion.
    Buy when price at lower BB AND RSI < 35 AND volume spike.
    Sell when price at upper BB OR RSI > 65.
    Short-term (5-15 day hold).
    """
    name = "Mean Reversion"
    short_name = "MEAN_REV"

    def generate_signals(self, data: dict) -> list:
        signals = []
        for ticker, df in data.items():
            df = add_indicators(df)
            if len(df) < 50:
                continue
            row = df.iloc[-1]

            rsi     = row.get("rsi", 50)
            bb_pct  = row.get("bb_pct", 0.5)
            vol_r   = row.get("vol_ratio", 1.0)
            price   = float(row["Close"])
            ema200  = row.get("ema_200", price)

            # Only in uptrend (price > 200 EMA) for longs
            in_uptrend = price > ema200

            if in_uptrend and rsi < 35 and bb_pct < 0.1 and vol_r > 1.2:
                sig = "BUY"
                strength = int(100 - rsi)
            elif rsi > 65 and bb_pct > 0.9:
                sig = "SELL"
                strength = int(rsi)
            else:
                sig = "NEUTRAL"
                strength = 50

            signals.append({
                "ticker":   ticker,
                "score":    (0.5 - bb_pct) + (50 - rsi) / 100,
                "price":    price,
                "rsi":      float(rsi),
                "bb_pct":   float(bb_pct),
                "strategy": self.short_name,
                "signal":   sig,
                "strength": strength,
            })
        return [s for s in signals if s["signal"] != "NEUTRAL"]


# ---------------------------------------------------------------------------
# Strategy 3: Multi-Factor Institutional (FII/MF Style)
# ---------------------------------------------------------------------------

class MultiFactorStrategy:
    """
    Combines Momentum + Quality + Low-Volatility factors.
    Proxy for Quality: higher price stability, lower drawdown, consistent trend.
    Monthly rebalance. Holds 8-12 positions.
    """
    name = "Multi-Factor"
    short_name = "MULTIFACTOR"

    def generate_signals(self, data: dict) -> list:
        records = []
        for ticker, df in data.items():
            df = add_indicators(df)
            if len(df) < 200:
                continue
            row = df.iloc[-1]
            close = df["Close"].squeeze()

            # Factor 1: Momentum (3m + 6m)
            mom_3m = float(row.get("roc_3m", 0) or 0)
            mom_6m = float(row.get("roc_6m", 0) or 0)
            mom_factor = 0.4 * mom_3m + 0.6 * mom_6m

            # Factor 2: Low Volatility (inverse 1yr realized vol)
            vol_1y = close.pct_change().rolling(252).std().iloc[-1]
            vol_factor = -float(vol_1y) if not pd.isna(vol_1y) else 0

            # Factor 3: Trend Quality (% above 200 EMA)
            ema200 = row.get("ema_200", float(close.iloc[-1]))
            trend_factor = (float(close.iloc[-1]) - float(ema200)) / float(ema200) if ema200 else 0

            composite = 0.4 * mom_factor + 0.3 * vol_factor * 10 + 0.3 * trend_factor

            records.append({
                "ticker":    ticker,
                "score":     composite,
                "price":     float(row["Close"]),
                "rsi":       float(row.get("rsi", 50)),
                "mom_3m":    round(mom_3m * 100, 2),
                "mom_6m":    round(mom_6m * 100, 2),
                "strategy":  self.short_name,
                "signal":    "NEUTRAL",
                "strength":  50,
            })

        records.sort(key=lambda x: x["score"], reverse=True)
        n = max(1, len(records) // 4)
        for i, r in enumerate(records):
            if i < n:
                r["signal"]   = "BUY"
                r["strength"] = min(100, int(60 + r["score"] * 100))
            elif i >= len(records) - n:
                r["signal"]   = "SELL"
                r["strength"] = max(0, int(40 + r["score"] * 100))
        return [r for r in records if r["signal"] != "NEUTRAL"]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Strategy 4b: SMA Crossover (Trend Following)
# ---------------------------------------------------------------------------

class SMAStrategy:
    """
    Simple Moving Average trend system.
    - Golden cross (50>200) or full alignment (price>20>50>200) → BUY
    - Death cross (50<200) or full inversion → SELL
    Strength scales with how cleanly the SMA stack is aligned.
    """
    name       = "SMA Crossover"
    short_name = "SMA"

    def generate_signals(self, data: dict) -> list:
        signals = []
        for ticker, df in data.items():
            if len(df) < 210:
                continue
            close = df["Close"].squeeze()
            sma20  = close.rolling(20).mean()
            sma50  = close.rolling(50).mean()
            sma200 = close.rolling(200).mean()

            curr      = float(close.iloc[-1])
            s20, s50, s200       = float(sma20.iloc[-1]),  float(sma50.iloc[-1]),  float(sma200.iloc[-1])
            s50p, s200p          = float(sma50.iloc[-2]),  float(sma200.iloc[-2])

            golden_cross = s50p <= s200p and s50 > s200
            death_cross  = s50p >= s200p and s50 < s200

            if   curr > s20 > s50 > s200: sig, strength, score = "BUY",  90, (curr-s200)/s200
            elif curr < s20 < s50 < s200: sig, strength, score = "SELL", 90, -(s200-curr)/s200
            elif golden_cross:             sig, strength, score = "BUY",  85, (s50-s200)/s200
            elif death_cross:              sig, strength, score = "SELL", 85, -(s200-s50)/s200
            elif curr > s50 > s200:        sig, strength, score = "BUY",  65, (curr-s50)/s50
            elif curr < s50 < s200:        sig, strength, score = "SELL", 65, -(s50-curr)/s50
            else:
                continue

            signals.append({
                "ticker":   ticker, "signal": sig, "price": round(curr, 2),
                "score":    round(score, 4), "rsi": 50.0,
                "strategy": self.short_name, "strength": strength,
                "sma20": round(s20,2), "sma50": round(s50,2), "sma200": round(s200,2),
            })
        return signals


# ---------------------------------------------------------------------------
# Strategy 4c: Fibonacci Retracement
# ---------------------------------------------------------------------------

class FibonacciStrategy:
    """
    Auto-detects 52-week swing high/low, computes Fib retracement levels
    (23.6%, 38.2%, 50%, 61.8%, 78.6%).
    BUY when price bounces off deep support fib (≥50%) in uptrend.
    SELL when price hits shallow resistance fib (≤38.2%) in downtrend.
    Only triggers when price is within 2% of a key Fib level.
    """
    name       = "Fibonacci Retracement"
    short_name = "FIBONACCI"
    FIB_LEVELS = [0.236, 0.382, 0.500, 0.618, 0.786]
    TOLERANCE  = 0.025   # 2.5% proximity to trigger

    def generate_signals(self, data: dict) -> list:
        signals = []
        for ticker, df in data.items():
            if len(df) < 252:
                continue
            year_df    = df.tail(252)
            swing_high = float(year_df["High"].max())
            swing_low  = float(year_df["Low"].min())
            rang       = swing_high - swing_low
            if rang < 1:
                continue

            curr     = float(df["Close"].iloc[-1])
            sma200   = float(df["Close"].rolling(200).mean().iloc[-1])
            uptrend  = curr > sma200

            # Fib levels from high to low (retracement from high)
            fib_prices = {lvl: swing_high - lvl * rang for lvl in self.FIB_LEVELS}

            for lvl, fp in fib_prices.items():
                if abs(curr - fp) / fp > self.TOLERANCE:
                    continue
                # BUY: near deep support in uptrend
                if uptrend and lvl >= 0.382 and curr >= fp * 0.99:
                    sig      = "BUY"
                    strength = min(95, int(50 + lvl * 60))
                    score    = lvl
                # SELL: near resistance in downtrend
                elif not uptrend and lvl <= 0.382 and curr <= fp * 1.01:
                    sig      = "SELL"
                    strength = min(85, int(50 + (1 - lvl) * 50))
                    score    = -lvl
                else:
                    continue

                signals.append({
                    "ticker":     ticker, "signal": sig, "price": round(curr, 2),
                    "score":      round(score, 4), "rsi": 50.0,
                    "strategy":   self.short_name, "strength": strength,
                    "fib_level":  lvl, "fib_price": round(fp, 2),
                    "swing_high": round(swing_high, 2), "swing_low": round(swing_low, 2),
                })
                break   # only the closest level
        return signals


# ---------------------------------------------------------------------------
# Strategy 4d: RSI Divergence
# ---------------------------------------------------------------------------

class RSIDivergenceStrategy:
    """
    Detects RSI divergence over a rolling 20-day window:
    - Bullish: price makes lower low BUT RSI makes higher low → exhausted sellers → BUY
    - Bearish: price makes higher high BUT RSI makes lower high → exhausted buyers → SELL
    Also catches extreme oversold/overbought with trend confirmation.
    """
    name       = "RSI Divergence"
    short_name = "RSI_DIV"

    def generate_signals(self, data: dict) -> list:
        signals = []
        for ticker, df in data.items():
            df = add_indicators(df)
            if len(df) < 50 or "rsi" not in df.columns:
                continue

            close    = df["Close"].squeeze()
            rsi_s    = df["rsi"]
            window   = 20

            # Recent window for divergence detection
            price_w = close.iloc[-window:]
            rsi_w   = rsi_s.iloc[-window:]

            curr_price = float(close.iloc[-1])
            curr_rsi   = float(rsi_s.iloc[-1])

            # Price extremes in window
            price_min_idx = price_w.idxmin()
            price_max_idx = price_w.idxmax()

            # Bullish divergence: price at new low but RSI higher than its value at price low
            price_low  = float(price_w.min())
            rsi_at_low = float(rsi_w[price_min_idx])

            if (curr_price <= price_low * 1.01           # near low
                    and curr_rsi > rsi_at_low + 5        # RSI higher than it was at price low
                    and curr_rsi < 45):                  # still in bearish RSI zone (room to rally)
                signals.append({
                    "ticker":   ticker, "signal": "BUY", "price": round(curr_price, 2),
                    "score":    (curr_rsi - rsi_at_low) / 100,
                    "rsi":      round(curr_rsi, 1),
                    "strategy": self.short_name, "strength": min(90, int(50 + curr_rsi)),
                })
                continue

            # Bearish divergence: price at new high but RSI lower
            price_high  = float(price_w.max())
            rsi_at_high = float(rsi_w[price_max_idx])

            if (curr_price >= price_high * 0.99
                    and curr_rsi < rsi_at_high - 5
                    and curr_rsi > 55):
                signals.append({
                    "ticker":   ticker, "signal": "SELL", "price": round(curr_price, 2),
                    "score":    -(rsi_at_high - curr_rsi) / 100,
                    "rsi":      round(curr_rsi, 1),
                    "strategy": self.short_name, "strength": min(90, int(50 + (100-curr_rsi))),
                })
        return signals


# ---------------------------------------------------------------------------
# Strategy 4e: Bollinger Squeeze (Volatility Breakout)
# ---------------------------------------------------------------------------

class BollingerSqueezeStrategy:
    """
    Bollinger Band squeeze + breakout:
    - Squeeze: BB width narrows to 6-month low (volatility contraction)
    - After squeeze resolves, price breaks above upper band → BUY
    - After squeeze resolves, price breaks below lower band → SELL
    High conviction because low-volatility periods are followed by big moves.
    """
    name       = "Bollinger Squeeze"
    short_name = "BB_SQUEEZE"

    def generate_signals(self, data: dict) -> list:
        signals = []
        for ticker, df in data.items():
            df = add_indicators(df)
            if len(df) < 130 or "bb_upper" not in df.columns:
                continue

            close    = df["Close"].squeeze()
            bb_upper = df["bb_upper"]
            bb_lower = df["bb_lower"]
            bb_mid   = df["bb_mid"]

            curr        = float(close.iloc[-1])
            band_width  = (bb_upper - bb_lower) / bb_mid
            bw_now      = float(band_width.iloc[-1])
            bw_6m_min   = float(band_width.iloc[-126:].min())
            bw_6m_max   = float(band_width.iloc[-126:].max())

            # Squeeze: current width near 6m minimum
            in_squeeze  = bw_now <= bw_6m_min * 1.10

            # Breakout direction
            broke_up    = curr > float(bb_upper.iloc[-1]) and not in_squeeze
            broke_down  = curr < float(bb_lower.iloc[-1]) and not in_squeeze

            if not broke_up and not broke_down:
                continue

            # Squeeze must have existed recently (within last 10 bars)
            recent_squeeze = (band_width.iloc[-10:] <= bw_6m_min * 1.15).any()
            if not recent_squeeze:
                continue

            sig      = "BUY" if broke_up else "SELL"
            strength = min(95, int(70 + (1 - bw_now / bw_6m_max) * 30))
            score    = (curr - float(bb_mid.iloc[-1])) / float(bb_mid.iloc[-1])

            signals.append({
                "ticker":   ticker, "signal": sig, "price": round(curr, 2),
                "score":    round(score, 4), "rsi": float(df["rsi"].iloc[-1]) if "rsi" in df.columns else 50.0,
                "strategy": self.short_name, "strength": strength,
                "bb_width": round(bw_now, 4),
            })
        return signals


# ---------------------------------------------------------------------------
# Strategy 4f: Volume Breakout
# ---------------------------------------------------------------------------

class VolumeBreakoutStrategy:
    """
    Volume-confirmed price breakout from consolidation:
    - Price breaks 20-day high/low on volume ≥ 2× 20-day average
    - Ensures the move is institutional, not a random spike
    """
    name       = "Volume Breakout"
    short_name = "VOL_BREAK"

    def generate_signals(self, data: dict) -> list:
        signals = []
        for ticker, df in data.items():
            if len(df) < 25 or "Volume" not in df.columns:
                continue

            close   = df["Close"].squeeze()
            vol     = df["Volume"].squeeze()
            curr    = float(close.iloc[-1])
            curr_v  = float(vol.iloc[-1])

            high_20 = float(close.iloc[-21:-1].max())
            low_20  = float(close.iloc[-21:-1].min())
            avg_v20 = float(vol.iloc[-21:-1].mean())

            if avg_v20 <= 0:
                continue
            vol_ratio = curr_v / avg_v20

            if vol_ratio < 2.0:   # require 2× avg volume
                continue

            broke_up   = curr > high_20
            broke_down = curr < low_20

            if not broke_up and not broke_down:
                continue

            sig      = "BUY" if broke_up else "SELL"
            strength = min(95, int(60 + min(vol_ratio, 5) * 7))
            score    = (curr - high_20) / high_20 if broke_up else (low_20 - curr) / low_20

            signals.append({
                "ticker":    ticker, "signal": sig, "price": round(curr, 2),
                "score":     round(score, 4),
                "rsi":       float(df["rsi"].iloc[-1]) if "rsi" in df.columns else 50.0,
                "strategy":  self.short_name, "strength": strength,
                "vol_ratio": round(vol_ratio, 1),
            })
        return signals


# ---------------------------------------------------------------------------
# Strategy 4: Sector Rotation (Macro / FII Style)
# ---------------------------------------------------------------------------

class SectorRotationStrategy:
    """
    Rank sectors by relative strength vs Nifty 50.
    Rotate into top 2 sectors, exit bottom 2.
    Monthly rebalance. Uses sector ETFs.
    """
    name = "Sector Rotation"
    short_name = "SECTOR_ROT"

    def generate_signals(self, etf_data: dict, index_data: pd.DataFrame) -> list:
        signals = []
        if index_data.empty:
            return signals

        nifty_ret_3m = index_data["Close"].pct_change(63).iloc[-1]

        sector_scores = []
        for sector, ticker in SECTOR_ETFS.items():
            if ticker not in etf_data or etf_data[ticker].empty:
                continue
            df = etf_data[ticker]
            if len(df) < 70:
                continue
            ret_3m = df["Close"].pct_change(63).iloc[-1]
            rs     = float(ret_3m) - float(nifty_ret_3m)   # relative strength

            sector_scores.append({
                "sector":  sector,
                "ticker":  ticker,
                "score":   rs,
                "ret_3m":  round(float(ret_3m) * 100, 2),
                "rs":      round(rs * 100, 2),
            })

        sector_scores.sort(key=lambda x: x["score"], reverse=True)
        n = len(sector_scores)
        for i, s in enumerate(sector_scores):
            if i < 2:
                sig = "BUY"; strength = min(100, int(60 + s["score"] * 200))
            elif i >= n - 2:
                sig = "SELL"; strength = max(0, int(40 - abs(s["score"]) * 200))
            else:
                sig = "NEUTRAL"; strength = 50
            signals.append({
                "ticker":   s["ticker"],
                "sector":   s["sector"],
                "score":    s["score"],
                "price":    float(DataFetcher.get_current_price(s["ticker"])),
                "rsi":      50.0,
                "ret_3m":   s["ret_3m"],
                "rs":       s["rs"],
                "strategy": self.short_name,
                "signal":   sig,
                "strength": strength,
            })
        return [s for s in signals if s["signal"] != "NEUTRAL"]


# ---------------------------------------------------------------------------
# Dynamic Stop-Loss / Take-Profit helper
# ---------------------------------------------------------------------------

def _compute_sl_tp(ticker: str, entry: float, strength: float) -> tuple[float, float, float]:
    """
    Return (stop_loss, take_profit, planned_rr) for a new position.

    Stop is set 1.5× ATR(14) below entry so it respects each stock's
    natural volatility — tight on stable stocks, wider on volatile ones.

    Take-profit is set at  stop_distance × planned_rr  above entry.
    planned_rr scales with signal conviction:
        strength < 70  → 1:2.0   (minimum — always at least double the risk)
        strength 70–79 → 1:2.5
        strength 80–89 → 1:3.0
        strength ≥ 90  → 1:4.0   (only for the highest-conviction setups)

    Falls back to flat STOP_LOSS_PCT / planned_rr × STOP_LOSS_PCT if ATR
    data is unavailable or looks wrong.
    """
    # Conviction-based RR target — kept realistic so targets are reachable
    # 1:2 is the institutional minimum; going higher means waiting months for TP
    if strength >= 90:
        planned_rr = 2.5
    elif strength >= 80:
        planned_rr = 2.0
    elif strength >= 70:
        planned_rr = 1.5
    else:
        planned_rr = 1.5

    try:
        df = DataFetcher.fetch(ticker, period="60d", interval="1d")
        if df is not None and not df.empty and len(df) >= ATR_PERIOD:
            high  = df["High"]
            low   = df["Low"]
            close = df["Close"]
            tr    = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low  - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr = tr.rolling(ATR_PERIOD).mean().iloc[-1]

            # Sanity: ATR must be positive and < 15% of price
            if pd.notna(atr) and 0 < atr < entry * 0.15:
                sl_dist = atr * ATR_SL_MULT
                sl  = round(max(entry - sl_dist, entry * 0.50), 2)   # floor at -50%
                tp  = round(entry + sl_dist * planned_rr, 2)
                logger.debug(
                    f"[SL/TP] {ticker}  ATR={atr:.2f}  SL={sl:.2f}  "
                    f"TP={tp:.2f}  RR=1:{planned_rr}"
                )
                return sl, tp, planned_rr
    except Exception as exc:
        logger.debug(f"[SL/TP] ATR compute failed for {ticker}: {exc}")

    # Flat fallback — keep same distance from entry, honour the RR target
    sl_dist = entry * STOP_LOSS_PCT
    sl  = round(entry - sl_dist, 2)
    tp  = round(entry + sl_dist * planned_rr, 2)
    logger.debug(
        f"[SL/TP] {ticker}  flat fallback  SL={sl:.2f}  TP={tp:.2f}  RR=1:{planned_rr}"
    )
    return sl, tp, planned_rr


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------

class Portfolio:
    """Paper trading portfolio with full position management."""

    def __init__(self):
        self.state = self._load()

    def _default_state(self) -> dict:
        return {
            "cash":         INITIAL_CAPITAL,
            "initial":      INITIAL_CAPITAL,
            "positions":    {},          # ticker → {qty, avg_price, strategy, entry_date}
            "realised_pnl": 0.0,
            "exit_log":     {},          # ticker → ISO timestamp of last exit (cooldown tracking)
            "created_at":   _now_ist().isoformat(),
            "last_updated": _now_ist().isoformat(),
        }

    def _load(self) -> dict:
        if PORTFOLIO_FILE.exists():
            with open(PORTFOLIO_FILE) as f:
                return json.load(f)
        state = self._default_state()
        self._save(state)
        return state

    def _save(self, state: dict = None):
        if state:
            self.state = state
        self.state["last_updated"] = _now_ist().isoformat()

        # ── Daily baseline: snapshot portfolio value at start of each new day ──
        today_str = _now_ist().strftime("%Y-%m-%d")
        if self.state.get("day_start_date") != today_str:
            self.state["day_start_date"]  = today_str
            self.state["day_start_value"] = round(self.get_total_value(), 2)

        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(self.state, f, indent=2)

    # ------------------------------------------------------------------ #

    def get_position_value(self, ticker: str) -> float:
        pos = self.state["positions"].get(ticker)
        if not pos:
            return 0.0
        price = DataFetcher.get_current_price(ticker)
        return pos["qty"] * price

    def get_total_value(self) -> float:
        equity = sum(self.get_position_value(t) for t in self.state["positions"])
        return self.state["cash"] + equity

    def get_unrealised_pnl(self) -> float:
        total = 0.0
        for ticker, pos in self.state["positions"].items():
            price = DataFetcher.get_current_price(ticker)
            total += (price - pos["avg_price"]) * pos["qty"]
        return total

    def available_cash(self) -> float:
        return self.state["cash"]

    # ------------------------------------------------------------------ #

    def in_cooldown(self, ticker: str) -> bool:
        """Return True if ticker was exited recently and should not be re-entered yet."""
        exit_log = self.state.get("exit_log", {})
        last_exit = exit_log.get(ticker)
        if not last_exit:
            return False
        try:
            elapsed_days = (_now_ist() - datetime.fromisoformat(last_exit)).days
            return elapsed_days < COOLDOWN_DAYS
        except Exception:
            return False

    def can_buy(self, ticker: str, price: float) -> bool:
        if len(self.state["positions"]) >= MAX_POSITIONS:
            return False
        if ticker in self.state["positions"]:
            return False
        if self.in_cooldown(ticker):
            logger.debug(f"SKIP {ticker} — in cooldown after recent exit")
            return False
        # Cash check: need at least min position size available
        total_val = self.get_total_value()
        min_floor = total_val * 0.005   # 0.5% floor
        return self.state["cash"] >= min_floor and price > 0

    def execute_buy(self, ticker: str, price: float, strategy: str,
                    reason: str = "", strength: float = 65.0) -> dict | None:

        # ── Hard gate 1: market must be open ────────────────────────────── #
        if not _market_open():
            logger.info(
                f"🚫 BLOCKED {ticker} [{strategy}] — market is closed "
                f"(current IST time is outside 9:15–15:30)"
            )
            return None

        # ── Hard gate 2: price sanity check ─────────────────────────────── #
        # Cross-verify the passed price against a fresh yfinance fetch.
        # Reject if the price deviates > 15% from the live quote — bad data.
        try:
            live = DataFetcher.get_current_price(ticker)
            if live and live > 0:
                deviation = abs(price - live) / live
                if deviation > 0.15:
                    logger.warning(
                        f"🚫 PRICE SANITY FAIL {ticker} — "
                        f"passed ₹{price:.2f} vs live ₹{live:.2f} "
                        f"({deviation:.0%} deviation > 15%) — aborting trade"
                    )
                    return None
                # Use the verified live price for execution
                price = live
        except Exception:
            pass

        if not self.can_buy(ticker, price):
            return None

        # ── Dynamic ATR-based SL/TP (single fetch, no duplicate download) ──────
        stop_loss, target, planned_rr = _compute_sl_tp(ticker, price, strength)
        risk_per_unit = round(price - stop_loss, 2)   # ₹ at risk per share

        # Derive ATR from the stop distance _compute_sl_tp already calculated.
        # ATR = stop_dist / ATR_SL_MULT — no extra download needed.
        atr_val: float | None = None
        implied_atr = risk_per_unit / ATR_SL_MULT
        if 0 < implied_atr < price * 0.15:
            atr_val = implied_atr

        # ── Dynamic position sizing via RiskManager ──────────────────────────
        total_val = self.get_total_value()
        rm = get_risk_manager()
        spend = rm.position_size(
            portfolio_value = total_val,
            available_cash  = self.state["cash"],
            price           = price,
            strength        = strength,
            atr             = atr_val,
        )
        qty = int(spend / price)
        if qty < 1:
            return None

        cost = qty * price
        self.state["cash"] -= cost
        self.state["positions"][ticker] = {
            "qty":           qty,
            "avg_price":     price,
            "strategy":      strategy,
            "entry_date":    _now_ist().isoformat(),
            "stop_loss":     stop_loss,      # always fixed -7% from entry — max loss is known
            "target":        target,
            "planned_rr":    planned_rr,
            "risk_per_unit": risk_per_unit,
            "strength":      round(strength, 1),
            "atr_at_entry":  atr_val,        # cached for chandelier trailing (only moves stop UP)
            "highest_close": price,          # tracks peak price for chandelier calculation
        }
        self._save()
        trade = self._log_trade(
            "BUY", ticker, qty, price, strategy, reason,
            stop_loss=stop_loss, target=target,
            planned_rr=planned_rr, risk_per_unit=risk_per_unit,
        )
        logger.info(
            f"BUY  {ticker:20s} qty={qty} @ ₹{price:.2f}  "
            f"SL=₹{stop_loss:.2f}  TP=₹{target:.2f}  RR=1:{planned_rr}  [{strategy}]"
        )
        # ── Live order execution (only when LIVE_TRADING=true) ───────────── #
        try:
            from angelone_trader import get_trader  # pylint: disable=import-outside-toplevel
            get_trader().buy(ticker, qty, price, segment="equity")
        except Exception as _le:
            logger.debug(f"[Trader] buy call skipped: {_le}")
        # ── Notify ──────────────────────────────────────────────────────────
        try:
            # Extract source group name if this is a Telegram-triggered trade
            source_group = None
            if strategy == "Telegram" and reason:
                # reason format: "BUY RELIANCE from <GroupName> (score 0.72)"
                import re as _re
                m = _re.search(r'from (.+?) \(score', reason)
                if m:
                    source_group = m.group(1).strip()
            get_notifier().notify_buy(
                ticker=ticker, price=price,
                stop_loss=stop_loss, take_profit=target,
                planned_rr=planned_rr, strength=strength,
                strategy=strategy, reason=reason,
                source_group=source_group,
                qty=qty, capital_used=cost,
            )
        except Exception:
            pass   # never let notifier crash the trade engine
        return trade

    def execute_sell(self, ticker: str, price: float, reason: str = "") -> dict | None:
        pos = self.state["positions"].get(ticker)
        if not pos:
            return None

        # Enforce minimum hold period — never exit same day as entry (prevents signal flip-flop)
        if reason not in ("STOP_LOSS", "TAKE_PROFIT", "PENNY_STOCK_EXIT",
                          "HEDGE_LIQUIDATION", "DEAD_MONEY", "GAP_DOWN",
                          "PARTIAL_PROFIT", "ROTATION_EXIT"):
            try:
                entry_dt  = datetime.fromisoformat(pos["entry_date"])
                held_days = (_now_ist() - entry_dt).days
                if held_days < MIN_HOLD_DAYS:
                    logger.debug(f"SKIP SELL {ticker} — held only {held_days}d (min {MIN_HOLD_DAYS}d)")
                    return None
            except Exception:
                pass

        qty      = pos["qty"]
        proceeds = qty * price
        pnl      = (price - pos["avg_price"]) * qty
        self.state["cash"]         += proceeds
        self.state["realised_pnl"] += pnl
        strategy = pos["strategy"]

        # Compute actual RR: how many units of risk did we make/lose?
        risk_per_unit = pos.get("risk_per_unit") or (pos["avg_price"] * STOP_LOSS_PCT)
        actual_rr     = round((price - pos["avg_price"]) / risk_per_unit, 3) if risk_per_unit > 0 else 0.0
        planned_rr    = pos.get("planned_rr")

        # ── Trade type for Indian tax classification ───────────────────────
        # INTRADAY  : bought & sold same calendar day  → speculative business income
        # STCG      : held 1 day – 364 days            → Short-Term Capital Gains (20%)
        # LTCG      : held ≥ 365 days                  → Long-Term Capital Gains (12.5% above ₹1.25L)
        try:
            entry_dt  = datetime.fromisoformat(pos["entry_date"])
            exit_dt   = _now_ist()
            hold_days = (exit_dt.date() - entry_dt.date()).days
            if hold_days == 0:
                trade_type = "INTRADAY"
            elif hold_days < 365:
                trade_type = "STCG"
            else:
                trade_type = "LTCG"
        except Exception:
            hold_days  = None
            trade_type = "STCG"   # safe default

        del self.state["positions"][ticker]

        # Record exit time for cooldown — prevents immediate re-entry
        if "exit_log" not in self.state:
            self.state["exit_log"] = {}
        self.state["exit_log"][ticker] = _now_ist().isoformat()

        self._save()
        trade = self._log_trade(
            "SELL", ticker, qty, price, strategy, reason, pnl=pnl,
            actual_rr=actual_rr, planned_rr=planned_rr,
            trade_type=trade_type, hold_days=hold_days,
        )
        rr_str   = f"  RR={actual_rr:+.2f}(planned 1:{planned_rr})" if planned_rr else ""
        type_str = f"  [{trade_type}/{hold_days}d]"
        logger.info(f"SELL {ticker:20s} qty={qty} @ ₹{price:.2f}  pnl=₹{pnl:.2f}  [{reason}]{rr_str}{type_str}")
        # ── Live order execution (only when LIVE_TRADING=true) ───────────── #
        try:
            from angelone_trader import get_trader  # pylint: disable=import-outside-toplevel
            get_trader().sell(ticker, qty, price, segment="equity")
        except Exception as _le:
            logger.debug(f"[Trader] sell call skipped: {_le}")
        # ── Notify ──────────────────────────────────────────────────────────
        try:
            avg_price = pos["avg_price"]
            pnl_pct   = round((price / avg_price - 1) * 100, 2)
            get_notifier().notify_sell(
                ticker=ticker, price=price, avg_price=avg_price,
                pnl=pnl, pnl_pct=pnl_pct,
                actual_rr=actual_rr, planned_rr=planned_rr,
                strategy=strategy, hold_days=hold_days or 0,
                trade_type=trade_type or "STCG",
                reason=reason, qty=qty,
                strength=pos.get("strength", 65.0),
            )
        except Exception:
            pass   # never let notifier crash the trade engine
        return trade

    def check_stops(self) -> list:
        """
        Institutional-grade stop management.

        TRAILING STOP — Chandelier Exit (industry standard):
          Uses ATR so the stop width reflects each stock's actual volatility.
          stop = highest_close_since_entry - ATR_MULT × ATR(14)
          This prevents getting stopped out by normal volatility while still
          locking in profits as the position moves in our favour.
          Stop ONLY moves up — never down.

        PARTIAL PROFIT BOOKING:
          When price hits 50% of the way to target (T1), sell half the position
          and tighten the stop to breakeven on the remainder. This locks in
          cash while letting the winner run — standard institutional practice.

        DEAD-MONEY EXIT:
          If a position is flat (< 3% up or down) after 20 trading days,
          exit and redeploy the capital. Dead positions have opportunity cost.

        GAP-DOWN PROTECTION:
          If price gaps down more than 4% from the previous stop level in a
          single cycle, exit immediately regardless of stop level.
        """
        triggered = []
        state_dirty = False

        for ticker, pos in list(self.state["positions"].items()):
            price = DataFetcher.get_current_price(ticker)
            if price <= 0:
                continue

            # ── Penny stock exit: close any position priced under ₹50 ──── #
            # These slipped in before the price filter was added. Exit cleanly.
            if price < 50:
                logger.info(f"PENNY EXIT {ticker} @ ₹{price:.2f} — below ₹50 minimum")
                trade = self.execute_sell(ticker, price, reason="PENNY_STOCK_EXIT")
                if trade:
                    triggered.append(trade)
                continue

            entry    = pos["avg_price"]
            pnl_pct  = (price / entry - 1) * 100
            prev_sl  = pos["stop_loss"]

            # ── Chandelier Exit trailing stop ────────────────────────────── #
            # Use cached ATR stored on the position at entry time.
            # We only fetch fresh data if ATR is not cached — avoids per-position
            # API calls on every 15-minute cycle (institutional standard: compute
            # stop at entry, trail mechanically from the stored ATR value).
            try:
                atr = pos.get("atr_at_entry")
                highest_close = pos.get("highest_close", max(price, entry))

                # Update highest close in position state
                if price > highest_close:
                    pos["highest_close"] = price
                    highest_close = price
                    state_dirty = True

                # If no cached ATR (position opened before this code), do a one-time fetch
                if not atr:
                    df_recent = DataFetcher.fetch(ticker, period="30d", interval="1d")
                    if df_recent is not None and not df_recent.empty and len(df_recent) >= 15:
                        hi = df_recent["High"]; lo = df_recent["Low"]; cl = df_recent["Close"]
                        tr = pd.concat([hi-lo, (hi-cl.shift(1)).abs(), (lo-cl.shift(1)).abs()], axis=1).max(axis=1)
                        atr = float(tr.rolling(14).mean().iloc[-1])
                        pos["atr_at_entry"] = atr
                        state_dirty = True

                if atr and atr > 0:
                    # Chandelier multiplier tightens as profits grow
                    if pnl_pct >= 15: mult = 2.0
                    elif pnl_pct >= 7: mult = 2.5
                    else:              mult = 3.0
                    chandelier_sl = round(highest_close - mult * atr, 2)

                    # HARD FLOOR: initial -7% stop is the MINIMUM — chandelier
                    # only activates above this level. Downside is always capped
                    # at the entry stop so we never risk more than planned.
                    initial_sl = round(entry * (1 - STOP_LOSS_PCT), 2)
                    floor      = max(initial_sl, prev_sl)   # stop never moves down
                    new_sl     = max(chandelier_sl, floor)
                    new_sl     = round(new_sl, 2)
                else:
                    new_sl = prev_sl
            except Exception:
                new_sl = prev_sl

            if new_sl > prev_sl:
                pos["stop_loss"] = new_sl
                state_dirty = True
                logger.info(
                    f"📈 CHANDELIER-STOP {ticker}  "
                    f"SL ₹{prev_sl:.2f} → ₹{new_sl:.2f}  ({pnl_pct:+.1f}%)"
                )

            # ── Gap-down protection ──────────────────────────────────────── #
            gap_down_pct = (prev_sl - price) / prev_sl * 100 if prev_sl > 0 else 0
            if gap_down_pct > 4 and price < entry:
                logger.warning(
                    f"⚡ GAP-DOWN {ticker} @ ₹{price:.2f}  "
                    f"gapped {gap_down_pct:.1f}% below stop — exit immediately"
                )
                trade = self.execute_sell(ticker, price, reason="GAP_DOWN_PROTECTION")
                if trade:
                    triggered.append(trade)
                continue

            # ── Stop-loss trigger ────────────────────────────────────────── #
            if price <= pos["stop_loss"]:
                reason = "CHANDELIER_STOP" if pos["stop_loss"] > entry else "STOP_LOSS"
                logger.warning(
                    f"🛑 {reason} {ticker} @ ₹{price:.2f}  "
                    f"entry=₹{entry:.2f}  sl=₹{pos['stop_loss']:.2f}  {pnl_pct:+.1f}%"
                )
                trade = self.execute_sell(ticker, price, reason=reason)
                if trade:
                    triggered.append(trade)
                continue

            # ── Take-profit trigger ──────────────────────────────────────── #
            if price >= pos["target"]:
                logger.info(
                    f"🎯 TAKE-PROFIT {ticker} @ ₹{price:.2f}  "
                    f"entry=₹{entry:.2f}  gain={pnl_pct:.1f}%"
                )
                trade = self.execute_sell(ticker, price, reason="TAKE_PROFIT")
                if trade:
                    triggered.append(trade)
                continue

            # ── Partial profit booking at T1 (50% of way to target) ──────── #
            t1 = entry + (pos["target"] - entry) * 0.50
            if price >= t1 and not pos.get("t1_booked"):
                half_qty = pos["qty"] // 2
                if half_qty >= 1:
                    logger.info(
                        f"💰 PARTIAL-PROFIT {ticker} @ ₹{price:.2f}  "
                        f"selling {half_qty}/{pos['qty']} shares at T1 ({pnl_pct:+.1f}%)"
                    )
                    # Execute partial sell
                    pnl = (price - entry) * half_qty
                    self.state["cash"]         += price * half_qty
                    self.state["realised_pnl"] += pnl
                    pos["qty"]      -= half_qty
                    pos["t1_booked"] = True
                    # Tighten stop to breakeven on remainder
                    breakeven_sl = round(entry * 1.005, 2)
                    if breakeven_sl > pos["stop_loss"]:
                        pos["stop_loss"] = breakeven_sl
                        logger.info(
                            f"   ↳ Stop tightened to breakeven ₹{breakeven_sl:.2f} on remainder"
                        )
                    self._log_trade(
                        "SELL", ticker, half_qty, price,
                        pos.get("strategy", ""), "PARTIAL_PROFIT_T1",
                        pnl=pnl,
                    )
                    state_dirty = True

            # ── Dead-money exit: flat for 20+ days ──────────────────────── #
            try:
                entry_dt = pos.get("entry_date", "")
                if entry_dt:
                    entry_ts  = _now_ist().fromisoformat(entry_dt)
                    days_held = (_now_ist() - entry_ts).days
                    if days_held >= 20 and abs(pnl_pct) < 3.0:
                        logger.info(
                            f"⌛ DEAD-MONEY EXIT {ticker}  "
                            f"held {days_held}d  only {pnl_pct:+.1f}% — redeploying capital"
                        )
                        trade = self.execute_sell(ticker, price, reason="DEAD_MONEY_EXIT")
                        if trade:
                            triggered.append(trade)
                        continue
            except Exception:
                pass

            logger.debug(
                f"📊 HOLD {ticker} @ ₹{price:.2f}  "
                f"SL=₹{pos['stop_loss']:.2f}  TP=₹{pos['target']:.2f}  {pnl_pct:+.1f}%"
            )

        if state_dirty:
            self._save()

        return triggered

    def _log_trade(self, action, ticker, qty, price, strategy, reason, pnl=None,
                   stop_loss=None, target=None, planned_rr=None,
                   actual_rr=None, risk_per_unit=None,
                   trade_type=None, hold_days=None) -> dict:
        log = []
        if TRADE_LOG_FILE.exists():
            with open(TRADE_LOG_FILE) as f:
                log = json.load(f)
        trade = {
            "id":            len(log) + 1,
            "action":        action,
            "ticker":        ticker,
            "qty":           qty,
            "price":         round(price, 2),
            "value":         round(qty * price, 2),
            "strategy":      strategy,
            "reason":        reason,
            "pnl":           round(pnl, 2) if pnl is not None else None,
            "time":          _now_ist().isoformat(),
            # ── RR metadata ──────────────────────────────────────────────
            # BUY  → stop_loss, target, planned_rr, risk_per_unit
            # SELL → actual_rr, trade_type, hold_days
            "stop_loss":     stop_loss,
            "target":        target,
            "planned_rr":    planned_rr,
            "actual_rr":     actual_rr,
            "risk_per_unit": risk_per_unit,
            # ── Tax classification ────────────────────────────────────────
            # INTRADAY = same-day (speculative, taxed at slab rate)
            # STCG     = 1–364 days (20% tax)
            # LTCG     = 365+ days (12.5% above ₹1.25L exemption)
            "trade_type":    trade_type,   # populated on SELL/exit
            "hold_days":     hold_days,    # populated on SELL/exit
        }
        log.append(trade)
        with open(TRADE_LOG_FILE, "w") as f:
            json.dump(log, f, indent=2)
        return trade

    def get_positions_display(self) -> list:
        result = []
        for ticker, pos in self.state["positions"].items():
            price    = DataFetcher.get_current_price(ticker)
            pnl      = (price - pos["avg_price"]) * pos["qty"]
            pnl_pct  = (price / pos["avg_price"] - 1) * 100 if pos["avg_price"] else 0
            result.append({
                "ticker":     ticker,
                "qty":        pos["qty"],
                "avg_price":  round(pos["avg_price"], 2),
                "ltp":        round(price, 2),
                "value":      round(price * pos["qty"], 2),
                "pnl":        round(pnl, 2),
                "pnl_pct":    round(pnl_pct, 2),
                "strategy":   pos["strategy"],
                "stop_loss":  pos.get("stop_loss"),
                "target":     pos.get("target"),
                "entry_date": pos.get("entry_date", ""),
            })
        result.sort(key=lambda x: x["pnl_pct"], reverse=True)
        return result

    def reset(self):
        state = self._default_state()
        self._save(state)
        if TRADE_LOG_FILE.exists():
            TRADE_LOG_FILE.write_text("[]")
        logger.info("Portfolio reset to ₹10,00,000")


# ---------------------------------------------------------------------------
# Strategy 4g: Market Structure (HH / HL / LH / LL)
# ---------------------------------------------------------------------------

class MarketStructureStrategy:
    """
    Detects pivot-based market structure from the analyst's Pine Script logic:
      - Higher High (HH) + Higher Low (HL) → Bullish trend → BUY
      - Lower Low (LL)  + Lower High (LH)  → Bearish trend → SELL
      - Mixed / neutral                     → no signal

    Pivot is the local extreme over ±pivot_length bars.
    Looks at the last 4 confirmed pivots (2 highs + 2 lows) to judge
    whether price structure is improving or deteriorating.

    Signal strength scales with how many consecutive confirmations exist:
      1 confirmation  → 0.55
      2 confirmations → 0.65
      3+ confirmations → 0.75
    """

    name       = "Market Structure"
    short_name = "MKT_STRUCT"

    PIVOT_LENGTH   = 5   # bars left & right (same default as Pine Script)
    MIN_PIVOTS     = 4   # need at least 2 highs + 2 lows to judge structure

    def _find_pivots(self, series: pd.Series, kind: str) -> list[tuple]:
        """
        Return list of (bar_index, price) for confirmed pivot highs or lows.
        A pivot high: series[i] == max over [i-L .. i+L]
        A pivot low:  series[i] == min over [i-L .. i+L]
        Only returns bars where both sides have at least PIVOT_LENGTH bars.
        """
        L = self.PIVOT_LENGTH
        pivots = []
        vals = series.values
        for i in range(L, len(vals) - L):
            window = vals[i - L: i + L + 1]
            if kind == "high" and vals[i] == max(window):
                pivots.append((i, float(vals[i])))
            elif kind == "low" and vals[i] == min(window):
                pivots.append((i, float(vals[i])))
        return pivots

    def _structure_label(self, pivots_high: list, pivots_low: list) -> str:
        """
        Determine market structure from the last 2 confirmed highs and 2 lows.
        Returns: "Bullish", "Bearish", "Consolidation", or "Neutral".
        """
        if len(pivots_high) < 2 or len(pivots_low) < 2:
            return "Neutral"
        last_hh, prev_hh = pivots_high[-1][1], pivots_high[-2][1]
        last_ll, prev_ll = pivots_low[-1][1],  pivots_low[-2][1]
        hh_up = last_hh > prev_hh   # Higher High
        hl_up = last_ll > prev_ll   # Higher Low
        ll_dn = last_ll < prev_ll   # Lower Low
        lh_dn = last_hh < prev_hh   # Lower High
        if hh_up and hl_up:
            return "Bullish"
        if ll_dn and lh_dn:
            return "Bearish"
        return "Consolidation"

    def _count_consecutive(self, pivots: list, kind: str) -> int:
        """
        Count how many consecutive pivot pairs confirm the direction.
        kind: "up" (each pivot > prev) or "down" (each pivot < prev)
        """
        count = 0
        for i in range(len(pivots) - 1, 0, -1):
            cur = pivots[i][1]
            prev = pivots[i - 1][1]
            if kind == "up"   and cur > prev: count += 1
            elif kind == "down" and cur < prev: count += 1
            else: break
        return count

    def generate_signals(self, stock_data: dict) -> list:
        signals = []
        now_str = _now_ist().isoformat()

        for ticker, df in stock_data.items():
            if df is None or len(df) < self.PIVOT_LENGTH * 2 + 10:
                continue
            try:
                ph = self._find_pivots(df["High"], "high")
                pl = self._find_pivots(df["Low"],  "low")

                if len(ph) < 2 or len(pl) < 2:
                    continue

                structure = self._structure_label(ph, pl)
                if structure not in ("Bullish", "Bearish"):
                    continue

                # Count consecutive confirmations for strength scaling
                if structure == "Bullish":
                    hh_count = self._count_consecutive(ph, "up")
                    hl_count = self._count_consecutive(pl, "up")
                    consec   = min(hh_count, hl_count)
                    action   = "BUY"
                    reason   = (
                        f"HH={ph[-1][1]:.2f}>{ph[-2][1]:.2f}, "
                        f"HL={pl[-1][1]:.2f}>{pl[-2][1]:.2f} "
                        f"({consec}× confirmed)"
                    )
                else:  # Bearish
                    lh_count = self._count_consecutive(ph, "down")
                    ll_count = self._count_consecutive(pl, "down")
                    consec   = min(lh_count, ll_count)
                    action   = "SELL"
                    reason   = (
                        f"LH={ph[-1][1]:.2f}<{ph[-2][1]:.2f}, "
                        f"LL={pl[-1][1]:.2f}<{pl[-2][1]:.2f} "
                        f"({consec}× confirmed)"
                    )

                strength = min(0.55 + consec * 0.10, 0.80)

                current_price = float(df["Close"].iloc[-1])
                is_penny = ticker in set(PENNY_UNIVERSE)

                signals.append({
                    "ticker":   ticker,
                    "action":   action,
                    "strength": round(strength, 3),
                    "strategy": self.name,
                    "reason":   reason,
                    "price":    round(current_price, 2),
                    "time":     now_str,
                    "is_penny": is_penny,
                })

            except Exception as e:
                logger.debug(f"MarketStructure error {ticker}: {e}")

        logger.info(f"MarketStructure: {len(signals)} signals")
        return signals


# ---------------------------------------------------------------------------
# Strategy 5: Telegram Signal Intelligence
# ---------------------------------------------------------------------------

class QuantAnalysisStrategy:
    """
    Quantitative analysis strategy using statistical methods:

    1. Z-Score mean reversion  — finds stocks statistically cheap/expensive
       relative to their own 60-day distribution
    2. Hurst Exponent          — detects whether each stock is trending (H>0.5)
       or mean-reverting (H<0.5) and picks the right trade style
    3. Volatility-adjusted momentum — Sharpe-like ranking using return / vol
    4. Statistical support/resistance — entry near lower Bollinger + 2σ support,
       target at mean, SL at 3σ below current

    Entry, stop-loss and target are all computed from actual price statistics,
    not fixed percentages.
    """
    name       = "Quant Analysis"
    short_name = "QUANT"

    LOOKBACK   = 60    # days for statistical window
    Z_BUY      = -1.5  # Z-score threshold to signal statistically cheap
    Z_SELL     =  1.8  # Z-score threshold to signal statistically expensive

    @staticmethod
    def _hurst(close: pd.Series, lags: int = 20) -> float:
        """
        Compute Hurst exponent using R/S analysis on log-returns.
        H > 0.55 → trending  |  H < 0.45 → mean-reverting  |  ~0.5 → random walk
        """
        try:
            log_ret = np.log(close / close.shift(1)).dropna().values
            if len(log_ret) < lags * 2:
                return 0.5
            rs_vals = []
            lag_vals = range(2, lags)
            for lag in lag_vals:
                chunks = [log_ret[i:i+lag] for i in range(0, len(log_ret)-lag, lag)]
                rs_chunk = []
                for chunk in chunks:
                    if len(chunk) < 2:
                        continue
                    m = np.mean(chunk)
                    dev = np.cumsum(chunk - m)
                    r = dev.max() - dev.min()
                    s = np.std(chunk, ddof=1)
                    if s > 0:
                        rs_chunk.append(r / s)
                if rs_chunk:
                    rs_vals.append(np.mean(rs_chunk))
            if len(rs_vals) < 4:
                return 0.5
            log_rs  = np.log(rs_vals)
            log_lag = np.log(list(range(2, lags)))[:len(log_rs)]
            h = float(np.polyfit(log_lag, log_rs, 1)[0])
            return max(0.0, min(1.0, h))
        except Exception:
            return 0.5

    def generate_signals(self, data: dict) -> list:
        signals = []

        for ticker, df in data.items():
            if df is None or df.empty or len(df) < self.LOOKBACK + 5:
                continue

            close  = df["Close"].dropna()
            window = close.iloc[-self.LOOKBACK:]

            # ── Z-score of current price vs 60-day window ─────────────────── #
            mean  = float(window.mean())
            std   = float(window.std())
            if std == 0:
                continue
            price   = float(close.iloc[-1])
            z_score = (price - mean) / std

            # ── Hurst exponent ────────────────────────────────────────────── #
            hurst = self._hurst(close.iloc[-120:] if len(close) >= 120 else close)

            # ── Volatility-adjusted momentum (Sharpe-like) ────────────────── #
            ret_20d = float(close.pct_change(20).iloc[-1])
            vol_20d = float(close.pct_change().iloc[-20:].std()) if len(close) >= 20 else 0.01
            sharpe  = ret_20d / vol_20d if vol_20d > 0 else 0

            # ── Determine signal ──────────────────────────────────────────── #
            signal = None
            strength = 0
            reason_parts = []

            if z_score <= self.Z_BUY:
                # Statistically cheap — prefer mean-reverting stocks
                if hurst < 0.55:  # mean-reverting or neutral
                    signal   = "BUY"
                    strength = min(100, int(65 + abs(z_score) * 10))
                    reason_parts.append(f"Z={z_score:.2f} (statistically cheap)")
                    reason_parts.append(f"Hurst={hurst:.2f} (mean-reverting)")
                elif hurst >= 0.55 and ret_20d > 0:
                    # Trending stock dipping — valid pullback buy
                    signal   = "BUY"
                    strength = min(100, int(60 + abs(z_score) * 8))
                    reason_parts.append(f"Z={z_score:.2f} pullback in uptrend")
                    reason_parts.append(f"Hurst={hurst:.2f} (trending)")

            elif z_score >= self.Z_SELL:
                signal   = "SELL"
                strength = min(100, int(60 + z_score * 8))
                reason_parts.append(f"Z={z_score:.2f} (statistically expensive)")
                reason_parts.append(f"Hurst={hurst:.2f}")

            if signal is None:
                continue

            # ── Statistical entry / SL / target ──────────────────────────── #
            if signal == "BUY":
                # Entry: current price (already at statistical low)
                entry  = round(price, 2)
                # SL: 2σ below current (beyond statistical range)
                sl     = round(max(price - 2.0 * std, price * 0.85), 2)
                # Target: mean reversion back to μ, then half-σ above
                target = round(mean + 0.5 * std, 2)
                if target <= entry:
                    target = round(entry * 1.10, 2)  # fallback: 10% above entry
            else:
                entry  = round(price, 2)
                sl     = round(price + 2.0 * std, 2)
                target = round(mean - 0.5 * std, 2)
                if target >= entry:
                    target = round(entry * 0.90, 2)

            if sharpe > 0.5:
                reason_parts.append(f"Sharpe={sharpe:.2f} ✅")

            signals.append({
                "ticker":       ticker,
                "signal":       signal,
                "price":        entry,
                "score":        round(z_score, 3),
                "rsi":          50.0,
                "strategy":     self.short_name,
                "strength":     strength,
                "entry_price":  entry,
                "stop_loss":    sl,
                "target":       target,
                "z_score":      round(z_score, 3),
                "hurst":        round(hurst, 3),
                "sharpe_20d":   round(sharpe, 3),
                "reason":       " | ".join(reason_parts),
            })

        # Return top 20 by abs(z_score) — most statistically extreme
        signals.sort(key=lambda x: abs(x["score"]), reverse=True)
        return signals[:20]


class SectorMomentumStrategy:
    """
    Ranks all NSE sectors on 1-day, 5-day and 20-day returns, then drills into
    the top-performing sectors and generates BUY signals for the best stocks
    within those sectors.

    Logic:
      1. Compute weighted sector score: 50% × 1d + 30% × 5d + 20% × 20d
      2. Top 3 sectors are "hot" — generate BUY signals for stocks inside them
      3. Bottom 2 sectors are "cold" — generate SELL signals for any held stocks
      4. Signal strength reflects how much the sector outperforms Nifty
    """
    name       = "Sector Momentum"
    short_name = "SECTOR_MOM"

    # Sector → constituent tickers (broad NSE universe)
    SECTOR_STOCKS: dict[str, list[str]] = {
        "Banking":      ["HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","AXISBANK.NS",
                         "KOTAKBANK.NS","INDUSINDBK.NS","BANDHANBNK.NS","FEDERALBNK.NS",
                         "IDFCFIRSTB.NS","YESBANK.NS"],
        "IT":           ["TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS",
                         "LTIM.NS","MPHASIS.NS","PERSISTENT.NS","COFORGE.NS"],
        "Pharma":       ["SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS",
                         "APOLLOHOSP.NS","LUPIN.NS","BIOCON.NS","LAURUSLABS.NS"],
        "Auto":         ["MARUTI.NS","TATAMOTORS.NS","M&M.NS","BAJAJ-AUTO.NS",
                         "HEROMOTOCO.NS","EICHERMOT.NS","MOTHERSON.NS","BHARATFORG.NS"],
        "FMCG":         ["ITC.NS","HINDUNILVR.NS","NESTLEIND.NS","BRITANNIA.NS",
                         "DABUR.NS","GODREJCP.NS","MARICO.NS","COLPAL.NS"],
        "Metal":        ["TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","VEDL.NS",
                         "SAIL.NS","NATIONALUM.NS","HINDZINC.NS","NMDC.NS"],
        "Energy":       ["RELIANCE.NS","ONGC.NS","BPCL.NS","IOC.NS",
                         "GAIL.NS","NTPC.NS","POWERGRID.NS","TATAPOWER.NS"],
        "CapGoods":     ["LT.NS","ABB.NS","SIEMENS.NS","BHEL.NS",
                         "HAVELLS.NS","CUMMINSIND.NS","THERMAX.NS","BEL.NS"],
        "Realty":       ["DLF.NS","GODREJPROP.NS","PRESTIGE.NS","OBEROIRLTY.NS",
                         "PHOENIXLTD.NS","SOBHA.NS"],
        "Consumer":     ["TITAN.NS","VOLTAS.NS","CROMPTON.NS","HAVELLS.NS",
                         "WHIRLPOOL.NS","VGUARD.NS"],
    }

    def generate_signals(self, all_data: dict, index_df: pd.DataFrame) -> list:
        signals = []

        # ── Step 1: Score each sector ─────────────────────────────────────── #
        sector_scores: list[dict] = []
        nifty_ret = {}
        for days, label in [(1, "1d"), (5, "5d"), (20, "20d")]:
            if len(index_df) > days:
                nifty_ret[label] = float(index_df["Close"].pct_change(days).iloc[-1])
            else:
                nifty_ret[label] = 0.0

        for sector, tickers in self.SECTOR_STOCKS.items():
            rets: dict[str, list[float]] = {"1d": [], "5d": [], "20d": []}
            for tk in tickers:
                df = all_data.get(tk)
                if df is None or df.empty or len(df) < 22:
                    continue
                close = df["Close"].dropna()
                for days, label in [(1, "1d"), (5, "5d"), (20, "20d")]:
                    if len(close) > days:
                        rets[label].append(float(close.pct_change(days).iloc[-1]))

            if not rets["1d"]:
                continue

            avg = {k: float(np.mean(v)) for k, v in rets.items() if v}
            # Relative to Nifty
            rs_1d  = avg.get("1d",  0) - nifty_ret["1d"]
            rs_5d  = avg.get("5d",  0) - nifty_ret["5d"]
            rs_20d = avg.get("20d", 0) - nifty_ret["20d"]

            # Weighted composite: recent momentum weighted most
            composite = 0.50 * rs_1d + 0.30 * rs_5d + 0.20 * rs_20d

            sector_scores.append({
                "sector":    sector,
                "tickers":   tickers,
                "composite": composite,
                "rs_1d":     round(rs_1d * 100, 2),
                "rs_5d":     round(rs_5d * 100, 2),
                "rs_20d":    round(rs_20d * 100, 2),
                "avg_1d":    round(avg.get("1d", 0) * 100, 2),
            })

        if not sector_scores:
            return signals

        sector_scores.sort(key=lambda x: x["composite"], reverse=True)
        n = len(sector_scores)
        hot_sectors  = sector_scores[:3]    # top 3
        cold_sectors = sector_scores[-2:]   # bottom 2

        logger.info(
            f"[SectorMom] Top: {[s['sector'] for s in hot_sectors]}  "
            f"Weak: {[s['sector'] for s in cold_sectors]}"
        )

        # ── Step 2: BUY signals — best stocks in hot sectors ──────────────── #
        for rank, sec in enumerate(hot_sectors):
            sector_strength = min(100, max(50, int(70 + sec["composite"] * 500)))
            stock_signals = []

            for tk in sec["tickers"]:
                df = all_data.get(tk)
                if df is None or df.empty or len(df) < 22:
                    continue
                df = add_indicators(df)
                if df.empty or "rsi" not in df.columns:
                    continue

                row   = df.iloc[-1]
                close = float(row["Close"])
                rsi   = float(row.get("rsi", 50))
                ema20 = float(row.get("ema_20", close))
                ema50 = float(row.get("ema_50", close))

                # Must be in uptrend and not overbought
                if close < ema20:
                    continue
                if rsi > 75:
                    continue

                # Stock-level score: reward momentum + room to run
                stock_score = 0
                if close > ema20: stock_score += 20
                if close > ema50: stock_score += 20
                if 40 < rsi < 65: stock_score += 30    # sweet spot
                elif 30 < rsi <= 40: stock_score += 20  # oversold bounce
                ret_1d = float(df["Close"].pct_change(1).iloc[-1])
                if ret_1d > 0: stock_score += 15
                ret_5d = float(df["Close"].pct_change(5).iloc[-1])
                if ret_5d > 0.02: stock_score += 15

                strength = min(100, int(sector_strength * 0.6 + stock_score * 0.4))
                if strength < 55:
                    continue

                stock_signals.append({
                    "ticker":   tk,
                    "signal":   "BUY",
                    "price":    round(close, 2),
                    "score":    round(sec["composite"], 4),
                    "rsi":      round(rsi, 1),
                    "strategy": self.short_name,
                    "strength": strength,
                    "reason":   (
                        f"{sec['sector']} sector +{sec['rs_1d']:.1f}% vs Nifty today "
                        f"(rank #{rank+1}) | stock RSI {rsi:.0f}"
                    ),
                })

            # Keep top 3 stocks per sector by strength
            stock_signals.sort(key=lambda x: x["strength"], reverse=True)
            signals += stock_signals[:3]

        # ── Step 3: SELL signals — stocks held in cold sectors ────────────── #
        cold_tickers = {tk for s in cold_sectors for tk in s["tickers"]}
        for sec in cold_sectors:
            for tk in sec["tickers"]:
                df = all_data.get(tk)
                if df is None or df.empty:
                    continue
                close = float(df["Close"].iloc[-1])
                signals.append({
                    "ticker":   tk,
                    "signal":   "SELL",
                    "price":    round(close, 2),
                    "score":    round(sec["composite"], 4),
                    "rsi":      50.0,
                    "strategy": self.short_name,
                    "strength": max(30, int(50 - abs(sec["composite"]) * 300)),
                    "reason":   (
                        f"{sec['sector']} sector underperforming Nifty "
                        f"({sec['rs_1d']:+.1f}% today)"
                    ),
                })

        return signals


class TelegramSignalStrategy:
    """
    Converts high-scoring Telegram group signals into trading signals.
    Only uses groups that have:
      - status == "active"
      - signals_evaluated >= MIN_EVALUATED_FOR_SCORE (10)
      - score >= MIN_GROUP_SCORE (0.55)
    Signal strength is proportional to the group's composite score.
    """
    name       = "Telegram Signals"
    short_name = "TELEGRAM"

    MIN_GROUP_SCORE   = 0.55
    MIN_EVALUATED     = 10
    MAX_SIGNAL_AGE_H  = 72   # only use signals received in the last 72 hours

    def generate_signals(self) -> list:
        try:
            from telegram_agent import get_telegram_agent  # pylint: disable=import-outside-toplevel
            agent = get_telegram_agent()
            if not agent.is_configured():
                return []

            # Build map of high-quality active groups
            good_groups = {
                str(g["id"]): g
                for g in agent.get_groups()
                if g.get("status") == "active"
                and g.get("signals_evaluated", 0) >= self.MIN_EVALUATED
                and g.get("score", 0) >= self.MIN_GROUP_SCORE
            }
            if not good_groups:
                return []

            # Recent pending signals from good groups only
            cutoff = (_now_ist() - timedelta(hours=self.MAX_SIGNAL_AGE_H)).isoformat()
            recent = [
                s for s in agent.get_signals(limit=1000)
                if s.get("status") == "pending"
                and s.get("received_at", "") >= cutoff
                and str(s.get("group_id", "")) in good_groups
            ]

            signals   = []
            seen      = set()
            for sig in recent:
                parsed = sig.get("parsed", {})
                ticker = parsed.get("ticker")
                if not ticker or ticker in seen:
                    continue
                group    = good_groups.get(str(sig["group_id"]), {})
                score    = group.get("score", 0.5)
                strength = min(100, int(score * 110))   # scale 0-1 → ~0-100

                signals.append({
                    "ticker":      ticker,
                    "signal":      parsed["direction"],
                    "price":       DataFetcher.get_current_price(ticker),
                    "score":       round(score, 4),
                    "rsi":         50.0,
                    "strategy":    self.short_name,
                    "strength":    strength,
                    "source":      sig.get("group_title", ""),
                    "specificity": parsed.get("specificity", 0.3),
                })
                seen.add(ticker)

            logger.info(f"TelegramSignalStrategy: {len(signals)} signals from {len(good_groups)} groups")
            return signals

        except Exception as e:
            logger.warning(f"TelegramSignalStrategy error: {e}")
            return []


# ---------------------------------------------------------------------------
# Signal Aggregator
# ---------------------------------------------------------------------------

class SignalAggregator:

    def __init__(self):
        self.momentum    = MomentumStrategy()
        self.mean_rev    = MeanReversionStrategy()
        self.multifactor = MultiFactorStrategy()
        self.sector_rot  = SectorRotationStrategy()
        self.sector_mom  = SectorMomentumStrategy()
        self.quant       = QuantAnalysisStrategy()
        self.sma         = SMAStrategy()
        self.fibonacci   = FibonacciStrategy()
        self.rsi_div     = RSIDivergenceStrategy()
        self.bb_squeeze  = BollingerSqueezeStrategy()
        self.vol_break   = VolumeBreakoutStrategy()
        self.mkt_struct  = MarketStructureStrategy()
        self.telegram    = TelegramSignalStrategy()

    def run(self) -> list:
        # ── Build full universe: NSE equity list (refreshed daily) + PENNY_UNIVERSE ──
        logger.info("Loading stock universe…")
        full_universe = load_nse_universe()   # all NSE EQ-series stocks
        # Merge penny universe; deduplicate
        all_tickers = list(dict.fromkeys(full_universe + PENNY_UNIVERSE))
        logger.info(f"Universe size: {len(all_tickers)} stocks (NSE + curated)")

        logger.info("Fetching market data (batch download)…")
        # Batch fetch entire universe — yfinance does this in a handful of HTTP calls
        all_data    = DataFetcher.fetch_multi(all_tickers, period="2y", batch_size=200)
        # Sector ETFs + index fetched separately (small list, already cached)
        etf_data    = DataFetcher.fetch_multi(list(SECTOR_ETFS.values()), period="2y")
        index_df    = DataFetcher.fetch("^NSEI", period="2y")
        logger.info(f"Data fetched for {len(all_data)} stocks")

        all_signals = []
        # ── All strategies run on the full universe ──
        all_signals += self.momentum.generate_signals(all_data)
        all_signals += self.mean_rev.generate_signals(all_data)
        all_signals += self.multifactor.generate_signals(all_data)
        all_signals += self.sector_rot.generate_signals(etf_data, index_df)
        all_signals += self.sector_mom.generate_signals(all_data, index_df)
        all_signals += self.quant.generate_signals(all_data)
        all_signals += self.sma.generate_signals(all_data)
        all_signals += self.fibonacci.generate_signals(all_data)
        all_signals += self.rsi_div.generate_signals(all_data)
        all_signals += self.bb_squeeze.generate_signals(all_data)
        all_signals += self.vol_break.generate_signals(all_data)
        all_signals += self.mkt_struct.generate_signals(all_data)
        # ── Intelligence layers ──
        all_signals += self.telegram.generate_signals()
        # ── News + commodity signals (imported lazily to avoid circular deps) ──
        try:
            from news_agent import get_news_agent  # pylint: disable=import-outside-toplevel
            all_signals += get_news_agent().get_all_signals()
        except Exception as e:
            logger.warning(f"News agent signals skipped: {e}")
        # ── Fundamental filter: re-score all signals ──
        try:
            from fundamental_analyzer import get_analyzer  # pylint: disable=import-outside-toplevel
            all_signals = get_analyzer().rescore_signals(all_signals)
        except Exception as e:
            logger.warning(f"Fundamental rescoring skipped: {e}")

        # ── Institutional signal filters ──────────────────────────────────────
        # Drop signals that no institution would touch:
        #   • Price < ₹20 (penny/illiquid)
        #   • Average daily value < ₹2 Cr (not enough liquidity to enter/exit cleanly)
        #   • RSI > 80 on a BUY (chasing overbought) or RSI < 20 on a SELL
        #   • Fundamental grade F (avoid complete junk)
        def _is_tradeable(sig: dict) -> bool:
            price = sig.get("price", 0)
            if price < 20:
                return False
            rsi = sig.get("rsi", 50)
            if sig.get("signal") == "BUY" and rsi > 82:
                return False
            if sig.get("signal") == "SELL" and rsi < 18:
                return False
            if sig.get("fund_grade") == "F":
                return False
            # Volume/liquidity check from already-fetched data
            ticker = sig.get("ticker", "")
            df = all_data.get(ticker)
            if df is not None and not df.empty and "Volume" in df.columns:
                avg_vol = float(df["Volume"].iloc[-20:].mean()) if len(df) >= 20 else float(df["Volume"].mean())
                avg_val = avg_vol * price
                if avg_val < 2_00_00_000:   # < ₹2 Cr daily turnover
                    return False
            return True

        all_signals = [s for s in all_signals if _is_tradeable(s)]
        logger.info(f"After institutional filters: {len(all_signals)} signals remain")

        # ── ATR-based SL/TP enrichment using already-downloaded data ─────────
        # We already have 2 years of OHLCV for every stock in all_data.
        # Use that to compute proper ATR(14) — zero extra API calls.
        ATR_PERIOD = 14
        for sig in all_signals:
            if sig.get("entry_price"):
                continue   # already has levels (e.g. from QUANT or Telegram parser)
            price = sig.get("price", 0)
            if not price:
                continue
            strength = sig.get("strength", 65)
            if strength >= 90:   rr = 2.5
            elif strength >= 80: rr = 2.0
            else:                rr = 1.5

            # Try ATR from in-memory data first
            atr_sl_mult = 1.5
            sl_dist = None
            ticker = sig.get("ticker", "")
            df = all_data.get(ticker)
            if df is not None and not df.empty and len(df) >= ATR_PERIOD + 1:
                try:
                    hi = df["High"]; lo = df["Low"]; cl = df["Close"]
                    tr = pd.concat([
                        hi - lo,
                        (hi - cl.shift(1)).abs(),
                        (lo - cl.shift(1)).abs(),
                    ], axis=1).max(axis=1)
                    atr = float(tr.rolling(ATR_PERIOD).mean().iloc[-1])
                    if pd.notna(atr) and 0 < atr < price * 0.15:
                        sl_dist = atr * atr_sl_mult
                except Exception:
                    pass

            # Flat fallback only if ATR computation failed
            if sl_dist is None:
                sl_dist = price * STOP_LOSS_PCT

            if sig.get("signal") == "BUY":
                sl = round(max(price - sl_dist, price * 0.50), 2)
                tp = round(price + sl_dist * rr, 2)
            else:
                sl = round(price + sl_dist, 2)
                tp = round(price - sl_dist * rr, 2)

            sig["entry_price"] = round(price, 2)
            sig["stop_loss"]   = sl
            sig["target"]      = tp
            sig["planned_rr"]  = rr

        # Save to file
        with open(SIGNALS_FILE, "w") as f:
            json.dump({
                "signals":    all_signals,
                "updated_at": _now_ist().isoformat(),
            }, f, indent=2)
        logger.info(f"Generated {len(all_signals)} signals")
        return all_signals, all_data


# ---------------------------------------------------------------------------
# Agent Orchestrator
# ---------------------------------------------------------------------------

class TradingAgent:
    """
    Orchestrates data fetch → signal generation → order execution → stop checks.
    Runs each cycle (called from the dashboard or scheduler).
    """

    def __init__(self):
        self.portfolio  = Portfolio()
        self.aggregator = SignalAggregator()

    def run_cycle(self) -> dict:
        """Full agent cycle: generate signals + execute paper trades."""
        logger.info("=== Agent Cycle Start ===")
        t0 = time.time()

        # 1. Check stop-loss / take-profit on open positions first
        stops = self.portfolio.check_stops()

        # 2. Generate fresh signals (also returns the already-fetched price data)
        signals, all_data = self.aggregator.run()

        # ── Aggregate signals per ticker ───────────────────────────────────
        # Multiple strategies may fire for the same ticker. We combine them:
        #   - composite_strength = average of all signals for that ticker
        #   - strategy_count     = how many strategies agree
        # This prevents trading on a single weak strategy firing alone.

        buy_agg:  dict[str, dict] = {}   # ticker → aggregated buy info
        sell_agg: dict[str, dict] = {}   # ticker → aggregated sell info

        for sig in signals:
            ticker   = sig["ticker"]
            action   = sig.get("signal", sig.get("action", ""))
            strength = sig.get("strength", 0)
            strategy = sig.get("strategy", "")
            price    = sig.get("price", 0)

            if action == "BUY":
                if ticker not in buy_agg:
                    buy_agg[ticker] = {"strengths": [], "strategies": [], "price": price}
                buy_agg[ticker]["strengths"].append(strength)
                buy_agg[ticker]["strategies"].append(strategy)
                if price > 0:
                    buy_agg[ticker]["price"] = price   # use latest non-zero price

            elif action == "SELL":
                if ticker not in sell_agg:
                    sell_agg[ticker] = {"strengths": [], "strategies": [], "price": price}
                sell_agg[ticker]["strengths"].append(strength)
                sell_agg[ticker]["strategies"].append(strategy)
                if price > 0:
                    sell_agg[ticker]["price"] = price

        # ── Risk check before any new buys ────────────────────────────────────
        rm = get_risk_manager()
        total_val = self.portfolio.get_total_value()
        rm.update_peak(total_val)   # keep drawdown guard fresh

        macro_score, macro_label, macro_breakdown = rm.macro_risk()
        dd_level, dd_pct = rm.drawdown_status(total_val)
        max_new_buys = rm.max_new_buys_this_cycle(total_val)

        logger.info(
            f"📊 Risk snapshot: macro={macro_label}({macro_score:.2f}) "
            f"drawdown={dd_level}({dd_pct*100:.1f}%) max_new_buys={max_new_buys}"
        )
        if macro_breakdown.get("events_soon"):
            for ev in macro_breakdown["events_soon"]:
                logger.warning(
                    f"⚠️  UPCOMING EVENT in {ev['days_away']}d: {ev['label']} ({ev['date']})"
                )

        # ── Execute BUY signals ────────────────────────────────────────────
        # Sort by composite strength descending (strongest conviction first)
        learning = get_learning_engine()
        buy_threshold = learning.get_threshold()

        buy_candidates = []
        for ticker, agg in buy_agg.items():
            # Use learning-weighted composite strength (accounts for strategy win rates)
            boosted = learning.weighted_strength(agg["strengths"], agg["strategies"])
            buy_candidates.append((ticker, boosted, agg["price"], agg["strategies"]))

        buy_candidates.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"📋 {len(buy_candidates)} BUY candidates after aggregation "
                    f"({len([c for c in buy_candidates if c[1] >= buy_threshold])} above threshold={buy_threshold})")

        executed = []
        new_buys_this_cycle = 0

        # ── Time-of-day filter: institutional execution windows ────────────
        now_ist  = _now_ist()
        now_time = now_ist.time()
        # Avoid first 15 min (wide spreads, stop-hunt candles) and last 30 min
        _EXEC_OPEN  = dt_time(9, 30)
        _EXEC_CLOSE = dt_time(15, 0)
        execution_window_open = _EXEC_OPEN <= now_time <= _EXEC_CLOSE
        if not execution_window_open:
            logger.info(
                f"⏰ EXECUTION WINDOW CLOSED ({now_time.strftime('%H:%M')} IST) — "
                f"no new buys (trades only 9:30–15:00)"
            )

        # ── Sector concentration: track current sector exposure ────────────
        sector_exposure: dict[str, float] = {}
        total_portfolio = self.portfolio.get_total_value()
        for tk, pos in self.portfolio.state["positions"].items():
            price_pos = DataFetcher.get_current_price(tk)
            val = pos["qty"] * price_pos
            for sector, tickers in SectorMomentumStrategy.SECTOR_STOCKS.items():
                if tk in tickers:
                    sector_exposure[sector] = sector_exposure.get(sector, 0) + val
                    break
        MAX_SECTOR_PCT = 0.30  # max 30% of portfolio in any single sector

        for ticker, composite_strength, price, strategies in buy_candidates:
            # Guard-rail 0: macro / drawdown hard limit
            if new_buys_this_cycle >= max_new_buys:
                logger.info(
                    f"🛡️  RISK HALT — max_new_buys={max_new_buys} reached "
                    f"(macro={macro_label}, drawdown={dd_level})"
                )
                break

            strat_str = "+".join(sorted(set(strategies)))

            # Guard-rail 1: minimum strength (dynamic — set by LearningEngine)
            if composite_strength < buy_threshold:
                logger.debug(
                    f"⏭  SKIP {ticker} — strength {composite_strength:.0f} < {buy_threshold} [{strat_str}]"
                )
                continue

            # Guard-rail 2: multi-strategy consensus
            # Institutional rule: single-strategy signals are noise.
            # Require at least 2 strategies to agree (or very high conviction single).
            n_strategies = len(set(strategies))
            if n_strategies < 2 and composite_strength < 80:
                logger.debug(
                    f"⏭  SKIP {ticker} — only 1 strategy ({strat_str}) "
                    f"with strength {composite_strength:.0f} < 80"
                )
                continue

            # Guard-rail 3: execution time window
            if not execution_window_open:
                logger.debug(f"⏭  SKIP {ticker} — outside execution window")
                continue

            # Guard-rail 4: volume / liquidity filter (uses already-fetched batch data — no extra API calls)
            try:
                df_vol = all_data.get(ticker)
                if df_vol is not None and not df_vol.empty and len(df_vol) >= 5:
                    avg_vol   = float(df_vol["Volume"].iloc[-20:].mean()) if len(df_vol) >= 20 else float(df_vol["Volume"].mean())
                    today_vol = float(df_vol["Volume"].iloc[-1])
                    avg_val   = avg_vol * float(df_vol["Close"].iloc[-1])
                    if avg_val < 1_00_00_000:   # ₹1 crore minimum ADV
                        logger.info(f"⏭  SKIP {ticker} — illiquid (avg daily ₹{avg_val/1e7:.1f}Cr < ₹1Cr)")
                        continue
                    if today_vol < avg_vol * 0.30:
                        logger.info(f"⏭  SKIP {ticker} — low volume today ({today_vol/avg_vol:.0%} of avg)")
                        continue
                # If ticker not in all_data, allow through — data may have been missing due to rate limit
            except Exception:
                pass

            # Guard-rail 5: minimum price — no penny stocks under ₹50
            if price > 0 and price < 50:
                logger.info(f"⏭  SKIP {ticker} — price ₹{price:.2f} below ₹50 minimum")
                continue

            # Guard-rail 6: sector concentration limit
            ticker_sector = None
            for sector, tickers in SectorMomentumStrategy.SECTOR_STOCKS.items():
                if ticker in tickers:
                    ticker_sector = sector
                    break
            if ticker_sector:
                current_exposure = sector_exposure.get(ticker_sector, 0) / total_portfolio
                if current_exposure >= MAX_SECTOR_PCT:
                    logger.info(
                        f"⏭  SKIP {ticker} — sector '{ticker_sector}' already at "
                        f"{current_exposure:.0%} of portfolio (max {MAX_SECTOR_PCT:.0%})"
                    )
                    continue

            if self.portfolio.in_cooldown(ticker):
                logger.info(f"⏳ COOLDOWN {ticker} — skipping (exited recently)")
                continue

            if price <= 0:
                price = DataFetcher.get_current_price(ticker)

            # Position rotation: if at max positions, exit the weakest holder
            # to make room for a higher-conviction signal
            positions = self.portfolio.state.get("positions", {})
            if len(positions) >= MAX_POSITIONS:
                # Find the position with lowest unrealised P&L % that isn't in profit >5%
                weakest_ticker = None
                weakest_pnl_pct = float("inf")
                for held_tk, held_pos in positions.items():
                    held_price = DataFetcher.get_current_price(held_tk) or held_pos["avg_price"]
                    held_pnl_pct = (held_price - held_pos["avg_price"]) / held_pos["avg_price"]
                    # Only rotate out if: flat or losing, AND new signal is meaningfully stronger
                    if held_pnl_pct < 0.05 and held_pnl_pct < weakest_pnl_pct:
                        weakest_pnl_pct = held_pnl_pct
                        weakest_ticker = held_tk

                if weakest_ticker and composite_strength > buy_threshold + 10:
                    rotate_price = DataFetcher.get_current_price(weakest_ticker) or 0
                    if rotate_price > 0:
                        self.portfolio.execute_sell(
                            weakest_ticker, rotate_price,
                            reason=f"ROTATION→{ticker} (new strength={composite_strength:.0f})"
                        )
                        logger.info(
                            f"🔄 ROTATE out {weakest_ticker} ({weakest_pnl_pct:+.1%}) "
                            f"→ in {ticker} (strength={composite_strength:.0f})"
                        )

            logger.info(
                f"🔎 Evaluating BUY {ticker} @ ₹{price:.2f}  "
                f"strength={composite_strength:.0f}  strategies={n_strategies}×[{strat_str}]"
            )
            trade = self.portfolio.execute_buy(
                ticker, price,
                strategy=strat_str,
                reason=f"SIGNAL strength={composite_strength:.0f} strategies={n_strategies}",
                strength=composite_strength,
            )
            if trade:
                executed.append(trade)
                new_buys_this_cycle += 1
                # Update sector exposure tracker
                if ticker_sector:
                    sector_exposure[ticker_sector] = (
                        sector_exposure.get(ticker_sector, 0) + trade["qty"] * price
                    )
                logger.info(
                    f"✅ BOUGHT {ticker}  qty={trade['qty']}  @ ₹{price:.2f}  [{strat_str}]"
                )

        # ── SELL signals → PUT options (institutional approach) ───────────
        # Equity desk is long-only. Bearish views are expressed via PUT options
        # on F&O-eligible stocks. Same logic a real desk would use.
        put_trades = []
        try:
            from fno_engine import get_fno_agent, is_fno_eligible  # noqa: PLC0415
            fno = get_fno_agent()

            # Build sell candidates (same aggregation logic as buys)
            sell_candidates = []
            for ticker, agg in sell_agg.items():
                composite = float(np.mean(agg["strengths"]))
                sell_candidates.append((ticker, composite, agg["price"], agg["strategies"]))
            sell_candidates.sort(key=lambda x: x[1], reverse=True)

            for ticker, strength, price, strategies in sell_candidates:
                if strength < 70:
                    continue   # too weak for an options trade
                if not execution_window_open:
                    break
                if self.portfolio.in_cooldown(ticker):
                    continue
                # Don't open a PUT if we already hold a long equity position
                # (that would be an internal hedge — use stop-loss instead)
                if ticker in self.portfolio.state.get("positions", {}):
                    logger.debug(f"[FNO] SKIP PUT {ticker} — we hold long equity; stop-loss handles this")
                    continue
                spot = price if price > 0 else DataFetcher.get_current_price(ticker)
                if not spot or spot <= 0:
                    continue
                strat_str = "+".join(sorted(set(strategies)))
                trade = fno.portfolio.execute_sell_signal_as_put(
                    ticker   = ticker,
                    spot     = spot,
                    strength = strength,
                    strategy = strat_str,
                    reason   = f"Equity SELL signal strength={strength:.0f}",
                )
                if trade:
                    put_trades.append(trade)
                    logger.info(f"🔴 SELL→PUT {ticker}  strength={strength:.0f}  [{strat_str}]")
        except Exception as _e:
            logger.warning(f"SELL→PUT routing error: {_e}")

        # ── Self-learning: update strategy weights from closed trades ─────────
        try:
            learn_summary = learning.learn_from_trades()
            if learn_summary["new_trades_processed"] > 0:
                logger.info(
                    f"🧠 Learning: processed {learn_summary['new_trades_processed']} trade(s) | "
                    f"overall_win_rate={learn_summary['overall_win_rate']:.1%} | "
                    f"threshold={learning.get_threshold()}"
                )
                for wc in learn_summary["weight_changes"]:
                    logger.info(
                        f"   ↕ {wc['strategy']} weight {wc['old']:.2f}→{wc['new']:.2f} "
                        f"(win_rate={wc['win_rate']:.1%})"
                    )
                if learn_summary["threshold_change"]:
                    tc = learn_summary["threshold_change"]
                    logger.info(f"   ↕ Buy threshold {tc['old']}→{tc['new']}")
        except Exception as exc:
            logger.warning(f"[Learning] learn_from_trades error: {exc}")

        elapsed = round(time.time() - t0, 1)
        summary = {
            "cycle_time_s":    elapsed,
            "signals_count":   len(signals),
            "buys_executed":   len([t for t in executed if t["action"] == "BUY"]),
            "sells_executed":  len([t for t in executed if t["action"] == "SELL"]),
            "stops_triggered": len(stops),
            "portfolio_value": round(self.portfolio.get_total_value(), 2),
            "timestamp":       _now_ist().isoformat(),
            "buy_threshold":   learning.get_threshold(),
        }
        logger.info(f"=== Cycle done in {elapsed}s | {summary} ===")
        return summary

    def get_dashboard_data(self) -> dict:
        """All data needed to render the dashboard (signals page, KPIs, charts)."""
        port = self.portfolio

        # Trade log
        trades = []
        if TRADE_LOG_FILE.exists():
            with open(TRADE_LOG_FILE) as f:
                trades = json.load(f)

        # Signals
        signals = []
        if SIGNALS_FILE.exists():
            with open(SIGNALS_FILE) as f:
                d = json.load(f)
                signals = d.get("signals", [])
                signals_updated = d.get("updated_at", "")
        else:
            signals_updated = ""

        total_val   = port.get_total_value()
        unreal_pnl  = port.get_unrealised_pnl()
        real_pnl    = port.state.get("realised_pnl", 0)
        total_pnl   = unreal_pnl + real_pnl
        # P&L % based on total deployed capital (₹13L = ₹10L + ₹2L F&O + ₹1L commodity)
        total_pnl_pct = (total_pnl / INITIAL_CAPITAL) * 100

        # Equity curve: reconstruct from trades + append live value
        equity_curve = _build_equity_curve(
            trades,
            current_value=total_val,
            created_at=port.state.get("created_at"),
            start_value=INITIAL_CAPITAL,  # ₹13L total deployed capital
        )

        # Strategy performance: closed P&L + unrealised from open positions
        strat_perf = _calc_strategy_perf(trades, port.get_positions_display())

        return {
            "portfolio": {
                "total_value":    round(total_val, 2),
                "cash":           round(port.state["cash"], 2),
                "invested":       round(total_val - port.state["cash"], 2),
                "initial":        INITIAL_CAPITAL,
                "realised_pnl":   round(real_pnl, 2),
                "unrealised_pnl": round(unreal_pnl, 2),
                "total_pnl":      round(total_pnl, 2),
                "total_pnl_pct":  round(total_pnl_pct, 2),
                "last_updated":   port.state.get("last_updated", ""),
            },
            "positions":      port.get_positions_display(),
            "trades":         list(reversed(trades[-50:])),   # last 50
            "signals":        _diversify_signals(signals, max_per_strategy=10, total=150),
            "signals_updated": signals_updated,
            "equity_curve":   equity_curve,
            "strategy_perf":  strat_perf,
            "today_pnl":      _calc_today_pnl(trades, total_val, equity_curve, INITIAL_CAPITAL, port.state),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _diversify_signals(signals: list, max_per_strategy: int = 5, total: int = 50) -> list:
    """
    Return a strategy-diversified view of signals so the dashboard never shows
    30 MOMENTUM rows and nothing else.

    Algorithm:
      1. Sort BUY signals by strength desc, SELL signals by strength desc.
      2. Round-robin across strategies, taking up to max_per_strategy from each.
      3. Cap at `total` signals.
    """
    from collections import defaultdict
    buckets: dict[str, list] = defaultdict(list)
    for sig in signals:
        if sig.get("signal") in ("BUY", "SELL"):
            buckets[sig.get("strategy", "OTHER")].append(sig)

    # Sort each bucket by strength descending
    for strat in buckets:
        buckets[strat].sort(key=lambda x: x.get("strength", 0), reverse=True)
        buckets[strat] = buckets[strat][:max_per_strategy]

    # Round-robin interleave
    result = []
    strat_lists = list(buckets.values())
    idx = 0
    while len(result) < total and any(strat_lists):
        if strat_lists:
            lst = strat_lists[idx % len(strat_lists)]
            if lst:
                result.append(lst.pop(0))
            if not lst:
                strat_lists.pop(idx % len(strat_lists))
                if not strat_lists:
                    break
            else:
                idx += 1
    return result


def _calc_today_pnl(trades: list, current_value: float, equity_curve: list,
                    initial: float, portfolio_state: dict = None) -> dict:
    """
    Today's P&L = current value minus today's opening baseline.

    Priority:
      1. portfolio_state["day_start_value"] — snapshotted at midnight/first save of the day
      2. Last equity-curve point before today
      3. Initial capital (first-ever day)
    """
    today_str = _now_ist().strftime("%Y-%m-%d")

    # Realised P&L from today's SELL trades
    today_realised = sum(
        float(t.get("pnl") or 0)
        for t in trades
        if t.get("action") == "SELL" and (t.get("time") or "").startswith(today_str)
    )

    # Best baseline: portfolio's own daily snapshot
    baseline = None
    if portfolio_state and portfolio_state.get("day_start_date") == today_str:
        baseline = portfolio_state.get("day_start_value")

    # Fallback: last equity-curve point before today
    if baseline is None:
        for pt in reversed(equity_curve):
            if pt["date"] < today_str:
                baseline = pt["value"]
                break

    # Last resort: initial capital
    if baseline is None:
        baseline = initial

    day_pnl     = round(current_value - baseline, 2)
    day_pnl_pct = round((day_pnl / baseline) * 100, 2) if baseline else 0.0

    return {
        "day_pnl":         day_pnl,
        "day_pnl_pct":     day_pnl_pct,
        "day_realised":    round(today_realised, 2),
        "day_trade_count": sum(1 for t in trades if t.get("action") == "SELL"
                               and (t.get("time") or "").startswith(today_str)),
    }


def _build_equity_curve(trades: list, current_value: float = None,
                        created_at: str = None, start_value: float = None) -> list:
    """
    Equity curve from trade log + current live portfolio value.
    Starts at the portfolio creation date using the ACTUAL first portfolio
    value (not INITIAL_CAPITAL, which is inflated by top-ups added later).
    """
    start_date = (created_at or _now_ist().isoformat())[:10]
    # Use the real starting value from trades if available, otherwise INITIAL_CAPITAL
    # This prevents the curve from showing ₹13L when trading started at ₹10L
    first_val = start_value if start_value else INITIAL_CAPITAL
    curve = [{"date": start_date, "value": first_val}]
    running = first_val
    daily = {}
    for t in trades:
        date = t["time"][:10]
        if t["pnl"] is not None:
            daily[date] = daily.get(date, 0) + t["pnl"]
    for date in sorted(daily):
        if date == start_date:
            continue   # skip duplicate start point
        running += daily[date]
        curve.append({"date": date, "value": round(running, 2)})
    # Always end with the current live value (includes unrealised P&L)
    if current_value is not None:
        today = _now_ist().strftime("%Y-%m-%d")
        if curve and curve[-1]["date"] == today:
            curve[-1]["value"] = round(current_value, 2)
        else:
            curve.append({"date": today, "value": round(current_value, 2)})
    return curve


def _calc_strategy_perf(trades: list, positions: list = None) -> list:
    """
    Strategy P&L = closed trade P&L + unrealised P&L from open positions.
    This way the chart is meaningful even when no positions have been closed yet.
    """
    strats: dict = {}

    # Closed trade P&L
    for t in trades:
        s = t.get("strategy", "UNKNOWN")
        if s not in strats:
            strats[s] = {"trades": 0, "wins": 0, "closed_pnl": 0.0, "open_pnl": 0.0, "open_count": 0}
        if t["action"] == "SELL" and t["pnl"] is not None:
            strats[s]["trades"] += 1
            strats[s]["closed_pnl"] += t["pnl"]
            if t["pnl"] > 0:
                strats[s]["wins"] += 1

    # Unrealised P&L from open positions per strategy
    if positions:
        for pos in positions:
            s = pos.get("strategy", "UNKNOWN")
            if s not in strats:
                strats[s] = {"trades": 0, "wins": 0, "closed_pnl": 0.0, "open_pnl": 0.0, "open_count": 0}
            strats[s]["open_pnl"]   += pos.get("pnl", 0.0)
            strats[s]["open_count"] += 1

    result = []
    for s, d in strats.items():
        win_rate  = round(d["wins"] / d["trades"] * 100, 1) if d["trades"] > 0 else 0
        total_pnl = d["closed_pnl"] + d["open_pnl"]
        result.append({
            "strategy":   s,
            "trades":     d["trades"],
            "wins":       d["wins"],
            "pnl":        round(total_pnl, 2),
            "closed_pnl": round(d["closed_pnl"], 2),
            "open_pnl":   round(d["open_pnl"], 2),
            "open_count": d["open_count"],
            "win_rate":   win_rate,
        })
    return result


# Singleton
_agent = None

def get_agent() -> TradingAgent:
    global _agent
    if _agent is None:
        _agent = TradingAgent()
    return _agent
