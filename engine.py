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
from datetime import datetime, timedelta
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
    "PFIZER.NS","SANOFI.NS","LAURUS.NS","GRANULES.NS","SYNGENE.NS",
    "AJANTPHARM.NS","SOLARA.NS","MARKSANS.NS","SUVEN.NS","NEULANDLAB.NS",

    # ══ HEALTHCARE / DIAGNOSTICS ══════════════════════════════════════════
    "APOLLOHOSP.NS","MAXHEALTH.NS","FORTIS.NS","NARAYANA.NS","ASTER.NS",
    "METROPOLIS.NS","LALPATHLAB.NS","THYROCARE.NS","KRSNAA.NS","VIJAYA.NS",

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
    "HAL.NS","COCHINSHIP.NS","MAZAGON.NS","GRSE.NS","MTAR.NS",
    "DATAPATTNS.NS","PARAS.NS","IDEAFORGE.NS",

    # ══ INFRASTRUCTURE / CONSTRUCTION ══════════════════════════════════════
    "ADANIPORTS.NS","ADANIENT.NS","GMRINFRA.NS","IRB.NS","SADBHAV.NS",
    "NCC.NS","KPITL.NS","PNCINFRA.NS","HG-INFRA.NS","RVNL.NS","IRCON.NS",

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
    "EIHOTEL.NS","INDHOTEL.NS","CHALET.NS","LEMON.NS","MAHINDRAHOLIDAYS.NS",

    # ══ TEXTILES & APPAREL ══════════════════════════════════════════════════
    "RAYMOND.NS","ARVIND.NS","WELSPUN.NS","TRIDENT.NS","VARDHMAN.NS",
    "KTEX.NS","NITIN.NS","GRASIM.NS",

    # ══ AGRICULTURE & FOOD PROCESSING ══════════════════════════════════════
    "KRBL.NS","LTFOODS.NS","AVANTIFEED.NS","WATERBASE.NS","GODREJAGRO.NS",
    "RUCHI.NS","PATANJALIFOODS.NS","KSEEDS.NS","DHANUKA.NS","PIIND.NS",

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
    "HAL.NS", "COCHINSHIP.NS", "MAZAGON.NS", "GRSE.NS",
    # Banking & Finance (turnaround plays)
    "YESBANK.NS", "IDFCFIRSTB.NS", "FEDERALBNK.NS", "BANDHANBNK.NS",
    # Chemicals & Pharma
    "LAURUS.NS", "GRANULES.NS", "SYNGENE.NS", "PCBL.NS",
    # Telecom
    "IDEA.NS",
    # Others
    "GMRINFRA.NS", "ADANIGREEN.NS",
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

INITIAL_CAPITAL = 1_000_000   # ₹10 lakhs
MAX_POSITION_PCT = 0.08        # max 8% per position
STOP_LOSS_PCT    = 0.07        # 7% stop loss
TAKE_PROFIT_PCT  = 0.20        # 20% take profit
MAX_POSITIONS    = 25          # max concurrent positions

# ---------------------------------------------------------------------------
# Data Layer
# ---------------------------------------------------------------------------

class DataFetcher:
    """Fetches and caches NSE market data via yfinance."""

    _cache: dict = {}

    @classmethod
    def fetch(cls, ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
        key = f"{ticker}_{period}_{interval}"
        if key in cls._cache:
            return cls._cache[key]
        try:
            df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
            if df.empty:
                logger.warning(f"No data for {ticker}")
                return pd.DataFrame()
            # Flatten MultiIndex columns if present
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

    @classmethod
    def get_current_price(cls, ticker: str) -> float:
        # Prefer live Angel One feed when available
        try:
            from angelone_feed import get_feed  # pylint: disable=import-outside-toplevel
            live_price = get_feed().get_price(ticker)
            if live_price and live_price > 0:
                return live_price
        except ImportError:
            pass
        # Fallback: yfinance (delayed / EOD)
        df = cls.fetch(ticker, period="5d")
        if df.empty:
            return 0.0
        return float(df["Close"].iloc[-1])

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
            "created_at":   datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
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
        self.state["last_updated"] = datetime.now().isoformat()
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

    def can_buy(self, ticker: str, price: float) -> bool:
        if len(self.state["positions"]) >= MAX_POSITIONS:
            return False
        if ticker in self.state["positions"]:
            return False
        total_val = self.get_total_value()
        max_spend = total_val * MAX_POSITION_PCT
        return self.state["cash"] >= max_spend and price > 0

    def execute_buy(self, ticker: str, price: float, strategy: str, reason: str = "") -> dict | None:
        if not self.can_buy(ticker, price):
            return None
        total_val = self.get_total_value()
        spend     = min(total_val * MAX_POSITION_PCT, self.state["cash"] * 0.95)
        qty       = int(spend / price)
        if qty < 1:
            return None
        cost = qty * price
        self.state["cash"] -= cost
        self.state["positions"][ticker] = {
            "qty":        qty,
            "avg_price":  price,
            "strategy":   strategy,
            "entry_date": datetime.now().isoformat(),
            "stop_loss":  round(price * (1 - STOP_LOSS_PCT), 2),
            "target":     round(price * (1 + TAKE_PROFIT_PCT), 2),
        }
        self._save()
        trade = self._log_trade("BUY", ticker, qty, price, strategy, reason)
        logger.info(f"BUY  {ticker:20s} qty={qty} @ ₹{price:.2f}  [{strategy}]")
        return trade

    def execute_sell(self, ticker: str, price: float, reason: str = "") -> dict | None:
        pos = self.state["positions"].get(ticker)
        if not pos:
            return None
        qty      = pos["qty"]
        proceeds = qty * price
        pnl      = (price - pos["avg_price"]) * qty
        self.state["cash"]         += proceeds
        self.state["realised_pnl"] += pnl
        strategy = pos["strategy"]
        del self.state["positions"][ticker]
        self._save()
        trade = self._log_trade("SELL", ticker, qty, price, strategy, reason, pnl=pnl)
        logger.info(f"SELL {ticker:20s} qty={qty} @ ₹{price:.2f}  pnl=₹{pnl:.2f}  [{reason}]")
        return trade

    def check_stops(self) -> list:
        """Check stop-loss and take-profit for all open positions."""
        triggered = []
        for ticker, pos in list(self.state["positions"].items()):
            price = DataFetcher.get_current_price(ticker)
            if price <= 0:
                continue
            if price <= pos["stop_loss"]:
                trade = self.execute_sell(ticker, price, reason="STOP_LOSS")
                if trade:
                    triggered.append(trade)
            elif price >= pos["target"]:
                trade = self.execute_sell(ticker, price, reason="TAKE_PROFIT")
                if trade:
                    triggered.append(trade)
        return triggered

    def _log_trade(self, action, ticker, qty, price, strategy, reason, pnl=None) -> dict:
        log = []
        if TRADE_LOG_FILE.exists():
            with open(TRADE_LOG_FILE) as f:
                log = json.load(f)
        trade = {
            "id":       len(log) + 1,
            "action":   action,
            "ticker":   ticker,
            "qty":      qty,
            "price":    round(price, 2),
            "value":    round(qty * price, 2),
            "strategy": strategy,
            "reason":   reason,
            "pnl":      round(pnl, 2) if pnl is not None else None,
            "time":     datetime.now().isoformat(),
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
        now_str = datetime.now().isoformat()

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
            cutoff = (datetime.now() - timedelta(hours=self.MAX_SIGNAL_AGE_H)).isoformat()
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

        # Save to file
        with open(SIGNALS_FILE, "w") as f:
            json.dump({
                "signals":    all_signals,
                "updated_at": datetime.now().isoformat(),
            }, f, indent=2)
        logger.info(f"Generated {len(all_signals)} signals")
        return all_signals


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

        # 1. Check stops first
        stops = self.portfolio.check_stops()

        # 2. Generate fresh signals
        signals = self.aggregator.run()

        # 3. Execute BUY signals (prioritise by strength desc)
        buy_signals = sorted(
            [s for s in signals if s["signal"] == "BUY"],
            key=lambda x: x.get("strength", 0),
            reverse=True,
        )
        executed = []
        for sig in buy_signals:
            ticker = sig["ticker"]
            price  = sig.get("price", 0)
            if price <= 0:
                price = DataFetcher.get_current_price(ticker)
            trade = self.portfolio.execute_buy(ticker, price, sig["strategy"], "SIGNAL")
            if trade:
                executed.append(trade)

        # 4. Execute SELL signals for held positions
        sell_signals = [s for s in signals if s["signal"] == "SELL"]
        for sig in sell_signals:
            ticker = sig["ticker"]
            if ticker in self.portfolio.state["positions"]:
                price = sig.get("price", 0) or DataFetcher.get_current_price(ticker)
                trade = self.portfolio.execute_sell(ticker, price, "SIGNAL_EXIT")
                if trade:
                    executed.append(trade)

        elapsed = round(time.time() - t0, 1)
        summary = {
            "cycle_time_s":    elapsed,
            "signals_count":   len(signals),
            "buys_executed":   len([t for t in executed if t["action"] == "BUY"]),
            "sells_executed":  len([t for t in executed if t["action"] == "SELL"]),
            "stops_triggered": len(stops),
            "portfolio_value": round(self.portfolio.get_total_value(), 2),
            "timestamp":       datetime.now().isoformat(),
        }
        logger.info(f"=== Cycle done in {elapsed}s | {summary} ===")
        return summary

    def get_dashboard_data(self) -> dict:
        """All data needed to render the dashboard."""
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
        total_pnl_pct = (total_pnl / INITIAL_CAPITAL) * 100

        # Equity curve: reconstruct from trades + append live value
        equity_curve = _build_equity_curve(
            trades,
            current_value=total_val,
            created_at=port.state.get("created_at"),
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
            "signals":        signals[:30],                   # top 30
            "signals_updated": signals_updated,
            "equity_curve":   equity_curve,
            "strategy_perf":  strat_perf,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_equity_curve(trades: list, current_value: float = None, created_at: str = None) -> list:
    """
    Equity curve from trade log + current live portfolio value.
    Starts at the portfolio creation date. Always appends today's actual
    total value (cash + unrealised positions) as the latest data point.
    """
    start_date = (created_at or datetime.now().isoformat())[:10]
    curve = [{"date": start_date, "value": INITIAL_CAPITAL}]
    running = INITIAL_CAPITAL
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
        today = datetime.now().strftime("%Y-%m-%d")
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
