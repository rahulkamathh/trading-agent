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

NIFTY50_TICKERS = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","SBIN.NS","BHARTIARTL.NS","BAJFINANCE.NS","KOTAKBANK.NS",
    "LT.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS","SUNPHARMA.NS",
    "TITAN.NS","WIPRO.NS","ULTRACEMCO.NS","NESTLEIND.NS","POWERGRID.NS",
    "NTPC.NS","M&M.NS","HCLTECH.NS","ONGC.NS","JSWSTEEL.NS",
    "TATAMOTORS.NS","ADANIENT.NS","COALINDIA.NS","BAJAJFINSV.NS","GRASIM.NS",
    "TECHM.NS","BPCL.NS","CIPLA.NS","DRREDDY.NS","EICHERMOT.NS",
    "APOLLOHOSP.NS","DIVISLAB.NS","TATACONSUM.NS","INDUSINDBK.NS","SBILIFE.NS",
    "HDFCLIFE.NS","ADANIPORTS.NS","UPL.NS","HEROMOTOCO.NS","BRITANNIA.NS",
    "TATASTEEL.NS","ITC.NS","BAJAJ-AUTO.NS","HINDALCO.NS","VEDL.NS",
]

SECTOR_ETFS = {
    "Nifty50":   "NIFTYBEES.NS",
    "Banking":   "BANKBEES.NS",
    "IT":        "ITBEES.NS",
    "Pharma":    "PHARMABEES.NS",
    "Gold":      "GOLDBEES.NS",
}

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
MAX_POSITIONS    = 15          # max concurrent positions

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
    def fetch_multi(cls, tickers: list, period: str = "2y") -> dict:
        result = {}
        for t in tickers:
            df = cls.fetch(t, period=period)
            if not df.empty:
                result[t] = df
        return result

    @classmethod
    def get_current_price(cls, ticker: str) -> float:
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
# Signal Aggregator
# ---------------------------------------------------------------------------

class SignalAggregator:

    def __init__(self):
        self.momentum   = MomentumStrategy()
        self.mean_rev   = MeanReversionStrategy()
        self.multifactor= MultiFactorStrategy()
        self.sector_rot = SectorRotationStrategy()

    def run(self) -> list:
        logger.info("Fetching market data…")
        stock_data = DataFetcher.fetch_multi(NIFTY50_TICKERS, period="2y")
        etf_data   = DataFetcher.fetch_multi(list(SECTOR_ETFS.values()), period="2y")
        index_df   = DataFetcher.fetch("^NSEI", period="2y")

        all_signals = []
        all_signals += self.momentum.generate_signals(stock_data)
        all_signals += self.mean_rev.generate_signals(stock_data)
        all_signals += self.multifactor.generate_signals(stock_data)
        all_signals += self.sector_rot.generate_signals(etf_data, index_df)

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

        # Equity history from trade log (reconstruct)
        equity_curve = _build_equity_curve(trades)

        # Strategy performance
        strat_perf = _calc_strategy_perf(trades)

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

def _build_equity_curve(trades: list) -> list:
    """Approximate equity curve from trade log."""
    curve = [{"date": datetime.now().strftime("%Y-%m-%d"), "value": INITIAL_CAPITAL}]
    running = INITIAL_CAPITAL
    daily = {}
    for t in trades:
        date = t["time"][:10]
        if t["pnl"] is not None:
            daily[date] = daily.get(date, 0) + t["pnl"]
    for date in sorted(daily):
        running += daily[date]
        curve.append({"date": date, "value": round(running, 2)})
    return curve


def _calc_strategy_perf(trades: list) -> list:
    strats: dict = {}
    for t in trades:
        s = t.get("strategy", "UNKNOWN")
        if s not in strats:
            strats[s] = {"trades": 0, "wins": 0, "pnl": 0.0}
        if t["action"] == "SELL" and t["pnl"] is not None:
            strats[s]["trades"] += 1
            strats[s]["pnl"]    += t["pnl"]
            if t["pnl"] > 0:
                strats[s]["wins"] += 1
    result = []
    for s, d in strats.items():
        win_rate = round(d["wins"] / d["trades"] * 100, 1) if d["trades"] > 0 else 0
        result.append({
            "strategy": s,
            "trades":   d["trades"],
            "wins":     d["wins"],
            "pnl":      round(d["pnl"], 2),
            "win_rate": win_rate,
        })
    return result


# Singleton
_agent = None

def get_agent() -> TradingAgent:
    global _agent
    if _agent is None:
        _agent = TradingAgent()
    return _agent
