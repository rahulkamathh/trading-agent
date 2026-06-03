"""
Signal Analyzer
===============
Runs the agent's own technical + fundamental analysis on a Telegram signal
and sends a verdict back to the notification Telegram group.

Pipeline (called from telegram_agent._handle_message after parsing):
  1. Fetch 6 months of OHLCV data
  2. Compute technical indicators: RSI, MACD, EMA trend, volume, ATR
  3. Fetch basic fundamentals via yfinance: P/E, debt-to-equity, revenue growth
  4. Score each dimension 0–100 and produce a composite conviction score
  5. Format and send a rich analysis message to TELEGRAM_NOTIFY_CHAT_ID

The analysis is non-blocking — runs in a daemon thread so it never delays
the main message handler.
"""

import logging
import os
import threading
from datetime import datetime
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Technical Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_ohlcv(ticker: str):
    """Fetch 6-month daily OHLCV. Returns DataFrame or None."""
    try:
        import yfinance as yf
        import pandas as pd
        hist = yf.download(ticker, period="6mo", interval="1d",
                           auto_adjust=True, progress=False)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        return hist if not hist.empty and len(hist) >= 20 else None
    except Exception as exc:
        logger.debug(f"[Analyzer] OHLCV fetch failed for {ticker}: {exc}")
        return None


def _rsi(close, period=14):
    import pandas as pd
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, float("nan"))
    return 100 - 100 / (1 + rs)


def _ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def _macd(close):
    fast = _ema(close, 12)
    slow = _ema(close, 26)
    macd_line   = fast - slow
    signal_line = _ema(macd_line, 9)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def _atr(df, period=14):
    import pandas as pd
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def run_technical_analysis(ticker: str, direction: str) -> dict:
    """
    Returns dict with:
      score       : 0–100 technical conviction
      rsi         : float
      macd_signal : "bullish" | "bearish" | "neutral"
      trend       : "uptrend" | "downtrend" | "sideways"
      volume_ok   : bool
      atr_pct     : float (ATR as % of price — proxy for volatility)
      reasons     : list[str]  (human-readable bullet points)
    """
    result = {
        "score": 50, "rsi": None, "macd_signal": "neutral",
        "trend": "sideways", "volume_ok": True, "atr_pct": None,
        "reasons": [], "error": None,
    }

    df = _fetch_ohlcv(ticker)
    if df is None:
        result["error"] = "No price data"
        result["score"] = 40
        return result

    close  = df["Close"]
    volume = df["Volume"]
    price  = float(close.iloc[-1])
    score  = 50
    reasons = []

    # ── RSI ──────────────────────────────────────────────────────────────── #
    rsi_series = _rsi(close)
    rsi = float(rsi_series.iloc[-1])
    result["rsi"] = round(rsi, 1)

    if direction == "BUY":
        if rsi < 30:
            score += 15; reasons.append(f"RSI {rsi:.0f} — oversold, good BUY zone ✅")
        elif rsi < 50:
            score += 10; reasons.append(f"RSI {rsi:.0f} — room to run ✅")
        elif rsi < 65:
            score += 0;  reasons.append(f"RSI {rsi:.0f} — neutral ⚪")
        elif rsi < 75:
            score -= 8;  reasons.append(f"RSI {rsi:.0f} — approaching overbought ⚠️")
        else:
            score -= 18; reasons.append(f"RSI {rsi:.0f} — overbought, risky entry ❌")
    else:  # SELL
        if rsi > 70:
            score += 15; reasons.append(f"RSI {rsi:.0f} — overbought, good SELL zone ✅")
        elif rsi > 55:
            score += 8;  reasons.append(f"RSI {rsi:.0f} — elevated ✅")
        elif rsi > 40:
            score += 0;  reasons.append(f"RSI {rsi:.0f} — neutral ⚪")
        else:
            score -= 15; reasons.append(f"RSI {rsi:.0f} — oversold, risky SELL ❌")

    # ── MACD ─────────────────────────────────────────────────────────────── #
    macd_line, signal_line, histogram = _macd(close)
    macd_val  = float(macd_line.iloc[-1])
    sig_val   = float(signal_line.iloc[-1])
    hist_val  = float(histogram.iloc[-1])
    hist_prev = float(histogram.iloc[-2]) if len(histogram) > 1 else 0

    if macd_val > sig_val and hist_val > hist_prev:
        result["macd_signal"] = "bullish"
        if direction == "BUY":
            score += 12; reasons.append("MACD bullish crossover / momentum building ✅")
        else:
            score -= 8;  reasons.append("MACD bullish — works against SELL ⚠️")
    elif macd_val < sig_val and hist_val < hist_prev:
        result["macd_signal"] = "bearish"
        if direction == "SELL":
            score += 12; reasons.append("MACD bearish momentum ✅")
        else:
            score -= 8;  reasons.append("MACD bearish — works against BUY ⚠️")
    else:
        reasons.append("MACD neutral / no clear crossover ⚪")

    # ── EMA Trend (20/50/200) ─────────────────────────────────────────────── #
    ema20  = float(_ema(close, 20).iloc[-1])
    ema50  = float(_ema(close, 50).iloc[-1])
    ema200 = float(_ema(close, 200).iloc[-1]) if len(close) >= 200 else None

    above_ema20 = price > ema20
    above_ema50 = price > ema50
    above_ema200 = (price > ema200) if ema200 else None

    if above_ema20 and above_ema50:
        result["trend"] = "uptrend"
        if direction == "BUY":
            score += 10; reasons.append("Price above EMA20 & EMA50 — uptrend confirmed ✅")
        else:
            score -= 6;  reasons.append("Price in uptrend — risky SELL ⚠️")
    elif not above_ema20 and not above_ema50:
        result["trend"] = "downtrend"
        if direction == "SELL":
            score += 10; reasons.append("Price below EMA20 & EMA50 — downtrend ✅")
        else:
            score -= 6;  reasons.append("Price below EMAs — weak BUY setup ⚠️")
    else:
        result["trend"] = "sideways"
        reasons.append("Mixed EMA signals — sideways market ⚪")

    if above_ema200 is True:
        if direction == "BUY":
            score += 5; reasons.append("Above 200 EMA — long-term uptrend ✅")
    elif above_ema200 is False:
        if direction == "BUY":
            score -= 5; reasons.append("Below 200 EMA — long-term downtrend ⚠️")

    # ── Volume ───────────────────────────────────────────────────────────── #
    avg_vol  = float(volume.rolling(20).mean().iloc[-1])
    last_vol = float(volume.iloc[-1])
    vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0
    result["volume_ok"] = vol_ratio >= 0.8

    if vol_ratio >= 1.5:
        score += 8;  reasons.append(f"Volume {vol_ratio:.1f}× average — strong participation ✅")
    elif vol_ratio >= 1.0:
        score += 3;  reasons.append(f"Volume normal ({vol_ratio:.1f}× avg) ⚪")
    else:
        score -= 5;  reasons.append(f"Volume low ({vol_ratio:.1f}× avg) — weak signal ⚠️")

    # ── ATR (volatility) ─────────────────────────────────────────────────── #
    atr_series = _atr(df)
    atr = float(atr_series.iloc[-1])
    atr_pct = atr / price * 100 if price > 0 else 0
    result["atr_pct"] = round(atr_pct, 2)

    if atr_pct > 5:
        score -= 5; reasons.append(f"High volatility (ATR {atr_pct:.1f}%) — wider SL needed ⚠️")

    result["score"] = max(10, min(100, round(score)))
    result["reasons"] = reasons
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Fundamental Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def run_fundamental_analysis(ticker: str) -> dict:
    """
    Fetches key fundamentals via yfinance and scores them.
    Returns dict with score, pe, debt_to_equity, revenue_growth, reasons.
    """
    result = {
        "score": 50, "pe": None, "debt_to_equity": None,
        "revenue_growth": None, "market_cap_cr": None,
        "reasons": [], "error": None,
    }

    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}

        score   = 50
        reasons = []

        # ── P/E Ratio ───────────────────────────────────────────────────── #
        pe = info.get("trailingPE") or info.get("forwardPE")
        result["pe"] = round(pe, 1) if pe else None
        if pe:
            if pe < 0:
                score -= 10; reasons.append(f"P/E negative (loss-making) ❌")
            elif pe < 15:
                score += 10; reasons.append(f"P/E {pe:.1f} — undervalued ✅")
            elif pe < 25:
                score += 5;  reasons.append(f"P/E {pe:.1f} — fair value ✅")
            elif pe < 40:
                score -= 5;  reasons.append(f"P/E {pe:.1f} — slightly expensive ⚠️")
            else:
                score -= 12; reasons.append(f"P/E {pe:.1f} — expensive ❌")

        # ── Debt to Equity ───────────────────────────────────────────────── #
        de = info.get("debtToEquity")
        result["debt_to_equity"] = round(de / 100, 2) if de else None  # yf gives as %, normalise
        if de is not None:
            if de < 30:
                score += 8;  reasons.append(f"Low debt (D/E {de/100:.1f}×) ✅")
            elif de < 80:
                score += 3;  reasons.append(f"Moderate debt (D/E {de/100:.1f}×) ⚪")
            elif de < 150:
                score -= 5;  reasons.append(f"High debt (D/E {de/100:.1f}×) ⚠️")
            else:
                score -= 12; reasons.append(f"Very high debt (D/E {de/100:.1f}×) ❌")

        # ── Revenue Growth ───────────────────────────────────────────────── #
        rev_growth = info.get("revenueGrowth")
        result["revenue_growth"] = round(rev_growth * 100, 1) if rev_growth else None
        if rev_growth is not None:
            if rev_growth > 0.20:
                score += 10; reasons.append(f"Strong revenue growth {rev_growth*100:.0f}% ✅")
            elif rev_growth > 0.05:
                score += 5;  reasons.append(f"Revenue growing {rev_growth*100:.0f}% ✅")
            elif rev_growth > -0.05:
                score += 0;  reasons.append(f"Revenue flat ({rev_growth*100:.0f}%) ⚪")
            else:
                score -= 8;  reasons.append(f"Revenue declining {rev_growth*100:.0f}% ❌")

        # ── Market Cap ───────────────────────────────────────────────────── #
        mc = info.get("marketCap")
        if mc:
            mc_cr = mc / 1e7   # convert to crores (1 crore = 10M INR)
            result["market_cap_cr"] = round(mc_cr)
            if mc_cr > 50000:
                score += 5; reasons.append(f"Large cap (₹{mc_cr/100:.0f}k Cr) — liquid ✅")
            elif mc_cr > 5000:
                reasons.append(f"Mid cap (₹{mc_cr:.0f} Cr) ⚪")
            else:
                score -= 3; reasons.append(f"Small cap (₹{mc_cr:.0f} Cr) — higher risk ⚠️")

        # ── Profit Margins ───────────────────────────────────────────────── #
        margin = info.get("profitMargins")
        if margin is not None:
            if margin > 0.15:
                score += 5; reasons.append(f"Strong profit margin {margin*100:.0f}% ✅")
            elif margin > 0.05:
                pass   # normal, don't comment
            elif margin < 0:
                score -= 8; reasons.append(f"Loss-making (margin {margin*100:.0f}%) ❌")

        result["score"]   = max(10, min(100, round(score)))
        result["reasons"] = reasons

    except Exception as exc:
        result["error"] = str(exc)
        logger.debug(f"[Analyzer] Fundamentals failed for {ticker}: {exc}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Composite Analyzer + Telegram Reply
# ═══════════════════════════════════════════════════════════════════════════════

def _bar(score: int, width: int = 8) -> str:
    """Visual score bar: ████░░░░ 72/100"""
    filled = round(score / 100 * width)
    return "█" * filled + "░" * (width - filled) + f" {score}/100"


def _score_emoji(score: int) -> str:
    if score >= 75: return "🟢"
    if score >= 55: return "🟡"
    return "🔴"


def analyze_and_reply(
    signal: dict,
    group_title: str,
    async_send: bool = True,
) -> None:
    """
    Entry point called from telegram_agent._handle_message.
    Runs analysis in a daemon thread (async_send=True) so it never blocks.

    signal dict has keys: ticker, direction, entry_price, targets, stop_loss,
    trade_type, specificity, timeline_raw (from SignalParser.parse → "parsed" sub-dict)
    + group_id, group_title, raw_text, received_at at the top level.
    """
    if async_send:
        t = threading.Thread(
            target=_do_analyze_and_reply,
            args=(signal, group_title),
            daemon=True,
        )
        t.start()
    else:
        _do_analyze_and_reply(signal, group_title)


def _do_analyze_and_reply(signal: dict, group_title: str) -> None:
    """Blocking version — runs in background thread."""
    parsed    = signal.get("parsed", {})
    ticker    = parsed.get("ticker", "")
    direction = parsed.get("direction", "BUY")
    entry     = parsed.get("entry_price")
    targets   = parsed.get("targets", [])
    sl        = parsed.get("stop_loss")
    trade_type = parsed.get("trade_type", "")
    timeline  = parsed.get("timeline_raw", "unspecified")
    raw_text  = signal.get("raw_text", "")

    if not ticker:
        return

    logger.info(f"[Analyzer] Analyzing {ticker} {direction} from '{group_title}'")

    # ── Run analyses ─────────────────────────────────────────────────────── #
    ta  = run_technical_analysis(ticker, direction)
    fa  = run_fundamental_analysis(ticker)

    # ── Composite score ───────────────────────────────────────────────────── #
    # Weight: 60% technical (tradeable timeframe), 40% fundamental
    composite = round(ta["score"] * 0.60 + fa["score"] * 0.40)

    # Boost if signal has entry + targets + SL (high specificity)
    specificity = parsed.get("specificity", 0.3)
    if specificity >= 0.9:
        composite = min(100, composite + 5)

    # ── Decision ─────────────────────────────────────────────────────────── #
    THRESHOLD = 60

    # Check if market is open before deciding to trade
    try:
        from engine import _market_open  # pylint: disable=import-outside-toplevel
        market_is_open = _market_open()
    except Exception:
        market_is_open = False

    will_trade = composite >= THRESHOLD and direction == "BUY" and market_is_open

    # ── Actually execute the trade if conviction is high enough ──────────── #
    trade_result = None
    if will_trade:
        try:
            from engine import get_agent, DataFetcher  # pylint: disable=import-outside-toplevel
            agent = get_agent()
            # Always use live price — never trust Telegram message price directly
            price = DataFetcher.get_current_price(ticker)
            if price and price > 0:
                trade_result = agent.portfolio.execute_buy(
                    ticker, price,
                    strategy="TelegramSignal",
                    reason=f"Telegram signal from {group_title} | conviction {composite}/100"
                )
        except Exception as _te:
            logger.warning(f"[Analyzer] Trade execution error for {ticker}: {_te}")

    if not market_is_open and composite >= THRESHOLD and direction == "BUY":
        action_str = "🕐 <b>QUEUED</b> — market closed, will evaluate at next open"
    elif will_trade:
        if trade_result:
            qty   = trade_result.get("qty", 0)
            price = trade_result.get("price", 0)
            action_str = f"✅ <b>EXECUTED</b> — Bought {qty} × ₹{price:,.2f}"
        else:
            action_str = "⚠️ <b>SKIPPED execution</b> — insufficient cash or max positions reached"
    elif composite < THRESHOLD:
        action_str = "⏭ <b>SKIPPING</b> — below conviction threshold"
    else:
        action_str = "📋 <b>SELL signal noted</b> — monitoring positions"

    # ── Format message ────────────────────────────────────────────────────── #
    # Price levels block
    price_lines = []
    if entry:
        price_lines.append(f"  Entry:   ₹{entry:,.2f}")
    if targets:
        tgt_str = " / ".join(f"₹{t:,.2f}" for t in targets[:3])
        price_lines.append(f"  Targets: {tgt_str}")
    if sl:
        price_lines.append(f"  SL:      ₹{sl:,.2f}")
    if trade_type:
        price_lines.append(f"  Type:    {trade_type.upper()}")
    if timeline and timeline != "unspecified":
        price_lines.append(f"  Timeline:{timeline}")

    # Technical reasons (top 3)
    ta_bullets = "\n".join(f"  • {r}" for r in ta["reasons"][:3])
    fa_bullets = "\n".join(f"  • {r}" for r in fa["reasons"][:3])

    msg = (
        f"🤖 <b>Agent Analysis — {ticker} {direction}</b>\n"
        f"📡 Source: {group_title}\n"
        f"─────────────────────────\n"
    )

    if price_lines:
        msg += "\n".join(price_lines) + "\n─────────────────────────\n"

    msg += (
        f"\n📊 <b>Technical</b> {_score_emoji(ta['score'])} {_bar(ta['score'])}\n"
        f"{ta_bullets}\n"
        f"\n💼 <b>Fundamental</b> {_score_emoji(fa['score'])} {_bar(fa['score'])}\n"
        f"{fa_bullets}\n"
        f"\n⚡ <b>Conviction</b>  {_score_emoji(composite)} {_bar(composite)}\n"
        f"\n{action_str}"
    )

    if ta.get("error") and fa.get("error"):
        msg += "\n⚠️ <i>Limited data available — analysis may be incomplete</i>"

    # ── Send to notification group ────────────────────────────────────────── #
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id   = os.environ.get("TELEGRAM_NOTIFY_CHAT_ID", "").strip()

    if not bot_token or not chat_id:
        logger.debug("[Analyzer] Bot token/chat_id not set — skipping reply")
        return

    try:
        url  = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        resp = requests.post(url, json={
            "chat_id":    chat_id,
            "text":       msg,
            "parse_mode": "HTML",
        }, timeout=10)
        if resp.status_code == 200:
            logger.info(f"[Analyzer] Sent analysis for {ticker} to group")
        else:
            logger.warning(f"[Analyzer] Telegram send failed: {resp.text[:200]}")
    except Exception as exc:
        logger.warning(f"[Analyzer] Could not send analysis: {exc}")
