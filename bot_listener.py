"""
Telegram Bot Command Listener
==============================
Polls the Telegram Bot API for incoming messages and responds to ticker
analysis requests in the "stock updates" group (or any allowed chat).

Supported commands
------------------
  /analyze RELIANCE        → full TA + fundamental analysis + screener.in link
  /a RELIANCE              → shorthand
  /grade RELIANCE          → just the A/B/C/D grade and one-line verdict
  RELIANCE                 → bare ticker (if it's a known NSE symbol)
  /pairs                   → list all tracked forex pairs + live prices
  /help                    → list commands

The bot ignores messages from other bots and only processes messages from
chats listed in ALLOWED_CHAT_IDS (defaults to TELEGRAM_NOTIFY_CHAT_ID).

Runs in a daemon thread — started from app.py on boot.
"""

import logging
import os
import re
import threading
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NSE ticker resolution
# ---------------------------------------------------------------------------

def _build_nse_map() -> dict[str, str]:
    """Map short symbol → yfinance ticker (e.g. "RELIANCE" → "RELIANCE.NS")."""
    try:
        from engine import NIFTY50_TICKERS  # pylint: disable=import-outside-toplevel
        m = {}
        for t in NIFTY50_TICKERS:
            short = t.replace(".NS", "").replace("^", "").upper()
            m[short] = t
        # Index aliases
        m["NIFTY"]     = "^NSEI"
        m["NIFTY50"]   = "^NSEI"
        m["BANKNIFTY"] = "^NSEBANK"
        m["SENSEX"]    = "^BSESN"
        return m
    except Exception:
        return {}


_NSE_MAP: dict[str, str] = {}
_NSE_MAP_LOCK = threading.Lock()


def _get_nse_map() -> dict[str, str]:
    global _NSE_MAP
    with _NSE_MAP_LOCK:
        if not _NSE_MAP:
            _NSE_MAP = _build_nse_map()
    return _NSE_MAP


def _resolve_ticker(raw: str) -> Optional[str]:
    """Return yfinance ticker for a raw symbol, or None if not found."""
    s = raw.upper().strip().replace(".NS", "").replace(".BO", "")
    m = _get_nse_map()
    if s in m:
        return m[s]
    # Fuzzy: try appending .NS directly
    if re.fullmatch(r"[A-Z&]{2,20}", s):
        return s + ".NS"   # let yfinance handle validity
    return None


def _screener_url(ticker: str) -> str:
    """Convert yfinance ticker to screener.in URL."""
    sym = ticker.replace(".NS", "").replace(".BO", "").replace("^", "")
    return f"https://www.screener.in/company/{sym}/"


# ---------------------------------------------------------------------------
# Grade helper (shared with morning digest)
# ---------------------------------------------------------------------------

def grade(ta_score: int, fa_score: int) -> tuple[str, str]:
    """
    Returns (letter_grade, emoji) based on composite TA+FA score.
    Weights: 60% TA, 40% FA (same as signal_analyzer.py).
    """
    composite = ta_score * 0.60 + fa_score * 0.40
    if composite >= 78:  return "A", "🟢"
    if composite >= 62:  return "B", "🔵"
    if composite >= 48:  return "C", "🟡"
    return "D", "🔴"


# ---------------------------------------------------------------------------
# Analysis formatter
# ---------------------------------------------------------------------------

def _format_analysis(ticker: str, raw: str) -> str:
    """Run TA + FA on ticker and return a formatted HTML message."""
    from signal_analyzer import (  # pylint: disable=import-outside-toplevel
        run_technical_analysis, run_fundamental_analysis, _bar, _score_emoji
    )

    ta = run_technical_analysis(ticker, "BUY")   # neutral direction for query
    fa = run_fundamental_analysis(ticker)

    sym = ticker.replace(".NS", "").replace("^", "")
    grade_letter, grade_emoji = grade(ta["score"], fa["score"])
    composite = round(ta["score"] * 0.60 + fa["score"] * 0.40)

    # Current price
    price_line = ""
    try:
        import yfinance as yf  # pylint: disable=import-outside-toplevel
        info  = yf.Ticker(ticker).fast_info
        price = getattr(info, "last_price", None) or getattr(info, "regular_market_price", None)
        if price:
            price_line = f"💰 <b>Price:</b> ₹{price:,.2f}\n"
    except Exception:
        pass

    # Fundamental extras
    fa_extras = ""
    if fa.get("pe"):         fa_extras += f"P/E {fa['pe']:.1f}  "
    if fa.get("revenue_growth") is not None:
        fa_extras += f"Rev Growth {fa['revenue_growth']:.0f}%  "
    if fa.get("debt_to_equity") is not None:
        fa_extras += f"D/E {fa['debt_to_equity']:.1f}×"

    ta_bullets = "\n".join(f"  • {r}" for r in ta["reasons"][:4])
    fa_bullets = "\n".join(f"  • {r}" for r in fa["reasons"][:3])
    if not ta_bullets: ta_bullets = "  • Insufficient data"
    if not fa_bullets: fa_bullets = "  • Insufficient data"

    screener = _screener_url(ticker)

    msg = (
        f"🔍 <b>Analysis: {sym}</b>  {grade_emoji} Grade <b>{grade_letter}</b>\n"
        f"{price_line}"
        f"─────────────────────────\n"
        f"\n📊 <b>Technical</b> {_score_emoji(ta['score'])} {_bar(ta['score'])}\n"
        f"  RSI: {ta['rsi'] or '—'}  |  Trend: {ta.get('trend','—').title()}  |  MACD: {ta.get('macd_signal','—').title()}\n"
        f"{ta_bullets}\n"
        f"\n💼 <b>Fundamental</b> {_score_emoji(fa['score'])} {_bar(fa['score'])}\n"
        f"  {fa_extras or '—'}\n"
        f"{fa_bullets}\n"
        f"\n⚡ <b>Composite</b>  {_score_emoji(composite)} {_bar(composite)}\n"
        f"\n🔗 <a href=\"{screener}\">View on Screener.in</a>"
    )
    return msg


# ---------------------------------------------------------------------------
# Bot API helpers
# ---------------------------------------------------------------------------

class TelegramBotPoller:
    """
    Long-polls the Telegram Bot API for new messages and dispatches commands.
    Thread-safe; safe to run as a daemon thread.
    """

    POLL_TIMEOUT  = 30    # seconds for long poll
    ERROR_SLEEP   = 15    # seconds to wait after an API error
    ANALYSIS_COOLDOWN = 30  # seconds before the same user can request again

    def __init__(self):
        self._token        = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self._allowed_chat = os.environ.get("TELEGRAM_NOTIFY_CHAT_ID", "").strip()
        self._offset       = 0
        self._cooldowns: dict[int, float] = {}   # user_id → last_request_ts
        self._running      = False

    def _api(self, method: str, **kwargs) -> Optional[dict]:
        if not self._token:
            return None
        url = f"https://api.telegram.org/bot{self._token}/{method}"
        try:
            r = requests.post(url, json=kwargs, timeout=self.POLL_TIMEOUT + 5)
            data = r.json()
            return data if data.get("ok") else None
        except Exception as exc:
            logger.debug(f"[BotListener] API error {method}: {exc}")
            return None

    def _send(self, chat_id: int, text: str, reply_to: Optional[int] = None):
        params = {
            "chat_id":    chat_id,
            "text":       text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        if reply_to:
            params["reply_to_message_id"] = reply_to
        self._api("sendMessage", **params)

    def _send_typing(self, chat_id: int):
        self._api("sendChatAction", chat_id=chat_id, action="typing")

    def _parse_command(self, text: str) -> tuple[Optional[str], Optional[str]]:
        """
        Parse message text.
        Returns (command, argument) or (None, None) if not a recognised command.

        Examples:
          "/analyze RELIANCE"  → ("analyze", "RELIANCE")
          "/a HDFCBANK"        → ("analyze", "HDFCBANK")
          "/grade ITC"         → ("grade",   "ITC")
          "/help"              → ("help",    None)
          "/pairs"             → ("pairs",   None)
          "RELIANCE"           → ("analyze", "RELIANCE")  if known ticker
          "RELIANCE analysis"  → ("analyze", "RELIANCE")
        """
        t = text.strip()

        # Slash commands
        m = re.match(r"^/(\w+)(?:\s+(.+))?$", t, re.I)
        if m:
            cmd = m.group(1).lower()
            arg = (m.group(2) or "").strip().upper() or None
            if cmd in ("a", "analyze", "analyse", "check", "stock", "analysis"):
                return "analyze", arg
            if cmd in ("g", "grade"):
                return "grade", arg
            if cmd == "help":
                return "help", None
            if cmd == "pairs":
                return "pairs", None
            return None, None

        # Bare ticker (1–15 uppercase letters, possibly with &)
        bare = re.match(r"^([A-Z&]{2,15})(?:\s+analysis|analysis)?$", t.upper())
        if bare:
            sym = bare.group(1)
            ticker = _resolve_ticker(sym)
            if ticker:
                return "analyze", sym

        return None, None

    def _handle_message(self, msg: dict):
        chat_id    = msg["chat"]["id"]
        msg_id     = msg["message_id"]
        user_id    = msg.get("from", {}).get("id", 0)
        is_bot     = msg.get("from", {}).get("is_bot", False)
        text       = (msg.get("text") or "").strip()

        if is_bot or not text:
            return

        # Only respond to the allowed group (or DMs to the bot)
        chat_type = msg["chat"]["type"]
        if self._allowed_chat:
            allowed_id = str(abs(int(self._allowed_chat)))
            this_id    = str(abs(chat_id))
            if chat_type != "private" and this_id != allowed_id:
                return   # ignore other groups/channels

        cmd, arg = self._parse_command(text)
        if not cmd:
            return

        # Cooldown check (per user)
        now = time.time()
        if user_id in self._cooldowns and now - self._cooldowns[user_id] < self.ANALYSIS_COOLDOWN:
            remaining = int(self.ANALYSIS_COOLDOWN - (now - self._cooldowns[user_id]))
            self._send(chat_id, f"⏳ Please wait {remaining}s before another request.", reply_to=msg_id)
            return
        self._cooldowns[user_id] = now

        if cmd == "help":
            self._send(chat_id, (
                "🤖 <b>Trading Agent Bot Commands</b>\n\n"
                "/analyze <b>TICKER</b> — Full TA + Fundamental analysis with grade &amp; Screener.in link\n"
                "/grade <b>TICKER</b> — Quick A/B/C/D grade only\n"
                "/pairs — Live forex pair prices\n"
                "/help — This message\n\n"
                "<i>You can also just type a stock symbol like <b>RELIANCE</b> or <b>HDFCBANK</b></i>"
            ), reply_to=msg_id)
            return

        if cmd == "pairs":
            self._handle_pairs(chat_id, msg_id)
            return

        if cmd in ("analyze", "grade") and not arg:
            self._send(chat_id, "❓ Please provide a ticker, e.g. <code>/analyze RELIANCE</code>", reply_to=msg_id)
            return

        ticker = _resolve_ticker(arg)
        if not ticker:
            self._send(chat_id, f"❓ Could not find <b>{arg}</b> in NSE universe. Try the full symbol, e.g. <code>RELIANCE</code>", reply_to=msg_id)
            return

        self._send_typing(chat_id)

        if cmd == "grade":
            self._handle_grade(chat_id, msg_id, ticker, arg)
        else:
            self._handle_analyze(chat_id, msg_id, ticker, arg)

    def _handle_analyze(self, chat_id: int, msg_id: int, ticker: str, raw: str):
        try:
            msg = _format_analysis(ticker, raw)
            self._send(chat_id, msg, reply_to=msg_id)
        except Exception as exc:
            logger.warning(f"[BotListener] Analysis failed for {ticker}: {exc}")
            self._send(chat_id, f"⚠️ Could not complete analysis for <b>{raw}</b>. Try again shortly.", reply_to=msg_id)

    def _handle_grade(self, chat_id: int, msg_id: int, ticker: str, raw: str):
        try:
            from signal_analyzer import run_technical_analysis, run_fundamental_analysis  # noqa
            ta = run_technical_analysis(ticker, "BUY")
            fa = run_fundamental_analysis(ticker)
            g, emoji = grade(ta["score"], fa["score"])
            composite = round(ta["score"] * 0.60 + fa["score"] * 0.40)
            sym = ticker.replace(".NS", "")
            verdict = {
                "A": "Strong buy — most indicators aligned ✅",
                "B": "Good setup — worth considering ✅",
                "C": "Weak setup — caution advised ⚠️",
                "D": "Avoid — signals are against entry ❌",
            }[g]
            self._send(chat_id,
                f"{emoji} <b>{sym}</b>  Grade <b>{g}</b>  ({composite}/100)\n"
                f"{verdict}\n"
                f"🔗 <a href=\"{_screener_url(ticker)}\">Screener.in</a>",
                reply_to=msg_id
            )
        except Exception as exc:
            self._send(chat_id, f"⚠️ Could not grade <b>{raw}</b>.", reply_to=msg_id)

    def _handle_pairs(self, chat_id: int, msg_id: int):
        try:
            from forex_engine import FOREX_PAIRS, get_price  # noqa
            lines = ["💱 <b>Live Forex Pairs</b>\n"]
            for pair, meta in FOREX_PAIRS.items():
                price = get_price(pair)
                sym   = pair.replace("=X", "")
                lines.append(f"  <code>{sym:<10}</code> {price:.5f}" if price else f"  <code>{sym:<10}</code> —")
            self._send(chat_id, "\n".join(lines), reply_to=msg_id)
        except Exception as exc:
            self._send(chat_id, "⚠️ Could not fetch forex prices.", reply_to=msg_id)

    def run(self):
        """Main poll loop — blocks forever, call from a daemon thread."""
        if not self._token:
            logger.warning("[BotListener] TELEGRAM_BOT_TOKEN not set — bot commands disabled")
            return

        self._running = True
        logger.info("[BotListener] Starting bot command listener (long-poll)")

        while self._running:
            try:
                data = self._api(
                    "getUpdates",
                    offset=self._offset,
                    timeout=self.POLL_TIMEOUT,
                    allowed_updates=["message"],
                )
                if not data:
                    time.sleep(self.ERROR_SLEEP)
                    continue

                for update in data.get("result", []):
                    self._offset = update["update_id"] + 1
                    msg = update.get("message")
                    if msg:
                        try:
                            self._handle_message(msg)
                        except Exception as exc:
                            logger.warning(f"[BotListener] Handler error: {exc}")

            except Exception as exc:
                logger.warning(f"[BotListener] Poll error: {exc}")
                time.sleep(self.ERROR_SLEEP)


# ---------------------------------------------------------------------------
# Singleton + start helper
# ---------------------------------------------------------------------------

_poller: Optional[TelegramBotPoller] = None


def start_bot_listener() -> None:
    """Start the bot command listener in a daemon thread. Call once from app.py."""
    global _poller
    _poller = TelegramBotPoller()
    t = threading.Thread(target=_poller.run, daemon=True, name="bot-listener")
    t.start()
    logger.info("[BotListener] Bot command listener thread started")
