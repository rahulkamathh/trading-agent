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
            # Module 16 recovery commands
            if cmd in ("status",):
                return "status", None
            if cmd in ("pnl",):
                return "pnl", None
            if cmd in ("positions",):
                return "positions", None
            if cmd in ("orders",):
                return "orders", None
            if cmd in ("reconcile",):
                return "reconcile", None
            if cmd in ("killswitch", "kill"):
                return "killswitch", arg   # arg may be "on" or "off"
            if cmd in ("resume",):
                return "resume", None
            if cmd in ("resume_live",):
                return "resume_live", None
            if cmd in ("mode",):
                return "mode", arg         # arg may be "live" or "paper"
            if cmd in ("audit",):
                return "audit", None
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

        # ── Module 16 recovery commands — no cooldown applied ────────────────
        if cmd == "status":
            self._handle_status(chat_id, msg_id)
            return
        if cmd == "pnl":
            self._handle_pnl(chat_id, msg_id)
            return
        if cmd == "positions":
            self._handle_positions(chat_id, msg_id)
            return
        if cmd == "orders":
            self._handle_orders(chat_id, msg_id)
            return
        if cmd == "reconcile":
            self._handle_reconcile(chat_id, msg_id, user_id)
            return
        if cmd == "killswitch":
            self._handle_killswitch(chat_id, msg_id, arg, user_id)
            return
        if cmd == "resume":
            self._handle_resume(chat_id, msg_id, user_id)
            return
        if cmd == "resume_live":
            self._handle_resume_live(chat_id, msg_id, user_id)
            return
        if cmd == "mode":
            self._handle_mode(chat_id, msg_id, arg, user_id)
            return
        if cmd == "audit":
            self._handle_audit(chat_id, msg_id)
            return

        # Cooldown check (per user) — for analysis commands only
        now = time.time()
        if user_id in self._cooldowns and now - self._cooldowns[user_id] < self.ANALYSIS_COOLDOWN:
            remaining = int(self.ANALYSIS_COOLDOWN - (now - self._cooldowns[user_id]))
            self._send(chat_id, f"⏳ Please wait {remaining}s before another request.", reply_to=msg_id)
            return
        self._cooldowns[user_id] = now

        if cmd == "help":
            self._send(chat_id, (
                "🤖 <b>Trading Agent Bot Commands</b>\n\n"
                "<b>Analysis</b>\n"
                "/analyze <b>TICKER</b> — Full TA + Fundamental analysis\n"
                "/grade <b>TICKER</b> — Quick A/B/C/D grade\n"
                "/pairs — Live forex pair prices\n\n"
                "<b>Portfolio</b>\n"
                "/status — System status + failure level\n"
                "/pnl — Today's P&amp;L summary\n"
                "/positions — Open positions\n"
                "/orders — Pending orders\n\n"
                "<b>Recovery (Module 16)</b>\n"
                "/killswitch on|off — Activate/deactivate kill switch\n"
                "/resume — Resume after RISK failure\n"
                "/resume_live — Re-enable live after CRITICAL failure\n"
                "/reconcile — Sync broker → internal state\n"
                "/mode live|paper — Check or switch trading mode\n"
                "/audit — Last 10 recovery events\n\n"
                "<i>You can also type a stock symbol like <b>RELIANCE</b></i>"
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

    # ── Module 16 recovery command handlers ──────────────────────────────────

    def _handle_status(self, chat_id: int, msg_id: int):
        """Show system status: failure level, mode, P&L, positions."""
        self._send_typing(chat_id)
        try:
            from failure_recovery import get_recovery_manager  # noqa
            rm   = get_recovery_manager()
            st   = rm.status()
            level = st["failure_level"]
            mode  = st["trading_mode"]
            emoji_level = {"NONE": "✅", "SOFT": "⚠️", "RISK": "🚨", "CRITICAL": "🔴"}.get(level, "❓")
            emoji_mode  = "🟢" if mode == "LIVE" else "🟡"

            # Portfolio snapshot
            try:
                from engine import get_agent as _ga
                agent = _ga()
                val   = agent.portfolio.get_total_value()
                cash  = agent.portfolio.available_cash()
                pos   = agent.portfolio.state.get("positions", {})
                initial = agent.portfolio.state.get("initial_capital", 1_000_000)
                pnl   = val - initial
                pnl_str = f"{'+'if pnl>=0 else ''}₹{abs(pnl):,.0f} ({'+'if pnl>=0 else ''}{pnl/initial*100:.1f}%)"
            except Exception:
                val = cash = 0; pos = {}; pnl_str = "—"

            lines = [
                f"{emoji_level} <b>FAILURE LEVEL: {level}</b>",
                f"{emoji_mode} <b>MODE: {mode}</b>",
            ]
            if st["failure_reason"]:
                lines.append(f"Reason: {st['failure_reason'][:120]}")
            lines += [
                "",
                f"Portfolio: ₹{val:,.0f}",
                f"P&amp;L: {pnl_str}",
                f"Cash: ₹{cash:,.0f}",
                f"Open positions: {len(pos)}",
                "",
            ]
            if level == "SOFT":
                lines.append("ℹ️ Auto-retrying every 60s. No action needed.")
            elif level == "RISK":
                lines.append("⚡ Send <code>/resume</code> to unlock trading.")
            elif level == "CRITICAL":
                recon = "✅ Done" if st["reconciled"] else "❌ Required"
                lines.append(f"Reconciliation: {recon}")
                if not st["reconciled"]:
                    lines.append("Step 1: <code>/reconcile</code>")
                    lines.append("Step 2: <code>/resume_live</code>")
                else:
                    lines.append("Send <code>/resume_live</code> to re-enable live.")
            self._send(chat_id, "\n".join(lines), reply_to=msg_id)
        except Exception as e:
            self._send(chat_id, f"⚠️ Status error: {e}", reply_to=msg_id)

    def _handle_pnl(self, chat_id: int, msg_id: int):
        """Today's P&L breakdown."""
        self._send_typing(chat_id)
        try:
            import json as _json
            from pathlib import Path as _P
            from zoneinfo import ZoneInfo
            from datetime import datetime as _dt
            today = _dt.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d")

            eq_pnl = fno_pnl = 0.0
            eq_trades = fno_trades = 0

            tf = _P("data/trade_log.json")
            if tf.exists():
                for t in _json.loads(tf.read_text()):
                    if t.get("action") == "SELL" and (t.get("time") or "").startswith(today):
                        eq_pnl += float(t.get("pnl") or 0)
                        eq_trades += 1

            ff = _P("data/fno_trades.json")
            if ff.exists():
                for t in _json.loads(ff.read_text()):
                    if (t.get("action") in ("SELL","CLOSE") and
                            (t.get("time") or t.get("timestamp") or "").startswith(today)):
                        fno_pnl += float(t.get("pnl") or 0)
                        fno_trades += 1

            total = eq_pnl + fno_pnl
            e = "✅" if total >= 0 else "❌"
            self._send(chat_id, (
                f"{e} <b>Today's P&amp;L — {today}</b>\n\n"
                f"Equity: {'+'if eq_pnl>=0 else ''}₹{eq_pnl:,.0f}  ({eq_trades} trades)\n"
                f"F&amp;O:   {'+'if fno_pnl>=0 else ''}₹{fno_pnl:,.0f}  ({fno_trades} trades)\n"
                f"<b>Total:  {'+'if total>=0 else ''}₹{total:,.0f}</b>"
            ), reply_to=msg_id)
        except Exception as e:
            self._send(chat_id, f"⚠️ P&L error: {e}", reply_to=msg_id)

    def _handle_positions(self, chat_id: int, msg_id: int):
        """List open equity + F&O positions."""
        self._send_typing(chat_id)
        try:
            from engine import get_agent as _ga
            pos = _ga().portfolio.state.get("positions", {})
            if not pos:
                eq_lines = ["No open equity positions."]
            else:
                eq_lines = [f"<b>📈 Equity Positions ({len(pos)})</b>"]
                for ticker, p in list(pos.items())[:15]:
                    sym = ticker.replace(".NS", "")
                    pnl = p.get("unrealised_pnl", 0)
                    eq_lines.append(
                        f"  {sym}  ×{p.get('qty','?')}  "
                        f"avg ₹{p.get('avg_price',0):.1f}  "
                        f"P&amp;L {'+'if pnl>=0 else ''}₹{pnl:,.0f}"
                    )

            fno_lines = []
            try:
                from fno_engine import get_fno_agent as _fga
                fno_pos = _fga().portfolio.state.get("positions", {})
                if fno_pos:
                    fno_lines = [f"\n<b>⚡ F&amp;O Positions ({len(fno_pos)})</b>"]
                    for pid, p in list(fno_pos.items())[:10]:
                        und = p.get("underlying","?").replace(".NS","")
                        fno_lines.append(
                            f"  {und} {p.get('strike','?')}{(p.get('option_type') or '?')[0].upper()}  "
                            f"×{p.get('qty_lots','?')} lots  "
                            f"entry ₹{p.get('entry_premium',0):.2f}"
                        )
            except Exception:
                pass

            self._send(chat_id, "\n".join(eq_lines + fno_lines), reply_to=msg_id)
        except Exception as e:
            self._send(chat_id, f"⚠️ Positions error: {e}", reply_to=msg_id)

    def _handle_orders(self, chat_id: int, msg_id: int):
        """Show pending/live orders from broker (if connected)."""
        self._send_typing(chat_id)
        try:
            from kite_broker import get_broker as _gb  # noqa
            orders = _gb().get_orders() or []
            pending = [o for o in orders if o.get("status") in ("TRIGGER PENDING", "OPEN", "AMO REQ REC")]
            if not pending:
                self._send(chat_id, "📋 No pending orders.", reply_to=msg_id)
                return
            lines = [f"📋 <b>Pending Orders ({len(pending)})</b>"]
            for o in pending[:10]:
                lines.append(
                    f"  {o.get('tradingsymbol','')}  "
                    f"{o.get('transaction_type','')}  "
                    f"×{o.get('quantity','')}  "
                    f"@ ₹{o.get('price',0):.2f}  "
                    f"[{o.get('order_type','')}]"
                )
            self._send(chat_id, "\n".join(lines), reply_to=msg_id)
        except ImportError:
            self._send(chat_id, "📋 Broker not configured — paper mode, no live orders.", reply_to=msg_id)
        except Exception as e:
            self._send(chat_id, f"⚠️ Orders error: {e}", reply_to=msg_id)

    def _handle_reconcile(self, chat_id: int, msg_id: int, user_id: int):
        """Run broker reconciliation."""
        self._send_typing(chat_id)
        self._send(chat_id, "🔍 Running reconciliation — syncing broker → internal state…", reply_to=msg_id)
        try:
            from failure_recovery import get_recovery_manager  # noqa
            report = get_recovery_manager().reconcile()
            overall = report.get("overall", "?")
            mismatches = len(report.get("mismatches", []))
            repaired   = len(report.get("repaired", []))
            emoji = "✅" if overall in ("PASS","PAPER_PASS","REPAIRED") else "❌"
            lines = [
                f"{emoji} <b>Reconciliation: {overall}</b>",
                f"Mismatches: {mismatches}",
                f"Repaired: {repaired}",
            ]
            if report.get("error"):
                lines.append(f"Note: {report['error']}")
            if report.get("repaired"):
                lines.append("\nRepairs made:")
                for r in report["repaired"][:5]:
                    lines.append(f"  • {r}")
            st = get_recovery_manager().status()
            if st["failure_level"] == "CRITICAL":
                lines.append("\n✔ Reconciliation done. Send <code>/resume_live</code> to re-enable live trading.")
            self._send(chat_id, "\n".join(lines), reply_to=msg_id)
        except Exception as e:
            self._send(chat_id, f"⚠️ Reconcile error: {e}", reply_to=msg_id)

    def _handle_killswitch(self, chat_id: int, msg_id: int, arg: Optional[str], user_id: int):
        """Activate or deactivate the daily-loss kill switch."""
        try:
            # Import from app context
            import requests as _req
            base = "http://localhost:5001"
            if arg and arg.upper() in ("ON", "ACTIVATE"):
                r = _req.post(f"{base}/api/kill_switch/activate",
                              json={"reason": f"Manual activation by Telegram user {user_id}"}, timeout=5)
                d = r.json()
                self._send(chat_id, f"🛑 Kill switch <b>ACTIVATED</b>.\n{d.get('reason','')}", reply_to=msg_id)
            elif arg and arg.upper() in ("OFF", "RESET", "DEACTIVATE"):
                r = _req.post(f"{base}/api/kill_switch/reset", timeout=5)
                self._send(chat_id, "✅ Kill switch <b>DEACTIVATED</b>. Trading resumed.", reply_to=msg_id)
            else:
                r = _req.get(f"{base}/api/kill_switch/status", timeout=5)
                d = r.json()
                active = d.get("active", False)
                emoji  = "🛑" if active else "✅"
                state  = "ACTIVE" if active else "INACTIVE"
                reason = d.get("reason", "—")
                self._send(chat_id,
                    f"{emoji} Kill switch: <b>{state}</b>\n"
                    f"Reason: {reason}\n\n"
                    f"Use <code>/killswitch on</code> or <code>/killswitch off</code>",
                    reply_to=msg_id)
        except Exception as e:
            self._send(chat_id, f"⚠️ Kill switch error: {e}", reply_to=msg_id)

    def _handle_resume(self, chat_id: int, msg_id: int, user_id: int):
        """Manual resume after RISK failure."""
        self._send_typing(chat_id)
        try:
            from failure_recovery import get_recovery_manager  # noqa
            ok, msg_text = get_recovery_manager().resume(
                user=str(user_id), method="telegram"
            )
            emoji = "✅" if ok else "❌"
            self._send(chat_id, f"{emoji} {msg_text}", reply_to=msg_id)
        except Exception as e:
            self._send(chat_id, f"⚠️ Resume error: {e}", reply_to=msg_id)

    def _handle_resume_live(self, chat_id: int, msg_id: int, user_id: int):
        """Re-enable live trading after CRITICAL failure."""
        self._send_typing(chat_id)
        try:
            from failure_recovery import get_recovery_manager  # noqa
            ok, msg_text = get_recovery_manager().resume_live(
                user=str(user_id), method="telegram"
            )
            emoji = "✅" if ok else "❌"
            self._send(chat_id, f"{emoji} {msg_text}", reply_to=msg_id)
        except Exception as e:
            self._send(chat_id, f"⚠️ Resume live error: {e}", reply_to=msg_id)

    def _handle_mode(self, chat_id: int, msg_id: int, arg: Optional[str], user_id: int):
        """Show or describe current trading mode."""
        try:
            from failure_recovery import get_recovery_manager  # noqa
            rm  = get_recovery_manager()
            st  = rm.status()
            mode = st["trading_mode"]
            level = st["failure_level"]

            if not arg:
                emoji = "🟢" if mode == "LIVE" else "🟡"
                self._send(chat_id,
                    f"{emoji} <b>Current mode: {mode}</b>\n"
                    f"Failure level: {level}\n\n"
                    f"Modes are set automatically by the recovery framework:\n"
                    f"• LIVE → PAPER on CRITICAL failure\n"
                    f"• PAPER → LIVE via <code>/resume_live</code> after reconciliation",
                    reply_to=msg_id)
            else:
                self._send(chat_id,
                    f"ℹ️ Mode is controlled by the recovery framework.\n"
                    f"Current mode: <b>{mode}</b> (Level: {level})\n"
                    f"To switch back to LIVE: resolve the failure and use <code>/resume_live</code>.",
                    reply_to=msg_id)
        except Exception as e:
            self._send(chat_id, f"⚠️ Mode error: {e}", reply_to=msg_id)

    def _handle_audit(self, chat_id: int, msg_id: int):
        """Show last 10 recovery audit events."""
        try:
            from failure_recovery import get_recovery_manager  # noqa
            entries = get_recovery_manager().audit_log(10)
            if not entries:
                self._send(chat_id, "📋 Recovery audit log is empty.", reply_to=msg_id)
                return
            lines = ["📋 <b>Recovery Audit Log (last 10)</b>\n"]
            for e in reversed(entries):
                ts = e.get("ts", "")[-8:]  # HH:MM:SS
                lines.append(f"  <code>{ts}</code> {e.get('event','')} — {e.get('detail','')[:60]}")
            self._send(chat_id, "\n".join(lines), reply_to=msg_id)
        except Exception as e:
            self._send(chat_id, f"⚠️ Audit error: {e}", reply_to=msg_id)

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
