"""
Trade Notification Engine
==========================
Sends formatted alerts whenever the agent executes a paper trade.

Channels
--------
1. Telegram Bot  — primary (free, instant, richly formatted with MarkdownV2)
2. Twilio SMS    — optional fallback, fires only for high-conviction trades (strength ≥ 80)

Configuration (add to .env / Railway Variables)
------------------------------------------------
# Telegram bot  ← REQUIRED for Telegram alerts
TELEGRAM_BOT_TOKEN=7123456789:AAF...
TELEGRAM_NOTIFY_CHAT_ID=-1001234567890   # group chat_id OR your personal chat_id

# Twilio SMS    ← OPTIONAL, only needed for SMS
TWILIO_ACCOUNT_SID=ACxxxx
TWILIO_AUTH_TOKEN=xxxx
TWILIO_FROM_NUMBER=+1415...
TWILIO_TO_NUMBER=+91980...     # comma-separated for multiple recipients

# Tuning
NOTIFY_SMS_MIN_STRENGTH=80     # minimum signal strength to trigger SMS (default 80)
NOTIFY_ENABLED=true            # set to 'false' to silence all notifications

How to get your Telegram chat_id
---------------------------------
1. Create a bot via @BotFather → copy the token
2. Add the bot to your group (or start a DM)
3. Send any message in the group/DM
4. Visit: https://api.telegram.org/bot<TOKEN>/getUpdates
5. Look for "chat": {"id": ...} in the response — that's your TELEGRAM_NOTIFY_CHAT_ID
"""

import logging
import os
import threading
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
_IST = ZoneInfo("Asia/Kolkata")

# ── Throttle: max 1 notification per ticker per 60 seconds ──────────────────
_last_sent: dict[str, float] = {}
_THROTTLE_SECS = 60
_lock = threading.Lock()


def _throttle_ok(key: str) -> bool:
    import time
    with _lock:
        last = _last_sent.get(key, 0)
        now  = time.time()
        if now - last < _THROTTLE_SECS:
            return False
        _last_sent[key] = now
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# Telegram Bot
# ═══════════════════════════════════════════════════════════════════════════════

class TelegramBotNotifier:
    """
    Sends messages via the Telegram Bot API using plain `requests`.
    No extra dependency needed — requests is already in requirements.txt.
    """

    def __init__(self):
        self._token   = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self._chat_id = os.getenv("TELEGRAM_NOTIFY_CHAT_ID", "").strip()
        self._ok      = bool(self._token and self._chat_id)
        if not self._ok:
            logger.info("[Notifier] Telegram bot not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_NOTIFY_CHAT_ID missing)")

    @property
    def configured(self) -> bool:
        return self._ok

    def send(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message. Returns True on success."""
        if not self._ok:
            return False
        try:
            import requests
            url  = f"https://api.telegram.org/bot{self._token}/sendMessage"
            resp = requests.post(url, json={
                "chat_id":    self._chat_id,
                "text":       text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            }, timeout=10)
            if not resp.ok:
                logger.warning(f"[Notifier] Telegram send failed: {resp.status_code} {resp.text[:200]}")
                return False
            return True
        except Exception as exc:
            logger.warning(f"[Notifier] Telegram send error: {exc}")
            return False

    def send_async(self, text: str):
        """Fire-and-forget in a daemon thread so it never blocks the trade engine."""
        threading.Thread(target=self.send, args=(text,), daemon=True).start()


# ═══════════════════════════════════════════════════════════════════════════════
# Twilio SMS
# ═══════════════════════════════════════════════════════════════════════════════

class TwilioSMSNotifier:
    """
    Sends SMS via Twilio REST API.
    Falls back gracefully if twilio package is not installed or creds missing.
    """

    def __init__(self):
        self._sid   = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
        self._token = os.getenv("TWILIO_AUTH_TOKEN",  "").strip()
        self._from  = os.getenv("TWILIO_FROM_NUMBER", "").strip()
        to_raw      = os.getenv("TWILIO_TO_NUMBER",   "").strip()
        self._to    = [n.strip() for n in to_raw.split(",") if n.strip()] if to_raw else []
        self._ok    = bool(self._sid and self._token and self._from and self._to)
        self._min_strength = int(os.getenv("NOTIFY_SMS_MIN_STRENGTH", "80"))
        if not self._ok:
            logger.info("[Notifier] Twilio SMS not configured (TWILIO_* vars missing)")

    @property
    def configured(self) -> bool:
        return self._ok

    def send(self, body: str, strength: float = 0.0) -> bool:
        """Send SMS to all configured recipients. Returns True if all succeeded."""
        if not self._ok:
            return False
        if strength < self._min_strength:
            return False   # only SMS high-conviction trades
        ok = True
        try:
            from twilio.rest import Client  # type: ignore[import-untyped]
            client = Client(self._sid, self._token)
            for number in self._to:
                try:
                    client.messages.create(body=body[:1600], from_=self._from, to=number)
                    logger.info(f"[Notifier] SMS sent to {number}")
                except Exception as exc:
                    logger.warning(f"[Notifier] SMS failed to {number}: {exc}")
                    ok = False
        except ImportError:
            logger.warning("[Notifier] twilio package not installed — run: pip install twilio")
            ok = False
        except Exception as exc:
            logger.warning(f"[Notifier] Twilio error: {exc}")
            ok = False
        return ok

    def send_async(self, body: str, strength: float = 0.0):
        threading.Thread(target=self.send, args=(body, strength), daemon=True).start()


# ═══════════════════════════════════════════════════════════════════════════════
# Message builders
# ═══════════════════════════════════════════════════════════════════════════════

def _ist_now_str() -> str:
    return datetime.now(_IST).strftime("%d %b %Y, %H:%M IST")


def _rr_label(rr: float | None) -> str:
    if rr is None:
        return "—"
    return f"1:{rr:.1f}" if rr >= 0 else f"{rr:.2f}R"


def _pnl_emoji(pnl: float) -> str:
    if pnl > 0:
        return "✅"
    if pnl < 0:
        return "🔴"
    return "➖"


def build_buy_alert_html(
    ticker:       str,
    price:        float,
    stop_loss:    float | None,
    take_profit:  float | None,
    planned_rr:   float | None,
    strength:     float,
    strategy:     str,
    reason:       str = "",
    source_group: str | None = None,
    qty:          int = 0,
    capital_used: float = 0.0,
) -> str:
    """Rich HTML message for Telegram."""
    short = ticker.replace(".NS", "").replace("^", "")

    sl_str = f"₹{stop_loss:,.2f}" if stop_loss else "—"
    tp_str = f"₹{take_profit:,.2f}" if take_profit else "—"
    rr_str = _rr_label(planned_rr)
    cap_str = f"₹{capital_used:,.0f}" if capital_used else "—"

    # Strength bar (5 blocks)
    filled = round(strength / 20)
    bar    = "█" * filled + "░" * (5 - filled)

    # Strategy emoji
    strat_emoji = {
        "Momentum":      "🚀",
        "MeanReversion": "🔄",
        "MultiFactor":   "🔬",
        "SectorRotation":"🌐",
        "SMA":           "📈",
        "Fibonacci":     "🌀",
        "RSIDivergence": "📊",
        "BollingerSqueeze":"🗜️",
        "VolumeBreakout":"💥",
        "Telegram":      "📡",
        "News":          "📰",
        "Commodity":     "🛢️",
        "Fundamental":   "📋",
        "MANUAL":        "👤",
    }.get(strategy, "⚡")

    lines = [
        f"<b>🟢 PAPER BUY — {short}</b>",
        f"",
        f"💰 <b>Entry</b>  ₹{price:,.2f}",
        f"🛑 <b>Stop Loss</b>  {sl_str}",
        f"🎯 <b>Target</b>  {tp_str}",
        f"⚖️ <b>RR</b>  {rr_str}",
        f"",
        f"{strat_emoji} <b>Strategy</b>  {strategy}",
        f"📊 <b>Strength</b>  {bar} {strength:.0f}/100",
    ]

    if qty:
        lines.append(f"📦 <b>Qty</b>  {qty} shares  ({cap_str})")

    if source_group:
        lines.append(f"📡 <b>Source</b>  {source_group}")

    if reason:
        lines.append(f"💬 <b>Signal</b>  <i>{reason[:120]}</i>")

    lines += [
        f"",
        f"🕐 {_ist_now_str()}",
        f"<i>Paper trade — not real money</i>",
    ]

    return "\n".join(lines)


def build_sell_alert_html(
    ticker:      str,
    price:       float,
    avg_price:   float,
    pnl:         float,
    pnl_pct:     float,
    actual_rr:   float | None,
    planned_rr:  float | None,
    strategy:    str,
    hold_days:   int = 0,
    trade_type:  str = "INTRADAY",
    reason:      str = "",
    qty:         int = 0,
) -> str:
    short     = ticker.replace(".NS", "").replace("^", "")
    emoji     = _pnl_emoji(pnl)
    rr_actual = _rr_label(actual_rr)
    rr_plan   = _rr_label(planned_rr)
    hold_str  = "Intraday" if hold_days == 0 else f"{hold_days}d"

    tax_note  = {"INTRADAY": "Intraday (speculative)", "STCG": "STCG @20%", "LTCG": "LTCG @12.5%"}.get(trade_type, trade_type)

    lines = [
        f"{emoji} <b>PAPER SELL — {short}</b>",
        f"",
        f"📤 <b>Exit</b>  ₹{price:,.2f}   (entry ₹{avg_price:,.2f})",
        f"💵 <b>P&amp;L</b>  ₹{pnl:+,.0f}  ({pnl_pct:+.2f}%)",
        f"⚖️ <b>RR achieved</b>  {rr_actual}  (planned {rr_plan})",
        f"⏱ <b>Hold</b>  {hold_str}  ·  {tax_note}",
    ]

    if qty:
        lines.append(f"📦 <b>Qty</b>  {qty} shares")

    if reason:
        lines.append(f"💬 <b>Reason</b>  <i>{reason[:80]}</i>")

    lines += [
        f"",
        f"🕐 {_ist_now_str()}",
        f"<i>Paper trade — not real money</i>",
    ]

    return "\n".join(lines)


def build_buy_alert_sms(
    ticker: str, price: float, stop_loss: float | None,
    take_profit: float | None, planned_rr: float | None,
    strength: float, strategy: str,
) -> str:
    """Compact plain-text SMS (≤160 chars preferred)."""
    short  = ticker.replace(".NS", "").replace("^", "")
    sl_str = f"SL:{stop_loss:.0f}" if stop_loss else ""
    tp_str = f"TP:{take_profit:.0f}" if take_profit else ""
    rr_str = f"RR 1:{planned_rr:.1f}" if planned_rr else ""
    parts  = [p for p in [sl_str, tp_str, rr_str] if p]
    detail = " | ".join(parts)
    return f"[PAPER BUY] {short} @{price:.0f} {detail} | Str:{strength:.0f} | {strategy} | {_ist_now_str()}"


def build_sell_alert_sms(
    ticker: str, price: float, pnl: float, pnl_pct: float,
    actual_rr: float | None, hold_days: int,
) -> str:
    short    = ticker.replace(".NS", "").replace("^", "")
    rr_str   = f"RR:{actual_rr:.2f}" if actual_rr else ""
    hold_str = "Intraday" if hold_days == 0 else f"{hold_days}d"
    return f"[PAPER SELL] {short} @{price:.0f} P&L:{pnl:+.0f} ({pnl_pct:+.1f}%) {rr_str} | Hold:{hold_str} | {_ist_now_str()}"


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton facade
# ═══════════════════════════════════════════════════════════════════════════════

class Notifier:
    """
    Single entry point.  Call once at startup via get_notifier().
    """

    def __init__(self):
        self._enabled = os.getenv("NOTIFY_ENABLED", "true").lower() not in ("false", "0", "no")
        self._tg  = TelegramBotNotifier()
        self._sms = TwilioSMSNotifier()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def get_status(self) -> dict:
        return {
            "enabled":             self._enabled,
            "telegram_configured": self._tg.configured,
            "sms_configured":      self._sms.configured,
            "sms_min_strength":    self._sms._min_strength,
            "telegram_chat_id":    self._tg._chat_id or None,
        }

    def notify_buy(
        self,
        ticker:       str,
        price:        float,
        stop_loss:    float | None = None,
        take_profit:  float | None = None,
        planned_rr:   float | None = None,
        strength:     float = 65.0,
        strategy:     str   = "",
        reason:       str   = "",
        source_group: str | None = None,
        qty:          int   = 0,
        capital_used: float = 0.0,
    ):
        if not self._enabled:
            return
        if not _throttle_ok(f"buy_{ticker}"):
            return

        html = build_buy_alert_html(
            ticker, price, stop_loss, take_profit, planned_rr,
            strength, strategy, reason, source_group, qty, capital_used,
        )
        sms  = build_buy_alert_sms(ticker, price, stop_loss, take_profit, planned_rr, strength, strategy)

        self._tg.send_async(html)
        self._sms.send_async(sms, strength=strength)
        logger.info(f"[Notifier] BUY alert fired → {ticker} @ ₹{price:.2f}")

    def notify_sell(
        self,
        ticker:     str,
        price:      float,
        avg_price:  float,
        pnl:        float,
        pnl_pct:    float,
        actual_rr:  float | None = None,
        planned_rr: float | None = None,
        strategy:   str   = "",
        hold_days:  int   = 0,
        trade_type: str   = "INTRADAY",
        reason:     str   = "",
        qty:        int   = 0,
        strength:   float = 0.0,
    ):
        if not self._enabled:
            return
        if not _throttle_ok(f"sell_{ticker}"):
            return

        html = build_sell_alert_html(
            ticker, price, avg_price, pnl, pnl_pct,
            actual_rr, planned_rr, strategy, hold_days, trade_type, reason, qty,
        )
        sms  = build_sell_alert_sms(ticker, price, pnl, pnl_pct, actual_rr, hold_days)

        self._tg.send_async(html)
        # SMS on sell only if it was a win (positive P&L) with meaningful strength
        if pnl > 0:
            self._sms.send_async(sms, strength=strength)
        logger.info(f"[Notifier] SELL alert fired → {ticker} @ ₹{price:.2f}  P&L ₹{pnl:+,.0f}")

    def send_test(self) -> dict:
        """Send a test alert to all configured channels. Returns results dict."""
        test_msg_html = (
            "<b>🔔 Trading Agent — Test Alert</b>\n\n"
            "✅ Telegram bot is connected and working!\n"
            "This is a test from your NSE Paper Trading Agent.\n\n"
            f"🕐 {_ist_now_str()}\n"
            "<i>Paper trade — not real money</i>"
        )
        test_msg_sms = f"[TRADING AGENT] Test alert from your NSE Paper Trading Agent. {_ist_now_str()}"

        tg_ok  = self._tg.send(test_msg_html)
        sms_ok = self._sms.send(test_msg_sms, strength=999)  # bypass strength gate for test

        return {
            "telegram": {"sent": tg_ok,  "configured": self._tg.configured},
            "sms":      {"sent": sms_ok, "configured": self._sms.configured},
        }

    def send_closing_report(
        self,
        date:             str,
        day_pnl:          float,
        portfolio_value:  float,
        initial_capital:  float,
        today_trades:     list,        # list of trade dicts from trade_log
        open_positions:   list,        # list of position dicts
        win_rate:         float | None = None,
    ):
        """
        Full end-of-day closing report — mirrors the dashboard Daily Report page.
        Sent automatically at 15:30 IST. Splits into multiple messages if too long.
        """
        if not self._enabled:
            return

        total_pnl_pct = round((portfolio_value / initial_capital - 1) * 100, 2)
        day_emoji     = "📈" if day_pnl >= 0 else "📉"
        total_emoji   = "🟢" if portfolio_value >= initial_capital else "🔴"
        wr_str        = f"{win_rate * 100:.0f}%" if win_rate is not None else "—"
        sells_today   = [t for t in today_trades if t.get("action") == "SELL"]
        buys_today    = [t for t in today_trades if t.get("action") == "BUY"]

        # ── Header ───────────────────────────────────────────────────────────
        header = [
            f"{day_emoji} <b>Closing Report — {date}</b>",
            f"",
            f"{total_emoji} <b>Portfolio</b>  ₹{portfolio_value:,.0f}  ({total_pnl_pct:+.2f}% all-time)",
            f"💵 <b>Day P&amp;L</b>  ₹{day_pnl:+,.0f}",
            f"🎯 <b>Win rate today</b>  {wr_str}",
            f"🔢 <b>Trades</b>  {len(sells_today)} closed · {len(buys_today)} opened · {len(open_positions)} still open",
            f"",
        ]

        # ── Closed trades (sells) ─────────────────────────────────────────
        trade_lines = []
        if sells_today:
            trade_lines.append("<b>📋 Today's closed trades:</b>")
            for t in sells_today:
                short  = (t.get("ticker") or "").replace(".NS", "").replace("^", "")
                pnl    = t.get("pnl") or 0
                rr     = t.get("actual_rr")
                hdays  = t.get("hold_days", 0)
                ttype  = t.get("trade_type", "")
                emoji  = "✅" if pnl > 0 else "❌"
                rr_str = f" · RR {rr:+.2f}" if rr is not None else ""
                hold   = f"{hdays}d" if hdays else "intraday"
                tax    = {"INTRADAY": "Int", "STCG": "STCG", "LTCG": "LTCG"}.get(ttype, "")
                trade_lines.append(
                    f"  {emoji} <b>{short}</b>  ₹{pnl:+,.0f}{rr_str}  <i>({hold} · {tax})</i>"
                )

        # ── Open positions ────────────────────────────────────────────────
        pos_lines = []
        if open_positions:
            pos_lines.append("")
            pos_lines.append("<b>📂 Open positions carried forward:</b>")
            for p in open_positions[:8]:   # cap at 8 to keep message readable
                short   = (p.get("ticker") or "").replace(".NS", "").replace("^", "")
                pnl_pct = p.get("pnl_pct", 0)
                sl      = p.get("stop_loss")
                tp      = p.get("target")
                e       = "🟢" if pnl_pct >= 0 else "🔴"
                sl_str  = f"  SL ₹{sl:,.0f}" if sl else ""
                tp_str  = f"  TP ₹{tp:,.0f}" if tp else ""
                pos_lines.append(f"  {e} <b>{short}</b>  {pnl_pct:+.1f}%{sl_str}{tp_str}")
            if len(open_positions) > 8:
                pos_lines.append(f"  … +{len(open_positions) - 8} more")

        footer = ["", f"🕐 {_ist_now_str()}", "<i>Paper trade — not real money</i>"]

        # Telegram message limit is 4096 chars — split if needed
        body = "\n".join(header + trade_lines + pos_lines + footer)
        if len(body) <= 4000:
            self._tg.send_async(body)
        else:
            # Send header + trades first, then positions
            self._tg.send_async("\n".join(header + trade_lines + footer))
            if pos_lines:
                self._tg.send_async("\n".join(pos_lines + footer))

        logger.info(f"[Notifier] Closing report fired for {date}")

    # Keep old name as alias for backward compatibility
    def send_daily_summary(self, date, day_pnl, trades_today, win_rate,
                           portfolio_value, top_win="", top_loss=""):
        """Thin wrapper kept for backward compat — prefer send_closing_report."""
        emoji  = "📈" if day_pnl >= 0 else "📉"
        wr_str = f"{win_rate * 100:.0f}%" if win_rate is not None else "—"
        lines  = [
            f"{emoji} <b>Daily Summary — {date}</b>", "",
            f"💵 <b>Day P&amp;L</b>  ₹{day_pnl:+,.0f}",
            f"📊 <b>Portfolio</b>  ₹{portfolio_value:,.0f}",
            f"🔢 <b>Trades today</b>  {trades_today}",
            f"🎯 <b>Win rate</b>  {wr_str}",
        ]
        if top_win:  lines.append(f"🏆 <b>Best</b>  {top_win}")
        if top_loss: lines.append(f"⚠️ <b>Worst</b>  {top_loss}")
        lines += ["", f"🕐 {_ist_now_str()}", "<i>Paper trade — not real money</i>"]
        self._tg.send_async("\n".join(lines))
        logger.info(f"[Notifier] Daily summary fired for {date}")


# ── Singleton ────────────────────────────────────────────────────────────────

_notifier: Notifier | None = None


def get_notifier() -> Notifier:
    global _notifier  # noqa: PLW0603
    if _notifier is None:
        _notifier = Notifier()
    return _notifier
