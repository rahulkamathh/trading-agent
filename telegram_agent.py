"""
Telegram Signal Intelligence Agent
====================================
Autonomously discovers NSE trading signal groups on Telegram via cold search,
monitors them for signals, parses signals (ticker + direction + targets + SL +
timeline stated in the message), evaluates outcomes after the stated timeline
expires, and scores each group continuously.

High-scoring groups feed into TelegramSignalStrategy in engine.py.

Setup (one-time):
  1. Go to https://my.telegram.org/apps → create an app → copy API ID + Hash
  2. Add to .env:
       TELEGRAM_API_ID=...
       TELEGRAM_API_HASH=...
       TELEGRAM_PHONE=+91xxxxxxxxxx   ← your phone number
  3. First run will send an OTP to your Telegram app. Enter it in the terminal.
     Session is saved to data/telegram_session — no OTP needed after that.
"""

import asyncio
import json
import logging
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from threading import Thread

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR     = Path(__file__).parent / "data"
GROUPS_FILE  = DATA_DIR / "telegram_groups.json"
SIGNALS_FILE = DATA_DIR / "telegram_signals.json"
SESSION_FILE = str(DATA_DIR / "telegram_session")

DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# NSE universe (for ticker matching)
# ---------------------------------------------------------------------------

_NSE_SHORT: dict[str, str] = {}   # "RELIANCE" → "RELIANCE.NS"

def _build_ticker_map():
    global _NSE_SHORT
    try:
        from engine import NIFTY50_TICKERS, SECTOR_ETFS
        all_tickers = list(NIFTY50_TICKERS) + list(SECTOR_ETFS.values())
        _NSE_SHORT = {t.replace(".NS", "").replace("^", ""): t for t in all_tickers}
        # Special aliases
        _NSE_SHORT["NIFTY"]     = "^NSEI"
        _NSE_SHORT["BANKNIFTY"] = "^NSEBANK"
        _NSE_SHORT["NIFTY50"]   = "^NSEI"
    except Exception:
        pass

_build_ticker_map()

# ---------------------------------------------------------------------------
# Discovery keywords
# ---------------------------------------------------------------------------

DISCOVERY_KEYWORDS = [
    "NSE stock signals",
    "Nifty intraday tips",
    "NSE buy sell calls",
    "equity trading signals India",
    "stock tips NSE BSE",
    "swing trade signals NSE",
    "Nifty 50 positional calls",
    "intraday calls NSE today",
    "momentum stocks NSE",
    "NSE technical analysis signals",
    "multibagger stocks NSE",
    "options trading signals NSE",
]

# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

MIN_EVALUATED_FOR_SCORE = 10   # need ≥10 evaluated signals before we trust a group
MIN_WIN_RATE_TO_KEEP    = 0.52 # drop groups below this after MIN_EVALUATED_FOR_SCORE
MAX_ACTIVE_GROUPS       = 40   # keep at most 40 active groups
MAX_GROUPS_TO_JOIN      = 15   # cap per discovery cycle — avoid Telegram flood bans
PROBATION_THRESHOLD     = 5    # promote from probation after 5 evaluated signals
DEFAULT_TIMELINE_DAYS   = 3    # fallback if no timeline found in message


# ===========================================================================
# Signal Parser
# ===========================================================================

class SignalParser:
    """
    Parses raw Telegram message text into a structured signal dict.
    Returns None if no valid NSE signal is found.
    """

    # Direction
    _BUY_RE  = re.compile(
        r'\b(buy|long|bullish|accumulate|entry|ce buy|call buy|go long)\b', re.I)
    _SELL_RE = re.compile(
        r'\b(sell|short|bearish|exit|book\s*profit|put buy|pe buy|go short)\b', re.I)

    # Prices  — handles "@ 2500", "entry: 2,500", "CMP 2500.50", "ltp 2500"
    _ENTRY_RE  = re.compile(
        r'(?:@|entry|cmp|ltp|price|at)[:\s]*(\d[\d,]*(?:\.\d+)?)', re.I)
    _TARGET_RE = re.compile(
        r'(?:tgt|target|tp|t1|t2|t3|t4)[:\s]*(\d[\d,]*(?:\.\d+)?)', re.I)
    _SL_RE     = re.compile(
        r'(?:\bsl\b|stop|stoploss|stop.?loss)[:\s]*(\d[\d,]*(?:\.\d+)?)', re.I)

    # Timeline rules: (pattern, days) — days=None means extract from match groups
    _TIMELINE_RULES = [
        (re.compile(r'\bintraday\b|\bbtst\b|\bsame.?day\b|\btoday\b', re.I), 1),
        (re.compile(r'\b1\s*day\b|\bone\s*day\b', re.I), 1),
        (re.compile(r'\b2\s*days?\b|\btwo\s*days?\b', re.I), 2),
        (re.compile(r'\b3\s*days?\b|\bthree\s*days?\b', re.I), 3),
        (re.compile(r'\b4\s*days?\b|\bfour\s*days?\b', re.I), 4),
        (re.compile(r'\b5\s*days?\b|\bfive\s*days?\b', re.I), 5),
        # "2-3 days", "3 to 5 days" — take the upper bound
        (re.compile(r'\b(\d+)\s*[-–to]+\s*(\d+)\s*days?\b', re.I), None),
        (re.compile(r'\b1\s*week\b|\bone\s*week\b|\bweekly\b', re.I), 7),
        (re.compile(r'\b2\s*weeks?\b|\btwo\s*weeks?\b', re.I), 14),
        (re.compile(r'\b3\s*weeks?\b|\bthree\s*weeks?\b', re.I), 21),
        (re.compile(r'\b1\s*month\b|\bone\s*month\b|\bmonthly\b', re.I), 30),
        (re.compile(r'\b2\s*months?\b|\btwo\s*months?\b', re.I), 60),
        (re.compile(r'\bpositional\b', re.I), 21),
        (re.compile(r'\bswing\b', re.I), 10),
        (re.compile(r'\bshort.?term\b', re.I), 7),
        (re.compile(r'\bmedium.?term\b', re.I), 30),
    ]

    @classmethod
    def parse(cls, text: str, group_id: int, group_title: str, message_id: int) -> dict | None:
        """Parse message. Returns None if no actionable NSE signal found."""
        if not text or len(text.strip()) < 15:
            return None

        text_upper = text.upper()

        # 1. Find ticker
        ticker = cls._find_ticker(text_upper)
        if not ticker:
            return None

        # 2. Direction
        is_buy  = bool(cls._BUY_RE.search(text))
        is_sell = bool(cls._SELL_RE.search(text))
        if not is_buy and not is_sell:
            return None
        # If both found, buy takes priority (common in "buy X, sell Y at …" messages)
        direction = "BUY" if is_buy else "SELL"

        # 3. Prices
        def clean_price(s: str) -> float:
            return float(s.replace(",", ""))

        entry_m  = cls._ENTRY_RE.search(text)
        target_m = cls._TARGET_RE.findall(text)
        sl_m     = cls._SL_RE.search(text)

        entry_price = clean_price(entry_m.group(1)) if entry_m else None
        targets     = [clean_price(t) for t in target_m] if target_m else []
        stop_loss   = clean_price(sl_m.group(1)) if sl_m else None

        # Basic sanity: if we have entry + targets, targets should be > entry for BUY
        if entry_price and targets:
            if direction == "BUY":
                targets = [t for t in targets if t > entry_price * 0.9]
            else:
                targets = [t for t in targets if t < entry_price * 1.1]

        # 4. Specificity score 0–1
        specificity = 0.3   # base: ticker + direction
        if entry_price: specificity += 0.2
        if targets:     specificity += 0.3
        if stop_loss:   specificity += 0.2

        # 5. Timeline
        timeline_raw, timeline_days = cls._find_timeline(text)
        evaluate_at = (datetime.now() + timedelta(days=timeline_days)).isoformat()

        return {
            "id":          str(uuid.uuid4()),
            "group_id":    group_id,
            "group_title": group_title,
            "message_id":  message_id,
            "raw_text":    text[:600],
            "parsed": {
                "ticker":        ticker,
                "direction":     direction,
                "entry_price":   round(entry_price, 2) if entry_price else None,
                "targets":       [round(t, 2) for t in targets],
                "stop_loss":     round(stop_loss, 2) if stop_loss else None,
                "timeline_raw":  timeline_raw,
                "timeline_days": timeline_days,
                "specificity":   round(specificity, 2),
                "evaluate_at":   evaluate_at,
            },
            "received_at": datetime.now().isoformat(),
            "status":      "pending",
            "outcome":     None,
        }

    @classmethod
    def _find_ticker(cls, text_upper: str) -> str | None:
        """Return the first NSE ticker found in the message, or None."""
        best = None
        best_len = 0
        for short, full in _NSE_SHORT.items():
            # Word-boundary match — require at least 2 chars
            if len(short) < 2:
                continue
            pattern = rf'(?<![A-Z]){re.escape(short)}(?![A-Z])'
            if re.search(pattern, text_upper):
                # Prefer longer match to avoid "LT" matching inside "LTBEES"
                if len(short) > best_len:
                    best = full
                    best_len = len(short)
        return best

    @classmethod
    def _find_timeline(cls, text: str) -> tuple[str, int]:
        """Return (raw_text, days). Falls back to DEFAULT_TIMELINE_DAYS."""
        for pattern, days in cls._TIMELINE_RULES:
            m = pattern.search(text)
            if m:
                if days is None:
                    # Range pattern — take the upper bound
                    nums = [int(x) for x in m.groups() if x and x.isdigit()]
                    days = max(nums) if nums else DEFAULT_TIMELINE_DAYS
                return m.group(0).strip(), int(days)
        return "unspecified", DEFAULT_TIMELINE_DAYS


# ===========================================================================
# Group Scorer
# ===========================================================================

class GroupScorer:
    """Scores groups and evaluates pending signal outcomes."""

    @staticmethod
    def compute_score(group: dict) -> float:
        """
        Composite score (0–1):
          50% win rate
          25% avg specificity
          15% activity (signals/week, capped at 10)
          10% freshness (decays over 7 days since last signal)
        """
        evaluated = group.get("signals_evaluated", 0)
        if evaluated < 1:
            return 0.0

        win_rate    = group.get("win_rate") or 0.5
        specificity = group.get("avg_specificity") or 0.3
        spw         = min(group.get("signals_per_week", 0), 10) / 10.0

        last_sig = group.get("last_signal_at")
        if last_sig:
            try:
                days_ago  = (datetime.now() - datetime.fromisoformat(last_sig)).total_seconds() / 86400
                freshness = max(0.0, 1.0 - days_ago / 7.0)
            except Exception:
                freshness = 0.0
        else:
            freshness = 0.0

        score = (0.50 * win_rate +
                 0.25 * specificity +
                 0.15 * spw +
                 0.10 * freshness)
        return round(score, 4)

    @staticmethod
    def should_drop(group: dict) -> bool:
        if group.get("added_manually"):
            return False   # never auto-drop manually added groups
        evaluated = group.get("signals_evaluated", 0)
        if evaluated < MIN_EVALUATED_FOR_SCORE:
            return False
        win_rate = group.get("win_rate") or 1.0
        return win_rate < MIN_WIN_RATE_TO_KEEP

    @staticmethod
    def evaluate_signal(signal: dict) -> dict | None:
        """
        If the signal's timeline has elapsed, fetch current price and compute outcome.
        Returns updated signal dict, or None if not yet time.
        """
        parsed  = signal.get("parsed", {})
        eval_at = parsed.get("evaluate_at")
        if not eval_at:
            return None
        try:
            if datetime.now() < datetime.fromisoformat(eval_at):
                return None
        except Exception:
            return None

        ticker    = parsed.get("ticker")
        direction = parsed.get("direction", "BUY")
        entry     = parsed.get("entry_price")
        targets   = parsed.get("targets", [])
        sl        = parsed.get("stop_loss")

        if not ticker:
            signal["status"] = "eval_error"
            return signal

        try:
            import yfinance as yf
            hist = yf.download(ticker, period="5d", interval="1d",
                               auto_adjust=True, progress=False)
            if isinstance(hist.columns, __import__("pandas").MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            if hist.empty:
                signal["status"] = "eval_error"
                return signal

            current = float(hist["Close"].iloc[-1])
            if not entry:
                entry = float(hist["Close"].iloc[0])

            pnl_pct    = ((current - entry) / entry) * 100
            won        = (direction == "BUY" and pnl_pct > 0) or \
                         (direction == "SELL" and pnl_pct < 0)
            target_hit = (any(current >= t for t in targets) if direction == "BUY"
                          else any(current <= t for t in targets)) if targets else False
            sl_hit     = ((current <= sl) if direction == "BUY"
                          else (current >= sl)) if sl else False

            if target_hit:
                status = "hit_target"
            elif sl_hit:
                status = "hit_sl"
            else:
                status = "expired"

            signal["status"]  = status
            signal["outcome"] = {
                "entry_price":   round(entry, 2),
                "price_at_eval": round(current, 2),
                "pnl_pct":       round(pnl_pct, 2),
                "won":           won,
                "target_hit":    target_hit,
                "sl_hit":        sl_hit,
                "evaluated_at":  datetime.now().isoformat(),
            }
        except Exception as e:
            logger.warning(f"Signal eval error ({ticker}): {e}")
            signal["status"] = "eval_error"

        return signal


# ===========================================================================
# Telegram Agent
# ===========================================================================

class TelegramAgent:
    """
    Background async agent that:
      1. Searches Telegram for NSE signal groups (cold start)
      2. Auto-joins public groups
      3. Listens for new messages and parses signals
      4. Evaluates signal outcomes after the stated timeline
      5. Scores groups and drops underperformers
    """

    def __init__(self):
        self._client   = None
        self._loop     = None
        self._thread   = None
        self._running  = False
        self._configured = False
        self._status   = "not_started"
        self._discovery_running = False
        self._parser   = SignalParser()
        self._scorer   = GroupScorer()
        self._groups   = self._load_groups()
        self._signals  = self._load_signals()

    # ── Persistence ──────────────────────────────────────────────────────── #

    def _load_groups(self) -> dict:
        if GROUPS_FILE.exists():
            try:
                with open(GROUPS_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"groups": {}, "last_discovery_at": None}

    def _save_groups(self):
        with open(GROUPS_FILE, "w") as f:
            json.dump(self._groups, f, indent=2)

    def _load_signals(self) -> list:
        if SIGNALS_FILE.exists():
            try:
                with open(SIGNALS_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save_signals(self):
        # Keep only last 3000 signals
        with open(SIGNALS_FILE, "w") as f:
            json.dump(self._signals[-3000:], f, indent=2)

    # ── Public API ───────────────────────────────────────────────────────── #

    def start(self):
        """Start the Telegram agent. Reads credentials from environment."""
        import os
        api_id   = os.getenv("TELEGRAM_API_ID")
        api_hash = os.getenv("TELEGRAM_API_HASH")
        phone    = os.getenv("TELEGRAM_PHONE")
        if not all([api_id, api_hash, phone]):
            self._status = "not_configured"
            logger.warning(
                "Telegram agent not started — set TELEGRAM_API_ID, "
                "TELEGRAM_API_HASH, TELEGRAM_PHONE in .env"
            )
            return
        self._configured = True
        self._loop   = asyncio.new_event_loop()
        self._thread = Thread(
            target=self._run_loop,
            args=(api_id, api_hash, phone),
            daemon=True,
            name="telegram-agent",
        )
        self._thread.start()
        logger.info("Telegram agent thread started")

    def is_configured(self) -> bool:
        return self._configured

    def get_status(self) -> str:
        return self._status

    def get_groups(self) -> list:
        return sorted(
            self._groups.get("groups", {}).values(),
            key=lambda g: g.get("score", 0),
            reverse=True,
        )

    def get_signals(self, limit: int = 100) -> list:
        return list(reversed(self._signals[-limit:]))

    def get_stats(self) -> dict:
        groups = self._groups.get("groups", {}).values()
        active    = [g for g in groups if g.get("status") == "active"]
        probation = [g for g in groups if g.get("status") == "probation"]
        dropped   = [g for g in groups if g.get("status") == "dropped"]
        evaluated = [s for s in self._signals if s.get("outcome")]
        wins      = [s for s in evaluated if s.get("outcome", {}).get("won")]
        pending   = [s for s in self._signals if s.get("status") == "pending"]
        return {
            "status":            self._status,
            "configured":        self._configured,
            "active_groups":     len(active),
            "probation_groups":  len(probation),
            "dropped_groups":    len(dropped),
            "total_signals":     len(self._signals),
            "pending_signals":   len(pending),
            "evaluated_signals": len(evaluated),
            "overall_win_rate":  round(len(wins) / len(evaluated), 3) if evaluated else None,
            "last_discovery":    self._groups.get("last_discovery_at"),
            "discovery_running": self._discovery_running,
        }

    def add_group_manual(self, identifier: str) -> dict:
        """
        Add a group by @username or t.me invite link.
        identifier: "@channelname" | "https://t.me/channelname" | "https://t.me/joinchat/HASH"
        """
        if not self._configured or not self._loop:
            return {"ok": False, "error": "Telegram agent not running. Set credentials in .env."}
        future = asyncio.run_coroutine_threadsafe(
            self._async_add_manual(identifier), self._loop
        )
        try:
            return future.result(timeout=30)
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def trigger_discovery(self) -> dict:
        """Manually trigger a group discovery cycle."""
        if not self._configured or not self._loop:
            return {"ok": False, "error": "Telegram agent not running."}
        if self._discovery_running:
            return {"ok": False, "error": "Discovery already in progress."}
        asyncio.run_coroutine_threadsafe(self._discover_groups(), self._loop)
        return {"ok": True, "message": "Discovery started in background."}

    # ── Internal async machinery ──────────────────────────────────────────── #

    def _run_loop(self, api_id: str, api_hash: str, phone: str):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._async_main(api_id, api_hash, phone))

    async def _async_main(self, api_id: str, api_hash: str, phone: str):
        try:
            from telethon import TelegramClient, events
        except ImportError:
            self._status = "error: telethon not installed (pip install telethon)"
            logger.error("telethon not installed")
            return

        self._client = TelegramClient(SESSION_FILE, int(api_id), api_hash, loop=self._loop)

        try:
            await self._client.start(phone=phone)
            self._status = "connected"
            self._running = True
            logger.info("Telegram agent: connected")

            # Listen to new messages from monitored groups
            @self._client.on(events.NewMessage)
            async def _on_message(event):
                gid = str(event.chat_id)
                if gid in self._groups.get("groups", {}):
                    await self._handle_message(event)

            # Startup tasks
            await self._discover_groups()
            await self._evaluate_pending_signals()

            # Periodic maintenance loop
            while self._running:
                await asyncio.sleep(3600)           # every hour
                await self._evaluate_pending_signals()
                await self._rescore_all_groups()
                # Re-discover every 6 hours
                last = self._groups.get("last_discovery_at")
                if not last or (
                    (datetime.now() - datetime.fromisoformat(last)).total_seconds() > 21600
                ):
                    await self._discover_groups()

        except Exception as e:
            self._status = f"error: {e}"
            logger.error(f"Telegram agent crashed: {e}", exc_info=True)

    async def _handle_message(self, event):
        try:
            text = event.message.text or ""
            if len(text.strip()) < 15:
                return
            chat  = await event.get_chat()
            gid   = str(event.chat_id)
            title = getattr(chat, "title", gid)

            signal = self._parser.parse(text, event.chat_id, title, event.message.id)
            if signal:
                self._signals.append(signal)
                self._save_signals()
                # Update group's last_signal_at and signal count
                if gid in self._groups.get("groups", {}):
                    g = self._groups["groups"][gid]
                    g["last_signal_at"]  = datetime.now().isoformat()
                    g["signals_tracked"] = g.get("signals_tracked", 0) + 1
                    self._save_groups()
                logger.info(
                    f"Signal: {signal['parsed']['ticker']} "
                    f"{signal['parsed']['direction']} from {title}"
                )
        except Exception as e:
            logger.warning(f"Message handler error: {e}")

    async def _discover_groups(self):
        """Cold-search Telegram for NSE signal groups and auto-join public ones."""
        if self._discovery_running:
            return
        self._discovery_running = True
        self._status = "discovering"
        logger.info("Telegram: starting discovery cycle")
        joined = 0
        try:
            from telethon.tl.functions.contacts import SearchRequest
            from telethon.tl.functions.channels import JoinChannelRequest
            from telethon.errors import FloodWaitError, UserAlreadyParticipantError

            found_chats = {}
            for keyword in DISCOVERY_KEYWORDS:
                try:
                    result = await self._client(SearchRequest(q=keyword, limit=20))
                    for chat in result.chats:
                        cid = str(getattr(chat, "id", ""))
                        if cid and cid not in found_chats:
                            found_chats[cid] = chat
                    await asyncio.sleep(2)      # polite rate-limiting
                except FloodWaitError as e:
                    logger.warning(f"Flood wait {e.seconds}s during discovery")
                    await asyncio.sleep(e.seconds + 5)
                except Exception as e:
                    logger.warning(f"Search error for '{keyword}': {e}")

            existing = self._groups.get("groups", {})
            for cid, chat in found_chats.items():
                if joined >= MAX_GROUPS_TO_JOIN:
                    logger.info(f"Telegram: hit join cap ({MAX_GROUPS_TO_JOIN}), stopping discovery")
                    break
                if cid in existing:
                    continue
                username = getattr(chat, "username", None)
                if not username:
                    continue   # can't join without a username
                try:
                    await self._client(JoinChannelRequest(username))
                    existing[cid] = self._new_group_record(chat, source="search")
                    joined += 1
                    logger.info(f"Joined: {getattr(chat, 'title', username)} (@{username})")
                    await asyncio.sleep(2)   # slightly longer delay to avoid flood ban
                except UserAlreadyParticipantError:
                    existing[cid] = self._new_group_record(chat, source="search")
                except FloodWaitError as e:
                    logger.warning(f"Flood wait {e.seconds}s while joining — stopping discovery")
                    break   # stop joining entirely when Telegram rate-limits us
                except Exception as e:
                    logger.warning(f"Could not join @{username}: {e}")

            self._groups["groups"] = existing
            self._groups["last_discovery_at"] = datetime.now().isoformat()
            self._save_groups()
            logger.info(f"Discovery done — {joined} new groups joined")
        except Exception as e:
            logger.error(f"Discovery error: {e}", exc_info=True)
        finally:
            self._discovery_running = False
            self._status = "connected"

    async def _async_add_manual(self, identifier: str) -> dict:
        """Join a group by @username or t.me invite link and track it."""
        try:
            from telethon.tl.functions.channels import JoinChannelRequest
            from telethon.tl.functions.messages import ImportChatInviteRequest
            from telethon.errors import UserAlreadyParticipantError

            identifier = identifier.strip()

            # Detect invite link vs username
            is_invite = "joinchat" in identifier or (
                "t.me/" in identifier and "+" in identifier.split("t.me/")[-1]
            )

            if is_invite:
                # Extract hash
                hash_ = identifier.split("/")[-1].lstrip("+")
                try:
                    await self._client(ImportChatInviteRequest(hash_))
                except UserAlreadyParticipantError:
                    pass
                entity = await self._client.get_entity(identifier)
            else:
                username = identifier.lstrip("@").replace("https://t.me/", "").strip("/")
                try:
                    await self._client(JoinChannelRequest(username))
                except UserAlreadyParticipantError:
                    pass
                entity = await self._client.get_entity(username)

            gid = str(entity.id)
            record = self._new_group_record(entity, source="manual")
            record["added_manually"] = True
            record["status"]         = "active"   # manually added → trust immediately
            self._groups.setdefault("groups", {})[gid] = record
            self._save_groups()
            return {"ok": True, "title": getattr(entity, "title", identifier), "id": gid}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def _evaluate_pending_signals(self):
        """Evaluate all pending signals whose stated timeline has elapsed."""
        updated = False
        for i, sig in enumerate(self._signals):
            if sig.get("status") != "pending":
                continue
            result = self._scorer.evaluate_signal(sig)
            if result:
                self._signals[i] = result
                updated = True
                gid = str(sig["group_id"])
                if gid in self._groups.get("groups", {}):
                    g = self._groups["groups"][gid]
                    g["signals_evaluated"] = g.get("signals_evaluated", 0) + 1
                    if result.get("outcome", {}).get("won"):
                        g["wins"] = g.get("wins", 0) + 1
                    ev = g["signals_evaluated"]
                    g["win_rate"] = round(g["wins"] / ev, 3) if ev else None
        if updated:
            self._save_signals()
            await self._rescore_all_groups()

    async def _rescore_all_groups(self):
        """Recompute scores, promote from probation, drop underperformers."""
        groups = self._groups.get("groups", {})
        for gid, g in groups.items():
            # Recompute signals_per_week
            group_sigs = [s for s in self._signals if str(s["group_id"]) == gid]
            if group_sigs:
                first_ts = datetime.fromisoformat(group_sigs[0]["received_at"])
                weeks    = max(1, (datetime.now() - first_ts).days / 7)
                g["signals_tracked"] = len(group_sigs)
                g["signals_per_week"] = round(len(group_sigs) / weeks, 1)
                # avg specificity
                g["avg_specificity"] = round(
                    sum(s["parsed"].get("specificity", 0.3) for s in group_sigs)
                    / len(group_sigs), 2
                )
            g["score"] = self._scorer.compute_score(g)

            # Promote from probation
            if g.get("status") == "probation" and \
               g.get("signals_evaluated", 0) >= PROBATION_THRESHOLD:
                g["status"] = "active"
                logger.info(f"Promoted from probation: {g.get('title')} (score={g['score']})")

            # Drop underperformers
            if g.get("status") == "active" and self._scorer.should_drop(g):
                g["status"] = "dropped"
                logger.info(
                    f"Dropped: {g.get('title')} "
                    f"(win_rate={g.get('win_rate')}, evaluated={g.get('signals_evaluated')})"
                )

        self._groups["groups"] = groups
        self._save_groups()

    @staticmethod
    def _new_group_record(chat, source: str = "search") -> dict:
        return {
            "id":                getattr(chat, "id", 0),
            "title":             getattr(chat, "title", "Unknown"),
            "username":          getattr(chat, "username", None),
            "type":              "channel" if getattr(chat, "broadcast", False) else "group",
            "joined_at":         datetime.now().isoformat(),
            "status":            "probation",
            "member_count":      getattr(chat, "participants_count", 0),
            "signals_tracked":   0,
            "signals_evaluated": 0,
            "wins":              0,
            "win_rate":          None,
            "avg_specificity":   0.0,
            "signals_per_week":  0.0,
            "score":             0.0,
            "last_signal_at":    None,
            "source":            source,
            "added_manually":    False,
        }


# ===========================================================================
# Singleton
# ===========================================================================

_telegram_agent: TelegramAgent | None = None


def get_telegram_agent() -> TelegramAgent:
    global _telegram_agent
    if _telegram_agent is None:
        _telegram_agent = TelegramAgent()
    return _telegram_agent
