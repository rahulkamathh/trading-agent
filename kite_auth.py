"""
Kite Connect — Automated Daily Authentication
=============================================
Handles the Zerodha Kite Connect OAuth flow automatically each morning.

Flow:
  1. Playwright headless browser navigates to Kite login
  2. Fills user_id + password + TOTP (generated from secret via pyotp)
  3. Captures request_token from OAuth redirect URL
  4. Exchanges request_token for access_token via Kite API
  5. Saves access_token + expiry to data/kite_token.json

Self-healing:
  - If the login page structure changes (new fields, CAPTCHA detected),
    sends a Telegram alert and falls back to manual token entry mode.

Required env vars:
  KITE_API_KEY       — from developers.kite.trade
  KITE_API_SECRET    — from developers.kite.trade
  KITE_USER_ID       — your Zerodha client ID (e.g. AB1234)
  KITE_PASSWORD      — your Zerodha login password
  KITE_TOTP_SECRET   — raw TOTP secret (shown when setting up 2FA)
  KITE_PAPER_MODE    — "true"/"false" (default: "true")

Optional:
  TELEGRAM_BOT_TOKEN — for auth alerts
  TELEGRAM_CHAT_ID   — for auth alerts
"""

import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
TOKEN_FILE = DATA_DIR / "kite_token.json"

# ---------------------------------------------------------------------------
# Config (from env vars)
# ---------------------------------------------------------------------------
KITE_API_KEY     = os.environ.get("KITE_API_KEY", "")
KITE_API_SECRET  = os.environ.get("KITE_API_SECRET", "")
KITE_USER_ID     = os.environ.get("KITE_USER_ID", "")
KITE_PASSWORD    = os.environ.get("KITE_PASSWORD", "")
KITE_TOTP_SECRET = os.environ.get("KITE_TOTP_SECRET", "")
KITE_PAPER_MODE  = os.environ.get("KITE_PAPER_MODE", "true").lower() == "true"

# Login page URL for OAuth
KITE_LOGIN_URL   = f"https://kite.trade/connect/login?api_key={KITE_API_KEY}&v=3"
KITE_TOKEN_URL   = "https://api.kite.trade/session/token"

# ---------------------------------------------------------------------------
# Token store
# ---------------------------------------------------------------------------

def save_token(access_token: str, request_token: str = "") -> None:
    """Persist access token to disk with IST expiry (next 6:30am)."""
    now = datetime.now(IST)
    # Kite tokens expire at 6:00am next day
    if now.hour < 6:
        expiry = now.replace(hour=6, minute=0, second=0, microsecond=0)
    else:
        expiry = (now + timedelta(days=1)).replace(hour=6, minute=0, second=0, microsecond=0)

    token_data = {
        "access_token":  access_token,
        "request_token": request_token,
        "generated_at":  now.isoformat(),
        "expires_at":    expiry.isoformat(),
        "api_key":       KITE_API_KEY,
    }
    with open(TOKEN_FILE, "w") as f:
        json.dump(token_data, f, indent=2)
    logger.info(f"[KiteAuth] Token saved. Expires at {expiry.strftime('%Y-%m-%d %H:%M IST')}")


def load_token() -> dict | None:
    """Load saved token. Returns None if expired or not found."""
    if not TOKEN_FILE.exists():
        return None
    try:
        with open(TOKEN_FILE) as f:
            data = json.load(f)
        expires_at = datetime.fromisoformat(data["expires_at"])
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=IST)
        now = datetime.now(IST)
        if now >= expires_at:
            logger.info("[KiteAuth] Saved token has expired.")
            return None
        return data
    except Exception as e:
        logger.warning(f"[KiteAuth] Could not load token: {e}")
        return None


def get_access_token() -> str | None:
    """Return valid access token if available, else None."""
    data = load_token()
    return data["access_token"] if data else None


def is_token_valid() -> bool:
    return load_token() is not None


def token_info() -> dict:
    """Return token metadata for the dashboard."""
    data = load_token()
    if not data:
        return {
            "valid": False,
            "access_token": None,
            "generated_at": None,
            "expires_at": None,
            "paper_mode": KITE_PAPER_MODE,
            "api_key_configured": bool(KITE_API_KEY),
            "credentials_configured": all([KITE_API_KEY, KITE_API_SECRET, KITE_USER_ID, KITE_PASSWORD, KITE_TOTP_SECRET]),
        }
    return {
        "valid": True,
        "access_token": data["access_token"][:8] + "****",  # masked
        "generated_at": data.get("generated_at"),
        "expires_at": data.get("expires_at"),
        "paper_mode": KITE_PAPER_MODE,
        "api_key_configured": bool(KITE_API_KEY),
        "credentials_configured": True,
    }


# ---------------------------------------------------------------------------
# Checksum helper (Kite requires sha256 of api_key + request_token + api_secret)
# ---------------------------------------------------------------------------

def _compute_checksum(request_token: str) -> str:
    raw = f"{KITE_API_KEY}{request_token}{KITE_API_SECRET}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Exchange request_token for access_token
# ---------------------------------------------------------------------------

def exchange_request_token(request_token: str) -> str | None:
    """
    POST to Kite session/token endpoint to get access_token.
    Returns access_token string or None on failure.
    """
    import requests as req
    checksum = _compute_checksum(request_token)
    try:
        resp = req.post(
            KITE_TOKEN_URL,
            data={
                "api_key":       KITE_API_KEY,
                "request_token": request_token,
                "checksum":      checksum,
            },
            headers={"X-Kite-Version": "3"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "success":
            access_token = data["data"]["access_token"]
            save_token(access_token, request_token)
            return access_token
        else:
            logger.error(f"[KiteAuth] Token exchange failed: {data}")
            return None
    except Exception as e:
        logger.error(f"[KiteAuth] Token exchange error: {e}")
        return None


# ---------------------------------------------------------------------------
# Playwright automated login
# ---------------------------------------------------------------------------

def _send_telegram_alert(msg: str) -> None:
    """Send Telegram alert if configured."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id   = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not bot_token or not chat_id:
        return
    try:
        import requests as req
        req.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception:
        pass


def _generate_totp() -> str:
    """Generate current TOTP code from secret."""
    import pyotp
    return pyotp.TOTP(KITE_TOTP_SECRET).now()


def login_with_playwright() -> str | None:
    """
    Automated Kite login via Playwright headless Chromium.
    Returns request_token on success, None on failure.

    Self-healing: if login page structure changes (unexpected fields,
    CAPTCHA detected), sends Telegram alert and returns None.
    """
    if not all([KITE_API_KEY, KITE_USER_ID, KITE_PASSWORD, KITE_TOTP_SECRET]):
        logger.warning("[KiteAuth] Missing credentials — cannot auto-login")
        return None

    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        logger.error("[KiteAuth] playwright not installed. Run: pip install playwright && playwright install chromium")
        return None

    request_token = None

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()

        try:
            logger.info("[KiteAuth] Navigating to Kite login...")
            page.goto(KITE_LOGIN_URL, wait_until="networkidle", timeout=30_000)
            page.wait_for_timeout(2000)

            # ── Self-healing check: detect CAPTCHA ──────────────────────────
            page_text = page.content()
            if "captcha" in page_text.lower() or "recaptcha" in page_text.lower():
                msg = (
                    "🚨 <b>KAMATH TERMINAL — Kite Auth Alert</b>\n\n"
                    "CAPTCHA detected on Zerodha login page.\n"
                    "Manual login required. Please paste your access_token:\n"
                    f"<code>POST /api/kite/set_token</code>\n\n"
                    f"Time: {datetime.now(IST).strftime('%H:%M IST')}"
                )
                _send_telegram_alert(msg)
                logger.warning("[KiteAuth] CAPTCHA detected — manual login required")
                return None

            # ── Step 1: User ID ──────────────────────────────────────────────
            try:
                user_field = page.locator('input[type="text"]').first
                user_field.fill(KITE_USER_ID)
                page.wait_for_timeout(500)
            except PWTimeout:
                raise RuntimeError("Could not find user ID field")

            # ── Step 2: Password ─────────────────────────────────────────────
            try:
                pass_field = page.locator('input[type="password"]').first
                pass_field.fill(KITE_PASSWORD)
                page.wait_for_timeout(500)
            except PWTimeout:
                raise RuntimeError("Could not find password field")

            # ── Step 3: Submit login form ────────────────────────────────────
            page.keyboard.press("Enter")
            page.wait_for_timeout(2000)

            # ── Step 4: Detect TOTP screen ───────────────────────────────────
            # After login, Kite shows a TOTP input screen
            current_url = page.url
            if "login" in current_url or "signin" in current_url:
                # Check for 2FA input
                totp_inputs = page.locator('input[type="text"], input[type="number"], input[autocomplete="one-time-code"]')
                if totp_inputs.count() > 0:
                    totp_code = _generate_totp()
                    logger.info(f"[KiteAuth] Entering TOTP: {totp_code[:2]}****")
                    totp_inputs.first.fill(totp_code)
                    page.wait_for_timeout(500)
                    page.keyboard.press("Enter")
                    page.wait_for_timeout(3000)
                else:
                    # No TOTP field found — login may have already redirected
                    logger.info("[KiteAuth] No TOTP field — may have already passed 2FA")

            # ── Step 5: Capture request_token from redirect URL ──────────────
            # After successful login, Kite redirects to the app's redirect_uri
            # with ?request_token=XXXX&action=login&status=success
            # We wait for the URL to contain request_token
            for _ in range(15):  # wait up to 15 seconds
                current_url = page.url
                if "request_token=" in current_url:
                    from urllib.parse import urlparse, parse_qs
                    parsed = urlparse(current_url)
                    params = parse_qs(parsed.query)
                    tokens = params.get("request_token", [])
                    if tokens:
                        request_token = tokens[0]
                        logger.info(f"[KiteAuth] Got request_token: {request_token[:8]}****")
                        break
                page.wait_for_timeout(1000)

            if not request_token:
                # Page structure may have changed
                page_content = page.content()[:500]
                logger.error(f"[KiteAuth] Could not capture request_token. URL: {page.url[:200]}")
                msg = (
                    "🚨 <b>KAMATH TERMINAL — Kite Auth Failed</b>\n\n"
                    "Could not capture request_token after login.\n"
                    "Zerodha may have changed their login page.\n"
                    "Manual token entry required:\n"
                    "<code>POST /api/kite/set_token {\"access_token\": \"...\"}</code>\n\n"
                    f"Time: {datetime.now(IST).strftime('%H:%M IST')}"
                )
                _send_telegram_alert(msg)

        except Exception as e:
            logger.error(f"[KiteAuth] Login failed: {e}")
            _send_telegram_alert(
                f"🚨 <b>KAMATH TERMINAL — Kite Auth Error</b>\n\n"
                f"Login error: {e}\n"
                f"Time: {datetime.now(IST).strftime('%H:%M IST')}"
            )
            request_token = None
        finally:
            browser.close()

    return request_token


# ---------------------------------------------------------------------------
# Full auth flow
# ---------------------------------------------------------------------------

def run_auth() -> bool:
    """
    Complete auth flow: login → get request_token → exchange for access_token.
    Returns True on success.
    """
    if not KITE_API_KEY:
        logger.info("[KiteAuth] KITE_API_KEY not configured — skipping auth")
        return False

    # Check if current token is still valid
    if is_token_valid():
        logger.info("[KiteAuth] Existing token is still valid — skipping re-auth")
        return True

    logger.info("[KiteAuth] Starting automated Kite login...")
    request_token = login_with_playwright()

    if not request_token:
        logger.error("[KiteAuth] Failed to get request_token")
        return False

    access_token = exchange_request_token(request_token)
    if access_token:
        logger.info("[KiteAuth] ✅ Auth successful")
        _send_telegram_alert(
            f"✅ <b>KAMATH TERMINAL</b>\n"
            f"Zerodha Kite authenticated successfully.\n"
            f"Paper mode: {'ON' if KITE_PAPER_MODE else '🔴 LIVE TRADING'}\n"
            f"Time: {datetime.now(IST).strftime('%H:%M IST')}"
        )
        return True
    else:
        logger.error("[KiteAuth] Token exchange failed")
        return False


# ---------------------------------------------------------------------------
# Background scheduler: run auth daily at 8:30am IST
# ---------------------------------------------------------------------------

_auth_thread: threading.Thread | None = None
_auth_lock = threading.Lock()


def start_daily_auth_scheduler() -> None:
    """Start a background thread that re-authenticates every morning at 8:30 IST."""
    global _auth_thread

    if not KITE_API_KEY:
        logger.info("[KiteAuth] No API key configured — auth scheduler not started")
        return

    def _scheduler_loop():
        logger.info("[KiteAuth] Daily auth scheduler started")
        while True:
            now = datetime.now(IST)
            # Target: 8:30am IST today or tomorrow
            target = now.replace(hour=8, minute=30, second=0, microsecond=0)
            if now >= target:
                target += timedelta(days=1)

            wait_sec = (target - now).total_seconds()
            logger.info(f"[KiteAuth] Next auth at {target.strftime('%Y-%m-%d %H:%M IST')} (in {wait_sec/3600:.1f}h)")
            time.sleep(wait_sec)

            # Run auth
            success = run_auth()
            if not success:
                # Retry after 5 minutes
                logger.warning("[KiteAuth] Auth failed — retrying in 5 minutes")
                time.sleep(300)
                run_auth()

    with _auth_lock:
        if _auth_thread is None or not _auth_thread.is_alive():
            _auth_thread = threading.Thread(target=_scheduler_loop, daemon=True, name="kite-auth-scheduler")
            _auth_thread.start()


if __name__ == "__main__":
    # Run auth immediately when executed directly
    logging.basicConfig(level=logging.INFO)
    success = run_auth()
    print(f"Auth {'succeeded' if success else 'failed'}")
    if success:
        print(f"Token info: {token_info()}")
