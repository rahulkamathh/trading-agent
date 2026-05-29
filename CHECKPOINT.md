# 🔖 Stable Checkpoint — "kill the ifvg"

**Git tag:** `stable-pre-new-markets`  
**Commit:** `7f05e39`  
**Date:** 2026-05-29  
**Repo:** https://github.com/rahulkamathh/trading-agent  

If Rahul says **"kill the ifvg"**, restore to this state:
```bash
git checkout stable-pre-new-markets
# or to create a fresh branch from it:
git checkout -b recovered-stable stable-pre-new-markets
git push origin recovered-stable
```
Then redeploy on Railway by pushing `recovered-stable` as the new main.

---

## What's in this checkpoint (everything working)

### Core engine
- **engine.py** — full NSE universe (~260 tickers), 4 strategies (momentum, mean-reversion, multi-factor, sector rotation), ATR-based dynamic SL/TP, conviction-scaled RR (1:2 to 1:4), intraday/delivery tax classification
- **learning_engine.py** — self-learning strategy weights, win-rate tracking, dynamic buy threshold
- **risk_manager.py** — dynamic position sizing (0.5% portfolio risk per trade, no hardcoded amounts), EventCalendar (MSCI rebalancing, F&O expiry, RBI MPC, Budget), India VIX monitor, drawdown circuit-breakers (CAUTION/STOP/HALT at -2%/-4%/-6%)

### Telegram intelligence
- **telegram_agent.py** — auto-discovers NSE signal groups, universal signal parser (handles all Indian channel formats: BUY ABOVE, ACCUMULATE, slash targets, arrow targets, price ranges, BTST/positional), group scoring/leaderboard
- **signal_analyzer.py** — TA (RSI, MACD, EMA trend, volume, ATR) + fundamental (P/E, D/E, revenue growth, market cap) analysis on every parsed signal; sends verdict with visual score bars to personal Telegram group
- **notifier.py** — Telegram Bot alerts on every paper trade BUY/SELL; daily closing report to "stock updates" group; optional Twilio SMS for high-conviction trades

### Dashboard (dashboard.html)
- Overview, Charts (TradingView links on click), Portfolio, Positions, Signals, Trades
- Telegram page: Group Leaderboard, Signal Feed, Group Feed (chat bubble UI)
- Controls: Manual buy/sell, Risk Manager panel (macro score, drawdown, events), Notifications setup
- Self-Learning page: strategy weights, threshold history
- Daily Report page

### Infrastructure
- **app.py** — Flask server, all API endpoints including `/api/risk`, `/api/telegram/reparse`
- **mise.toml** — disables GitHub attestation check (fixes Railway build)
- Deployed on Railway at `trading-agent-rahul.up.railway.app`

### Key env vars (set in Railway Variables)
- `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, `TELEGRAM_PHONE`, `TELEGRAM_STRING_SESSION`
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_NOTIFY_CHAT_ID` (= -5283747539, "stock updates" group)
- `TWILIO_*` (optional SMS)

---

## User preferences & working style
- **Rahul Kamath** — rahulkamath522@gmail.com
- Wants everything dynamic, nothing hardcoded (amounts, sizes, thresholds)
- Prefers concise explanations, no excessive bullet-point formatting
- Pushes to GitHub via `/tmp/push-to-github` clone (stale index.lock on main repo prevents direct git ops)
- Railway auto-deploys on push to `main` branch of `rahulkamathh/trading-agent`
- Says "kill the ifvg" → rollback to this checkpoint
