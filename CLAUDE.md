# Indian Institutional Trading Agent

## Project Overview
A Python/Flask paper trading system for the Indian stock market (NSE).
No broker connection — 100% paper trading with ₹10,00,000 starting capital.
Runs 4 institutional-grade strategies on Nifty 50 equities, sector ETFs, and indices.

## How to Run
```bash
# Install deps and launch (one command)
bash run.sh

# Or manually:
pip install -r requirements.txt
python app.py
# Then open http://localhost:5001
```

## Architecture

```
indian_trading_agent/
├── engine.py          # Core trading engine (data, strategies, portfolio)
├── app.py             # Flask REST API + background agent loop
├── templates/
│   └── dashboard.html # Single-file HTML/JS dashboard (Chart.js)
├── data/
│   ├── portfolio.json # Persistent portfolio state (cash, positions)
│   ├── trade_log.json # All executed trades
│   └── signals.json   # Latest generated signals
├── requirements.txt
├── run.sh             # One-click launcher
└── CLAUDE.md          # This file
```

## Key Files & Their Roles

### engine.py
The entire trading brain. Key classes:

- **`DataFetcher`** — yfinance wrapper; fetches NSE data (.NS suffix), caches in memory
  - `DataFetcher.fetch(ticker, period, interval)` → DataFrame
  - `DataFetcher.get_current_price(ticker)` → float
  - `DataFetcher.clear_cache()` — call when you want fresh data

- **`MomentumStrategy`** — Cross-sectional 12-1 month momentum (Jegadeesh-Titman)
- **`MeanReversionStrategy`** — RSI + Bollinger Band oversold/overbought
- **`MultiFactorStrategy`** — Composite of momentum + low-vol + trend quality
- **`SectorRotationStrategy`** — Sector relative strength vs Nifty, rotates ETFs

- **`Portfolio`** — Paper trading engine
  - State stored in `data/portfolio.json`
  - `execute_buy(ticker, price, strategy, reason)` → trade dict or None
  - `execute_sell(ticker, price, reason)` → trade dict or None
  - `check_stops()` — enforces 7% stop-loss and 20% take-profit automatically

- **`SignalAggregator`** — Runs all 4 strategies and saves results to `data/signals.json`
- **`TradingAgent`** — Top-level orchestrator: fetch → signal → execute → stop check
- **`get_agent()`** — Singleton factory; always use this to get the agent instance

### app.py
Flask server on port 5001. Key endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serves the dashboard HTML |
| GET | `/api/dashboard` | All dashboard data in one call |
| GET | `/api/positions` | Open positions |
| GET | `/api/signals` | Latest signals |
| GET | `/api/trades` | Full trade log |
| GET | `/api/market_overview` | Nifty 50 + Bank Nifty quotes |
| POST | `/api/run_cycle` | Trigger a full agent cycle |
| POST | `/api/manual_buy` | `{"ticker": "RELIANCE"}` |
| POST | `/api/manual_sell` | `{"ticker": "RELIANCE.NS"}` |
| POST | `/api/reset` | Reset portfolio to ₹10L |
| POST | `/api/set_interval` | `{"seconds": 900}` — auto-run interval |

### templates/dashboard.html
Standalone single-file app. Fetches all data from the Flask API via `fetch()`.
No build step required. Uses Chart.js from CDN.

## Configuration Constants (engine.py)
```python
INITIAL_CAPITAL  = 1_000_000   # ₹10 lakhs
MAX_POSITION_PCT = 0.08        # 8% max per position
STOP_LOSS_PCT    = 0.07        # 7% stop-loss
TAKE_PROFIT_PCT  = 0.20        # 20% take-profit
MAX_POSITIONS    = 15          # max concurrent positions
```

## Universe
- **Nifty 50 stocks**: `NIFTY50_TICKERS` list (50 tickers, `.NS` suffix)
- **Sector ETFs**: `SECTOR_ETFS` dict — Nifty50, Banking, IT, Pharma, Gold
- **Indices**: `^NSEI` (Nifty 50), `^NSEBANK` (Bank Nifty) — for regime filtering

## Data Source
Yahoo Finance via `yfinance`. All NSE tickers use `.NS` suffix (e.g. `RELIANCE.NS`).
Data history available: ~25 years for most Nifty 50 stocks, ~15 years for ETFs.
Note: NSE was founded in 1994; pre-NSE data is not available digitally.

## Adding a New Strategy
1. Create a class with a `generate_signals(self, data: dict) -> list` method
2. Each signal dict must have: `ticker`, `signal` (BUY/SELL/NEUTRAL), `price`, `score`, `strategy`, `strength` (0-100)
3. Instantiate it in `SignalAggregator.__init__()` and call it in `SignalAggregator.run()`

## Adding a New API Endpoint
Add a Flask route to `app.py` — the agent singleton is available via `get_agent()`.

## Common Development Tasks

```bash
# Run only the engine (no Flask) to test strategies
python3 -c "from engine import get_agent; a = get_agent(); print(a.get_dashboard_data())"

# Reset portfolio from command line
python3 -c "from engine import Portfolio; Portfolio().reset()"

# Check current portfolio
python3 -c "
from engine import get_agent
a = get_agent()
print(f'Value: ₹{a.portfolio.get_total_value():,.0f}')
print(f'Cash:  ₹{a.portfolio.available_cash():,.0f}')
print(f'Positions: {len(a.portfolio.state[\"positions\"])}')
"

# Fetch a stock and view indicators
python3 -c "
from engine import DataFetcher, add_indicators
df = DataFetcher.fetch('RELIANCE.NS', period='1y')
df = add_indicators(df)
print(df[['Close','rsi','ema_200','adx']].tail(5))
"
```

## Dependencies
- `flask` — web server
- `yfinance` — market data (NSE via Yahoo Finance)
- `pandas` / `numpy` — data processing
- `ta` — technical analysis indicators (RSI, Bollinger, ADX, EMA, ATR)
- `requests` — HTTP (used by yfinance internally)

## Notes for Claude Code
- Always use `get_agent()` singleton — never instantiate `TradingAgent` directly
- Portfolio state is persisted to `data/portfolio.json` — don't delete this mid-session
- yfinance rate-limits aggressively; avoid calling `DataFetcher.fetch()` in tight loops
- The dashboard auto-refreshes every 60s; the background thread auto-runs every 15min
- All prices are in INR (₹); quantities are whole shares (no fractional shares)
