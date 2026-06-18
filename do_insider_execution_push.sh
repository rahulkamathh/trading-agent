#!/bin/bash
set -e
REPO="/Users/rahulkamath/Desktop/indian_trading_agent copy 2"
cd "$REPO"

echo "=== py_compile checks ==="
python3 -m py_compile "$REPO/insider_agent.py" && echo "insider_agent.py OK"
python3 -m py_compile "$REPO/execution_agent.py" && echo "execution_agent.py OK"

echo "=== /tmp workaround push ==="
cp -r "$REPO" /tmp/agent_insider_exec_push
cd /tmp/agent_insider_exec_push

cp "$REPO/insider_agent.py" .
cp "$REPO/execution_agent.py" .

rm -f .git/index.lock .git/HEAD.lock

git pull --rebase origin main

git add insider_agent.py execution_agent.py

git commit -m "feat: insider activity agent + smart execution agent

- insider_agent.py: InsiderActivityAgent with get_insider_agent() singleton
  * fetch_bulk_deals(): NSE bulk/block deal API with volume-spike fallback
  * get_insider_signals(): ACCUMULATION / DISTRIBUTION / NEUTRAL per ticker
  * get_dashboard_data(): signals, bulk_deals, unusual_volume, summary (60 min cache)

- execution_agent.py: SmartExecutionAgent with get_execution_agent() singleton
  * optimal_entry_window(): 5d 5-min intraday volume profile, best IST windows
  * optimal_position_size(): ATR(14)-based sizing capped at 8% of portfolio
  * suggest_limit_price(): VWAP-anchored limit for BUY/SELL orders
  * get_pre_trade_analysis(): GO / WAIT / NO decision
  * get_dashboard_data(): pre-trade analysis for top-5 BUY signals (15 min cache)
"

git push origin main
echo "=== DONE ==="
