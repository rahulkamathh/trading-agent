#!/bin/bash
REPO="/Users/rahulkamath/Desktop/indian_trading_agent copy 2"

echo "=== py_compile checks ==="
python3 -m py_compile "$REPO/insider_agent.py" && echo "insider_agent.py OK"
python3 -m py_compile "$REPO/execution_agent.py" && echo "execution_agent.py OK"

echo "=== /tmp workaround push ==="
# Fresh clone-like copy from the remote to get latest commits
rm -rf /tmp/agent_insider_exec_push2
git clone https://github.com/rahulkamathh/trading-agent.git /tmp/agent_insider_exec_push2

cd /tmp/agent_insider_exec_push2
git config user.email "rahulkamath522@gmail.com"
git config user.name "Rahul Kamath"

# Copy our new files in
cp "$REPO/insider_agent.py" .
cp "$REPO/execution_agent.py" .

git add insider_agent.py execution_agent.py

git commit -m "feat: insider activity agent + smart execution agent

- insider_agent.py: InsiderActivityAgent with get_insider_agent() singleton
  * fetch_bulk_deals(): NSE bulk/block deal API with volume-spike fallback (3x 20d avg)
  * get_insider_signals(): ACCUMULATION / DISTRIBUTION / NEUTRAL per ticker
  * get_dashboard_data(): signals, bulk_deals, unusual_volume, summary (60 min cache)

- execution_agent.py: SmartExecutionAgent with get_execution_agent() singleton
  * optimal_entry_window(): 5d 5-min intraday volume profile, best IST windows
  * optimal_position_size(): ATR(14)-based sizing capped at 8% of portfolio
  * suggest_limit_price(): VWAP-anchored limit for BUY/SELL orders (0.1% offset)
  * get_pre_trade_analysis(): GO / WAIT / NO decision combining all three
  * get_dashboard_data(): pre-trade analysis for top-5 BUY signals (15 min cache)
"

git push origin main
echo ""
echo "=== DONE ==="
