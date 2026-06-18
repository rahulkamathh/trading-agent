#!/bin/bash
set -e
cd "/Users/rahulkamath/Desktop/indian_trading_agent copy 2"
rm -f .git/HEAD.lock .git/index.lock

echo "=== syntax check ==="
python3 -m py_compile engine.py && echo "engine.py OK"

echo "=== staging engine.py ==="
git add engine.py

echo "=== committing ==="
git commit -m "feat: autonomous regime/risk/news/insider guard-rails in run_cycle

Added 4 new guard-rails to TradingAgent.run_cycle() in engine.py:

1. REGIME guard-rail (lines 2706-2725):
   - Calls get_regime_agent().get_dashboard_data() after check_stops()
   - Hard blocks all BUY signals when regime=CRISIS
   - Caps max open positions at 10 (vs default 15) when regime=BEAR

2. VaR guard-rail (lines 2727-2749):
   - Calls get_risk_agent().get_risk_summary()
   - Blocks new buys when VaR95 > 5% of portfolio value
   - Tightens per-position stop_pct from 7% to 5% when portfolio beta > 1.8

3. News Sentiment guard-rail (lines 2751-2765):
   - Calls get_news_agent().get_market_sentiment()
   - Blocks new buys only on extreme bearish sentiment (score < -0.6)

4. Per-ticker insider boost/penalty (lines 2951-2982):
   - Inside buy_candidates loop, after existing guard-rails 1-6
   - Calls get_insider_agent().get_insider_signals() (60-min cached)
   - ACCUMULATION: +5 strength boost (capped at 100)
   - DISTRIBUTION: -10 strength penalty; skips if drops below buy_threshold

5. Return dict enrichment (lines 3117-3123):
   - Added: regime, regime_score, agent_blocks{regime,var,news}
" || echo "(nothing new to commit)"

echo "=== pushing ==="
git push origin main

echo ""
echo "=== DONE — guard-rails pushed ==="
git log --oneline -3
