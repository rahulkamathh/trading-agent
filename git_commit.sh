#!/bin/bash
set -e
REPO="/Users/rahulkamath/Desktop/indian_trading_agent copy 2"
cd "$REPO"
rm -f .git/HEAD.lock .git/index.lock
git add risk_agent.py regime_agent.py
git commit -m "feat: portfolio risk agent + regime detection agent"
git push origin main
echo "SUCCESS"
