#!/bin/bash
set -e
cd "/Users/rahulkamath/Desktop/indian_trading_agent copy 2"
rm -f .git/HEAD.lock .git/index.lock
git add fii_agent.py fundamental_agent.py
git commit -m "feat: FII/DII flow agent + fundamental screener agent"
git pull --rebase origin main 2>/dev/null || true
git push origin main
echo "DONE"
