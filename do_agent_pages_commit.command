#!/bin/bash
cd "/Users/rahulkamath/Desktop/indian_trading_agent copy 2"

git config user.email "rahulkamath522@gmail.com"
git config user.name  "Rahul Kamath"

echo "=== Pulling latest ==="
git pull --rebase origin main

echo "=== Staging files ==="
git add app.py templates/dashboard.html

echo "=== Diff stat ==="
git diff --cached --stat

echo "=== Committing ==="
git commit -m "feat: wire all 11 agents into API + add 6 new dashboard pages (Risk, Regime, News, Tax, FII/DII, Execution)"

echo "=== Pushing ==="
git push origin main

echo ""
echo "=== DONE ==="
