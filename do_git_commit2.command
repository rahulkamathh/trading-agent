#!/bin/bash
cd "/Users/rahulkamath/Desktop/indian_trading_agent copy 2"
rm -f .git/HEAD.lock .git/index.lock

echo "=== fetching remote ==="
git fetch origin main

echo ""
echo "=== local log (top 3) ==="
git log --oneline -3

echo ""
echo "=== remote log (top 3) ==="
git log --oneline origin/main -3

echo ""
echo "=== rebasing local commit on top of remote ==="
git rebase origin/main

echo ""
echo "=== pushing ==="
git push origin main

echo ""
echo "=== final log ==="
git log --oneline -6

echo ""
echo "=== DONE ==="
