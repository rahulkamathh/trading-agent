#!/bin/bash
cd "/Users/rahulkamath/Desktop/indian_trading_agent copy 2"
rm -f .git/HEAD.lock .git/index.lock

echo "=== fetch remote ==="
git fetch origin main

echo ""
echo "=== rebase our commit on top of remote ==="
git rebase origin/main

echo ""
echo "=== pushing ==="
git push origin main

echo ""
echo "=== recent log ==="
git log --oneline -6

echo ""
echo "=== DONE ==="
