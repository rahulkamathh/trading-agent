#!/bin/bash
cd "/Users/rahulkamath/Desktop/indian_trading_agent copy 2"
rm -f .git/HEAD.lock .git/index.lock

echo "=== stashing ALL local changes (including untracked) ==="
git stash push --include-untracked -m "temp-stash-for-rebase"

echo ""
echo "=== fetching remote ==="
git fetch origin main

echo ""
echo "=== local log (top 4) ==="
git log --oneline -4

echo ""
echo "=== remote log (top 4) ==="
git log --oneline origin/main -4

echo ""
echo "=== rebasing our commit on top of remote ==="
git rebase origin/main

echo ""
echo "=== popping stash ==="
git stash pop

echo ""
echo "=== pushing ==="
git push origin main

echo ""
echo "=== final log ==="
git log --oneline -6

echo ""
echo "=== DONE ==="
