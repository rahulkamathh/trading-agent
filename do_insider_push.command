#!/bin/bash
cd "$(dirname "$0")"
git pull --rebase origin main
git add insider_agent.py execution_agent.py
git diff --cached --quiet || git commit -m "feat: insider activity agent + smart execution agent"
git push origin main
echo "DONE — press any key to close"
read -n1
