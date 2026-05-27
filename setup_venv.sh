#!/bin/bash
# Creates a .venv inside the project so VS Code / Pylance finds all packages
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Creating virtual environment…"
python3 -m venv .venv

echo "Activating and installing dependencies…"
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "✅  Done!  Virtual environment ready at .venv/"
echo ""
echo "Next step in VS Code:"
echo "  1. Cmd+Shift+P → 'Python: Select Interpreter'"
echo "  2. Choose: ./.venv/bin/python"
echo "  (Pylance will now resolve Flask, yfinance, pandas, ta, etc.)"
echo ""
echo "To start the agent with the venv:"
echo "  source .venv/bin/activate && python app.py"
