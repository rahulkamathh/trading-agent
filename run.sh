#!/bin/bash
# ─────────────────────────────────────────────
#  Indian Institutional Trading Agent — Launcher
# ─────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "================================================"
echo "  🇮🇳  Indian Institutional Trading Agent"
echo "================================================"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "❌  python3 not found. Please install Python 3.9+."
  exit 1
fi

# Install dependencies if needed
echo "📦  Checking dependencies…"
pip3 install -r requirements.txt -q --break-system-packages 2>/dev/null || \
pip3 install -r requirements.txt -q 2>/dev/null || true

echo ""
echo "✅  Dependencies ready"
echo "🌐  Dashboard → http://localhost:5001"
echo "📄  Mode: Paper Trading (₹10,00,000)"
echo ""
echo "Press Ctrl+C to stop."
echo ""

python3 app.py
