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

# ── 1. Ensure Python 3 is available ──────────────
if ! command -v python3 &>/dev/null; then
  echo "❌  python3 not found. Install Python 3.9+ and try again."
  exit 1
fi

# ── 2. Create venv if it doesn't exist ───────────
if [ ! -f ".venv/bin/activate" ]; then
  echo "🔧  Creating virtual environment (.venv)…"
  python3 -m venv .venv
fi

# ── 3. Activate venv ─────────────────────────────
# shellcheck disable=SC1091
source .venv/bin/activate

# ── 4. Install / upgrade dependencies ────────────
echo "📦  Checking dependencies…"
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "✅  Dependencies ready"
echo "🌐  Dashboard → http://localhost:5001"
echo "📄  Mode: Paper Trading (₹10,00,000)"
echo ""
echo "Press Ctrl+C to stop."
echo ""

# ── 5. Run ───────────────────────────────────────
python app.py
