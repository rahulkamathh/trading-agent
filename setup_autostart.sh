#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  Auto-start setup for Indian Trading Agent
#  Run this ONCE: bash setup_autostart.sh
#  After that the agent starts automatically on every Mac login.
# ─────────────────────────────────────────────────────────────

# Find the directory this script lives in
AGENT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$(which python3)"
PLIST_NAME="com.tradingagent.app"
PLIST_PATH="$HOME/Library/LaunchAgents/$PLIST_NAME.plist"
LOG_DIR="$AGENT_DIR/logs"

# Create logs directory
mkdir -p "$LOG_DIR"

echo "📁 Agent directory : $AGENT_DIR"
echo "🐍 Python           : $PYTHON"
echo "📋 Launch agent     : $PLIST_PATH"
echo ""

# Write the launchd plist
cat > "$PLIST_PATH" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$PLIST_NAME</string>

  <key>ProgramArguments</key>
  <array>
    <string>$PYTHON</string>
    <string>$AGENT_DIR/app.py</string>
  </array>

  <key>WorkingDirectory</key>
  <string>$AGENT_DIR</string>

  <!-- Auto-restart if it crashes -->
  <key>KeepAlive</key>
  <true/>

  <!-- Start on login -->
  <key>RunAtLoad</key>
  <true/>

  <!-- Log output -->
  <key>StandardOutPath</key>
  <string>$LOG_DIR/agent.log</string>
  <key>StandardErrorPath</key>
  <string>$LOG_DIR/agent_error.log</string>

  <!-- Throttle restarts: wait 10s before restarting after a crash -->
  <key>ThrottleInterval</key>
  <integer>10</integer>
</dict>
</plist>
EOF

# Load it immediately (no need to log out/in)
launchctl unload "$PLIST_PATH" 2>/dev/null
launchctl load -w "$PLIST_PATH"

echo "✅ Done! Trading agent is now running in the background."
echo ""
echo "   Dashboard → http://localhost:5001"
echo "   Logs      → $LOG_DIR/agent.log"
echo ""
echo "   To stop the agent:    launchctl unload ~/Library/LaunchAgents/$PLIST_NAME.plist"
echo "   To start it again:    launchctl load   ~/Library/LaunchAgents/$PLIST_NAME.plist"
echo "   To remove autostart:  rm ~/Library/LaunchAgents/$PLIST_NAME.plist"
