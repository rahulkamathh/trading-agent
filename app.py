"""
Flask API Server — Indian Institutional Trading Agent
Run: python app.py
Dashboard: http://localhost:5001
"""

import threading
import time
import json
from datetime import datetime
from pathlib import Path
from flask import Flask, jsonify, render_template, request, send_from_directory

from engine import get_agent, DataFetcher, INITIAL_CAPITAL, INDEX_TICKERS

app = Flask(__name__, template_folder="templates", static_folder="static")

# Background agent loop state
_agent_running   = False
_last_cycle_info = {}
_auto_interval   = 900   # 15 minutes default
_auto_thread     = None

# ──────────────────────────────────────────────
# HTML Dashboard
# ──────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("templates", "dashboard.html")

# ──────────────────────────────────────────────
# Dashboard Data
# ──────────────────────────────────────────────

@app.route("/api/dashboard")
def api_dashboard():
    try:
        data = get_agent().get_dashboard_data()
        data["agent_running"]    = _agent_running
        data["last_cycle"]       = _last_cycle_info
        data["auto_interval_s"]  = _auto_interval
        return jsonify({"ok": True, "data": data})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/portfolio")
def api_portfolio():
    agent = get_agent()
    return jsonify({
        "ok": True,
        "portfolio": {
            "total_value":    round(agent.portfolio.get_total_value(), 2),
            "cash":           round(agent.portfolio.state["cash"], 2),
            "realised_pnl":   round(agent.portfolio.state.get("realised_pnl", 0), 2),
            "unrealised_pnl": round(agent.portfolio.get_unrealised_pnl(), 2),
        }
    })


@app.route("/api/positions")
def api_positions():
    return jsonify({"ok": True, "positions": get_agent().portfolio.get_positions_display()})


@app.route("/api/trades")
def api_trades():
    from engine import TRADE_LOG_FILE
    trades = []
    if TRADE_LOG_FILE.exists():
        with open(TRADE_LOG_FILE) as f:
            trades = json.load(f)
    return jsonify({"ok": True, "trades": list(reversed(trades))})


@app.route("/api/signals")
def api_signals():
    from engine import SIGNALS_FILE
    if SIGNALS_FILE.exists():
        with open(SIGNALS_FILE) as f:
            data = json.load(f)
        return jsonify({"ok": True, **data})
    return jsonify({"ok": True, "signals": [], "updated_at": ""})


# ──────────────────────────────────────────────
# Agent Controls
# ──────────────────────────────────────────────

@app.route("/api/run_cycle", methods=["POST"])
def api_run_cycle():
    global _agent_running, _last_cycle_info
    if _agent_running:
        return jsonify({"ok": False, "error": "Agent already running"})
    _agent_running = True
    try:
        summary = get_agent().run_cycle()
        _last_cycle_info = summary
        return jsonify({"ok": True, "summary": summary})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        _agent_running = False


@app.route("/api/manual_buy", methods=["POST"])
def api_manual_buy():
    body     = request.get_json(force=True)
    ticker   = body.get("ticker", "").upper()
    if not ticker.endswith(".NS"):
        ticker += ".NS"
    price    = DataFetcher.get_current_price(ticker)
    if price <= 0:
        return jsonify({"ok": False, "error": f"Could not fetch price for {ticker}"}), 400
    trade = get_agent().portfolio.execute_buy(ticker, price, "MANUAL", "Manual buy")
    if trade:
        return jsonify({"ok": True, "trade": trade})
    return jsonify({"ok": False, "error": "Could not execute buy (check capital / position limits)"}), 400


@app.route("/api/manual_sell", methods=["POST"])
def api_manual_sell():
    body   = request.get_json(force=True)
    ticker = body.get("ticker", "")
    price  = DataFetcher.get_current_price(ticker)
    trade  = get_agent().portfolio.execute_sell(ticker, price, "MANUAL")
    if trade:
        return jsonify({"ok": True, "trade": trade})
    return jsonify({"ok": False, "error": f"No position in {ticker}"}), 400


@app.route("/api/reset", methods=["POST"])
def api_reset():
    get_agent().portfolio.reset()
    DataFetcher.clear_cache()
    return jsonify({"ok": True, "message": "Portfolio reset to ₹10,00,000"})


@app.route("/api/set_interval", methods=["POST"])
def api_set_interval():
    global _auto_interval
    body = request.get_json(force=True)
    secs = int(body.get("seconds", 900))
    _auto_interval = max(60, secs)
    return jsonify({"ok": True, "interval": _auto_interval})


# ──────────────────────────────────────────────
# Market Overview
# ──────────────────────────────────────────────

@app.route("/api/market_overview")
def api_market_overview():
    results = {}
    for name, ticker in INDEX_TICKERS.items():
        df = DataFetcher.fetch(ticker, period="5d")
        if not df.empty:
            close = df["Close"].squeeze()
            price   = float(close.iloc[-1])
            prev    = float(close.iloc[-2]) if len(close) > 1 else price
            chg_pct = (price / prev - 1) * 100
            results[name] = {
                "price":   round(price, 2),
                "chg_pct": round(chg_pct, 2),
                "ticker":  ticker,
            }
    return jsonify({"ok": True, "indices": results})


# ──────────────────────────────────────────────
# Auto-run background thread
# ──────────────────────────────────────────────

def _background_loop():
    global _agent_running, _last_cycle_info
    while True:
        time.sleep(_auto_interval)
        if not _agent_running:
            _agent_running = True
            try:
                summary = get_agent().run_cycle()
                _last_cycle_info = summary
            except Exception as e:
                print(f"[BG] Error in auto-cycle: {e}")
            finally:
                _agent_running = False


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Start background auto-run thread
    t = threading.Thread(target=_background_loop, daemon=True)
    t.start()

    print("\n" + "=" * 60)
    print("  🇮🇳  Indian Institutional Trading Agent  —  Paper Mode")
    print("=" * 60)
    print("  Dashboard → http://localhost:5001")
    print(f"  Capital  → ₹{INITIAL_CAPITAL:,.0f}")
    print("  Strategies: Momentum | Mean Reversion | Multi-Factor | Sector Rotation")
    print("=" * 60 + "\n")

    app.run(host="0.0.0.0", port=5001, debug=False)
