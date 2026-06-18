"""
Daily Report Agent
==================
Generates end-of-day portfolio reports for the Kamath Terminal.
Reads trade_log.json and portfolio.json; fetches live market data from yfinance.
"""

import json
import os
import threading
import logging
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo

import yfinance as yf

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = DATA_DIR / "reports"
TRADE_LOG_PATH = DATA_DIR / "trade_log.json"
PORTFOLIO_PATH = DATA_DIR / "portfolio.json"
SIGNALS_PATH = DATA_DIR / "signals.json"


def _load_json(path: Path, default):
    try:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
    return default


def _now_ist() -> datetime:
    return datetime.now(IST)


class DailyReportAgent:
    def __init__(self):
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Market data helpers
    # ------------------------------------------------------------------
    def _fetch_market_data(self) -> dict:
        """Fetch Nifty 50, Bank Nifty, and India VIX from yfinance."""
        result = {
            "nifty_change_pct": 0.0,
            "banknifty_change_pct": 0.0,
            "vix": 0.0,
        }
        try:
            tickers = yf.download(
                "^NSEI ^NSEBANK ^INDIAVIX",
                period="2d",
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if tickers.empty:
                return result

            close = tickers["Close"]
            for col, key in [("^NSEI", "nifty_change_pct"), ("^NSEBANK", "banknifty_change_pct")]:
                if col in close.columns and len(close[col].dropna()) >= 2:
                    vals = close[col].dropna().values
                    pct = round(float((vals[-1] - vals[-2]) / vals[-2] * 100), 2)
                    result[key] = pct

            if "^INDIAVIX" in close.columns:
                vix_vals = close["^INDIAVIX"].dropna().values
                if len(vix_vals) >= 1:
                    result["vix"] = round(float(vix_vals[-1]), 1)
        except Exception as e:
            logger.warning(f"Market data fetch error: {e}")
        return result

    # ------------------------------------------------------------------
    # Core report generation
    # ------------------------------------------------------------------
    def generate_report(self, date_str: str = None) -> dict:
        """
        Generate EOD report for the given date (default: today IST).
        Returns a dict matching the documented schema.
        """
        if date_str is None:
            date_str = _now_ist().strftime("%Y-%m-%d")

        generated_at = _now_ist().isoformat(timespec="seconds")

        trade_log: list = _load_json(TRADE_LOG_PATH, [])
        portfolio: dict = _load_json(PORTFOLIO_PATH, {})
        signals_data: dict = _load_json(SIGNALS_PATH, {})

        # ── Portfolio snapshot ────────────────────────────────────────
        cash = float(portfolio.get("cash", 0))
        initial = float(portfolio.get("initial", 1_000_000))
        positions = portfolio.get("positions", {})
        realised_pnl = float(portfolio.get("realised_pnl", 0))

        # Calculate invested value and unrealised P&L from open positions
        invested = 0.0
        unrealised_pnl = 0.0
        position_details = []
        positions_in_profit = 0
        positions_in_loss = 0
        largest_position_value = 0.0

        for ticker, pos in positions.items():
            entry_price = float(pos.get("avg_price", pos.get("price", 0)))
            qty = int(pos.get("qty", 0))
            # Try to get current price from yfinance quickly
            try:
                info = yf.Ticker(ticker).fast_info
                current_price = float(info.get("last_price", entry_price) or entry_price)
            except Exception:
                current_price = entry_price

            cost = entry_price * qty
            mkt_value = current_price * qty
            pos_pnl = mkt_value - cost
            pos_pnl_pct = (pos_pnl / cost * 100) if cost > 0 else 0.0

            invested += mkt_value
            unrealised_pnl += pos_pnl
            largest_position_value = max(largest_position_value, mkt_value)

            if pos_pnl >= 0:
                positions_in_profit += 1
            else:
                positions_in_loss += 1

            position_details.append({
                "ticker": ticker,
                "qty": qty,
                "entry_price": entry_price,
                "current_price": current_price,
                "pnl": round(pos_pnl, 2),
                "pnl_pct": round(pos_pnl_pct, 2),
            })

        total_value = cash + invested
        total_pnl = total_value - initial
        total_pnl_pct = round((total_pnl / initial * 100), 2) if initial else 0.0

        # Daily P&L: use day_start_value if available
        day_start_value = float(portfolio.get("day_start_value", initial))
        daily_pnl = total_value - day_start_value
        daily_pnl_pct = round((daily_pnl / day_start_value * 100), 2) if day_start_value else 0.0

        largest_position_pct = round((largest_position_value / total_value * 100), 2) if total_value else 0.0
        cash_pct = round((cash / total_value * 100), 2) if total_value else 0.0

        portfolio_snapshot = {
            "total_value": round(total_value),
            "cash": round(cash),
            "invested": round(invested),
            "open_positions": len(positions),
            "daily_pnl": round(daily_pnl),
            "daily_pnl_pct": daily_pnl_pct,
            "total_pnl": round(total_pnl),
            "total_pnl_pct": total_pnl_pct,
        }

        # ── Today's trades ────────────────────────────────────────────
        todays_trades = [
            {
                "ticker": t.get("ticker", ""),
                "action": t.get("action", ""),
                "qty": t.get("qty", 0),
                "price": t.get("price", 0),
                "strategy": t.get("strategy", ""),
            }
            for t in trade_log
            if str(t.get("time", "")).startswith(date_str)
        ]

        # ── Strategy performance (all-time from trade_log) ────────────
        strategy_perf: dict = {}
        for t in trade_log:
            strat = t.get("strategy", "UNKNOWN")
            if strat not in strategy_perf:
                strategy_perf[strat] = {"trades": 0, "wins": 0, "total_pnl": 0.0}
            strategy_perf[strat]["trades"] += 1
            pnl = t.get("pnl")
            if pnl is not None:
                strategy_perf[strat]["total_pnl"] += float(pnl)
                if float(pnl) > 0:
                    strategy_perf[strat]["wins"] += 1

        strategy_performance = {}
        for strat, data in strategy_perf.items():
            win_rate = round(data["wins"] / data["trades"] * 100) if data["trades"] > 0 else 0
            strategy_performance[strat] = {
                "trades": data["trades"],
                "win_rate": win_rate,
                "total_pnl": round(data["total_pnl"], 2),
            }

        # ── Top 3 gainers / losers from open positions ────────────────
        sorted_positions = sorted(position_details, key=lambda x: x["pnl_pct"], reverse=True)
        top_3_gainers = [
            {"ticker": p["ticker"], "pnl_pct": p["pnl_pct"], "pnl": p["pnl"]}
            for p in sorted_positions[:3]
            if p["pnl_pct"] > 0
        ]
        top_3_losers = [
            {"ticker": p["ticker"], "pnl_pct": p["pnl_pct"], "pnl": p["pnl"]}
            for p in reversed(sorted_positions[-3:])
            if p["pnl_pct"] < 0
        ]

        # ── Risk metrics ──────────────────────────────────────────────
        risk_metrics = {
            "cash_pct": cash_pct,
            "largest_position_pct": largest_position_pct,
            "positions_in_loss": positions_in_loss,
            "positions_in_profit": positions_in_profit,
        }

        # ── Market summary ────────────────────────────────────────────
        market_summary = self._fetch_market_data()

        # ── Tomorrow's signals ────────────────────────────────────────
        signals_list = signals_data.get("signals", []) if isinstance(signals_data, dict) else signals_data
        if not isinstance(signals_list, list):
            signals_list = []

        buy_signals = [s for s in signals_list if s.get("signal") == "BUY"]
        strong_buys = [s for s in buy_signals if float(s.get("strength", 0)) > 85]
        top_5 = sorted(buy_signals, key=lambda x: float(x.get("strength", 0)), reverse=True)[:5]
        top_5_signals = [
            {
                "ticker": s.get("ticker", ""),
                "strategy": s.get("strategy", ""),
                "strength": s.get("strength", 0),
                "score": round(float(s.get("score", 0)), 3),
            }
            for s in top_5
        ]

        signals_tomorrow = {
            "strong_buys": len(strong_buys),
            "buys": len(buy_signals),
            "top_5_signals": top_5_signals,
        }

        # ── Formatted text ────────────────────────────────────────────
        formatted_text = self._format_telegram_message(
            date_str=date_str,
            portfolio_snapshot=portfolio_snapshot,
            todays_trades=todays_trades,
            top_3_gainers=top_3_gainers,
            top_3_losers=top_3_losers,
            market_summary=market_summary,
            signals_tomorrow=signals_tomorrow,
        )

        report = {
            "date": date_str,
            "generated_at": generated_at,
            "portfolio_snapshot": portfolio_snapshot,
            "todays_trades": todays_trades,
            "strategy_performance": strategy_performance,
            "top_3_gainers": top_3_gainers,
            "top_3_losers": top_3_losers,
            "risk_metrics": risk_metrics,
            "market_summary": market_summary,
            "signals_tomorrow": signals_tomorrow,
            "formatted_text": formatted_text,
        }

        return report

    # ------------------------------------------------------------------
    # Telegram-ready formatter
    # ------------------------------------------------------------------
    def _format_telegram_message(
        self,
        date_str: str,
        portfolio_snapshot: dict,
        todays_trades: list,
        top_3_gainers: list,
        top_3_losers: list,
        market_summary: dict,
        signals_tomorrow: dict,
    ) -> str:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            display_date = dt.strftime("%-d %b %Y")
        except Exception:
            display_date = date_str

        ps = portfolio_snapshot
        total_pnl_sign = "+" if ps["total_pnl"] >= 0 else ""
        daily_pnl_sign = "+" if ps["daily_pnl"] >= 0 else ""

        lines = [
            "📊 KAMATH TERMINAL — EOD REPORT",
            f"Date: {display_date}",
            "",
            "💼 PORTFOLIO",
            f"Value: ₹{ps['total_value']:,.0f} ({total_pnl_sign}{ps['total_pnl_pct']:.2f}% overall)",
            f"Today: {daily_pnl_sign}₹{ps['daily_pnl']:,.0f} ({daily_pnl_sign}{ps['daily_pnl_pct']:.2f}%)",
            f"Cash: ₹{ps['cash']:,.0f} ({ps['_cash_pct'] if '_cash_pct' in ps else 0:.1f}%)",
            "",
        ]

        # Recalculate cash pct inline
        cash_pct = round(ps["cash"] / ps["total_value"] * 100, 1) if ps["total_value"] else 0
        lines[7] = f"Cash: ₹{ps['cash']:,.0f} ({cash_pct:.1f}%)"

        # Today's trades
        trade_count = len(todays_trades)
        lines.append(f"📈 TODAY'S TRADES ({trade_count})")
        if todays_trades:
            for t in todays_trades:
                action = t["action"]
                ticker = t["ticker"].replace(".NS", "")
                price = t["price"]
                strat = t.get("strategy", "")
                lines.append(f"• {action} {ticker} @ ₹{price:,.0f} | {strat}")
        else:
            lines.append("• No trades today")

        lines.append("")

        # Gainers
        lines.append("🏆 TOP PERFORMERS")
        if top_3_gainers:
            for g in top_3_gainers:
                ticker = g["ticker"].replace(".NS", "")
                lines.append(f"• {ticker} +{g['pnl_pct']:.2f}% (₹{g['pnl']:,.0f})")
        else:
            lines.append("• No positions in profit")

        lines.append("")

        # Losers
        lines.append("📉 LAGGARDS")
        if top_3_losers:
            for l in top_3_losers:
                ticker = l["ticker"].replace(".NS", "")
                lines.append(f"• {ticker} {l['pnl_pct']:.2f}% (₹{l['pnl']:,.0f})")
        else:
            lines.append("• No positions in loss")

        lines.append("")

        # Signals
        total_signals = signals_tomorrow["buys"]
        strong = signals_tomorrow["strong_buys"]
        lines.append("🎯 TOMORROW'S SIGNALS")
        lines.append(f"{total_signals} active | {strong} strong buys")

        lines.append("")

        # Market
        nifty = market_summary.get("nifty_change_pct", 0)
        bnifty = market_summary.get("banknifty_change_pct", 0)
        vix = market_summary.get("vix", 0)
        nifty_sign = "+" if nifty >= 0 else ""
        bnifty_sign = "+" if bnifty >= 0 else ""
        lines.append("📊 MARKET")
        lines.append(f"Nifty: {nifty_sign}{nifty:.2f}% | Bank Nifty: {bnifty_sign}{bnifty:.2f}% | VIX: {vix}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_report(self, report: dict) -> str:
        """Save report to data/reports/YYYY-MM-DD.json. Returns file path."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        date_str = report.get("date", _now_ist().strftime("%Y-%m-%d"))
        out_path = REPORTS_DIR / f"{date_str}.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {out_path}")
        return str(out_path)

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------
    def get_dashboard_data(self) -> dict:
        """Return today's report (generates fresh each call)."""
        return self.generate_report()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_REPORT_AGENT_INSTANCE = None
_REPORT_AGENT_LOCK = threading.Lock()


def get_report_agent() -> DailyReportAgent:
    global _REPORT_AGENT_INSTANCE
    if _REPORT_AGENT_INSTANCE is None:
        with _REPORT_AGENT_LOCK:
            if _REPORT_AGENT_INSTANCE is None:
                _REPORT_AGENT_INSTANCE = DailyReportAgent()
    return _REPORT_AGENT_INSTANCE
