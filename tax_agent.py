"""
Tax Agent
=========
Tracks Indian equity tax implications (STCG / LTCG / Intraday) for all
paper trades and surfaces actionable tax-optimisation tips.

Indian tax rules implemented:
  - Intraday (same-day buy+sell): 30 % slab (speculative)
  - STCG  (held < 365 days)     : 20 % flat
  - LTCG  (held ≥ 365 days)     : 12.5 % on gains above ₹1,25,000 threshold
"""

import json
import logging
import threading
from collections import defaultdict, deque
from datetime import datetime, date
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

TRADE_LOG_FILE = DATA_DIR / "trade_log.json"
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"

# Tax rates
INTRADAY_RATE = 0.30
STCG_RATE = 0.20
LTCG_RATE = 0.125
LTCG_EXEMPT = 125_000   # ₹1.25 lakh exemption per FY

# Indian Financial Year starts April 1
FY_START_MONTH = 4
FY_START_DAY = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fy_start(ref: date) -> date:
    """Return the start of the Indian FY that contains *ref*."""
    if ref.month >= FY_START_MONTH:
        return date(ref.year, FY_START_MONTH, FY_START_DAY)
    return date(ref.year - 1, FY_START_MONTH, FY_START_DAY)


def _parse_date(ts: str) -> date:
    """Parse ISO timestamp or date string into a date object."""
    try:
        return datetime.fromisoformat(ts).date()
    except ValueError:
        return datetime.strptime(ts[:10], "%Y-%m-%d").date()


# ---------------------------------------------------------------------------
# TaxAgent
# ---------------------------------------------------------------------------

class TaxAgent:
    """Singleton tax analysis and optimisation agent."""

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse_trades(self, trades: list) -> dict:
        """
        Analyse a list of trade dicts from trade_log.json.
        Matches SELL trades to their corresponding BUY via FIFO per ticker.
        Returns a summary dict with per-trade tax details and totals.
        """
        # Build FIFO queues of BUY lots per ticker
        buy_queues: dict[str, deque] = defaultdict(deque)
        analysed_trades = []

        intraday_pnl = 0.0
        stcg_pnl = 0.0
        ltcg_pnl = 0.0

        for trade in trades:
            action = trade.get("action", "").upper()
            ticker = trade.get("ticker", "")
            qty = float(trade.get("qty", 0))
            price = float(trade.get("price", 0))
            ts = trade.get("time", "")

            if not ts or qty <= 0 or price <= 0:
                continue

            trade_date = _parse_date(ts)

            if action == "BUY":
                buy_queues[ticker].append({
                    "date": trade_date,
                    "price": price,
                    "qty": qty,
                })
                continue

            if action != "SELL":
                continue

            # Match SELL against BUY lots (FIFO)
            remaining_sell = qty
            sell_price = price
            sell_date = trade_date

            while remaining_sell > 0 and buy_queues[ticker]:
                lot = buy_queues[ticker][0]
                matched_qty = min(lot["qty"], remaining_sell)

                buy_date = lot["date"]
                buy_price = lot["price"]
                hold_days = (sell_date - buy_date).days

                gross_pnl = (sell_price - buy_price) * matched_qty

                # Classify
                if buy_date == sell_date:
                    category = "INTRADAY"
                    tax_rate = INTRADAY_RATE
                    tax_amount = round(max(gross_pnl, 0) * tax_rate, 2)
                    intraday_pnl += gross_pnl
                elif hold_days < 365:
                    category = "STCG"
                    tax_rate = STCG_RATE
                    tax_amount = round(max(gross_pnl, 0) * tax_rate, 2)
                    stcg_pnl += gross_pnl
                else:
                    category = "LTCG"
                    tax_rate = LTCG_RATE
                    # Exemption applied at summary level; trade-level tax shown without exemption
                    tax_amount = round(max(gross_pnl, 0) * tax_rate, 2)
                    ltcg_pnl += gross_pnl

                analysed_trades.append({
                    "ticker": ticker,
                    "buy_date": str(buy_date),
                    "sell_date": str(sell_date),
                    "buy_price": round(buy_price, 2),
                    "sell_price": round(sell_price, 2),
                    "qty": matched_qty,
                    "gross_pnl": round(gross_pnl, 2),
                    "hold_days": hold_days,
                    "tax_category": category,
                    "tax_rate_pct": round(tax_rate * 100, 1),
                    "tax_amount": tax_amount,
                })

                lot["qty"] -= matched_qty
                remaining_sell -= matched_qty
                if lot["qty"] <= 0:
                    buy_queues[ticker].popleft()

        # LTCG exemption: only the amount above ₹1.25L is taxable
        ltcg_taxable = max(0.0, ltcg_pnl - LTCG_EXEMPT) if ltcg_pnl > 0 else 0.0
        ltcg_tax = round(ltcg_taxable * LTCG_RATE, 2)

        intraday_tax = round(max(intraday_pnl, 0) * INTRADAY_RATE, 2)
        stcg_tax = round(max(stcg_pnl, 0) * STCG_RATE, 2)

        total_tax = intraday_tax + stcg_tax + ltcg_tax
        net_pnl_after_tax = round(intraday_pnl + stcg_pnl + ltcg_pnl - total_tax, 2)

        return {
            "summary": {
                "intraday_pnl": round(intraday_pnl, 2),
                "intraday_tax": intraday_tax,
                "stcg_pnl": round(stcg_pnl, 2),
                "stcg_tax": stcg_tax,
                "ltcg_pnl": round(ltcg_pnl, 2),
                "ltcg_taxable_pnl": round(ltcg_taxable, 2),
                "ltcg_tax": ltcg_tax,
                "total_tax_liability": round(total_tax, 2),
                "net_pnl_after_tax": net_pnl_after_tax,
            },
            "trades": analysed_trades,
            "optimisation_tips": [],  # populated separately
        }

    def get_optimisation_tips(self, positions: dict, trades: list) -> list:
        """
        Analyse open positions and realised trades to generate actionable
        tax-saving tips.

        *positions* : dict of {ticker: position_dict} from portfolio.json
        *trades*    : list of trade dicts from trade_log.json (all BUY/SELL)
        """
        tips = []
        today = date.today()

        # Build last-buy-date map per ticker from trade log
        last_buy: dict[str, dict] = {}
        for trade in trades:
            if trade.get("action", "").upper() == "BUY":
                ticker = trade.get("ticker", "")
                ts = trade.get("time", "")
                if ticker and ts:
                    last_buy[ticker] = {
                        "date": _parse_date(ts),
                        "price": float(trade.get("price", 0)),
                        "qty": float(trade.get("qty", 0)),
                    }

        # Compute existing STCG from closed trades for offset analysis
        analysis = self.analyse_trades(trades)
        existing_stcg_pnl = analysis["summary"]["stcg_pnl"]
        existing_ltcg_pnl = analysis["summary"]["ltcg_pnl"]

        for ticker, pos in positions.items():
            if not isinstance(pos, dict):
                continue

            buy_price = float(pos.get("avg_price", pos.get("entry_price", 0)))
            current_price = float(pos.get("current_price", buy_price))
            qty = float(pos.get("qty", pos.get("quantity", 0)))

            if qty <= 0 or buy_price <= 0:
                continue

            # Use last_buy date if available, fallback to position timestamp
            if ticker in last_buy:
                buy_date = last_buy[ticker]["date"]
            else:
                bought_at = pos.get("bought_at", pos.get("entry_time", ""))
                if not bought_at:
                    continue
                try:
                    buy_date = _parse_date(bought_at)
                except Exception:
                    continue

            hold_days = (today - buy_date).days
            unrealised_pnl = (current_price - buy_price) * qty

            # --- TIP 1: Hold for LTCG (if within 30 days of 1-year mark) ---
            days_to_ltcg = 365 - hold_days
            if 0 < days_to_ltcg <= 30 and unrealised_pnl > 0:
                stcg_tax_if_sold_now = unrealised_pnl * STCG_RATE
                ltcg_tax_after_wait = max(0, unrealised_pnl - LTCG_EXEMPT) * LTCG_RATE
                saving = round(stcg_tax_if_sold_now - ltcg_tax_after_wait, 2)
                if saving > 0:
                    tips.append({
                        "type": "HOLD_FOR_LTCG",
                        "ticker": ticker,
                        "action": (
                            f"Hold {ticker} for {days_to_ltcg} more days to qualify for LTCG "
                            f"rate (12.5%) instead of STCG rate (20%). "
                            f"Potential tax saving: ₹{saving:,.0f}"
                        ),
                        "tax_saving": saving,
                        "days_remaining": days_to_ltcg,
                    })

            # --- TIP 2: Tax-loss harvesting ---
            if unrealised_pnl < 0 and existing_stcg_pnl > 0 and hold_days < 365:
                loss_to_harvest = abs(unrealised_pnl)
                tax_saving = round(min(loss_to_harvest, existing_stcg_pnl) * STCG_RATE, 2)
                if tax_saving > 0:
                    tips.append({
                        "type": "HARVEST_LOSS",
                        "ticker": ticker,
                        "action": (
                            f"Harvest unrealised loss of ₹{loss_to_harvest:,.0f} on {ticker} "
                            f"to offset ₹{existing_stcg_pnl:,.0f} STCG gains. "
                            f"Estimated tax saving: ₹{tax_saving:,.0f}"
                        ),
                        "tax_saving": tax_saving,
                        "unrealised_loss": round(unrealised_pnl, 2),
                    })

            # --- TIP 3: STCG gain that can be deferred ---
            if unrealised_pnl > 0 and hold_days < 365 and days_to_ltcg > 30:
                stcg_tax = round(unrealised_pnl * STCG_RATE, 2)
                ltcg_tax_est = round(max(0, unrealised_pnl - LTCG_EXEMPT) * LTCG_RATE, 2)
                saving = round(stcg_tax - ltcg_tax_est, 2)
                if saving > 500:
                    tips.append({
                        "type": "DEFER_FOR_LTCG",
                        "ticker": ticker,
                        "action": (
                            f"Consider deferring sale of {ticker} until {buy_date.replace(year=buy_date.year+1)} "
                            f"to convert ₹{unrealised_pnl:,.0f} STCG (tax ₹{stcg_tax:,.0f}) "
                            f"to LTCG (tax ~₹{ltcg_tax_est:,.0f}). "
                            f"Potential saving: ₹{saving:,.0f}"
                        ),
                        "tax_saving": saving,
                        "unrealised_gain": round(unrealised_pnl, 2),
                    })

        # --- TIP 4: LTCG threshold check ---
        if existing_ltcg_pnl > 0:
            remaining_exempt = max(0.0, LTCG_EXEMPT - existing_ltcg_pnl)
            taxable_ltcg = max(0.0, existing_ltcg_pnl - LTCG_EXEMPT)
            if remaining_exempt > 0:
                tips.append({
                    "type": "LTCG_THRESHOLD",
                    "ticker": None,
                    "action": (
                        f"You have ₹{existing_ltcg_pnl:,.0f} in LTCG gains. "
                        f"₹1,25,000 is exempt per FY — you still have ₹{remaining_exempt:,.0f} "
                        f"of exemption remaining. Only ₹{taxable_ltcg:,.0f} is taxable."
                    ),
                    "tax_saving": 0,
                })
            else:
                tips.append({
                    "type": "LTCG_THRESHOLD",
                    "ticker": None,
                    "action": (
                        f"LTCG gains of ₹{existing_ltcg_pnl:,.0f} exceed the ₹1,25,000 "
                        f"annual exemption. Effective LTCG tax applies on ₹{taxable_ltcg:,.0f}."
                    ),
                    "tax_saving": 0,
                })

        return tips

    def get_dashboard_data(self) -> dict:
        """
        Loads trade_log.json and portfolio.json, runs full tax analysis,
        and returns a combined dashboard payload.
        """
        today = date.today()
        fy_start = _fy_start(today)

        # Load trades
        trades = []
        if TRADE_LOG_FILE.exists():
            try:
                with open(TRADE_LOG_FILE) as f:
                    trades = json.load(f)
                if not isinstance(trades, list):
                    trades = []
            except Exception as exc:
                logger.warning(f"Could not load trade_log.json: {exc}")
                trades = []

        # Load portfolio (positions)
        positions = {}
        if PORTFOLIO_FILE.exists():
            try:
                with open(PORTFOLIO_FILE) as f:
                    portfolio = json.load(f)
                positions = portfolio.get("positions", {})
            except Exception as exc:
                logger.warning(f"Could not load portfolio.json: {exc}")

        # Filter trades to current FY for tax analysis
        fy_trades = []
        for trade in trades:
            ts = trade.get("time", "")
            if not ts:
                continue
            try:
                td = _parse_date(ts)
                if td >= fy_start:
                    fy_trades.append(trade)
            except Exception:
                continue

        analysis = self.analyse_trades(fy_trades)
        tips = self.get_optimisation_tips(positions, trades)

        return {
            "summary": analysis["summary"],
            "trades": analysis["trades"],
            "optimisation_tips": tips,
            "fy_start": str(fy_start),
            "last_updated": datetime.now().isoformat(),
        }


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_TAX_INSTANCE: TaxAgent | None = None
_TAX_LOCK = threading.Lock()


def get_tax_agent() -> TaxAgent:
    """Return the singleton TaxAgent instance."""
    global _TAX_INSTANCE
    if _TAX_INSTANCE is None:
        with _TAX_LOCK:
            if _TAX_INSTANCE is None:
                _TAX_INSTANCE = TaxAgent()
    return _TAX_INSTANCE
