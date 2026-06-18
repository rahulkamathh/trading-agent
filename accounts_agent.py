"""
Accounts Agent — CA-Style Financial Accounting
===============================================
Acts as a Chartered Accountant for the trading portfolio.

Responsibilities:
  1. Full trade ledger with per-trade cost breakdowns
     (signal price → exec price → slippage cost → fees itemised)
  2. P&L waterfall: Gross → after slippage → after fees → after tax
  3. Tax computation: STCG (20%), LTCG (12.5% above ₹1.25L), Intraday (30%)
  4. Advance tax schedule (Indian IT Act — due Jun 15, Sep 15, Dec 15, Mar 15)
  5. Fee drag analysis (STT, exchange, GST, stamp duty, SEBI) as % of returns
  6. Monthly P&L statement (gross vs net per month)
  7. Tax-loss harvesting alerts on open positions
  8. FY summary (April → March)

Data source: data/trade_log.json + data/portfolio.json
Fees: back-calculated for historical trades without fee data, using the exact
      Zerodha CNC charge model (same as _compute_zerodha_costs in engine.py)
"""

import json
import logging
import threading
import time
from collections import defaultdict, deque
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

DATA_DIR        = Path(__file__).parent / "data"
TRADE_LOG_FILE  = DATA_DIR / "trade_log.json"
PORTFOLIO_FILE  = DATA_DIR / "portfolio.json"

_CACHE_TTL = 300   # 5-minute cache

# ── Indian tax rates ──────────────────────────────────────────────────────────
INTRADAY_RATE  = 0.30
STCG_RATE      = 0.20
LTCG_RATE      = 0.125
LTCG_EXEMPT    = 125_000   # ₹1.25L per FY

# ── Zerodha CNC charges (mirrored from engine.py) ────────────────────────────
SLIPPAGE_BPS   = 5   # 5 basis points per side

def _zerodha_fees(qty: float, price: float, action: str) -> dict:
    """Zerodha CNC charge model — identical to engine._compute_zerodha_costs."""
    turnover         = qty * price
    brokerage        = 0.0
    stt              = turnover * 0.001
    exchange_charges = turnover * 0.0000297
    stamp_duty       = (turnover * 0.00015) if action == "BUY" else 0.0
    sebi_charges     = turnover * 0.000001
    gst              = (brokerage + exchange_charges + sebi_charges) * 0.18
    total            = brokerage + stt + exchange_charges + gst + sebi_charges + stamp_duty
    return {
        "brokerage":        round(brokerage,        4),
        "stt":              round(stt,              4),
        "exchange_charges": round(exchange_charges, 4),
        "gst":              round(gst,              4),
        "sebi_charges":     round(sebi_charges,     4),
        "stamp_duty":       round(stamp_duty,       4),
        "total":            round(total,            4),
    }

def _slippage_cost(qty: float, price: float, action: str) -> float:
    """Estimated slippage cost (5 bps per side)."""
    factor = SLIPPAGE_BPS / 10_000
    return round(qty * price * factor, 4)

# ── Date helpers ──────────────────────────────────────────────────────────────

def _parse_dt(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return datetime.strptime(ts[:10], "%Y-%m-%d")

def _parse_date(ts: str) -> date:
    return _parse_dt(ts).date()

def _fy_label(d: date) -> str:
    """Return FY label, e.g. 'FY2025-26' for any date in April 2025 – March 2026."""
    if d.month >= 4:
        return f"FY{d.year}-{str(d.year + 1)[2:]}"
    return f"FY{d.year - 1}-{str(d.year)[2:]}"

def _month_label(d: date) -> str:
    return d.strftime("%b %Y")   # e.g. "Jun 2026"

def _current_fy() -> str:
    return _fy_label(datetime.now(IST).date())

def _advance_tax_schedule(total_tax: float, fy: str) -> list:
    """
    Indian IT Act advance tax instalments.
    Applicable only if total tax liability > ₹10,000.
    FY start year extracted from fy label, e.g. 'FY2025-26' → 2025.
    """
    if total_tax <= 10_000:
        return []
    try:
        start_yr = int(fy[2:6])
    except Exception:
        return []
    return [
        {"due": f"15 Jun {start_yr}",   "cumulative_pct": 15,  "amount": round(total_tax * 0.15, 0)},
        {"due": f"15 Sep {start_yr}",   "cumulative_pct": 45,  "amount": round(total_tax * 0.30, 0)},  # 45-15=30 new
        {"due": f"15 Dec {start_yr}",   "cumulative_pct": 75,  "amount": round(total_tax * 0.30, 0)},  # 75-45=30 new
        {"due": f"15 Mar {start_yr+1}", "cumulative_pct": 100, "amount": round(total_tax * 0.25, 0)},  # 100-75=25 new
    ]


# ── AccountsAgent ─────────────────────────────────────────────────────────────

class AccountsAgent:
    """CA-style accounting engine for the trading portfolio."""

    def __init__(self):
        self._lock       = threading.Lock()
        self._cache: dict | None = None
        self._cache_ts: float    = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_dashboard_data(self, fy: str | None = None) -> dict:
        """
        Return full accounting report for the dashboard.
        fy: e.g. 'FY2025-26'. Defaults to current FY.
        """
        with self._lock:
            now = time.time()
            if self._cache and now - self._cache_ts < _CACHE_TTL:
                return self._cache if fy is None else self._filter_fy(self._cache, fy or _current_fy())
            result = self._compute()
            self._cache    = result
            self._cache_ts = now
            return result if fy is None else self._filter_fy(result, fy or _current_fy())

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _compute(self) -> dict:
        trades   = self._load_trades()
        port     = self._load_portfolio()
        positions = port.get("positions", {})

        # ── Build enriched ledger ────────────────────────────────────────
        ledger   = self._build_ledger(trades)

        # ── Aggregate by FY ──────────────────────────────────────────────
        by_fy    = self._aggregate_by_fy(ledger)
        cur_fy   = _current_fy()
        cur      = by_fy.get(cur_fy, self._empty_fy_bucket())

        # ── Monthly summary ──────────────────────────────────────────────
        monthly  = self._monthly_summary(ledger)

        # ── Fee breakdown (lifetime) ─────────────────────────────────────
        fee_breakdown = self._fee_breakdown(ledger)

        # ── Tax-loss harvesting ──────────────────────────────────────────
        tlh = self._tlh_opportunities(positions, cur.get("stcg_pnl", 0) + cur.get("ltcg_pnl", 0))

        # ── Advance tax ──────────────────────────────────────────────────
        adv_tax = _advance_tax_schedule(cur.get("total_tax", 0), cur_fy)

        # ── Lifetime totals ───────────────────────────────────────────────
        life_fees     = sum(t["fees_total"]    for t in ledger)
        life_slip     = sum(t["slippage_cost"] for t in ledger)
        life_gross    = sum(t["gross_pnl"]     for t in ledger if t["action"] == "SELL")
        life_net      = life_gross - life_fees - life_slip

        return {
            "current_fy":     cur_fy,
            "current_fy_data": cur,
            "by_fy":          by_fy,
            "monthly":        monthly,
            "fee_breakdown":  fee_breakdown,
            "advance_tax":    adv_tax,
            "tlh_opportunities": tlh,
            "lifetime": {
                "gross_pnl":    round(life_gross, 2),
                "total_fees":   round(life_fees,  2),
                "total_slippage": round(life_slip, 2),
                "net_pnl":      round(life_net,   2),
                "fee_drag_pct": round(life_fees / abs(life_gross) * 100, 2) if life_gross else 0.0,
            },
            "ledger":         list(reversed(ledger[-100:])),   # last 100 for dashboard
            "last_updated":   datetime.now(IST).isoformat(),
        }

    # ------------------------------------------------------------------
    # Ledger builder
    # ------------------------------------------------------------------

    def _build_ledger(self, trades: list) -> list:
        """
        Convert raw trade_log.json entries into enriched ledger rows.
        For BUY trades: records acquisition cost.
        For SELL trades: matches FIFO lots, computes gross P&L, fees, slippage.
        Fees/slippage are taken from trade record if present, otherwise estimated.
        """
        buy_queues: dict[str, deque] = defaultdict(deque)
        ledger: list[dict] = []

        for t in trades:
            action = t.get("action", "").upper()
            ticker = t.get("ticker", "")
            qty    = float(t.get("qty", 0))
            price  = float(t.get("price", 0))
            ts     = t.get("time", "")

            if not ts or qty <= 0 or price <= 0:
                continue

            trade_date = _parse_date(ts)

            # Get fees from log if present, else estimate
            fees_dict    = t.get("fees") or _zerodha_fees(qty, price, action)
            fees_total   = t.get("fees_total") or fees_dict.get("total", 0)
            slippage_raw = t.get("slippage_cost")
            slip_cost    = slippage_raw if slippage_raw is not None else _slippage_cost(qty, price, action)

            if action == "BUY":
                buy_queues[ticker].append({
                    "date":        trade_date,
                    "price":       price,
                    "qty":         qty,
                    "fees_total":  fees_total,
                    "fees_dict":   fees_dict,
                    "slip_cost":   slip_cost,
                })
                ledger.append({
                    "action":        "BUY",
                    "ticker":        ticker,
                    "date":          str(trade_date),
                    "qty":           qty,
                    "signal_price":  price,
                    "exec_price":    price,
                    "gross_pnl":     0.0,
                    "fees_total":    round(fees_total, 2),
                    "fees_breakdown": fees_dict,
                    "slippage_cost": round(slip_cost, 2),
                    "net_pnl":       round(-(fees_total + slip_cost), 2),
                    "tax_category":  None,
                    "tax_amount":    0.0,
                    "hold_days":     None,
                    "strategy":      t.get("strategy", ""),
                    "fy":            _fy_label(trade_date),
                    "month":         _month_label(trade_date),
                })
                continue

            if action != "SELL":
                continue

            # SELL — match FIFO lots
            remaining = qty
            sell_date  = trade_date

            while remaining > 0 and buy_queues[ticker]:
                lot         = buy_queues[ticker][0]
                matched_qty = min(lot["qty"], remaining)

                buy_date  = lot["date"]
                buy_price = lot["price"]
                hold_days = (sell_date - buy_date).days

                gross_pnl = (price - buy_price) * matched_qty

                # Pro-rate buy fees/slip to matched qty
                lot_frac      = matched_qty / max(lot["qty"], 1)
                buy_fees      = round(lot["fees_total"]  * lot_frac, 4)
                buy_slip      = round(lot["slip_cost"]   * lot_frac, 4)
                sell_frac     = matched_qty / max(qty, 1)
                sell_fees     = round(fees_total * sell_frac, 4)
                sell_slip     = round(slip_cost  * sell_frac, 4)
                total_costs   = buy_fees + buy_slip + sell_fees + sell_slip
                net_pnl       = gross_pnl - total_costs

                # Tax category
                if buy_date == sell_date:
                    cat      = "INTRADAY"
                    tax_rate = INTRADAY_RATE
                elif hold_days < 365:
                    cat      = "STCG"
                    tax_rate = STCG_RATE
                else:
                    cat      = "LTCG"
                    tax_rate = LTCG_RATE
                tax_amount = round(max(net_pnl, 0) * tax_rate, 2)

                ledger.append({
                    "action":        "SELL",
                    "ticker":        ticker,
                    "date":          str(sell_date),
                    "buy_date":      str(buy_date),
                    "hold_days":     hold_days,
                    "qty":           matched_qty,
                    "buy_price":     round(buy_price, 2),
                    "signal_price":  round(price, 2),
                    "exec_price":    round(price, 2),
                    "gross_pnl":     round(gross_pnl, 2),
                    "buy_fees":      round(buy_fees, 2),
                    "sell_fees":     round(sell_fees, 2),
                    "fees_total":    round(buy_fees + sell_fees, 2),
                    "fees_breakdown": fees_dict,
                    "buy_slippage":  round(buy_slip, 2),
                    "sell_slippage": round(sell_slip, 2),
                    "slippage_cost": round(buy_slip + sell_slip, 2),
                    "total_costs":   round(total_costs, 2),
                    "net_pnl":       round(net_pnl, 2),
                    "tax_category":  cat,
                    "tax_rate_pct":  round(tax_rate * 100, 1),
                    "tax_amount":    tax_amount,
                    "strategy":      t.get("strategy", ""),
                    "fy":            _fy_label(sell_date),
                    "month":         _month_label(sell_date),
                })

                lot["qty"] -= matched_qty
                remaining  -= matched_qty
                if lot["qty"] <= 0:
                    buy_queues[ticker].popleft()

        return ledger

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def _aggregate_by_fy(self, ledger: list) -> dict:
        by_fy: dict[str, dict] = {}

        for row in ledger:
            if row["action"] != "SELL":
                continue
            fy = row["fy"]
            if fy not in by_fy:
                by_fy[fy] = self._empty_fy_bucket()
            b = by_fy[fy]

            b["gross_pnl"]    += row["gross_pnl"]
            b["total_fees"]   += row["fees_total"]
            b["total_slippage"] += row["slippage_cost"]
            b["total_costs"]  += row["total_costs"]
            b["net_pnl"]      += row["net_pnl"]
            b["trade_count"]  += 1

            cat = row["tax_category"]
            if cat == "INTRADAY":
                b["intraday_pnl"] += row["gross_pnl"]
            elif cat == "STCG":
                b["stcg_pnl"]     += row["gross_pnl"]
            elif cat == "LTCG":
                b["ltcg_pnl"]     += row["gross_pnl"]

        # Compute tax per FY
        for fy, b in by_fy.items():
            intraday_tax = round(max(b["intraday_pnl"], 0) * INTRADAY_RATE, 2)
            stcg_tax     = round(max(b["stcg_pnl"], 0) * STCG_RATE, 2)
            ltcg_taxable = max(0, b["ltcg_pnl"] - LTCG_EXEMPT) if b["ltcg_pnl"] > 0 else 0
            ltcg_tax     = round(ltcg_taxable * LTCG_RATE, 2)
            total_tax    = intraday_tax + stcg_tax + ltcg_tax

            b.update({
                "intraday_tax":   intraday_tax,
                "stcg_tax":       stcg_tax,
                "ltcg_taxable":   round(ltcg_taxable, 2),
                "ltcg_tax":       ltcg_tax,
                "total_tax":      round(total_tax, 2),
                "take_home":      round(b["net_pnl"] - total_tax, 2),
                "gross_pnl":      round(b["gross_pnl"], 2),
                "net_pnl":        round(b["net_pnl"], 2),
                "total_fees":     round(b["total_fees"], 2),
                "total_slippage": round(b["total_slippage"], 2),
                "total_costs":    round(b["total_costs"], 2),
                "fee_drag_pct":   round(b["total_fees"] / abs(b["gross_pnl"]) * 100, 2)
                                  if b["gross_pnl"] else 0.0,
                "advance_tax":    _advance_tax_schedule(total_tax, fy),
            })

        return by_fy

    def _monthly_summary(self, ledger: list) -> list:
        monthly: dict[str, dict] = {}
        for row in ledger:
            if row["action"] != "SELL":
                continue
            m = row["month"]
            if m not in monthly:
                monthly[m] = {
                    "month": m, "gross_pnl": 0.0,
                    "total_fees": 0.0, "total_slippage": 0.0,
                    "net_pnl": 0.0, "trade_count": 0,
                }
            b = monthly[m]
            b["gross_pnl"]    += row["gross_pnl"]
            b["total_fees"]   += row["fees_total"]
            b["total_slippage"] += row["slippage_cost"]
            b["net_pnl"]      += row["net_pnl"]
            b["trade_count"]  += 1

        # Round and sort chronologically
        result = []
        for m, b in monthly.items():
            b["gross_pnl"]    = round(b["gross_pnl"], 2)
            b["total_fees"]   = round(b["total_fees"], 2)
            b["total_slippage"] = round(b["total_slippage"], 2)
            b["net_pnl"]      = round(b["net_pnl"], 2)
            result.append(b)
        result.sort(key=lambda x: datetime.strptime(x["month"], "%b %Y"))
        return result

    def _fee_breakdown(self, ledger: list) -> dict:
        totals = {
            "stt": 0.0, "exchange_charges": 0.0,
            "gst": 0.0, "sebi_charges": 0.0,
            "stamp_duty": 0.0, "brokerage": 0.0, "slippage": 0.0,
        }
        for row in ledger:
            fb = row.get("fees_breakdown") or {}
            totals["stt"]              += fb.get("stt", 0)
            totals["exchange_charges"] += fb.get("exchange_charges", 0)
            totals["gst"]              += fb.get("gst", 0)
            totals["sebi_charges"]     += fb.get("sebi_charges", 0)
            totals["stamp_duty"]       += fb.get("stamp_duty", 0)
            totals["brokerage"]        += fb.get("brokerage", 0)
            totals["slippage"]         += row.get("slippage_cost", 0)
        return {k: round(v, 2) for k, v in totals.items()}

    def _tlh_opportunities(self, positions: dict, realised_gains: float) -> list:
        """
        Identify open positions with unrealised losses that could be crystallised
        to offset realised gains (Tax-Loss Harvesting).
        Only flags positions if there are gains to offset.
        """
        if realised_gains <= 0:
            return []
        alerts = []
        for ticker, pos in positions.items():
            avg_price = pos.get("avg_price", 0)
            qty       = pos.get("qty", 0)
            # Try to get current price from engine DataFetcher
            try:
                from engine import DataFetcher  # pylint: disable=import-outside-toplevel
                cur_price = DataFetcher.get_current_price(ticker) or avg_price
            except Exception:
                cur_price = avg_price
            unreal_pnl = (cur_price - avg_price) * qty
            if unreal_pnl >= -1000:   # only flag if loss > ₹1,000
                continue
            tax_saving = round(abs(unreal_pnl) * STCG_RATE, 2)
            alerts.append({
                "ticker":       ticker,
                "qty":          qty,
                "avg_price":    round(avg_price, 2),
                "current_price": round(cur_price, 2),
                "unrealised_pnl": round(unreal_pnl, 2),
                "potential_tax_saving": tax_saving,
                "note": (
                    f"Crystallising ₹{abs(unreal_pnl):,.0f} loss "
                    f"could save ₹{tax_saving:,.0f} in STCG tax"
                ),
            })
        alerts.sort(key=lambda x: x["unrealised_pnl"])
        return alerts[:10]

    def _filter_fy(self, data: dict, fy: str) -> dict:
        """Return data scoped to a specific FY."""
        result = dict(data)
        result["current_fy"] = fy
        result["current_fy_data"] = data["by_fy"].get(fy, self._empty_fy_bucket())
        result["ledger"] = [r for r in data["ledger"] if r.get("fy") == fy]
        result["advance_tax"] = _advance_tax_schedule(
            result["current_fy_data"].get("total_tax", 0), fy
        )
        return result

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------

    def _load_trades(self) -> list:
        if not TRADE_LOG_FILE.exists():
            return []
        try:
            with open(TRADE_LOG_FILE) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"[AccountsAgent] Could not load trade log: {e}")
            return []

    def _load_portfolio(self) -> dict:
        if not PORTFOLIO_FILE.exists():
            return {}
        try:
            with open(PORTFOLIO_FILE) as f:
                return json.load(f)
        except Exception:
            return {}

    @staticmethod
    def _empty_fy_bucket() -> dict:
        return {
            "gross_pnl": 0.0, "net_pnl": 0.0, "take_home": 0.0,
            "total_fees": 0.0, "total_slippage": 0.0, "total_costs": 0.0,
            "intraday_pnl": 0.0, "intraday_tax": 0.0,
            "stcg_pnl": 0.0, "stcg_tax": 0.0,
            "ltcg_pnl": 0.0, "ltcg_taxable": 0.0, "ltcg_tax": 0.0,
            "total_tax": 0.0, "fee_drag_pct": 0.0,
            "trade_count": 0, "advance_tax": [],
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_INSTANCE: AccountsAgent | None = None
_LOCK = threading.Lock()

def get_accounts_agent() -> AccountsAgent:
    global _INSTANCE
    if _INSTANCE is None:
        with _LOCK:
            if _INSTANCE is None:
                _INSTANCE = AccountsAgent()
    return _INSTANCE
