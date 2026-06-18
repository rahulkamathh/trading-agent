#!/bin/bash
cd "$(dirname "$0")"
rm -f .git/HEAD.lock .git/index.lock
git add app.py fno_engine.py templates/dashboard.html
git commit -m "feat: capital allocation config + fix P&L calendar separation

CAPITAL ALLOCATION (new):
- /api/capital_config GET/POST — save total/equity/FnO split to data/capital_config.json
- Controls page: Capital Allocation panel with inputs + visual bar
- Master page: initial capital loaded from config (no more hardcoded ₹15L)
- Combined value = equity portfolio + FnO (cash + deployed + unrealised)

P&L CALENDAR FIX:
- /api/pnl_calendar now shows EQUITY trades only (was combining equity+FnO+commodity)
- /api/fno/pnl_calendar already shows FnO only — no change needed
- Both desks now show independent, non-overlapping P&L calendars
"
git push
echo "Done."
