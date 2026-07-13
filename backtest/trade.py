"""Trade record produced by backtest.engine.backtest().

From docs/superpowers/plans/2026-07-13-phase2-strategy-engine.md, Task 3
Step 1. A `risk` field is added on top of the plan's code block (see the
deliverables note in that Task): risk is the distance, in points, between
the signal-bar close and the stop, fixed at signal time. It is carried
through unchanged to the filled Trade so that `r_multiple` can be computed
against the *intended* risk even though the actual fill price (`entry`,
the next bar's open) can differ slightly from the signal-bar close that
anchors the stop/target.

Phase 5 (docs/superpowers/plans/2026-07-13-phase5-exits-costs-volfilter.md,
Task 1 Step 5 / Global Constraints -- "Trade schema is EXTENDED
ADDITIVELY") appends `net_pnl` and `exit_reason` AFTER all the existing
fields, and keeps `outcome`/field order otherwise unchanged so Phase-4's
`_serialize_trade`/charts (which read `t.outcome`) keep working untouched.
`net_pnl` mirrors `pnl_usd` (gross) when no `CostModel` is supplied;
`exit_reason` is one of "stop"/"target"/"trail"/"time"/"partial_scaleout"/
"partial_remainder_stop"/"partial_remainder_target" (empty string for a
trade that was never actually closed through the exit-mode machinery,
which should not happen for any trade in `trades`).
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Trade:
    entry_time: datetime
    direction: str          # "Long" | "Short"
    entry: float
    stop: float
    target: float
    risk: float              # points: abs(signal_close - stop), fixed at signal time
    exit_time: datetime | None = None
    exit: float | None = None
    pnl_usd: float | None = None
    r_multiple: float | None = None
    outcome: str = "Open"    # "Win" | "Loss" | "Open"
    net_pnl: float | None = None  # pnl_usd minus CostModel costs (== pnl_usd if no CostModel)
    exit_reason: str = ""         # "stop" | "target" | "trail" | "time" | "partial_scaleout" |
                                   # "partial_remainder_stop" | "partial_remainder_target"
