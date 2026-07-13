"""Trade record produced by backtest.engine.backtest().

From docs/superpowers/plans/2026-07-13-phase2-strategy-engine.md, Task 3
Step 1. A `risk` field is added on top of the plan's code block (see the
deliverables note in that Task): risk is the distance, in points, between
the signal-bar close and the stop, fixed at signal time. It is carried
through unchanged to the filled Trade so that `r_multiple` can be computed
against the *intended* risk even though the actual fill price (`entry`,
the next bar's open) can differ slightly from the signal-bar close that
anchors the stop/target.
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
