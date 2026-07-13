"""Strategy parameters for the FYP IFVG+CISD NQ engine.

From docs/superpowers/plans/2026-07-13-phase4-parameter-tuning.md, Task 1
Step 1 (Global Constraints). Defaults MUST reproduce Phase 2 exactly --
they are the Pine-source values the engine was originally hardcoded to
(PT_VALUE/MAX_TRADES_PER_DAY stay engine-side constants, not params, per
the plan).

Phase 5 (docs/superpowers/plans/2026-07-13-phase5-exits-costs-volfilter.md,
Task 1 Step 4) adds `exit_mode` and `vol_filter`, both defaulting to the
Phase-2/4 base behavior so existing callers -- and the golden-fixture
regression -- are unaffected. `exit_mode="fixed_1_5R"` selects the
ORIGINAL, unmodified `backtest.engine._try_exit` stop/target logic (see
engine.py's dispatch); the other 4 modes route through backtest/exits.py.
`vol_filter` is a descriptive label for the Task-2 grid/walk-forward (its
threshold-selection machinery lives there) -- `run_execution` itself is
gated by the separate `atr`/`vol_threshold` arguments, not this field, so
`vol_filter="off"` alone doesn't change `run_execution`'s behavior."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyParams:
    fvg_threshold: float = 0.0
    rr: float = 1.5
    ema_length: int = 20
    swing_lookback: int = 8
    session_start: str = "09:30"
    session_end: str = "10:30"
    exit_mode: str = "fixed_1_5R"  # "fixed_1_5R" | "breakeven_1R" | "trail_swing" | "partial_1R" | "time_stop"
    vol_filter: str = "off"        # "off" | "p25" | "p50" | "p75" (descriptive; see docstring)
