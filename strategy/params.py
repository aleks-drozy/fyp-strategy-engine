"""Strategy parameters for the FYP IFVG+CISD NQ engine.

From docs/superpowers/plans/2026-07-13-phase4-parameter-tuning.md, Task 1
Step 1 (Global Constraints). Defaults MUST reproduce Phase 2 exactly --
they are the Pine-source values the engine was originally hardcoded to
(PT_VALUE/MAX_TRADES_PER_DAY stay engine-side constants, not params, per
the plan)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyParams:
    fvg_threshold: float = 0.0
    rr: float = 1.5
    ema_length: int = 20
    swing_lookback: int = 8
    session_start: str = "09:30"
    session_end: str = "10:30"
