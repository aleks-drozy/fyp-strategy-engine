"""Phase-5 grid: exit_mode x vol_filter (20 combos, entry fixed at defaults).

From docs/superpowers/plans/2026-07-13-phase5-exits-costs-volfilter.md, Task
2 Step 1 (Global Constraints -- "Grid = 20"). Unlike Phase-4's `tuning/grid.py`
(which sweeps `fvg_threshold`/`rr`/`ema_length`/`swing_lookback`), the
Phase-5 sweep is CONDITIONAL on the Phase-4-null default entry: every combo
here uses `StrategyParams()`'s entry fields untouched (`fvg_threshold=0.0`,
`rr=1.5`, `ema_length=20`, `swing_lookback=8`, default session) and varies
ONLY `exit_mode` and `vol_filter`.

PRE-REGISTRATION FREEZE (anti-meta-overfitting, same convention as
`tuning/grid.py`): `EXIT_MODES`/`VOL_FILTERS` and their Cartesian product in
`build_grid_p5()` are committed BEFORE `run_phase5.py` is ever executed. Do
not change these after seeing an OOS result.

`EXIT_MODES` x `VOL_FILTERS` = 5 x 4 = 20 combos, including
`(fixed_1_5R, off)` -- i.e. `StrategyParams()` itself -- which MUST be
present: it is both the base/baseline arm of the net success rule (Global
Constraints, condition (b)) and `tuning.walkforward_p5.select_params_p5`'s
fallback target.

Tuples (not sets), same reasoning as `tuning/grid.py`: order must be fixed
and reproducible across runs/Python versions.
"""

from strategy.params import StrategyParams

EXIT_MODES = ("fixed_1_5R", "breakeven_1R", "trail_swing", "partial_1R", "time_stop")
VOL_FILTERS = ("off", "p25", "p50", "p75")


def build_grid_p5() -> list[StrategyParams]:
    """The full Phase-5 grid: Cartesian product of EXIT_MODES x VOL_FILTERS
    as `StrategyParams`, entry fields fixed at `StrategyParams()` defaults.

    20 combos, no duplicates, includes `StrategyParams()` (== the
    `(fixed_1_5R, off)` base baseline).
    """
    return [
        StrategyParams(exit_mode=exit_mode, vol_filter=vol_filter)
        for exit_mode in EXIT_MODES
        for vol_filter in VOL_FILTERS
    ]
