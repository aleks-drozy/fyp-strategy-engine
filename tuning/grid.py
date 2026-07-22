"""Coarse parameter grid for Phase-4 walk-forward tuning.

From the Phase-4 parameter-tuning spec
(docs/specs/2026-07-13-phase4-parameter-tuning-design.md) -- the coarse
144-combo grid.

PRE-REGISTRATION FREEZE (anti-meta-overfitting): `FVG_GRID`, `RR_GRID`,
`EMA_GRID`, `SWING_GRID` and their Cartesian product in `build_grid()` are
committed as constants BEFORE `run_phase4.py` is ever executed. Do not
change these values after seeing an OOS result -- that requires a new dated
spec and an explicitly-labelled new experiment, never a silent edit here.

Session (`session_start`/`session_end`) is fixed at the `StrategyParams`
default for the base run -- it is NOT gridded, so the grid is exactly
FVG_GRID x RR_GRID x EMA_GRID x SWING_GRID = 4 x 4 x 3 x 3 = 144 combos,
including the all-defaults point (`StrategyParams()` itself must be
reproducible by some combination of the four grids -- it is, since each
grid contains its corresponding default value).

Tuples (not sets) are used deliberately: `tuning.walkforward` derives a
"random" combo index deterministically from `(fold_i * 37) % 144` against
`build_grid()`'s output order, so that order must be fixed and reproducible
across runs/Python versions -- a `set`'s iteration order is not a
contract to rely on for that.
"""

from strategy.params import StrategyParams

FVG_GRID = (0.0, 0.02, 0.05, 0.10)
RR_GRID = (1.0, 1.5, 2.0, 3.0)
EMA_GRID = (10, 20, 50)
SWING_GRID = (5, 8, 12)


def build_grid() -> list[StrategyParams]:
    """The full coarse grid: Cartesian product of FVG_GRID x RR_GRID x
    EMA_GRID x SWING_GRID as `StrategyParams` (session fixed at defaults).

    144 combos, no duplicates, includes `StrategyParams()` (all defaults).
    """
    return [
        StrategyParams(fvg_threshold=fvg, rr=rr, ema_length=ema, swing_lookback=swing)
        for fvg in FVG_GRID
        for rr in RR_GRID
        for ema in EMA_GRID
        for swing in SWING_GRID
    ]
