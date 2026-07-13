"""Tests for tuning.grid.build_grid.

Phase 4, Task 2 Step 2 (docs/superpowers/plans/2026-07-13-phase4-parameter-tuning.md).
"""

from strategy.params import StrategyParams
from tuning.grid import EMA_GRID, FVG_GRID, RR_GRID, SWING_GRID, build_grid


def test_grid_has_144_combos():
    assert len(build_grid()) == 144
    assert len(FVG_GRID) * len(RR_GRID) * len(EMA_GRID) * len(SWING_GRID) == 144


def test_defaults_are_a_grid_point():
    assert StrategyParams() in build_grid()


def test_no_duplicate_combos():
    grid = build_grid()
    assert len(set(grid)) == len(grid)


def test_grid_only_varies_the_four_gridded_fields():
    default = StrategyParams()
    for p in build_grid():
        assert p.session_start == default.session_start
        assert p.session_end == default.session_end
        assert p.fvg_threshold in FVG_GRID
        assert p.rr in RR_GRID
        assert p.ema_length in EMA_GRID
        assert p.swing_lookback in SWING_GRID
