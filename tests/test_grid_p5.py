"""Tests for tuning.grid_p5.build_grid_p5.

From the Phase-5 exits/costs/vol-filter spec
(docs/specs/2026-07-13-phase5-exits-costs-volfilter-design.md).
"""

from strategy.params import StrategyParams
from tuning.grid_p5 import EXIT_MODES, VOL_FILTERS, build_grid_p5


def test_grid_has_20_combos():
    assert len(build_grid_p5()) == 20
    assert len(EXIT_MODES) * len(VOL_FILTERS) == 20


def test_base_fixed_1_5r_off_is_a_grid_point():
    assert StrategyParams() in build_grid_p5()
    assert StrategyParams(exit_mode="fixed_1_5R", vol_filter="off") in build_grid_p5()


def test_no_duplicate_combos():
    grid = build_grid_p5()
    assert len(set(grid)) == len(grid)


def test_grid_only_varies_exit_mode_and_vol_filter():
    default = StrategyParams()
    for p in build_grid_p5():
        assert p.fvg_threshold == default.fvg_threshold
        assert p.rr == default.rr
        assert p.ema_length == default.ema_length
        assert p.swing_lookback == default.swing_lookback
        assert p.session_start == default.session_start
        assert p.session_end == default.session_end
        assert p.exit_mode in EXIT_MODES
        assert p.vol_filter in VOL_FILTERS


def test_every_exit_mode_x_vol_filter_pair_present_exactly_once():
    grid = build_grid_p5()
    pairs = [(p.exit_mode, p.vol_filter) for p in grid]
    expected = [(em, vf) for em in EXIT_MODES for vf in VOL_FILTERS]
    assert pairs == expected  # fixed, reproducible order (tuples, not sets)
