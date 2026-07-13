"""Smoke test for the core call run_phase5.py makes: walk_forward_p5(load_nq(),
grid, folds, cost_model) doesn't raise, returns the expected structure, and
net metrics (not gross) are present -- on a SMALL grid over a SHORT real
slice (not the full 20x4 sweep).

Task 3 Step 4 (docs/superpowers/plans/2026-07-13-phase5-exits-costs-volfilter.md).
This only proves plumbing (real data flows through walk_forward_p5 end-to-end
without error); leakage/selection/net-routing correctness is covered by
tests/test_walkforward_p5.py's synthetic-data tests. Skips if the raw Phase-1
NQ CSV isn't present locally (it's ~70MB and not committed).
"""
import os

import pandas as pd
import pytest

from backtest.costs import CostModel
from nqdata.load import DEFAULT_PATH, load_nq
from strategy.params import StrategyParams
from tuning.walkforward import Fold
from tuning.walkforward_p5 import walk_forward_p5

TZ = "America/New_York"


@pytest.mark.skipif(not os.path.exists(DEFAULT_PATH), reason="raw NQ data not present")
def test_walk_forward_p5_runs_on_real_short_slice_small_grid():
    df = load_nq()
    short_slice = df.loc["2023-01-01":"2023-03-31"]
    assert len(short_slice) > 0

    # A tiny 4-combo grid (incl. the required default), entry fixed at
    # StrategyParams() defaults -- varying only exit_mode/vol_filter, same
    # constraint walk_forward_p5 enforces on the real 20-combo grid.
    grid = [
        StrategyParams(),
        StrategyParams(exit_mode="breakeven_1R", vol_filter="p50"),
        StrategyParams(exit_mode="trail_swing", vol_filter="off"),
        StrategyParams(exit_mode="time_stop", vol_filter="p75"),
    ]
    folds = [
        Fold(
            train_start=pd.Timestamp("2023-01-01", tz=TZ),
            train_end=pd.Timestamp("2023-02-15", tz=TZ),
            test_start=pd.Timestamp("2023-02-15", tz=TZ),
            test_end=pd.Timestamp("2023-03-31", tz=TZ),
        )
    ]

    result = walk_forward_p5(short_slice, grid, folds, cost_model=CostModel())

    assert result["grid_size"] == len(grid)
    assert isinstance(result["oos_trades_tuned"], list)
    assert isinstance(result["oos_trades_default"], list)
    assert len(result["folds"]) == 1
    assert len(result["eligibility_table"]) == 1
    assert set(result["stitched_by_combo"].keys()) == set(grid)
    assert set(result["stitched_net_oos_pf_by_combo"].keys()) == set(grid)

    fr = result["folds"][0]
    # the no-leakage structural guarantee: test window strictly after train window.
    assert fr["train_end"] == fr["test_start"]
    assert fr["test_start"] >= fr["train_end"]
    assert fr["train_start"] < fr["test_start"] < fr["test_end"]

    assert len(fr["oos_pf_distribution"]) == len(grid)
    assert 0.0 <= fr["selected_oos_percentile"] <= 1.0
    assert isinstance(fr["fallback_used"], bool)
    assert fr["selected_params"] in grid

    # net metrics (not gross) are present at every level.
    for key in ("net_profit_factor", "net_win_rate", "net_total_pnl", "net_max_drawdown", "net_expectancy", "n_trades"):
        assert key in fr["oos_net_metrics"]
        assert key in fr["oos_net_default_metrics"]

    # if any OOS trades were realized, net_pnl must differ from gross pnl_usd
    # for every trade (CostModel always charges commission_rt > 0) -- a
    # direct end-to-end proof that costs were actually applied, not just
    # structurally present as a field.
    for t in fr["oos_trades"]:
        assert t.net_pnl != pytest.approx(t.pnl_usd)
        assert t.exit_reason != ""
