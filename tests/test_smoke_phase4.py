"""Smoke test for the core call run_phase4.py makes: walk_forward(load_nq(),
grid, folds) doesn't raise, returns the expected structure, and each fold's
test window is strictly after its train window -- on a SMALL grid over a
SHORT real slice (not the full 144x4 sweep).

Task 3 Step 4 (docs/superpowers/plans/2026-07-13-phase4-parameter-tuning.md).
This only proves plumbing (real data flows through walk_forward end-to-end
without error); leakage/selection correctness is covered by
tests/test_walkforward.py's synthetic-data tests. Skips if the raw Phase-1
NQ CSV isn't present locally (it's ~70MB and not committed).
"""
import os

import pandas as pd
import pytest

from nqdata.load import DEFAULT_PATH, load_nq
from strategy.params import StrategyParams
from tuning.walkforward import Fold, walk_forward

TZ = "America/New_York"


@pytest.mark.skipif(not os.path.exists(DEFAULT_PATH), reason="raw NQ data not present")
def test_walk_forward_runs_on_real_short_slice_small_grid():
    df = load_nq()
    short_slice = df.loc["2023-01-01":"2023-03-31"]
    assert len(short_slice) > 0

    # A tiny 4-combo grid (incl. the required default) over a single short
    # fold carved out of the slice's own date range.
    grid = [
        StrategyParams(),
        StrategyParams(fvg_threshold=0.02, rr=2.0, ema_length=10, swing_lookback=5),
        StrategyParams(fvg_threshold=0.05, rr=1.0, ema_length=50, swing_lookback=12),
        StrategyParams(fvg_threshold=0.10, rr=3.0, ema_length=20, swing_lookback=8),
    ]
    folds = [
        Fold(
            train_start=pd.Timestamp("2023-01-01", tz=TZ),
            train_end=pd.Timestamp("2023-02-15", tz=TZ),
            test_start=pd.Timestamp("2023-02-15", tz=TZ),
            test_end=pd.Timestamp("2023-03-31", tz=TZ),
        )
    ]

    result = walk_forward(short_slice, grid, folds)

    assert result["grid_size"] == len(grid)
    assert isinstance(result["oos_trades_tuned"], list)
    assert isinstance(result["oos_trades_default"], list)
    assert len(result["folds"]) == 1

    fr = result["folds"][0]
    # the no-leakage structural guarantee: test window strictly after train window.
    assert fr["train_end"] == fr["test_start"]
    assert fr["test_start"] >= fr["train_end"]
    assert fr["train_start"] < fr["test_start"] < fr["test_end"]

    assert len(fr["oos_pf_distribution"]) == len(grid)
    assert 0.0 <= fr["selected_oos_percentile"] <= 1.0
    assert isinstance(fr["fallback_used"], bool)
    assert fr["selected_params"] in grid
