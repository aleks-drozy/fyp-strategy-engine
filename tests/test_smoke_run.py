"""Smoke test for the core call run_backtest.py makes: backtest(load_nq())
doesn't raise and returns a list[Trade], on a small real 2-day slice.

Task 5 Step 3 (docs/superpowers/plans/2026-07-13-phase2-strategy-engine.md).
This only proves plumbing (real data loads and flows through the engine
without error) -- backtest correctness is covered by tests/test_engine.py's
synthetic fixtures. Skips if the raw Phase-1 NQ CSV isn't present locally
(it's ~70MB and not committed).
"""
import os

import pytest

from backtest.engine import backtest
from backtest.trade import Trade
from nqdata.load import DEFAULT_PATH, load_nq


@pytest.mark.skipif(not os.path.exists(DEFAULT_PATH), reason="raw NQ data not present")
def test_backtest_runs_on_real_two_day_slice():
    df = load_nq()
    two_days = df.loc["2023-01-03":"2023-01-04"]
    assert len(two_days) > 0

    trades = backtest(two_days)

    assert isinstance(trades, list)
    assert all(isinstance(t, Trade) for t in trades)
