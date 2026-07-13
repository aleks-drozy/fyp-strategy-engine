"""Primary regression lock for the Phase-5 exit/cost/vol-filter refactor.

Task 1 Step 6 (docs/superpowers/plans/2026-07-13-phase5-exits-costs-volfilter.md,
Global Constraints -- "Behavior preservation"). Proves the refactor didn't
change the base path: `run_execution` with `exit_mode="fixed_1_5R"`,
`vol_filter` off (no `atr`/`vol_threshold` passed), `cost_model=None`
reproduces `tests/fixtures/phase2_golden_trades.json` trade-for-trade on
`entry_time/direction/entry/stop/target/exit/pnl_usd` -- the SAME fixture
and slice as tests/test_engine_default_regression.py (Phase 4's own lock),
but exercised through `run_execution` directly (not the `backtest()`
convenience wrapper) with the new Phase-5 keyword arguments explicit at
their behavior-preserving defaults, rather than implicitly absent.

This is NOT a self-comparison: the fixture was captured from the ORIGINAL,
pre-Phase-5 engine (see tests/test_engine_default_regression.py's
docstring for the capture provenance) -- reproducing it trade-for-trade
after the exit-mode dispatch/cost-model/vol-filter refactor is the thing
actually being proven.

Runs unconditionally except for the raw-data-absent escape hatch (the
fixture itself is committed; only regenerating `df_slice` from raw data
requires the ~70MB CSV that isn't committed) -- same convention as
tests/test_engine_default_regression.py.
"""
import json
import os

import pytest

from backtest.engine import compute_signal_layer, run_execution
from nqdata.load import DEFAULT_PATH, load_nq
from strategy.params import StrategyParams

FIXTURE_PATH = "tests/fixtures/phase2_golden_trades.json"

SLICE_START = "2024-05-01"
SLICE_END = "2024-07-31"


def _load_fixture() -> dict:
    with open(FIXTURE_PATH) as f:
        return json.load(f)


def _trade_to_dict(tr) -> dict:
    return {
        "entry_time": tr.entry_time.isoformat(),
        "direction": tr.direction,
        "entry": tr.entry,
        "stop": tr.stop,
        "target": tr.target,
        "exit": tr.exit,
        "pnl_usd": tr.pnl_usd,
    }


def _fixture_trade_to_dict(d: dict) -> dict:
    # Same key set as _trade_to_dict, minus the fixture's own r_multiple
    # (this test only asserts the fields the plan names: entry_time/
    # direction/entry/stop/target/exit/pnl_usd).
    return {k: d[k] for k in ("entry_time", "direction", "entry", "stop", "target", "exit", "pnl_usd")}


@pytest.mark.skipif(not os.path.exists(DEFAULT_PATH), reason="raw NQ data not present")
def test_fixed_1_5r_no_vol_filter_no_costs_reproduces_golden_trades_trade_for_trade():
    fixture = _load_fixture()
    assert fixture["slice_start"] == SLICE_START
    assert fixture["slice_end"] == SLICE_END
    assert fixture["n_trades"] > 0  # sanity: the slice was never a 0-trade fallback

    df = load_nq()
    df_slice = df.loc[SLICE_START:SLICE_END]

    params = StrategyParams(exit_mode="fixed_1_5R", vol_filter="off")
    layer = compute_signal_layer(df_slice, params)

    # Phase-5 keyword args explicit at their behavior-preserving defaults --
    # cost_model=None, atr=None, vol_threshold=None -- rather than merely
    # relying on run_execution's own defaults, so this test would fail
    # loudly if those defaults ever changed.
    trades = run_execution(layer, params, fill_mode="next_open", cost_model=None, atr=None, vol_threshold=None)

    assert len(trades) == fixture["n_trades"]
    actual = [_trade_to_dict(tr) for tr in trades]
    expected = [_fixture_trade_to_dict(d) for d in fixture["trades"]]
    assert actual == expected


@pytest.mark.skipif(not os.path.exists(DEFAULT_PATH), reason="raw NQ data not present")
def test_phase5_trade_fields_are_additive_and_inert_on_the_base_path():
    """Every closed trade on the base path gets `exit_reason` set to "stop"
    or "target" (mirrored from `outcome`, since `_try_exit` itself is
    untouched) and `net_pnl == pnl_usd` (no CostModel supplied) -- proving
    the additive schema fields don't perturb the regression-locked values
    themselves, just add two independent, backward-compatible fields."""
    df = load_nq()
    df_slice = df.loc[SLICE_START:SLICE_END]
    params = StrategyParams()
    trades = run_execution(compute_signal_layer(df_slice, params), params)

    assert len(trades) > 0
    for tr in trades:
        assert tr.exit_reason in ("stop", "target")
        assert (tr.exit_reason == "stop") == (tr.outcome == "Loss")
        assert (tr.exit_reason == "target") == (tr.outcome == "Win")
        assert tr.net_pnl == pytest.approx(tr.pnl_usd)
