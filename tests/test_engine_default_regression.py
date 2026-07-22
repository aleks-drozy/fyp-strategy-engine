"""Golden-trade regression lock for the parameterized engine.

From the Phase-4 parameter-tuning spec
(docs/specs/2026-07-13-phase4-parameter-tuning-design.md).

NOT a self-comparison (`backtest(df) == backtest(df, StrategyParams())` would
be tautological -- same code path, always passes even if the refactor
changed every trade). Instead this asserts the parameterized
`backtest(df_slice, StrategyParams())` reproduces trade-for-trade the fixture
captured from the ORIGINAL, pre-refactor engine
(tests/fixtures/phase2_golden_trades.json, see that commit's message for the
exact capture script/commit) on the fixed deterministic slice
`load_nq().loc["2024-05-01":"2024-07-31"]` (53 trades).

Primary lock: runs unconditionally except for the raw-data-absent
escape hatch (the fixture itself is committed; only the ability to
regenerate `df_slice` from raw data requires the ~70MB CSV that isn't
committed).

Secondary confirmation: the full-data default run still yields
605 trades and profit_factor ~= 0.85886, matching the committed
backtest_results.json. Skippable if raw data is absent.
"""
import json
import os

import pytest

from backtest.engine import backtest
from metrics import profit_factor
from nqdata.load import DEFAULT_PATH, load_nq
from strategy.params import StrategyParams

FIXTURE_PATH = "tests/fixtures/phase2_golden_trades.json"

SLICE_START = "2024-05-01"
SLICE_END = "2024-07-31"

FULL_DATA_EXPECTED_TRADES = 605
FULL_DATA_EXPECTED_PF = 0.858860901517489


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
        "r_multiple": tr.r_multiple,
    }


@pytest.mark.skipif(not os.path.exists(DEFAULT_PATH), reason="raw NQ data not present")
def test_default_params_reproduce_golden_trades_trade_for_trade():
    fixture = _load_fixture()
    assert fixture["slice_start"] == SLICE_START
    assert fixture["slice_end"] == SLICE_END
    assert fixture["n_trades"] > 0  # sanity: the slice was never a 0-trade fallback

    df = load_nq()
    df_slice = df.loc[SLICE_START:SLICE_END]

    trades = backtest(df_slice, StrategyParams())

    assert len(trades) == fixture["n_trades"]
    actual = [_trade_to_dict(tr) for tr in trades]
    assert actual == fixture["trades"]


@pytest.mark.skipif(not os.path.exists(DEFAULT_PATH), reason="raw NQ data not present")
def test_default_params_full_data_matches_committed_results():
    df = load_nq()
    trades = backtest(df, StrategyParams())

    assert len(trades) == FULL_DATA_EXPECTED_TRADES
    pnls = [tr.pnl_usd for tr in trades]
    assert profit_factor(pnls) == pytest.approx(FULL_DATA_EXPECTED_PF, rel=1e-9)
