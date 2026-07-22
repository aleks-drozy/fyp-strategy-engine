"""Integration tests for the Phase-5 wiring inside backtest.engine.run_execution
that isn't otherwise exercised: the `atr`/`vol_threshold` entry gate (incl.
the `len(atr) == len(sig)` guard), and the full per-bar-loop dispatch to a
non-fixed_1_5R exit mode / a CostModel, end to end (not just the isolated
`exits.manage_bar` unit tests in tests/test_exits.py).

From the Phase-5 exits/costs/vol-filter spec
(docs/specs/2026-07-13-phase5-exits-costs-volfilter-design.md).
Reuses test_engine.py's monkeypatch style: a fixed, hand-picked signal via
`double_confirmation` and a constant EMA far below/above price so the
long/short filter is trivially satisfied, leaving only the execution-loop
wiring under test.
"""

import numpy as np
import pandas as pd
import pytest

import backtest.engine as engine
from backtest.costs import CostModel
from backtest.engine import backtest, compute_signal_layer, run_execution
from strategy.params import StrategyParams


def _session_frame(rows, day="2025-01-21", start="09:30"):
    idx = pd.date_range(f"{day} {start}", periods=len(rows), freq="1min", tz="US/Eastern")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1
    return df


def _fake_signals(sig_list):
    def _f(ifvg, cisd):
        return pd.Series(list(sig_list))
    return _f


def _fake_ema(low=True):
    def _f(df, period=20):
        val = -1_000_000.0 if low else 1_000_000.0
        return pd.Series(np.full(len(df), val), index=df.index)
    return _f


# Same fixture as test_engine.py's ROWS_LONG_WIN: signal at index 9
# (close=100 -> stop=95, risk=5, target=107.5), fills at index 10's open
# (100, no gap) and hits the 1.5R target same bar.
ROWS_LONG_WIN = [
    (99, 100, 98, 99), (99, 100, 98, 99), (99, 100, 98, 99), (99, 100, 98, 99),
    (99, 100, 98, 99), (99, 100, 95, 97), (97, 100, 96, 99), (99, 101, 98, 100),
    (100, 101, 98, 100), (99, 101, 99, 100), (100, 108, 99, 101),
]


def _rig(df, sig_bar=9):
    sig = [""] * len(df)
    sig[sig_bar] = "Long"
    return _fake_signals(sig)


# --- vol filter gate -------------------------------------------------------


def test_vol_threshold_skips_entry_when_atr_below_threshold(monkeypatch):
    df = _session_frame(ROWS_LONG_WIN)
    monkeypatch.setattr(engine, "double_confirmation", _rig(df))
    monkeypatch.setattr(engine, "compute_ema", _fake_ema(low=True))

    layer = compute_signal_layer(df)
    atr = np.full(len(df), 1.0)  # every bar's atr == 1.0, below the threshold

    trades = run_execution(layer, StrategyParams(), atr=atr, vol_threshold=2.0)

    assert trades == []  # the signal at bar 9 never becomes a pending entry


def test_vol_threshold_allows_entry_when_atr_at_or_above_threshold(monkeypatch):
    df = _session_frame(ROWS_LONG_WIN)
    monkeypatch.setattr(engine, "double_confirmation", _rig(df))
    monkeypatch.setattr(engine, "compute_ema", _fake_ema(low=True))

    layer = compute_signal_layer(df)
    atr = np.full(len(df), 5.0)  # every bar's atr >= threshold

    trades = run_execution(layer, StrategyParams(), atr=atr, vol_threshold=2.0)

    assert len(trades) == 1
    assert trades[0].outcome == "Win"


def test_atr_none_vol_threshold_none_is_a_no_op_gate(monkeypatch):
    """The default (both None) must behave exactly like no gate at all --
    this is what keeps the base path byte-identical."""
    df = _session_frame(ROWS_LONG_WIN)
    monkeypatch.setattr(engine, "double_confirmation", _rig(df))
    monkeypatch.setattr(engine, "compute_ema", _fake_ema(low=True))

    layer = compute_signal_layer(df)
    trades = run_execution(layer, StrategyParams(), atr=None, vol_threshold=None)

    assert len(trades) == 1


def test_atr_length_mismatch_raises_assertion(monkeypatch):
    df = _session_frame(ROWS_LONG_WIN)
    monkeypatch.setattr(engine, "double_confirmation", _rig(df))
    monkeypatch.setattr(engine, "compute_ema", _fake_ema(low=True))

    layer = compute_signal_layer(df)
    short_atr = np.full(len(df) - 1, 5.0)  # deliberately mis-sized

    with pytest.raises(AssertionError):
        run_execution(layer, StrategyParams(), atr=short_atr, vol_threshold=1.0)


# --- full-pipeline dispatch to a non-fixed_1_5R exit mode -------------------


def test_full_pipeline_dispatches_to_breakeven_1r_and_sets_exit_reason(monkeypatch):
    """End-to-end sanity for the engine.py wiring (mgmt_state creation at
    fill time, dispatch away from `_try_exit`) -- the per-mode price/reason
    math itself is unit-tested in tests/test_exits.py."""
    rows = ROWS_LONG_WIN[:10] + [
        (100, 106, 99, 105),   # 10: fill @ open=100; h=106 reaches +1R (105) same bar, no stop breach -> activates
        (101, 102, 100.0, 101),  # 11: pulls back to entry (100) -> breakeven stop hit
    ]
    df = _session_frame(rows)
    monkeypatch.setattr(engine, "double_confirmation", _rig(df))
    monkeypatch.setattr(engine, "compute_ema", _fake_ema(low=True))

    params = StrategyParams(exit_mode="breakeven_1R")
    trades = backtest(df, params)

    assert len(trades) == 1
    tr = trades[0]
    assert tr.exit_reason == "stop"
    assert tr.entry == 100.0
    assert tr.exit == pytest.approx(100.0)  # breakeven = entry
    assert tr.net_pnl == pytest.approx(tr.pnl_usd)  # no cost_model supplied


def test_full_pipeline_applies_cost_model_on_fixed_1_5r_path(monkeypatch):
    df = _session_frame(ROWS_LONG_WIN)
    monkeypatch.setattr(engine, "double_confirmation", _rig(df))
    monkeypatch.setattr(engine, "compute_ema", _fake_ema(low=True))

    cm = CostModel()
    trades = backtest(df, StrategyParams(), cost_model=cm)

    assert len(trades) == 1
    tr = trades[0]
    assert tr.exit_reason == "target"
    assert tr.pnl_usd == pytest.approx(150.0)  # unchanged from test_engine.py's equivalent case
    # target = limit fill: commission($5) + entry tick($5) + 0 exit slippage = $10
    assert tr.net_pnl == pytest.approx(140.0)
