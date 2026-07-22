"""Tests for backtest.engine.backtest -- the bar-by-bar execution loop.

From the Phase-2 strategy-engine spec
(docs/specs/2026-07-12-phase2-strategy-engine-design.md).

`compute_ifvg`/`compute_cisd`/`double_confirmation` are already characterized
against Pine logic in tests/test_ifvg.py, test_cisd.py and test_signals.py.
These tests isolate the EXECUTION loop -- fill timing,
stop-first/gap-through exits, the 1-trade/day cap, the session gate, and
no-lookahead -- by monkeypatching `double_confirmation` (to fire an exact,
hand-picked signal at an exact bar position) and `compute_ema` (to a
constant far below/above price, so the long/short EMA filter is trivially
satisfied). The real strategy.session.in_session_mask is left untouched, so
the session gate is exercised for real.
"""

import numpy as np
import pandas as pd
import pytest

import backtest.engine as engine
from backtest.engine import backtest

# --- shared fixtures -------------------------------------------------------


def _session_frame(rows, day="2025-01-21", start="09:30"):
    """rows: list of (open, high, low, close); consecutive 1-min bars."""
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


# 11 bars: 9 bars of swing history (indices 0-8, swing low=95 at index 5),
# a signal bar at index 9 (close=100 -> stop=95, risk=5, target=107.5), and
# a fill bar at index 10 (open=100, no gap) whose range hits the target.
ROWS_LONG_WIN = [
    (99, 100, 98, 99),     # 0
    (99, 100, 98, 99),     # 1
    (99, 100, 98, 99),     # 2  <- swing window l[2:10] starts here
    (99, 100, 98, 99),     # 3
    (99, 100, 98, 99),     # 4
    (99, 100, 95, 97),     # 5  swing low = 95
    (97, 100, 96, 99),     # 6
    (99, 101, 98, 100),    # 7
    (100, 101, 98, 100),   # 8
    (99, 101, 99, 100),    # 9  signal bar, close = 100
    (100, 108, 99, 101),   # 10 fill @ open=100, hits target (107.5) same bar
]


# --- (a) long hits target -> Win with correct pnl_usd -----------------------


def test_long_hits_target_wins_with_correct_pnl(monkeypatch):
    df = _session_frame(ROWS_LONG_WIN)
    sig = [""] * len(df)
    sig[9] = "Long"
    monkeypatch.setattr(engine, "double_confirmation", _fake_signals(sig))
    monkeypatch.setattr(engine, "compute_ema", _fake_ema(low=True))

    trades = backtest(df)

    assert len(trades) == 1
    tr = trades[0]
    assert tr.direction == "Long"
    assert tr.outcome == "Win"
    assert tr.entry == 100.0
    assert tr.stop == 95.0
    assert tr.target == 107.5
    assert tr.exit == 107.5
    assert tr.exit_time == df.index[10]
    assert tr.pnl_usd == pytest.approx((tr.target - tr.entry) * 20)
    assert tr.pnl_usd == pytest.approx(150.0)
    assert tr.r_multiple == pytest.approx(1.5)


# --- (b) stop-first when a bar spans both, with gap-through fill ------------


def test_stop_first_with_gap_through_on_bar_spanning_both(monkeypatch):
    rows = ROWS_LONG_WIN[:10] + [
        (100, 102, 99, 101),  # 10 fill bar: quiet, neither stop nor target hit
        (93, 110, 90, 95),    # 11 gaps below stop AND spans the target
    ]
    df = _session_frame(rows)
    sig = [""] * len(df)
    sig[9] = "Long"
    monkeypatch.setattr(engine, "double_confirmation", _fake_signals(sig))
    monkeypatch.setattr(engine, "compute_ema", _fake_ema(low=True))

    trades = backtest(df)

    assert len(trades) == 1
    tr = trades[0]
    assert tr.entry == 100.0
    assert tr.outcome == "Loss"                # stop wins the same-bar tie-break
    assert tr.exit == 93.0                     # gap-through: worse of stop(95)/open(93)
    assert tr.exit_time == df.index[11]
    assert tr.pnl_usd == pytest.approx(-140.0)
    assert tr.r_multiple == pytest.approx(-1.4)
    assert backtest.same_bar_span_count == 1


# --- (c) max-1-trade/day -----------------------------------------------------


def test_max_one_trade_per_day(monkeypatch):
    rows = ROWS_LONG_WIN + [
        (100, 101, 99, 100),   # 11 quiet spacer
        (100, 102, 95, 100),   # 12 a second, otherwise-valid Long signal bar
        (100, 108, 99, 101),   # 13 would hit target too, if wrongly allowed
    ]
    df = _session_frame(rows)
    sig = [""] * len(df)
    sig[9] = "Long"
    sig[12] = "Long"
    monkeypatch.setattr(engine, "double_confirmation", _fake_signals(sig))
    monkeypatch.setattr(engine, "compute_ema", _fake_ema(low=True))

    trades = backtest(df)

    assert len(trades) == 1
    assert trades[0].entry_time == df.index[10]


# --- (d) no entry outside session --------------------------------------------


def test_no_entry_outside_session(monkeypatch):
    # Same price action as the winning long, but shifted to 09:15-09:25 --
    # entirely before the 09:30 session open.
    df = _session_frame(ROWS_LONG_WIN, start="09:15")
    assert not engine.in_session_mask(df.index).any()  # sanity: truly out of session

    sig = [""] * len(df)
    sig[9] = "Long"
    monkeypatch.setattr(engine, "double_confirmation", _fake_signals(sig))
    monkeypatch.setattr(engine, "compute_ema", _fake_ema(low=True))

    trades = backtest(df)

    assert trades == []


# --- (e) no-lookahead: an in-flight future bar can't change entry/stop/target --


def test_no_lookahead_in_flight_bar_does_not_change_entry_stop_target(monkeypatch):
    """Bars 0-9 reuse ROWS_LONG_WIN's swing history + signal bar (signal at
    index 9 -> entry=100, stop=95, target=107.5). Bar 10 is a quiet fill bar
    (no same-bar exit), so the trade FILLS and stays OPEN across bar 11 and
    is only resolved on a LATER, FIXED bar (13, identical in both frames,
    which hits the target). Bar 11 -- strictly after entry (bar 10) and
    strictly before the exit-determining bar (13), i.e. while the trade is
    in-flight -- is the only bar that differs between the two frames: a wide
    intrabar spike that still stays short of both stop(95) and target(107.5),
    so it can't itself trigger an exit and the two frames are forced to
    resolve on the same later bar. This proves the IN-FLIGHT decision (not
    just a closed trade) ignores a future bar, unlike a mutation placed after
    the trade has already closed."""
    rows_common_tail = [
        (100, 102, 99, 100),   # 12 quiet in-flight buffer bar (identical both frames)
        (100, 108, 99, 101),   # 13 exit-determining bar (identical both frames): hits target
    ]
    rows_quiet = ROWS_LONG_WIN[:10] + [
        (100, 102, 99, 101),   # 10 fill bar: quiet, no same-bar exit
        (101, 103, 100, 101),  # 11 in-flight bar: quiet
    ] + rows_common_tail
    rows_spike = ROWS_LONG_WIN[:10] + [
        (100, 102, 99, 101),   # 10 fill bar: quiet, no same-bar exit (same as quiet frame)
        (101, 107, 96, 101),   # 11 in-flight bar: wide spike, still short of stop/target
    ] + rows_common_tail

    df_quiet = _session_frame(rows_quiet)
    df_spike = _session_frame(rows_spike)

    sig = [""] * len(df_quiet)
    sig[9] = "Long"
    monkeypatch.setattr(engine, "double_confirmation", _fake_signals(sig))
    monkeypatch.setattr(engine, "compute_ema", _fake_ema(low=True))

    trades_quiet = backtest(df_quiet)
    trades_spike = backtest(df_spike)

    assert len(trades_quiet) == 1
    assert len(trades_spike) == 1
    tr_quiet, tr_spike = trades_quiet[0], trades_spike[0]

    # Sanity: the trade really did stay open across the mutated bar (11) and
    # was only resolved on the later, fixed bar (13) -- confirms this isn't
    # the vacuous "a closed trade isn't retroactively rewritten" case.
    assert tr_quiet.entry_time == df_quiet.index[10]
    assert tr_quiet.exit_time == df_quiet.index[13]
    assert tr_spike.entry_time == df_spike.index[10]
    assert tr_spike.exit_time == df_spike.index[13]

    # entry/stop/target are decided at/before the signal bar (9) and locked
    # in at the fill bar (10) -- both strictly before the mutated in-flight
    # bar (11) -- so they must be identical no matter what bar 11 does.
    # (exit/outcome/pnl legitimately match too here since the
    # exit-determining bar is unchanged, but the invariant under test is
    # entry/stop/target.)
    assert (tr_quiet.entry, tr_quiet.stop, tr_quiet.target) == (
        tr_spike.entry, tr_spike.stop, tr_spike.target,
    )
