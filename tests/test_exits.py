"""Tests for backtest.exits -- the 4 new (non-fixed_1_5R) exit-mode handlers.

Task 1 Step 3 (docs/superpowers/plans/2026-07-13-phase5-exits-costs-volfilter.md).

`fixed_1_5R` is NOT retested here: `backtest.engine.run_execution` routes it
straight to the ORIGINAL, byte-for-byte unmodified `_try_exit` (see
engine.py's `_manage_open_trade`), so its behavior is already covered by
tests/test_engine.py (unit-level) and tests/test_engine_p5_regression.py
(the golden-fixture regression lock) -- reimplementing/retesting it inside
exits.py would risk two divergent copies of the same logic.

Every scenario hand-builds a `Trade` directly (bypassing signal detection)
with `direction="Long", signal_close=100, stop=95 -> risk=5, entry=100.5`
(entry deliberately != signal_close, to prove R-anchored levels use
signal_close -- reconstructed from stop+risk -- while breakeven explicitly
moves to `entry`). Levels: target(1.5R)=107.5, +1R=105, +3R=115. Bars are
fed one at a time through `exits.manage_bar`, mirroring how
`backtest.engine.run_execution` drives it.
"""

import numpy as np
import pandas as pd
import pytest

import backtest.exits as exits
from backtest.costs import CostModel
from backtest.engine import PT_VALUE
from backtest.trade import Trade

ENTRY = 100.5
STOP = 95.0
SIGNAL_CLOSE = 100.0
RISK = 5.0
TARGET_1_5R = 107.5
R1 = 105.0
R3 = 115.0


def _trade(direction="Long", entry=ENTRY, stop=STOP, signal_close=SIGNAL_CLOSE, rr=1.5):
    risk = abs(signal_close - stop)
    target = signal_close + risk * rr if direction == "Long" else signal_close - risk * rr
    return Trade(
        entry_time=pd.Timestamp("2025-01-21 09:40", tz="US/Eastern"),
        direction=direction,
        entry=entry,
        stop=stop,
        target=target,
        risk=risk,
    )


def _run(trade, exit_mode, bars, swing=3, cost_model=None, start="2025-01-21 09:40"):
    """bars: list of (open, high, low, close) 1-min OHLC tuples. Feeds them
    one at a time through `exits.manage_bar` and stops at the first close."""
    o = np.array([b[0] for b in bars], dtype=float)
    h = np.array([b[1] for b in bars], dtype=float)
    l = np.array([b[2] for b in bars], dtype=float)
    c = np.array([b[3] for b in bars], dtype=float)
    idx = pd.date_range(start, periods=len(bars), freq="1min", tz="US/Eastern")
    state = exits.init_state(trade)
    trades: list[Trade] = []
    counters = {"same_bar_span": 0}
    for i in range(len(bars)):
        exits.manage_bar(trade, state, exit_mode, i, o, h, l, c, idx, swing, PT_VALUE, cost_model, trades, counters)
        if trade.outcome != "Open":
            break
    return state, trades, counters


# --- fixed R/level reconstruction -------------------------------------------


def test_level_reconstructs_signal_close_anchored_r_not_entry():
    tr = _trade()
    assert exits._level(tr, 1.0) == pytest.approx(R1)
    assert exits._level(tr, 1.5) == pytest.approx(TARGET_1_5R) == pytest.approx(tr.target)
    assert exits._level(tr, 3.0) == pytest.approx(R3)


# --- breakeven_1R ------------------------------------------------------------


def test_breakeven_never_reaches_1r_original_stop_applies():
    tr = _trade()
    bars = [
        (100.5, 103, 99, 102),  # quiet, no stop, no +1R
        (100, 101, 94, 95),     # breach original stop (l=94<=95); open=100 -> gap fill min(95,100)=95
    ]
    state, trades, _ = _run(tr, "breakeven_1R", bars)
    assert tr.outcome == "Loss"
    assert tr.exit_reason == "stop"
    assert tr.exit == pytest.approx(95.0)
    assert tr.pnl_usd == pytest.approx(-110.0)
    assert trades == [tr]
    assert state["activated_1r"] is False


def test_breakeven_activates_at_1r_then_pullback_exits_at_entry_near_zero():
    tr = _trade()
    bars = [
        (100.5, 103, 99, 102),   # no +1R yet
        (101, 106, 100, 105),    # h=106 >= 105(+1R); l=100 doesn't breach stop(95) -> activate, stop -> entry
        (101, 102, 100.0, 101),  # pulls back to entry: l=100.0 <= 100.5 -> stop hit
    ]
    state, trades, _ = _run(tr, "breakeven_1R", bars)
    assert tr.exit_reason == "stop"
    assert tr.exit == pytest.approx(ENTRY)  # moved stop = the FILL entry, not signal_close
    assert tr.pnl_usd == pytest.approx(0.0)  # ~0 gross, before costs
    assert trades == [tr]


def test_breakeven_hits_1_5r_target_directly():
    tr = _trade()
    bars = [(100.5, 108, 99, 107)]  # h=108 >= target(107.5)
    _run(tr, "breakeven_1R", bars)
    assert tr.exit_reason == "target"
    assert tr.exit == pytest.approx(TARGET_1_5R)
    assert tr.pnl_usd == pytest.approx((TARGET_1_5R - ENTRY) * PT_VALUE)
    assert tr.outcome == "Win"


def test_breakeven_intrabar_stop_first_no_phantom_activation():
    """Blocker 3: a bar spanning BOTH the current stop and +1R must resolve
    as a stop -- no breakeven activation/credit on that bar."""
    tr = _trade()
    bars = [(100, 110, 90, 100)]  # l=90<=95(stop) AND h=110>=105(+1R), same bar
    state, trades, _ = _run(tr, "breakeven_1R", bars)
    assert tr.exit_reason == "stop"
    assert tr.exit == pytest.approx(95.0)  # gap-through: worse of stop(95)/open(100)
    assert tr.pnl_usd == pytest.approx(-110.0)  # full -1R-ish loss, NOT ~0 (which a phantom breakeven would give)
    assert state["activated_1r"] is False


# --- trail_swing ---------------------------------------------------------


def test_trail_activates_ratchets_and_exits_on_pullback():
    tr = _trade()
    bars = [
        (100, 102, 99, 101),    # i0
        (101, 103, 100, 102),   # i1
        (102, 106, 101, 105),   # i2: h=106>=105 -> activate; swing(l[0:3])=99 -> stop=max(95,99)=99
        (105, 107, 104, 106),   # i3: swing(l[1:4])=100 -> ratchet stop=max(99,100)=100
        (106, 108, 105, 107),   # i4: swing(l[2:5])=101 -> ratchet stop=max(100,101)=101
        (107, 109, 100, 101),   # i5: l=100<=101 -> breach -> exit "trail" @ min(101,107)=101
    ]
    state, trades, _ = _run(tr, "trail_swing", bars, swing=3)
    assert tr.exit_reason == "trail"
    assert tr.exit == pytest.approx(101.0)
    assert tr.pnl_usd == pytest.approx((101.0 - ENTRY) * PT_VALUE)
    assert tr.outcome == "Win"  # trailing exit that locked in a real profit -- NOT mislabeled "Loss"
    assert trades == [tr]


def test_trail_runner_beyond_1_5r_keeps_running_no_fixed_target_cap():
    tr = _trade()
    bars = [
        (100, 102, 99, 101),
        (101, 103, 100, 102),
        (102, 106, 101, 105),   # activate, stop -> 99
        (105, 111, 104, 110),   # h=111 > target(107.5) -- but trail_swing has NO fixed target
    ]
    state, trades, _ = _run(tr, "trail_swing", bars, swing=3)
    assert tr.outcome == "Open"  # still running -- no target check ever fires
    assert trades == []
    assert state["activated_1r"] is True


def test_trail_intrabar_stop_first_no_phantom_activation():
    tr = _trade()
    bars = [(100, 110, 90, 100)]  # spans both original stop(95) and +1R(105)
    state, trades, _ = _run(tr, "trail_swing", bars, swing=3)
    assert tr.exit_reason == "stop"  # not "trail" -- activation never happened on this bar
    assert tr.exit == pytest.approx(95.0)
    assert tr.pnl_usd == pytest.approx(-110.0)
    assert state["activated_1r"] is False


# --- partial_1R ------------------------------------------------------------


def test_partial_pre_activation_full_stop_no_scaleout_credit():
    tr = _trade()
    bars = [(100, 101, 94, 95)]  # breach stop(95) before ever reaching +1R
    state, trades, _ = _run(tr, "partial_1R", bars, swing=3)
    assert tr.exit_reason == "stop"
    assert tr.exit == pytest.approx(95.0)
    assert tr.pnl_usd == pytest.approx(-110.0)  # full single-unit loss, nothing banked
    assert state["half_closed"] is False


def test_partial_intrabar_stop_first_no_phantom_scaleout():
    """Blocker 3: a bar spanning both the stop and +1R must resolve as a
    full stop -- no scale-out credit, no half_closed."""
    tr = _trade()
    bars = [(100, 110, 90, 100)]
    state, trades, _ = _run(tr, "partial_1R", bars, swing=3)
    assert tr.exit_reason == "stop"
    assert tr.pnl_usd == pytest.approx(-110.0)
    assert state["half_closed"] is False


def test_partial_activation_then_remainder_hits_3r_target():
    tr = _trade()
    bars = [
        (100.5, 103, 99, 102),  # no activation yet
        (101, 106, 100, 105),   # h=106>=105 -> activate: bank 0.5*(105-100.5)*20=45; stop->entry; remainder target=115
        (105, 109, 104, 108),   # remainder in flight
        (108, 116, 107, 115),   # h=116>=115(3R) -> remainder target hit
    ]
    state, trades, _ = _run(tr, "partial_1R", bars, swing=3)
    assert tr.exit_reason == "partial_remainder_target"
    assert tr.exit == pytest.approx(115.0)
    leg1_gross = 0.5 * (105.0 - ENTRY) * PT_VALUE   # 45.0
    leg2_gross = 0.5 * (115.0 - ENTRY) * PT_VALUE   # 145.0
    assert tr.pnl_usd == pytest.approx(leg1_gross + leg2_gross)
    assert tr.pnl_usd == pytest.approx(190.0)
    assert tr.outcome == "Win"


def test_partial_activation_then_remainder_hits_breakeven_stop():
    tr = _trade()
    bars = [
        (100.5, 103, 99, 102),
        (101, 106, 100, 105),   # activate: bank 45.0; remainder stop -> entry (100.5)
        (105, 107, 104, 106),
        (105, 106, 99, 100),    # l=99<=100.5 -> remainder stop breached; open=105 -> fill min(100.5,105)=100.5
    ]
    state, trades, _ = _run(tr, "partial_1R", bars, swing=3)
    assert tr.exit_reason == "partial_remainder_stop"
    assert tr.exit == pytest.approx(ENTRY)
    leg1_gross = 0.5 * (105.0 - ENTRY) * PT_VALUE  # 45.0
    leg2_gross = 0.5 * (ENTRY - ENTRY) * PT_VALUE  # 0.0 (remainder exits flat at breakeven)
    assert tr.pnl_usd == pytest.approx(leg1_gross + leg2_gross)
    assert tr.pnl_usd == pytest.approx(45.0)
    assert tr.outcome == "Win"  # the banked half still nets a real, positive blended P&L


def test_partial_cost_model_charges_two_commissions_correct_per_leg_slippage():
    cm = CostModel()
    tr = _trade()
    bars = [
        (100.5, 103, 99, 102),
        (101, 106, 100, 105),   # activate: leg1 gross=45.0 (limit scale-out)
        (105, 107, 104, 106),
        (105, 106, 99, 100),    # remainder stop (market): leg2 gross=0.0
    ]
    _run(tr, "partial_1R", bars, swing=3, cost_model=cm)
    assert tr.pnl_usd == pytest.approx(45.0)
    # leg1 (partial_scaleout, limit, charge_entry=True): commission($5) + entry tick($5) + 0 = $10 -> net 35.0
    # leg2 (partial_remainder_stop, market, charge_entry=False): commission($5) + 0 + exit tick($5) = $10 -> net -10.0
    assert tr.net_pnl == pytest.approx(25.0)


# --- time_stop ---------------------------------------------------------


def test_time_stop_still_open_at_11_00_et_force_exits_at_close():
    tr = _trade()
    tr.entry_time = pd.Timestamp("2025-01-21 10:55", tz="US/Eastern")
    ts = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-01-21 10:58", tz="US/Eastern"),
            pd.Timestamp("2025-01-21 10:59", tz="US/Eastern"),
            pd.Timestamp("2025-01-21 11:00", tz="US/Eastern"),  # first bar >= TIME_STOP_ET
        ]
    )
    bars = [
        (100.5, 103, 99, 102),  # quiet, before 11:00
        (102, 104, 101, 103),   # quiet, before 11:00
        (103, 105, 102, 104),   # quiet price action, but time >= 11:00 -> forced close-price exit
    ]
    o = np.array([b[0] for b in bars]); h = np.array([b[1] for b in bars])
    l = np.array([b[2] for b in bars]); c = np.array([b[3] for b in bars])
    state = exits.init_state(tr)
    trades: list[Trade] = []
    counters = {"same_bar_span": 0}
    for i in range(len(bars)):
        exits.manage_bar(tr, state, "time_stop", i, o, h, l, c, ts, 3, PT_VALUE, None, trades, counters)
        if tr.outcome != "Open":
            break
    assert tr.exit_reason == "time"
    assert tr.exit == pytest.approx(104.0)  # the close of the first bar at/after 11:00
    assert tr.exit_time == ts[2]
    assert tr.pnl_usd == pytest.approx((104.0 - ENTRY) * PT_VALUE)


def test_time_stop_stop_and_target_still_take_priority_over_time():
    tr = _trade()
    tr.entry_time = pd.Timestamp("2025-01-21 10:55", tz="US/Eastern")
    ts = pd.DatetimeIndex([pd.Timestamp("2025-01-21 11:05", tz="US/Eastern")])
    bars = [(100.5, 108, 99, 107)]  # h=108 >= target(107.5), even though time already >= 11:00
    o = np.array([b[0] for b in bars]); h = np.array([b[1] for b in bars])
    l = np.array([b[2] for b in bars]); c = np.array([b[3] for b in bars])
    state = exits.init_state(tr)
    trades: list[Trade] = []
    counters = {"same_bar_span": 0}
    exits.manage_bar(tr, state, "time_stop", 0, o, h, l, c, ts, 3, PT_VALUE, None, trades, counters)
    assert tr.exit_reason == "target"  # not "time"
    assert tr.exit == pytest.approx(TARGET_1_5R)
