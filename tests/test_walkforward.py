"""Tests for tuning.walkforward -- the leakage-critical core of Phase 4.

Phase 4, Task 2 Step 3 (docs/superpowers/plans/2026-07-13-phase4-parameter-tuning.md).

Deliberately data-light: synthetic frames and tiny grids only. The real
`make_folds()` dates (2023-2025) are exercised as pure date arithmetic (no
data needed); the positional-slice-alignment and walk_forward-integration
tests use small synthetic OHLCV frames with their OWN small fold windows,
so nothing here needs the (gitignored, ~70MB, not-always-present) raw NQ
CSV or a full 144x4 run.
"""

import numpy as np
import pandas as pd
import pytest

from backtest.trade import Trade
from strategy.params import StrategyParams
from tuning.walkforward import (
    MIN_IS_TRADES,
    Fold,
    _slice_layer,
    _window_trades,
    make_folds,
    select_params,
    walk_forward,
)

TZ = "America/New_York"


# --- shared synthetic-data helpers -------------------------------------------


def _multi_day_session_df(n_days: int, bars_per_day: int = 20, start_day: str = "2024-01-01", seed: int = 0) -> pd.DataFrame:
    """`n_days` trading days (weekdays only) of `bars_per_day` 1-min bars
    starting 09:30 ET each day -- entirely inside the default 09:30-10:30
    session, so `in_session_mask` is True throughout. A fresh small random
    walk per day (not compounded across days) keeps prices well away from
    zero regardless of `n_days`.
    """
    rng = np.random.default_rng(seed)
    frames = []
    day = pd.Timestamp(start_day, tz=TZ)
    count = 0
    while count < n_days:
        if day.dayofweek < 5:  # Mon-Fri
            idx = pd.date_range(f"{day.date()} 09:30", periods=bars_per_day, freq="1min", tz=TZ)
            base = 100.0 + count * 0.1
            walk = np.cumsum(rng.normal(0, 0.3, bars_per_day))
            close = base + walk
            open_ = np.r_[close[0], close[:-1]]
            high = np.maximum(open_, close) + rng.uniform(0, 0.5, bars_per_day)
            low = np.minimum(open_, close) - rng.uniform(0, 0.5, bars_per_day)
            frames.append(
                pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1}, index=idx)
            )
            count += 1
        day += pd.Timedelta(days=1)
    return pd.concat(frames)


def _raw_layer_from_df(df: pd.DataFrame) -> dict:
    """A minimal layer dict (arbitrary sig/ema arrays) for exercising
    `_slice_layer` in isolation from the real signal-computation logic."""
    n = len(df)
    return {
        "sig": np.arange(n),
        "ema_v": df["close"].to_numpy(dtype=float),
        "sess": np.ones(n, dtype=bool),
        "o": df["open"].to_numpy(dtype=float),
        "h": df["high"].to_numpy(dtype=float),
        "l": df["low"].to_numpy(dtype=float),
        "c": df["close"].to_numpy(dtype=float),
        "days": df.index.tz_convert("America/New_York").date,
        "index": df.index,
    }


def _params(tag: float) -> StrategyParams:
    """Distinct, hashable StrategyParams stand-ins for selection tests."""
    return StrategyParams(rr=1.0 + tag)


def _trade_at(ts: pd.Timestamp) -> Trade:
    return Trade(entry_time=ts, direction="Long", entry=1.0, stop=1.0, target=1.0, risk=1.0)


# --- make_folds(): exact half-open dates -------------------------------------


def test_make_folds_exact_half_open_dates():
    folds = make_folds()
    assert len(folds) == 4
    expected = [
        ("2023-01-01", "2024-01-01", "2024-07-01"),
        ("2023-07-01", "2024-07-01", "2025-01-01"),
        ("2024-01-01", "2025-01-01", "2025-07-01"),
        ("2024-07-01", "2025-07-01", "2025-12-12"),
    ]
    for fold, (train_start, test_start, test_end) in zip(folds, expected):
        assert fold.train_start == pd.Timestamp(train_start, tz=TZ)
        assert fold.train_end == pd.Timestamp(test_start, tz=TZ)
        assert fold.test_start == pd.Timestamp(test_start, tz=TZ)
        assert fold.test_end == pd.Timestamp(test_end, tz=TZ)
        assert fold.train_end == fold.test_start  # half-open contiguity: train=[.,test_start), test=[test_start,.)


def test_make_folds_test_windows_contiguous_non_overlapping_cover_oos_span():
    folds = make_folds()
    assert folds[0].test_start == pd.Timestamp("2024-01-01", tz=TZ)
    assert folds[-1].test_end == pd.Timestamp("2025-12-12", tz=TZ)
    for a, b in zip(folds, folds[1:]):
        assert a.test_end == b.test_start  # contiguous, non-overlapping


def test_make_folds_dec31_session_bar_belongs_to_train_not_lost():
    """A Dec-31 09:30 ET bar sits one half-open window below the Jan-1
    00:00 boundary -- it must land in F1's TRAIN window, not be dropped or
    miscounted into test."""
    f1 = make_folds()[0]
    dec31_bar = pd.DatetimeIndex([pd.Timestamp("2023-12-31 09:30", tz=TZ)])

    a_tr, b_tr = dec31_bar.searchsorted(f1.train_start, side="left"), dec31_bar.searchsorted(f1.train_end, side="left")
    assert b_tr - a_tr == 1  # included in [train_start, train_end)

    a_te, b_te = dec31_bar.searchsorted(f1.test_start, side="left"), dec31_bar.searchsorted(f1.test_end, side="left")
    assert b_te - a_te == 0  # NOT included in [test_start, test_end)


# --- positional-slice alignment (the 1-bar-leak guard) -----------------------


def test_positional_slice_alignment_half_open_disjoint_and_bar_for_bar():
    df = _multi_day_session_df(n_days=12, bars_per_day=15, seed=1)
    layer = _raw_layer_from_df(df)

    # Split at a day boundary (midnight) so the boundary never coincides
    # with an actual bar timestamp (bars only exist at 09:30+).
    start = df.index[0].normalize()
    mid = df.index[len(df) // 2].normalize()
    end = df.index[-1].normalize() + pd.Timedelta(days=1)

    train = _slice_layer(layer, start, mid)
    test = _slice_layer(layer, mid, end)

    # Independent ground truth via boolean masking (not searchsorted) --
    # the half-open window semantics, computed a different way.
    expected_train = df.loc[(df.index >= start) & (df.index < mid)]
    expected_test = df.loc[(df.index >= mid) & (df.index < end)]

    assert train["index"].equals(expected_train.index)
    assert test["index"].equals(expected_test.index)
    assert np.array_equal(train["c"], expected_train["close"].to_numpy())
    assert np.array_equal(test["c"], expected_test["close"].to_numpy())

    a, b = layer["index"].searchsorted(start, side="left"), layer["index"].searchsorted(mid, side="left")
    assert len(train["index"]) == b - a
    a2, b2 = layer["index"].searchsorted(mid, side="left"), layer["index"].searchsorted(end, side="left")
    assert len(test["index"]) == b2 - a2

    assert train["index"].max() < test["index"].min()  # strictly disjoint
    assert len(train["index"]) + len(test["index"]) == len(df)  # full coverage, no gaps or dups


def test_slice_layer_clips_to_data_edge_when_boundary_past_last_bar():
    """F4's real test_end (2025-12-12) is past the data's last bar -- this
    must clip to the data edge (i.e. include everything up to the last
    bar), not raise or silently drop the tail."""
    df = _multi_day_session_df(n_days=5, bars_per_day=10, seed=2)
    layer = _raw_layer_from_df(df)
    far_future = df.index[-1] + pd.Timedelta(days=365)

    sliced = _slice_layer(layer, df.index[0], far_future)

    assert len(sliced["index"]) == len(df)
    assert sliced["index"].equals(df.index)


# --- select_params: floor, tie-breaks, fallback -------------------------------


def test_select_params_picks_max_pf_within_floor():
    results = [
        {"params": _params(0), "profit_factor": 1.2, "n_trades": 60, "max_drawdown": 500.0},
        {"params": _params(1), "profit_factor": 1.5, "n_trades": 55, "max_drawdown": 300.0},
        {"params": _params(2), "profit_factor": 1.1, "n_trades": 200, "max_drawdown": 100.0},
    ]
    sel = select_params(results)
    assert sel.params == _params(1)
    assert sel.fallback_used is False


def test_select_params_ignores_higher_pf_combo_below_floor():
    winner = _params(0)
    lucky_low_n = _params(1)
    results = [
        {"params": winner, "profit_factor": 1.3, "n_trades": MIN_IS_TRADES, "max_drawdown": 200.0},
        {"params": lucky_low_n, "profit_factor": 5.0, "n_trades": MIN_IS_TRADES - 1, "max_drawdown": 10.0},
    ]
    sel = select_params(results)
    assert sel.params == winner
    assert sel.fallback_used is False


def test_select_params_tie_break_higher_n_then_lower_drawdown():
    a, b = _params(0), _params(1)
    tied_pf = [
        {"params": a, "profit_factor": 1.4, "n_trades": 60, "max_drawdown": 500.0},
        {"params": b, "profit_factor": 1.4, "n_trades": 90, "max_drawdown": 500.0},
    ]
    assert select_params(tied_pf).params == b  # higher n_trades wins the first tie-break

    c, d = _params(2), _params(3)
    tied_pf_and_n = [
        {"params": c, "profit_factor": 1.4, "n_trades": 90, "max_drawdown": 700.0},
        {"params": d, "profit_factor": 1.4, "n_trades": 90, "max_drawdown": 300.0},
    ]
    assert select_params(tied_pf_and_n).params == d  # lower max_drawdown wins the second tie-break


def test_select_params_fallback_when_none_meet_floor():
    results = [
        {"params": _params(0), "profit_factor": 3.0, "n_trades": 10, "max_drawdown": 5.0},
        {"params": _params(1), "profit_factor": 2.0, "n_trades": MIN_IS_TRADES - 1, "max_drawdown": 5.0},
    ]
    sel = select_params(results)
    assert sel.params == StrategyParams()
    assert sel.fallback_used is True


def test_select_params_empty_results_falls_back():
    sel = select_params([])
    assert sel.params == StrategyParams()
    assert sel.fallback_used is True


# --- _window_trades: no-leakage on trades -------------------------------------


def test_window_trades_half_open_filtering():
    inside_start = pd.Timestamp("2024-01-01 09:30", tz=TZ)
    inside_mid = pd.Timestamp("2024-03-01 09:30", tz=TZ)
    at_end_boundary = pd.Timestamp("2024-07-01 00:00", tz=TZ)  # excluded: belongs to the NEXT window
    trades = [_trade_at(inside_start), _trade_at(inside_mid), _trade_at(at_end_boundary)]

    windowed = _window_trades(trades, pd.Timestamp("2024-01-01", tz=TZ), pd.Timestamp("2024-07-01", tz=TZ))

    assert windowed == trades[:2]
    assert trades[2] not in windowed


def test_window_trades_test_window_trade_never_leaks_into_train_filter():
    train_start = pd.Timestamp("2023-01-01", tz=TZ)
    train_end = pd.Timestamp("2024-01-01", tz=TZ)  # == test_start (half-open contiguity)
    test_end = pd.Timestamp("2024-07-01", tz=TZ)

    test_window_trade = _trade_at(pd.Timestamp("2024-03-15 09:45", tz=TZ))
    all_trades = [test_window_trade]

    assert _window_trades(all_trades, train_start, train_end) == []
    assert _window_trades(all_trades, train_end, test_end) == [test_window_trade]


# --- walk_forward integration: tiny grid, synthetic data, structural + no-leakage ---


def test_walk_forward_structure_and_no_leakage_tiny_grid_synthetic_data():
    df = _multi_day_session_df(n_days=40, bars_per_day=20, seed=7)

    grid = [
        StrategyParams(),  # required: fallback target + OOS-default baseline
        StrategyParams(fvg_threshold=0.02, rr=2.0, ema_length=10, swing_lookback=5),
        StrategyParams(fvg_threshold=0.05, rr=3.0, ema_length=50, swing_lookback=12),
    ]

    # Small custom folds carved out of the synthetic df's own date range --
    # NOT make_folds()'s real 2023-2025 dates, which need the full dataset.
    days = sorted(set(df.index.normalize()))
    b1 = days[len(days) // 3]
    b2 = days[2 * len(days) // 3]
    folds = [
        Fold(train_start=df.index[0].normalize(), train_end=b1, test_start=b1, test_end=b2),
        Fold(train_start=b1, train_end=b2, test_start=b2, test_end=days[-1] + pd.Timedelta(days=1)),
    ]

    result = walk_forward(df, grid, folds)

    assert result["grid_size"] == len(grid)
    assert result["min_is_trades"] == MIN_IS_TRADES
    assert len(result["folds"]) == 2
    assert isinstance(result["oos_trades_tuned"], list)
    assert isinstance(result["oos_trades_default"], list)

    for fr in result["folds"]:
        # no-leakage structure: train ends exactly where test begins, and
        # test never precedes train (strict fold ordering).
        assert fr["train_end"] == fr["test_start"]
        assert fr["test_start"] >= fr["train_end"]

        # null control: every grid combo's OOS PF was recorded.
        assert len(fr["oos_pf_distribution"]) == len(grid)
        assert 0.0 <= fr["selected_oos_percentile"] <= 1.0
        assert isinstance(fr["fallback_used"], bool)
        assert fr["selected_params"] in grid

        # independent no-leakage re-check: filtering this fold's reported
        # OOS trades to its own test window must be a no-op.
        assert _window_trades(fr["oos_trades"], fr["test_start"], fr["test_end"]) == fr["oos_trades"]


def test_walk_forward_rejects_grid_without_default():
    df = _multi_day_session_df(n_days=6, bars_per_day=10, seed=3)
    grid = [StrategyParams(rr=2.0)]  # missing StrategyParams() -- violates the contract
    folds = [
        Fold(
            train_start=df.index[0].normalize(),
            train_end=df.index[-1].normalize(),
            test_start=df.index[-1].normalize(),
            test_end=df.index[-1].normalize() + pd.Timedelta(days=1),
        )
    ]
    with pytest.raises(AssertionError):
        walk_forward(df, grid, folds)
