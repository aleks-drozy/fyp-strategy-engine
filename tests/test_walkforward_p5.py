"""Tests for tuning.walkforward_p5 -- the leakage-critical core of Phase 5.

From the Phase-5 exits/costs/vol-filter spec
(docs/specs/2026-07-13-phase5-exits-costs-volfilter-design.md).

Deliberately data-light, mirroring tests/test_walkforward.py's convention:
synthetic frames and tiny grids only. No 20x4 sweep over full real data --
this only proves the leakage-critical wiring (train-only ATR% percentiles,
net-not-gross routing, the pre-filter eligibility floor, ATR%-slice
alignment) at small scale.
"""

import numpy as np
import pandas as pd
import pytest

from backtest.costs import CostModel
from backtest.trade import Trade
from metrics import profit_factor
from strategy.params import StrategyParams
from tuning.walkforward import MIN_IS_TRADES, Fold
from tuning.walkforward_p5 import (
    _in_session_signal_count,
    _net_metrics,
    _precompute_p5,
    _slice_layer_p5,
    _vol_threshold,
    select_params_p5,
    walk_forward_p5,
)

TZ = "America/New_York"


# --- shared synthetic-data helpers (same pattern as tests/test_walkforward.py) --


def _multi_day_session_df(n_days: int, bars_per_day: int = 20, start_day: str = "2024-01-01", seed: int = 0) -> pd.DataFrame:
    """`n_days` trading days (weekdays only) of `bars_per_day` 1-min bars
    starting 09:30 ET each day -- entirely inside the default 09:30-10:30
    session. A fresh small random walk per day keeps prices away from zero."""
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


def _raw_layer_from_df(df: pd.DataFrame, sig=None, sess=None, atr_pct=None) -> dict:
    """A minimal layer dict with caller-controlled `sig`/`sess`/`atr_pct` --
    for exercising `_slice_layer_p5`/`_vol_threshold` in isolation from the
    real strategy-signal pipeline (same style as test_walkforward.py's
    `_raw_layer_from_df`, extended with `atr_pct`)."""
    n = len(df)
    return {
        "sig": sig if sig is not None else np.array([""] * n, dtype=object),
        "ema_v": df["close"].to_numpy(dtype=float),
        "sess": sess if sess is not None else np.ones(n, dtype=bool),
        "o": df["open"].to_numpy(dtype=float),
        "h": df["high"].to_numpy(dtype=float),
        "l": df["low"].to_numpy(dtype=float),
        "c": df["close"].to_numpy(dtype=float),
        "days": df.index.tz_convert("America/New_York").date,
        "index": df.index,
        "atr_pct": atr_pct if atr_pct is not None else np.zeros(n, dtype=float),
    }


def _flat_df(n: int, start: str = "2024-01-02 09:30") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1min", tz=TZ)
    return pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1}, index=idx)


def _trade_with_pnls(gross: float, net: float) -> Trade:
    return Trade(
        entry_time=pd.Timestamp("2024-01-02 09:31", tz=TZ),
        direction="Long",
        entry=100.0,
        stop=99.0,
        target=101.5,
        risk=1.0,
        pnl_usd=gross,
        net_pnl=net,
        outcome="Win" if gross > 0 else "Loss",
        exit_reason="target",
    )


# --- ATR%-slice alignment (Blocker 4) ---------------------------------------


def test_atr_pct_slice_is_bar_aligned_with_df_loc_window():
    df = _multi_day_session_df(n_days=12, bars_per_day=15, seed=1)
    layer = _precompute_p5(df)

    start = df.index[0].normalize()
    mid = df.index[len(df) // 2].normalize()

    sliced = _slice_layer_p5(layer, start, mid)
    expected_window = df.loc[(df.index >= start) & (df.index < mid)]

    full_atr_pct = pd.Series(layer["atr_pct"], index=df.index)
    expected_atr_pct = full_atr_pct.loc[expected_window.index].to_numpy()

    assert sliced["index"].equals(expected_window.index)
    assert len(sliced["atr_pct"]) == len(sliced["sig"]) == len(expected_window)
    assert np.array_equal(sliced["atr_pct"], expected_atr_pct)


def test_slice_layer_p5_disjoint_and_full_coverage_like_phase4():
    df = _multi_day_session_df(n_days=12, bars_per_day=15, seed=1)
    layer = _precompute_p5(df)

    start = df.index[0].normalize()
    mid = df.index[len(df) // 2].normalize()
    end = df.index[-1].normalize() + pd.Timedelta(days=1)

    train = _slice_layer_p5(layer, start, mid)
    test = _slice_layer_p5(layer, mid, end)

    assert train["index"].max() < test["index"].min()  # strictly disjoint
    assert len(train["index"]) + len(test["index"]) == len(df)  # full coverage, no gaps/dups


# --- vol-filter threshold: leak-free, train-only (pre-registered) ----------


def test_vol_threshold_off_is_none():
    df = _flat_df(10)
    layer = _raw_layer_from_df(df)
    assert _vol_threshold(layer, "off") is None


def test_vol_threshold_empty_population_degrades_to_none():
    df = _flat_df(10)
    layer = _raw_layer_from_df(df)  # sig all "" -> empty population
    assert _vol_threshold(layer, "p50") is None


def test_vol_threshold_train_only_invariant_to_test_window_atr_mutation():
    n = 40
    df = _flat_df(n)
    idx = df.index
    train_end = idx[20]  # train = bars[0:20], test = bars[20:40]

    sig = np.array(["Long" if i % 3 == 0 else "" for i in range(n)], dtype=object)
    sess = np.ones(n, dtype=bool)
    atr_a = np.arange(n, dtype=float)  # 0..39

    layer_a = _raw_layer_from_df(df, sig=sig, sess=sess, atr_pct=atr_a)
    thr_a = _vol_threshold(_slice_layer_p5(layer_a, idx[0], train_end), "p50")

    # Mutate ONLY the TEST-window portion of ATR% -- train values untouched.
    atr_b = atr_a.copy()
    atr_b[20:] += 1000.0
    layer_b = _raw_layer_from_df(df, sig=sig, sess=sess, atr_pct=atr_b)
    thr_b = _vol_threshold(_slice_layer_p5(layer_b, idx[0], train_end), "p50")

    assert thr_a == pytest.approx(thr_b)  # invariant to test-window-only mutation

    # Sanity: mutating the TRAIN-window portion DOES move the threshold --
    # proves this isn't vacuously passing (e.g. an accidentally-empty
    # population, or a threshold function that ignores atr_pct entirely).
    atr_c = atr_a.copy()
    atr_c[:20] += 1000.0
    layer_c = _raw_layer_from_df(df, sig=sig, sess=sess, atr_pct=atr_c)
    thr_c = _vol_threshold(_slice_layer_p5(layer_c, idx[0], train_end), "p50")

    assert thr_c != pytest.approx(thr_a)


def test_vol_threshold_equals_percentile_of_train_signal_population_directly():
    n = 30
    df = _flat_df(n)
    sig = np.array(["Long" if i % 2 == 0 else "" for i in range(n)], dtype=object)
    sess = np.ones(n, dtype=bool)
    atr = np.arange(n, dtype=float)
    layer = _raw_layer_from_df(df, sig=sig, sess=sess, atr_pct=atr)

    train = _slice_layer_p5(layer, df.index[0], df.index[15])  # bars [0,15)
    expected_population = atr[:15][sig[:15] != ""]  # every-other bar, in-session

    assert _vol_threshold(train, "p50") == pytest.approx(np.percentile(expected_population, 50))
    assert _vol_threshold(train, "p25") == pytest.approx(np.percentile(expected_population, 25))
    assert _vol_threshold(train, "p75") == pytest.approx(np.percentile(expected_population, 75))


# --- pre-filter in-session signal count (Blocker 5) -------------------------


def test_in_session_signal_count_ignores_out_of_session_and_blank_bars():
    n = 10
    df = _flat_df(n)
    sig = np.array(["Long", "", "Short", "Long", "", "", "Short", "Long", "", "Long"], dtype=object)
    sess = np.array([True, True, True, False, True, True, False, True, True, True])  # 2 out-of-session bars
    layer = _raw_layer_from_df(df, sig=sig, sess=sess)

    # signal bars: indices 0,2,3,6,7,9 -> 6 total; drop out-of-session 3 and 6 -> 4 remain
    assert _in_session_signal_count(layer) == 4


# --- net (not gross) selection (Blocker 2) ----------------------------------


def test_net_metrics_uses_net_pnl_not_gross_pnl_usd():
    trades = [_trade_with_pnls(100, 10), _trade_with_pnls(-10, -100)]
    m = _net_metrics(trades)
    gross_pf = profit_factor([t.pnl_usd for t in trades])
    net_pf = profit_factor([t.net_pnl for t in trades])
    assert m["net_profit_factor"] == pytest.approx(net_pf)
    assert m["net_profit_factor"] != pytest.approx(gross_pf)
    assert m["net_total_pnl"] == pytest.approx(-90.0)  # 10 + (-100), NOT 100 + (-10)


def test_select_params_p5_uses_net_pf_not_gross_pf():
    combo_a = StrategyParams(exit_mode="fixed_1_5R", vol_filter="off")
    combo_b = StrategyParams(exit_mode="trail_swing", vol_filter="off")

    # combo_a: gross PF is HIGHER, net PF is LOWER (heavy relative costs).
    trades_a = [_trade_with_pnls(100, 10), _trade_with_pnls(100, 10), _trade_with_pnls(-10, -100)]
    # combo_b: gross PF is LOWER, net PF is HIGHER (costs barely dent it).
    trades_b = [_trade_with_pnls(60, 55), _trade_with_pnls(60, 55), _trade_with_pnls(-50, -55)]

    gross_pf_a = profit_factor([t.pnl_usd for t in trades_a])
    gross_pf_b = profit_factor([t.pnl_usd for t in trades_b])
    assert gross_pf_a > gross_pf_b  # sanity: gross ranking would pick A

    m_a, m_b = _net_metrics(trades_a), _net_metrics(trades_b)
    m_a["params"], m_b["params"] = combo_a, combo_b
    assert m_b["net_profit_factor"] > m_a["net_profit_factor"]  # net ranking favors B

    selection = select_params_p5([m_a, m_b], eligible=True)
    assert selection.params == combo_b  # selection followed NET, not gross
    assert selection.fallback_used is False


def test_select_params_p5_ineligible_falls_back_regardless_of_pf():
    combo = StrategyParams(exit_mode="time_stop", vol_filter="p75")
    m = _net_metrics([_trade_with_pnls(100, 90)] * 5)
    m["params"] = combo

    selection = select_params_p5([m], eligible=False)  # fold-wide floor not cleared
    assert selection.params == StrategyParams()
    assert selection.fallback_used is True


def test_select_params_p5_tie_break_total_pnl_then_drawdown():
    a, b = StrategyParams(exit_mode="fixed_1_5R"), StrategyParams(exit_mode="time_stop")
    ra = {"params": a, "net_profit_factor": 1.5, "net_total_pnl": 100.0, "net_max_drawdown": 50.0}
    rb = {"params": b, "net_profit_factor": 1.5, "net_total_pnl": 150.0, "net_max_drawdown": 50.0}
    assert select_params_p5([ra, rb], eligible=True).params == b  # higher net_total_pnl wins

    c, d = StrategyParams(exit_mode="breakeven_1R"), StrategyParams(exit_mode="partial_1R")
    rc = {"params": c, "net_profit_factor": 1.5, "net_total_pnl": 100.0, "net_max_drawdown": 80.0}
    rd = {"params": d, "net_profit_factor": 1.5, "net_total_pnl": 100.0, "net_max_drawdown": 30.0}
    assert select_params_p5([rc, rd], eligible=True).params == d  # lower drawdown wins the 2nd tie-break


# --- walk_forward_p5 integration: structure, no-leakage, eligibility table --


def test_walk_forward_p5_rejects_grid_without_default():
    df = _multi_day_session_df(n_days=6, bars_per_day=10, seed=3)
    grid = [StrategyParams(exit_mode="time_stop")]
    folds = [
        Fold(
            train_start=df.index[0].normalize(),
            train_end=df.index[-1].normalize(),
            test_start=df.index[-1].normalize(),
            test_end=df.index[-1].normalize() + pd.Timedelta(days=1),
        )
    ]
    with pytest.raises(ValueError):
        walk_forward_p5(df, grid, folds)


def test_walk_forward_p5_rejects_grid_with_varying_entry_field():
    df = _multi_day_session_df(n_days=6, bars_per_day=10, seed=3)
    grid = [StrategyParams(), StrategyParams(rr=2.0, exit_mode="time_stop")]
    folds = [
        Fold(
            train_start=df.index[0].normalize(),
            train_end=df.index[-1].normalize(),
            test_start=df.index[-1].normalize(),
            test_end=df.index[-1].normalize() + pd.Timedelta(days=1),
        )
    ]
    with pytest.raises(ValueError):
        walk_forward_p5(df, grid, folds)


def test_walk_forward_p5_structure_no_leakage_and_eligibility_table():
    df = _multi_day_session_df(n_days=40, bars_per_day=20, seed=7)

    grid = [
        StrategyParams(),  # required: fallback target + base baseline
        StrategyParams(exit_mode="breakeven_1R", vol_filter="p50"),
        StrategyParams(exit_mode="trail_swing", vol_filter="p75"),
    ]

    days = sorted(set(df.index.normalize()))
    b1 = days[len(days) // 3]
    b2 = days[2 * len(days) // 3]
    folds = [
        Fold(train_start=df.index[0].normalize(), train_end=b1, test_start=b1, test_end=b2),
        Fold(train_start=b1, train_end=b2, test_start=b2, test_end=days[-1] + pd.Timedelta(days=1)),
    ]

    result = walk_forward_p5(df, grid, folds, cost_model=CostModel())

    assert result["grid_size"] == len(grid)
    assert result["min_is_trades"] == MIN_IS_TRADES
    assert len(result["folds"]) == 2
    assert len(result["eligibility_table"]) == 2
    assert set(result["stitched_by_combo"].keys()) == set(grid)
    assert set(result["stitched_net_oos_pf_by_combo"].keys()) == set(grid)

    for fr, elig in zip(result["folds"], result["eligibility_table"]):
        # no-leakage structure: train ends exactly where test begins.
        assert fr["train_end"] == fr["test_start"]
        assert fr["test_start"] >= fr["train_end"]

        # null control: every grid combo's net OOS PF was recorded.
        assert len(fr["oos_pf_distribution"]) == len(grid)
        assert 0.0 <= fr["selected_oos_percentile"] <= 1.0
        assert isinstance(fr["fallback_used"], bool)
        assert fr["selected_params"] in grid

        # eligibility is fold-wide (Blocker 5), computed from the pre-filter count.
        assert fr["eligible"] == (fr["pre_filter_in_session_signal_count"] >= MIN_IS_TRADES)
        assert elig["eligible"] == fr["eligible"]
        assert elig["pre_filter_in_session_signal_count"] == fr["pre_filter_in_session_signal_count"]
        assert len(elig["combos"]) == len(grid)
        for combo_row in elig["combos"]:
            assert combo_row["is_realized_trades"] >= 0
            assert combo_row["oos_realized_trades"] >= 0

        # independent no-leakage re-check on the reported OOS trades.
        for t in fr["oos_trades"]:
            assert fr["test_start"] <= t.entry_time < fr["test_end"]


def test_walk_forward_p5_reports_net_not_gross_end_to_end():
    df = _multi_day_session_df(n_days=30, bars_per_day=20, seed=11)
    grid = [StrategyParams(), StrategyParams(exit_mode="time_stop", vol_filter="off")]

    days = sorted(set(df.index.normalize()))
    b1 = days[len(days) // 2]
    folds = [
        Fold(
            train_start=df.index[0].normalize(),
            train_end=b1,
            test_start=b1,
            test_end=days[-1] + pd.Timedelta(days=1),
        )
    ]

    result = walk_forward_p5(df, grid, folds, cost_model=CostModel())
    fr = result["folds"][0]

    oos_trades = fr["oos_trades"]
    recomputed_net_pf = profit_factor([t.net_pnl for t in oos_trades])
    assert fr["oos_net_metrics"]["net_profit_factor"] == pytest.approx(recomputed_net_pf)

    # CostModel always charges commission_rt (> 0) on every closed trade --
    # net_pnl must differ from pnl_usd for every trade whenever a CostModel
    # is supplied (a direct end-to-end check that net, not gross, was used).
    for t in oos_trades:
        assert t.net_pnl != pytest.approx(t.pnl_usd)
