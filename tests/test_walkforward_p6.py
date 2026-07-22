"""Phase 6: fold tiling (pinned rule), roll-spanning trade exclusion,
and the H-A / H-B runners (H-B provably selection-free)."""
import numpy as np
import pandas as pd
import pytest

from backtest.trade import Trade
from strategy.params import StrategyParams
from tuning.walkforward_p6 import (
    BASE_CONFIG,
    HB_CONFIGS,
    exclude_roll_spanning_trades,
    make_folds_tiled,
    run_HB,
)


def _index(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="1min", tz="US/Eastern")


# --- make_folds_tiled: the pinned tiling rule (Important I4) ----------------

def test_tiling_three_year_history():
    idx = _index("2020-02-01 09:30", "2023-01-15 16:00")
    folds = make_folds_tiled(idx)
    starts = [str(f.test_start.date()) for f in folds]
    # first boundary >= 2020-02-01 + 12mo -> 2021-07-01 (2021-01-01 < first_bar+12mo)
    assert starts[0] == "2021-07-01"
    assert starts == ["2021-07-01", "2022-01-01", "2022-07-01"]
    # 2023-01-01 boundary exists but its test span (to 2023-01-15) < 3mo -> stub, not formed
    for f in folds:
        assert f.train_end == f.test_start
        assert (f.test_start - f.train_start).days >= 360
        assert f.test_end > f.test_start


def test_tiling_stub_fold_rule_both_sides():
    # data edge 4 months past the last boundary -> final fold IS formed (clipped)
    idx = _index("2020-02-01 09:30", "2022-11-05 16:00")
    folds = make_folds_tiled(idx)
    last = folds[-1]
    assert str(last.test_start.date()) == "2022-07-01"
    assert str(last.test_end.date()) == "2022-11-05"
    # data edge 2 months past the boundary -> stub (< 3mo) -> NOT formed
    idx2 = _index("2020-02-01 09:30", "2022-09-01 16:00")
    folds2 = make_folds_tiled(idx2)
    assert str(folds2[-1].test_start.date()) == "2022-01-01"


def test_tiling_too_short_history_raises():
    with pytest.raises(ValueError):
        make_folds_tiled(_index("2022-01-01 09:30", "2023-03-01 16:00"))  # < 12mo+3mo usable


# --- roll-spanning trade exclusion (Blocker B2) ------------------------------

def _mk_trade(entry: str, exit_: str) -> Trade:
    t = Trade(entry_time=pd.Timestamp(entry, tz="US/Eastern"), direction="Long",
              entry=100.0, stop=99.0, target=101.5, risk=1.0)
    t.exit_time = pd.Timestamp(exit_, tz="US/Eastern")
    t.exit = 101.5
    t.outcome = "Win"
    t.pnl_usd = 30.0
    t.net_pnl = 20.0
    return t


def test_roll_spanning_trade_excluded_same_day_kept():
    # roll boundary detected for trading day 2023-03-10 (sits in the break before it)
    rolls = [pd.Timestamp("2023-03-10")]
    spanning = _mk_trade("2023-03-09 10:00", "2023-03-10 11:00")   # holds across the break
    same_day = _mk_trade("2023-03-09 10:00", "2023-03-09 11:30")   # intraday, before the roll
    after = _mk_trade("2023-03-10 10:00", "2023-03-10 11:00")      # entirely after the roll
    kept, n = exclude_roll_spanning_trades([spanning, same_day, after], rolls)
    assert n == 1
    assert spanning not in kept and same_day in kept and after in kept


def test_roll_exclusion_no_rolls_is_noop():
    t = _mk_trade("2023-03-09 10:00", "2023-03-12 11:00")
    kept, n = exclude_roll_spanning_trades([t], [])
    assert kept == [t] and n == 0


# --- H-B runner: provably selection-free -------------------------------------

def _synthetic_instrument_df(n_days=460, seed=7) -> pd.DataFrame:
    """~1.5y of per-day RTH sessions with enough movement to trade."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2021-01-04", periods=n_days)
    parts = [pd.date_range(f"{d.date()} 09:30", f"{d.date()} 11:30", freq="1min",
                           tz="US/Eastern", inclusive="left") for d in days]
    idx = parts[0].append(parts[1:])
    shocks = rng.normal(0, 0.4, len(idx))
    close = 1000 + np.cumsum(shocks)
    high = close + rng.uniform(0.2, 1.5, len(idx))
    low = close - rng.uniform(0.2, 1.5, len(idx))
    open_ = close + rng.normal(0, 0.3, len(idx))
    return pd.DataFrame({"open": open_, "high": np.maximum.reduce([open_, close, high]),
                         "low": np.minimum.reduce([open_, close, low]),
                         "close": close, "volume": 0}, index=idx)


def test_run_HB_never_calls_the_selector(monkeypatch):
    import tuning.walkforward_p5 as wf5
    import tuning.walkforward_p6 as wf6
    from strategy.instrument import SPECS

    def _boom(*a, **k):
        raise AssertionError("select_params_p5 must NEVER be called by run_HB")

    monkeypatch.setattr(wf5, "select_params_p5", _boom)
    df = _synthetic_instrument_df()
    result = run_HB(df, SPECS["NQ"], HB_CONFIGS["B1_partial_1R_p50"], roll_dates=[])
    assert result["n_folds"] >= 1
    assert "oos_trades_config" in result and "oos_trades_base" in result


def test_run_HB_base_config_equals_base_arm():
    from strategy.instrument import SPECS
    df = _synthetic_instrument_df()
    result = run_HB(df, SPECS["NQ"], BASE_CONFIG, roll_dates=[])
    cfg = [(t.entry_time, t.net_pnl) for t in result["oos_trades_config"]]
    base = [(t.entry_time, t.net_pnl) for t in result["oos_trades_base"]]
    assert cfg == base  # config==base -> identical arms


def test_hb_configs_are_the_preregistered_ones():
    assert HB_CONFIGS["B1_partial_1R_p50"] == StrategyParams(exit_mode="partial_1R", vol_filter="p50")
    assert HB_CONFIGS["B2_trail_swing_p50"] == StrategyParams(exit_mode="trail_swing", vol_filter="p50")
    assert BASE_CONFIG == StrategyParams()
