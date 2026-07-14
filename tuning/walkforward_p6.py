"""Phase-6 cross-instrument confirmation runners.

CONFIRMATION, NOT EXPLORATION (Global Constraints): nothing here re-tunes,
re-searches, or re-selects anything in response to Phase-6 data. `run_HA`
runs the FROZEN Phase-5 walk-forward procedure (`tuning.walkforward_p5.
walk_forward_p5`, reused verbatim -- not reimplemented) over folds tiled
across each instrument's full history; `run_HB` runs ONE fixed exit config
per fold with NO selector call anywhere (its test monkeypatches the selector
to raise, proving it). The only Phase-6-specific mechanics are:

  - `make_folds_tiled` (Important I4, pinned): test windows start at every
    Jan-01/Jul-01 ET boundary `t` with `t >= first_bar + 12mo` and
    `t < last_bar`; train = `[t - 12mo, t)` exactly (never extended to
    first_bar); test = `[t, min(t + 6mo, data_end))`; the final fold is
    formed only if its test span is >= 3 months. < 1 usable fold -> raises.
  - `exclude_roll_spanning_trades` (Blocker B2): a trade whose holding
    window spans a detected contract-roll boundary (from
    `validation_report_p6.json`) books phantom splice P&L on unadjusted
    data; such trades are EXCLUDED identically in EVERY arm (tuned, base,
    every combo of the null) and counted -- never force-flattened, which
    would add a new exit path to frozen machinery.
"""

from __future__ import annotations

import pandas as pd

from backtest.costs import CostModel
from backtest.trade import Trade
from strategy.instrument import InstrumentSpec
from strategy.params import StrategyParams
from tuning.walkforward import Fold
from tuning.walkforward_p5 import (
    VOL_FILTERS,
    _in_session_signal_count,
    _net_metrics,
    _precompute_p5,
    _slice_layer_p5,
    _vol_threshold,
    walk_forward_p5,
)

FOLD_TZ = "America/New_York"
TRAIN_MONTHS = 12
TEST_MONTHS = 6
MIN_LAST_TEST_MONTHS = 3

# The two pre-registered fixed configs (H-B; spec "Pre-registered hypotheses").
HB_CONFIGS = {
    "B1_partial_1R_p50": StrategyParams(exit_mode="partial_1R", vol_filter="p50"),
    "B2_trail_swing_p50": StrategyParams(exit_mode="trail_swing", vol_filter="p50"),
}
BASE_CONFIG = StrategyParams()  # fixed_1_5R / off


def make_folds_tiled(index: pd.DatetimeIndex) -> list[Fold]:
    """Tile the frozen 12mo-train/6mo-test rolling folds over an
    instrument's actual history (pinned rule -- see module docstring)."""
    if index.tz is None:
        raise ValueError("make_folds_tiled requires a tz-aware index")
    first_bar = index.min().tz_convert(FOLD_TZ)
    last_bar = index.max().tz_convert(FOLD_TZ)

    # candidate test-start boundaries: every Jan-01 / Jul-01 in range
    years = range(first_bar.year, last_bar.year + 1)
    boundaries = []
    for y in years:
        for m in (1, 7):
            boundaries.append(pd.Timestamp(year=y, month=m, day=1, tz=FOLD_TZ))
    boundaries = sorted(b for b in boundaries
                        if b >= first_bar + pd.DateOffset(months=TRAIN_MONTHS) and b < last_bar)

    folds: list[Fold] = []
    for t in boundaries:
        test_end = min(t + pd.DateOffset(months=TEST_MONTHS), last_bar)
        if test_end <= t + pd.DateOffset(months=MIN_LAST_TEST_MONTHS) and test_end < t + pd.DateOffset(months=TEST_MONTHS):
            continue  # stub final fold: test span < 3 months -> not formed
        folds.append(Fold(
            train_start=t - pd.DateOffset(months=TRAIN_MONTHS),
            train_end=t,
            test_start=t,
            test_end=test_end,
        ))
    if not folds:
        raise ValueError(
            f"history too short to form a single 12mo-train/>=3mo-test fold: "
            f"{first_bar} .. {last_bar}"
        )
    return folds


def _trade_date(ts: pd.Timestamp) -> pd.Timestamp:
    """Same futures trading-day convention as validate_p6.trade_date."""
    return (ts + pd.Timedelta(hours=6)).normalize()


def exclude_roll_spanning_trades(
    trades: list[Trade], roll_dates: list[pd.Timestamp]
) -> tuple[list[Trade], int]:
    """Drop trades whose holding window spans a roll boundary (Blocker B2).

    A roll boundary detected for trading-day D sits in the session break
    IMMEDIATELY BEFORE D's first bar. A trade spans it iff it is open across
    that break: entry trading-day < D <= exit trading-day. Applied
    identically to every arm; returns (kept, n_excluded)."""
    if not roll_dates:
        return trades, 0
    rolls = sorted(pd.Timestamp(d).tz_localize(FOLD_TZ) if pd.Timestamp(d).tz is None
                   else pd.Timestamp(d).tz_convert(FOLD_TZ) for d in roll_dates)
    kept: list[Trade] = []
    n_excluded = 0
    for t in trades:
        if t.exit_time is None:
            kept.append(t)
            continue
        d_in = _trade_date(t.entry_time.tz_convert(FOLD_TZ))
        d_out = _trade_date(t.exit_time.tz_convert(FOLD_TZ))
        spans = any(d_in < r <= d_out for r in rolls)
        if spans:
            n_excluded += 1
        else:
            kept.append(t)
    return kept, n_excluded


def _apply_exclusions(report: dict, roll_dates: list) -> dict:
    """Apply the roll exclusion to every trade list in a walk_forward_p5-shaped
    report, identically across arms, and attach the counts."""
    n_total = 0
    for key in ("oos_trades_tuned", "oos_trades_default"):
        kept, n = exclude_roll_spanning_trades(report[key], roll_dates)
        report[key] = kept
        n_total += n
    by_combo = report.get("stitched_by_combo")
    if by_combo is not None:
        for k in list(by_combo):
            kept, n = exclude_roll_spanning_trades(by_combo[k], roll_dates)
            by_combo[k] = kept
    # stitched summary metrics must reflect the SAME post-exclusion trade
    # lists (identical treatment across arms)
    report["stitched_tuned_net"] = _net_metrics(report["oos_trades_tuned"])
    report["stitched_default_net"] = _net_metrics(report["oos_trades_default"])
    report["n_roll_spanning_trades_excluded_tuned_plus_base"] = n_total
    return report


def run_HA(df: pd.DataFrame, spec: InstrumentSpec, grid: list[StrategyParams],
           roll_dates: list, cost_model: CostModel | None = None) -> dict:
    """H-A: the FROZEN Phase-5 procedure, unchanged, over tiled folds."""
    cm = cost_model if cost_model is not None else spec.cost_model()
    folds = make_folds_tiled(df.index)
    report = walk_forward_p5(df, grid, folds, cm, spec=spec)
    report["n_folds"] = len(folds)
    return _apply_exclusions(report, roll_dates)


def run_HB(df: pd.DataFrame, spec: InstrumentSpec, config: StrategyParams,
           roll_dates: list, cost_model: CostModel | None = None) -> dict:
    """H-B: ONE fixed config per fold. No selection anywhere -- the per-fold
    vol threshold is the same leak-free train-window percentile computation
    Phase 5 froze, but the CONFIG never varies and no selector is invoked."""
    from backtest.engine import run_execution

    cm = cost_model if cost_model is not None else spec.cost_model()
    folds = make_folds_tiled(df.index)
    layer = _precompute_p5(df)
    idx = layer["index"]

    fold_rows = []
    stitched_cfg: list[Trade] = []
    stitched_base: list[Trade] = []
    for fold in folds:
        a_tr = idx.searchsorted(fold.train_start, side="left")
        b_tr = idx.searchsorted(fold.train_end, side="left")
        a_te = idx.searchsorted(fold.test_start, side="left")
        b_te = idx.searchsorted(fold.test_end, side="left")
        if b_tr != a_te:
            raise ValueError("train window must end exactly where the test window begins")
        if b_tr > a_tr and b_te > a_te and not (idx[a_tr:b_tr].max() < idx[a_te:b_te].min()):
            raise ValueError("train/test slices must be strictly disjoint in time")

        train_layer = _slice_layer_p5(layer, fold.train_start, fold.train_end)
        test_layer = _slice_layer_p5(layer, fold.test_start, fold.test_end)
        thr = _vol_threshold(train_layer, config.vol_filter)
        thr_base = _vol_threshold(train_layer, BASE_CONFIG.vol_filter)  # None ("off")

        cfg_trades = run_execution(test_layer, config, cost_model=cm,
                                   atr=test_layer["atr_pct"], vol_threshold=thr, spec=spec)
        base_trades = run_execution(test_layer, BASE_CONFIG, cost_model=cm,
                                    atr=test_layer["atr_pct"], vol_threshold=thr_base, spec=spec)
        stitched_cfg.extend(cfg_trades)
        stitched_base.extend(base_trades)
        fold_rows.append({
            "train_start": fold.train_start, "test_start": fold.test_start,
            "test_end": fold.test_end,
            "pre_filter_in_session_signal_count": _in_session_signal_count(train_layer),
            "cfg_net": {k: v for k, v in _net_metrics(cfg_trades).items() if k != "trades"},
            "base_net": {k: v for k, v in _net_metrics(base_trades).items() if k != "trades"},
        })

    stitched_cfg, n_ex_cfg = exclude_roll_spanning_trades(stitched_cfg, roll_dates)
    stitched_base, n_ex_base = exclude_roll_spanning_trades(stitched_base, roll_dates)
    return {
        "config": config,
        "n_folds": len(folds),
        "folds": fold_rows,
        "oos_trades_config": stitched_cfg,
        "oos_trades_base": stitched_base,
        "n_roll_spanning_trades_excluded": n_ex_cfg + n_ex_base,
    }
