"""Walk-forward parameter tuning: no-leakage in-sample selection + honest
out-of-sample evaluation, with a selection-luck null control.

From docs/superpowers/plans/2026-07-13-phase4-parameter-tuning.md, Task 2
Step 5. The Global Constraints govern the anti-leakage rules; the key ones
are restated inline at their enforcement points below, not just here.

PRECOMPUTE-AND-SLICE. `compute_cisd` is param-free; `compute_ifvg` depends
only on `fvg_threshold` (session is fixed at the `StrategyParams` default
for the base grid); `compute_ema` depends only on `ema_length`. None of
`compute_cisd`/`compute_ifvg`/`compute_ema`/`double_confirmation` look
ahead -- each bar's output depends only on bars <= it (verified in Task-1/2
review). So each is computed EXACTLY ONCE over the FULL `df`, cached by the
grid value that determines it, and then sliced per (fold, window) via
`index.searchsorted(ts, side="left")` -- a purely positional, half-open
`[start, end)` slice. Slicing a precomputed full-series array to a LATER
window can only ever hand a bar MORE past history than it would have seen
computed fresh on that window alone (since the causal recursion is warmed
up from further back) -- never any future bar. That is what makes
precompute-then-slice leak-free rather than merely convenient.

Selection (`select_params`) is computed from -- and only from -- the
TRAIN-window slice's results. The TEST window is used solely to (a)
evaluate the ONE selected combo (the OOS headline) and (b) run every other
grid combo too, purely for the null control / selection-luck diagnostics
-- the test window never feeds back into which combo got selected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.engine import run_execution
from backtest.trade import Trade
from metrics import max_drawdown, profit_factor, total_pnl, win_rate
from strategy.cisd import compute_cisd
from strategy.ema import compute_ema
from strategy.ifvg import compute_ifvg
from strategy.params import StrategyParams
from strategy.session import in_session_mask
from strategy.signals import double_confirmation
from tuning.grid import EMA_GRID, FVG_GRID

MIN_IS_TRADES = 50  # pre-registered selection floor (Global Constraints)

# Pre-registered fold windows (Global Constraints: "rolling 12mo train / 6mo
# test / 6mo step"), expressed directly in half-open form:
# (train_start, test_start, test_end) with train = [train_start, test_start)
# and test = [test_start, test_end). F4's test_end (2025-12-12) is past the
# data's last bar (2025-12-11 20:52 ET) -- `_slice_layer`'s searchsorted
# clips to the data edge automatically (searchsorted returns `len(index)`
# for an out-of-range boundary), so no special-casing is needed here.
_FOLD_SPECS = (
    ("2023-01-01", "2024-01-01", "2024-07-01"),
    ("2023-07-01", "2024-07-01", "2025-01-01"),
    ("2024-01-01", "2025-01-01", "2025-07-01"),
    ("2024-07-01", "2025-07-01", "2025-12-12"),
)
_FOLD_TZ = "America/New_York"


@dataclass(frozen=True)
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # == test_start (half-open train window end)
    test_start: pd.Timestamp
    test_end: pd.Timestamp   # half-open test window end (exclusive)


@dataclass(frozen=True)
class Selection:
    params: StrategyParams
    fallback_used: bool


def make_folds() -> list[Fold]:
    """The 4 pre-registered walk-forward folds, as half-open date windows:
    train = [train_start, test_start), test = [test_start, test_end)."""
    folds = []
    for train_start, test_start, test_end in _FOLD_SPECS:
        ts0 = pd.Timestamp(train_start, tz=_FOLD_TZ)
        ts1 = pd.Timestamp(test_start, tz=_FOLD_TZ)
        ts2 = pd.Timestamp(test_end, tz=_FOLD_TZ)
        folds.append(Fold(train_start=ts0, train_end=ts1, test_start=ts1, test_end=ts2))
    return folds


def select_params(is_results: list[dict]) -> Selection:
    """Pre-registered selection objective (Global Constraints): max
    in-sample profit factor among combos with >= MIN_IS_TRADES trades;
    ties broken by higher trade count, then lower max_drawdown. Falls back
    to `StrategyParams()` (flagged via `fallback_used`) if no combo clears
    the floor.

    `is_results` entries need only `params`, `profit_factor`, `n_trades`,
    `max_drawdown` -- this function reads ONLY these in-sample numbers. It
    never sees a test window; that is what makes selection leak-free.
    """
    eligible = [r for r in is_results if r["n_trades"] >= MIN_IS_TRADES]
    if not eligible:
        return Selection(params=StrategyParams(), fallback_used=True)

    best = eligible[0]
    for r in eligible[1:]:
        if _is_better(r, best):
            best = r
    return Selection(params=best["params"], fallback_used=False)


def _is_better(a: dict, b: dict) -> bool:
    """True if in-sample result `a` beats `b` under the pre-registered
    objective: higher profit_factor; tie -> higher n_trades; tie -> lower
    max_drawdown."""
    if a["profit_factor"] != b["profit_factor"]:
        return a["profit_factor"] > b["profit_factor"]
    if a["n_trades"] != b["n_trades"]:
        return a["n_trades"] > b["n_trades"]
    return a["max_drawdown"] < b["max_drawdown"]


def _window_trades(trades: list[Trade], start: pd.Timestamp, end: pd.Timestamp) -> list[Trade]:
    """Trades whose `entry_time` falls in the half-open `[start, end)`
    window. A defensive/auditing helper: with precompute-and-slice,
    `run_execution` run on a window-sliced layer can only ever produce
    entries inside that window by construction, but this gives an
    independent, slicing-free way to prove no test-window trade ever leaks
    into a train-window result (or vice versa) -- filtering an already
    correct trade list with this must be a no-op.
    """
    return [t for t in trades if start <= t.entry_time < end]


def _precompute(df: pd.DataFrame) -> dict:
    """Compute the param-dependent pieces of the signal layer ONCE over the
    FULL `df`, cached by the grid value that determines each:
      - `cisd`: param-free.
      - `sig` (double_confirmation of ifvg+cisd): depends only on
        `fvg_threshold` (session fixed at the default for the base grid).
      - `ema`: depends only on `ema_length`.
    `swing_lookback` and `rr` don't touch the signal layer at all -- they
    only affect `run_execution`'s stop lookback / target distance, which is
    cheap to redo per combo on top of this shared, precomputed layer.
    """
    default = StrategyParams()
    in_sess = in_session_mask(df.index, default.session_start, default.session_end)
    cisd = compute_cisd(df)

    sig_by_fvg = {
        fvg: double_confirmation(compute_ifvg(df, in_sess, fvg), cisd).to_numpy()
        for fvg in FVG_GRID
    }
    ema_by_len = {length: compute_ema(df, length).to_numpy(dtype=float) for length in EMA_GRID}

    o, h, l, c = (df[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    return {
        "sig_by_fvg": sig_by_fvg,
        "ema_by_len": ema_by_len,
        "sess": in_sess.to_numpy(dtype=bool),
        "o": o,
        "h": h,
        "l": l,
        "c": c,
        "days": df.index.tz_convert("America/New_York").date,
        "index": df.index,
    }


def _layer_for_params(pre: dict, params: StrategyParams) -> dict:
    """Assemble the full-series signal layer for `params` from the
    precomputed per-value caches (no recomputation)."""
    return {
        "sig": pre["sig_by_fvg"][params.fvg_threshold],
        "ema_v": pre["ema_by_len"][params.ema_length],
        "sess": pre["sess"],
        "o": pre["o"],
        "h": pre["h"],
        "l": pre["l"],
        "c": pre["c"],
        "days": pre["days"],
        "index": pre["index"],
    }


def _slice_layer(layer: dict, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Half-open positional slice `[start, end)` of a full-series layer via
    `index.searchsorted(ts, side="left")`. Every array in the layer is cut
    with the SAME `[a, b)` bounds, so they stay bar-for-bar aligned with
    each other and with `df` -- this is the "1-bar-leak guard" the plan
    calls out: a +1 error here would carry the next bar's signal/EMA into
    every decision in the window.
    """
    idx = layer["index"]
    a = idx.searchsorted(start, side="left")
    b = idx.searchsorted(end, side="left")
    return {
        "sig": layer["sig"][a:b],
        "ema_v": layer["ema_v"][a:b],
        "sess": layer["sess"][a:b],
        "o": layer["o"][a:b],
        "h": layer["h"][a:b],
        "l": layer["l"][a:b],
        "c": layer["c"][a:b],
        "days": layer["days"][a:b],
        "index": idx[a:b],
    }


def _run_metrics(layer: dict, params: StrategyParams) -> dict:
    trades = run_execution(layer, params)
    pnls = [t.pnl_usd for t in trades]
    return {
        "params": params,
        "trades": trades,
        "profit_factor": profit_factor(pnls),
        "win_rate": win_rate(pnls),
        "total_pnl": total_pnl(pnls),
        "max_drawdown": max_drawdown(pnls),
        "n_trades": len(trades),
    }


def _bootstrap_pf_ci(
    pnls: list[float],
    rng: np.random.Generator,
    n_boot: int = 1000,
    alpha: float = 0.05,
    cap: float = 1e6,
) -> tuple[float, float]:
    """Bootstrap CI on profit factor: resample `pnls` with replacement
    `n_boot` times using a caller-supplied, already-seeded
    `numpy.random.Generator` (deterministic and reproducible per fold --
    never `random`/time-based, per the plan). PF is capped at `cap` before
    taking percentiles so an all-win resample (gross_loss == 0 -> PF = inf
    under `metrics.profit_factor`'s convention) can't blow up the
    percentile computation.
    """
    arr = np.asarray(pnls, dtype=float)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0)
    resample_idx = rng.integers(0, n, size=(n_boot, n))
    samples = arr[resample_idx]
    gross_profit = np.clip(samples, 0, None).sum(axis=1)
    gross_loss = -np.clip(samples, None, 0).sum(axis=1)
    pf = np.full(n_boot, cap)
    has_loss = gross_loss > 0
    pf[has_loss] = np.minimum(gross_profit[has_loss] / gross_loss[has_loss], cap)
    pf[~has_loss & (gross_profit == 0)] = 0.0  # no wins, no losses -> PF 0 (matches metrics.profit_factor)
    lo, hi = np.percentile(pf, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def walk_forward(df: pd.DataFrame, grid: list[StrategyParams], folds: list[Fold]) -> dict:
    """Run the full walk-forward: for each fold, select params from the
    TRAIN slice only, evaluate the selection OOS, and -- for the null
    control -- also evaluate every other grid combo OOS plus the default
    baseline.

    `grid` MUST include `StrategyParams()` (the fallback target and the
    OOS-default baseline) and must not vary `session_start`/`session_end`
    (the precomputed signal-layer cache assumes the default session).
    """
    default = StrategyParams()
    assert default in grid, (
        "grid must include StrategyParams() -- it is both the select_params() "
        "fallback target and the OOS-default baseline"
    )
    assert all(
        p.session_start == default.session_start and p.session_end == default.session_end
        for p in grid
    ), "walk_forward's precomputed signal layer is cached at the default session only"

    pre = _precompute(df)
    idx = pre["index"]

    fold_reports = []
    stitched_tuned: list[Trade] = []
    stitched_default: list[Trade] = []

    for fold_i, fold in enumerate(folds):
        # --- No-leakage sanity check (holds by construction from
        # make_folds(); re-verified here defensively per the Global
        # Constraints: "half-open boundaries strictly disjoint"). ---
        a_tr = idx.searchsorted(fold.train_start, side="left")
        b_tr = idx.searchsorted(fold.train_end, side="left")
        a_te = idx.searchsorted(fold.test_start, side="left")
        b_te = idx.searchsorted(fold.test_end, side="left")
        assert b_tr == a_te, "train window must end exactly where the test window begins"
        if b_tr > a_tr and b_te > a_te:
            assert idx[a_tr:b_tr].max() < idx[a_te:b_te].min(), (
                "train/test slices must be strictly disjoint in time"
            )

        # --- IS: selection reads ONLY the train-window slice ---
        is_results = [
            _run_metrics(_slice_layer(_layer_for_params(pre, params), fold.train_start, fold.train_end), params)
            for params in grid
        ]

        selection = select_params(is_results)
        best_is = next(r for r in is_results if r["params"] == selection.params)

        rng = np.random.default_rng(fold_i)  # fixed seed per fold, not random/time-based
        ci_lo, ci_hi = _bootstrap_pf_ci([t.pnl_usd for t in best_is["trades"]], rng)
        n_combos_within_winner_ci = sum(
            1
            for r in is_results
            if r["params"] != selection.params and ci_lo <= r["profit_factor"] <= ci_hi
        )

        # --- OOS: the selected combo's headline AND the null control (every
        # grid combo, so the report can show the pick's percentile rank vs
        # the full distribution and the median-combo/random-combo nulls). ---
        oos_results = [
            _run_metrics(_slice_layer(_layer_for_params(pre, params), fold.test_start, fold.test_end), params)
            for params in grid
        ]

        oos_pf_distribution = [r["profit_factor"] for r in oos_results]
        selected_oos = next(r for r in oos_results if r["params"] == selection.params)
        default_oos = next(r for r in oos_results if r["params"] == default)
        random_oos = oos_results[(fold_i * 37) % len(grid)]  # deterministic sanity anchor, not random/time-based

        selected_pf = selected_oos["profit_factor"]
        selected_oos_percentile = sum(1 for x in oos_pf_distribution if x <= selected_pf) / len(oos_pf_distribution)
        median_combo_oos_pf = float(np.median(oos_pf_distribution))

        # Boundary artifact: a trade still open at test_end is simply never
        # appended by run_execution (it only returns CLOSED trades), so
        # dropping it is automatic here -- the next fold's execution starts
        # flat regardless.
        stitched_tuned.extend(selected_oos["trades"])
        stitched_default.extend(default_oos["trades"])

        fold_reports.append(
            {
                "fold_index": fold_i,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "selected_params": selection.params,
                "fallback_used": selection.fallback_used,
                "is_pf": best_is["profit_factor"],
                "is_n": best_is["n_trades"],
                "is_max_drawdown": best_is["max_drawdown"],
                "oos_metrics": {
                    k: selected_oos[k] for k in ("profit_factor", "win_rate", "total_pnl", "max_drawdown", "n_trades")
                },
                "oos_default_metrics": {
                    k: default_oos[k] for k in ("profit_factor", "win_rate", "total_pnl", "max_drawdown", "n_trades")
                },
                "oos_trades": selected_oos["trades"],
                "oos_pf_distribution": oos_pf_distribution,
                "selected_oos_percentile": selected_oos_percentile,
                "median_combo_oos_pf": median_combo_oos_pf,
                "random_combo_oos_pf": random_oos["profit_factor"],
                "n_combos_within_winner_ci": n_combos_within_winner_ci,
            }
        )

    return {
        "folds": fold_reports,
        "oos_trades_tuned": stitched_tuned,
        "oos_trades_default": stitched_default,
        "grid_size": len(grid),
        "min_is_trades": MIN_IS_TRADES,
    }
