"""Phase-5 grid + net walk-forward: leak-free ATR% vol-filter percentiles,
NET-metric selection/null, and a selection floor decoupled from the filter
under test.

From docs/superpowers/plans/2026-07-13-phase5-exits-costs-volfilter.md, Task
2 Step 3. Reuses Phase-4's `tuning/walkforward.py` machinery (`make_folds`,
`Fold`, `Selection`, `_slice_layer`, `MIN_IS_TRADES`, `PF_CAP`) but does NOT
reuse `_precompute`/`_layer_for_params`/`_run_metrics`/`select_params`
unchanged -- each is either inapplicable (grid_p5 varies only
`exit_mode`/`vol_filter`, entry fixed at `StrategyParams()` defaults, so
there is exactly ONE signal layer, not one per grid value) or would
silently reintroduce a closed blocker from the adversarial review (gross
metrics, a per-combo selection floor). The five blockers this module closes:

  1. **NET metrics everywhere (Blocker 2).** `_net_metrics` below computes
     PF/total/drawdown/expectancy from `t.net_pnl`, never `t.pnl_usd`.
     Selection (`select_params_p5`), the 20-combo OOS null
     (`oos_pf_distribution`), the median/p75-combo stats, and the stitched
     aggregates ALL route through it -- Phase-4's gross `_run_metrics` is
     never called here.
  2. **ATR% lives inside the sliced layer (Blocker 4).** `_precompute_p5`
     appends `atr_pct = compute_atr(df)/df.close*100` to the SAME layer dict
     `compute_signal_layer` returns; `_slice_layer_p5` wraps Phase-4's
     `_slice_layer` (identical `[a:b]` bounds for sig/ema/ohlc/days/index)
     and slices `atr_pct` with the SAME `a, b` positions, asserting
     alignment. `run_execution` is always given this sliced array, never a
     full-series one.
  3. **Train-only vol-filter population (leak-free).** `_vol_threshold`
     computes the `vol_filter`-th percentile of ATR% over every in-session
     signal bar of a TRAIN-sliced layer ONLY -- before the vol gate, the
     1-trade/day cap, and independent of `exit_mode`. It is computed ONCE
     per fold (`vol_thresholds`, keyed by `VOL_FILTERS`) and reused for BOTH
     the IS run and the OOS run of every combo sharing that `vol_filter` --
     the OOS gate never recomputes from (or even sees) the test slice.
  4. **Selection floor on the PRE-FILTER signal count (Blocker 5).**
     `_in_session_signal_count` counts in-session double-confirmation signal
     bars BEFORE any gate -- a property of the entry base, identical across
     all 20 combos. `eligible` is a single fold-wide flag from this count,
     NOT each combo's own (post-filter) realized in-sample trade count --
     so a p50/p75 arm's thin realized-trade count doesn't auto-disqualify
     it. `select_params_p5` takes this flag; the per-fold/per-combo
     eligibility + realized-trade table is a required return value.
  5. **Reuses, not re-derives, the no-leakage fold guard** -- the same
     half-open disjointness check as Phase-4's `walk_forward`, re-verified
     per fold defensively.

Selection objective (net, in-sample): max net PF among eligible combos;
tie -> higher net total_pnl; tie -> lower net max_drawdown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.engine import compute_signal_layer, run_execution
from backtest.trade import Trade
from metrics import max_drawdown, profit_factor, total_pnl, win_rate
from strategy.atr import compute_atr
from strategy.params import StrategyParams
from tuning.grid_p5 import VOL_FILTERS
from tuning.walkforward import MIN_IS_TRADES, PF_CAP, Fold, Selection, _slice_layer

_PERCENTILES = {"p25": 25.0, "p50": 50.0, "p75": 75.0}  # "off" is handled separately -> None

# grid_p5 varies ONLY exit_mode/vol_filter -- every other StrategyParams field
# must equal StrategyParams()'s default on every grid combo (Global
# Constraints: "entry params fixed at StrategyParams() defaults"), since
# `_precompute_p5` builds exactly ONE signal layer shared by all 20 combos.
_ENTRY_FIELDS = ("fvg_threshold", "rr", "ema_length", "swing_lookback", "session_start", "session_end")


def _validate_grid_p5(grid: list[StrategyParams], default: StrategyParams) -> None:
    if default not in grid:
        raise ValueError(
            "grid_p5 must include StrategyParams() (fixed_1_5R/off) -- it is both the "
            "base baseline (success-rule condition (b)) and select_params_p5()'s fallback target"
        )
    for p in grid:
        for field in _ENTRY_FIELDS:
            if getattr(p, field) != getattr(default, field):
                raise ValueError(
                    "walk_forward_p5's precomputed signal layer is entry-fixed at "
                    f"StrategyParams() defaults -- {field!r} varies on {p!r}"
                )


def _precompute_p5(df: pd.DataFrame) -> dict:
    """The FULL-series, entry-fixed signal layer (`compute_signal_layer` at
    `StrategyParams()`) plus ATR% (Blocker 4), computed ONCE and shared by
    every one of the 20 combos -- unlike Phase-4's `_precompute`, there is
    no per-grid-value caching to do here since entry params never vary.
    """
    default = StrategyParams()
    layer = dict(compute_signal_layer(df, default))
    atr_pct = (compute_atr(df) / df["close"] * 100.0).to_numpy(dtype=float)
    if len(atr_pct) != len(layer["sig"]):
        raise ValueError("atr_pct must align bar-for-bar with the signal layer")
    layer["atr_pct"] = atr_pct
    return layer


def _slice_layer_p5(layer: dict, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Wraps Phase-4's `_slice_layer` (identical `[a, b)` half-open bounds
    for sig/ema/ohlc/days/index) and slices `atr_pct` with the SAME `a, b`
    positions (Blocker 4) -- `atr_pct` can never drift out of alignment with
    the rest of the layer, and the alignment is asserted, not assumed."""
    sliced = _slice_layer(layer, start, end)
    idx = layer["index"]
    a = idx.searchsorted(start, side="left")
    b = idx.searchsorted(end, side="left")
    sliced["atr_pct"] = layer["atr_pct"][a:b]
    if len(sliced["atr_pct"]) != len(sliced["sig"]):
        raise ValueError("sliced atr_pct must align bar-for-bar with the sliced signal layer")
    return sliced


def _in_session_signal_count(sliced_layer: dict) -> int:
    """The PRE-FILTER in-session signal count (Blocker 5): every bar where a
    double-confirmation signal fired (`sig[i] != ""`) inside the session
    (`sess[i]`) -- BEFORE the vol gate, the 1-trade/day cap, or `exit_mode`.
    A property of the entry base alone, identical across all 20 combos."""
    sess = sliced_layer["sess"]
    sig = sliced_layer["sig"]
    return int(np.count_nonzero(sess & (sig != "")))


def _vol_threshold(train_sliced_layer: dict, vol_filter: str) -> float | None:
    """The vol-filter threshold (leak-free, Global Constraints): the
    `vol_filter`-th percentile of ATR% over the PINNED POPULATION -- every
    in-session signal bar in the TRAIN slice, before the vol gate/1-per-day
    cap/exit_mode. `"off"` -> None (no gate). The caller MUST always pass a
    TRAIN-sliced layer -- this function has no way to enforce that itself,
    which is exactly why `walk_forward_p5` computes it only once per fold,
    from `train_layer`, and reuses it for both the IS and OOS runs."""
    if vol_filter == "off":
        return None
    sess = train_sliced_layer["sess"]
    sig = train_sliced_layer["sig"]
    atr_pct = train_sliced_layer["atr_pct"]
    population = atr_pct[sess & (sig != "")]
    if population.size == 0:
        return None  # no train-window signal bars to gate against -- degrades to "off"
    return float(np.percentile(population, _PERCENTILES[vol_filter]))


def _net_metrics(trades: list[Trade]) -> dict:
    """Net-aware metrics builder (Blocker 2): PF/win-rate/total/drawdown/
    expectancy computed from `t.net_pnl`, NEVER Phase-4's gross
    `_run_metrics`/`t.pnl_usd`. Every consumer in this module -- selection,
    the OOS null, the median/p75-combo stats, and the stitched aggregates --
    routes through this."""
    net_pnls = [t.net_pnl for t in trades]
    n = len(trades)
    return {
        "trades": trades,
        "n_trades": n,
        "net_profit_factor": profit_factor(net_pnls),
        "net_win_rate": win_rate(net_pnls),
        "net_total_pnl": total_pnl(net_pnls),
        "net_max_drawdown": max_drawdown(net_pnls),
        "net_expectancy": (total_pnl(net_pnls) / n) if n else 0.0,
    }


def select_params_p5(is_results: list[dict], eligible: bool) -> Selection:
    """Phase-5 selection (Blocker 5): `eligible` is a single fold-wide flag
    -- computed by the caller from the PRE-FILTER in-session signal count
    (`_in_session_signal_count`), identical across all 20 combos -- NOT each
    combo's own realized in-sample trade count (that IS the filter under
    test; a p75 arm would always look thin and auto-disqualify itself).
    When eligible, every entry in `is_results` is a candidate; objective:
    max `net_profit_factor`, tie -> higher `net_total_pnl`, tie -> lower
    `net_max_drawdown`. Falls back to `StrategyParams()` (`fallback_used`)
    when the fold-wide floor isn't cleared, mirroring Phase-4's
    `select_params`/`Selection` contract."""
    if not eligible or not is_results:
        return Selection(params=StrategyParams(), fallback_used=True)
    best = is_results[0]
    for r in is_results[1:]:
        if _is_better_p5(r, best):
            best = r
    return Selection(params=best["params"], fallback_used=False)


def _is_better_p5(a: dict, b: dict) -> bool:
    """True if in-sample net result `a` beats `b`: higher net_profit_factor;
    tie -> higher net_total_pnl; tie -> lower net_max_drawdown."""
    if a["net_profit_factor"] != b["net_profit_factor"]:
        return a["net_profit_factor"] > b["net_profit_factor"]
    if a["net_total_pnl"] != b["net_total_pnl"]:
        return a["net_total_pnl"] > b["net_total_pnl"]
    return a["net_max_drawdown"] < b["net_max_drawdown"]


def walk_forward_p5(
    df: pd.DataFrame,
    grid: list[StrategyParams],
    folds: list[Fold],
    cost_model=None,
    spec=None,
) -> dict:
    """Run the Phase-5 walk-forward: for each fold, compute the leak-free
    train-only vol-filter thresholds, select params from the TRAIN slice
    (net PF, eligibility-gated), evaluate the selection OOS (net), and --
    for the null control / condition-(d) distribution -- also evaluate
    every other grid combo OOS (net), stitched per-combo across folds.

    `grid` MUST include `StrategyParams()` and must not vary any entry
    field (`_validate_grid_p5`) -- the precomputed signal layer is built
    once, entry-fixed, and shared by every combo.
    """
    default = StrategyParams()
    _validate_grid_p5(grid, default)

    # Phase-6 instrument threading: `spec=None` preserves the exact Phase-5
    # call path (engine default = NQ spec), keeping the pinned F4 regression.
    from strategy.instrument import SPECS
    spec = spec if spec is not None else SPECS["NQ"]

    layer = _precompute_p5(df)
    idx = layer["index"]

    fold_reports = []
    stitched_tuned: list[Trade] = []
    stitched_default: list[Trade] = []
    stitched_by_combo: dict[StrategyParams, list[Trade]] = {p: [] for p in grid}
    eligibility_table = []

    for fold_i, fold in enumerate(folds):
        # --- No-leakage sanity check (Phase-4's guard, re-verified per fold
        # defensively; holds by construction from make_folds()). ---
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

        pre_filter_count = _in_session_signal_count(train_layer)
        eligible = pre_filter_count >= MIN_IS_TRADES

        # Vol thresholds: computed ONCE per fold, from the TRAIN slice ONLY
        # (leak-free), and reused for BOTH the IS run and the OOS run of
        # every combo sharing a given vol_filter -- the OOS gate never
        # recomputes from, or even sees, the test slice.
        vol_thresholds = {vf: _vol_threshold(train_layer, vf) for vf in VOL_FILTERS}

        # --- IS: every combo, net metrics, train slice only ---
        is_results = []
        for params in grid:
            trades = run_execution(
                train_layer,
                params,
                cost_model=cost_model,
                atr=train_layer["atr_pct"],
                vol_threshold=vol_thresholds[params.vol_filter],
                spec=spec,
            )
            m = _net_metrics(trades)
            m["params"] = params
            is_results.append(m)

        selection = select_params_p5(is_results, eligible)
        best_is = next(r for r in is_results if r["params"] == selection.params)

        # --- OOS: every combo (null control + condition-(d) distribution),
        # net metrics, test slice, SAME train-derived vol_thresholds. ---
        oos_results = []
        for params in grid:
            trades = run_execution(
                test_layer,
                params,
                cost_model=cost_model,
                atr=test_layer["atr_pct"],
                vol_threshold=vol_thresholds[params.vol_filter],
                spec=spec,
            )
            m = _net_metrics(trades)
            m["params"] = params
            oos_results.append(m)

        for r in oos_results:
            stitched_by_combo[r["params"]].extend(r["trades"])

        selected_oos = next(r for r in oos_results if r["params"] == selection.params)
        default_oos = next(r for r in oos_results if r["params"] == default)

        # Cap net PF (all-win combos give gross_loss==0 -> inf) so an inf
        # can't distort the median/p75-combo null (same convention as
        # Phase-4's oos_pf_distribution).
        oos_pf_distribution = [min(r["net_profit_factor"], PF_CAP) for r in oos_results]
        selected_pf = min(selected_oos["net_profit_factor"], PF_CAP)
        selected_oos_percentile = sum(1 for x in oos_pf_distribution if x <= selected_pf) / len(oos_pf_distribution)
        median_combo_oos_pf = float(np.median(oos_pf_distribution))
        p75_combo_oos_pf = float(np.percentile(oos_pf_distribution, 75))

        stitched_tuned.extend(selected_oos["trades"])
        stitched_default.extend(default_oos["trades"])

        eligibility_table.append(
            {
                "fold_index": fold_i,
                "pre_filter_in_session_signal_count": pre_filter_count,
                "eligible": eligible,
                "combos": [
                    {
                        "exit_mode": p.exit_mode,
                        "vol_filter": p.vol_filter,
                        "vol_threshold": vol_thresholds[p.vol_filter],
                        "is_realized_trades": next(r for r in is_results if r["params"] == p)["n_trades"],
                        "oos_realized_trades": next(r for r in oos_results if r["params"] == p)["n_trades"],
                    }
                    for p in grid
                ],
            }
        )

        fold_reports.append(
            {
                "fold_index": fold_i,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "selected_params": selection.params,
                "fallback_used": selection.fallback_used,
                "eligible": eligible,
                "pre_filter_in_session_signal_count": pre_filter_count,
                "is_net_pf": best_is["net_profit_factor"],
                "is_net_total_pnl": best_is["net_total_pnl"],
                "is_n": best_is["n_trades"],
                "is_net_max_drawdown": best_is["net_max_drawdown"],
                "oos_net_metrics": {
                    k: selected_oos[k]
                    for k in (
                        "net_profit_factor",
                        "net_win_rate",
                        "net_total_pnl",
                        "net_max_drawdown",
                        "net_expectancy",
                        "n_trades",
                    )
                },
                "oos_net_default_metrics": {
                    k: default_oos[k]
                    for k in (
                        "net_profit_factor",
                        "net_win_rate",
                        "net_total_pnl",
                        "net_max_drawdown",
                        "net_expectancy",
                        "n_trades",
                    )
                },
                "oos_trades": selected_oos["trades"],
                "oos_pf_distribution": oos_pf_distribution,
                "selected_oos_percentile": selected_oos_percentile,
                "median_combo_oos_pf": median_combo_oos_pf,
                "p75_combo_oos_pf": p75_combo_oos_pf,
            }
        )

    # Per-combo stitched-across-folds net OOS trades (condition-(d)'s p75
    # distribution needs the SAME stitched-PF statistic as the pick, not a
    # median-of-fold-medians) + capped PF summary per combo.
    stitched_net_oos_pf_by_combo = {
        params: min(profit_factor([t.net_pnl for t in trades]), PF_CAP)
        for params, trades in stitched_by_combo.items()
    }

    return {
        "folds": fold_reports,
        "oos_trades_tuned": stitched_tuned,
        "oos_trades_default": stitched_default,
        "stitched_tuned_net": _net_metrics(stitched_tuned),
        "stitched_default_net": _net_metrics(stitched_default),
        "stitched_by_combo": stitched_by_combo,
        "stitched_net_oos_pf_by_combo": stitched_net_oos_pf_by_combo,
        "grid_size": len(grid),
        "min_is_trades": MIN_IS_TRADES,
        "eligibility_table": eligibility_table,
    }
