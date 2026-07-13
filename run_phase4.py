"""Runner: single-shot Phase-4 walk-forward parameter-tuning verdict.

From docs/superpowers/plans/2026-07-13-phase4-parameter-tuning.md, Task 3.
`load_nq()` -> `walk_forward(df, build_grid(), make_folds())`, then assembles
`phase4_results.json` (the pre-registered, falsifiable OOS verdict) and the
5 `charts/phase4_*.png` diagnostics.

PRE-REGISTRATION: the grid (`tuning/grid.py`), folds/objective/floor
(`tuning/walkforward.py`) are frozen BEFORE this file is executed. This
runner records a SHA-256 hash of that frozen config plus the current git
commit SHA into the results JSON, so the stitched-OOS number is observed
exactly once (Global Constraints: "single-shot"). Do not edit the grid,
folds, `MIN_IS_TRADES`, or the selection objective after seeing a result --
that requires a new dated spec and an explicitly-labelled new experiment.

Run: .venv/Scripts/python run_phase4.py
Requires: data/raw/Dataset_NQ_1min_2022_2025.csv (Phase-1 raw data, not
committed). The full 144-combo x 4-fold sweep re-runs `run_execution` (a
pure-Python per-bar loop) ~1150 times over train/test window slices, plus a
one-time precompute of `compute_ifvg`/`compute_cisd` per grid value over the
full ~1.05M-row series -- expect several minutes.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest.trade import Trade
from metrics import max_drawdown, profit_factor, total_pnl, win_rate
from nqdata.load import load_nq
from strategy.params import StrategyParams
from tuning.grid import build_grid
from tuning.walkforward import MIN_IS_TRADES, make_folds, walk_forward

OBJECTIVE_NAME = "max_pf_min50trades"
CHARTS_DIR = Path("charts")
RESULTS_PATH = Path("phase4_results.json")
PARAM_FIELDS = ("fvg_threshold", "rr", "ema_length", "swing_lookback")


# --- pre-registration freeze: config hash + git SHA -------------------------


def _config_hash(grid: list[StrategyParams], folds, min_is_trades: int, objective_name: str) -> str:
    """SHA-256 of the frozen config: sorted list of all grid combos' field
    tuples + the fold date-windows + MIN_IS_TRADES + the objective name.
    Any change to these after seeing an OOS result must produce a DIFFERENT
    hash (a silent overwrite with the same hash would defeat the point).
    """
    combo_tuples = sorted(
        (p.fvg_threshold, p.rr, p.ema_length, p.swing_lookback, p.session_start, p.session_end) for p in grid
    )
    fold_windows = [
        [f.train_start.isoformat(), f.train_end.isoformat(), f.test_start.isoformat(), f.test_end.isoformat()]
        for f in folds
    ]
    payload = {
        "combos": combo_tuples,
        "fold_windows": fold_windows,
        "min_is_trades": min_is_trades,
        "objective": objective_name,
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


# --- serialization helpers ---------------------------------------------------


def _json_safe(obj):
    """Recursively replace inf/-inf/NaN floats with None so json.dumps
    produces strict, portable JSON (mirrors run_backtest.py's helper)."""
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _serialize_trade(t: Trade) -> dict:
    """Compact per-trade record -- enough to redraw equity curves, not the
    full Trade (entry/stop/target/exit prices omitted)."""
    return {
        "entry_time": t.entry_time.isoformat(),
        "direction": t.direction,
        "pnl_usd": t.pnl_usd,
        "r_multiple": t.r_multiple,
        "outcome": t.outcome,
    }


def _aggregate(trades: list[Trade]) -> dict:
    pnls = [t.pnl_usd for t in trades]
    return {
        "profit_factor": profit_factor(pnls),
        "win_rate": win_rate(pnls),
        "total_pnl": total_pnl(pnls),
        "max_drawdown": max_drawdown(pnls),
        "n": len(trades),
    }


def _trades_in_window(trades: list[Trade], start: pd.Timestamp, end: pd.Timestamp) -> list[Trade]:
    return [t for t in trades if start <= t.entry_time < end]


# --- results assembly ---------------------------------------------------


def _build_results(df: pd.DataFrame, grid: list[StrategyParams], folds, t_start: float) -> dict:
    result = walk_forward(df, grid, folds)
    fold_reports = result["folds"]

    config_hash = _config_hash(grid, folds, MIN_IS_TRADES, OBJECTIVE_NAME)
    git_sha = _git_sha()

    # --- stitched OOS: all 4 folds, and excluding fallback folds -----------
    stitched_tuned_all = result["oos_trades_tuned"]
    stitched_default_all = result["oos_trades_default"]

    non_fallback = [fr for fr in fold_reports if not fr["fallback_used"]]
    fallback_fold_count = len(fold_reports) - len(non_fallback)

    # Per-fold tuned trades are already stored per fold (`oos_trades`); default
    # trades are only stored stitched (all folds) in `result`, so recover the
    # per-fold subset by windowing on that fold's disjoint, non-overlapping
    # test span -- safe because no trade spans a fold boundary (Global
    # Constraints: boundary artifacts are dropped, not carried over).
    stitched_tuned_excl = [t for fr in non_fallback for t in fr["oos_trades"]]
    stitched_default_excl = [
        t for fr in non_fallback for t in _trades_in_window(stitched_default_all, fr["test_start"], fr["test_end"])
    ]

    stitched_oos = {
        "all_folds": {
            "tuned": _aggregate(stitched_tuned_all),
            "default": _aggregate(stitched_default_all),
        },
        "excluding_fallback_folds": {
            "tuned": _aggregate(stitched_tuned_excl),
            "default": _aggregate(stitched_default_excl),
        },
        "fallback_fold_count": fallback_fold_count,
    }

    # --- success rule: 4 pre-registered conditions, headline = all-folds stitched OOS ---
    tuned_pf = stitched_oos["all_folds"]["tuned"]["profit_factor"]
    default_pf = stitched_oos["all_folds"]["default"]["profit_factor"]

    cond_a = tuned_pf > 1.0
    margin = tuned_pf - default_pf
    cond_b = margin >= 0.10
    folds_tuned_beats_default = sum(
        1 for fr in fold_reports if fr["oos_metrics"]["profit_factor"] > fr["oos_default_metrics"]["profit_factor"]
    )
    cond_c = folds_tuned_beats_default >= 3
    median_of_fold_median_combo_pf = float(np.median([fr["median_combo_oos_pf"] for fr in fold_reports]))
    cond_d = tuned_pf > median_of_fold_median_combo_pf

    success_rule = {
        "condition_a_tuned_pf_gt_1": {"tuned_stitched_oos_pf": tuned_pf, "passes": bool(cond_a)},
        "condition_b_margin_gte_0_10": {
            "tuned_stitched_oos_pf": tuned_pf,
            "default_stitched_oos_pf": default_pf,
            "margin": margin,
            "passes": bool(cond_b),
        },
        "condition_c_wins_3_of_4_folds": {
            "folds_tuned_beats_default": folds_tuned_beats_default,
            "total_folds": len(fold_reports),
            "passes": bool(cond_c),
        },
        "condition_d_beats_median_combo_null": {
            "tuned_stitched_oos_pf": tuned_pf,
            "median_of_fold_median_combo_oos_pf": median_of_fold_median_combo_pf,
            "passes": bool(cond_d),
        },
        "robust_improvement": bool(cond_a and cond_b and cond_c and cond_d),
    }

    # --- overfitting / stability diagnostics --------------------------------
    in_sample_to_oos_decay = [
        {
            "fold_index": fr["fold_index"],
            "is_pf": fr["is_pf"],
            "oos_pf": fr["oos_metrics"]["profit_factor"],
            "decay": fr["is_pf"] - fr["oos_metrics"]["profit_factor"],
        }
        for fr in fold_reports
    ]

    parameter_stability = {
        field: [getattr(fr["selected_params"], field) for fr in fold_reports] for field in PARAM_FIELDS
    }
    parameter_stability["caveat"] = (
        "n=4 folds with overlapping (6mo-shifted) train windows; the 4 selected "
        "param-sets are NOT independent draws, which inflates apparent stability. "
        "Descriptive only -- no statistical power."
    )

    # --- per-fold report -----------------------------------------------------
    folds_out = []
    for fr in fold_reports:
        folds_out.append(
            {
                "fold_index": fr["fold_index"],
                "train_start": fr["train_start"].isoformat(),
                "train_end": fr["train_end"].isoformat(),
                "test_start": fr["test_start"].isoformat(),
                "test_end": fr["test_end"].isoformat(),
                "selected_params": asdict(fr["selected_params"]),
                "fallback_used": fr["fallback_used"],
                "is_pf": fr["is_pf"],
                "is_n": fr["is_n"],
                "is_max_drawdown": fr["is_max_drawdown"],
                "oos_metrics": fr["oos_metrics"],
                "oos_default_metrics": fr["oos_default_metrics"],
                "oos_trades": [_serialize_trade(t) for t in fr["oos_trades"]],
                "oos_pf_distribution": fr["oos_pf_distribution"],
                "selected_oos_percentile": fr["selected_oos_percentile"],
                "median_combo_oos_pf": fr["median_combo_oos_pf"],
                "random_combo_oos_pf": fr["random_combo_oos_pf"],
                "n_combos_within_winner_ci": fr["n_combos_within_winner_ci"],
            }
        )

    return {
        "config_hash": config_hash,
        "git_sha": git_sha,
        "grid_size": result["grid_size"],
        "min_is_trades": result["min_is_trades"],
        "objective": OBJECTIVE_NAME,
        "run_seconds": round(time.time() - t_start, 1),
        "folds": folds_out,
        "stitched_oos": stitched_oos,
        "success_rule": success_rule,
        "in_sample_to_oos_decay": in_sample_to_oos_decay,
        "parameter_stability": parameter_stability,
    }, {
        # kept out of the JSON payload but handed to the chart functions
        "stitched_tuned_all": stitched_tuned_all,
        "stitched_default_all": stitched_default_all,
    }


# --- charts ---------------------------------------------------------------


def _plot_equity_curve(stitched_tuned: list[Trade], stitched_default: list[Trade]) -> None:
    def curve(trades):
        running = 0.0
        out = []
        for t in trades:
            running += t.pnl_usd
            out.append(running)
        return out

    tuned_curve = curve(stitched_tuned)
    default_curve = curve(stitched_default)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(tuned_curve) + 1), tuned_curve, color="#2563eb", linewidth=1.5, label="tuned")
    ax.plot(range(1, len(default_curve) + 1), default_curve, color="#f59e0b", linewidth=1.5, label="default")
    ax.axhline(0.0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax.set_title("Stitched OOS cumulative P&L -- tuned vs default")
    ax.set_xlabel("OOS trade # (chronological, stitched across folds)")
    ax.set_ylabel("Cumulative P&L (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "phase4_equity_curve.png", dpi=150)
    plt.close(fig)


def _plot_oos_pf_per_fold(folds_out: list[dict]) -> None:
    labels = [f"F{fr['fold_index'] + 1}" for fr in folds_out]
    tuned_vals = [min(fr["oos_metrics"]["profit_factor"], 5.0) for fr in folds_out]
    default_vals = [min(fr["oos_default_metrics"]["profit_factor"], 5.0) for fr in folds_out]

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([xi - width / 2 for xi in x], tuned_vals, width=width, label="tuned", color="#2563eb")
    ax.bar([xi + width / 2 for xi in x], default_vals, width=width, label="default", color="#f59e0b")
    ax.axhline(1.0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("OOS profit factor (capped at 5.0 for display)")
    ax.set_title("Per-fold OOS profit factor -- tuned vs default")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "phase4_oos_pf_per_fold.png", dpi=150)
    plt.close(fig)


def _plot_param_stability(parameter_stability: dict, folds_out: list[dict]) -> None:
    fields = list(PARAM_FIELDS)
    labels = [f"F{fr['fold_index'] + 1}" for fr in folds_out]

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.axis("off")
    table_data = [[str(v) for v in parameter_stability[field]] for field in fields]
    table = ax.table(
        cellText=table_data,
        rowLabels=fields,
        colLabels=labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    ax.set_title("Selected parameter per fold (stability table)\n"
                  "caveat: overlapping train windows inflate apparent agreement (n=4, descriptive only)",
                  fontsize=9)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "phase4_param_stability.png", dpi=150)
    plt.close(fig)


def _plot_is_vs_oos_pf(in_sample_to_oos_decay: list[dict]) -> None:
    labels = [f"F{d['fold_index'] + 1}" for d in in_sample_to_oos_decay]
    is_vals = [min(d["is_pf"], 5.0) for d in in_sample_to_oos_decay]
    oos_vals = [min(d["oos_pf"], 5.0) if d["oos_pf"] == d["oos_pf"] else 0.0 for d in in_sample_to_oos_decay]

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([xi - width / 2 for xi in x], is_vals, width=width, label="in-sample (selection) PF", color="#10b981")
    ax.bar([xi + width / 2 for xi in x], oos_vals, width=width, label="OOS (selected combo) PF", color="#2563eb")
    ax.axhline(1.0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Profit factor (capped at 5.0 for display)")
    ax.set_title("In-sample vs OOS profit factor per fold (the overfitting gap)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "phase4_is_vs_oos_pf.png", dpi=150)
    plt.close(fig)


def _plot_selection_luck_null(folds_out: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, fr in zip(axes.flat, folds_out):
        dist = [min(v, 5.0) for v in fr["oos_pf_distribution"]]
        ax.hist(dist, bins=24, color="#93c5fd", edgecolor="#1e40af", alpha=0.85)
        selected_pf = min(fr["oos_metrics"]["profit_factor"], 5.0)
        median_pf = min(fr["median_combo_oos_pf"], 5.0)
        ax.axvline(selected_pf, color="#dc2626", linewidth=2, label="selected pick")
        ax.axvline(median_pf, color="#111827", linewidth=1.5, linestyle="--", label="median combo")
        ax.set_title(
            f"F{fr['fold_index'] + 1}: selected pick @ p{fr['selected_oos_percentile'] * 100:.0f}",
            fontsize=10,
        )
        ax.set_xlabel("OOS profit factor (144 combos, capped at 5.0)")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)

    fig.suptitle("Selection-luck null: all 144 combos' OOS PF per fold, selected pick vs median-combo null")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "phase4_selection_luck_null.png", dpi=150)
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    CHARTS_DIR.mkdir(exist_ok=True)

    print("Loading NQ data ...")
    df = load_nq()
    print(f"  {len(df):,} rows, {df.index[0]} .. {df.index[-1]} ({time.time() - t0:.1f}s)")

    grid = build_grid()
    folds = make_folds()
    print(f"Grid size: {len(grid)}, folds: {len(folds)}")

    print("Running walk-forward (144 combos x 4 folds, IS-select + OOS-eval + null control) ...")
    t1 = time.time()
    results, extra = _build_results(df, grid, folds, t0)
    print(f"  walk_forward + assembly done in {time.time() - t1:.1f}s")

    RESULTS_PATH.write_text(json.dumps(_json_safe(results), indent=2, default=str))
    print(f"Wrote {RESULTS_PATH}")

    print("Plotting charts ...")
    _plot_equity_curve(extra["stitched_tuned_all"], extra["stitched_default_all"])
    _plot_oos_pf_per_fold(results["folds"])
    _plot_param_stability(results["parameter_stability"], results["folds"])
    _plot_is_vs_oos_pf(results["in_sample_to_oos_decay"])
    _plot_selection_luck_null(results["folds"])
    print(f"Wrote charts to {CHARTS_DIR}/ (phase4_*.png)")

    sr = results["success_rule"]
    print("\n=== HEADLINE (pre-registered, single-shot) ===")
    print(f"config_hash={results['config_hash'][:16]}...  git_sha={results['git_sha'][:12]}")
    print(
        f"Tuned stitched-OOS PF = {sr['condition_a_tuned_pf_gt_1']['tuned_stitched_oos_pf']:.4f} vs "
        f"default = {sr['condition_b_margin_gte_0_10']['default_stitched_oos_pf']:.4f} "
        f"(margin {sr['condition_b_margin_gte_0_10']['margin']:+.4f})"
    )
    print(
        f"(a) PF>1.0: {sr['condition_a_tuned_pf_gt_1']['passes']}  "
        f"(b) margin>=0.10: {sr['condition_b_margin_gte_0_10']['passes']}  "
        f"(c) wins {sr['condition_c_wins_3_of_4_folds']['folds_tuned_beats_default']}/4 folds: "
        f"{sr['condition_c_wins_3_of_4_folds']['passes']}  "
        f"(d) beats median-combo null: {sr['condition_d_beats_median_combo_null']['passes']}"
    )
    print(f"ROBUST IMPROVEMENT: {sr['robust_improvement']}")
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
