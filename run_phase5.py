"""Runner: single-shot Phase-5 net out-of-sample verdict (costs + exits +
volatility filter).

From the Phase-5 exits/costs/vol-filter spec
(docs/specs/2026-07-13-phase5-exits-costs-volfilter-design.md).
`load_nq()` -> `walk_forward_p5(df, build_grid_p5(), make_folds(),
CostModel())`, then assembles `phase5_results.json` (the pre-registered,
falsifiable NET OOS verdict) and the 5 `charts/phase5_*.png` diagnostics.

PRE-REGISTRATION: the grid (`tuning/grid_p5.py`), folds/objective/floor
(`tuning/walkforward.py`, `tuning/walkforward_p5.py`), the cost constants
(`backtest/costs.py`), the exit-mode sequencing (`backtest/exits.py`) and the
vol-filter definition are frozen BEFORE this file is executed. This runner
records a SHA-256 hash of that ENTIRE frozen design plus the current git
commit SHA into the results JSON, so the stitched-OOS number is observed
exactly once (the spec's "single-shot" rule). Do not edit the grid,
folds, MIN_IS_TRADES/MIN_OOS_TRADES, the objective, the vol-filter
definition, the ATR method, the exit sequencing, the cost constants, or
TIME_STOP_ET after seeing a result -- that requires a new dated spec and an
explicitly-labelled new experiment.

Cost sensitivity (0x/1x/2x) is derived ALGEBRAICALLY from the single sweep's
already-computed per-trade gross `pnl_usd` and 1x `net_pnl`, not by
re-running the 20x4 sweep three times: `CostModel.leg_cost` is linear in
`multiplier` for every fill leg (`cost * multiplier`), so for any trade,
`cost_at_m = cost_at_1x * m`, hence `net_at_m = gross - (gross - net_at_1x) *
m`. This holds exactly even for `partial_1R`'s two-leg trades (the identity
holds leg-by-leg, and net_pnl is the SUM of leg net_pnls, so it holds for
the summed total too). `CostModel.multiplier` is documented as "reported,
never used for selection" -- consistent with recomputing the ALREADY
selected picks' cost exposure rather than re-selecting under each multiplier.

Run: .venv/Scripts/python run_phase5.py
Requires: data/raw/Dataset_NQ_1min_2022_2025.csv (Phase-1 raw data, not
committed). The 20-combo x 4-fold x 2 (IS+OOS) sweep re-runs `run_execution`
(a pure-Python per-bar loop) ~160 times over train/test window slices, plus a
one-time precompute of the shared entry-fixed signal layer (+ ATR%) over the
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

from backtest.costs import CostModel
from backtest.exits import TIME_STOP_ET
from backtest.trade import Trade
from metrics import max_drawdown, profit_factor, total_pnl, win_rate
from nqdata.load import load_nq
from strategy.params import StrategyParams
from tuning.grid_p5 import VOL_FILTERS, build_grid_p5
from tuning.walkforward import MIN_IS_TRADES, PF_CAP, _bootstrap_pf_ci, make_folds
from tuning.walkforward_p5 import walk_forward_p5

MIN_OOS_TRADES = 60  # pre-registered OOS-sample gate (success rule, condition (e))
OBJECTIVE_NAME = "max_net_pf_among_eligible_combos__tiebreak_net_total_pnl_then_lower_net_max_drawdown"
BOOTSTRAP_SEED = 42  # fixed, not random/time-based (condition (e)'s bootstrap)
BOOTSTRAP_N = 1000
CHARTS_DIR = Path("charts")
RESULTS_PATH = Path("phase5_results.json")

_EXIT_MODE_ABBR = {
    "fixed_1_5R": "fixed_1.5R",
    "breakeven_1R": "breakeven_1R",
    "trail_swing": "trail_swing",
    "partial_1R": "partial_1R",
    "time_stop": "time_stop",
}


# --- pre-registration freeze: config hash + git SHA -------------------------


def _config_hash(
    grid: list[StrategyParams],
    folds,
    min_is_trades: int,
    min_oos_trades: int,
    objective_name: str,
    cost_model: CostModel,
) -> str:
    """SHA-256 of the WHOLE frozen Phase-5 design (the spec requires the
    config hash to cover the whole frozen design): every grid combo's full
    field tuple (entry fields + exit_mode + vol_filter -- not just the
    varying two, so a change to the entry-fixed defaults also changes the
    hash), the fold date-windows, MIN_IS_TRADES, MIN_OOS_TRADES, the
    objective, the vol-filter definition (ATR% + pinned population +
    percentile set), the ATR method (period + Wilder RMA), the exit-mode
    intrabar sequencing tag, the cost constants, and TIME_STOP_ET. `git_sha`
    is deliberately NOT part of this hash (it is recorded as a separate
    sibling field, same convention as Phase-4's `_config_hash`) -- the design
    is what must be frozen pre-registration; the commit it happened to run
    at is provenance, not part of the design.
    """
    combo_tuples = sorted(
        (
            p.fvg_threshold,
            p.rr,
            p.ema_length,
            p.swing_lookback,
            p.session_start,
            p.session_end,
            p.exit_mode,
            p.vol_filter,
        )
        for p in grid
    )
    fold_windows = [
        [f.train_start.isoformat(), f.train_end.isoformat(), f.test_start.isoformat(), f.test_end.isoformat()]
        for f in folds
    ]
    vol_filter_definition = {
        "variable": "ATR% = ATR14 / close * 100, evaluated at the signal bar",
        "population": (
            "every in-session double-confirmation signal bar in the TRAIN window, "
            "computed before the vol gate and independent of the 1-trade/day cap and exit_mode"
        ),
        "percentiles": {"off": None, "p25": 25.0, "p50": 50.0, "p75": 75.0},
        "vol_filters": list(VOL_FILTERS),
    }
    atr_method = {"period": 14, "method": "wilder_rma", "alpha": "1/period", "ewm_adjust": False}
    exit_sequencing_tag = (
        "stop_first_gap_through__no_phantom_credit_v1: on each managed bar, the CURRENT "
        "pre-activation stop is evaluated FIRST using stop-first/gap-through-at-worse-of-"
        "stop-or-open; if breached, the bar resolves as a stop-type exit and +1R activation/"
        "breakeven-move/trail-ratchet/partial-scaleout may NOT also happen on that same bar; "
        "R and all managed levels (1R, 1.5R, 3R) anchored at the SIGNAL-BAR CLOSE"
    )
    payload = {
        "combos": combo_tuples,
        "fold_windows": fold_windows,
        "min_is_trades": min_is_trades,
        "min_oos_trades": min_oos_trades,
        "objective": objective_name,
        "vol_filter_definition": vol_filter_definition,
        "atr_method": atr_method,
        "exit_sequencing_tag": exit_sequencing_tag,
        "cost_constants": asdict(cost_model),
        "time_stop_et": TIME_STOP_ET,
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
    produces strict, portable JSON (mirrors run_phase4.py's helper)."""
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _serialize_trade(t: Trade) -> dict:
    """Compact per-trade record -- enough to redraw equity curves and audit
    cost routing, not the full Trade (entry/stop/target/exit prices
    omitted). Includes `net_pnl`/`exit_reason` on top of Phase-4's fields
    since Phase 5's whole point is the net-vs-gross comparison."""
    return {
        "entry_time": t.entry_time.isoformat(),
        "direction": t.direction,
        "pnl_usd": t.pnl_usd,
        "net_pnl": t.net_pnl,
        "r_multiple": t.r_multiple,
        "outcome": t.outcome,
        "exit_reason": t.exit_reason,
    }


def _net_aggregate(trades: list[Trade]) -> dict:
    """Net-metric aggregate for a stitched trade list (Blocker 2: net
    everywhere). Mirrors `walkforward_p5._net_metrics`'s
    fields but drops the raw `trades`/`params` keys (not JSON-safe /
    redundant at this level) and adds `gross_total_pnl` so charts and the
    writeup can show the gross-vs-net erosion from costs directly."""
    net_pnls = [t.net_pnl for t in trades]
    gross_pnls = [t.pnl_usd for t in trades]
    n = len(trades)
    return {
        "n_trades": n,
        "net_profit_factor": profit_factor(net_pnls),
        "net_win_rate": win_rate(net_pnls),
        "net_total_pnl": total_pnl(net_pnls),
        "net_max_drawdown": max_drawdown(net_pnls),
        "net_expectancy_per_trade": (total_pnl(net_pnls) / n) if n else 0.0,
        "gross_total_pnl": total_pnl(gross_pnls),
    }


def _trades_in_window(trades: list[Trade], start: pd.Timestamp, end: pd.Timestamp) -> list[Trade]:
    return [t for t in trades if start <= t.entry_time < end]


def _capped(pf: float) -> float:
    """Cap a profit factor at PF_CAP (all-win combos give gross_loss==0 ->
    inf under metrics.profit_factor's convention) -- guards every headline
    number that lands in the JSON/success-rule against an unserializable
    inf, and is REQUIRED before condition (d)'s comparison (review note:
    the 20-combo null distribution is already capped, so an uncapped tuned
    pick could clear (d) trivially)."""
    return min(pf, PF_CAP)


# --- cost sensitivity (algebraic recompute, no re-sweep) --------------------


def _net_pnls_at_multiplier(trades: list[Trade], multiplier: float) -> list[float]:
    """Net P&L at cost `multiplier`, derived algebraically from each
    trade's already-computed gross `pnl_usd` and 1x `net_pnl` -- see module
    docstring for why this is exact (not an approximation) even for
    partial_1R's two-leg trades."""
    out = []
    for t in trades:
        cost_1x = t.pnl_usd - t.net_pnl
        out.append(t.pnl_usd - cost_1x * multiplier)
    return out


def _cost_sensitivity(stitched_tuned: list[Trade], stitched_default: list[Trade]) -> dict:
    """Recompute stitched tuned & base NET PF + total_pnl at cost
    multipliers 0x/1x/2x (the spec's pre-registered cost sensitivity).
    `multiplier` is never used for selection -- these are the SAME picks
    (per-fold selected exit_mode/vol_filter, and the fixed base combo)
    already chosen under the pre-registered 1x cost model; only their
    reported cost exposure changes."""
    out = {}
    for mult in (0.0, 1.0, 2.0):
        tuned_net = _net_pnls_at_multiplier(stitched_tuned, mult)
        base_net = _net_pnls_at_multiplier(stitched_default, mult)
        tuned_pf = _capped(profit_factor(tuned_net))
        base_pf = _capped(profit_factor(base_net))
        out[f"{mult:g}x"] = {
            "multiplier": mult,
            "tuned": {"net_profit_factor": tuned_pf, "net_total_pnl": total_pnl(tuned_net)},
            "base": {"net_profit_factor": base_pf, "net_total_pnl": total_pnl(base_net)},
            "tuned_pf_gt_1": bool(tuned_pf > 1.0),
            "tuned_beats_base_margin_ge_0_10": bool((tuned_pf - base_pf) >= 0.10),
        }
    return out


# --- results assembly ---------------------------------------------------


def _build_results(df: pd.DataFrame, grid: list[StrategyParams], folds, cost_model: CostModel, t_start: float) -> dict:
    result = walk_forward_p5(df, grid, folds, cost_model=cost_model)
    fold_reports = result["folds"]
    eligibility_table = result["eligibility_table"]

    config_hash = _config_hash(grid, folds, MIN_IS_TRADES, MIN_OOS_TRADES, OBJECTIVE_NAME, cost_model)
    git_sha = _git_sha()

    # --- stitched OOS: all 4 folds, and excluding fallback folds -----------
    stitched_tuned_all = result["oos_trades_tuned"]
    stitched_default_all = result["oos_trades_default"]

    non_fallback = [fr for fr in fold_reports if not fr["fallback_used"]]
    fallback_fold_count = len(fold_reports) - len(non_fallback)

    stitched_tuned_excl = [t for fr in non_fallback for t in fr["oos_trades"]]
    stitched_default_excl = [
        t for fr in non_fallback for t in _trades_in_window(stitched_default_all, fr["test_start"], fr["test_end"])
    ]

    stitched_oos = {
        "all_folds": {
            "tuned": _net_aggregate(stitched_tuned_all),
            "base": _net_aggregate(stitched_default_all),
        },
        "excluding_fallback_folds": {
            "tuned": _net_aggregate(stitched_tuned_excl),
            "base": _net_aggregate(stitched_default_excl),
        },
        "fallback_fold_count": fallback_fold_count,
    }

    # --- 20-combo stitched-net-OOS-PF null (condition (d)'s distribution) --
    # Deterministic order (build_grid_p5()'s order), not dict-iteration order.
    combo_null_distribution = [
        {
            "exit_mode": p.exit_mode,
            "vol_filter": p.vol_filter,
            "stitched_net_oos_pf": result["stitched_net_oos_pf_by_combo"][p],  # already capped by walk_forward_p5
        }
        for p in grid
    ]
    combo_pfs = [r["stitched_net_oos_pf"] for r in combo_null_distribution]
    p75_all_combos = float(np.percentile(combo_pfs, 75))
    median_all_combos = float(np.median(combo_pfs))

    # --- success rule: 5 pre-registered conditions, headline = all-folds stitched net OOS ---
    tuned_pf_raw = stitched_oos["all_folds"]["tuned"]["net_profit_factor"]
    base_pf_raw = stitched_oos["all_folds"]["base"]["net_profit_factor"]
    tuned_pf = _capped(tuned_pf_raw)
    base_pf = _capped(base_pf_raw)

    cond_a = tuned_pf > 1.0
    margin = tuned_pf - base_pf
    cond_b = margin >= 0.10

    folds_tuned_beats_base = sum(
        1
        for fr in fold_reports
        if _capped(fr["oos_net_metrics"]["net_profit_factor"]) > _capped(fr["oos_net_default_metrics"]["net_profit_factor"])
    )
    cond_c = folds_tuned_beats_base >= 3

    # Condition (d): CAP the tuned pick's stitched net PF at PF_CAP BEFORE
    # comparing against the (already-capped) 20-combo distribution's p75 --
    # review note: otherwise an all-win pick clears (d) trivially.
    cond_d = tuned_pf >= p75_all_combos

    n_oos_trades = stitched_oos["all_folds"]["tuned"]["n_trades"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    # alpha=0.10 -> percentiles [5, 95]: the lower bound IS the 5th
    # percentile condition (e) is scored on (reusing the 2-sided-CI helper,
    # which defaults to alpha=0.05 / a 95% CI -- alpha=0.10 is what turns
    # its lower percentile into exactly "5th pct").
    ci_lo, ci_hi = _bootstrap_pf_ci(
        [t.net_pnl for t in stitched_tuned_all], rng, n_boot=BOOTSTRAP_N, alpha=0.10, cap=PF_CAP
    )
    cond_e_trade_count = n_oos_trades >= MIN_OOS_TRADES
    cond_e_ci = ci_lo > 1.0
    cond_e = cond_e_trade_count and cond_e_ci

    success_rule = {
        "condition_a_tuned_net_pf_gt_1": {"tuned_stitched_net_oos_pf": tuned_pf, "passes": bool(cond_a)},
        "condition_b_margin_gte_0_10": {
            "tuned_stitched_net_oos_pf": tuned_pf,
            "base_stitched_net_oos_pf": base_pf,
            "margin": margin,
            "passes": bool(cond_b),
        },
        "condition_c_wins_3_of_4_folds": {
            "folds_tuned_beats_base": folds_tuned_beats_base,
            "total_folds": len(fold_reports),
            "passes": bool(cond_c),
        },
        "condition_d_beats_p75_combo_null": {
            "tuned_stitched_net_oos_pf_capped": tuned_pf,
            "p75_of_20_combos_stitched_net_oos_pf": p75_all_combos,
            "median_of_20_combos_stitched_net_oos_pf": median_all_combos,
            "passes": bool(cond_d),
        },
        "condition_e_oos_sample_gate": {
            "n_oos_trades": n_oos_trades,
            "min_oos_trades": MIN_OOS_TRADES,
            "trade_count_passes": bool(cond_e_trade_count),
            "bootstrap_ci_lower_5th_pct": ci_lo,
            "bootstrap_ci_upper_95th_pct": ci_hi,
            "bootstrap_n_boot": BOOTSTRAP_N,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "ci_lower_gt_1_passes": bool(cond_e_ci),
            "passes": bool(cond_e),
        },
        "robust_improvement": bool(cond_a and cond_b and cond_c and cond_d and cond_e),
    }

    # --- cost sensitivity (all-folds stitched tuned vs base) ---------------
    cost_sensitivity = _cost_sensitivity(stitched_tuned_all, stitched_default_all)

    # --- per-fold report -----------------------------------------------------
    folds_out = []
    for fr, elig in zip(fold_reports, eligibility_table):
        p = fr["selected_params"]
        folds_out.append(
            {
                "fold_index": fr["fold_index"],
                "train_start": fr["train_start"].isoformat(),
                "train_end": fr["train_end"].isoformat(),
                "test_start": fr["test_start"].isoformat(),
                "test_end": fr["test_end"].isoformat(),
                "selected_exit_mode": p.exit_mode,
                "selected_vol_filter": p.vol_filter,
                "selected_params": asdict(p),
                "fallback_used": fr["fallback_used"],
                "eligible": fr["eligible"],
                "pre_filter_in_session_signal_count": fr["pre_filter_in_session_signal_count"],
                "is_net_pf": fr["is_net_pf"],
                "is_net_total_pnl": fr["is_net_total_pnl"],
                "is_n": fr["is_n"],
                "is_net_max_drawdown": fr["is_net_max_drawdown"],
                "oos_net_metrics": fr["oos_net_metrics"],  # incl. net_expectancy, n_trades
                "oos_net_base_metrics": fr["oos_net_default_metrics"],
                "oos_trades": [_serialize_trade(t) for t in fr["oos_trades"]],
                "oos_pf_distribution": fr["oos_pf_distribution"],
                "selected_oos_percentile": fr["selected_oos_percentile"],
                "median_combo_oos_pf": fr["median_combo_oos_pf"],
                "p75_combo_oos_pf": fr["p75_combo_oos_pf"],
                "combo_eligibility": elig["combos"],
            }
        )

    return {
        "config_hash": config_hash,
        "git_sha": git_sha,
        "grid_size": result["grid_size"],
        "min_is_trades": MIN_IS_TRADES,
        "min_oos_trades": MIN_OOS_TRADES,
        "objective": OBJECTIVE_NAME,
        "cost_model": asdict(cost_model),
        "run_seconds": round(time.time() - t_start, 1),
        "folds": folds_out,
        "stitched_oos": stitched_oos,
        "combo_null_distribution": combo_null_distribution,
        "success_rule": success_rule,
        "cost_sensitivity": cost_sensitivity,
    }, {
        # kept out of the JSON payload but handed to the chart functions
        "stitched_tuned_all": stitched_tuned_all,
        "stitched_default_all": stitched_default_all,
    }


# --- charts ---------------------------------------------------------------


def _plot_equity_curve(stitched_tuned: list[Trade], stitched_default: list[Trade]) -> None:
    def curve(vals):
        running = 0.0
        out = []
        for v in vals:
            running += v
            out.append(running)
        return out

    tuned_net_curve = curve([t.net_pnl for t in stitched_tuned])
    tuned_gross_curve = curve([t.pnl_usd for t in stitched_tuned])
    base_net_curve = curve([t.net_pnl for t in stitched_default])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(tuned_net_curve) + 1), tuned_net_curve, color="#2563eb", linewidth=1.5, label="tuned (net)")
    ax.plot(range(1, len(base_net_curve) + 1), base_net_curve, color="#f59e0b", linewidth=1.5, label="base (net)")
    ax.plot(
        range(1, len(tuned_gross_curve) + 1),
        tuned_gross_curve,
        color="#9ca3af",
        linewidth=1.2,
        linestyle="--",
        label="tuned (gross, no costs)",
    )
    ax.axhline(0.0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax.set_title("Stitched OOS cumulative P&L -- tuned (net) vs base (net) vs tuned (gross)")
    ax.set_xlabel("OOS trade # (chronological, stitched across folds)")
    ax.set_ylabel("Cumulative P&L (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "phase5_equity_curve.png", dpi=150)
    plt.close(fig)


def _plot_oos_pf_per_fold(folds_out: list[dict]) -> None:
    labels = [f"F{fr['fold_index'] + 1}" for fr in folds_out]
    tuned_vals = [min(fr["oos_net_metrics"]["net_profit_factor"], 5.0) for fr in folds_out]
    base_vals = [min(fr["oos_net_base_metrics"]["net_profit_factor"], 5.0) for fr in folds_out]

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([xi - width / 2 for xi in x], tuned_vals, width=width, label="tuned (net)", color="#2563eb")
    ax.bar([xi + width / 2 for xi in x], base_vals, width=width, label="base (net)", color="#f59e0b")
    ax.axhline(1.0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Net OOS profit factor (capped at 5.0 for display)")
    ax.set_title("Per-fold net OOS profit factor -- tuned vs base")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "phase5_oos_pf_per_fold.png", dpi=150)
    plt.close(fig)


def _plot_selected_exit_vol_stability(folds_out: list[dict]) -> None:
    labels = [f"F{fr['fold_index'] + 1}" for fr in folds_out]
    exit_row = [_EXIT_MODE_ABBR[fr["selected_exit_mode"]] for fr in folds_out]
    vol_row = [fr["selected_vol_filter"] for fr in folds_out]
    fallback_row = ["yes" if fr["fallback_used"] else "no" for fr in folds_out]

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.axis("off")
    table = ax.table(
        cellText=[exit_row, vol_row, fallback_row],
        rowLabels=["exit_mode", "vol_filter", "fallback_used"],
        colLabels=labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    ax.set_title(
        "Selected exit_mode + vol_filter per fold (stability)\n"
        "caveat: n=4 overlapping-train-window folds, descriptive only",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "phase5_selected_exit_vol_stability.png", dpi=150)
    plt.close(fig)


def _plot_cost_sensitivity(cost_sensitivity: dict) -> None:
    mults = ["0x", "1x", "2x"]
    tuned_pnl = [cost_sensitivity[m]["tuned"]["net_total_pnl"] for m in mults]
    base_pnl = [cost_sensitivity[m]["base"]["net_total_pnl"] for m in mults]

    x = range(len(mults))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([xi - width / 2 for xi in x], tuned_pnl, width=width, label="tuned", color="#2563eb")
    ax.bar([xi + width / 2 for xi in x], base_pnl, width=width, label="base", color="#f59e0b")
    ax.axhline(0.0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax.set_xticks(list(x))
    ax.set_xticklabels(mults)
    ax.set_xlabel("Cost multiplier")
    ax.set_ylabel("Stitched net total P&L (USD)")
    ax.set_title("Cost sensitivity: stitched net total P&L at 0x / 1x / 2x costs")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "phase5_cost_sensitivity.png", dpi=150)
    plt.close(fig)


def _plot_combo_null(combo_null_distribution: list[dict], tuned_pf: float, median_pf: float, p75_pf: float) -> None:
    records = sorted(combo_null_distribution, key=lambda r: r["stitched_net_oos_pf"])
    labels = [f"{_EXIT_MODE_ABBR[r['exit_mode']]}\n{r['vol_filter']}" for r in records]
    vals = [min(r["stitched_net_oos_pf"], 5.0) for r in records]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    colors = ["#93c5fd"] * len(records)
    ax.bar(range(len(records)), vals, color=colors, edgecolor="#1e40af")
    ax.axhline(min(tuned_pf, 5.0), color="#dc2626", linewidth=2, label=f"selected pick (realized) = {tuned_pf:.3f}")
    ax.axhline(min(median_pf, 5.0), color="#111827", linewidth=1.5, linestyle="--", label=f"median = {median_pf:.3f}")
    ax.axhline(min(p75_pf, 5.0), color="#059669", linewidth=1.5, linestyle=":", label=f"p75 = {p75_pf:.3f}")
    ax.set_xticks(range(len(records)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Stitched net OOS profit factor (20 fixed combos, capped at 5.0)")
    ax.set_title("Selection-luck null: each of the 20 combos' OWN stitched net-OOS PF\nvs the realized (per-fold-selected) tuned pick")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "phase5_combo_null.png", dpi=150)
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    CHARTS_DIR.mkdir(exist_ok=True)

    print("Loading NQ data ...")
    df = load_nq()
    print(f"  {len(df):,} rows, {df.index[0]} .. {df.index[-1]} ({time.time() - t0:.1f}s)")

    grid = build_grid_p5()
    folds = make_folds()
    cost_model = CostModel()
    print(f"Grid size: {len(grid)}, folds: {len(folds)}")

    print("Running Phase-5 net walk-forward (20 combos x 4 folds, IS-select + OOS-eval + net null) ...")
    t1 = time.time()
    results, extra = _build_results(df, grid, folds, cost_model, t0)
    print(f"  walk_forward_p5 + assembly done in {time.time() - t1:.1f}s")

    RESULTS_PATH.write_text(json.dumps(_json_safe(results), indent=2, default=str))
    print(f"Wrote {RESULTS_PATH}")

    print("Plotting charts ...")
    _plot_equity_curve(extra["stitched_tuned_all"], extra["stitched_default_all"])
    _plot_oos_pf_per_fold(results["folds"])
    _plot_selected_exit_vol_stability(results["folds"])
    _plot_cost_sensitivity(results["cost_sensitivity"])
    cd = results["success_rule"]["condition_d_beats_p75_combo_null"]
    _plot_combo_null(
        results["combo_null_distribution"],
        tuned_pf=cd["tuned_stitched_net_oos_pf_capped"],
        median_pf=cd["median_of_20_combos_stitched_net_oos_pf"],
        p75_pf=cd["p75_of_20_combos_stitched_net_oos_pf"],
    )
    print(f"Wrote charts to {CHARTS_DIR}/ (phase5_*.png)")

    sr = results["success_rule"]
    print("\n=== HEADLINE (pre-registered, single-shot, NET) ===")
    print(f"config_hash={results['config_hash'][:16]}...  git_sha={results['git_sha'][:12]}")
    print(
        f"Tuned stitched-OOS net PF = {sr['condition_a_tuned_net_pf_gt_1']['tuned_stitched_net_oos_pf']:.4f} vs "
        f"base = {sr['condition_b_margin_gte_0_10']['base_stitched_net_oos_pf']:.4f} "
        f"(margin {sr['condition_b_margin_gte_0_10']['margin']:+.4f})"
    )
    print(
        f"(a) net PF>1.0: {sr['condition_a_tuned_net_pf_gt_1']['passes']}  "
        f"(b) margin>=0.10: {sr['condition_b_margin_gte_0_10']['passes']}  "
        f"(c) wins {sr['condition_c_wins_3_of_4_folds']['folds_tuned_beats_base']}/"
        f"{sr['condition_c_wins_3_of_4_folds']['total_folds']} folds: "
        f"{sr['condition_c_wins_3_of_4_folds']['passes']}  "
        f"(d) beats p75-combo null: {sr['condition_d_beats_p75_combo_null']['passes']}  "
        f"(e) OOS-sample gate: {sr['condition_e_oos_sample_gate']['passes']}"
    )
    print(f"ROBUST IMPROVEMENT (NET): {sr['robust_improvement']}")
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
