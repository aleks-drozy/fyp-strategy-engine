"""Runner: execute the real bar-by-bar backtest over the full Phase-1 NQ
dataset, validate the generated trades against both real TradingView logs,
and write backtest_results.json + charts/*.png.

From the Phase-2 strategy-engine spec
(docs/specs/2026-07-12-phase2-strategy-engine-design.md).

Run: .venv/Scripts/python run_backtest.py
Requires: data/raw/Dataset_NQ_1min_2022_2025.csv (Phase-1 raw data, not
committed -- ~1.05M 1-min rows) and the two real logs at
C:/Users/Alex/Projects/Trading-Strategy-Monte-Carlo-Simulation/data/. Both
compute_ifvg/compute_cisd and the engine loop are pure-Python per-bar loops
over ~1.05M rows, so a full run takes a few minutes.
"""
from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest.engine import backtest
from metrics import max_drawdown, profit_factor, total_pnl, win_rate
from nqdata.load import load_nq
from validate_trades import _entry_date, compare, parse_tv_log

REAL_LOG_PATHS = {
    "2023_2024": "C:/Users/Alex/Projects/Trading-Strategy-Monte-Carlo-Simulation/data/NQ1_2023_2024.csv",
    "optimised": "C:/Users/Alex/Projects/Trading-Strategy-Monte-Carlo-Simulation/data/NQ1_optimised.csv",
}

# Phase-1 raw-data coverage bounds (the Phase-2 spec's data window).
DATA_WINDOW_START = date(2022, 12, 26)
DATA_WINDOW_END = date(2025, 12, 11)

# Known in-window real baselines (verified; see the Phase-2 spec) -- asserted
# below so a parsing/window regression fails loudly instead of silently
# shipping a wrong headline number.
KNOWN_BASELINES = {
    "2023_2024": {"n_real_in_window": 95, "real_total_pnl": -4600.0},
    "optimised": {"n_real_in_window": 59, "real_total_pnl": 18115.0},
}

CHARTS_DIR = Path("charts")
RESULTS_PATH = Path("backtest_results.json")


def _json_safe(obj):
    """Recursively replace inf/-inf/NaN floats (e.g. an undefined profit_factor
    when a window has zero losing trades) with None so the output is strict,
    portable JSON -- Python's json.dumps would otherwise silently emit the
    non-standard Infinity/-Infinity/NaN tokens."""
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _log_window(real_df) -> tuple[date, date]:
    dmin = real_df["entry_date"].min()
    dmax = real_df["entry_date"].max()
    win_start = max(dmin, DATA_WINDOW_START)
    win_end = min(dmax, DATA_WINDOW_END)
    return win_start, win_end


def _generated_out_of_all_windows(generated: list, windows: dict[str, tuple[date, date]]) -> dict:
    """Honesty requirement from the T4 review: count + bucket the GENERATED
    trades that fall outside BOTH logs' compare() windows, so they don't
    silently vanish from every precision denominator. Buckets: before the
    first window, in the gap between the two (disjoint) windows, and after
    the last window. Windows are sorted by start so this generalizes even
    if the two logs' chronological order ever changes.
    """
    ordered = sorted(windows.items(), key=lambda kv: kv[1][0])
    (first_name, (first_start, first_end)), (second_name, (second_start, second_end)) = ordered
    if first_end > second_start:
        raise ValueError(
            f"expected disjoint, non-overlapping log windows but got "
            f"{first_name}={first_start}..{first_end} overlapping "
            f"{second_name}={second_start}..{second_end}"
        )

    def _covered(d: date) -> bool:
        return (first_start <= d <= first_end) or (second_start <= d <= second_end)

    buckets = {"before_first_window": [], "inter_log_gap": [], "after_last_window": []}
    for tr in generated:
        d = _entry_date(tr.entry_time)
        if _covered(d):
            continue
        if d < first_start:
            buckets["before_first_window"].append(d)
        elif d > second_end:
            buckets["after_last_window"].append(d)
        else:
            buckets["inter_log_gap"].append(d)

    def _summary(dates: list[date]) -> dict:
        return {
            "count": len(dates),
            "date_range": [min(dates).isoformat(), max(dates).isoformat()] if dates else None,
        }

    return {
        "count": sum(len(v) for v in buckets.values()),
        "before_first_window": _summary(buckets["before_first_window"]),
        "inter_log_gap": {
            **_summary(buckets["inter_log_gap"]),
            "gap_bounds": [first_end.isoformat(), second_start.isoformat()],
        },
        "after_last_window": _summary(buckets["after_last_window"]),
    }


def _generated_aggregate(trades: list) -> dict:
    pnls = [tr.pnl_usd for tr in trades]
    long_n = sum(1 for tr in trades if tr.direction == "Long")
    short_n = sum(1 for tr in trades if tr.direction == "Short")
    return {
        "profit_factor": profit_factor(pnls),
        "win_rate": win_rate(pnls),
        "total_pnl": total_pnl(pnls),
        "max_drawdown": max_drawdown(pnls),
        "direction_mix": {"Long": long_n, "Short": short_n},
    }


def _plot_equity_curve(trades: list) -> None:
    running = 0.0
    curve = []
    for tr in trades:
        running += tr.pnl_usd
        curve.append(running)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(curve) + 1), curve, color="#2563eb", linewidth=1.5)
    ax.axhline(0.0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax.set_title("Generated trades -- cumulative P&L (equity curve)")
    ax.set_xlabel("Trade # (chronological)")
    ax.set_ylabel("Cumulative P&L (USD)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "equity_curve.png", dpi=150)
    plt.close(fig)


def _plot_coverage(log_reports: dict) -> None:
    labels = ["real_in_window", "matched", "missed", "extra"]
    log_names = list(log_reports.keys())
    x = range(len(labels))
    width = 0.8 / len(log_names)
    colors = ["#2563eb", "#f59e0b"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(log_names):
        r = log_reports[name]
        values = [r["n_real_in_window"], r["n_matched"], r["n_missed"], r["n_extra"]]
        offsets = [xi + (i - (len(log_names) - 1) / 2) * width for xi in x]
        ax.bar(offsets, values, width=width, label=name, color=colors[i % len(colors)])

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Trade count")
    ax.set_title("Coverage vs real logs (in-window)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "coverage.png", dpi=150)
    plt.close(fig)


def _plot_pf_wr(log_reports: dict) -> None:
    log_names = list(log_reports.keys())
    fig, (ax_pf, ax_wr) = plt.subplots(1, 2, figsize=(11, 5))
    width = 0.35
    x = range(len(log_names))

    for ax, metric, title in ((ax_pf, "profit_factor", "Profit factor"), (ax_wr, "win_rate", "Win rate")):
        gen_vals = [log_reports[n]["aggregate"]["generated"][metric] for n in log_names]
        real_vals = [log_reports[n]["aggregate"]["real"][metric] for n in log_names]
        # inf PF (e.g. no losers in-window) plots as NaN-safe cap so bars stay visible
        gen_vals = [v if v not in (float("inf"),) else 0.0 for v in gen_vals]
        real_vals = [v if v not in (float("inf"),) else 0.0 for v in real_vals]

        ax.bar([xi - width / 2 for xi in x], gen_vals, width=width, label="generated", color="#2563eb")
        ax.bar([xi + width / 2 for xi in x], real_vals, width=width, label="real", color="#f59e0b")
        ax.set_xticks(list(x))
        ax.set_xticklabels(log_names)
        ax.set_title(f"{title} (generated vs real, same window)")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "generated_vs_real_pf_wr.png", dpi=150)
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    CHARTS_DIR.mkdir(exist_ok=True)

    print("Loading NQ data ...")
    df = load_nq()
    print(f"  {len(df):,} rows, {df.index[0]} .. {df.index[-1]} ({time.time() - t0:.1f}s)")

    print("Running backtest (bar-by-bar, pure Python -- this takes a few minutes) ...")
    t1 = time.time()
    trades = backtest(df)
    same_bar_span_count = backtest.same_bar_span_count
    print(f"  {len(trades)} generated trades, {same_bar_span_count} same-bar-span exits "
          f"({time.time() - t1:.1f}s)")

    print("Parsing real logs ...")
    real_logs = {name: parse_tv_log(path) for name, path in REAL_LOG_PATHS.items()}

    print("Computing per-log windows + compare() reports ...")
    windows: dict[str, tuple[date, date]] = {}
    log_reports: dict[str, dict] = {}
    for name, real_df in real_logs.items():
        win_start, win_end = _log_window(real_df)
        windows[name] = (win_start, win_end)
        log_reports[name] = compare(trades, real_df, win_start, win_end)
        print(f"  {name}: window {win_start}..{win_end} -- "
              f"n_real_in_window={log_reports[name]['n_real_in_window']} "
              f"n_matched={log_reports[name]['n_matched']} "
              f"n_missed={log_reports[name]['n_missed']} "
              f"n_extra={log_reports[name]['n_extra']}")

    # --- Sanity assert the known in-window real baselines --------------------
    for name, baseline in KNOWN_BASELINES.items():
        report = log_reports[name]
        if report["n_real_in_window"] != baseline["n_real_in_window"]:
            raise RuntimeError(
                f"{name}: expected {baseline['n_real_in_window']} real in-window trades, "
                f"got {report['n_real_in_window']} -- parsing/window regression"
            )
        real_pnl = report["aggregate"]["real"]["total_pnl"]
        if abs(real_pnl - baseline["real_total_pnl"]) >= 1e-6:
            raise RuntimeError(
                f"{name}: expected real total_pnl {baseline['real_total_pnl']}, "
                f"got {real_pnl} -- parsing/window regression"
            )
    print("Sanity baselines OK: 2023-24 = 95/-4600.0, optimised = 59/+18115.0")

    out_of_all_windows = _generated_out_of_all_windows(trades, windows)
    print(f"Generated trades outside both log windows: {out_of_all_windows['count']}")

    generated_aggregate = _generated_aggregate(trades)

    results = {
        "generated": {
            "total_trades": len(trades),
            "aggregate": generated_aggregate,
            "same_bar_span_count": same_bar_span_count,
            "out_of_all_windows": out_of_all_windows,
        },
        "logs": {
            name: {
                "source_path": REAL_LOG_PATHS[name],
                "log_date_min": real_logs[name]["entry_date"].min().isoformat(),
                "log_date_max": real_logs[name]["entry_date"].max().isoformat(),
                "compare": report,
            }
            for name, report in log_reports.items()
        },
        "meta": {
            "data_path": "data/raw/Dataset_NQ_1min_2022_2025.csv",
            "data_rows": len(df),
            "data_date_min": df.index[0].isoformat(),
            "data_date_max": df.index[-1].isoformat(),
            "fill_mode": "next_open",
            "run_seconds": round(time.time() - t0, 1),
        },
    }

    RESULTS_PATH.write_text(json.dumps(_json_safe(results), indent=2, default=str))
    print(f"Wrote {RESULTS_PATH}")

    print("Plotting charts ...")
    _plot_equity_curve(trades)
    _plot_coverage(log_reports)
    _plot_pf_wr(log_reports)
    print(f"Wrote charts to {CHARTS_DIR}/")

    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
