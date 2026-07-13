"""Validate generated trades against the two real TradingView trade logs.

From docs/superpowers/plans/2026-07-13-phase2-strategy-engine.md, Task 4.

Honesty constraints this module enforces (see the plan's Global Constraints):

- The two real logs are DISJOINT date ranges (2023-24 vs 2025-26). `compare()`
  window-clips BOTH the generated set and the real set to
  `[win_start, win_end]` before comparing, so a generated trade that falls
  outside a given log's coverage window is never scored as a false positive
  against that log.
- Absolute price/PnL comparisons are NOT valid headline metrics:
  `nqdata.load_nq()` is a back-adjusted continuous NQ series, while the real
  logs are unadjusted front-month prints (large, time-varying offset --
  e.g. load_nq 15017 vs real 12362 on 2023-03-06). `matched_entry_price_delta`
  is exposed purely as a secondary diagnostic; it is NEVER used to decide a
  match or to compute precision/recall.
- The match key is (entry NY session-date, direction) only. Generated trades
  are never filtered or tuned to improve the match -- the only filtering
  `compare()` performs is the window clip, applied identically to both sets.
"""
from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from metrics import profit_factor, total_pnl, win_rate

REAL_LOG_COLUMNS = ["entry_date", "direction", "entry", "exit", "pnl_usd"]
GENERATED_COLUMNS = ["entry_date", "direction", "entry", "pnl_usd"]


def parse_tv_log(path: str) -> pd.DataFrame:
    """Parse a TradingView "List of Trades" CSV export into one row per
    round-trip trade.

    Entry/Exit rows pair on `Trade #` (each trade number appears exactly
    twice: one `Entry long`/`Entry short` row and one `Exit long`/`Exit
    short` row -- TradingView emits the Exit row before the Entry row within
    a pair, so order in the file is not relied on). `direction` is derived
    from the Entry row's `Type` ("long" in Type.lower() -> "Long", else
    "Short"). `entry_date` is the NY session date of the Entry row's `Date
    and time` (the export's timestamps are already NY/exchange-local, so no
    further tz conversion is applied). `entry`/`exit` are the paired `Price
    USD` values; `pnl_usd` is the `Net P&L USD` value (identical on both
    rows of a pair in the real exports).
    """
    raw = pd.read_csv(path)
    raw["Type"] = raw["Type"].astype(str)

    entries = raw[raw["Type"].str.startswith("Entry")].set_index("Trade #")
    exits = raw[raw["Type"].str.startswith("Exit")].set_index("Trade #")

    rows = []
    for trade_no, erow in entries.iterrows():
        xrow = exits.loc[trade_no]
        direction = "Long" if "long" in erow["Type"].lower() else "Short"
        entry_dt = pd.to_datetime(erow["Date and time"])
        rows.append({
            "entry_date": entry_dt.date(),
            "direction": direction,
            "entry": float(erow["Price USD"]),
            "exit": float(xrow["Price USD"]),
            "pnl_usd": float(xrow["Net P&L USD"]),
        })

    out = pd.DataFrame(rows, columns=REAL_LOG_COLUMNS)
    return out.sort_values("entry_date", kind="stable").reset_index(drop=True)


def _to_date(value) -> date:
    """Normalize a date-like value (str, datetime.date, pd.Timestamp) to a
    plain datetime.date for window-bound comparisons."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.Timestamp(value).date()


def _entry_date(entry_time) -> date:
    """NY session date of a generated Trade's entry_time. entry_time is a
    tz-aware US/Eastern datetime when produced by backtest.engine.backtest
    (the df it runs over is US/Eastern-indexed); tz-naive values (e.g. in
    unit tests) are treated as already NY-local."""
    ts = pd.Timestamp(entry_time)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("America/New_York")
    return ts.date()


def _side_aggregate(df: pd.DataFrame) -> dict:
    """PF / win-rate / total PnL / direction mix for one side (generated or
    real), over whatever rows are already window-filtered in `df`."""
    pnls = df["pnl_usd"].astype(float).tolist() if len(df) else []
    long_n = int((df["direction"] == "Long").sum()) if len(df) else 0
    short_n = int((df["direction"] == "Short").sum()) if len(df) else 0
    return {
        "profit_factor": profit_factor(pnls),
        "win_rate": win_rate(pnls),
        "total_pnl": total_pnl(pnls),
        "direction_mix": {"Long": long_n, "Short": short_n},
    }


def compare(generated: list, real: pd.DataFrame, win_start, win_end) -> dict:
    """Compare generated trades to one real log, both window-clipped to
    `[win_start, win_end]` (inclusive both ends).

    Matching is on (entry_date, direction) only -- never on absolute price.
    A real trade and a generated trade on the same date but opposite
    directions do NOT match (each counts once as missed and once as extra,
    per the plan's Task 4 spec). No filtering or tuning of `generated` is
    performed beyond the window clip.
    """
    win_start_d = _to_date(win_start)
    win_end_d = _to_date(win_end)

    real_dates = real["entry_date"].map(_to_date)
    real_mask = (real_dates >= win_start_d) & (real_dates <= win_end_d)
    real_in = real.loc[real_mask].copy()
    real_in["entry_date"] = real_dates.loc[real_mask]

    n_real_in_window = int(len(real_in))
    n_real_excluded = int(len(real) - n_real_in_window)

    gen_records = []
    for tr in generated:
        d = _entry_date(tr.entry_time)
        if win_start_d <= d <= win_end_d:
            gen_records.append({
                "entry_date": d,
                "direction": tr.direction,
                "entry": tr.entry,
                "pnl_usd": tr.pnl_usd,
            })
    generated_in = pd.DataFrame(gen_records, columns=GENERATED_COLUMNS)
    n_generated_in_window = int(len(generated_in))

    # (entry_date, direction) is verified unique per day within each real
    # log, and the engine caps generated trades at MAX_TRADES_PER_DAY == 1,
    # so both sides are unique per key -- a plain dict keyed on (date,
    # direction) is safe (no need to guard against duplicate keys).
    real_by_key = {(r.entry_date, r.direction): r for r in real_in.itertuples()}
    gen_by_key = {(g.entry_date, g.direction): g for g in generated_in.itertuples()}

    matched_keys = set(real_by_key) & set(gen_by_key)
    n_matched = len(matched_keys)
    n_missed = n_real_in_window - n_matched
    n_extra = n_generated_in_window - n_matched

    precision = (n_matched / (n_matched + n_extra)) if (n_matched + n_extra) else 0.0
    recall = (n_matched / (n_matched + n_missed)) if (n_matched + n_missed) else 0.0

    matched_entry_price_delta = [
        float(gen_by_key[k].entry - real_by_key[k].entry) for k in sorted(matched_keys)
    ]

    return {
        "win_start": win_start_d.isoformat(),
        "win_end": win_end_d.isoformat(),
        "n_real_in_window": n_real_in_window,
        "n_real_excluded": n_real_excluded,
        "n_generated_in_window": n_generated_in_window,
        "n_matched": n_matched,
        "n_missed": n_missed,
        "n_extra": n_extra,
        "precision": precision,
        "recall": recall,
        "matched_entry_price_delta": matched_entry_price_delta,
        "aggregate": {
            "generated": _side_aggregate(generated_in),
            "real": _side_aggregate(real_in),
        },
    }
