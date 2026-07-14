"""Hardened validation gates for the Phase-6 per-instrument datasets.

Adapts the Phase-1 validator (`validate.py`) per instrument, plus the
blocker-fix gates from the adversarial review:

  B1 - divergent-duplicate stats (identical vs divergent + percentiles)
       and an in-session anomaly-window detector (bad prints).
  B2 - contract-roll boundary detection (quarterly splice jumps), hard-failed
       if the yearly count is far from ~4/yr, and asserted to have mixed
       signs (proof of a splice artifact, not directional drift).
  I2 - timezone evidence hardened per DST-regime AND per calendar year
       (hard gate: raises on ambiguity).
  I3 - per-half-year session-day counts + per-session 09:30-10:30 bar
       completeness (descriptive; not a hard gate).
  I1 - cross-vendor sanity: per-day correlation of 1-min log returns between
       the independent-vendor NQ and the Phase-1 NQ loader on their overlap
       (hard gate).

Frozen module-level constants (never tuned after seeing Phase-6 data):
  ANOMALY_WINDOW_BARS, ANOMALY_K, ROLL_DAY_WINDOW, ROLL_K2, ROLL_MONTHS,
  ROLL_EXPECTED_PER_YEAR, CROSS_VENDOR_MIN_CORR, CROSS_VENDOR_MIN_COMMON_BARS.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from data.convert_p6 import parquet_path, read_dup_stats
from nqdata.load import load_nq
from nqdata.load_p6 import load_instrument

# --- frozen detector constants (B1/B2/I1) -----------------------------------
ANOMALY_WINDOW_BARS = 390       # ~1 session of 1-min bars: rolling robust-range window
ANOMALY_K = 20.0                # bar range > K x rolling median range -> flagged bad print

ROLL_DAY_WINDOW = (5, 18)       # day-of-month window searched for quarterly roll splices
ROLL_K2 = 3.0                   # flag if |gap| > K2 x that year's median |session-break gap|
ROLL_MONTHS = (3, 6, 9, 12)     # quarterly futures contract months
ROLL_EXPECTED_PER_YEAR = (2, 6) # accepted average rolls/year range; outside -> hard-fail

TICK_SIZES = {"ES": 0.25, "NQ": 0.25, "YM": 1.0}

CROSS_VENDOR_START = "2022-01-01"
CROSS_VENDOR_END = "2025-01-31"
CROSS_VENDOR_MIN_CORR = 0.9
CROSS_VENDOR_MIN_COMMON_BARS = 30

REPORT_PATH = "data/validation_report_p6.json"


# --- basic OHLC sanity -------------------------------------------------------

def count_ohlc_violations(df: pd.DataFrame) -> int:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    return int(((h < o) | (h < c) | (l > o) | (l > c) | (h < l)).sum())


# --- per-instrument tick grid ------------------------------------------------

def check_grid(df: pd.DataFrame, tick_size: float, tol: float = 1e-6) -> dict:
    cols = ["open", "high", "low", "close"]
    vals = df[cols].to_numpy().ravel()
    ratio = vals / tick_size
    resid = np.abs(ratio - np.round(ratio))
    n_off = int((resid > tol).sum())
    if n_off:
        raise ValueError(
            f"grid violation: {n_off} OHLC values are not on the {tick_size} tick grid"
        )
    return {"grid_ok": True, "tick_size": tick_size, "n_off_grid": 0}


# --- B1: in-session anomaly-window detector ---------------------------------

def detect_anomaly_windows(df: pd.DataFrame) -> list[dict]:
    rng = (df["high"] - df["low"]).to_numpy()
    s = pd.Series(rng)
    robust = s.rolling(ANOMALY_WINDOW_BARS, min_periods=50).median()
    threshold = ANOMALY_K * robust
    flagged = ((s > threshold) & robust.notna() & (robust > 0)).to_numpy()

    windows: list[dict] = []
    if flagged.any():
        idx = df.index
        change = np.diff(np.concatenate(([0], flagged.astype(int), [0])))
        starts = np.flatnonzero(change == 1)
        ends = np.flatnonzero(change == -1) - 1
        thr_vals = threshold.to_numpy()
        for s_, e_ in zip(starts, ends):
            windows.append({
                "start": str(idx[s_]),
                "end": str(idx[e_]),
                "n_bars": int(e_ - s_ + 1),
                "max_range": float(rng[s_:e_ + 1].max()),
                "threshold": float(np.nanmax(thr_vals[s_:e_ + 1])),
            })
    return windows


# --- shared trading-day convention -------------------------------------------

def trade_date(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Futures trading-day label: these instruments trade nearly 24h with the
    single real close being the 17:00-18:00 ET daily maintenance break (the
    spike-established quiet/reopen hours), not midnight. Bars at/after 18:00
    ET belong to the *next* calendar day's trading session (Globex
    convention), so the natural session boundary is shifted by +6h before
    normalizing to a date."""
    return (idx + pd.Timedelta(hours=6)).normalize()


# --- B2: contract-roll boundary detection -----------------------------------

def compute_session_breaks(df: pd.DataFrame) -> pd.DataFrame:
    """Per trading-day gap between that day's first open and the *previous*
    trading day's last close (sequential shift over the actual day index --
    correctly skips weekends/holidays)."""
    day = trade_date(df.index)
    g = df.groupby(day)
    first_open = g["open"].first()
    last_close = g["close"].last()
    prev_close = last_close.shift(1)
    gap = first_open - prev_close
    out = pd.DataFrame({"first_open": first_open, "prev_close": prev_close, "gap": gap})
    return out.dropna()


def detect_roll_boundaries(df: pd.DataFrame) -> dict:
    breaks = compute_session_breaks(df)
    if breaks.empty:
        raise ValueError("roll boundary detection: no session breaks available")

    breaks = breaks.copy()
    breaks["year"] = breaks.index.year
    breaks["month"] = breaks.index.month
    breaks["day"] = breaks.index.day
    median_abs_gap_by_year = breaks.groupby("year")["gap"].apply(lambda s: s.abs().median())

    roll_rows: list[dict] = []
    for year, g in breaks.groupby("year"):
        med = median_abs_gap_by_year.loc[year]
        if pd.isna(med) or med == 0:
            continue
        for month in ROLL_MONTHS:
            sub = g[(g["month"] == month)
                    & (g["day"] >= ROLL_DAY_WINDOW[0])
                    & (g["day"] <= ROLL_DAY_WINDOW[1])]
            if sub.empty:
                continue
            i_max = sub["gap"].abs().idxmax()
            max_gap = float(sub.loc[i_max, "gap"])
            if abs(max_gap) > ROLL_K2 * med:
                roll_rows.append({
                    "date": str(pd.Timestamp(i_max).date()),
                    "year": int(year),
                    "month": int(month),
                    "gap": max_gap,
                })

    n_years = int(breaks["year"].nunique())
    n_rolls = len(roll_rows)
    avg_per_year = n_rolls / n_years if n_years else 0.0
    signs = [1 if r["gap"] > 0 else -1 for r in roll_rows]
    signs_mixed = len(set(signs)) > 1 if signs else False

    result = {
        "roll_boundaries": roll_rows,
        "n_rolls": n_rolls,
        "n_years": n_years,
        "avg_per_year": avg_per_year,
        "signs_mixed": signs_mixed,
    }
    if not (ROLL_EXPECTED_PER_YEAR[0] <= avg_per_year <= ROLL_EXPECTED_PER_YEAR[1]):
        raise ValueError(
            f"roll boundary count hard-fail: {avg_per_year:.2f}/yr not in "
            f"{ROLL_EXPECTED_PER_YEAR}: {result}"
        )
    if n_rolls > 0 and not signs_mixed:
        raise ValueError(
            f"roll boundary signs are not mixed (looks like directional drift, "
            f"not a splice artifact): {result}"
        )
    return result


# --- I2: hardened timezone evidence (hard gate) -----------------------------

def compute_tz_evidence(df: pd.DataFrame) -> dict:
    idx = df.index
    if idx.tz is None:
        raise ValueError("tz evidence requires a tz-aware index")

    local_naive = idx.tz_localize(None)
    utc_naive = idx.tz_convert("UTC").tz_localize(None)
    offset_hours = (local_naive - utc_naive) / pd.Timedelta(hours=1)
    regime = np.where(offset_hours == -4, "EDT", np.where(offset_hours == -5, "EST", "OTHER"))
    n_other = int((regime == "OTHER").sum())
    if n_other:
        raise ValueError(f"tz evidence: {n_other} rows have a non-EDT/EST US/Eastern offset")

    tmp = pd.DataFrame({"year": idx.year.to_numpy(), "regime": regime, "hour": idx.hour.to_numpy()})
    counts = tmp.groupby(["year", "regime", "hour"]).size()

    per_bucket: dict = {}
    ok = True
    for (year, reg), grp in counts.groupby(level=[0, 1]):
        hour_counts = grp.droplevel([0, 1]).reindex(range(24), fill_value=0)
        quiet_hour = int(hour_counts.idxmin())
        reopen_hour = (quiet_hour + 1) % 24
        quiet_n = int(hour_counts.loc[quiet_hour])
        reopen_n = int(hour_counts.loc[reopen_hour])
        # Frozen criterion (spec I2, literal): quiet_hour==17 and reopen_hour==18.
        # (quiet_n/reopen_n are reported for diagnostic context only -- they are
        # NOT part of the gating condition, since count ratios are not a frozen
        # constant and using them here would be an ad hoc, undisclosed gate.)
        good = (quiet_hour == 17) and (reopen_hour == 18)
        per_bucket[f"{int(year)}-{reg}"] = {
            "quiet_hour": quiet_hour, "reopen_hour": reopen_hour,
            "quiet_n": quiet_n, "reopen_n": reopen_n, "ok": good,
        }
        ok = ok and good

    quiet_hour_by_year: dict = {}
    for year in sorted(set(tmp["year"].tolist())):
        yh = tmp.loc[tmp["year"] == year].groupby("hour").size().reindex(range(24), fill_value=0)
        quiet_hour_by_year[str(int(year))] = int(yh.idxmin())

    result = {
        "quiet_hour_by_dst_regime_and_year": per_bucket,
        "quiet_hour_by_year": quiet_hour_by_year,
        "ok": ok,
    }
    if not ok:
        raise ValueError(f"tz evidence hard-fail (ambiguous ET localization): {result}")
    return result


# --- I3: per-fold session integrity (descriptive) ---------------------------

def compute_session_integrity(df: pd.DataFrame) -> dict:
    day = trade_date(df.index)
    unique_days = pd.DatetimeIndex(sorted(set(day)))

    half_labels = np.where(unique_days.month <= 6, "H1", "H2")
    half_key = pd.Series([f"{d.year}-{h}" for d, h in zip(unique_days, half_labels)])
    per_halfyear = half_key.value_counts().sort_index()
    per_halfyear_out_of_range = {
        k: int(v) for k, v in per_halfyear.items() if not (115 <= v <= 135)
    }

    window = df.between_time("09:30", "10:30", inclusive="left")
    counts = window.groupby(trade_date(window.index)).size().reindex(unique_days, fill_value=0)
    incomplete = {str(d.date()): int(n) for d, n in counts.items() if n < 50}

    return {
        "per_halfyear_session_days": {k: int(v) for k, v in per_halfyear.items()},
        "per_halfyear_out_of_range": per_halfyear_out_of_range,
        "per_session_completeness": {
            "n_sessions_total": int(len(unique_days)),
            "n_incomplete": len(incomplete),
            "incomplete_sample": dict(list(incomplete.items())[:25]),
        },
    }


# --- I1: cross-vendor sanity (hard gate) ------------------------------------

def cross_vendor_gate(
    nq_indep: pd.DataFrame,
    nq_phase1: pd.DataFrame,
    start: str = CROSS_VENDOR_START,
    end: str = CROSS_VENDOR_END,
) -> dict:
    a = nq_indep.loc[start:end, "close"]
    b = nq_phase1.loc[start:end, "close"]
    joined = pd.DataFrame({"a": a, "b": b}).dropna()
    if joined.empty:
        raise ValueError("cross-vendor gate: no common bars in the overlap window")

    ret_a = np.log(joined["a"]).diff()
    ret_b = np.log(joined["b"]).diff()
    rets = pd.DataFrame({"a": ret_a, "b": ret_b}).dropna()

    day = rets.index.normalize()
    daily_corr: dict = {}
    failing_days: list[str] = []
    for d, g in rets.groupby(day):
        if len(g) < CROSS_VENDOR_MIN_COMMON_BARS:
            continue
        corr = g["a"].corr(g["b"])
        if pd.isna(corr):
            continue
        daily_corr[str(d.date())] = float(corr)
        if corr < CROSS_VENDOR_MIN_CORR:
            failing_days.append(str(d.date()))

    corr_vals = np.array(list(daily_corr.values()))
    dist = {
        "n_days": int(len(corr_vals)),
        "min": float(corr_vals.min()) if len(corr_vals) else None,
        "p5": float(np.percentile(corr_vals, 5)) if len(corr_vals) else None,
        "median": float(np.median(corr_vals)) if len(corr_vals) else None,
        "mean": float(corr_vals.mean()) if len(corr_vals) else None,
    }

    lag0 = float(rets["a"].corr(rets["b"]))
    lag_plus1 = float(rets["a"].corr(rets["b"].shift(1)))
    lag_minus1 = float(rets["a"].corr(rets["b"].shift(-1)))

    offset = joined["a"] - joined["b"]
    all_days = offset.index.normalize().unique()
    step = max(1, len(all_days) // 6)
    sample_days = all_days[::step][:6]
    offset_samples = {
        str(d.date()): float(offset[offset.index.normalize() == d].median())
        for d in sample_days
    }

    result = {
        "n_common_bars": int(len(joined)),
        "daily_correlation_distribution": dist,
        "failing_days": failing_days,
        "lag0_corr": lag0,
        "lag_plus1_corr": lag_plus1,
        "lag_minus1_corr": lag_minus1,
        "lag0_exceeds_neighbors": bool(lag0 > lag_plus1 and lag0 > lag_minus1),
        "offset_curve_samples": offset_samples,
    }
    if failing_days:
        raise ValueError(
            f"cross-vendor gate hard-fail: {len(failing_days)} days below "
            f"{CROSS_VENDOR_MIN_CORR} corr (first 10): {failing_days[:10]}"
        )
    if not result["lag0_exceeds_neighbors"]:
        raise ValueError(f"cross-vendor gate hard-fail: lag-0 corr does not exceed +/-1min lags: {result}")
    return result


# --- per-instrument report ---------------------------------------------------

def validate_instrument(df: pd.DataFrame, sym: str, *, dup_stats: dict | None = None) -> dict:
    sym_u = sym.upper()
    ohlc_viol = count_ohlc_violations(df)
    nan_ct = int(df[["open", "high", "low", "close", "volume"]].isna().any(axis=1).sum())
    spacing = df.index.to_series().diff().dropna()
    pct_1min = float((spacing == pd.Timedelta(minutes=1)).mean()) if len(spacing) else 0.0

    grid = check_grid(df, TICK_SIZES[sym_u])
    tz_evidence = compute_tz_evidence(df)
    roll_info = detect_roll_boundaries(df)
    anomaly_windows = detect_anomaly_windows(df)
    session_integrity = compute_session_integrity(df)

    report = {
        "symbol": sym_u,
        "n_rows": int(len(df)),
        "date_min": str(df.index.min()),
        "date_max": str(df.index.max()),
        "pct_1min_spacing_after_sort": pct_1min,
        "n_ohlc_violations": ohlc_viol,
        "n_nan": nan_ct,
        "grid_ok": grid["grid_ok"],
        "grid": grid,
        "session_days": int(pd.Index(trade_date(df.index)).nunique()),
        "tz_evidence": tz_evidence,
        "roll_boundaries": roll_info["roll_boundaries"],
        "roll_summary": {k: v for k, v in roll_info.items() if k != "roll_boundaries"},
        "n_anomaly_windows": len(anomaly_windows),
        "anomaly_windows": anomaly_windows[:50],
        **session_integrity,
    }
    if dup_stats:
        report["n_dups_identical"] = dup_stats.get("n_dups_identical")
        report["n_dups_divergent"] = dup_stats.get("n_dups_divergent")
        report["dup_divergence_pctiles"] = dup_stats.get("dup_divergence_pctiles")
        report["dups_dropped"] = dup_stats.get("dups_dropped")
        report["n_in_raw"] = dup_stats.get("n_in")
    return report


def main() -> dict:
    """Validate every instrument and write the committed report.

    Each per-instrument / cross-vendor hard gate still RAISES inside
    ``validate_instrument`` / ``cross_vendor_gate`` (unit-tested directly).
    At this orchestration layer we catch that raise per section so one
    instrument's hard-gate failure doesn't prevent writing a report that
    documents every OTHER section's real result -- a hard gate failure is a
    legitimate, disclosed BLOCKED outcome, not something to silently paper
    over or let crash into an empty report.
    """
    report: dict = {"instruments": {}}
    frames: dict[str, pd.DataFrame] = {}

    for sym in ("ES", "NQ", "YM"):
        df = load_instrument(sym)
        frames[sym] = df
        dup_stats = read_dup_stats(parquet_path(sym))
        try:
            info = validate_instrument(df, sym, dup_stats=dup_stats)
            info["gate_status"] = "PASS"
        except ValueError as e:
            info = {"symbol": sym, "gate_status": "HARD_GATE_FAILED", "error": str(e)}
        report["instruments"][sym] = info

    try:
        cv = cross_vendor_gate(frames["NQ"], load_nq())
        cv["gate_status"] = "PASS"
    except ValueError as e:
        cv = {"gate_status": "HARD_GATE_FAILED", "error": str(e)}
    report["cross_vendor_gate"] = cv

    all_pass = all(v.get("gate_status") == "PASS" for v in report["instruments"].values())
    all_pass = all_pass and report["cross_vendor_gate"].get("gate_status") == "PASS"
    report["overall_status"] = "PASS" if all_pass else "BLOCKED"

    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    return report


if __name__ == "__main__":
    r = main()
    print(json.dumps({"status": "ok", "instruments": list(r["instruments"])}, indent=2))
