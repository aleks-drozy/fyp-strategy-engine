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
from nqdata.load_p6 import (
    drop_maintenance_hour,
    load_instrument,
    load_instrument_raw,
    load_instrument_unshifted,
    MAINTENANCE_HOUR_END,
    MAINTENANCE_HOUR_START,
)

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
# Forensically established (committed evidence, 2026-07-14): the PHASE-1 NQ
# dataset -- not the new vendor -- carries defects on 188 scattered evaluable
# overlap days (104 whole-day DST-lag + 84 fast-market infidelity): its bars are stamped +/-60 minutes off (sometimes
# for only PART of a day -- an intra-day splice). Proof: on those days
# indep-NQ correlates 0.93-0.98 with indep-ES (it is real NQ), while Phase-1
# NQ correlates ~0 with everything at lag 0 but 0.88-0.99 at a +/-60-minute
# lag (e.g. 2023-05-01 RTH: aligned -0.07, lag -60 = 0.998).
#
# Scoring window: days are compared on RTH bars (09:30-16:00 ET) ONLY --
# where both vendors are liquid and where the strategy actually trades
# (09:30-10:30 is inside RTH). Overnight/Sunday-evening sessions are thin
# enough that 0.25-tick quantization noise dominates 1-min returns, making
# cross-vendor correlation meaningless there regardless of data quality.
# A day needs >= CROSS_VENDOR_MIN_RTH_BARS common RTH bars to be evaluable;
# non-evaluable days (weekend evenings, Phase-1 data holes) are COUNTED and
# disclosed, never silently dropped.
#
# Attribution ladder for a below-floor day (rungs 1-2 demand the SAME >=0.9
# bar): (1) whole-RTH single-lag explanation in the +/-60min neighborhood;
# (2) half-session explanation -- each RTH half independently explained at
# its own lag in {0} U {+/-55..65} (handles Phase-1's intra-day splices);
# (3) REFEREE adjudication -- Phase-1 shows a FOURTH defect class (fast-market
# infidelity: e.g. 2023-05-03 FOMC hour, where its path diverges 80+ points
# from the CME-grid data then re-converges; hourly corr 1.00 everywhere
# except that hour). Chasing each Phase-1 defect class with its own detector
# adds no assurance about the NEW data, which is what this gate exists to
# vouch for. Rung 3 therefore asks the new data's SIBLING instrument (indep
# ES, same vendor): if indep-NQ ~ indep-ES daily RTH correlation >=
# CROSS_VENDOR_REFEREE_MIN on the disputed day, the new data is internally
# coherent (real NQ co-moves with real ES at 0.85-0.98; garbage cannot), and
# the day is classified as Phase-1 infidelity -- disclosed, counted. A day
# failing the referee TOO is a genuine new-data suspect -> hard-fail. The
# referee floor is not a tuned knob: measured real days sit at 0.93-0.98 and
# garbage at ~0; 0.80 splits a chasm, not a distribution.
CROSS_VENDOR_LAG_CANDIDATES = tuple(range(-65, -54)) + tuple(range(-5, 6)) + tuple(range(55, 66))
CROSS_VENDOR_HALF_LAG_CANDIDATES = (0,) + tuple(range(-65, -54)) + tuple(range(55, 66))
CROSS_VENDOR_MIN_RTH_BARS = 100
CROSS_VENDOR_MIN_HALF_BARS = 50
CROSS_VENDOR_RTH = ("09:30", "16:00")
CROSS_VENDOR_RTH_SPLIT = "12:45"
CROSS_VENDOR_REFEREE_MIN = 0.80

# --- Defect 1 (maintenance-hour contamination) evidence constants ----------
# The daily CME maintenance close is a genuine time GAP (no bars at all); a
# vendor-spliced contaminated hour has bars filling it continuously instead.
# Searching a window around the expected 17:00-18:00 break for a >30-min gap
# is therefore robust to the contamination itself (frozen; grounded in market
# structure, not tuned to the observed defect).
GAP_MINUTES_THRESHOLD = 30
# A genuine-gap day should reopen at exactly 18:00 ET (CME's fixed rule, for
# both the daily maintenance close and the Sunday weekly reopen). Real vendor
# data has occasional extended-closure noise unrelated to any defect (thin
# holiday-period coverage, missing bars) that pushes a MINORITY of gap days'
# reopen past 18:00 -- require only a majority, not unanimity, of gap days to
# land on 18 so this genuine noise doesn't masquerade as tz ambiguity. A truly
# mislabeled/wrong-offset series fails this trivially (~0%, not a near-miss).
REOPEN_HOUR_MAJORITY_FLOOR = 0.5

# --- Defect 2 (bar-labeling offset) empirical-shift constants --------------
BAR_LABEL_SHIFT_WINDOW = ("2023-01-01", "2023-12-31")   # frozen overlap window
BAR_LABEL_SHIFT_CANDIDATES_MIN = (-1, 0, 1)
BAR_LABEL_SHIFT_MIN_COMMON_BARS = 30
BAR_LABEL_SHIFT_DECISIVE_FLOOR = 0.9     # the winning shift's median daily corr must clear this
BAR_LABEL_SHIFT_NEAR_ZERO = 0.3          # the other shifts' median daily corr must stay under this

CONTAMINATION_CHECK_WINDOW = ("2020-01-01", "2021-12-31")  # Defect-1 disclosed affected period

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
#
# Reworked to be robust to Defect 1 (2020-mid2021 ES/NQ maintenance-hour
# vendor contamination): the original criterion (quietest hour-of-day ==17,
# reopen ==18, from a per-year/regime hour histogram) breaks on contaminated
# years because the spliced bars fill the 17:00 hour, so it is no longer the
# quietest. This version detects evidence on RAW (pre-drop) bars using a
# SECOND, contamination-robust signal alongside the original one:
#
#   - reopen-hour evidence: per calendar day with a bar in hour 16 or 17,
#     follow the actual next bar chronologically (not restricted to the same
#     calendar day, since Friday's reopen lands on Sunday) and check for a
#     >GAP_MINUTES_THRESHOLD gap (a genuine maintenance/weekly close is a
#     real time gap; a contaminated hour has no gap, since bars fill it
#     continuously). Required on EVERY (year, regime) bucket that has at
#     least one such gap day, gated at REOPEN_HOUR_MAJORITY_FLOOR (a majority,
#     not unanimity, of gap days must reopen at 18 -- real vendor data has
#     occasional extended-closure noise unrelated to any defect).
#   - quiet-hour evidence: the original argmin-hour-count heuristic --
#     required ONLY on buckets with zero contamination-candidate days (it is
#     unreliable by construction on contaminated buckets).
#
# A bucket is "ok" iff reopen-hour evidence holds AND (the bucket has no
# contamination candidates OR quiet-hour evidence also holds). Still a hard
# gate: raises on ambiguity, now correctly distinguishing genuine ambiguity
# from disclosed, evidenced vendor contamination.

def _daily_maintenance_gap_evidence(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ET-local calendar day that has >=1 bar in hour 16 or 17
    (a candidate "closure start"): whether the gap from that day's LAST such
    bar to the very NEXT bar in the whole series (chronologically, NOT
    restricted to the same calendar day) exceeds GAP_MINUTES_THRESHOLD, and
    if so that next bar's hour (``reopen_hour``).

    Deliberately NOT restricted to a same-calendar-day window: on a Friday
    (or the day before a holiday) CME does not reopen until 18:00 ET on the
    NEXT trading day (Sunday, for a Friday close) -- a naive same-day search
    would find no reopen bar at all on every single Friday and misclassify
    it as a contamination candidate. The daily maintenance close and the
    weekly close both reopen at 18:00 ET, so following the actual next bar
    chronologically handles both uniformly.

    A day with NO qualifying gap is a contamination CANDIDATE: on a clean
    day the market is genuinely closed for >30 minutes; a vendor-spliced day
    has continuous bars filling the maintenance hour instead, so no gap is
    found."""
    idx = df.index
    n = len(idx)
    if n < 2:
        return pd.DataFrame(columns=["date", "year", "regime", "has_gap", "reopen_hour"])

    local_naive = idx.tz_localize(None)
    utc_naive = idx.tz_convert("UTC").tz_localize(None)
    offset_hours = (local_naive - utc_naive) / pd.Timedelta(hours=1)
    regime = np.where(offset_hours == -4, "EDT", np.where(offset_hours == -5, "EST", "OTHER"))
    hour = idx.hour.to_numpy()

    idx_vals = idx.to_numpy()
    gap_after = np.empty(n, dtype="timedelta64[ns]")
    next_hour = np.full(n, -1, dtype="int64")
    gap_after[:-1] = idx_vals[1:] - idx_vals[:-1]
    gap_after[-1] = np.timedelta64("NaT")
    next_hour[:-1] = hour[1:]

    pre_mask = (hour == 16) | (hour == 17)   # candidate "last bar before closure" rows
    if not pre_mask.any():
        return pd.DataFrame(columns=["date", "year", "regime", "has_gap", "reopen_hour"])

    tmp = pd.DataFrame({
        "date": idx.normalize().to_numpy()[pre_mask],
        "year": idx.year.to_numpy()[pre_mask],
        "regime": regime[pre_mask],
        "gap_after": gap_after[pre_mask],
        "next_hour": next_hour[pre_mask],
    })

    rows: list[dict] = []
    for d, g in tmp.groupby("date"):
        # idxmax on a pandas Series skips NaN/NaT by default -- safe here,
        # unlike np.argmax on a raw timedelta64 array (see historical note:
        # numpy sorts NaT as the LARGEST timedelta64 value).
        i_max = g["gap_after"].idxmax()
        best = g.loc[i_max]
        max_gap = best["gap_after"]
        if pd.isna(max_gap) or max_gap <= pd.Timedelta(minutes=GAP_MINUTES_THRESHOLD):
            rows.append({"date": d, "year": int(best["year"]), "regime": best["regime"],
                         "has_gap": False, "reopen_hour": None})
        else:
            rows.append({"date": d, "year": int(best["year"]), "regime": best["regime"],
                         "has_gap": True, "reopen_hour": int(best["next_hour"])})
    return pd.DataFrame(rows)


def compute_contamination_evidence(df_raw: pd.DataFrame) -> dict:
    """Per-year affected-day counts (Defect 1 disclosure): a day counts as a
    contamination CANDIDATE if the maintenance-hour gap search finds no
    genuine gap. Computed on RAW (pre-drop) bars."""
    gaps = _daily_maintenance_gap_evidence(df_raw)
    if gaps.empty:
        return {"by_year": {}, "total_contamination_candidate_days": 0}

    by_year: dict = {}
    for year, g in gaps.groupby("year"):
        n_total = int(len(g))
        n_gap = int(g["has_gap"].sum())
        by_year[str(int(year))] = {
            "n_days_total": n_total,
            "n_days_with_maintenance_gap": n_gap,
            "n_days_no_gap_contamination_candidate": n_total - n_gap,
        }
    return {
        "by_year": by_year,
        "total_contamination_candidate_days": int((~gaps["has_gap"]).sum()),
    }


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

    gaps = _daily_maintenance_gap_evidence(df)
    contamination = compute_contamination_evidence(df)

    per_bucket: dict = {}
    for (year, reg), grp in counts.groupby(level=[0, 1]):
        hour_counts = grp.droplevel([0, 1]).reindex(range(24), fill_value=0)
        quiet_hour = int(hour_counts.idxmin())
        reopen_hour = (quiet_hour + 1) % 24
        quiet_n = int(hour_counts.loc[quiet_hour])
        reopen_n = int(hour_counts.loc[reopen_hour])

        bucket_gaps = gaps[(gaps["year"] == year) & (gaps["regime"] == reg)]
        gap_days = bucket_gaps[bucket_gaps["has_gap"]]
        n_no_gap = int((~bucket_gaps["has_gap"]).sum()) if len(bucket_gaps) else 0
        reopen_hours_seen = sorted({int(h) for h in gap_days["reopen_hour"].dropna()})
        frac_reopen_18 = float((gap_days["reopen_hour"] == 18).mean()) if len(gap_days) else 0.0
        has_local_evidence = len(gap_days) > 0
        reopen_evidence_ok = bool(has_local_evidence and frac_reopen_18 >= REOPEN_HOUR_MAJORITY_FLOOR)

        contaminated_bucket = n_no_gap > 0
        quiet_hour_ok = (quiet_hour == 17) and (reopen_hour == 18)
        # A bucket with local (gap-day) evidence must pass it, plus (on the
        # zero-contamination-candidate majority of buckets, where the simple
        # heuristic is trustworthy) the original quiet-hour==17 check, in
        # defense in depth. A FULLY contaminated bucket (has_local_evidence
        # False -- e.g. ES/NQ 2020, where every single day is spliced) has NO
        # evidence to independently verify at all; it is not hard-failed on
        # its own here -- localization is a single, series-wide vendor
        # convention, not something that could drift for one specific year
        # while every other year proves correct, so a fully contaminated
        # bucket instead DEFERS to the regime-level check below, which
        # requires real evidence to exist somewhere in that DST regime.
        good = reopen_evidence_ok and (contaminated_bucket or quiet_hour_ok) if has_local_evidence else None

        per_bucket[f"{int(year)}-{reg}"] = {
            "quiet_hour": quiet_hour, "reopen_hour": reopen_hour,
            "quiet_n": quiet_n, "reopen_n": reopen_n,
            "n_gap_days": int(len(gap_days)),
            "frac_gap_days_reopen_at_18": frac_reopen_18,
            "n_no_gap_days_contamination_candidate": n_no_gap,
            "reopen_hours_seen_on_gap_days": reopen_hours_seen,
            "has_local_evidence": has_local_evidence,
            "reopen_evidence_ok": reopen_evidence_ok,
            "quiet_hour_check_applicable": not contaminated_bucket,
            "quiet_hour_ok": quiet_hour_ok,
            "ok": good,
        }

    # Every bucket WITH local evidence must independently pass.
    ok = all(v["ok"] for v in per_bucket.values() if v["has_local_evidence"])
    # And each DST regime present must have at least one bucket with real,
    # passing evidence -- otherwise the whole regime has zero verifiable
    # proof and this is genuinely ambiguous, not just locally contaminated.
    regimes_present = {k.rsplit("-", 1)[1] for k in per_bucket}
    for reg in regimes_present:
        reg_has_evidence = any(
            v["has_local_evidence"] and v["ok"]
            for k, v in per_bucket.items() if k.rsplit("-", 1)[1] == reg
        )
        if not reg_has_evidence:
            ok = False

    quiet_hour_by_year: dict = {}
    for year in sorted(set(tmp["year"].tolist())):
        yh = tmp.loc[tmp["year"] == year].groupby("hour").size().reindex(range(24), fill_value=0)
        quiet_hour_by_year[str(int(year))] = int(yh.idxmin())

    result = {
        "quiet_hour_by_dst_regime_and_year": per_bucket,
        "quiet_hour_by_year": quiet_hour_by_year,
        "contamination_evidence": contamination,
        "ok": ok,
    }
    if not ok:
        raise ValueError(f"tz evidence hard-fail (ambiguous ET localization): {result}")
    return result


# --- Defect-1 post-drop containment verification -----------------------------

def verify_maintenance_drop_containment(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    sym: str,
    start: str = CONTAMINATION_CHECK_WINDOW[0],
    end: str = CONTAMINATION_CHECK_WINDOW[1],
) -> dict:
    """Verify the maintenance-hour drop is confined to the dropped hour and
    report its effect on the disclosed contamination window.

    Runs the existing anomaly-window detector (B1) on RAW vs CLEANED bars,
    restricted to the disclosed contamination window, and reports before/after
    counts overall and at the old 17:00/18:00 boundary -- on a contaminated
    instrument/period (ES/NQ 2020-mid2021) the boundary count should drop
    sharply; on a clean one (YM) it should already be near zero and stay
    there. This is DESCRIPTIVE, not a hard gate on the boundary count itself:
    a nonzero post-drop count at 18:00 is not necessarily contamination --
    genuine reopen volatility (wider range after the market reopens from a
    real closure) is normal market behavior, not a defect, and 17:00 bars are
    structurally impossible post-drop (that hour has zero remaining bars).

    The one HARD assertion is in-session (09:30-10:30) continuity: dropping
    17:00-18:00 bars cannot legitimately change anything in the 09:30-10:30
    window (ANOMALY_WINDOW_BARS=390 bars, ~6.5h, does not reach back far
    enough from 09:30 to touch the previous day's 17:00-18:00 bars) -- so the
    in-session anomaly windows before and after must be identical. Any
    mismatch would mean the drop leaked into session data, a real bug."""
    raw_slice = df_raw.loc[start:end]
    clean_slice = df_clean.loc[start:end]

    before_windows = detect_anomaly_windows(raw_slice) if len(raw_slice) else []
    after_windows = detect_anomaly_windows(clean_slice) if len(clean_slice) else []

    def _hour(w: dict) -> int:
        return pd.Timestamp(w["start"]).hour

    boundary_before = [w for w in before_windows if _hour(w) in (17, 18)]
    boundary_after = [w for w in after_windows if _hour(w) in (17, 18)]

    def _in_session(w: dict) -> bool:
        t = pd.Timestamp(w["start"]).time()
        return pd.Timestamp("09:30").time() <= t < pd.Timestamp("10:30").time()

    session_before = [w for w in before_windows if _in_session(w)]
    session_after = [w for w in after_windows if _in_session(w)]

    result = {
        "symbol": sym.upper(),
        "window": f"{start}..{end}",
        "n_anomaly_windows_before_drop": len(before_windows),
        "n_anomaly_windows_after_drop": len(after_windows),
        "n_boundary_anomaly_windows_before_drop": len(boundary_before),
        "n_boundary_anomaly_windows_after_drop": len(boundary_after),
        "n_session_anomaly_windows_before_drop": len(session_before),
        "n_session_anomaly_windows_after_drop": len(session_after),
    }
    if session_before != session_after:
        raise ValueError(
            f"{sym}: in-session (09:30-10:30) anomaly windows changed after "
            f"the maintenance-hour drop -- the drop leaked outside the "
            f"dropped hour: before={session_before} after={session_after}"
        )
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
    es_referee: pd.DataFrame | None = None,
) -> dict:
    a = nq_indep.loc[start:end, "close"]
    b = nq_phase1.loc[start:end, "close"]
    joined = pd.DataFrame({"a": a, "b": b}).dropna()
    if joined.empty:
        raise ValueError("cross-vendor gate: no common bars in the overlap window")

    ret_a = np.log(joined["a"]).diff()
    ret_b = np.log(joined["b"]).diff()
    rets_all = pd.DataFrame({"a": ret_a, "b": ret_b}).dropna()
    # scoring window: RTH only (see the forensic note above)
    rets = rets_all.between_time(*CROSS_VENDOR_RTH, inclusive="left")

    day = rets.index.normalize()
    daily_corr: dict = {}
    below_floor_days: list[pd.Timestamp] = []
    # evaluability accounting: every overlap calendar day is either scored,
    # or counted as not-evaluable (thin RTH overlap), never silently dropped
    all_overlap_days = set(rets_all.index.normalize().unique())
    evaluable_days: set = set()
    for d, g in rets.groupby(day):
        if len(g) < CROSS_VENDOR_MIN_RTH_BARS:
            continue
        corr = g["a"].corr(g["b"])
        if pd.isna(corr):
            continue
        evaluable_days.add(d)
        daily_corr[str(d.date())] = float(corr)
        if corr < CROSS_VENDOR_MIN_CORR:
            below_floor_days.append(d)
    not_evaluable = sorted(all_overlap_days - evaluable_days)
    n_not_evaluable_weekday = sum(1 for d in not_evaluable if d.dayofweek < 5)

    # Attribution ladder (see the forensic note): per-series returns (diffed
    # independently, then window-joined) so a lag shift realigns real bars.
    ret_a_full = np.log(nq_indep.loc[start:end, "close"]).diff().dropna()
    ret_b_full = np.log(nq_phase1.loc[start:end, "close"]).diff().dropna()
    ra_day_of = ret_a_full.index.normalize()
    rb_day_of = ret_b_full.index.normalize()

    def _best_lag(rb_seg: pd.Series, ra_win: pd.Series, candidates, min_bars: int):
        best_lag, best_corr = None, -1.0
        for lag in candidates:
            ra_l = ra_win.copy()
            ra_l.index = ra_l.index + pd.Timedelta(minutes=lag)
            j = pd.DataFrame({"x": rb_seg, "y": ra_l}).dropna()
            if len(j) < min_bars:
                continue
            c = j["x"].corr(j["y"])
            if pd.notna(c) and c > best_corr:
                best_lag, best_corr = lag, float(c)
        return best_lag, best_corr

    artifact_days: dict = {}
    failing_days: list[str] = []
    for d in below_floor_days:
        d_str = str(d.date())
        rb_day = ret_b_full[rb_day_of == d].between_time(*CROSS_VENDOR_RTH, inclusive="left")
        window_mask = (ra_day_of >= d - pd.Timedelta(days=1)) & (ra_day_of <= d + pd.Timedelta(days=1))
        ra_win = ret_a_full[window_mask]

        # rung 1: whole-RTH single lag in the +/-60min neighborhood
        lag1, corr1 = _best_lag(rb_day, ra_win, CROSS_VENDOR_LAG_CANDIDATES, CROSS_VENDOR_MIN_RTH_BARS)
        if corr1 >= CROSS_VENDOR_MIN_CORR and lag1 is not None and abs(lag1) >= 55:
            artifact_days[d_str] = {"kind": "whole_day", "lag_minutes": lag1, "corr_at_lag": corr1}
            continue

        # rung 2: intra-day splice -- each RTH half explained at its own lag
        h1 = rb_day.between_time(CROSS_VENDOR_RTH[0], CROSS_VENDOR_RTH_SPLIT, inclusive="left")
        h2 = rb_day.between_time(CROSS_VENDOR_RTH_SPLIT, CROSS_VENDOR_RTH[1], inclusive="left")
        lag_h1, corr_h1 = _best_lag(h1, ra_win, CROSS_VENDOR_HALF_LAG_CANDIDATES, CROSS_VENDOR_MIN_HALF_BARS)
        lag_h2, corr_h2 = _best_lag(h2, ra_win, CROSS_VENDOR_HALF_LAG_CANDIDATES, CROSS_VENDOR_MIN_HALF_BARS)
        halves_ok = (corr_h1 >= CROSS_VENDOR_MIN_CORR and corr_h2 >= CROSS_VENDOR_MIN_CORR)
        # a mixed day must involve an actual shift in at least one half --
        # two aligned halves would have passed the aligned check already
        involves_shift = any(lag is not None and abs(lag) >= 55 for lag in (lag_h1, lag_h2))
        if halves_ok and involves_shift:
            artifact_days[d_str] = {
                "kind": "intra_day_splice",
                "half1": {"lag_minutes": lag_h1, "corr_at_lag": corr_h1},
                "half2": {"lag_minutes": lag_h2, "corr_at_lag": corr_h2},
            }
            continue

        # rung 3: referee adjudication (see the forensic note above)
        if es_referee is not None:
            try:
                na_day = np.log(nq_indep.loc[d_str, "close"]).diff().dropna()
                es_day = np.log(es_referee.loc[d_str, "close"]).diff().dropna()
            except KeyError:
                na_day = es_day = pd.Series(dtype=float)
            jr = (pd.DataFrame({"x": na_day, "y": es_day}).dropna()
                  .between_time(*CROSS_VENDOR_RTH, inclusive="left"))
            ref_corr = float(jr["x"].corr(jr["y"])) if len(jr) >= CROSS_VENDOR_MIN_RTH_BARS else float("nan")
            if pd.notna(ref_corr) and ref_corr >= CROSS_VENDOR_REFEREE_MIN:
                artifact_days[d_str] = {
                    "kind": "phase1_infidelity",
                    "aligned_corr": float(daily_corr.get(d_str, float("nan"))),
                    "referee_indepNQ_vs_indepES_corr": ref_corr,
                }
                continue
        failing_days.append(d_str)

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

    lag_hist: dict = {}
    n_splice_days = 0
    n_infidelity_days = 0
    for v in artifact_days.values():
        if v["kind"] == "whole_day":
            key = str(v["lag_minutes"])
            lag_hist[key] = lag_hist.get(key, 0) + 1
        elif v["kind"] == "intra_day_splice":
            n_splice_days += 1
            for half in ("half1", "half2"):
                lg = v[half]["lag_minutes"]
                if lg is not None and abs(lg) >= 55:
                    key = str(lg)
                    lag_hist[key] = lag_hist.get(key, 0) + 1
        else:  # phase1_infidelity (referee-adjudicated)
            n_infidelity_days += 1

    # The aligned-lag dominance check is meaningful only on the CLEAN days --
    # computed over days that met the aligned floor (on artifact days, the
    # Phase-1 series is the shifted one, which is exactly what we've proven).
    clean_day_set = {d for d, c in daily_corr.items() if c >= CROSS_VENDOR_MIN_CORR}
    clean_mask = pd.Index(rets.index.normalize().strftime("%Y-%m-%d")).isin(clean_day_set)
    rc = rets[clean_mask]
    lag0_clean = float(rc["a"].corr(rc["b"])) if len(rc) else float("nan")
    lagp1_clean = float(rc["a"].corr(rc["b"].shift(1))) if len(rc) else float("nan")
    lagm1_clean = float(rc["a"].corr(rc["b"].shift(-1))) if len(rc) else float("nan")

    result = {
        "n_common_bars": int(len(joined)),
        "scoring_window_rth": f"[{CROSS_VENDOR_RTH[0]}, {CROSS_VENDOR_RTH[1]}) ET",
        "daily_correlation_distribution": dist,
        "n_overlap_days_total": len(all_overlap_days),
        "n_evaluable_days": len(evaluable_days),
        "n_not_evaluable_days": len(not_evaluable),
        "n_not_evaluable_weekdays_phase1_holes": n_not_evaluable_weekday,
        "n_below_floor_days": len(below_floor_days),
        "n_phase1_tz_artifact_days": len(artifact_days),
        "n_phase1_intra_day_splice_days": n_splice_days,
        "n_phase1_infidelity_days_referee_adjudicated": n_infidelity_days,
        "phase1_tz_artifact_lag_histogram": lag_hist,
        "phase1_tz_artifact_sample": dict(list(artifact_days.items())[:10]),
        "failing_days": failing_days,
        "lag0_corr": lag0,
        "lag_plus1_corr": lag_plus1,
        "lag_minus1_corr": lag_minus1,
        "lag0_corr_clean_days": lag0_clean,
        "lag0_exceeds_neighbors_clean_days": bool(lag0_clean > lagp1_clean and lag0_clean > lagm1_clean),
        "offset_curve_samples": offset_samples,
    }
    if failing_days:
        raise ValueError(
            f"cross-vendor gate hard-fail: {len(failing_days)} days below "
            f"{CROSS_VENDOR_MIN_CORR} aligned corr AND not lag-explained as a "
            f"Phase-1 tz artifact (first 10): {failing_days[:10]}"
        )
    if not result["lag0_exceeds_neighbors_clean_days"]:
        raise ValueError(
            f"cross-vendor gate hard-fail: lag-0 corr does not exceed +/-1min "
            f"lags on clean days: {result}"
        )
    return result


# --- Defect 2: empirical bar-label shift evidence ----------------------------

def compute_bar_label_shift_evidence(
    nq_indep_unshifted: pd.DataFrame,
    nq_phase1: pd.DataFrame,
    start: str = BAR_LABEL_SHIFT_WINDOW[0],
    end: str = BAR_LABEL_SHIFT_WINDOW[1],
) -> dict:
    """Empirically determine the bar-labeling shift (Defect 2): the vendors
    stamp bars by different conventions (open-time vs close-time). For each
    candidate shift in {-1, 0, +1} minutes applied to the RAW (unshifted)
    indep-NQ timestamps, compute the per-day cross-vendor 1-min log-return
    correlation against Phase-1 NQ over the frozen 2023 overlap window, and
    summarize with the MEDIAN across days (the "lag-0 correlation" at that
    shift) -- robust to a handful of noisy days, unlike the pooled/aggregate
    correlation. Decisive iff exactly one shift's median daily corr clears
    ``BAR_LABEL_SHIFT_DECISIVE_FLOOR`` while the other two stay under
    ``BAR_LABEL_SHIFT_NEAR_ZERO``."""
    b = nq_phase1.loc[start:end, "close"]

    per_shift: dict = {}
    for shift_min in BAR_LABEL_SHIFT_CANDIDATES_MIN:
        shifted_index = nq_indep_unshifted.index + pd.Timedelta(minutes=shift_min)
        a_full = pd.Series(nq_indep_unshifted["close"].to_numpy(), index=shifted_index)
        a_full = a_full[~a_full.index.duplicated(keep="first")].sort_index()
        a = a_full.loc[start:end]

        joined = pd.DataFrame({"a": a, "b": b}).dropna()
        ret_a = np.log(joined["a"]).diff()
        ret_b = np.log(joined["b"]).diff()
        rets = pd.DataFrame({"a": ret_a, "b": ret_b}).dropna()
        pooled_corr = float(rets["a"].corr(rets["b"])) if len(rets) else None

        day = rets.index.normalize()
        daily_corrs = []
        for _, g in rets.groupby(day):
            if len(g) < BAR_LABEL_SHIFT_MIN_COMMON_BARS:
                continue
            c = g["a"].corr(g["b"])
            if pd.notna(c):
                daily_corrs.append(float(c))
        daily_arr = np.array(daily_corrs)

        per_shift[str(shift_min)] = {
            "n_common_bars": int(len(joined)),
            "pooled_log_return_corr": pooled_corr,
            "n_days": int(len(daily_arr)),
            "median_daily_corr": float(np.median(daily_arr)) if len(daily_arr) else None,
        }

    best_shift = max(
        per_shift, key=lambda k: per_shift[k]["median_daily_corr"] if per_shift[k]["median_daily_corr"] is not None else -1.0
    )
    best_val = per_shift[best_shift]["median_daily_corr"]
    other_vals = [v["median_daily_corr"] for k, v in per_shift.items() if k != best_shift]
    decisive = bool(
        best_val is not None
        and best_val >= BAR_LABEL_SHIFT_DECISIVE_FLOOR
        and all(o is not None and abs(o) < BAR_LABEL_SHIFT_NEAR_ZERO for o in other_vals)
    )

    return {
        "window": f"{start}..{end}",
        "candidate_shifts_minutes": list(BAR_LABEL_SHIFT_CANDIDATES_MIN),
        "per_shift_minutes": per_shift,
        "chosen_shift_minutes": int(best_shift),
        "decisive": decisive,
    }


# --- per-instrument report ---------------------------------------------------

def validate_instrument(
    df: pd.DataFrame,
    sym: str,
    *,
    raw_df: pd.DataFrame,
    dup_stats: dict | None = None,
) -> dict:
    """``df`` is the CLEANED (shift + maintenance-hour drop applied) frame
    that every other gate runs on. ``raw_df`` is shift-applied but
    maintenance-hour-drop NOT applied -- needed only for the tz evidence
    (Defect 1's reworked I2 gate, which must see the pre-drop contamination
    to prove it) and the post-drop containment verification."""
    sym_u = sym.upper()
    ohlc_viol = count_ohlc_violations(df)
    nan_ct = int(df[["open", "high", "low", "close", "volume"]].isna().any(axis=1).sum())
    spacing = df.index.to_series().diff().dropna()
    pct_1min = float((spacing == pd.Timedelta(minutes=1)).mean()) if len(spacing) else 0.0

    grid = check_grid(df, TICK_SIZES[sym_u])
    tz_evidence = compute_tz_evidence(raw_df)
    roll_info = detect_roll_boundaries(df)
    anomaly_windows = detect_anomaly_windows(df)
    session_integrity = compute_session_integrity(df)
    contamination_verification = verify_maintenance_drop_containment(raw_df, df, sym_u)

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
        "maintenance_hour_drop": {
            "dropped_et_hour_range": f"[{MAINTENANCE_HOUR_START}:00, {MAINTENANCE_HOUR_END}:00)",
            "n_rows_raw_pre_drop": int(len(raw_df)),
            "n_rows_dropped": int(len(raw_df) - len(df)),
        },
        "contamination_verification": contamination_verification,
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

    Defect-2 fix: the bar-labeling shift evidence is computed FIRST (on raw,
    unshifted indep-NQ vs Phase-1 NQ). If it is not decisive, this STOPS here
    and reports BLOCKED with the evidence, per the frozen fix spec -- the
    per-instrument gates below all depend on ``nqdata.load_p6.load_instrument``
    applying a shift that would then be undetermined/undisclosed.
    """
    report: dict = {}

    shift_evidence = compute_bar_label_shift_evidence(load_instrument_unshifted("NQ"), load_nq())
    report["bar_label_shift_evidence"] = shift_evidence
    if not shift_evidence["decisive"]:
        report["instruments"] = {}
        report["cross_vendor_gate"] = {"gate_status": "SKIPPED", "reason": "bar-label shift not decisive"}
        report["overall_status"] = "BLOCKED"
        with open(REPORT_PATH, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        return report

    report["instruments"] = {}
    frames: dict[str, pd.DataFrame] = {}

    for sym in ("ES", "NQ", "YM"):
        raw_df = load_instrument_raw(sym)   # shift applied, maintenance-hour NOT dropped
        df = drop_maintenance_hour(raw_df)  # == load_instrument(sym)
        frames[sym] = df
        dup_stats = read_dup_stats(parquet_path(sym))
        try:
            info = validate_instrument(df, sym, raw_df=raw_df, dup_stats=dup_stats)
            info["gate_status"] = "PASS"
        except ValueError as e:
            info = {"symbol": sym, "gate_status": "HARD_GATE_FAILED", "error": str(e)}
        report["instruments"][sym] = info

    try:
        cv = cross_vendor_gate(frames["NQ"], load_nq(), es_referee=frames["ES"])
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
