import numpy as np
import pandas as pd
import pytest

import validate_p6


def _ohlc_frame(index, open_, high, low, close):
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 0}, index=index
    )


# --- OHLC violations ----------------------------------------------------------

def test_count_ohlc_violations_flags_bad_bar():
    idx = pd.date_range("2023-01-03 09:30", periods=2, freq="1min", tz="US/Eastern")
    df = _ohlc_frame(idx, [100.0, 100.0], [100.5, 99.0], [99.0, 99.0], [100.0, 101.0])
    # row 1: high(99.0) < open(100.0) and < close(101.0) -> violation
    assert validate_p6.count_ohlc_violations(df) == 1


def test_count_ohlc_violations_clean():
    idx = pd.date_range("2023-01-03 09:30", periods=2, freq="1min", tz="US/Eastern")
    df = _ohlc_frame(idx, [100.0, 100.5], [101.0, 101.5], [99.5, 100.0], [100.5, 101.0])
    assert validate_p6.count_ohlc_violations(df) == 0


# --- per-instrument tick grid --------------------------------------------------

def test_check_grid_flags_off_grid_close():
    idx = pd.date_range("2023-01-03 09:30", periods=3, freq="1min", tz="US/Eastern")
    df = _ohlc_frame(idx, [100, 101, 102], [100, 101, 102], [100, 101, 102], [100, 101, 102.25])
    with pytest.raises(ValueError):
        validate_p6.check_grid(df, validate_p6.TICK_SIZES["YM"])


def test_check_grid_passes_on_grid():
    idx = pd.date_range("2023-01-03 09:30", periods=3, freq="1min", tz="US/Eastern")
    vals = [100.0, 100.25, 100.5]
    df = _ohlc_frame(idx, vals, vals, vals, vals)
    result = validate_p6.check_grid(df, validate_p6.TICK_SIZES["ES"])
    assert result["grid_ok"] is True


# --- B1: anomaly-window detector -----------------------------------------------

def test_detect_anomaly_windows_flags_injected_bad_print():
    n = 500
    idx = pd.date_range("2023-01-03 09:30", periods=n, freq="1min", tz="US/Eastern")
    rng = np.random.default_rng(0)
    base = 100 + np.cumsum(rng.normal(0, 0.01, n))
    high = base + 0.1
    low = base - 0.1
    bad_i = 300
    high = high.copy()
    low = low.copy()
    high[bad_i] = base[bad_i] + 25.0
    low[bad_i] = base[bad_i] - 25.0
    df = _ohlc_frame(idx, base, high, low, base)

    windows = validate_p6.detect_anomaly_windows(df)

    assert len(windows) == 1
    assert windows[0]["n_bars"] == 1
    assert pd.Timestamp(windows[0]["start"]) == idx[bad_i]


def test_detect_anomaly_windows_no_false_positive_on_quiet_series():
    n = 500
    idx = pd.date_range("2023-01-03 09:30", periods=n, freq="1min", tz="US/Eastern")
    rng = np.random.default_rng(1)
    base = 100 + np.cumsum(rng.normal(0, 0.01, n))
    df = _ohlc_frame(idx, base, base + 0.1, base - 0.1, base)
    assert validate_p6.detect_anomaly_windows(df) == []


# --- B2: roll boundary detection -----------------------------------------------

def _daily_frame(year, jump_dates=None):
    jump_dates = jump_dates or {}
    days = pd.bdate_range(f"{year}-01-01", f"{year}-12-31")
    idx, rows = [], []
    price = 100.0
    for i, d in enumerate(days):
        noise = 0.05 if i % 2 == 0 else -0.05
        gap = jump_dates.get(d.date(), 0.0) + noise
        open_ = price + gap
        close_ = open_ + 0.02
        t_open = d + pd.Timedelta(hours=9, minutes=30)
        t_close = d + pd.Timedelta(hours=15, minutes=59)
        idx += [t_open, t_close]
        rows += [
            {"open": open_, "high": open_ + 0.2, "low": open_ - 0.2, "close": open_},
            {"open": close_, "high": close_ + 0.2, "low": close_ - 0.2, "close": close_},
        ]
        price = close_
    tz_idx = pd.DatetimeIndex(idx).tz_localize(
        "US/Eastern", ambiguous="NaT", nonexistent="shift_forward"
    )
    df = pd.DataFrame(rows, index=tz_idx)
    df["volume"] = 0
    return df[df.index.notna()].sort_index()


def test_detect_roll_boundaries_finds_all_four_injected_jumps():
    year = 2023
    jump_dates = {
        pd.Timestamp(f"{year}-03-10").date(): 50.0,
        pd.Timestamp(f"{year}-06-12").date(): -50.0,
        pd.Timestamp(f"{year}-09-14").date(): 50.0,
        pd.Timestamp(f"{year}-12-15").date(): -50.0,
    }
    df = _daily_frame(year, jump_dates)

    result = validate_p6.detect_roll_boundaries(df)

    assert result["n_rolls"] == 4
    assert result["signs_mixed"] is True
    detected_dates = {r["date"] for r in result["roll_boundaries"]}
    assert detected_dates == {str(d) for d in jump_dates}


def test_detect_roll_boundaries_hard_fails_on_no_roll_series():
    df = _daily_frame(2023, jump_dates=None)
    with pytest.raises(ValueError):
        validate_p6.detect_roll_boundaries(df)


# --- I2: hardened tz evidence ---------------------------------------------------

def _synthetic_naive_index(windows, hour_shift=None):
    idx = []
    for start, n in windows:
        cur = pd.Timestamp(start)
        for d in range(n):
            day = cur + pd.Timedelta(days=d)
            quiet_hour = hour_shift(day) if hour_shift else 17
            for hour in range(24):
                if hour == quiet_hour:
                    continue
                for minute in range(0, 60, 5):
                    idx.append(day + pd.Timedelta(hours=hour, minutes=minute))
    return pd.DatetimeIndex(sorted(idx))


def _frame_from_naive_index(naive_idx):
    tz_idx = naive_idx.tz_localize("US/Eastern", ambiguous="NaT", nonexistent="shift_forward")
    tz_idx = tz_idx[tz_idx.notna()].sort_values()
    n = len(tz_idx)
    price = 100.0 + 0.01 * np.arange(n)
    return _ohlc_frame(tz_idx, price, price + 0.1, price - 0.1, price)


def test_tz_evidence_true_et_series_passes():
    idx = _synthetic_naive_index(
        [("2015-01-05", 5), ("2015-07-06", 5), ("2016-01-04", 5), ("2016-07-04", 5)]
    )
    df = _frame_from_naive_index(idx)
    result = validate_p6.compute_tz_evidence(df)
    assert result["ok"] is True


def test_tz_evidence_fixed_offset_series_fails():
    # Fixed UTC-5 year-round: naive quiet hour is correct (17) only during the
    # EST calendar months; during EDT months it drifts to 16 -- exactly the
    # signature of a mislabeled fixed-offset feed, not genuine ET wall clock.
    idx = _synthetic_naive_index(
        [("2015-01-05", 5), ("2015-07-06", 5)],
        hour_shift=lambda d: 16 if 4 <= d.month <= 10 else 17,
    )
    df = _frame_from_naive_index(idx)
    with pytest.raises(ValueError):
        validate_p6.compute_tz_evidence(df)


# --- Defect 1: contamination-robust tz evidence + contamination_evidence --------

def _synthetic_naive_index_partial_contamination(start, n_days, contaminated_day_offsets):
    """Like ``_synthetic_naive_index`` but on ``contaminated_day_offsets`` days
    the 17:00 hour is NOT skipped (simulating vendor-spliced bars filling the
    daily maintenance break instead of leaving a genuine gap)."""
    idx = []
    cur = pd.Timestamp(start)
    for d in range(n_days):
        day = cur + pd.Timedelta(days=d)
        contaminated = d in contaminated_day_offsets
        for hour in range(24):
            if hour == 17 and not contaminated:
                continue
            for minute in range(0, 60, 5):
                idx.append(day + pd.Timedelta(hours=hour, minutes=minute))
    return pd.DatetimeIndex(sorted(idx))


def test_tz_evidence_robust_to_partial_contamination():
    clean_idx = _synthetic_naive_index([("2015-01-05", 5)])
    contam_idx = _synthetic_naive_index_partial_contamination(
        "2020-01-06", 5, contaminated_day_offsets={0, 1, 2}
    )
    idx = clean_idx.append(contam_idx)
    df = _frame_from_naive_index(idx)

    result = validate_p6.compute_tz_evidence(df)
    assert result["ok"] is True

    contamination = result["contamination_evidence"]["by_year"]
    assert contamination["2020"]["n_days_no_gap_contamination_candidate"] == 3
    assert contamination["2015"]["n_days_no_gap_contamination_candidate"] == 0

    bucket = result["quiet_hour_by_dst_regime_and_year"]["2020-EST"]
    assert bucket["reopen_evidence_ok"] is True          # proven from the 2 clean gap-days
    assert bucket["quiet_hour_check_applicable"] is False  # contaminated -> heuristic not relied on


def test_tz_evidence_fully_contaminated_bucket_hard_fails():
    # Every day contaminated -> no gap-day evidence exists at all for this
    # bucket; there is genuinely no proof of correct localization, so this
    # must still hard-fail (contamination-robustness must not silently
    # rubber-stamp a bucket with zero real evidence).
    idx = _synthetic_naive_index_partial_contamination(
        "2020-01-06", 5, contaminated_day_offsets={0, 1, 2, 3, 4}
    )
    df = _frame_from_naive_index(idx)
    with pytest.raises(ValueError):
        validate_p6.compute_tz_evidence(df)


def test_compute_contamination_evidence_zero_on_clean_series():
    idx = _synthetic_naive_index([("2015-01-05", 5), ("2016-01-04", 5)])
    df = _frame_from_naive_index(idx)
    result = validate_p6.compute_contamination_evidence(df)
    assert result["total_contamination_candidate_days"] == 0


# --- I3: session integrity (descriptive) ----------------------------------------

def test_session_integrity_complete_session():
    idx = pd.date_range("2023-01-03 09:30", periods=70, freq="1min", tz="US/Eastern")
    df = _ohlc_frame(idx, 100.0, 100.1, 99.9, 100.0)
    result = validate_p6.compute_session_integrity(df)
    assert result["per_session_completeness"]["n_sessions_total"] == 1
    assert result["per_session_completeness"]["n_incomplete"] == 0


def test_session_integrity_flags_incomplete_session():
    idx = pd.date_range("2023-01-03 09:30", periods=30, freq="1min", tz="US/Eastern")
    df = _ohlc_frame(idx, 100.0, 100.1, 99.9, 100.0)
    result = validate_p6.compute_session_integrity(df)
    assert result["per_session_completeness"]["n_incomplete"] == 1


# --- I1: cross-vendor gate -------------------------------------------------------

def _make_corr_pair(n_days=5, corr_break_day=None, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03 09:30", periods=n_days * 390, freq="1min", tz="US/Eastern")
    shocks = rng.normal(0, 0.001, len(idx))
    price_a = 100 + np.cumsum(shocks)
    noise_b = rng.normal(0, 0.00005, len(idx))
    price_b = price_a + noise_b + 2655.0  # mimics a back-adjustment style offset

    if corr_break_day is not None:
        day = idx.normalize()
        mask = day == pd.Timestamp(corr_break_day, tz="US/Eastern")
        indep_shocks = rng.normal(0, 0.001, int(mask.sum()))
        price_b = price_b.copy()
        price_b[mask] = 100 + np.cumsum(indep_shocks) + 2655.0

    df_a = _ohlc_frame(idx, price_a, price_a, price_a, price_a)
    df_b = _ohlc_frame(idx, price_b, price_b, price_b, price_b)
    return df_a, df_b


def test_cross_vendor_gate_passes_on_correlated_series():
    a, b = _make_corr_pair(n_days=5)
    result = validate_p6.cross_vendor_gate(a, b, start="2022-01-01", end="2022-01-10")
    assert result["lag0_exceeds_neighbors_clean_days"] is True
    assert result["failing_days"] == []
    assert result["n_phase1_tz_artifact_days"] == 0


def test_cross_vendor_gate_hard_fails_on_decorrelated_day():
    # a genuinely-corrupted day (independent random walk in one series) is
    # NOT lag-explainable and must still hard-fail
    a, b = _make_corr_pair(n_days=5, corr_break_day="2022-01-04")
    with pytest.raises(ValueError):
        validate_p6.cross_vendor_gate(a, b, start="2022-01-01", end="2022-01-10")


def _make_tz_artifact_pair(shift_day="2022-01-04", shift_minutes=60, seed=0):
    """Phase-1-style tz artifact: on one day, series b's bars are the SAME
    market data stamped +shift_minutes (a vendor DST bug), not garbage.
    Built from disjoint per-day RTH sessions (09:30-16:00) so a +60-minute
    shift never collides with a neighboring day's bars."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2022-01-03", periods=5, freq="B")
    parts = [pd.date_range(f"{d.date()} 09:30", f"{d.date()} 16:00", freq="1min",
                           tz="US/Eastern", inclusive="left") for d in days]
    idx = parts[0].append(parts[1:])
    shocks = rng.normal(0, 0.001, len(idx))
    price = 100 + np.cumsum(shocks)
    df_a = _ohlc_frame(idx, price, price, price, price)

    idx_b = idx.to_series().copy()
    mask = np.asarray(idx.normalize() == pd.Timestamp(shift_day, tz="US/Eastern"))
    idx_b[mask] = idx_b[mask] + pd.Timedelta(minutes=shift_minutes)
    df_b = _ohlc_frame(pd.DatetimeIndex(idx_b), price + 2655.0, price + 2655.0,
                       price + 2655.0, price + 2655.0)
    df_b = df_b[~df_b.index.duplicated(keep="first")].sort_index()
    return df_a, df_b


def test_cross_vendor_gate_lag_explains_phase1_tz_artifact_day():
    # a day where the OTHER vendor's stamps are +60min (the forensically
    # proven Phase-1 defect) is attributed, disclosed, and passes the gate
    a, b = _make_tz_artifact_pair()
    result = validate_p6.cross_vendor_gate(a, b, start="2022-01-01", end="2022-01-10")
    assert result["failing_days"] == []
    assert result["n_phase1_tz_artifact_days"] >= 1
    lags = {int(k) for k in result["phase1_tz_artifact_lag_histogram"]}
    assert lags <= set(range(-65, -54)) | set(range(55, 66))


def _make_intra_day_splice_pair(splice_day="2022-01-04", shift_minutes=60, seed=0):
    """Phase-1's nastier defect shape: only PART of a day is shifted (an
    intra-day splice, e.g. real 2023-05-01). First half aligned, second half
    of the splice day stamped +shift_minutes in series b."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2022-01-03", periods=5, freq="B")
    parts = [pd.date_range(f"{d.date()} 09:30", f"{d.date()} 16:00", freq="1min",
                           tz="US/Eastern", inclusive="left") for d in days]
    idx = parts[0].append(parts[1:])
    shocks = rng.normal(0, 0.001, len(idx))
    price = 100 + np.cumsum(shocks)
    df_a = _ohlc_frame(idx, price, price, price, price)

    idx_b = idx.to_series().copy()
    day = idx.normalize()
    after_split = idx.time >= pd.Timestamp("12:45").time()
    mask = np.asarray((day == pd.Timestamp(splice_day, tz="US/Eastern")) & after_split)
    idx_b[mask] = idx_b[mask] + pd.Timedelta(minutes=shift_minutes)
    df_b = _ohlc_frame(pd.DatetimeIndex(idx_b), price + 2655.0, price + 2655.0,
                       price + 2655.0, price + 2655.0)
    df_b = df_b[~df_b.index.duplicated(keep="first")].sort_index()
    return df_a, df_b


def test_cross_vendor_gate_explains_intra_day_splice():
    a, b = _make_intra_day_splice_pair()
    result = validate_p6.cross_vendor_gate(a, b, start="2022-01-01", end="2022-01-10")
    assert result["failing_days"] == []
    assert result["n_phase1_intra_day_splice_days"] >= 1


def _make_referee_scenario(bad_series: str, day="2022-01-04", seed=0):
    """Three-series scenario for rung-3 referee tests: indep-NQ (a), Phase-1
    NQ (b), and indep-ES (the referee, ~0.9-correlated with real NQ). On
    `day`, either b degrades mildly (Phase-1 fast-market infidelity: partial
    corruption, aligned corr ~0.5-0.8, unexplainable by any lag) or a is
    replaced with garbage (new-data corruption)."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2022-01-03", periods=5, freq="B")
    parts = [pd.date_range(f"{d.date()} 09:30", f"{d.date()} 16:00", freq="1min",
                           tz="US/Eastern", inclusive="left") for d in days]
    idx = parts[0].append(parts[1:])
    shocks = rng.normal(0, 0.001, len(idx))
    price = 100 + np.cumsum(shocks)
    # referee: real co-moving sibling (shares most of the shock + own noise)
    es_price = 50 + np.cumsum(0.8 * shocks + rng.normal(0, 0.0004, len(idx)))

    a_price = price.copy()
    b_price = price + 2655.0
    mask = np.asarray(idx.normalize() == pd.Timestamp(day, tz="US/Eastern"))
    if bad_series == "phase1":
        # infidelity: b keeps the trend but half its 1-min moves are wrong
        corrupt = np.where(rng.random(int(mask.sum())) < 0.5,
                           rng.normal(0, 0.001, int(mask.sum())), 0.0)
        b_price = b_price.copy()
        b_price[mask] = b_price[mask] + np.cumsum(corrupt)
    else:  # bad_series == "indep": a is garbage that day
        a_price = a_price.copy()
        a_price[mask] = 100 + np.cumsum(rng.normal(0, 0.001, int(mask.sum())))

    df_a = _ohlc_frame(idx, a_price, a_price, a_price, a_price)
    df_b = _ohlc_frame(idx, b_price, b_price, b_price, b_price)
    df_es = _ohlc_frame(idx, es_price, es_price, es_price, es_price)
    return df_a, df_b, df_es


def test_cross_vendor_gate_referee_adjudicates_phase1_infidelity():
    a, b, es = _make_referee_scenario("phase1")
    result = validate_p6.cross_vendor_gate(a, b, start="2022-01-01", end="2022-01-10",
                                           es_referee=es)
    assert result["failing_days"] == []
    assert result["n_phase1_infidelity_days_referee_adjudicated"] >= 1


def test_cross_vendor_gate_referee_still_fails_corrupted_new_data():
    a, b, es = _make_referee_scenario("indep")
    with pytest.raises(ValueError):
        validate_p6.cross_vendor_gate(a, b, start="2022-01-01", end="2022-01-10",
                                      es_referee=es)


def test_cross_vendor_gate_counts_not_evaluable_thin_days():
    # a Sunday-evening-style stub (no RTH bars) must be counted, not scored
    a, b = _make_corr_pair(n_days=5)
    stub_idx = pd.date_range("2022-01-09 18:00", periods=120, freq="1min", tz="US/Eastern")
    stub_a = _ohlc_frame(stub_idx, 100.0, 100.1, 99.9, 100.0)
    stub_b = _ohlc_frame(stub_idx, 2755.0, 2755.1, 2754.9, 2755.0)
    a2 = pd.concat([a, stub_a]).sort_index()
    b2 = pd.concat([b, stub_b]).sort_index()
    result = validate_p6.cross_vendor_gate(a2, b2, start="2022-01-01", end="2022-01-10")
    assert result["failing_days"] == []
    assert result["n_not_evaluable_days"] >= 1


# --- Defect 2: empirical bar-label shift evidence --------------------------------

def _make_offset_labeled_pair(true_shift_minutes, n_days=5, seed=0):
    """Build a Phase-1-like series ``b`` and an indep-vendor-like RAW
    (unshifted) series ``a`` such that ``a``'s literal vendor timestamps are
    ``true_shift_minutes`` EARLIER than ``b``'s for the same underlying bar --
    i.e. shifting ``a``'s index forward by ``true_shift_minutes`` recovers
    perfect alignment with ``b``."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-03 09:30", periods=n_days * 390, freq="1min", tz="US/Eastern")
    shocks = rng.normal(0, 0.001, len(idx))
    price = 100 + np.cumsum(shocks)
    b = pd.DataFrame({"close": price}, index=idx)
    a_unshifted = pd.DataFrame({"close": price}, index=idx - pd.Timedelta(minutes=true_shift_minutes))
    return a_unshifted, b


def test_bar_label_shift_evidence_recovers_known_decisive_shift():
    a_unshifted, b = _make_offset_labeled_pair(true_shift_minutes=1, n_days=5)
    result = validate_p6.compute_bar_label_shift_evidence(a_unshifted, b, start="2023-01-01", end="2023-01-10")

    assert result["chosen_shift_minutes"] == 1
    assert result["decisive"] is True
    assert result["per_shift_minutes"]["1"]["median_daily_corr"] >= validate_p6.BAR_LABEL_SHIFT_DECISIVE_FLOOR
    assert abs(result["per_shift_minutes"]["0"]["median_daily_corr"]) < validate_p6.BAR_LABEL_SHIFT_NEAR_ZERO
    assert abs(result["per_shift_minutes"]["-1"]["median_daily_corr"]) < validate_p6.BAR_LABEL_SHIFT_NEAR_ZERO


def test_bar_label_shift_evidence_not_decisive_when_no_shift_aligns():
    # Independent (uncorrelated) series at every candidate shift -> none of
    # the three should clear the decisive floor.
    rng = np.random.default_rng(3)
    idx = pd.date_range("2023-01-03 09:30", periods=5 * 390, freq="1min", tz="US/Eastern")
    a_unshifted = pd.DataFrame(
        {"close": 100 + np.cumsum(rng.normal(0, 0.001, len(idx)))}, index=idx
    )
    b = pd.DataFrame(
        {"close": 200 + np.cumsum(rng.normal(0, 0.001, len(idx)))}, index=idx
    )
    result = validate_p6.compute_bar_label_shift_evidence(a_unshifted, b, start="2023-01-01", end="2023-01-10")
    assert result["decisive"] is False


# --- Defect 1: post-drop containment verification ---------------------------------

def test_verify_maintenance_drop_containment_passes_when_boundary_is_clean():
    idx = pd.date_range("2020-01-03 09:00", periods=1000, freq="1min", tz="US/Eastern")
    price = 100.0 + 0.01 * np.arange(len(idx))
    df = _ohlc_frame(idx, price, price + 0.1, price - 0.1, price)
    result = validate_p6.verify_maintenance_drop_containment(
        df, df, "NQ", start="2020-01-01", end="2020-01-10"
    )
    assert result["n_boundary_anomaly_windows_after_drop"] == 0


def test_verify_maintenance_drop_containment_reports_but_does_not_hard_fail_residual_boundary_anomaly():
    # A residual anomaly AT the 18:00 reopen (post-drop) is NOT necessarily
    # contamination -- genuine reopen volatility is normal market behavior.
    # This must be reported, not hard-failed (only in-session drift hard-fails).
    n = 500
    idx = pd.date_range("2020-01-03 16:00", periods=n, freq="1min", tz="US/Eastern")
    rng = np.random.default_rng(2)
    base = 100 + np.cumsum(rng.normal(0, 0.01, n))
    high = base + 0.1
    low = base - 0.1
    bad_i = 65  # 17:05 -- past the rolling min_periods
    high = high.copy()
    low = low.copy()
    high[bad_i] = base[bad_i] + 25.0
    low[bad_i] = base[bad_i] - 25.0
    df = _ohlc_frame(idx, base, high, low, base)

    result = validate_p6.verify_maintenance_drop_containment(df, df, "NQ", start="2020-01-01", end="2020-01-10")
    assert result["n_boundary_anomaly_windows_after_drop"] >= 1


def test_verify_maintenance_drop_containment_hard_fails_on_in_session_drift():
    # Simulates a bug where the "cleaned" frame's in-session (09:30-10:30)
    # bars differ from raw -- the drop leaked outside the dropped hour.
    n = 500
    idx = pd.date_range("2020-01-03 09:00", periods=n, freq="1min", tz="US/Eastern")
    rng = np.random.default_rng(4)
    base = 100 + np.cumsum(rng.normal(0, 0.01, n))
    df_raw = _ohlc_frame(idx, base, base + 0.1, base - 0.1, base)

    bad_i = 65  # within 09:30-10:30
    assert pd.Timestamp("09:30").time() <= idx[bad_i].time() < pd.Timestamp("10:30").time()
    high = (base + 0.1).copy()
    low = (base - 0.1).copy()
    high[bad_i] = base[bad_i] + 25.0
    low[bad_i] = base[bad_i] - 25.0
    df_clean = _ohlc_frame(idx, base, high, low, base)  # "cleaned" frame has an extra in-session anomaly

    with pytest.raises(ValueError):
        validate_p6.verify_maintenance_drop_containment(
            df_raw, df_clean, "NQ", start="2020-01-01", end="2020-01-10"
        )
