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
    assert result["lag0_exceeds_neighbors"] is True
    assert result["failing_days"] == []


def test_cross_vendor_gate_hard_fails_on_decorrelated_day():
    a, b = _make_corr_pair(n_days=5, corr_break_day="2022-01-04")
    with pytest.raises(ValueError):
        validate_p6.cross_vendor_gate(a, b, start="2022-01-01", end="2022-01-10")
