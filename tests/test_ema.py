import pandas as pd
from strategy.ema import compute_ema


def _frame(closes, day="2025-01-21", t0=" 09:32"):
    idx = pd.date_range(f"{day}{t0}", periods=len(closes), freq="1min", tz="US/Eastern")
    df = pd.DataFrame({"close": closes}, index=idx)
    return df


def test_compute_ema_matches_ewm_adjust_false():
    closes = [100.0, 101.5, 99.0, 102.25, 103.0, 101.75, 104.5, 100.0, 99.5, 105.0]
    df = _frame(closes)
    result = compute_ema(df, period=20)
    expected = df["close"].ewm(span=20, adjust=False).mean()
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_no_nan_values():
    # ewm(adjust=False) seeds from bar 0, so there should be no NaN values even
    # though fewer than `period` bars are supplied.
    closes = [100.0, 101.0, 102.0]
    df = _frame(closes)
    result = compute_ema(df, period=20)
    assert result.isna().sum() == 0
    assert len(result) == len(df)
