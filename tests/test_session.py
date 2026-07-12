import pandas as pd
from strategy.session import in_session_mask


def _idx(times):  # times: list of "YYYY-MM-DD HH:MM"
    return pd.DatetimeIndex(pd.to_datetime(times)).tz_localize("US/Eastern")


def test_bounds_and_weekday():
    idx = _idx(["2025-01-21 09:29", "2025-01-21 09:30", "2025-01-21 10:29",
                "2025-01-21 10:30", "2025-01-25 09:45"])  # last is a Saturday
    m = in_session_mask(idx).tolist()
    assert m == [False, True, True, False, False]


def test_requires_tz():
    import pytest
    with pytest.raises(ValueError):
        in_session_mask(pd.DatetimeIndex(pd.to_datetime(["2025-01-21 09:30"])))
