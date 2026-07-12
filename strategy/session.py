"""NY trading-session mask. Strategy window = 09:30-10:30 America/New_York, weekdays,
left-inclusive/right-exclusive. From FYP_BOT_1_3.pine lines 13-33."""
import pandas as pd


def in_session_mask(index: pd.DatetimeIndex, start: str = "09:30", end: str = "10:30") -> pd.Series:
    if index.tz is None:
        raise ValueError("index must be tz-aware (US/Eastern)")
    ny = index.tz_convert("America/New_York")
    sh, sm = (int(x) for x in start.split(":"))
    eh, em = (int(x) for x in end.split(":"))
    mins = ny.hour * 60 + ny.minute
    in_win = (mins >= sh * 60 + sm) & (mins < eh * 60 + em)
    is_weekday = ny.dayofweek < 5  # Mon..Fri
    return pd.Series(in_win & is_weekday, index=index, name="in_session")
