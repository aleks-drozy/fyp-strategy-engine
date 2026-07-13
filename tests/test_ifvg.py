import pandas as pd
from strategy.ifvg import compute_ifvg
from strategy.session import in_session_mask


def _frame(rows, day="2025-01-21", t0=" 09:32"):
    # rows: list of (open,high,low,close); consecutive 1-min bars
    idx = pd.date_range(f"{day}{t0}", periods=len(rows), freq="1min", tz="US/Eastern")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1
    return df


def test_bullish_fvg_forms_and_inverts_in_session():
    # bar2 low(105) > bar0 high(100) => bullish FVG {top=105, bottom=100};
    # later a close below 100 inverts it -> ifvgState "Bearish"
    rows = [(99, 100, 98, 99.5), (101, 104, 100, 103), (106, 108, 105, 107),
            (104, 104, 101, 102), (101, 101, 99, 99.0)]
    df = _frame(rows)
    st = compute_ifvg(df, in_session_mask(df.index))
    assert st.iloc[2] in ("None", "Bullish")   # FVG created, not yet inverted
    assert st.iloc[4] == "Bearish"             # inverted bullish FVG -> Bearish signal


def test_no_fvg_created_out_of_session():
    df = _frame([(99, 100, 98, 99.5), (101, 104, 100, 103), (106, 108, 105, 107)],
                t0=" 08:00")  # pre-session
    st = compute_ifvg(df, in_session_mask(df.index))
    assert set(st.unique()) <= {"None"}
