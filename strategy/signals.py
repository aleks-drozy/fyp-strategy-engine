"""Double-confirmation entry trigger. From FYP_BOT_1_3.pine lines 430-457, 508-509.
Fires on the bar where CISD flips, if the IFVG is (already or newly) on the same side."""
import pandas as pd


def double_confirmation(ifvg: pd.Series, cisd: pd.Series) -> pd.Series:
    ifvg = list(ifvg)
    cisd = list(cisd)
    n = len(ifvg)
    cisd_bull = [c == "Bullish" for c in cisd]
    out = [""] * n
    for i in range(1, n):
        cisd_turned_bull = (not cisd_bull[i - 1]) and cisd_bull[i]
        cisd_turned_bear = cisd_bull[i - 1] and (not cisd_bull[i])
        ifvg_turned_bull = (ifvg[i - 1] != "Bullish") and (ifvg[i] == "Bullish")
        ifvg_turned_bear = (ifvg[i - 1] != "Bearish") and (ifvg[i] == "Bearish")
        bull_double = (
            (ifvg[i - 1] == "Bullish" and ifvg[i] == "Bullish" and cisd_turned_bull)
            or (ifvg_turned_bull and cisd_turned_bull)
        )
        bear_double = (
            (ifvg[i - 1] == "Bearish" and ifvg[i] == "Bearish" and cisd_turned_bear)
            or (ifvg_turned_bear and cisd_turned_bear)
        )
        if bull_double:
            out[i] = "Long"
        elif bear_double:
            out[i] = "Short"
    return pd.Series(out, name="signal")
