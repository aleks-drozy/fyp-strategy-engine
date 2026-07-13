import pandas as pd
from strategy.cisd import compute_cisd


def _frame(rows, day="2025-01-21", t0=" 09:32"):
    # rows: list of (open, high, low, close); consecutive 1-min bars
    idx = pd.date_range(f"{day}{t0}", periods=len(rows), freq="1min", tz="US/Eastern")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1
    return df


def test_smoke_output_shape_and_values():
    # ~30 synthetic bars: an up leg, a pullback, another up leg, a longer down leg.
    rows = []
    price = 100.0
    for direction, n, step in [(+1, 3, 1.0), (-1, 2, 1.0), (+1, 5, 1.0), (-1, 8, 2.0), (+1, 6, 3.0), (-1, 6, 1.5)]:
        for _ in range(n):
            o = price
            c = price + direction * step
            hi = max(o, c) + 0.3
            lo = min(o, c) - 0.3
            rows.append((o, hi, lo, c))
            price = c
    df = _frame(rows)
    st = compute_cisd(df)
    assert len(st) == len(df)
    assert st.isna().sum() == 0
    assert set(st.unique()) <= {"Bullish", "Bearish"}


def test_corrected_neighbor_index_pullback_break_branch():
    """Characterization test locking the corrected pullback-break branch (Blocker 4).

    This 9-bar sequence drives a bearish pullback (is_bearish_pullback, tracked via
    bullish_break_idx) that resolves at bar 5, where lows[5] breaks below
    struct_bottom while bullish_break_idx=4 (offset=1). The bullish-break/max block
    then computes struct_top = max(h1, h2) with h1 = highs[bullish_break_idx] =
    highs[4] = 98.59:
      - corrected h2 = highs[breakIdx-1] = highs[3] = 99.01  -> struct_top = 99.01
      - original buggy h2 = highs[breakIdx+1] = highs[5] = 96.79 -> struct_top = 98.59
    Bar 7's high is 98.97 -- between the two thresholds. Under the buggy 98.59
    threshold it re-triggers an early "highs > struct_top" structure break, which
    updates the active bull CISD level from 96.57 down to 95.57. Under the corrected
    99.01 threshold, 98.97 does not clear it, so the bull CISD level stays at 96.57.
    That different level then flips whether bar 8's close (96.38) counts as crossing
    below it: it does versus the corrected 96.57 level (-> "Bearish"), but not versus
    the buggy 95.57 level (-> stays "Bullish"). Hence the pinned output differs at
    index 8 ("Bearish" here vs "Bullish" under the bug).

    A revert to breakIdx+1 changes this pinned output (index 8 becomes "Bullish") and
    fails this test -- that is the point: it locks the corrected branch against
    regression.
    """
    rows = [
        (100.0, 100.86, 97.81, 97.92),
        (97.92, 98.1, 96.48, 96.57),
        (96.57, 97.41, 96.05, 97.08),
        (97.08, 99.01, 96.37, 98.54),
        (98.54, 98.59, 96.42, 96.62),
        (96.62, 96.79, 95.46, 95.57),
        (95.57, 98.08, 95.06, 98.0),
        (98.0, 98.97, 97.81, 98.71),
        (98.71, 98.81, 95.8, 96.38),
    ]
    df = _frame(rows)
    st = compute_cisd(df)
    expected = ["Bearish", "Bearish", "Bearish", "Bearish", "Bearish",
                "Bearish", "Bearish", "Bullish", "Bearish"]
    assert st.tolist() == expected
