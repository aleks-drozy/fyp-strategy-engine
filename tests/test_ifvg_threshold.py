"""Tests for the `fvg_threshold` parameter added to strategy.ifvg.compute_ifvg.

From the Phase-4 parameter-tuning spec
(docs/specs/2026-07-13-phase4-parameter-tuning-design.md).

Two synthetic in-session frames, each with a single bullish gap-then-invert
event at bar index 2 (gap) / bar index 3 (inversion, giving state "Bearish"
per strategy/ifvg.py's Bullish-FVG-inverted -> "Bearish" rule):
  - `_small_gap_frame()`: gap pct ~0.30%
  - `_large_gap_frame()`: gap pct ~5.00%

At `fvg_threshold=0.0` both gaps qualify (states change to "Bearish" once
inverted). At a threshold between the two pct's (1.0), only the large gap
still qualifies -- the small gap is filtered out entirely, so it's never
created and therefore never inverts, and bar 3 stays "None".
"""

import pandas as pd

from strategy.ifvg import compute_ifvg
from strategy.session import in_session_mask

SMALL_GAP_PCT = 0.30   # (100.30 - 100) / 100 * 100
LARGE_GAP_PCT = 5.00   # (105.00 - 100) / 100 * 100


def _frame(rows, day="2025-01-21", t0=" 09:32"):
    idx = pd.date_range(f"{day}{t0}", periods=len(rows), freq="1min", tz="US/Eastern")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1
    return df


def _small_gap_frame():
    # bar0.high=100 is the gap reference; bar2.low=100.30 -> bullish gap,
    # pct = 0.30%. bar3.close=95 < 100 -> inverts -> state "Bearish".
    rows = [
        (99, 100, 98, 99.5),        # 0
        (100, 100.2, 99.8, 100.0),  # 1  filler
        (100.3, 100.4, 100.3, 100.35),  # 2  gap: low=100.3 > high[0]=100
        (100.0, 100.0, 90.0, 95.0),     # 3  close=95 < 100 -> inverts
    ]
    return _frame(rows)


def _large_gap_frame():
    # bar0.high=100 is the gap reference; bar2.low=105.00 -> bullish gap,
    # pct = 5.00%. bar3.close=95 < 100 -> inverts -> state "Bearish".
    rows = [
        (99, 100, 98, 99.5),         # 0
        (100, 100.2, 99.8, 100.0),   # 1  filler
        (105.0, 105.2, 105.0, 105.1),  # 2  gap: low=105.0 > high[0]=100
        (105.0, 105.0, 90.0, 95.0),    # 3  close=95 < 100 -> inverts
    ]
    return _frame(rows)


def test_threshold_zero_admits_both_gaps():
    small = _small_gap_frame()
    large = _large_gap_frame()

    st_small = compute_ifvg(small, in_session_mask(small.index), fvg_threshold=0.0)
    st_large = compute_ifvg(large, in_session_mask(large.index), fvg_threshold=0.0)

    assert st_small.iloc[3] == "Bearish"
    assert st_large.iloc[3] == "Bearish"


def test_threshold_between_pcts_filters_small_gap_only():
    threshold = 1.0  # above SMALL_GAP_PCT (0.30), below LARGE_GAP_PCT (5.00)
    small = _small_gap_frame()
    large = _large_gap_frame()

    st_small = compute_ifvg(small, in_session_mask(small.index), fvg_threshold=threshold)
    st_large = compute_ifvg(large, in_session_mask(large.index), fvg_threshold=threshold)

    # small gap never qualifies -> never created -> never inverts -> stays None
    assert st_small.iloc[3] == "None"
    # large gap still qualifies and inverts as before
    assert st_large.iloc[3] == "Bearish"


def test_monotonicity_higher_threshold_admits_no_more_signals():
    df = _small_gap_frame()
    sess = in_session_mask(df.index)

    st_zero = compute_ifvg(df, sess, fvg_threshold=0.0)
    st_high = compute_ifvg(df, sess, fvg_threshold=1.0)

    n_zero = int((st_zero != "None").sum())
    n_high = int((st_high != "None").sum())
    assert n_high <= n_zero
    # sanity: this frame's gap is actually filtered at the higher threshold,
    # so the inequality is strict here (not a vacuous <=).
    assert n_high < n_zero
