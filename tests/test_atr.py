"""Tests for strategy.atr.compute_atr.

Task 1 Step 1 (docs/superpowers/plans/2026-07-13-phase5-exits-costs-volfilter.md).
Hand-computed TR/ATR on a tiny 5-bar frame (period=2, alpha=0.5, so the
Wilder recursion is easy to verify by hand); bar-0 fallback (no NaN); and a
no-lookahead check (truncating the df to fewer bars can't change ATR values
already computed on the earlier bars).
"""

import numpy as np
import pandas as pd
import pytest

from strategy.atr import compute_atr


def _frame(rows):
    """rows: list of (high, low, close)."""
    idx = pd.date_range("2025-01-21", periods=len(rows), freq="1min")
    df = pd.DataFrame(rows, columns=["high", "low", "close"], index=idx)
    df["open"] = df["close"]  # unused by compute_atr; present for realism
    return df


# H, L, C chosen so every bar's TR is driven by a different one of the 3 legs.
ROWS = [
    (10, 8, 9),    # 0: TR = H-L = 2                          (no prev close)
    (11, 9, 10),   # 1: TR = max(2, |11-9|=2, |9-9|=0) = 2
    (9, 7, 8),     # 2: TR = max(2, |9-10|=1, |7-10|=3) = 3
    (12, 11, 11),  # 3: TR = max(1, |12-8|=4, |11-8|=3) = 4
    (13, 12, 12),  # 4: TR = max(1, |13-11|=2, |12-11|=1) = 2
]

EXPECTED_TR = [2.0, 2.0, 3.0, 4.0, 2.0]

# Wilder RMA, period=2 -> alpha=0.5, seeded at TR[0]:
# ATR0=2; ATR1=.5*2+.5*2=2; ATR2=.5*3+.5*2=2.5; ATR3=.5*4+.5*2.5=3.25; ATR4=.5*2+.5*3.25=2.625
EXPECTED_ATR_PERIOD2 = [2.0, 2.0, 2.5, 3.25, 2.625]


def test_tr_and_atr_values_on_hand_frame():
    df = _frame(ROWS)

    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    assert tr.tolist() == EXPECTED_TR

    atr = compute_atr(df, period=2)
    assert atr.tolist() == pytest.approx(EXPECTED_ATR_PERIOD2, abs=1e-9)


def test_bar_zero_has_no_nan_and_falls_back_to_high_minus_low():
    df = _frame(ROWS)
    atr = compute_atr(df, period=14)
    assert not atr.isna().any()
    assert atr.iloc[0] == df["high"].iloc[0] - df["low"].iloc[0] == 2.0


def test_index_alignment_and_default_period():
    df = _frame(ROWS)
    atr = compute_atr(df)  # default period=14
    assert list(atr.index) == list(df.index)
    assert len(atr) == len(df)


def test_no_lookahead_future_bar_cannot_change_earlier_atr():
    rng = np.random.default_rng(0)
    n = 40
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.1, 2.0, n)
    low = close - rng.uniform(0.1, 2.0, n)
    df_full = _frame(list(zip(high, low, close)))

    atr_full = compute_atr(df_full, period=14)

    for cut in (5, 10, 25, n):
        atr_prefix = compute_atr(df_full.iloc[:cut], period=14)
        assert atr_prefix.tolist() == atr_full.iloc[:cut].tolist(), (
            f"truncating to {cut} bars changed an already-computed ATR value -- lookahead bug"
        )
