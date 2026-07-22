"""Average True Range (ATR), Wilder's RMA smoothing.

From the Phase-5 exits/costs/vol-filter spec
(docs/specs/2026-07-13-phase5-exits-costs-volfilter-design.md). Used by the
Phase-5 volatility filter (ATR% = ATR14 / close * 100, wired up in the
walk-forward) and threaded through `run_execution` as
an optional pre-sliced `atr` array for the entry-side vol gate.

TR[i] = max(high[i]-low[i], |high[i]-close[i-1]|, |low[i]-close[i-1]|)

ATR here is Wilder's RMA (alpha = 1/period) of TR, NOT a plain SMA -- this
is the industry-standard "ATR" (Wilder, 1978) and matches TradingView's
ta.atr(). Causal: TR[i] and ATR[i] only ever read bars <= i (pandas
`.shift(1)` for the previous close and `.ewm(adjust=False)` for the
recursive smoothing -- both strictly backward-looking), so a later bar can
never change an earlier ATR value.

Bar 0 has no previous close: TR[0] falls back to `high[0]-low[0]` (the
other two legs of the max are NaN and dropped by `.max(axis=1)`'s default
skipna). ATR is then seeded from TR[0] and recurses forward -- the same
seed-from-bar-0 convention as strategy/ema.py's compute_ema -- so the
returned Series has no NaN values (no leading-NaN warm-up run to special
case downstream).
"""

import pandas as pd


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute ATR (Average True Range) via Wilder's RMA.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame with 'high', 'low', 'close' columns.
    period : int
        Wilder smoothing length. Default 14 (the conventional ATR period).

    Returns
    -------
    pd.Series
        ATR values, aligned to df.index, no NaN (see module docstring for
        the bar-0 seeding convention).
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)  # skipna=True (default): bar 0's two prev-close legs are
    # NaN and dropped, leaving high[0]-low[0] -- see module docstring.

    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    atr.name = f"atr_{period}"
    return atr
