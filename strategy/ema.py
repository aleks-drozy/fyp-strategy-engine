"""
EMA (Exponential Moving Average) computation.

Ported unchanged from Trading_Dashboard-master/backend/strategy/ema.py.

Logic extracted from docs/reference/FYP_BOT_1_3.pine line 499:
    ema = ta.ema(emaSource, emaLength)   # emaLength = 20, emaSource = close

pandas ewm(adjust=False) matches TradingView's recursive EMA formula exactly:
    EMA[t] = alpha * close[t] + (1 - alpha) * EMA[t-1]
    alpha = 2 / (length + 1)

Do NOT use:
- TA-Lib (C binary, unreliable on Render)
- rolling().mean() (that is SMA)
- pandas ewm(adjust=True)
"""

import pandas as pd


def compute_ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Compute EMA of close prices.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with 'close' column.
    period : int
        EMA period. Default 20 (matches Pine source emaLength=20).

    Returns
    -------
    pd.Series
        EMA values. Index matches df.index.
        ewm(adjust=False) seeds from bar 0, so there are no NaN values (unlike a
        rolling window). The seed-warmup difference vs Pine's ta.ema (which also
        seeds from the first bar) is negligible after the first trading session.
    """
    result = df["close"].ewm(span=period, adjust=False).mean()
    result.name = "ema_20"
    return result
