"""
IFVG (Inverted Fair Value Gap) computation.

Ported from Trading_Dashboard-master/backend/strategy/ifvg.py, logic extracted
verbatim from docs/reference/FYP_BOT_1_3.pine.

Key rules from Pine source:
- Bullish FVG forms when: low > high[2]  (gap between bar-2 high and current low)
- Bearish FVG forms when: high < low[2]  (gap between current high and bar-2 low)
- Inversion: endMethod="Close" (default)
    Bullish FVG inverted when: close < fvg.bottom  (close < high[2] at formation)
    Bearish FVG inverted when: close > fvg.top     (close > low[2] at formation)
- IFVG state: most recent inverted FVG wins.
    Inverted bullish FVG  -> ifvgState = "Bearish"
    Inverted bearish FVG  -> ifvgState = "Bullish"
- Expiry: bar_index - fvg.invertBar > ifvgLookback (10) -> "Expired"
- Session reset: ifvgState resets to "None" when entering a new trading session.
  Since fixture CSVs encode the Pine output directly, we replicate the reset by
  treating ifvgState as "None" at the start of each day (same as Pine's daily clear
  of fvgArray via array.clear on new calendar day).

Added vs the original port (Phase-2, Task 1 Step 5): an explicit `in_session` gate,
mirroring Pine lines 320/335 (FVG creation gated on inTradingSession) and lines
425-426 (state reset outside the session). FVG creation is now gated on session,
and the emitted state is forced to "None" for any out-of-session bar.
"""

import pandas as pd
import numpy as np
from typing import Literal

IFVGState = Literal["Bullish", "Bearish", "None", "Expired"]

IFVG_LOOKBACK = 10  # bars after inversion before expiry


def compute_ifvg(df: pd.DataFrame, in_session: pd.Series) -> pd.Series:
    """
    Compute IFVG state for each bar.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with columns: open, high, low, close, volume.
        Index should be timestamps. Rows are in chronological order.
    in_session : pd.Series
        Boolean mask (same index as df) — True where the bar is inside the
        strategy's trading session. FVG creation is gated on this, and the
        emitted state is forced to "None" outside it.

    Returns
    -------
    pd.Series
        String series with values: "Bullish", "Bearish", "None", "Expired".
        Index matches df.index.
    """
    n = len(df)
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    session = in_session.to_numpy(dtype=bool)

    # State output
    states = np.full(n, "None", dtype=object)

    # FVG tracking: list of dicts (most-recent first within each day)
    # Each entry: {top, bottom, is_bullish, start_bar, is_inverted, invert_bar}
    fvg_array: list[dict] = []

    # Daily reset tracking (Pine: array.clear(fvgArray) on new calendar day)
    last_trade_day = -1
    ifvg_state = "None"

    dates = df.index

    for i in range(2, n):
        # --- Daily reset (mirrors Pine session reset + array.clear) ---
        current_day = dates[i].date() if hasattr(dates[i], "date") else i
        if current_day != last_trade_day:
            fvg_array = []
            last_trade_day = current_day
            ifvg_state = "None"

        # --- FVG detection (Pine lines 307-316), gated on session (Pine lines 320/335) ---
        # Bullish gap: low[i] > high[i-2]
        bullish_gap = lows[i] > highs[i - 2]
        if bullish_gap and session[i]:
            fvg_array.insert(0, {
                "top": lows[i],
                "bottom": highs[i - 2],
                "is_bullish": True,
                "start_bar": i,
                "is_inverted": False,
                "invert_bar": None,
            })

        # Bearish gap: high[i] < low[i-2]
        bearish_gap = highs[i] < lows[i - 2]
        if bearish_gap and session[i]:
            fvg_array.insert(0, {
                "top": lows[i - 2],
                "bottom": highs[i],
                "is_bullish": False,
                "start_bar": i,
                "is_inverted": False,
                "invert_bar": None,
            })

        # --- Inversion check (Pine lines 350-369) ---
        for fvg in fvg_array:
            if not fvg["is_inverted"]:
                if fvg["is_bullish"]:
                    # Bullish FVG breached: close < fvg.bottom
                    if closes[i] < fvg["bottom"]:
                        fvg["is_inverted"] = True
                        fvg["invert_bar"] = i
                else:
                    # Bearish FVG breached: close > fvg.top
                    if closes[i] > fvg["top"]:
                        fvg["is_inverted"] = True
                        fvg["invert_bar"] = i

        # --- IFVG state update (Pine lines 416-427) ---
        # Find most recent inverted FVG (array is most-recent first)
        ifvg_state = "None"
        for fvg in fvg_array:
            if fvg["is_inverted"]:
                # Expiry check (Pine lines 472-484)
                bars_since_inversion = i - fvg["invert_bar"]
                if bars_since_inversion <= IFVG_LOOKBACK:
                    # Inverted bullish FVG -> "Bearish"; inverted bearish FVG -> "Bullish"
                    ifvg_state = "Bearish" if fvg["is_bullish"] else "Bullish"
                else:
                    ifvg_state = "Expired"
                break  # Only most recent inverted FVG matters

        # --- Session gate on emitted state (Pine lines 425-426) ---
        if not session[i]:
            ifvg_state = "None"

        states[i] = ifvg_state

    return pd.Series(states, index=df.index, name="ifvg_state", dtype=object)
