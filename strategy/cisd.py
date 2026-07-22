"""
CISD (Change in State of Delivery) computation.

Ported from Trading_Dashboard-master/backend/strategy/cisd.py, logic extracted
verbatim from docs/reference/FYP_BOT_1_3.pine -- EXCEPT for a corrected
off-by-one neighbor index (Phase 2, Blocker 4), documented below.

CISD tracks market structure flips:
- Bearish structure break creates a Bull CISD level (potentialTopPrice / open of pullback start)
- Bullish structure break creates a Bear CISD level (potentialBottomPrice / open of pullback start)
- currentState (bool): True=Bullish when close crosses above bearCisdLevel; False=Bearish when below bullCisdLevel

State output: "Bullish" when currentState is True, "Bearish" when False.

Pine source reference: lines 37-289 (CISD section) and lines 267-288 (state logic).

CORRECTED NEIGHBOR INDEX (Blocker 4):
Pine line 195 (bullish-break/max block) uses
    math.max(high[bar_index-bullishBreakIndex], high[bar_index-bullishBreakIndex+1])
In Pine "bars-ago" notation, high[N] means the bar N bars before bar_index, i.e.
actual array index = bar_index - N. With offset = bar_index - bullishBreakIndex:
    high[bar_index-bullishBreakIndex]   -> index bar_index-offset       == bullishBreakIndex
    high[bar_index-bullishBreakIndex+1] -> index bar_index-(offset+1)   == bullishBreakIndex-1
So the second neighbor is ONE BAR EARLIER (breakIdx-1), not one bar later. The
original port computed `highs[i-offset+1]` (== highs[breakIdx+1], one bar LATER,
the wrong direction). This has been corrected to `highs[i-offset-1]`
(== highs[breakIdx-1]), guarded with `>= 0`. The bearish-break/min block
(Pine line 217) is mirrored the same way: `lows[i-offset-1]`, guarded `>= 0`.
This struct_top/struct_bottom feeds every downstream structure break -> CISD
level -> currentState -> signal, so getting the neighbor bar right matters.
See tests/test_cisd.py for the characterization test pinning this branch.
"""

import pandas as pd
import numpy as np

import math


def compute_cisd(df: pd.DataFrame) -> pd.Series:
    """
    Compute CISD state for each bar.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with columns: open, high, low, close, volume.
        Index should be timestamps. Rows are in chronological order.

    Returns
    -------
    pd.Series
        String series with values: "Bullish" or "Bearish".
        Index matches df.index.
    """
    n = len(df)
    opens = df["open"].values.astype(float)
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)

    # --- Variable declarations (Pine var keyword = persistent across bars) ---
    # MarketStructure
    struct_top = 0.0
    struct_bottom = 0.0
    struct_is_bullish = False

    # CISD level arrays (Pine: array<cisd>)
    # We keep only the most recent active level per direction (keepLevels=false default)
    # cisd_levels_bu: list of {price, completed}
    # cisd_levels_be: list of {price, completed}
    cisd_levels_bu: list[dict] = []
    cisd_levels_be: list[dict] = []

    is_bullish_pullback = False
    is_bearish_pullback = False
    potential_top = float("nan")
    potential_bottom = float("nan")
    bullish_break_idx = -1   # bar_index where bearish pullback started (bullishBreakIndex)
    bearish_break_idx = -1   # bar_index where bullish pullback started (bearishBreakIndex)

    # CISD state variables
    bull_cisd_level = float("nan")
    bear_cisd_level = float("nan")
    current_state = False  # False = Bearish

    # Output
    states = np.full(n, "Bearish", dtype=object)

    for i in range(1, n):
        # --- Pullback detection (Pine lines 149-164) ---
        # bearishPullbackDetected = close[1] > open[1]  (prior bar was bullish)
        bearish_pullback_detected = closes[i - 1] > opens[i - 1]
        # bullishPullbackDetected = close[1] < open[1]  (prior bar was bearish)
        bullish_pullback_detected = closes[i - 1] < opens[i - 1]

        if bearish_pullback_detected and not is_bearish_pullback:
            is_bearish_pullback = True
            potential_top = opens[i - 1]
            bullish_break_idx = i - 1

        if bullish_pullback_detected and not is_bullish_pullback:
            is_bullish_pullback = True
            potential_bottom = opens[i - 1]
            bearish_break_idx = i - 1

        # --- Update potential levels (Pine lines 169-185) ---
        if is_bullish_pullback:
            if opens[i] < potential_bottom:
                potential_bottom = opens[i]
                bearish_break_idx = i
            if (closes[i] < opens[i]) and (opens[i] > potential_bottom):
                potential_bottom = opens[i]
                bearish_break_idx = i

        if is_bearish_pullback:
            if opens[i] > potential_top:
                potential_top = opens[i]
                bullish_break_idx = i
            if (closes[i] > opens[i]) and opens[i] < potential_top:
                potential_top = opens[i]
                bullish_break_idx = i

        # --- Structure update: Bearish break creates Bull CISD (Pine lines 190-207) ---
        if lows[i] < struct_bottom:
            struct_bottom = lows[i]
            struct_is_bullish = False

            if is_bearish_pullback and (i - bullish_break_idx) != 0:
                # Corrected neighbor index (Blocker 4): max of high[breakIdx] and
                # high[breakIdx-1] -- was erroneously high[breakIdx+1] in the original port.
                offset = i - bullish_break_idx
                h1 = highs[i - offset]
                if (i - offset - 1) >= 0:
                    struct_top = max(h1, highs[i - offset - 1])
                else:
                    struct_top = h1
                is_bearish_pullback = False
                # Create bullish CISD level at potentialTopPrice
                b = {"price": potential_top, "completed": False}
                cisd_levels_be.append(b)
            elif closes[i - 1] > opens[i - 1] and closes[i] < opens[i]:
                struct_top = highs[i - 1]
                is_bearish_pullback = False
                b = {"price": potential_top, "completed": False}
                cisd_levels_be.append(b)

        # --- Structure update: Bullish break creates Bear CISD (Pine lines 212-229) ---
        if highs[i] > struct_top:
            struct_is_bullish = True
            struct_top = highs[i]

            if is_bullish_pullback and (i - bearish_break_idx) != 0:
                # Mirrored fix (Blocker 4): min of low[breakIdx] and low[breakIdx-1]
                # -- was erroneously low[breakIdx+1] in the original port.
                offset = i - bearish_break_idx
                l1 = lows[i - offset]
                if (i - offset - 1) >= 0:
                    struct_bottom = min(l1, lows[i - offset - 1])
                else:
                    struct_bottom = l1
                is_bullish_pullback = False
                bu = {"price": potential_bottom, "completed": False}
                cisd_levels_bu.append(bu)
            elif closes[i - 1] < opens[i - 1] and closes[i] > opens[i]:
                struct_bottom = lows[i - 1]
                is_bullish_pullback = False
                bu = {"price": potential_bottom, "completed": False}
                cisd_levels_bu.append(bu)

        # --- Trim to 1 active level (keepLevels=false, Pine lines 233-243) ---
        while len(cisd_levels_bu) > 1:
            cisd_levels_bu.pop(0)
        while len(cisd_levels_be) > 1:
            cisd_levels_be.pop(0)

        # --- Completion checks (Pine lines 247-263) ---
        if cisd_levels_bu:
            latest = cisd_levels_bu[0]
            if closes[i] < latest["price"] and not latest["completed"] and closes[i - 1] > latest["price"]:
                latest["completed"] = True

        if cisd_levels_be:
            latest = cisd_levels_be[0]
            if closes[i] > latest["price"] and not latest["completed"] and closes[i - 1] < latest["price"]:
                latest["completed"] = True

        # --- Extract CISD level prices (Pine lines 273-276) ---
        if cisd_levels_bu:
            bull_cisd_level = cisd_levels_bu[0]["price"]
        else:
            bull_cisd_level = float("nan")

        if cisd_levels_be:
            bear_cisd_level = cisd_levels_be[0]["price"]
        else:
            bear_cisd_level = float("nan")

        # --- State crossover (Pine lines 280-288) ---
        # bullCross = not na(bearCisdLevel) and close > bearCisdLevel and close[1] <= bearCisdLevel
        bull_cross = (
            not math.isnan(bear_cisd_level)
            and closes[i] > bear_cisd_level
            and closes[i - 1] <= bear_cisd_level
        )
        # bearCross = not na(bullCisdLevel) and close < bullCisdLevel and close[1] >= bullCisdLevel
        bear_cross = (
            not math.isnan(bull_cisd_level)
            and closes[i] < bull_cisd_level
            and closes[i - 1] >= bull_cisd_level
        )

        if bull_cross:
            current_state = True
        elif bear_cross:
            current_state = False

        states[i] = "Bullish" if current_state else "Bearish"

    return pd.Series(states, index=df.index, name="cisd_state", dtype=object)
