"""
Swing pivot detection — faithful Python translation of Pine Script leg() and
getCurrentStructure() functions from LuxAlgo Smart Money Concepts.

Pine Script logic at bar i:
    newLegHigh = high[size] > ta.highest(size)
              → high[i-size] > max(high[i-size+1 .. i])
    newLegLow  = low[size]  < ta.lowest(size)
              → low[i-size]  < min(low[i-size+1 .. i])

A pivot HIGH at bar p is confirmed at bar p+size when no subsequent bar
within [p+1, p+size] has a higher high. Same logic inverted for pivot LOW.

The leg state machine (BEARISH_LEG / BULLISH_LEG) prevents recording two
consecutive highs or two consecutive lows without an intervening opposite,
matching Pine's `var leg` behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BEARISH_LEG: int = 0   # leg value after a pivot HIGH is confirmed
BULLISH_LEG: int = 1   # leg value after a pivot LOW is confirmed

BULLISH: int = +1
BEARISH: int = -1


@dataclass(frozen=True)
class PivotPoint:
    """A confirmed swing high or swing low."""
    bar_index: int       # position in OHLCV array where the pivot OCCURRED
    level: float         # price of the pivot (high.iloc[bar_index] or low.iloc[bar_index])
    is_high: bool        # True = swing high, False = swing low
    confirmed_at: int    # bar index at which the pivot was confirmed (bar_index + size)


def detect_pivots(
    highs: pd.Series,
    lows: pd.Series,
    size: int,
) -> list[PivotPoint]:
    """
    Detect swing pivot highs and lows using Pine Script's right-side confirmation.

    Args:
        highs:  Series of bar highs.
        lows:   Series of bar lows (same length and index).
        size:   Number of right-side bars required to confirm a pivot.
                Pine: swingsLengthInput (swing=50, internal=5).

    Returns:
        Chronologically ordered list of PivotPoint objects.
    """
    h = highs.to_numpy(dtype=np.float64)
    lo = lows.to_numpy(dtype=np.float64)
    n = len(h)

    leg: int = BEARISH_LEG  # Pine: var leg = 0
    pivots: list[PivotPoint] = []

    for i in range(size, n):
        pivot_idx = i - size

        # Pine: ta.highest(size) = max of bars [i-size+1 .. i]  (size bars AFTER the pivot)
        max_right = float(np.max(h[pivot_idx + 1 : i + 1]))
        min_right = float(np.min(lo[pivot_idx + 1 : i + 1]))

        new_leg_high = h[pivot_idx] > max_right   # pivot high confirmed
        new_leg_low  = lo[pivot_idx] < min_right  # pivot low confirmed

        prev_leg = leg
        if new_leg_high:
            leg = BEARISH_LEG
        elif new_leg_low:
            leg = BULLISH_LEG
        # else: carry forward (Pine: leg unchanged)

        if leg == prev_leg:
            continue  # no state change → no new pivot recorded

        if leg == BEARISH_LEG:
            # startOfBearishLeg: leg BULLISH→BEARISH → pivot HIGH confirmed
            pivots.append(PivotPoint(
                bar_index=pivot_idx,
                level=float(h[pivot_idx]),
                is_high=True,
                confirmed_at=i,
            ))
        else:
            # startOfBullishLeg: leg BEARISH→BULLISH → pivot LOW confirmed
            pivots.append(PivotPoint(
                bar_index=pivot_idx,
                level=float(lo[pivot_idx]),
                is_high=False,
                confirmed_at=i,
            ))

    return pivots
