"""
Market structure detection: BOS and CHoCH.

Translates Pine Script displayStructure() function.

BOS   (Break of Structure)   = break in the direction of the current trend (continuation)
CHoCH (Change of Character)  = break AGAINST the current trend (reversal signal)

Pine crossover/crossunder logic:
    ta.crossover(close, level)  → close[1] < level and close[0] >= level
    ta.crossunder(close, level) → close[1] > level and close[0] <= level
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd

from zeus.strategy.smc.pivot import PivotPoint, BULLISH, BEARISH


class StructureType(IntEnum):
    BOS   = 1  # Break of Structure  — trend continuation
    CHOCH = 2  # Change of Character — trend reversal


@dataclass(frozen=True)
class StructureEvent:
    """A single BOS or CHoCH event detected at a specific bar."""
    bar_index: int
    structure_type: StructureType
    direction: int       # BULLISH (+1) or BEARISH (-1)
    level: float         # price level that was broken
    pivot_bar: int       # bar index of the pivot whose level was broken
    is_internal: bool    # True = 5-bar internal structure; False = swing structure


def detect_structure(
    closes: pd.Series,
    pivots: list[PivotPoint],
    is_internal: bool = False,
) -> list[StructureEvent]:
    """
    Detect BOS and CHoCH events from confirmed pivots and close prices.

    Processes bars in chronological order, tracking:
    - The last unbroken swing high (for bullish breaks)
    - The last unbroken swing low (for bearish breaks)
    - The current trend bias (BULLISH / BEARISH)

    Pivots are registered as they are confirmed (at confirmed_at bar), then
    close prices are checked for crossover/crossunder, matching Pine's execution
    order: getCurrentStructure() → displayStructure().

    Args:
        closes:      Series of close prices.
        pivots:      PivotPoint list from detect_pivots().
        is_internal: Whether these are internal (5-bar) pivots.

    Returns:
        Chronologically ordered list of StructureEvent.
    """
    c  = closes.to_numpy(dtype=np.float64)
    n  = len(c)

    # Index pivots by confirmation bar for O(1) lookup
    by_confirmation: dict[int, list[PivotPoint]] = {}
    for p in pivots:
        by_confirmation.setdefault(p.confirmed_at, []).append(p)

    events: list[StructureEvent] = []

    # Active pivot trackers (most recently confirmed, not yet broken)
    active_high_level   = np.nan
    active_high_bar     = -1
    active_high_crossed = True   # True means "already used / no pending level"

    active_low_level    = np.nan
    active_low_bar      = -1
    active_low_crossed  = True

    trend = 0  # 0=undefined, BULLISH=+1, BEARISH=-1

    for i in range(n):
        # Register pivots confirmed at this bar BEFORE checking for breaks
        # (matches Pine: getCurrentStructure runs before displayStructure)
        for p in by_confirmation.get(i, []):
            if p.is_high:
                active_high_level   = p.level
                active_high_bar     = p.bar_index
                active_high_crossed = False
            else:
                active_low_level    = p.level
                active_low_bar      = p.bar_index
                active_low_crossed  = False

        if i == 0:
            continue  # need prev close for crossover

        prev_c = c[i - 1]
        curr_c = c[i]

        # Bullish break: close crosses ABOVE the active swing high (ta.crossover)
        if (
            not active_high_crossed
            and not np.isnan(active_high_level)
            and prev_c < active_high_level
            and curr_c >= active_high_level
        ):
            stype = StructureType.CHOCH if trend == BEARISH else StructureType.BOS
            active_high_crossed = True
            trend = BULLISH
            events.append(StructureEvent(
                bar_index=i,
                structure_type=stype,
                direction=BULLISH,
                level=active_high_level,
                pivot_bar=active_high_bar,
                is_internal=is_internal,
            ))

        # Bearish break: close crosses BELOW the active swing low (ta.crossunder)
        if (
            not active_low_crossed
            and not np.isnan(active_low_level)
            and prev_c > active_low_level
            and curr_c <= active_low_level
        ):
            stype = StructureType.CHOCH if trend == BULLISH else StructureType.BOS
            active_low_crossed = True
            trend = BEARISH
            events.append(StructureEvent(
                bar_index=i,
                structure_type=stype,
                direction=BEARISH,
                level=active_low_level,
                pivot_bar=active_low_bar,
                is_internal=is_internal,
            ))

    return events
