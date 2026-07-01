"""
LTF liquidity sweep entry confirmation.

When price enters an HTF S&D zone, the 1M entry bar must perform a
"liquidity sweep" of the previous bar's extreme before reversing:

  LONG  : bar low  < prev bar low   AND  bar close > prev bar low
           ↑ wick sweeps sell-side stops    ↑ closes back above = bulls absorb

  SHORT : bar high > prev bar high  AND  bar close < prev bar high
           ↑ wick sweeps buy-side stops     ↑ closes back below = bears absorb

This is the ICT/SMC "inducement sweep" or "pin bar at level" pattern.
Price must fake a breakout to collect the resting liquidity, then immediately
reverse — confirming institutional interest at the zone.

The check scans the last *lookback* LTF bars (default 3) so a sweep that
occurred 1–2 bars before the current trigger bar is still accepted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from zeus.strategy.smc.pivot import BULLISH, BEARISH


def ltf_liquidity_sweep(
    df:        pd.DataFrame,
    bar_index: int,
    direction: int,
    lookback:  int = 3,
) -> tuple[bool, str]:
    """
    Return (confirmed, reason).

    Scans the last *lookback* bars (inclusive of bar_index) for a bar that:
      - LONG  : low < previous bar low  AND  close > previous bar low
      - SHORT : high > previous bar high AND  close < previous bar high

    Args:
        df:        LTF OHLCV DataFrame (1M bars).
        bar_index: Current bar index (entry candidate).
        direction: BULLISH (+1) or BEARISH (−1).
        lookback:  Number of bars to scan (default 3).

    Returns:
        (True, description) if a valid sweep is found in the window.
        (False, reason)     if no sweep was detected.
    """
    if bar_index < 1:
        return False, "insufficient bars for sweep check"

    highs  = df["high"].to_numpy(dtype=float)
    lows   = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)

    # Scan from oldest to newest within the lookback window
    start = max(1, bar_index - lookback + 1)   # need i-1 so start >= 1

    for i in range(start, bar_index + 1):
        prev_low  = lows[i - 1]
        prev_high = highs[i - 1]

        if direction == BULLISH:
            if lows[i] < prev_low and closes[i] > prev_low:
                return (
                    True,
                    f"sell-side sweep bar={i}: low {lows[i]:.2f} < prev_low {prev_low:.2f}, "
                    f"close {closes[i]:.2f} recovered",
                )

        elif direction == BEARISH:
            if highs[i] > prev_high and closes[i] < prev_high:
                return (
                    True,
                    f"buy-side sweep bar={i}: high {highs[i]:.2f} > prev_high {prev_high:.2f}, "
                    f"close {closes[i]:.2f} reversed",
                )

    direction_name = "sell-side" if direction == BULLISH else "buy-side"
    return (
        False,
        f"no {direction_name} LTF sweep in last {lookback} bars "
        f"(bar {bar_index - lookback + 1}–{bar_index})",
    )
