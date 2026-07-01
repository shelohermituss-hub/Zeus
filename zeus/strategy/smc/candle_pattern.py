"""
1M candle pattern entry filter — Piste B gate (gate 6.4).

Patterns
--------
  LONG : hammer / pin bar  — lower wick ≥ min_wick_ratio of total range,
                             close in upper half of the bar.
         OR bullish engulfing — current body fully engulfs previous body,
                                current candle closes bullish.
  SHORT: shooting star      — upper wick ≥ min_wick_ratio of total range,
                              close in lower half of the bar.
         OR bearish engulfing — current body fully engulfs previous body,
                                current candle closes bearish.

Returns (confirmed: bool, reason: str).
"""
from __future__ import annotations

import pandas as pd

from zeus.strategy.smc.pivot import BULLISH, BEARISH


def detect_entry_candle(
    df: pd.DataFrame,
    bar_index: int,
    direction: int,
    min_wick_ratio: float = 0.60,
) -> tuple[bool, str]:
    """
    Detect a qualifying 1M entry candle pattern at bar_index.

    For longs: hammer (long lower wick closing in upper half)
               or bullish engulfing.
    For shorts: shooting star (long upper wick closing in lower half)
                or bearish engulfing.

    Args:
        df:             LTF (1M) OHLCV DataFrame.
        bar_index:      Index of the entry bar to inspect.
        direction:      BULLISH or BEARISH.
        min_wick_ratio: Minimum wick / total_range to qualify as pin bar.

    Returns:
        (True, description)  if a qualifying pattern is found.
        (False, reason)      if no pattern found.
    """
    if bar_index < 1:
        return False, "insufficient bars for pattern check"

    o = float(df["open"].iloc[bar_index])
    h = float(df["high"].iloc[bar_index])
    l = float(df["low"].iloc[bar_index])
    c = float(df["close"].iloc[bar_index])

    candle_range = h - l
    if candle_range <= 0.0:
        return False, "zero-range candle — no pattern"

    body_top    = max(o, c)
    body_bottom = min(o, c)
    lower_wick  = body_bottom - l
    upper_wick  = h - body_top
    mid_price   = (h + l) / 2.0

    if direction == BULLISH:
        lower_ratio = lower_wick / candle_range
        if lower_ratio >= min_wick_ratio and c >= mid_price:
            return (
                True,
                f"hammer: lower_wick={lower_wick:.2f} ({lower_ratio:.0%} of range), "
                f"close={c:.2f} in upper half",
            )

        if c > o:
            prev_o = float(df["open"].iloc[bar_index - 1])
            prev_c = float(df["close"].iloc[bar_index - 1])
            prev_top    = max(prev_o, prev_c)
            prev_bottom = min(prev_o, prev_c)
            if o <= prev_bottom and c >= prev_top:
                return (
                    True,
                    f"bullish engulfing: engulfed body [{prev_bottom:.2f},{prev_top:.2f}]",
                )

        return (
            False,
            f"no bullish pattern: lower_wick={lower_ratio:.0%} "
            f"(need ≥{min_wick_ratio:.0%}), close {'above' if c >= mid_price else 'below'} mid",
        )

    elif direction == BEARISH:
        upper_ratio = upper_wick / candle_range
        if upper_ratio >= min_wick_ratio and c <= mid_price:
            return (
                True,
                f"shooting_star: upper_wick={upper_wick:.2f} ({upper_ratio:.0%} of range), "
                f"close={c:.2f} in lower half",
            )

        if c < o:
            prev_o = float(df["open"].iloc[bar_index - 1])
            prev_c = float(df["close"].iloc[bar_index - 1])
            prev_top    = max(prev_o, prev_c)
            prev_bottom = min(prev_o, prev_c)
            if o >= prev_top and c <= prev_bottom:
                return (
                    True,
                    f"bearish engulfing: engulfed body [{prev_bottom:.2f},{prev_top:.2f}]",
                )

        return (
            False,
            f"no bearish pattern: upper_wick={upper_ratio:.0%} "
            f"(need ≥{min_wick_ratio:.0%}), close {'below' if c <= mid_price else 'above'} mid",
        )

    return False, f"unknown direction {direction}"
