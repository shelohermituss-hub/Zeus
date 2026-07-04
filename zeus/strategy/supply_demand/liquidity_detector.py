"""
Liquidity Pool Detector
=======================

Identifies swing-high/swing-low based liquidity pools on M15 bars and
detects liquidity sweeps (wick beyond level + close back inside).

Concepts
--------
Buy-side liquidity  : clustered at swing HIGHS (shorts' stop-losses reside above)
Sell-side liquidity : clustered at swing LOWS  (longs' stop-losses reside below)

Liquidity Sweep (entry confirmation)
  - Sell-side sweep : bar wicks BELOW a swing low and CLOSES ABOVE it → bullish
  - Buy-side sweep  : bar wicks ABOVE a swing high and CLOSES BELOW it → bearish

Pool-based TP (replacing fixed-R target)
  - Long  : nearest swing HIGH above entry → implied RR = (pool - entry) / sl_dist
  - Short : nearest swing LOW  below entry → implied RR = (entry - pool) / sl_dist
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def find_swing_levels(
    df:         pd.DataFrame,
    left_bars:  int = 5,
    right_bars: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute swing-high and swing-low arrays for a full OHLC dataframe.

    A bar i is a swing high if its high is the maximum in
    [i - left_bars, i + right_bars].  Same logic for swing lows.

    Returns
    -------
    swing_highs : float array, NaN where no swing, else the high price
    swing_lows  : float array, NaN where no swing, else the low price
    """
    n      = len(df)
    highs  = df["high"].to_numpy(dtype=float)
    lows   = df["low"].to_numpy(dtype=float)

    swing_highs = np.full(n, np.nan)
    swing_lows  = np.full(n, np.nan)

    for i in range(left_bars, n - right_bars):
        win_h = highs[i - left_bars : i + right_bars + 1]
        win_l = lows[i - left_bars : i + right_bars + 1]

        if highs[i] == win_h.max():
            swing_highs[i] = highs[i]
        if lows[i] == win_l.min():
            swing_lows[i] = lows[i]

    return swing_highs, swing_lows


def find_pool_target(
    swing_highs: np.ndarray,
    swing_lows:  np.ndarray,
    bar_idx:     int,          # current M15 bar index (signal entry bar)
    direction:   str,          # "long" or "short"
    entry:       float,
    pool_lookback: int = 200,  # how many M15 bars back to search
    min_rr:      float = 1.5,  # skip pools that imply RR below this
    sl_dist:     float = 0.0,  # needed to compute implied RR filter
) -> float | None:
    """
    Find the nearest swing-level liquidity pool that is a valid TP target.

    For longs : nearest swing HIGH above entry within lookback bars.
    For shorts: nearest swing LOW  below entry within lookback bars.

    Returns the pool price level, or None if no valid pool found.
    """
    start = max(0, bar_idx - pool_lookback)

    if direction == "long":
        candidates = swing_highs[start:bar_idx]
        levels = candidates[~np.isnan(candidates)]
        levels = levels[levels > entry]
        if len(levels) == 0:
            return None
        nearest = float(levels.min())
        if sl_dist > 0 and (nearest - entry) / sl_dist < min_rr:
            return None
        return nearest

    else:  # short
        candidates = swing_lows[start:bar_idx]
        levels = candidates[~np.isnan(candidates)]
        levels = levels[levels < entry]
        if len(levels) == 0:
            return None
        nearest = float(levels.max())
        if sl_dist > 0 and (entry - nearest) / sl_dist < min_rr:
            return None
        return nearest


def has_liquidity_sweep(
    df:            pd.DataFrame,
    bar_idx:       int,          # M15 bar index of signal entry
    swing_highs:   np.ndarray,
    swing_lows:    np.ndarray,
    direction:     str,          # "long" or "short"
    sweep_lookback: int = 20,    # M15 bars to look back for sweep
    min_wick_pips:  float = 3.0,
    pip_size:       float = 0.0001,
) -> bool:
    """
    Returns True if there was a liquidity sweep in the sweep_lookback bars
    ending at bar_idx (inclusive).

    Sell-side sweep (bullish, confirms long):
        Bar.low < swing_low - min_wick  AND  Bar.close > swing_low

    Buy-side sweep (bearish, confirms short):
        Bar.high > swing_high + min_wick  AND  Bar.close < swing_high
    """
    min_wick = min_wick_pips * pip_size
    start    = max(0, bar_idx - sweep_lookback + 1)

    highs  = df["high"].to_numpy(dtype=float)
    lows   = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)

    for i in range(start, bar_idx + 1):
        lo = lows[i]
        hi = highs[i]
        cl = closes[i]

        if direction == "long":
            # Need a swing low that was swept
            pool_start = max(0, i - 200)
            sl_levels  = swing_lows[pool_start:i]
            sl_levels  = sl_levels[~np.isnan(sl_levels)]
            for level in sl_levels:
                if lo < level - min_wick and cl > level:
                    return True

        else:  # short
            pool_start = max(0, i - 200)
            sh_levels  = swing_highs[pool_start:i]
            sh_levels  = sh_levels[~np.isnan(sh_levels)]
            for level in sh_levels:
                if hi > level + min_wick and cl < level:
                    return True

    return False
