"""
Fair Value Gap (FVG) detection and mitigation.

Translates Pine Script drawFairValueGaps() and deleteFairValueGaps().

Three-bar FVG pattern (bars A, B, C):
    A = 2 bars ago  (last2High, last2Low)
    B = 1 bar ago   (lastOpen, lastClose) — the impulse bar
    C = current bar (currentHigh, currentLow)

Bullish FVG (gap above A, below C):
    currentLow > last2High          → gap exists
    lastClose > last2High           → impulse bar confirms direction
    barDeltaPercent > threshold     → impulse bar has enough momentum
    → top = currentLow, bottom = last2High

Bearish FVG (gap below A, above C):
    currentHigh < last2Low          → gap exists
    lastClose < last2Low            → impulse bar confirms direction
    -barDeltaPercent > threshold    → impulse bar has enough momentum
    → top = last2Low, bottom = currentHigh

Mitigation (Pine: deleteFairValueGaps):
    Bullish: low < fvg.bottom (price filled gap from below)
    Bearish: high > fvg.top   (price filled gap from above)
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from zeus.strategy.smc.pivot import BULLISH, BEARISH


@dataclass(frozen=True)
class FairValueGap:
    """A detected FVG zone."""
    bar_index: int    # bar index of C (the bar at which the FVG was detected)
    top: float
    bottom: float
    direction: int         # BULLISH (+1) or BEARISH (-1)
    mitigated_at: int = -1  # -1 = still active


def detect_fvg(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    opens: pd.Series,
    auto_threshold: bool = True,
) -> list[FairValueGap]:
    """
    Detect Fair Value Gaps and their mitigation.

    Args:
        highs, lows, closes, opens: OHLCV series (same length and index).
        auto_threshold:             Match Pine's auto-threshold behaviour.
                                    When True, filters weak FVGs using the
                                    cumulative mean absolute bar-delta.

    Returns:
        List of FairValueGap objects (active and mitigated).
    """
    h  = highs.to_numpy(dtype=np.float64)
    lo = lows.to_numpy(dtype=np.float64)
    c  = closes.to_numpy(dtype=np.float64)
    o  = opens.to_numpy(dtype=np.float64)
    n  = len(h)

    # Pine: barDeltaPercent = (close[1] - open[1]) / (open[1] * 100)
    # threshold = cumulative mean of abs(barDeltaPercent) * 2
    with np.errstate(divide="ignore", invalid="ignore"):
        bar_delta_pct = np.where(o != 0, (c - o) / (o * 100), 0.0)

    cum_threshold = np.zeros(n)
    if auto_threshold:
        abs_delta = np.abs(bar_delta_pct)
        cum_sum = np.cumsum(abs_delta)
        indices = np.arange(1, n + 1, dtype=np.float64)
        cum_threshold = 2.0 * cum_sum / indices

    fvgs: list[FairValueGap] = []

    for i in range(2, n):
        last_close    = c[i - 1]
        last_open     = o[i - 1]
        two_ago_high  = h[i - 2]
        two_ago_low   = lo[i - 2]
        curr_high     = h[i]
        curr_low      = lo[i]

        # Middle bar impulse delta
        mid_delta = bar_delta_pct[i - 1]
        thr = cum_threshold[i] if auto_threshold else 0.0

        bullish = (
            curr_low > two_ago_high
            and last_close > two_ago_high
            and mid_delta > thr
        )
        bearish = (
            curr_high < two_ago_low
            and last_close < two_ago_low
            and -mid_delta > thr
        )

        if bullish:
            fvgs.append(FairValueGap(
                bar_index=i,
                top=curr_low,
                bottom=two_ago_high,
                direction=BULLISH,
            ))
        if bearish:
            fvgs.append(FairValueGap(
                bar_index=i,
                top=two_ago_low,
                bottom=curr_high,
                direction=BEARISH,
            ))

    # Detect mitigation
    result: list[FairValueGap] = []
    for fvg in fvgs:
        mitigated_at = -1
        for j in range(fvg.bar_index + 1, n):
            if fvg.direction == BULLISH and lo[j] < fvg.bottom:
                mitigated_at = j
                break
            if fvg.direction == BEARISH and h[j] > fvg.top:
                mitigated_at = j
                break
        result.append(replace(fvg, mitigated_at=mitigated_at))

    return result


def get_active_fvgs(
    fvgs: list[FairValueGap],
    at_bar: int,
) -> list[FairValueGap]:
    """Return FVGs that are still open (not mitigated) at the given bar."""
    return [
        fvg for fvg in fvgs
        if fvg.bar_index <= at_bar
        and (fvg.mitigated_at == -1 or fvg.mitigated_at > at_bar)
    ]
