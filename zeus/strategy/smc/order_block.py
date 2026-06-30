"""
Order Block detection and mitigation.

Translates Pine Script storeOrderBlock() and deleteOrderBlocks().

When a BOS/CHoCH occurs the bot searches the range [pivot_bar, breakout_bar)
for the most extreme candle in the direction opposing the breakout.

Volatile bars are filtered via a parsed high/low:
    high_vol_bar = (high - low) >= 2 * ATR(200)
    parsed_high  = low  if high_vol_bar else high
    parsed_low   = high if high_vol_bar else low
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from zeus.strategy.smc.pivot import BULLISH, BEARISH
from zeus.strategy.smc.structure import StructureEvent


@dataclass(frozen=True)
class OrderBlock:
    """A detected order block zone."""
    bar_index: int      # bar index of the OB candle
    high: float
    low: float
    direction: int      # BULLISH (+1) = demand zone; BEARISH (-1) = supply zone
    is_internal: bool
    detected_at: int    # bar index of the structure break that created this OB
    mitigated_at: int = -1   # -1 = still active


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
    """Compute ATR(period) using Wilder's smoothing (SMA approximation for simplicity)."""
    h, lo, c = highs, lows, closes
    tr = np.maximum(
        h[1:] - lo[1:],
        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(lo[1:] - c[:-1])),
    )
    tr_full = np.concatenate([[np.nan], tr])
    return pd.Series(tr_full).rolling(period, min_periods=1).mean().to_numpy()


def _parsed_high_low(
    highs: np.ndarray,
    lows: np.ndarray,
    atr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute parsed high/low with volatility filter.

    Pine:
        highVolatilityBar = (high - low) >= 2 * atr
        parsedHigh        = highVolatilityBar ? low  : high
        parsedLow         = highVolatilityBar ? high : low
    """
    high_vol = (highs - lows) >= 2 * atr
    return np.where(high_vol, lows, highs), np.where(high_vol, highs, lows)


def detect_order_blocks(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    structure_events: list[StructureEvent],
    atr_period: int = 200,
    mitigation: str = "highlow",
) -> list[OrderBlock]:
    """
    Derive order blocks from structure breakout events.

    For each event, scan [pivot_bar, breakout_bar) in the parsed arrays:
    - Bullish breakout → bar with the lowest parsed_low  (demand zone)
    - Bearish breakout → bar with the highest parsed_high (supply zone)

    Mitigation is detected by scanning bars after the OB was created:
    - Bullish OB mitigated when: low < ob.low  (price dips below the demand zone)
    - Bearish OB mitigated when: high > ob.high (price pushes above the supply zone)

    Args:
        mitigation: "highlow" uses high/low for mitigation check; "close" uses close.
    """
    h  = highs.to_numpy(dtype=np.float64)
    lo = lows.to_numpy(dtype=np.float64)
    c  = closes.to_numpy(dtype=np.float64)
    n  = len(h)

    atr        = _atr(h, lo, c, atr_period)
    parsed_h, parsed_lo = _parsed_high_low(h, lo, atr)

    order_blocks: list[OrderBlock] = []

    for ev in structure_events:
        start = ev.pivot_bar
        end   = ev.bar_index  # exclusive (Pine: .slice(pivotIndex, bar_index))

        if start < 0 or end <= start or end > n:
            continue

        ph_slice = parsed_h[start:end]
        pl_slice = parsed_lo[start:end]

        if len(ph_slice) == 0:
            continue

        if ev.direction == BULLISH:
            ob_rel = int(np.argmin(pl_slice))
        else:
            ob_rel = int(np.argmax(ph_slice))

        ob_idx = start + ob_rel

        order_blocks.append(OrderBlock(
            bar_index=ob_idx,
            high=float(h[ob_idx]),
            low=float(lo[ob_idx]),
            direction=ev.direction,
            is_internal=ev.is_internal,
            detected_at=ev.bar_index,
        ))

    # Post-process: detect mitigation
    result: list[OrderBlock] = []
    for ob in order_blocks:
        mitigated_at = -1
        for j in range(ob.detected_at + 1, n):
            if ob.direction == BEARISH:
                src = c[j] if mitigation == "close" else h[j]
                if src > ob.high:
                    mitigated_at = j
                    break
            else:
                src = c[j] if mitigation == "close" else lo[j]
                if src < ob.low:
                    mitigated_at = j
                    break
        result.append(replace(ob, mitigated_at=mitigated_at))

    return result


def get_active_order_blocks(
    order_blocks: list[OrderBlock],
    at_bar: int,
    max_count: int = 5,
) -> list[OrderBlock]:
    """
    Return order blocks that are active (non-mitigated) at the given bar.

    Returns at most max_count, most recently detected first.
    """
    active = [
        ob for ob in order_blocks
        if ob.detected_at <= at_bar
        and (ob.mitigated_at == -1 or ob.mitigated_at > at_bar)
    ]
    return active[-max_count:]
