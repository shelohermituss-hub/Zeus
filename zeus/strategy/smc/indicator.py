"""
Smart Money Concepts indicator — aggregates all SMC sub-modules into a single
analysis pass over an OHLCV DataFrame.

This is the Python equivalent of the full LuxAlgo SMC Pine Script indicator,
producing the same signals (BOS, CHoCH, Order Blocks, FVGs) over historical data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from zeus.strategy.smc.pivot import PivotPoint, detect_pivots, BULLISH, BEARISH
from zeus.strategy.smc.structure import StructureEvent, detect_structure
from zeus.strategy.smc.order_block import OrderBlock, detect_order_blocks, get_active_order_blocks
from zeus.strategy.smc.fvg import FairValueGap, detect_fvg, get_active_fvgs
from zeus.strategy.smc.fibonacci import FibZone, detect_fib_zones
from zeus.strategy.smc.liquidity import (
    LiquidityLevel, LiquiditySweep,
    detect_liquidity_levels, detect_sweeps,
)
from zeus.strategy.smc.session import SessionRange, detect_session_ranges


@dataclass
class SMCResult:
    """Full SMC analysis snapshot for an OHLCV series."""

    # Pivot points
    swing_pivots: list[PivotPoint]       = field(default_factory=list)
    internal_pivots: list[PivotPoint]    = field(default_factory=list)

    # Market structure
    swing_structure: list[StructureEvent]    = field(default_factory=list)
    internal_structure: list[StructureEvent] = field(default_factory=list)

    # Order blocks
    swing_obs: list[OrderBlock]    = field(default_factory=list)
    internal_obs: list[OrderBlock] = field(default_factory=list)

    # Fair Value Gaps
    fvgs: list[FairValueGap] = field(default_factory=list)

    # Fibonacci zones (swing-level only — covers factor 2 and 10)
    fib_zones: list[FibZone] = field(default_factory=list)

    # Liquidity levels and sweeps (factor 4)
    liquidity_levels: list[LiquidityLevel] = field(default_factory=list)
    liquidity_sweeps: list[LiquiditySweep] = field(default_factory=list)

    # Session ranges: Asian / London / NY H/L (factor 7)
    session_ranges: list[SessionRange] = field(default_factory=list)

    # Current bias derived from most recent structure event
    swing_bias: int    = 0   # BULLISH=+1, BEARISH=-1, 0=undefined
    internal_bias: int = 0


def analyze(
    df: pd.DataFrame,
    swing_length: int = 50,
    internal_length: int = 5,
    atr_period: int = 200,
    ob_mitigation: str = "highlow",
    show_fvg: bool = True,
    fvg_auto_threshold: bool = True,
) -> SMCResult:
    """
    Run the complete SMC analysis on an OHLCV DataFrame.

    Column requirements: high, low, close, open (open optional — defaults to
    previous close when absent, which slightly affects FVG threshold accuracy).

    Args:
        df:                OHLCV DataFrame.
        swing_length:      Pine's swingsLengthInput (default 50).
        internal_length:   Internal structure lookback (Pine hardcodes 5).
        atr_period:        ATR period for order block volatility filter.
        ob_mitigation:     OB mitigation check: "highlow" or "close".
        show_fvg:          Whether to compute Fair Value Gaps.
        fvg_auto_threshold: Auto-threshold for FVG impulse filter.

    Returns:
        SMCResult with all computed SMC signals.
    """
    highs  = df["high"]
    lows   = df["low"]
    closes = df["close"]
    opens  = df["open"] if "open" in df.columns else closes.shift(1).fillna(closes.iloc[0])

    # 1 — Pivot detection
    swing_pivots    = detect_pivots(highs, lows, swing_length)
    internal_pivots = detect_pivots(highs, lows, internal_length)

    # 2 — Market structure (BOS / CHoCH)
    swing_struct    = detect_structure(closes, swing_pivots, is_internal=False)
    internal_struct = detect_structure(closes, internal_pivots, is_internal=True)

    # 3 — Order blocks
    swing_obs    = detect_order_blocks(highs, lows, closes, swing_struct,    atr_period, ob_mitigation)
    internal_obs = detect_order_blocks(highs, lows, closes, internal_struct, atr_period, ob_mitigation)

    # 4 — Fair Value Gaps
    fvgs = detect_fvg(highs, lows, closes, opens, fvg_auto_threshold) if show_fvg else []

    # 5 — Fibonacci retracement zones (swing pivots → factors 2 & 10)
    fib_zones = detect_fib_zones(swing_pivots)

    # 6 — Liquidity levels and sweeps (factor 4)
    liq_levels = detect_liquidity_levels(swing_pivots)
    liq_sweeps = detect_sweeps(
        list(highs), list(lows), list(closes), liq_levels
    )

    # 7 — Session ranges: Asian / London / NY H/L (factor 7)
    # Only available when the DataFrame carries datetime index information
    sess_ranges = (
        detect_session_ranges(df)
        if isinstance(df.index, pd.DatetimeIndex)
        else []
    )

    # 8 — Derive current bias from latest structure event
    swing_bias    = next((e.direction for e in reversed(swing_struct)),    0)
    internal_bias = next((e.direction for e in reversed(internal_struct)), 0)

    return SMCResult(
        swing_pivots=swing_pivots,
        internal_pivots=internal_pivots,
        swing_structure=swing_struct,
        internal_structure=internal_struct,
        swing_obs=swing_obs,
        internal_obs=internal_obs,
        fvgs=fvgs,
        fib_zones=fib_zones,
        liquidity_levels=liq_levels,
        liquidity_sweeps=liq_sweeps,
        session_ranges=sess_ranges,
        swing_bias=swing_bias,
        internal_bias=internal_bias,
    )
