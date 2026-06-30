"""
SMC-based trading strategy — generates LONG/SHORT signals from SMC signals.

Entry logic:
    LONG  conditions:
        1. Internal structure is BULLISH (last internal event = BOS or CHoCH upward)
        2. Current close is at or near (ob_proximity_pct) an active BULLISH order block
           OR an active bullish FVG overlaps the current price

    SHORT conditions (mirror):
        1. Internal structure is BEARISH
        2. Current close is at or near an active BEARISH order block
           OR an active bearish FVG overlaps the current price

Confidence:
    - CHoCH (reversal) at OB  → 0.90
    - BOS   (continuation) at OB → 0.75
    - FVG entry only (no OB) → 0.60

No signal is emitted when data is insufficient (fewer bars than
2 × max(swing_length, internal_length)).
"""
from __future__ import annotations

import pandas as pd

from zeus.strategy.base import Signal, SignalType, Strategy
from zeus.strategy.smc.indicator import SMCResult, analyze
from zeus.strategy.smc.order_block import get_active_order_blocks
from zeus.strategy.smc.fvg import get_active_fvgs
from zeus.strategy.smc.pivot import BULLISH, BEARISH
from zeus.strategy.smc.structure import StructureType


class SMCStrategy(Strategy):
    """
    Implements a Smart Money Concepts strategy using internal structure as
    the primary bias filter and order blocks / FVGs as entry triggers.
    """

    def __init__(
        self,
        swing_length: int = 50,
        internal_length: int = 5,
        ob_proximity_pct: float = 0.003,   # ±0.3 % of OB to consider "at the zone"
        atr_period: int = 200,
        ob_mitigation: str = "highlow",
    ) -> None:
        self.swing_length    = swing_length
        self.internal_length = internal_length
        self.ob_proximity_pct = ob_proximity_pct
        self.atr_period      = atr_period
        self.ob_mitigation   = ob_mitigation

        self._min_bars = max(swing_length, internal_length) * 2

    def generate_signal(self, df: pd.DataFrame, bar_index: int) -> Signal:
        """Run SMC analysis on df[:bar_index+1] and return a trading signal."""
        window = df.iloc[: bar_index + 1]

        if len(window) < self._min_bars:
            return Signal(SignalType.NONE, 0.0, "insufficient data", bar_index)

        result: SMCResult = analyze(
            window,
            swing_length=self.swing_length,
            internal_length=self.internal_length,
            atr_period=self.atr_period,
            ob_mitigation=self.ob_mitigation,
        )

        price = float(df["close"].iloc[bar_index])
        return self._evaluate(result, price, bar_index)

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #

    def _evaluate(self, result: SMCResult, price: float, bar_index: int) -> Signal:
        bias = result.internal_bias

        if bias == 0:
            return Signal(SignalType.NONE, 0.0, "no internal bias", bar_index)

        # Check order block entry
        ob_result = self._ob_entry(result, price, bias, bar_index)
        if ob_result is not None:
            return ob_result

        # Check FVG entry (lower confidence — no OB confluence)
        fvg_result = self._fvg_entry(result, price, bias, bar_index)
        if fvg_result is not None:
            return fvg_result

        return Signal(SignalType.NONE, 0.0, "no entry condition", bar_index)

    def _ob_entry(
        self,
        result: SMCResult,
        price: float,
        bias: int,
        bar_index: int,
    ) -> Signal | None:
        active_obs = get_active_order_blocks(result.internal_obs, bar_index)
        prox = self.ob_proximity_pct

        for ob in reversed(active_obs):
            if ob.direction != bias:
                continue
            lo_zone = ob.low  * (1 - prox)
            hi_zone = ob.high * (1 + prox)
            if lo_zone <= price <= hi_zone:
                # Determine if last internal event was CHoCH (reversal → higher confidence)
                last_event = next(
                    (e for e in reversed(result.internal_structure) if e.direction == bias), None
                )
                is_choch   = last_event is not None and last_event.structure_type == StructureType.CHOCH
                confidence = 0.90 if is_choch else 0.75
                stype      = SignalType.LONG if bias == BULLISH else SignalType.SHORT
                tag        = "CHoCH" if is_choch else "BOS"
                reason     = (
                    f"internal {tag} {'bullish' if bias == BULLISH else 'bearish'} "
                    f"+ price at OB [{ob.low:.4f}-{ob.high:.4f}]"
                )
                return Signal(stype, confidence, reason, bar_index)
        return None

    def _fvg_entry(
        self,
        result: SMCResult,
        price: float,
        bias: int,
        bar_index: int,
    ) -> Signal | None:
        active_fvgs = get_active_fvgs(result.fvgs, bar_index)

        for fvg in reversed(active_fvgs):
            if fvg.direction != bias:
                continue
            if fvg.bottom <= price <= fvg.top:
                stype  = SignalType.LONG if bias == BULLISH else SignalType.SHORT
                reason = (
                    f"price inside {'bullish' if bias == BULLISH else 'bearish'} FVG "
                    f"[{fvg.bottom:.4f}-{fvg.top:.4f}]"
                )
                return Signal(stype, 0.60, reason, bar_index)
        return None
