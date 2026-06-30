"""
SMC-based trading strategy — 10-factor confluence scoring engine.

Replaces the earlier 2-factor approach (internal structure + OB/FVG)
with the full multi-factor confluence engine defined in confluence.py.
A signal is only emitted when BOTH gate factors fire AND the total
active-factor count meets the minimum score threshold.

Signal generation
-----------------
At each bar the strategy:
1. Runs the full SMC analysis (analyze()) over the available window.
2. Calls best_confluence() to evaluate all 10 factors for both directions.
3. Maps the winning ConfluenceScore to a Signal:
   - direction  → SignalType.LONG (+1) or SHORT (-1)
   - confidence → active_count / 10  (continuous 0.0–1.0)
   - reason     → human-readable active-factor summary for logging

Gate factors (1 and 8) must both fire regardless of total score.
No signal is emitted when data is insufficient or no confluent setup exists.
"""
from __future__ import annotations

import pandas as pd

from zeus.strategy.base import Signal, SignalType, Strategy
from zeus.strategy.confluence import ConfluenceScore, best_confluence
from zeus.strategy.smc.indicator import SMCResult, analyze
from zeus.strategy.smc.pivot import BULLISH


class SMCStrategy(Strategy):
    """
    Smart Money Concepts strategy driven by a 10-factor confluence score.

    A trade setup requires:
    - Factor 1  (GATE): swing market structure aligned with the direction.
    - Factor 8  (GATE): internal structure bias confirms the direction.
    - Total active-factor count ≥ min_score (default 4 out of 10).

    Both gate factors must fire for a signal regardless of total score.

    Args:
        swing_length:          Lookback for swing pivot detection (Pine default 50).
        internal_length:       Lookback for internal structure pivots (Pine default 5).
        atr_period:            ATR period for order-block volatility filter.
        ob_mitigation:         OB mitigation mode: "highlow" or "close".
        vol_num_bins:          Price grid resolution for the volume profile.
        vol_value_area_pct:    Fraction of volume the value area must cover (0.70 = 70%).
        min_score:             Minimum active factors required (both gates + score).
        poc_tolerance_pct:     ±% band around POC for factor 6.
        session_tolerance_pct: ±% band around session H/L for factor 7.
        fib_50_tolerance_pct:  ±% band around Fibonacci 50% for factor 10.
        sweep_lookback:        Max bars since last liquidity sweep for factor 4.
    """

    def __init__(
        self,
        swing_length:          int   = 50,
        internal_length:       int   = 5,
        atr_period:            int   = 200,
        ob_mitigation:         str   = "highlow",
        vol_num_bins:          int   = 100,
        vol_value_area_pct:    float = 0.70,
        min_score:             float = 4.0,
        poc_tolerance_pct:     float = 0.003,
        session_tolerance_pct: float = 0.003,
        fib_50_tolerance_pct:  float = 0.003,
        sweep_lookback:        int   = 10,
    ) -> None:
        self.swing_length          = swing_length
        self.internal_length       = internal_length
        self.atr_period            = atr_period
        self.ob_mitigation         = ob_mitigation
        self.vol_num_bins          = vol_num_bins
        self.vol_value_area_pct    = vol_value_area_pct
        self.min_score             = min_score
        self.poc_tolerance_pct     = poc_tolerance_pct
        self.session_tolerance_pct = session_tolerance_pct
        self.fib_50_tolerance_pct  = fib_50_tolerance_pct
        self.sweep_lookback        = sweep_lookback

        self._min_bars = max(swing_length, internal_length) * 2

    def generate_signal(self, df: pd.DataFrame, bar_index: int) -> Signal:
        """
        Run SMC analysis on df[:bar_index+1] and return a trading signal.

        Returns Signal(NONE) when:
        - The window has fewer than _min_bars bars.
        - No direction meets the minimum score + gate requirements.
        """
        window = df.iloc[: bar_index + 1]

        if len(window) < self._min_bars:
            return Signal(SignalType.NONE, 0.0, "insufficient data", bar_index)

        result: SMCResult = analyze(
            window,
            swing_length=self.swing_length,
            internal_length=self.internal_length,
            atr_period=self.atr_period,
            ob_mitigation=self.ob_mitigation,
            vol_num_bins=self.vol_num_bins,
            vol_value_area_pct=self.vol_value_area_pct,
        )

        price = float(df["close"].iloc[bar_index])
        cs = best_confluence(
            result, price, bar_index,
            min_score=self.min_score,
            poc_tolerance_pct=self.poc_tolerance_pct,
            session_tolerance_pct=self.session_tolerance_pct,
            fib_50_tolerance_pct=self.fib_50_tolerance_pct,
            sweep_lookback=self.sweep_lookback,
        )

        if cs is None:
            return Signal(SignalType.NONE, 0.0, "no confluent signal", bar_index)

        return self._signal_from_score(cs, bar_index, self.min_score)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _signal_from_score(cs: ConfluenceScore, bar_index: int, min_score: float = 4.0) -> Signal:
        """Convert a tradeable ConfluenceScore to a Signal."""
        stype = SignalType.LONG if cs.direction == BULLISH else SignalType.SHORT

        active_names    = [f.name for f in cs.factors if f.active]
        direction_label = "BULLISH" if cs.direction == BULLISH else "BEARISH"
        grade = cs.grade(min_score=min_score)
        reason = (
            f"{direction_label} grade={grade.value} score={cs.active_count}/10 "
            f"[{', '.join(active_names)}]"
        )

        return Signal(stype, cs.confidence, reason, bar_index)
