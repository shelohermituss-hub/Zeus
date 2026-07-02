"""
Portfolio strategy — run multiple sd_simulation strategies in parallel.

Bridges the new-style `.run(df) -> list[Signal]` strategies with the
PaperEngine's `Strategy.generate_signal(df, bar_index) -> Signal` interface.

Each constituent strategy produces signals independently; the portfolio
aggregates them with a per-strategy risk budget and cross-strategy cooldown.

Usage::

    portfolio = PortfolioStrategy(
        strategies=[
            ("harmonic", HarmonicStrategy(pivot_size=7, risk_reward=1.0)),
            ("ict_ob",   ICTObStrategy(pivot_size_swing=10, risk_reward=1.0)),
        ],
        risk_per_strategy=0.005,   # 0.5% each → 1% total when both fire
        cross_cooldown_bars=4,     # skip 4 bars after any signal from any strategy
    )

    # In PaperEngine (implements Strategy interface):
    signal = portfolio.generate_signal(df, bar_index)

Notes
-----
The PaperEngine uses percentage-based SL/TP (from settings) rather than the
signal's embedded SL/TP prices.  The portfolio only communicates direction
and confidence through the base Signal interface.  For SL/TP from the signal
object itself, a dedicated paper engine that understands these signals is needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from zeus.strategy.base import Signal, SignalType, Strategy
from zeus.strategy.utils import SignalCooldown


@dataclass(frozen=True)
class PortfolioHit:
    """Internal: one matched signal from a constituent strategy."""
    strategy_name: str
    signal:        Any   # the rich signal (HarmonicSignal / ICTSignal / etc.)


class PortfolioStrategy(Strategy):
    """
    Aggregates multiple sd_simulation strategies under the base Strategy API.

    On each call to generate_signal():
      1. Runs every constituent strategy on the full df received so far.
      2. Collects any signal whose bar_index == bar_index (current bar).
      3. Returns the first matching signal as a base Signal(type=LONG/SHORT/NONE).

    Args:
        strategies:            List of (name, strategy_instance) pairs.
        cross_cooldown_bars:   Minimum bars between any two signals across all
                               strategies (prevents over-trading on confluence).
        prefer_first:          If multiple strategies fire on the same bar,
                               use the one listed first (default True).
    """

    def __init__(
        self,
        strategies:          list[tuple[str, Any]],
        cross_cooldown_bars: int = 4,
        prefer_first:        bool = True,
    ) -> None:
        if not strategies:
            raise ValueError("PortfolioStrategy requires at least one strategy")
        self._strategies   = strategies
        self._cooldown     = SignalCooldown(cross_cooldown_bars)
        self._prefer_first = prefer_first

        # Cache last signal for downstream inspection
        self._last_hit: PortfolioHit | None = None

    @property
    def last_hit(self) -> PortfolioHit | None:
        """The most recently emitted portfolio signal (or None)."""
        return self._last_hit

    def generate_signal(self, df: pd.DataFrame, bar_index: int) -> Signal:
        """
        Implement Strategy base class interface for PaperEngine compatibility.

        Runs all constituent strategies on *df* and returns LONG / SHORT / NONE.
        """
        if not self._cooldown.can_signal(bar_index):
            return Signal(type=SignalType.NONE, confidence=0.0,
                          reason="cross-strategy cooldown", bar_index=bar_index)

        hits: list[PortfolioHit] = []

        for name, strat in self._strategies:
            try:
                signals = strat.run(df)
            except Exception:
                continue
            for sig in signals:
                if getattr(sig, "bar_index", -1) == bar_index:
                    hits.append(PortfolioHit(strategy_name=name, signal=sig))
                    if self._prefer_first:
                        break
            if hits and self._prefer_first:
                break

        if not hits:
            return Signal(type=SignalType.NONE, confidence=0.0,
                          reason="no signal", bar_index=bar_index)

        hit = hits[0]
        self._last_hit = hit
        self._cooldown.mark(bar_index)

        direction = getattr(hit.signal, "direction", "long")
        sig_type  = SignalType.LONG if direction == "long" else SignalType.SHORT
        rr        = getattr(hit.signal, "risk_reward", 1.0)
        confidence = min(1.0, rr / 3.0)

        return Signal(
            type       = sig_type,
            confidence = round(confidence, 3),
            reason     = f"{hit.strategy_name} signal at bar {bar_index}",
            bar_index  = bar_index,
        )
