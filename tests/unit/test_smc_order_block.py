"""
Unit tests for zeus.strategy.smc.order_block.

Tests OB detection and mitigation on controlled OHLCV series.
"""

import pandas as pd
import pytest

from zeus.strategy.smc.order_block import (
    OrderBlock,
    detect_order_blocks,
    get_active_order_blocks,
)
from zeus.strategy.smc.pivot import BULLISH, BEARISH
from zeus.strategy.smc.structure import StructureEvent, StructureType


def _ev(direction: int, pivot_bar: int, bar_index: int, is_internal: bool = False) -> StructureEvent:
    """Helper to build a minimal StructureEvent for testing."""
    stype = StructureType.BOS
    return StructureEvent(
        bar_index=bar_index,
        structure_type=stype,
        direction=direction,
        level=0.0,
        pivot_bar=pivot_bar,
        is_internal=is_internal,
    )


def _df(n: int, base: float = 100.0) -> pd.DataFrame:
    """Flat OHLCV DataFrame with controlled dip at index 5."""
    h = [base + 1] * n
    lo = [base - 1] * n
    c  = [base] * n
    # Plant a dip at bar 5 to create a clear bullish OB location
    lo[5] = base - 5
    c[5]  = base - 3
    return pd.DataFrame(
        {"high": pd.Series(h, dtype=float),
         "low":  pd.Series(lo, dtype=float),
         "close":pd.Series(c, dtype=float)},
    )


class TestOrderBlockDetection:
    def test_bullish_ob_in_correct_range(self):
        """
        For a bullish structure event (pivot=3, break=10), the OB must be
        the bar with the lowest parsedLow in bars [3, 10).
        """
        n = 15
        df = _df(n)

        event = _ev(BULLISH, pivot_bar=3, bar_index=10)
        obs = detect_order_blocks(df["high"], df["low"], df["close"], [event], atr_period=3)

        assert len(obs) == 1
        ob = obs[0]
        assert ob.direction == BULLISH
        # OB bar must be within [3, 10)
        assert 3 <= ob.bar_index < 10

    def test_bearish_ob_in_correct_range(self):
        """
        For a bearish structure event (pivot=2, break=8), the OB must be
        the bar with the highest parsedHigh in bars [2, 8).
        """
        n = 12
        h  = [100.0] * n
        lo = [99.0]  * n
        c  = [100.0] * n
        h[4] = 110.0   # spike at bar 4 → should become the bearish OB
        df = pd.DataFrame({"high": h, "low": lo, "close": c})

        event = _ev(BEARISH, pivot_bar=2, bar_index=8)
        obs = detect_order_blocks(df["high"], df["low"], df["close"], [event], atr_period=3)

        assert len(obs) == 1
        ob = obs[0]
        assert ob.direction == BEARISH
        assert 2 <= ob.bar_index < 8

    def test_ob_is_active_before_mitigation(self):
        """OB must not be mitigated at the bar it was detected."""
        n = 10
        df = _df(n)
        event = _ev(BULLISH, pivot_bar=2, bar_index=7)
        obs = detect_order_blocks(df["high"], df["low"], df["close"], [event], atr_period=3)

        assert len(obs) == 1
        ob = obs[0]
        active = get_active_order_blocks(obs, at_bar=7)
        assert len(active) == 1

    def test_bullish_ob_mitigated_when_low_breaches(self):
        """Bullish OB is mitigated when a bar's low falls below ob.low."""
        n = 15
        h  = [100.0] * n
        lo = [99.0]  * n
        c  = [100.0] * n
        # Create a very low bar inside the event window to set up the OB
        lo[5] = 90.0
        c[5]  = 91.0
        # Then breach after detection
        lo[12] = 88.0   # below OB.low = 90.0
        c[12]  = 89.0
        df = pd.DataFrame({"high": h, "low": lo, "close": c})

        event = _ev(BULLISH, pivot_bar=3, bar_index=9)
        obs = detect_order_blocks(
            df["high"], df["low"], df["close"], [event],
            atr_period=3, mitigation="highlow",
        )

        assert len(obs) == 1
        assert obs[0].mitigated_at == 12

    def test_empty_events_returns_empty_obs(self):
        """No structure events → no order blocks."""
        df = _df(20)
        obs = detect_order_blocks(df["high"], df["low"], df["close"], [], atr_period=3)
        assert obs == []

    def test_get_active_filters_future_obs(self):
        """OBs detected after at_bar must not appear in active list."""
        ob = OrderBlock(
            bar_index=3, high=105.0, low=100.0,
            direction=BULLISH, is_internal=False,
            detected_at=5, mitigated_at=-1,
        )
        active = get_active_order_blocks([ob], at_bar=4)
        assert active == []

    def test_max_count_limits_returned_obs(self):
        """get_active_order_blocks must respect max_count."""
        obs = [
            OrderBlock(bar_index=i, high=100.0, low=99.0, direction=BULLISH,
                       is_internal=False, detected_at=i, mitigated_at=-1)
            for i in range(10)
        ]
        active = get_active_order_blocks(obs, at_bar=20, max_count=3)
        assert len(active) == 3
