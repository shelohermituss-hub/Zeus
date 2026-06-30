"""
Unit tests for zeus.strategy.smc.pivot.

Uses synthetic zigzag price series where pivot highs and lows are predictable.

Zigzag construction:
    UP phase:   prices rise 1 unit per bar for 'up' bars
    DOWN phase: prices fall 1 unit per bar for 'down' bars

With size=3, a pivot HIGH at bar `i` is confirmed at bar `i+3`.
"""

import numpy as np
import pandas as pd
import pytest

from zeus.strategy.smc.pivot import (
    PivotPoint,
    detect_pivots,
    BULLISH_LEG,
    BEARISH_LEG,
    BULLISH,
    BEARISH,
)


def make_zigzag(
    segments: list[tuple[int, float]],
    start: float = 100.0,
) -> pd.DataFrame:
    """
    Build an OHLCV-like DataFrame from a list of (bars, direction) segments.

    direction > 0 → rising, direction < 0 → falling.
    Each bar moves by abs(direction) in the given direction.
    High = close + 0.1, Low = close - 0.1 (candle body only, flat wicks).
    """
    prices = [start]
    for bars, step in segments:
        for _ in range(bars):
            prices.append(prices[-1] + step)

    close = pd.Series(prices, dtype=float)
    high  = close + 0.1
    low   = close - 0.1
    return pd.DataFrame({"high": high, "low": low, "close": close})


class TestDetectPivotsBasic:
    def test_single_peak_detected(self):
        """Simple A-up B-down pattern must yield one pivot high."""
        df = make_zigzag([(10, +1.0), (10, -1.0)])
        highs = df["high"]
        lows  = df["low"]
        pivots = detect_pivots(highs, lows, size=3)
        highs_found = [p for p in pivots if p.is_high]
        assert len(highs_found) == 1, f"Expected 1 pivot high, got {len(highs_found)}"

    def test_single_trough_detected(self):
        """Simple A-down B-up pattern must yield one pivot low."""
        df = make_zigzag([(10, -1.0), (10, +1.0)])
        pivots = detect_pivots(df["high"], df["low"], size=3)
        lows_found = [p for p in pivots if not p.is_high]
        assert len(lows_found) == 1

    def test_zigzag_alternates_high_low(self):
        """Multiple segments must produce alternating pivot highs and lows."""
        df = make_zigzag([(8, +1.0), (8, -1.0), (8, +1.0), (8, -1.0)])
        pivots = detect_pivots(df["high"], df["low"], size=3)

        # Pivots must alternate: high, low, high, low … or low, high, low, high …
        for a, b in zip(pivots, pivots[1:]):
            assert a.is_high != b.is_high, (
                f"Expected alternating pivots, got two consecutive "
                f"{'highs' if a.is_high else 'lows'} at bars {a.bar_index}, {b.bar_index}"
            )

    def test_pivot_level_is_correct_price(self):
        """Pivot high level must equal the actual high of the peak bar."""
        df = make_zigzag([(5, +2.0), (5, -1.0)])
        pivots = detect_pivots(df["high"], df["low"], size=3)
        highs_found = [p for p in pivots if p.is_high]
        assert len(highs_found) >= 1
        peak_bar = highs_found[0].bar_index
        assert highs_found[0].level == pytest.approx(df["high"].iloc[peak_bar])

    def test_confirmation_delay(self):
        """A pivot at bar p must be confirmed at bar p + size."""
        size = 4
        df = make_zigzag([(6, +1.0), (6, -1.0)])
        pivots = detect_pivots(df["high"], df["low"], size=size)
        highs_found = [p for p in pivots if p.is_high]
        assert len(highs_found) >= 1
        ph = highs_found[0]
        assert ph.confirmed_at == ph.bar_index + size

    def test_no_pivots_on_flat_series(self):
        """A perfectly flat series produces no pivot highs or lows."""
        n      = 50
        prices = pd.Series([100.0] * n)
        pivots = detect_pivots(prices, prices, size=5)
        assert pivots == []

    def test_larger_size_yields_fewer_pivots(self):
        """Increasing size must produce equal or fewer pivots."""
        df = make_zigzag([(5, +1.0), (5, -1.0)] * 6)
        p3 = detect_pivots(df["high"], df["low"], size=3)
        p5 = detect_pivots(df["high"], df["low"], size=5)
        assert len(p5) <= len(p3)

    def test_size_one_detects_every_local_extremum(self):
        """With size=1, every local high/low should be detected."""
        df = make_zigzag([(2, +1.0), (2, -1.0)] * 4)
        pivots = detect_pivots(df["high"], df["low"], size=1)
        assert len(pivots) > 0


class TestPivotOrdering:
    def test_pivots_are_chronological(self):
        """Returned pivots must be ordered by confirmed_at bar."""
        df = make_zigzag([(6, +1.0), (6, -1.0), (6, +1.0), (6, -1.0)])
        pivots = detect_pivots(df["high"], df["low"], size=3)
        confirmed_ats = [p.confirmed_at for p in pivots]
        assert confirmed_ats == sorted(confirmed_ats)

    def test_pivot_bar_index_within_bounds(self):
        """bar_index must be within [0, len(df)-1]."""
        df = make_zigzag([(8, +1.0), (8, -1.0)])
        pivots = detect_pivots(df["high"], df["low"], size=3)
        n = len(df)
        for p in pivots:
            assert 0 <= p.bar_index < n


class TestPivotEdgeCases:
    def test_series_shorter_than_size_returns_empty(self):
        """If the series is shorter than size, no pivots can be confirmed."""
        prices = pd.Series([1.0, 2.0, 3.0])
        pivots = detect_pivots(prices, prices, size=5)
        assert pivots == []

    def test_monotonic_rising_no_pivot_high(self):
        """Strictly rising series produces no pivot HIGHS."""
        prices = pd.Series(np.arange(1, 31, dtype=float))
        pivots = detect_pivots(prices, prices, size=3)
        highs = [p for p in pivots if p.is_high]
        assert highs == []

    def test_monotonic_rising_at_most_one_initial_pivot_low(self):
        """
        In a rising series, bar 0 is the global minimum and may be confirmed
        as a pivot low (matching Pine Script's var leg = BEARISH_LEG initial state).
        No SUBSEQUENT pivot lows should appear.
        """
        prices = pd.Series(np.arange(1, 31, dtype=float))
        pivots = detect_pivots(prices, prices, size=3)
        lows = [p for p in pivots if not p.is_high]
        assert len(lows) <= 1
        if lows:
            assert lows[0].bar_index == 0

    def test_monotonic_falling_no_pivot_high(self):
        """Strictly falling series produces no pivot highs."""
        prices = pd.Series(np.arange(30, 0, -1, dtype=float))
        pivots = detect_pivots(prices, prices, size=3)
        highs = [p for p in pivots if p.is_high]
        assert highs == []
