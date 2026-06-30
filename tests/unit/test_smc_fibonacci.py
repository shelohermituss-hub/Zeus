"""
Unit tests for zeus.strategy.smc.fibonacci.

Covers FibZone geometry, OTE zone boundaries, level checks,
detect_fib_zones pairing, and get_latest_fib_zone filtering.
"""
import pytest

from zeus.strategy.smc.fibonacci import (
    FIB_500,
    FIB_618,
    FIB_786,
    FibZone,
    detect_fib_zones,
    get_latest_fib_zone,
)
from zeus.strategy.smc.pivot import BEARISH, BULLISH, PivotPoint


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _bullish_zone(swing_low=40_000.0, swing_high=50_000.0, formed_at=10) -> FibZone:
    """Standard BULLISH FibZone — impulsive leg went UP."""
    return FibZone(
        swing_high=swing_high,
        swing_low=swing_low,
        direction=BULLISH,
        leg_high_bar=8,
        leg_low_bar=0,
        formed_at=formed_at,
    )


def _bearish_zone(swing_low=40_000.0, swing_high=50_000.0, formed_at=10) -> FibZone:
    """Standard BEARISH FibZone — impulsive leg went DOWN."""
    return FibZone(
        swing_high=swing_high,
        swing_low=swing_low,
        direction=BEARISH,
        leg_high_bar=0,
        leg_low_bar=8,
        formed_at=formed_at,
    )


# ──────────────────────────────────────────────────────────────────────────────
# FibZone.price_at — BULLISH
# ──────────────────────────────────────────────────────────────────────────────

class TestPriceAtBullish:
    """BULLISH: price_at(f) = swing_high − range × f"""

    def test_fraction_zero_returns_swing_high(self):
        z = _bullish_zone(40_000, 50_000)
        assert z.price_at(0.0) == pytest.approx(50_000.0)

    def test_fraction_one_returns_swing_low(self):
        z = _bullish_zone(40_000, 50_000)
        assert z.price_at(1.0) == pytest.approx(40_000.0)

    def test_fraction_618(self):
        z = _bullish_zone(40_000, 50_000)
        # 50_000 − 10_000 × 0.618 = 43_820
        assert z.price_at(FIB_618) == pytest.approx(43_820.0)

    def test_fraction_786(self):
        z = _bullish_zone(40_000, 50_000)
        # 50_000 − 10_000 × 0.786 = 42_140
        assert z.price_at(FIB_786) == pytest.approx(42_140.0)

    def test_level_50(self):
        z = _bullish_zone(40_000, 50_000)
        assert z.level_50 == pytest.approx(45_000.0)


# ──────────────────────────────────────────────────────────────────────────────
# FibZone.price_at — BEARISH
# ──────────────────────────────────────────────────────────────────────────────

class TestPriceAtBearish:
    """BEARISH: price_at(f) = swing_low + range × f"""

    def test_fraction_zero_returns_swing_low(self):
        z = _bearish_zone(40_000, 50_000)
        assert z.price_at(0.0) == pytest.approx(40_000.0)

    def test_fraction_one_returns_swing_high(self):
        z = _bearish_zone(40_000, 50_000)
        assert z.price_at(1.0) == pytest.approx(50_000.0)

    def test_fraction_618(self):
        z = _bearish_zone(40_000, 50_000)
        # 40_000 + 10_000 × 0.618 = 46_180
        assert z.price_at(FIB_618) == pytest.approx(46_180.0)

    def test_fraction_786(self):
        z = _bearish_zone(40_000, 50_000)
        # 40_000 + 10_000 × 0.786 = 47_860
        assert z.price_at(FIB_786) == pytest.approx(47_860.0)

    def test_level_50(self):
        z = _bearish_zone(40_000, 50_000)
        assert z.level_50 == pytest.approx(45_000.0)


# ──────────────────────────────────────────────────────────────────────────────
# OTE zone boundaries
# ──────────────────────────────────────────────────────────────────────────────

class TestOTEBoundaries:
    def test_bullish_ote_bottom_is_at_786(self):
        # BULLISH retraces DOWN → deeper retrace (78.6%) gives lower price
        z = _bullish_zone(40_000, 50_000)
        assert z.ote_bottom == pytest.approx(42_140.0)   # price_at(0.786)

    def test_bullish_ote_top_is_at_618(self):
        z = _bullish_zone(40_000, 50_000)
        assert z.ote_top == pytest.approx(43_820.0)      # price_at(0.618)

    def test_bearish_ote_bottom_is_at_618(self):
        # BEARISH bounces UP → shallower retrace (61.8%) gives lower price
        z = _bearish_zone(40_000, 50_000)
        assert z.ote_bottom == pytest.approx(46_180.0)   # price_at(0.618)

    def test_bearish_ote_top_is_at_786(self):
        z = _bearish_zone(40_000, 50_000)
        assert z.ote_top == pytest.approx(47_860.0)      # price_at(0.786)

    def test_ote_top_always_greater_than_bottom(self):
        for z in (_bullish_zone(), _bearish_zone()):
            assert z.ote_top > z.ote_bottom


# ──────────────────────────────────────────────────────────────────────────────
# is_in_ote
# ──────────────────────────────────────────────────────────────────────────────

class TestIsInOTE:
    def test_price_inside_bullish_ote(self):
        z = _bullish_zone(40_000, 50_000)
        assert z.is_in_ote(43_000.0)   # between 42_140 and 43_820

    def test_price_above_bullish_ote_rejected(self):
        z = _bullish_zone(40_000, 50_000)
        assert not z.is_in_ote(44_000.0)   # above 43_820

    def test_price_below_bullish_ote_rejected(self):
        z = _bullish_zone(40_000, 50_000)
        assert not z.is_in_ote(41_000.0)   # below 42_140

    def test_price_at_ote_boundary_is_inside(self):
        z = _bullish_zone(40_000, 50_000)
        assert z.is_in_ote(z.ote_bottom)
        assert z.is_in_ote(z.ote_top)

    def test_price_inside_bearish_ote(self):
        z = _bearish_zone(40_000, 50_000)
        assert z.is_in_ote(47_000.0)   # between 46_180 and 47_860

    def test_price_below_bearish_ote_rejected(self):
        z = _bearish_zone(40_000, 50_000)
        assert not z.is_in_ote(45_000.0)


# ──────────────────────────────────────────────────────────────────────────────
# is_near_50
# ──────────────────────────────────────────────────────────────────────────────

class TestIsNear50:
    def test_price_at_exact_50_is_near(self):
        z = _bullish_zone(40_000, 50_000)
        assert z.is_near_50(45_000.0)

    def test_price_within_tolerance_is_near(self):
        z = _bullish_zone(40_000, 50_000)
        # 0.3% of 45_000 = 135 USDT tolerance
        assert z.is_near_50(45_100.0)   # +100 USDT < 135

    def test_price_outside_tolerance_is_not_near(self):
        z = _bullish_zone(40_000, 50_000)
        assert not z.is_near_50(45_200.0)   # +200 USDT > 135

    def test_custom_tolerance(self):
        z = _bullish_zone(40_000, 50_000)
        assert z.is_near_50(45_500.0, tolerance_pct=0.02)   # 1% of 45_000=450 > 500? No
        assert not z.is_near_50(45_500.0, tolerance_pct=0.01)  # 0.01*45000=450 < 500


# ──────────────────────────────────────────────────────────────────────────────
# detect_fib_zones
# ──────────────────────────────────────────────────────────────────────────────

def _low_pivot(bar: int, level: float, confirmed: int) -> PivotPoint:
    return PivotPoint(bar_index=bar, level=level, is_high=False, confirmed_at=confirmed)


def _high_pivot(bar: int, level: float, confirmed: int) -> PivotPoint:
    return PivotPoint(bar_index=bar, level=level, is_high=True, confirmed_at=confirmed)


class TestDetectFibZones:
    def test_low_then_high_produces_bullish_zone(self):
        pivots = [
            _low_pivot(0, 40_000, 5),
            _high_pivot(10, 50_000, 15),
        ]
        zones = detect_fib_zones(pivots)
        assert len(zones) == 1
        assert zones[0].direction == BULLISH
        assert zones[0].swing_low == 40_000
        assert zones[0].swing_high == 50_000

    def test_high_then_low_produces_bearish_zone(self):
        pivots = [
            _high_pivot(0, 50_000, 5),
            _low_pivot(10, 40_000, 15),
        ]
        zones = detect_fib_zones(pivots)
        assert len(zones) == 1
        assert zones[0].direction == BEARISH

    def test_formed_at_is_confirmed_at_of_last_pivot(self):
        pivots = [
            _low_pivot(0, 40_000, 5),
            _high_pivot(10, 50_000, 17),
        ]
        zones = detect_fib_zones(pivots)
        assert zones[0].formed_at == 17

    def test_three_pivots_produce_two_zones(self):
        pivots = [
            _low_pivot(0, 40_000, 5),
            _high_pivot(10, 50_000, 15),
            _low_pivot(20, 45_000, 25),
        ]
        zones = detect_fib_zones(pivots)
        assert len(zones) == 2
        assert zones[0].direction == BULLISH
        assert zones[1].direction == BEARISH

    def test_empty_pivots_returns_empty_list(self):
        assert detect_fib_zones([]) == []

    def test_single_pivot_returns_empty_list(self):
        assert detect_fib_zones([_low_pivot(0, 40_000, 5)]) == []

    def test_zones_are_in_chronological_order(self):
        pivots = [
            _low_pivot(0, 38_000, 5),
            _high_pivot(10, 50_000, 15),
            _low_pivot(20, 44_000, 25),
            _high_pivot(30, 52_000, 35),
        ]
        zones = detect_fib_zones(pivots)
        assert len(zones) == 3
        formed_ats = [z.formed_at for z in zones]
        assert formed_ats == sorted(formed_ats)


# ──────────────────────────────────────────────────────────────────────────────
# get_latest_fib_zone
# ──────────────────────────────────────────────────────────────────────────────

class TestGetLatestFibZone:
    def _zones(self):
        return [
            FibZone(50_000, 40_000, BULLISH, 8, 0, formed_at=10),
            FibZone(50_000, 44_000, BEARISH, 0, 20, formed_at=22),
            FibZone(55_000, 44_000, BULLISH, 30, 20, formed_at=32),
        ]

    def test_returns_latest_bullish_zone(self):
        z = get_latest_fib_zone(self._zones(), at_bar=50, direction=BULLISH)
        assert z is not None
        assert z.formed_at == 32

    def test_returns_latest_bearish_zone(self):
        z = get_latest_fib_zone(self._zones(), at_bar=50, direction=BEARISH)
        assert z is not None
        assert z.formed_at == 22

    def test_respects_at_bar_filter(self):
        # Only zones formed at or before bar 15 are eligible
        z = get_latest_fib_zone(self._zones(), at_bar=15, direction=BULLISH)
        assert z is not None
        assert z.formed_at == 10

    def test_returns_none_when_no_zone_matches(self):
        z = get_latest_fib_zone(self._zones(), at_bar=5, direction=BULLISH)
        assert z is None

    def test_returns_none_on_empty_list(self):
        assert get_latest_fib_zone([], at_bar=100, direction=BULLISH) is None
