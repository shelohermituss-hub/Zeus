"""
Tests for zeus.strategy.supply_demand.fibonacci

Swing used throughout (bullish): low=1900, high=2100, range=200
    0.236 level = 2100 − 200×0.236 = 2052.8
    0.382 level = 2100 − 200×0.382 = 2023.6
    0.500 level = 2100 − 200×0.500 = 2000.0
    0.618 level = 2100 − 200×0.618 = 1976.4
    0.705 level = 2100 − 200×0.705 = 1959.0
    0.764 level = 2100 − 200×0.764 = 1947.2
    0.786 level = 2100 − 200×0.786 = 1842.8  ← wait: 2100−157.2 = 1942.8
    1.000 level = 2100 − 200×1.000 = 1900.0

Bearish swing: low=1900, high=2100, is_bullish=False
    0.764 level = 1900 + 200×0.764 = 2052.8
    0.500 level = 1900 + 200×0.500 = 2000.0  (same equilibrium)
"""
from __future__ import annotations

import pytest
import pandas as pd

from zeus.strategy.supply_demand.pivot_candle import PivotSide, analyze_candle
from zeus.strategy.supply_demand.zone_detector import SDZone, ZoneDetector
from zeus.strategy.supply_demand.fibonacci import (
    FibLevels,
    FibConfluence,
    compute_fib_levels,
    zone_fib_confluence,
    fib_level_price,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

SWING_LOW  = 1900.0
SWING_HIGH = 2100.0
RANGE      = SWING_HIGH - SWING_LOW   # 200.0

_TS = pd.Timestamp("2026-06-01 09:00")


def _make_bull_fibs() -> FibLevels:
    return compute_fib_levels(SWING_LOW, SWING_HIGH, is_bullish=True)


def _make_bear_fibs() -> FibLevels:
    return compute_fib_levels(SWING_LOW, SWING_HIGH, is_bullish=False)


def _make_zone(side: PivotSide, bottom: float, top: float, wick_extreme: float) -> SDZone:
    """Build a minimal SDZone for confluence testing (no real pivot candle needed)."""
    pc = analyze_candle(
        _TS,
        o=bottom, h=top + 0.001, l=wick_extreme, c=top,
    )
    from zeus.strategy.supply_demand.zone_detector import ZoneScore
    score = ZoneScore(bos=1.0, impulse=1.0, time=1.0, fresh=2.0, sweep=0.0)
    return SDZone(
        side=side,
        zone_top=top,
        zone_bottom=bottom,
        wick_extreme=wick_extreme,
        pivot=pc,
        pivot_bar=0,
        formed_at=_TS,
        bos_bar=5,
        bos_level=top + 1.0,
        base_candles=3,
        score=score,
    )


# ── compute_fib_levels — bullish ──────────────────────────────────────────────

class TestComputeFibLevelsBullish:
    def test_stores_swing_low_high(self):
        f = _make_bull_fibs()
        assert f.swing_low  == pytest.approx(SWING_LOW)
        assert f.swing_high == pytest.approx(SWING_HIGH)

    def test_is_bullish_true(self):
        assert _make_bull_fibs().is_bullish is True

    def test_level_0000_is_swing_high(self):
        # For bullish, 0.000 retracement = start of pullback = swing_high
        f = _make_bull_fibs()
        assert f.level_0000 == pytest.approx(SWING_HIGH)

    def test_level_1000_is_swing_low(self):
        # Full retracement = swing_low
        f = _make_bull_fibs()
        assert f.level_1000 == pytest.approx(SWING_LOW)

    def test_equilibrium_0500(self):
        f = _make_bull_fibs()
        assert f.level_0500 == pytest.approx(2000.0)

    def test_level_0764_ote_core(self):
        # 2100 − 200×0.764 = 1947.2
        f = _make_bull_fibs()
        assert f.level_0764 == pytest.approx(1947.2, rel=1e-6)

    def test_level_0618(self):
        # 2100 − 200×0.618 = 1976.4
        f = _make_bull_fibs()
        assert f.level_0618 == pytest.approx(1976.4, rel=1e-6)

    def test_level_0705_ote_low(self):
        # 2100 − 200×0.705 = 1959.0
        f = _make_bull_fibs()
        assert f.level_0705 == pytest.approx(1959.0, rel=1e-6)

    def test_level_0786_ote_high(self):
        # 2100 − 200×0.786 = 1942.8
        f = _make_bull_fibs()
        assert f.level_0786 == pytest.approx(1942.8, rel=1e-6)

    def test_levels_descend_for_bullish(self):
        # For bullish swing, higher ratio = lower price (deeper retracement)
        f = _make_bull_fibs()
        assert f.level_0236 > f.level_0382 > f.level_0500 > f.level_0618 > f.level_0705 > f.level_0764 > f.level_0786

    def test_swing_range(self):
        f = _make_bull_fibs()
        assert f.swing_range == pytest.approx(RANGE)

    def test_ote_property(self):
        f = _make_bull_fibs()
        assert f.ote == pytest.approx(f.level_0764)

    def test_equilibrium_property(self):
        f = _make_bull_fibs()
        assert f.equilibrium == pytest.approx(f.level_0500)

    def test_ote_low_is_lower_price(self):
        # For bullish: 0.786 level is deeper retracement → lower price
        f = _make_bull_fibs()
        assert f.ote_low  == pytest.approx(f.level_0786)
        assert f.ote_high == pytest.approx(f.level_0705)


# ── compute_fib_levels — bearish ──────────────────────────────────────────────

class TestComputeFibLevelsBearish:
    def test_is_bullish_false(self):
        assert _make_bear_fibs().is_bullish is False

    def test_level_0000_is_swing_low_for_bearish(self):
        # For bearish, 0.000 = start of pullback = swing_low
        f = _make_bear_fibs()
        assert f.level_0000 == pytest.approx(SWING_LOW)

    def test_level_1000_is_swing_high_for_bearish(self):
        f = _make_bear_fibs()
        assert f.level_1000 == pytest.approx(SWING_HIGH)

    def test_equilibrium_same_price_regardless_of_direction(self):
        # Equilibrium is always the midpoint of the range
        bull = _make_bull_fibs()
        bear = _make_bear_fibs()
        assert bull.equilibrium == pytest.approx(bear.equilibrium)

    def test_bearish_0764_level(self):
        # 1900 + 200×0.764 = 2052.8
        f = _make_bear_fibs()
        assert f.level_0764 == pytest.approx(2052.8, rel=1e-6)

    def test_levels_ascend_for_bearish(self):
        # For bearish, higher ratio = higher price (deeper retracement upward)
        f = _make_bear_fibs()
        assert f.level_0236 < f.level_0382 < f.level_0500 < f.level_0618 < f.level_0705 < f.level_0764 < f.level_0786

    def test_bearish_ote_low_is_0705_level(self):
        # For bearish: 0.705 level is lower price than 0.786
        f = _make_bear_fibs()
        assert f.ote_low  == pytest.approx(f.level_0705)
        assert f.ote_high == pytest.approx(f.level_0786)


# ── fib_level_price ────────────────────────────────────────────────────────────

class TestFibLevelPrice:
    def test_arbitrary_ratio_bullish(self):
        f = _make_bull_fibs()
        # ratio 0.33 → 2100 − 200×0.33 = 2034.0
        assert fib_level_price(f, 0.33) == pytest.approx(2034.0, rel=1e-6)

    def test_arbitrary_ratio_bearish(self):
        f = _make_bear_fibs()
        # ratio 0.33 → 1900 + 200×0.33 = 1966.0
        assert fib_level_price(f, 0.33) == pytest.approx(1966.0, rel=1e-6)

    def test_zero_ratio_returns_level_0000(self):
        f = _make_bull_fibs()
        assert fib_level_price(f, 0.0) == pytest.approx(f.level_0000)

    def test_one_ratio_returns_level_1000(self):
        f = _make_bull_fibs()
        assert fib_level_price(f, 1.0) == pytest.approx(f.level_1000)


# ── zone_fib_confluence — OTE ──────────────────────────────────────────────────

class TestZoneFibConfluenceOTE:
    """Bullish swing: OTE golden pocket is between level_0786=1942.8 and level_0705=1959.0."""

    def test_demand_zone_inside_ote_has_ote_true(self):
        fibs = _make_bull_fibs()
        # Zone fully inside golden pocket [1942.8, 1959.0]
        zone = _make_zone(PivotSide.DEMAND, bottom=1945.0, top=1955.0, wick_extreme=1940.0)
        conf = zone_fib_confluence(zone, fibs)
        assert conf.has_ote is True

    def test_demand_zone_overlapping_ote_bottom_has_ote_true(self):
        fibs = _make_bull_fibs()
        # Zone bottom is below ote_low=1942.8, top is inside pocket
        zone = _make_zone(PivotSide.DEMAND, bottom=1935.0, top=1950.0, wick_extreme=1930.0)
        conf = zone_fib_confluence(zone, fibs)
        assert conf.has_ote is True

    def test_demand_zone_above_ote_has_ote_false(self):
        fibs = _make_bull_fibs()
        # Zone is above the golden pocket (closer to 0.382 area ~2023)
        zone = _make_zone(PivotSide.DEMAND, bottom=1975.0, top=1985.0, wick_extreme=1970.0)
        conf = zone_fib_confluence(zone, fibs)
        assert conf.has_ote is False

    def test_demand_zone_below_ote_has_ote_false(self):
        fibs = _make_bull_fibs()
        # Zone is below the golden pocket (past full retrace area)
        zone = _make_zone(PivotSide.DEMAND, bottom=1910.0, top=1930.0, wick_extreme=1905.0)
        conf = zone_fib_confluence(zone, fibs)
        assert conf.has_ote is False

    def test_bearish_supply_zone_in_ote_has_ote_true(self):
        fibs = _make_bear_fibs()
        # Bearish OTE pocket: [level_0705=1959.0, level_0786=2052.8+something]
        # level_0705 = 1900 + 200×0.705 = 2041.0
        # level_0786 = 1900 + 200×0.786 = 2057.2
        ote_low  = fibs.ote_low   # min(level_0705, level_0786)
        ote_high = fibs.ote_high  # max(level_0705, level_0786)
        mid = (ote_low + ote_high) / 2.0
        zone = _make_zone(PivotSide.SUPPLY, bottom=mid - 2.0, top=mid + 2.0, wick_extreme=mid + 5.0)
        conf = zone_fib_confluence(zone, fibs)
        assert conf.has_ote is True


# ── zone_fib_confluence — discount / premium ──────────────────────────────────

class TestZoneFibConfluenceDiscountPremium:
    """Equilibrium = 2000.0 for both directions."""

    def test_demand_below_equilibrium_is_in_discount(self):
        fibs = _make_bull_fibs()
        # Zone around 1960 — below equilibrium 2000
        zone = _make_zone(PivotSide.DEMAND, bottom=1955.0, top=1965.0, wick_extreme=1950.0)
        conf = zone_fib_confluence(zone, fibs)
        assert conf.is_in_discount is True
        assert conf.is_in_premium  is False

    def test_demand_above_equilibrium_not_in_discount(self):
        fibs = _make_bull_fibs()
        # Zone around 2030 — above equilibrium 2000 (premium territory)
        zone = _make_zone(PivotSide.DEMAND, bottom=2025.0, top=2035.0, wick_extreme=2020.0)
        conf = zone_fib_confluence(zone, fibs)
        assert conf.is_in_discount is False

    def test_supply_above_equilibrium_is_in_premium(self):
        fibs = _make_bull_fibs()
        # Zone around 2040 — above equilibrium 2000
        zone = _make_zone(PivotSide.SUPPLY, bottom=2035.0, top=2045.0, wick_extreme=2050.0)
        conf = zone_fib_confluence(zone, fibs)
        assert conf.is_in_premium  is True
        assert conf.is_in_discount is False

    def test_supply_below_equilibrium_not_in_premium(self):
        fibs = _make_bull_fibs()
        # Zone around 1960 — below equilibrium
        zone = _make_zone(PivotSide.SUPPLY, bottom=1955.0, top=1965.0, wick_extreme=1970.0)
        conf = zone_fib_confluence(zone, fibs)
        assert conf.is_in_premium is False


# ── zone_fib_confluence — score_bonus ─────────────────────────────────────────

class TestZoneFibConfluenceScoreBonus:
    def test_ote_with_discount_gives_2_0(self):
        fibs = _make_bull_fibs()
        # Zone at OTE level + below equilibrium → 1.5 + 0.5 = 2.0
        zone = _make_zone(PivotSide.DEMAND, bottom=1945.0, top=1955.0, wick_extreme=1940.0)
        conf = zone_fib_confluence(zone, fibs)
        assert conf.has_ote is True
        assert conf.is_in_discount is True
        assert conf.score_bonus == pytest.approx(2.0)

    def test_ote_without_discount_gives_1_5(self):
        fibs = _make_bull_fibs()
        # OTE zone but midpoint above equilibrium 2000 — impossible in standard bullish move
        # Let's use a bearish OTE (supply zone) without premium condition
        # Supply below equilibrium → no premium bonus
        fibs2 = _make_bull_fibs()
        # Force a supply zone in OTE area but midpoint below equilibrium (unusual case)
        # We test the logic directly: OTE only → 1.5
        zone = _make_zone(PivotSide.DEMAND, bottom=1945.0, top=1955.0, wick_extreme=1940.0)
        conf = zone_fib_confluence(zone, fibs2)
        # This zone IS in discount, so it will be 2.0 — test the non-discount OTE case
        # Use a supply zone at OTE but below equilibrium
        supply_zone = _make_zone(PivotSide.SUPPLY, bottom=1945.0, top=1955.0, wick_extreme=1960.0)
        conf2 = zone_fib_confluence(supply_zone, fibs2)
        assert conf2.has_ote is True
        assert conf2.is_in_premium is False   # supply midpoint ~1950 < equilibrium 2000
        assert conf2.score_bonus == pytest.approx(1.5)

    def test_at_0618_without_ote_gives_0_5_plus_context(self):
        fibs = _make_bull_fibs()
        # 0.618 level for bullish = 1976.4
        # Zone centered at 1976.4, below equilibrium 2000 → 0.5 (0.618) + 0.5 (discount) = 1.0
        zone = _make_zone(PivotSide.DEMAND, bottom=1974.0, top=1978.0, wick_extreme=1970.0)
        conf = zone_fib_confluence(zone, fibs)
        assert conf.has_ote is False
        assert conf.nearest_level == "0.618"
        assert conf.is_in_discount is True
        assert conf.score_bonus == pytest.approx(1.0)

    def test_no_confluence_gives_low_bonus(self):
        fibs = _make_bull_fibs()
        # Zone at 0.236 level (2052.8) — above equilibrium, no OTE, not 0.618 nearest
        zone = _make_zone(PivotSide.DEMAND, bottom=2050.0, top=2055.0, wick_extreme=2047.0)
        conf = zone_fib_confluence(zone, fibs)
        assert conf.has_ote is False
        assert conf.is_in_discount is False
        assert conf.score_bonus == pytest.approx(0.0)

    def test_score_bonus_capped_at_2_0(self):
        fibs = _make_bull_fibs()
        zone = _make_zone(PivotSide.DEMAND, bottom=1945.0, top=1955.0, wick_extreme=1940.0)
        conf = zone_fib_confluence(zone, fibs)
        assert conf.score_bonus <= 2.0


# ── zone_fib_confluence — nearest level ───────────────────────────────────────

class TestNearestLevel:
    def test_nearest_level_at_0764(self):
        fibs = _make_bull_fibs()
        # Zone midpoint exactly at 0.764 level (1947.2)
        zone = _make_zone(PivotSide.DEMAND, bottom=1946.0, top=1948.4, wick_extreme=1943.0)
        conf = zone_fib_confluence(zone, fibs)
        assert conf.nearest_level == "0.764"

    def test_nearest_level_at_0500(self):
        fibs = _make_bull_fibs()
        # Midpoint at equilibrium 2000
        zone = _make_zone(PivotSide.DEMAND, bottom=1998.0, top=2002.0, wick_extreme=1995.0)
        conf = zone_fib_confluence(zone, fibs)
        assert conf.nearest_level == "0.500"

    def test_nearest_distance_is_non_negative(self):
        fibs = _make_bull_fibs()
        zone = _make_zone(PivotSide.DEMAND, bottom=1945.0, top=1955.0, wick_extreme=1940.0)
        conf = zone_fib_confluence(zone, fibs)
        assert conf.nearest_distance >= 0.0
