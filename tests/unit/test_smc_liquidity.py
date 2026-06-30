"""
Unit tests for zeus.strategy.smc.liquidity.

Covers level creation from pivots, sweep detection logic (BSL/SSL),
active-level filtering, and last_sweep helper.
"""
import pytest

from zeus.strategy.smc.liquidity import (
    LiqType,
    LiquidityLevel,
    LiquiditySweep,
    detect_liquidity_levels,
    detect_sweeps,
    get_active_levels,
    last_sweep,
)
from zeus.strategy.smc.pivot import BEARISH, BULLISH, PivotPoint


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _high(bar: int, level: float, confirmed: int = None) -> PivotPoint:
    return PivotPoint(bar_index=bar, level=level, is_high=True,
                      confirmed_at=confirmed if confirmed is not None else bar + 5)


def _low(bar: int, level: float, confirmed: int = None) -> PivotPoint:
    return PivotPoint(bar_index=bar, level=level, is_high=False,
                      confirmed_at=confirmed if confirmed is not None else bar + 5)


# ──────────────────────────────────────────────────────────────────────────────
# detect_liquidity_levels
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectLiquidityLevels:
    def test_high_pivot_creates_bsl(self):
        levels = detect_liquidity_levels([_high(10, 50_000)])
        assert len(levels) == 1
        assert levels[0].liq_type == LiqType.BSL
        assert levels[0].level == 50_000

    def test_low_pivot_creates_ssl(self):
        levels = detect_liquidity_levels([_low(5, 40_000)])
        assert len(levels) == 1
        assert levels[0].liq_type == LiqType.SSL
        assert levels[0].level == 40_000

    def test_formed_at_matches_pivot_confirmed_at(self):
        levels = detect_liquidity_levels([_high(10, 50_000, confirmed=15)])
        assert levels[0].formed_at == 15

    def test_bar_index_matches_pivot_bar(self):
        levels = detect_liquidity_levels([_high(10, 50_000, confirmed=15)])
        assert levels[0].bar_index == 10

    def test_multiple_pivots_produce_multiple_levels(self):
        pivots = [_high(0, 50_000), _low(5, 40_000), _high(10, 52_000)]
        levels = detect_liquidity_levels(pivots)
        assert len(levels) == 3
        assert levels[0].liq_type == LiqType.BSL
        assert levels[1].liq_type == LiqType.SSL
        assert levels[2].liq_type == LiqType.BSL

    def test_empty_pivots_returns_empty(self):
        assert detect_liquidity_levels([]) == []


# ──────────────────────────────────────────────────────────────────────────────
# detect_sweeps — BSL
# ──────────────────────────────────────────────────────────────────────────────

class TestBSLSweep:
    """BSL is swept when high > level AND close < level (wick above, close below)."""

    def _bsl(self, level=50_000.0, formed_at=0) -> LiquidityLevel:
        return LiquidityLevel(bar_index=0, level=level,
                              liq_type=LiqType.BSL, formed_at=formed_at)

    def test_bsl_sweep_detected(self):
        lv = self._bsl(50_000, formed_at=0)
        highs  = [51_000.0]   # wick above
        lows   = [49_000.0]
        closes = [49_500.0]   # close below level
        sweeps = detect_sweeps(highs, lows, closes, [lv])
        assert len(sweeps) == 1
        assert sweeps[0].liq_type == LiqType.BSL
        assert sweeps[0].direction == BEARISH

    def test_high_above_but_close_above_no_sweep(self):
        lv = self._bsl(50_000, formed_at=0)
        sweeps = detect_sweeps([51_000], [49_500], [50_500], [lv])
        assert sweeps == []

    def test_close_below_but_high_not_above_no_sweep(self):
        lv = self._bsl(50_000, formed_at=0)
        sweeps = detect_sweeps([49_900], [48_000], [48_500], [lv])
        assert sweeps == []

    def test_bsl_level_can_only_be_swept_once(self):
        lv = self._bsl(50_000, formed_at=0)
        # Two consecutive bars both satisfy sweep condition
        highs  = [51_000, 51_000]
        lows   = [49_000, 49_000]
        closes = [49_500, 49_500]
        sweeps = detect_sweeps(highs, lows, closes, [lv])
        assert len(sweeps) == 1   # only the first bar


# ──────────────────────────────────────────────────────────────────────────────
# detect_sweeps — SSL
# ──────────────────────────────────────────────────────────────────────────────

class TestSSLSweep:
    """SSL is swept when low < level AND close > level (wick below, close above)."""

    def _ssl(self, level=40_000.0, formed_at=0) -> LiquidityLevel:
        return LiquidityLevel(bar_index=1, level=level,
                              liq_type=LiqType.SSL, formed_at=formed_at)

    def test_ssl_sweep_detected(self):
        lv = self._ssl(40_000, formed_at=0)
        highs  = [41_000.0]
        lows   = [39_000.0]   # wick below
        closes = [40_500.0]   # close above level
        sweeps = detect_sweeps(highs, lows, closes, [lv])
        assert len(sweeps) == 1
        assert sweeps[0].liq_type == LiqType.SSL
        assert sweeps[0].direction == BULLISH

    def test_low_below_but_close_below_no_sweep(self):
        lv = self._ssl(40_000, formed_at=0)
        sweeps = detect_sweeps([40_500], [39_000], [39_500], [lv])
        assert sweeps == []

    def test_close_above_but_low_not_below_no_sweep(self):
        lv = self._ssl(40_000, formed_at=0)
        sweeps = detect_sweeps([41_000], [40_100], [40_800], [lv])
        assert sweeps == []

    def test_ssl_level_can_only_be_swept_once(self):
        lv = self._ssl(40_000, formed_at=0)
        highs  = [41_000, 41_000]
        lows   = [39_000, 39_000]
        closes = [40_500, 40_500]
        sweeps = detect_sweeps(highs, lows, closes, [lv])
        assert len(sweeps) == 1


# ──────────────────────────────────────────────────────────────────────────────
# formed_at gate
# ──────────────────────────────────────────────────────────────────────────────

class TestFormedAtGate:
    def test_level_not_yet_formed_is_not_swept(self):
        # formed_at=5 but bars only go to index 2
        lv = LiquidityLevel(bar_index=0, level=50_000,
                            liq_type=LiqType.BSL, formed_at=5)
        highs  = [51_000, 51_000, 51_000]
        lows   = [49_000, 49_000, 49_000]
        closes = [49_500, 49_500, 49_500]
        sweeps = detect_sweeps(highs, lows, closes, [lv])
        assert sweeps == []

    def test_level_formed_exactly_at_bar_is_swept(self):
        # formed_at=0, sweep attempt at bar 0
        lv = LiquidityLevel(bar_index=0, level=50_000,
                            liq_type=LiqType.BSL, formed_at=0)
        sweeps = detect_sweeps([51_000], [49_000], [49_500], [lv])
        assert len(sweeps) == 1


# ──────────────────────────────────────────────────────────────────────────────
# get_active_levels
# ──────────────────────────────────────────────────────────────────────────────

class TestGetActiveLevels:
    def _levels(self):
        return [
            LiquidityLevel(0, 50_000, LiqType.BSL, formed_at=5),
            LiquidityLevel(10, 40_000, LiqType.SSL, formed_at=15),
            LiquidityLevel(20, 52_000, LiqType.BSL, formed_at=25),
        ]

    def test_returns_only_formed_levels(self):
        active = get_active_levels(self._levels(), at_bar=10)
        assert len(active) == 1
        assert active[0].level == 50_000

    def test_all_levels_visible_at_late_bar(self):
        active = get_active_levels(self._levels(), at_bar=30)
        assert len(active) == 3

    def test_swept_level_excluded(self):
        sweep = LiquiditySweep(bar_index=6, level=50_000,
                               liq_type=LiqType.BSL, direction=BEARISH)
        active = get_active_levels(self._levels(), at_bar=30, sweeps=[sweep])
        prices = [lv.level for lv in active]
        assert 50_000 not in prices

    def test_future_sweep_does_not_exclude_level(self):
        # Sweep happens at bar 100, we query at bar 30 — level should still be active
        sweep = LiquiditySweep(bar_index=100, level=50_000,
                               liq_type=LiqType.BSL, direction=BEARISH)
        active = get_active_levels(self._levels(), at_bar=30, sweeps=[sweep])
        prices = [lv.level for lv in active]
        assert 50_000 in prices


# ──────────────────────────────────────────────────────────────────────────────
# last_sweep
# ──────────────────────────────────────────────────────────────────────────────

class TestLastSweep:
    def _sweeps(self):
        return [
            LiquiditySweep(5,  50_000, LiqType.BSL, BEARISH),
            LiquiditySweep(15, 40_000, LiqType.SSL, BULLISH),
            LiquiditySweep(25, 52_000, LiqType.BSL, BEARISH),
        ]

    def test_returns_most_recent_sweep(self):
        s = last_sweep(self._sweeps(), at_bar=30)
        assert s is not None
        assert s.bar_index == 25

    def test_respects_at_bar_cutoff(self):
        s = last_sweep(self._sweeps(), at_bar=20)
        assert s is not None
        assert s.bar_index == 15

    def test_direction_filter_bullish(self):
        s = last_sweep(self._sweeps(), at_bar=30, direction=BULLISH)
        assert s is not None
        assert s.direction == BULLISH
        assert s.bar_index == 15

    def test_direction_filter_bearish(self):
        s = last_sweep(self._sweeps(), at_bar=30, direction=BEARISH)
        assert s is not None
        assert s.bar_index == 25

    def test_returns_none_when_no_match(self):
        assert last_sweep(self._sweeps(), at_bar=1) is None

    def test_returns_none_on_empty_list(self):
        assert last_sweep([], at_bar=100) is None
