"""
Tests for zeus.strategy.supply_demand.wyckoff

Candle layout used in demand pattern tests:

    bar 0-3 : tight accumulation range [1.100, 1.110]
        avg candle range ≈ 0.008 ; accum_rng = 0.010 ≤ 3 × 0.008 = 0.024 ✓
    bar 4   : Spring — low=1.095 < 1.100, close=1.106 > 1.100
    bar 5   : MSS    — close=1.113 > 1.110

Supply mirror:
    bar 0-3 : same tight accumulation [1.100, 1.110]
    bar 4   : Upthrust — high=1.115 > 1.110, close=1.104 < 1.110
    bar 5   : MSS      — close=1.097 < 1.100
"""
from __future__ import annotations

import pandas as pd
import pytest

from zeus.strategy.supply_demand.pivot_candle import PivotSide
from zeus.strategy.supply_demand.wyckoff import WyckoffDetector, WyckoffPattern

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df(candles: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    idx = pd.date_range("2026-06-01 09:00", periods=len(candles), freq="1min")
    return pd.DataFrame(
        [{"open": o, "high": h, "low": l, "close": c} for o, h, l, c in candles],
        index=idx,
    )


# 4 tight accumulation bars, then Spring, then MSS
#
# Bar 0 sets accum_low = 1.100.  Bars 1-3 stay strictly ABOVE 1.100 so the
# algorithm cannot identify an early Spring at bar 1, 2, or 3.
# Only bar 4 (low=1.095) violates the accumulation floor.
_DEMAND_CANDLES = [
    (1.103, 1.108, 1.100, 1.106),   # 0 accum  low=1.100  ← floor
    (1.105, 1.109, 1.102, 1.107),   # 1 accum  low=1.102
    (1.104, 1.110, 1.101, 1.105),   # 2 accum  low=1.101
    (1.106, 1.109, 1.103, 1.104),   # 3 accum  low=1.103 > 1.100 — NOT a Spring
    (1.104, 1.108, 1.095, 1.106),   # 4 Spring low=1.095 < 1.100, close=1.106 > 1.100
    (1.108, 1.115, 1.107, 1.113),   # 5 MSS    close=1.113 > 1.110
]

_SUPPLY_CANDLES = [
    (1.104, 1.108, 1.101, 1.105),   # 0 accum
    (1.105, 1.109, 1.102, 1.106),   # 1 accum
    (1.105, 1.110, 1.101, 1.104),   # 2 accum
    (1.106, 1.109, 1.100, 1.103),   # 3 accum
    (1.106, 1.115, 1.103, 1.104),   # 4 Upthrust high=1.115 > 1.110, close=1.104 < 1.110
    (1.103, 1.104, 1.095, 1.097),   # 5 MSS    close=1.097 < 1.100
]


# ── TestDetectDemand ──────────────────────────────────────────────────────────

class TestDetectDemand:
    def test_detects_demand_pattern(self):
        df = _make_df(_DEMAND_CANDLES)
        det = WyckoffDetector()
        p = det.detect(df, PivotSide.DEMAND, end_idx=len(df))
        assert p is not None
        assert p.side == PivotSide.DEMAND

    def test_spring_bar_index(self):
        df = _make_df(_DEMAND_CANDLES)
        p = WyckoffDetector().detect(df, PivotSide.DEMAND, end_idx=len(df))
        assert p is not None
        assert p.manip_bar == 4

    def test_mss_bar_index(self):
        df = _make_df(_DEMAND_CANDLES)
        p = WyckoffDetector().detect(df, PivotSide.DEMAND, end_idx=len(df))
        assert p is not None
        assert p.mss_bar == 5

    def test_accum_boundaries(self):
        df = _make_df(_DEMAND_CANDLES)
        p = WyckoffDetector().detect(df, PivotSide.DEMAND, end_idx=len(df))
        assert p is not None
        assert p.accum_high == pytest.approx(1.110, abs=1e-5)
        assert p.accum_low  == pytest.approx(1.100, abs=1e-5)

    def test_manip_extreme_is_spring_low(self):
        df = _make_df(_DEMAND_CANDLES)
        p = WyckoffDetector().detect(df, PivotSide.DEMAND, end_idx=len(df))
        assert p is not None
        assert p.manip_extreme == pytest.approx(1.095, abs=1e-5)

    def test_mss_close(self):
        df = _make_df(_DEMAND_CANDLES)
        p = WyckoffDetector().detect(df, PivotSide.DEMAND, end_idx=len(df))
        assert p is not None
        assert p.mss_close == pytest.approx(1.113, abs=1e-5)

    def test_formed_at_is_mss_timestamp(self):
        df = _make_df(_DEMAND_CANDLES)
        p = WyckoffDetector().detect(df, PivotSide.DEMAND, end_idx=len(df))
        assert p is not None
        assert p.formed_at == df.index[5]

    def test_entry_price_equals_mss_close(self):
        df = _make_df(_DEMAND_CANDLES)
        p = WyckoffDetector().detect(df, PivotSide.DEMAND, end_idx=len(df))
        assert p is not None
        assert p.entry_price == p.mss_close

    def test_stop_loss_is_spring_low(self):
        df = _make_df(_DEMAND_CANDLES)
        p = WyckoffDetector().detect(df, PivotSide.DEMAND, end_idx=len(df))
        assert p is not None
        assert p.stop_loss_price == p.manip_extreme

    def test_score_in_range(self):
        df = _make_df(_DEMAND_CANDLES)
        p = WyckoffDetector().detect(df, PivotSide.DEMAND, end_idx=len(df))
        assert p is not None
        assert 0.0 <= p.score <= 10.0

    def test_accum_bars_count(self):
        df = _make_df(_DEMAND_CANDLES)
        p = WyckoffDetector().detect(df, PivotSide.DEMAND, end_idx=len(df))
        assert p is not None
        assert p.accum_bars >= 3  # at least min_accum_bars


# ── TestDetectSupply ──────────────────────────────────────────────────────────

class TestDetectSupply:
    def test_detects_supply_pattern(self):
        df = _make_df(_SUPPLY_CANDLES)
        p = WyckoffDetector().detect(df, PivotSide.SUPPLY, end_idx=len(df))
        assert p is not None
        assert p.side == PivotSide.SUPPLY

    def test_upthrust_bar_index(self):
        df = _make_df(_SUPPLY_CANDLES)
        p = WyckoffDetector().detect(df, PivotSide.SUPPLY, end_idx=len(df))
        assert p is not None
        assert p.manip_bar == 4

    def test_mss_bar_index(self):
        df = _make_df(_SUPPLY_CANDLES)
        p = WyckoffDetector().detect(df, PivotSide.SUPPLY, end_idx=len(df))
        assert p is not None
        assert p.mss_bar == 5

    def test_manip_extreme_is_upthrust_high(self):
        df = _make_df(_SUPPLY_CANDLES)
        p = WyckoffDetector().detect(df, PivotSide.SUPPLY, end_idx=len(df))
        assert p is not None
        assert p.manip_extreme == pytest.approx(1.115, abs=1e-5)

    def test_mss_close_below_accum_low(self):
        df = _make_df(_SUPPLY_CANDLES)
        p = WyckoffDetector().detect(df, PivotSide.SUPPLY, end_idx=len(df))
        assert p is not None
        assert p.mss_close < p.accum_low

    def test_score_in_range(self):
        df = _make_df(_SUPPLY_CANDLES)
        p = WyckoffDetector().detect(df, PivotSide.SUPPLY, end_idx=len(df))
        assert p is not None
        assert 0.0 <= p.score <= 10.0


# ── TestDetectReturnsNone ─────────────────────────────────────────────────────

class TestDetectReturnsNone:
    def test_doji_side_returns_none(self):
        df = _make_df(_DEMAND_CANDLES)
        p = WyckoffDetector().detect(df, PivotSide.DOJI, end_idx=len(df))
        assert p is None

    def test_too_few_bars_returns_none(self):
        # min_accum_bars=3, need at least 5 bars (3 accum + spring + mss)
        candles = _DEMAND_CANDLES[:4]   # only 4 bars
        df = _make_df(candles)
        p = WyckoffDetector().detect(df, PivotSide.DEMAND, end_idx=len(df))
        assert p is None

    def test_no_spring_below_accum_returns_none(self):
        # All bars stay inside the accumulation range — no Spring.
        # All lows ≥ 1.100 (the accumulation floor) so no Spring can fire.
        candles = [
            (1.103, 1.108, 1.100, 1.106),  # accum  low=1.100  ← floor
            (1.105, 1.109, 1.102, 1.107),  # accum
            (1.104, 1.110, 1.101, 1.105),  # accum
            (1.106, 1.109, 1.103, 1.104),  # accum
            (1.105, 1.109, 1.101, 1.107),  # inside range — low=1.101 > 1.100
            (1.106, 1.110, 1.102, 1.108),  # inside range — not MSS above 1.110
        ]
        df = _make_df(candles)
        p = WyckoffDetector().detect(df, PivotSide.DEMAND, end_idx=len(df))
        assert p is None

    def test_spring_present_but_no_mss_returns_none(self):
        # Spring occurs but price never closes above accum_high within mss_lookback.
        candles = [
            (1.103, 1.108, 1.100, 1.106),  # accum  low=1.100 ← floor
            (1.105, 1.109, 1.102, 1.107),  # accum
            (1.104, 1.110, 1.101, 1.105),  # accum
            (1.106, 1.109, 1.103, 1.104),  # accum  low=1.103 > 1.100
            (1.104, 1.108, 1.095, 1.106),  # Spring ✓
            (1.106, 1.109, 1.103, 1.108),  # close=1.108 < accum_high=1.110 — NO MSS
        ]
        df = _make_df(candles)
        p = WyckoffDetector(mss_lookback=1).detect(df, PivotSide.DEMAND, end_idx=len(df))
        assert p is None

    def test_wide_accumulation_returns_none(self):
        # Trending bars — accumulation range >> mean candle range
        candles = [
            (1.080, 1.090, 1.079, 1.088),  # range 0.011
            (1.088, 1.100, 1.087, 1.098),  # range 0.013
            (1.098, 1.108, 1.097, 1.106),  # range 0.011
            (1.106, 1.115, 1.105, 1.113),  # range 0.010 — accum_rng=0.036 >> 3×0.011=0.033
            (1.113, 1.120, 1.100, 1.115),  # would-be spring
            (1.115, 1.125, 1.113, 1.122),  # would-be mss
        ]
        df = _make_df(candles)
        # With default accum_range_mult=3.0, the trending accumulation is too wide
        p = WyckoffDetector(accum_range_mult=3.0).detect(df, PivotSide.DEMAND, end_idx=len(df))
        assert p is None

    def test_end_idx_zero_returns_none(self):
        df = _make_df(_DEMAND_CANDLES)
        p = WyckoffDetector().detect(df, PivotSide.DEMAND, end_idx=0)
        assert p is None

    def test_demand_detector_ignores_supply_pattern(self):
        df = _make_df(_SUPPLY_CANDLES)
        p = WyckoffDetector().detect(df, PivotSide.DEMAND, end_idx=len(df))
        # Supply pattern: upthrust + MSS below → demand scan finds no Spring
        assert p is None

    def test_supply_detector_ignores_demand_pattern(self):
        df = _make_df(_DEMAND_CANDLES)
        p = WyckoffDetector().detect(df, PivotSide.SUPPLY, end_idx=len(df))
        assert p is None


# ── TestEndIdx ────────────────────────────────────────────────────────────────

class TestEndIdx:
    def test_end_idx_excludes_mss_bar(self):
        # If we pass end_idx=5 (bar 5 = MSS excluded), pattern is incomplete
        df = _make_df(_DEMAND_CANDLES)
        p = WyckoffDetector().detect(df, PivotSide.DEMAND, end_idx=5)
        assert p is None

    def test_end_idx_includes_mss_bar(self):
        # end_idx=6 includes bar 5 (MSS) → pattern found
        df = _make_df(_DEMAND_CANDLES)
        p = WyckoffDetector().detect(df, PivotSide.DEMAND, end_idx=6)
        assert p is not None
        assert p.mss_bar == 5

    def test_absolute_indices_match_df_timestamps(self):
        # Verify that manip_bar / mss_bar are absolute df indices, not
        # slice-relative offsets.  Cross-check via df.index.
        df = _make_df(_DEMAND_CANDLES)
        p = WyckoffDetector().detect(df, PivotSide.DEMAND, end_idx=len(df))
        assert p is not None
        assert p.formed_at == df.index[p.mss_bar]
        assert p.manip_bar == 4
        assert p.mss_bar   == 5


# ── TestPatternWithGapBetweenSpringAndMss ─────────────────────────────────────

class TestPatternWithGapBetweenSpringAndMss:
    def test_mss_two_bars_after_spring(self):
        # Pattern with one neutral bar between Spring and MSS.
        # Bar 3 low=1.103 > accum_low=1.100 so it cannot trigger early Spring.
        candles = [
            (1.103, 1.108, 1.100, 1.106),  # 0 accum  low=1.100  ← floor
            (1.105, 1.109, 1.102, 1.107),  # 1 accum
            (1.104, 1.110, 1.101, 1.105),  # 2 accum
            (1.106, 1.109, 1.103, 1.104),  # 3 accum  low=1.103 > 1.100
            (1.104, 1.108, 1.095, 1.106),  # 4 Spring
            (1.106, 1.109, 1.104, 1.107),  # 5 neutral — inside range
            (1.108, 1.116, 1.107, 1.114),  # 6 MSS close=1.114 > 1.110
        ]
        df = _make_df(candles)
        p = WyckoffDetector().detect(df, PivotSide.DEMAND, end_idx=len(df))
        assert p is not None
        assert p.manip_bar == 4
        assert p.mss_bar   == 6


# ── TestScore ─────────────────────────────────────────────────────────────────

class TestScore:
    def test_score_bounded_0_to_10(self):
        df = _make_df(_DEMAND_CANDLES)
        p = WyckoffDetector().detect(df, PivotSide.DEMAND, end_idx=len(df))
        assert p is not None
        assert 0.0 <= p.score <= 10.0

    def test_large_sweep_gets_higher_score(self):
        # Pattern with a bigger Spring sweep → better score
        candles_small = list(_DEMAND_CANDLES)  # Spring low=1.095 (5 pips below 1.100)
        candles_big = [
            (1.103, 1.108, 1.100, 1.106),
            (1.105, 1.109, 1.102, 1.107),
            (1.104, 1.110, 1.101, 1.105),
            (1.106, 1.109, 1.103, 1.104),  # low=1.103 > 1.100 — not a Spring
            (1.104, 1.108, 1.080, 1.106),  # Spring low=1.080 (20 pips below!)
            (1.108, 1.115, 1.107, 1.113),
        ]
        df_small = _make_df(candles_small)
        df_big   = _make_df(candles_big)
        det = WyckoffDetector()
        p_small = det.detect(df_small, PivotSide.DEMAND, end_idx=len(df_small))
        p_big   = det.detect(df_big,   PivotSide.DEMAND, end_idx=len(df_big))
        assert p_small is not None
        assert p_big   is not None
        assert p_big.score >= p_small.score

    def test_faster_mss_gets_higher_score(self):
        # MSS 1 bar after Spring vs 4 bars after
        candles_fast = list(_DEMAND_CANDLES)  # MSS 1 bar after Spring
        candles_slow = [
            (1.103, 1.108, 1.100, 1.106),
            (1.105, 1.109, 1.102, 1.107),
            (1.104, 1.110, 1.101, 1.105),
            (1.106, 1.109, 1.103, 1.104),  # low=1.103 > 1.100 — not a Spring
            (1.104, 1.108, 1.095, 1.106),  # Spring at bar 4
            (1.106, 1.109, 1.104, 1.107),  # inside range
            (1.107, 1.108, 1.104, 1.107),  # inside range
            (1.107, 1.109, 1.105, 1.108),  # inside range
            (1.108, 1.115, 1.107, 1.113),  # MSS 4 bars after Spring
        ]
        df_fast = _make_df(candles_fast)
        df_slow = _make_df(candles_slow)
        det = WyckoffDetector()
        p_fast = det.detect(df_fast, PivotSide.DEMAND, end_idx=len(df_fast))
        p_slow = det.detect(df_slow, PivotSide.DEMAND, end_idx=len(df_slow))
        assert p_fast is not None
        assert p_slow is not None
        assert p_fast.score >= p_slow.score
