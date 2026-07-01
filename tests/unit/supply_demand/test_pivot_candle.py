"""
Tests for zeus.strategy.supply_demand.pivot_candle

Candle constructions used throughout:
  demand_pivot : o=1.100, h=1.110, l=1.080, c=1.108
      range=0.030, lower_wick=0.020(67%), body=0.008(27%), upper_wick=0.002(7%)
      → clear buyers-won rejection → DEMAND

  supply_pivot : o=1.108, h=1.130, l=1.100, c=1.102
      range=0.030, upper_wick=0.022(73%), body=0.006(20%), lower_wick=0.002(7%)
      → clear sellers-won rejection → SUPPLY

  doji         : o=1.100, h=1.120, l=1.080, c=1.100
      range=0.040, both wicks 50 %, no body → no clear winner → DOJI

  full_body    : o=1.080, h=1.120, l=1.080, c=1.120
      range=0.040, no wicks, full bullish body → DOJI (no rejection)
"""
from __future__ import annotations

import pandas as pd
import pytest

from zeus.strategy.supply_demand.pivot_candle import (
    MIN_SCORE,
    PivotSide,
    analyze_candle,
    find_last_pivot_candle,
    scan_for_pivots,
)

_TS = pd.Timestamp("2026-06-01 09:00:00")


# ── Helper ───────────────────────────────────────────────────────────────────

def _make_df(candles: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    idx = pd.date_range("2026-06-01 09:00", periods=len(candles), freq="15min")
    return pd.DataFrame(
        [{"open": o, "high": h, "low": l, "close": c} for o, h, l, c in candles],
        index=idx,
    )


# ── analyze_candle ────────────────────────────────────────────────────────────

class TestAnalyzeCandle:
    def test_demand_pivot_side(self):
        pc = analyze_candle(_TS, 1.100, 1.110, 1.080, 1.108)
        assert pc.side == PivotSide.DEMAND

    def test_demand_pivot_high_score(self):
        pc = analyze_candle(_TS, 1.100, 1.110, 1.080, 1.108)
        assert pc.score >= 6.0, f"Expected score ≥ 6, got {pc.score}"

    def test_demand_pivot_zone_levels(self):
        pc = analyze_candle(_TS, 1.100, 1.110, 1.080, 1.108)
        assert pc.body_high  == pytest.approx(1.108)
        assert pc.body_low   == pytest.approx(1.100)
        assert pc.wick_high  == pytest.approx(1.110)
        assert pc.wick_low   == pytest.approx(1.080)

    def test_demand_pivot_wick_ratios(self):
        pc = analyze_candle(_TS, 1.100, 1.110, 1.080, 1.108)
        assert pc.lower_wick_ratio > 0.60
        assert pc.upper_wick_ratio < 0.10

    def test_supply_pivot_side(self):
        pc = analyze_candle(_TS, 1.108, 1.130, 1.100, 1.102)
        assert pc.side == PivotSide.SUPPLY

    def test_supply_pivot_high_score(self):
        pc = analyze_candle(_TS, 1.108, 1.130, 1.100, 1.102)
        assert pc.score >= 6.0, f"Expected score ≥ 6, got {pc.score}"

    def test_supply_pivot_zone_levels(self):
        pc = analyze_candle(_TS, 1.108, 1.130, 1.100, 1.102)
        assert pc.body_high == pytest.approx(1.108)
        assert pc.body_low  == pytest.approx(1.102)
        assert pc.wick_high == pytest.approx(1.130)
        assert pc.wick_low  == pytest.approx(1.100)

    def test_doji_equal_wicks(self):
        # symmetric doji → neither wick dominates
        pc = analyze_candle(_TS, 1.100, 1.120, 1.080, 1.100)
        assert pc.side == PivotSide.DOJI

    def test_full_body_no_wicks(self):
        # big bullish body, no wicks → no rejection → DOJI
        pc = analyze_candle(_TS, 1.080, 1.120, 1.080, 1.120)
        assert pc.side == PivotSide.DOJI

    def test_flat_candle_zero_range(self):
        pc = analyze_candle(_TS, 1.100, 1.100, 1.100, 1.100)
        assert pc.side  == PivotSide.DOJI
        assert pc.score == pytest.approx(0.0)
        assert pc.body_ratio == pytest.approx(0.0)

    def test_score_bounded_0_to_10(self):
        # perfect pin bar: zero body, all lower wick
        pc = analyze_candle(_TS, 1.110, 1.110, 1.080, 1.110)
        assert 0.0 <= pc.score <= 10.0

    def test_bearish_body_demand_pivot(self):
        # bearish close but long lower wick (buyers recovered) → still DEMAND
        # o=1.108, c=1.100 → bearish body, but lower wick is 0.020 vs body 0.008
        pc = analyze_candle(_TS, 1.108, 1.110, 1.080, 1.100)
        assert pc.side == PivotSide.DEMAND

    def test_score_demand_less_than_supply_for_pure_supply(self):
        # supply pin: long upper wick, tiny lower wick → score_supply > score_demand
        pc = analyze_candle(_TS, 1.108, 1.130, 1.100, 1.102)
        assert pc.side == PivotSide.SUPPLY


# ── find_last_pivot_candle ─────────────────────────────────────────────────────

class TestFindLastPivotCandle:
    def test_returns_last_qualifying_pivot(self):
        # [doji, supply, doji, DEMAND ← last, doji] → expect index 3
        candles = [
            (1.100, 1.120, 1.080, 1.100),   # 0 doji
            (1.108, 1.130, 1.100, 1.102),   # 1 supply
            (1.100, 1.120, 1.080, 1.100),   # 2 doji
            (1.100, 1.110, 1.080, 1.108),   # 3 demand ← last
            (1.100, 1.120, 1.080, 1.100),   # 4 doji (impulse starts here)
        ]
        df = _make_df(candles)
        pc = find_last_pivot_candle(df, 0, 4)  # exclude bar 4
        assert pc is not None
        assert pc.side == PivotSide.DEMAND
        assert pc.index == df.index[3]

    def test_side_filter_demand(self):
        # bars: supply at 1, demand at 3 → with side=DEMAND must return index 3
        candles = [
            (1.100, 1.120, 1.080, 1.100),
            (1.108, 1.130, 1.100, 1.102),   # 1 supply
            (1.100, 1.120, 1.080, 1.100),
            (1.100, 1.110, 1.080, 1.108),   # 3 demand
        ]
        df = _make_df(candles)
        pc = find_last_pivot_candle(df, 0, 4, side=PivotSide.DEMAND)
        assert pc is not None
        assert pc.side == PivotSide.DEMAND

    def test_side_filter_supply(self):
        candles = [
            (1.108, 1.130, 1.100, 1.102),   # 0 supply
            (1.100, 1.110, 1.080, 1.108),   # 1 demand
            (1.100, 1.120, 1.080, 1.100),   # 2 doji
        ]
        df = _make_df(candles)
        pc = find_last_pivot_candle(df, 0, 3, side=PivotSide.SUPPLY)
        assert pc is not None
        assert pc.side == PivotSide.SUPPLY
        assert pc.index == df.index[0]

    def test_returns_none_when_no_pivot(self):
        candles = [
            (1.100, 1.120, 1.080, 1.100),   # doji
            (1.100, 1.120, 1.080, 1.100),   # doji
        ]
        df = _make_df(candles)
        pc = find_last_pivot_candle(df, 0, 2)
        assert pc is None

    def test_min_score_filter(self):
        # demand pivot exists but has low score → filtered out by high min_score
        candles = [
            (1.100, 1.110, 1.080, 1.108),   # moderate demand pivot
        ]
        df = _make_df(candles)
        pc = find_last_pivot_candle(df, 0, 1, min_score=9.5)
        assert pc is None

    def test_single_bar_range(self):
        candles = [(1.100, 1.110, 1.080, 1.108)]
        df = _make_df(candles)
        pc = find_last_pivot_candle(df, 0, 1)
        assert pc is not None
        assert pc.side == PivotSide.DEMAND

    def test_empty_range(self):
        candles = [(1.100, 1.110, 1.080, 1.108)]
        df = _make_df(candles)
        pc = find_last_pivot_candle(df, 0, 0)  # start == end → empty
        assert pc is None


# ── scan_for_pivots ────────────────────────────────────────────────────────────

class TestScanForPivots:
    def test_returns_all_pivots(self):
        candles = [
            (1.100, 1.110, 1.080, 1.108),   # demand
            (1.100, 1.120, 1.080, 1.100),   # doji  (excluded)
            (1.108, 1.130, 1.100, 1.102),   # supply
            (1.100, 1.110, 1.080, 1.108),   # demand
        ]
        df = _make_df(candles)
        pivots = scan_for_pivots(df)
        assert len(pivots) == 3
        assert pivots[0].side == PivotSide.DEMAND
        assert pivots[1].side == PivotSide.SUPPLY
        assert pivots[2].side == PivotSide.DEMAND

    def test_chronological_order(self):
        candles = [
            (1.108, 1.130, 1.100, 1.102),   # supply at 0
            (1.100, 1.120, 1.080, 1.100),   # doji
            (1.100, 1.110, 1.080, 1.108),   # demand at 2
        ]
        df = _make_df(candles)
        pivots = scan_for_pivots(df)
        assert pivots[0].index < pivots[1].index

    def test_empty_df_returns_empty_list(self):
        df = pd.DataFrame(
            columns=["open", "high", "low", "close"],
            index=pd.DatetimeIndex([]),
        )
        assert scan_for_pivots(df) == []

    def test_respects_start_end_idx(self):
        candles = [
            (1.100, 1.110, 1.080, 1.108),   # 0 demand
            (1.100, 1.120, 1.080, 1.100),   # 1 doji
            (1.108, 1.130, 1.100, 1.102),   # 2 supply
        ]
        df = _make_df(candles)
        # Only scan bars 1–2 → should only see the supply
        pivots = scan_for_pivots(df, start_idx=1, end_idx=3)
        assert len(pivots) == 1
        assert pivots[0].side == PivotSide.SUPPLY

    def test_all_dojis_returns_empty(self):
        candles = [
            (1.100, 1.120, 1.080, 1.100),
            (1.100, 1.120, 1.080, 1.100),
        ]
        df = _make_df(candles)
        assert scan_for_pivots(df) == []

    def test_min_score_filters_weak_pivots(self):
        # demand pivot at moderate score; filter with high threshold → empty
        candles = [(1.100, 1.110, 1.080, 1.108)]
        df = _make_df(candles)
        # moderate pivot passes default min_score but fails high threshold
        all_pivots  = scan_for_pivots(df, min_score=MIN_SCORE)
        high_filter = scan_for_pivots(df, min_score=9.9)
        assert len(all_pivots) >= 1
        assert len(high_filter) == 0
