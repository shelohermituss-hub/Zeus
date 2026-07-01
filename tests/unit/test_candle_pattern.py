"""
Unit tests for candle_pattern.detect_entry_candle().

Covers: hammer, shooting star, bullish/bearish engulfing,
        and all edge / rejection cases.
"""
from __future__ import annotations

import pytest
import pandas as pd

from zeus.strategy.smc.candle_pattern import detect_entry_candle
from zeus.strategy.smc.pivot import BULLISH, BEARISH


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bar(o: float, h: float, l: float, c: float) -> pd.DataFrame:
    """Build a two-bar DataFrame: a neutral previous bar then the test bar."""
    mid = (h + l) / 2.0
    return pd.DataFrame(
        {
            "open":  [mid, o],
            "high":  [mid + 1, h],
            "low":   [mid - 1, l],
            "close": [mid, c],
            "volume": [1000.0, 1000.0],
        },
        index=pd.date_range("2026-01-06 08:00", periods=2, freq="1min", tz="UTC"),
    )


# ── Hammer / pin bar (LONG) ───────────────────────────────────────────────────

class TestHammer:
    def test_perfect_hammer_accepted(self):
        # range=10, lower wick=7, close=2998 > mid=2995 → ratio=70% ≥ 60%
        df = _bar(o=2998.0, h=3000.0, l=2990.0, c=2998.0)
        ok, reason = detect_entry_candle(df, bar_index=1, direction=BULLISH)
        assert ok, reason
        assert "hammer" in reason

    def test_hammer_exactly_at_threshold(self):
        # range=10, lower wick=6, close at mid or above → ratio=60% exactly
        df = _bar(o=2996.0, h=3000.0, l=2990.0, c=2996.0)
        ok, reason = detect_entry_candle(df, bar_index=1, direction=BULLISH, min_wick_ratio=0.60)
        assert ok, reason

    def test_hammer_rejected_below_threshold(self):
        # lower wick=5, range=10 → ratio=50% < 60%
        # prev bar has a big body so current bar cannot accidentally engulf it
        df = pd.DataFrame(
            {
                "open":  [2993.0, 2996.0],  # prev: big bullish body prevents engulfing
                "high":  [3002.0, 3001.0],
                "low":   [2991.0, 2991.0],
                "close": [3002.0, 2999.0],
                "volume": [1000.0, 1000.0],
            },
            index=pd.date_range("2026-01-06 08:00", periods=2, freq="1min", tz="UTC"),
        )
        # lower_wick = min(2996, 2999) - 2991 = 5, range = 10 → 50% < 60%
        ok, _ = detect_entry_candle(df, bar_index=1, direction=BULLISH)
        assert not ok

    def test_hammer_rejected_close_below_mid(self):
        # Long lower wick but close in lower half — not a valid hammer
        df = _bar(o=2993.0, h=3000.0, l=2990.0, c=2993.0)
        # lower_wick = 2993 - 2990 = 3, range = 10, mid = 2995 → close=2993 < 2995
        ok, _ = detect_entry_candle(df, bar_index=1, direction=BULLISH)
        assert not ok


# ── Shooting star (SHORT) ─────────────────────────────────────────────────────

class TestShootingStar:
    def test_shooting_star_accepted(self):
        # range=10, upper wick=7, close=2992 < mid=2995
        df = _bar(o=2992.0, h=3000.0, l=2990.0, c=2992.0)
        ok, reason = detect_entry_candle(df, bar_index=1, direction=BEARISH)
        assert ok, reason
        assert "shooting_star" in reason

    def test_shooting_star_rejected_close_above_mid(self):
        # Long upper wick but close in upper half
        df = _bar(o=2997.0, h=3000.0, l=2990.0, c=2997.0)
        # upper_wick = 3000 - 2997 = 3, range = 10, mid = 2995 → close=2997 > 2995
        ok, _ = detect_entry_candle(df, bar_index=1, direction=BEARISH)
        assert not ok

    def test_shooting_star_rejected_small_upper_wick(self):
        # upper wick=4, range=10 → 40% < 60%
        # prev bar has a big bearish body so current bar cannot accidentally engulf it
        df = pd.DataFrame(
            {
                "open":  [2997.0, 2993.0],  # prev: big bearish body prevents engulfing
                "high":  [2998.0, 2997.0],
                "low":   [2984.0, 2987.0],
                "close": [2984.0, 2990.0],
                "volume": [1000.0, 1000.0],
            },
            index=pd.date_range("2026-01-06 08:00", periods=2, freq="1min", tz="UTC"),
        )
        # upper_wick = 2997 - max(2993, 2990) = 4, range = 10 → 40% < 60%
        ok, _ = detect_entry_candle(df, bar_index=1, direction=BEARISH)
        assert not ok


# ── Bullish engulfing (LONG) ──────────────────────────────────────────────────

class TestBullishEngulfing:
    def test_bullish_engulfing_accepted(self):
        # prev bar: bearish, body [2996, 2998]
        # curr bar: bullish, open=2995 (≤ 2996), close=2999 (≥ 2998)
        df = pd.DataFrame(
            {
                "open":  [2998.0, 2995.0],
                "high":  [2999.0, 3000.0],
                "low":   [2995.0, 2994.0],
                "close": [2996.0, 2999.0],
                "volume": [1000.0, 1000.0],
            },
            index=pd.date_range("2026-01-06 08:00", periods=2, freq="1min", tz="UTC"),
        )
        ok, reason = detect_entry_candle(df, bar_index=1, direction=BULLISH)
        assert ok, reason
        assert "engulfing" in reason

    def test_partial_engulf_not_accepted(self):
        # curr open doesn't go below prev body bottom
        df = pd.DataFrame(
            {
                "open":  [2998.0, 2997.0],  # open=2997 > prev body bottom 2996
                "high":  [2999.0, 3001.0],
                "low":   [2995.0, 2994.0],
                "close": [2996.0, 3001.0],
                "volume": [1000.0, 1000.0],
            },
            index=pd.date_range("2026-01-06 08:00", periods=2, freq="1min", tz="UTC"),
        )
        ok, _ = detect_entry_candle(df, bar_index=1, direction=BULLISH)
        assert not ok


# ── Bearish engulfing (SHORT) ─────────────────────────────────────────────────

class TestBearishEngulfing:
    def test_bearish_engulfing_accepted(self):
        # prev bar: bullish, body [2996, 2998]
        # curr bar: bearish, open=2999 (≥ 2998), close=2995 (≤ 2996)
        df = pd.DataFrame(
            {
                "open":  [2996.0, 2999.0],
                "high":  [2999.0, 3000.0],
                "low":   [2994.0, 2994.0],
                "close": [2998.0, 2995.0],
                "volume": [1000.0, 1000.0],
            },
            index=pd.date_range("2026-01-06 08:00", periods=2, freq="1min", tz="UTC"),
        )
        ok, reason = detect_entry_candle(df, bar_index=1, direction=BEARISH)
        assert ok, reason
        assert "engulfing" in reason

    def test_bearish_no_engulf_when_not_fully_covering(self):
        df = pd.DataFrame(
            {
                "open":  [2996.0, 2997.0],  # open=2997 < prev body top 2998
                "high":  [2999.0, 2999.0],
                "low":   [2994.0, 2993.0],
                "close": [2998.0, 2994.0],
                "volume": [1000.0, 1000.0],
            },
            index=pd.date_range("2026-01-06 08:00", periods=2, freq="1min", tz="UTC"),
        )
        ok, _ = detect_entry_candle(df, bar_index=1, direction=BEARISH)
        assert not ok


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_insufficient_bars_returns_false(self):
        df = _bar(o=2998.0, h=3000.0, l=2990.0, c=2998.0)
        ok, reason = detect_entry_candle(df, bar_index=0, direction=BULLISH)
        assert not ok
        assert "insufficient" in reason

    def test_zero_range_candle(self):
        df = _bar(o=3000.0, h=3000.0, l=3000.0, c=3000.0)
        ok, reason = detect_entry_candle(df, bar_index=1, direction=BULLISH)
        assert not ok
        assert "zero-range" in reason

    def test_unknown_direction_returns_false(self):
        df = _bar(o=2998.0, h=3000.0, l=2990.0, c=2998.0)
        ok, reason = detect_entry_candle(df, bar_index=1, direction=99)
        assert not ok
        assert "unknown" in reason

    def test_custom_wick_ratio_more_lenient(self):
        # Bar: o=2996, h=3000, l=2992, c=2998
        # lower_wick = 2996 - 2992 = 4, range = 8 → 50% — below 60%, above 35%
        # prev bar has a big bearish body so the current bar cannot accidentally engulf it
        df = pd.DataFrame(
            {
                "open":  [2988.0, 2996.0],
                "high":  [2990.0, 3000.0],
                "low":   [2982.0, 2992.0],
                "close": [2982.0, 2998.0],
                "volume": [1000.0, 1000.0],
            },
            index=pd.date_range("2026-01-06 08:00", periods=2, freq="1min", tz="UTC"),
        )
        ok_strict, _ = detect_entry_candle(df, bar_index=1, direction=BULLISH, min_wick_ratio=0.60)
        ok_loose, _  = detect_entry_candle(df, bar_index=1, direction=BULLISH, min_wick_ratio=0.35)
        assert not ok_strict
        assert ok_loose

    def test_custom_wick_ratio_stricter(self):
        # Bar: o=2997, h=3000, l=2990, c=2998
        # lower_wick = 2997 - 2990 = 7, range = 10 → 70%
        # Accepted at 60% threshold, rejected at 80% threshold
        # prev bar has a big bearish body so the current bar cannot accidentally engulf it
        df = pd.DataFrame(
            {
                "open":  [2988.0, 2997.0],
                "high":  [2990.0, 3000.0],
                "low":   [2982.0, 2990.0],
                "close": [2982.0, 2998.0],
                "volume": [1000.0, 1000.0],
            },
            index=pd.date_range("2026-01-06 08:00", periods=2, freq="1min", tz="UTC"),
        )
        ok_normal, _ = detect_entry_candle(df, bar_index=1, direction=BULLISH, min_wick_ratio=0.60)
        ok_strict, _ = detect_entry_candle(df, bar_index=1, direction=BULLISH, min_wick_ratio=0.80)
        assert ok_normal
        assert not ok_strict
