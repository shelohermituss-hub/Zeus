"""
Unit tests for zeus.strategy.smc.session.

Uses synthetic 1-hour DataFrames with a DatetimeIndex so every session
boundary is deterministic. All timestamps are UTC.
"""
import pytest
import pandas as pd

from zeus.strategy.smc.session import (
    SessionType,
    SessionRange,
    detect_session_ranges,
    get_session_ranges,
    last_session_range,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _hourly_df(
    start: str = "2024-01-01 00:00",
    periods: int = 24,
    high: float = 50_100.0,
    low: float  = 49_900.0,
) -> pd.DataFrame:
    """Single-value high/low DataFrame with hourly resolution."""
    idx = pd.date_range(start, periods=periods, freq="1h")
    return pd.DataFrame(
        {"high": [high] * periods, "low": [low] * periods, "close": [50_000.0] * periods},
        index=idx,
    )


def _custom_df(highs: list[float], lows: list[float],
               start: str = "2024-01-01 00:00") -> pd.DataFrame:
    """Per-bar custom high/low, hourly, starting at start."""
    idx = pd.date_range(start, periods=len(highs), freq="1h")
    return pd.DataFrame(
        {"high": highs, "low": lows,
         "close": [(h + lo) / 2 for h, lo in zip(highs, lows)]},
        index=idx,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Session boundary detection — one full day (00:00–23:00)
# ──────────────────────────────────────────────────────────────────────────────
#
# With hourly bars starting 2024-01-01 00:00:
#   bar  0–7  → Asian   (00:00–07:00)   formed_at = 8
#   bar  8–15 → London  (08:00–15:00)   formed_at = 16
#   bar 13–20 → NY      (13:00–20:00)   formed_at = 21

class TestOneDayDetection:
    @pytest.fixture
    def ranges(self):
        return detect_session_ranges(_hourly_df(periods=24))

    def test_three_sessions_detected(self, ranges):
        assert len(ranges) == 3

    def test_asian_session_present(self, ranges):
        types = [r.session for r in ranges]
        assert SessionType.ASIAN in types

    def test_london_session_present(self, ranges):
        types = [r.session for r in ranges]
        assert SessionType.LONDON in types

    def test_ny_session_present(self, ranges):
        types = [r.session for r in ranges]
        assert SessionType.NY in types

    def test_asian_formed_at_bar_8(self, ranges):
        asian = next(r for r in ranges if r.session == SessionType.ASIAN)
        assert asian.formed_at == 8

    def test_asian_starts_at_bar_0(self, ranges):
        asian = next(r for r in ranges if r.session == SessionType.ASIAN)
        assert asian.start_bar == 0

    def test_london_formed_at_bar_16(self, ranges):
        london = next(r for r in ranges if r.session == SessionType.LONDON)
        assert london.formed_at == 16

    def test_ny_formed_at_bar_21(self, ranges):
        ny = next(r for r in ranges if r.session == SessionType.NY)
        assert ny.formed_at == 21

    def test_ny_starts_at_bar_13(self, ranges):
        ny = next(r for r in ranges if r.session == SessionType.NY)
        assert ny.start_bar == 13

    def test_ranges_sorted_by_formed_at(self, ranges):
        formed = [r.formed_at for r in ranges]
        assert formed == sorted(formed)


# ──────────────────────────────────────────────────────────────────────────────
# High / low tracking
# ──────────────────────────────────────────────────────────────────────────────

class TestHighLowTracking:
    def test_asian_high_is_max_of_session_bars(self):
        # Bars 0–7 are Asian; spike high on bar 3 (03:00)
        highs = [50_000.0] * 24
        lows  = [49_000.0] * 24
        highs[3] = 51_000.0   # bar 3 = 03:00 (inside Asian)
        df = _custom_df(highs, lows)
        ranges = detect_session_ranges(df, sessions=[SessionType.ASIAN])
        asian = ranges[0]
        assert asian.high == pytest.approx(51_000.0)
        assert asian.high_bar == 3

    def test_asian_low_is_min_of_session_bars(self):
        highs = [50_000.0] * 24
        lows  = [49_000.0] * 24
        lows[5] = 47_500.0   # bar 5 = 05:00 (inside Asian)
        df = _custom_df(highs, lows)
        ranges = detect_session_ranges(df, sessions=[SessionType.ASIAN])
        asian = ranges[0]
        assert asian.low == pytest.approx(47_500.0)
        assert asian.low_bar == 5

    def test_spike_outside_session_not_captured(self):
        # Bar 9 = 09:00 → London, should NOT affect Asian high
        highs = [50_000.0] * 24
        lows  = [49_000.0] * 24
        highs[9] = 55_000.0
        df = _custom_df(highs, lows)
        ranges = detect_session_ranges(df, sessions=[SessionType.ASIAN])
        asian = ranges[0]
        assert asian.high == pytest.approx(50_000.0)

    def test_london_ny_overlap_bars_contribute_to_both(self):
        # Bars 13–15 are in both London (8-16) and NY (13-21)
        highs = [50_000.0] * 24
        lows  = [49_000.0] * 24
        highs[14] = 52_000.0   # bar 14 = 14:00, inside both London and NY
        lows[14]  = 48_000.0
        df = _custom_df(highs, lows)
        ranges = detect_session_ranges(df)
        london = next(r for r in ranges if r.session == SessionType.LONDON)
        ny     = next(r for r in ranges if r.session == SessionType.NY)
        assert london.high == pytest.approx(52_000.0)
        assert ny.high     == pytest.approx(52_000.0)


# ──────────────────────────────────────────────────────────────────────────────
# SessionRange properties
# ──────────────────────────────────────────────────────────────────────────────

class TestSessionRangeProperties:
    @pytest.fixture
    def sr(self):
        return SessionRange(
            session=SessionType.ASIAN,
            high=50_200.0,
            low=49_800.0,
            high_bar=3,
            low_bar=6,
            start_bar=0,
            formed_at=8,
        )

    def test_mid_is_average(self, sr):
        assert sr.mid == pytest.approx(50_000.0)

    def test_is_near_high_within_tolerance(self, sr):
        # 0.3% of 50_200 = 150.6 → price 50_100 is within tolerance
        assert sr.is_near_high(50_100.0)

    def test_is_near_high_outside_tolerance(self, sr):
        assert not sr.is_near_high(49_000.0)

    def test_is_near_low_within_tolerance(self, sr):
        assert sr.is_near_low(49_850.0)

    def test_is_near_low_outside_tolerance(self, sr):
        assert not sr.is_near_low(51_000.0)

    def test_is_near_mid_within_tolerance(self, sr):
        assert sr.is_near_mid(50_000.0)

    def test_is_near_mid_outside_tolerance(self, sr):
        assert not sr.is_near_mid(48_000.0)

    def test_custom_tolerance(self, sr):
        # 1% of 50_200 = 502 → price 50_600 is within 1% but not 0.3%
        assert sr.is_near_high(50_600.0, tolerance_pct=0.01)
        assert not sr.is_near_high(50_600.0, tolerance_pct=0.003)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-day detection
# ──────────────────────────────────────────────────────────────────────────────

class TestMultiDay:
    @pytest.fixture
    def ranges(self):
        # 48 hours = 2 full trading days
        return detect_session_ranges(_hourly_df(periods=48))

    def test_six_sessions_over_two_days(self, ranges):
        assert len(ranges) == 6

    def test_two_asian_sessions(self, ranges):
        asians = [r for r in ranges if r.session == SessionType.ASIAN]
        assert len(asians) == 2

    def test_second_asian_formed_at_correct_bar(self, ranges):
        asians = [r for r in ranges if r.session == SessionType.ASIAN]
        # Day 1 Asian: formed_at=8; Day 2 Asian: formed_at=24+8=32
        assert asians[1].formed_at == 32

    def test_second_ny_formed_at_correct_bar(self, ranges):
        nys = [r for r in ranges if r.session == SessionType.NY]
        # Day 1 NY: formed_at=21; Day 2 NY: formed_at=24+21=45
        assert nys[1].formed_at == 45


# ──────────────────────────────────────────────────────────────────────────────
# Incomplete sessions
# ──────────────────────────────────────────────────────────────────────────────

class TestIncompleteSessions:
    def test_active_session_at_data_end_not_returned(self):
        # Data ends at bar 20 (20:00) — NY (13:00–21:00) is still active
        df = _hourly_df(periods=21)   # bars 0–20
        ranges = detect_session_ranges(df)
        types = [r.session for r in ranges]
        assert SessionType.NY not in types

    def test_completed_sessions_still_returned_when_last_incomplete(self):
        df = _hourly_df(periods=21)   # bars 0–20; Asian and London complete
        ranges = detect_session_ranges(df)
        types = [r.session for r in ranges]
        assert SessionType.ASIAN in types
        assert SessionType.LONDON in types

    def test_no_sessions_when_data_too_short(self):
        df = _hourly_df(periods=7)   # only 00:00–06:00, Asian never closes
        ranges = detect_session_ranges(df)
        assert ranges == []


# ──────────────────────────────────────────────────────────────────────────────
# Session filter parameter
# ──────────────────────────────────────────────────────────────────────────────

class TestSessionFilter:
    def test_filter_asian_only(self):
        ranges = detect_session_ranges(
            _hourly_df(periods=24),
            sessions=[SessionType.ASIAN],
        )
        assert all(r.session == SessionType.ASIAN for r in ranges)
        assert len(ranges) == 1

    def test_filter_two_sessions(self):
        ranges = detect_session_ranges(
            _hourly_df(periods=24),
            sessions=[SessionType.ASIAN, SessionType.LONDON],
        )
        types = {r.session for r in ranges}
        assert types == {SessionType.ASIAN, SessionType.LONDON}
        assert SessionType.NY not in types


# ──────────────────────────────────────────────────────────────────────────────
# Timezone-aware index
# ──────────────────────────────────────────────────────────────────────────────

class TestTimezone:
    def test_tz_aware_index_accepted(self):
        idx = pd.date_range("2024-01-01 00:00", periods=24, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {"high": [50_100.0] * 24, "low": [49_900.0] * 24, "close": [50_000.0] * 24},
            index=idx,
        )
        ranges = detect_session_ranges(df)
        assert len(ranges) == 3

    def test_non_datetime_index_raises(self):
        df = pd.DataFrame({"high": [50_000.0], "low": [49_000.0], "close": [49_500.0]})
        with pytest.raises(ValueError, match="DatetimeIndex"):
            detect_session_ranges(df)


# ──────────────────────────────────────────────────────────────────────────────
# get_session_ranges
# ──────────────────────────────────────────────────────────────────────────────

class TestGetSessionRanges:
    @pytest.fixture
    def ranges(self):
        return detect_session_ranges(_hourly_df(periods=48))

    def test_returns_only_formed_ranges(self, ranges):
        # At bar 5 nothing is formed yet (Asian formed_at=8)
        active = get_session_ranges(ranges, at_bar=5)
        assert active == []

    def test_returns_asian_at_bar_8(self, ranges):
        active = get_session_ranges(ranges, at_bar=8)
        assert len(active) == 1
        assert active[0].session == SessionType.ASIAN

    def test_session_type_filter(self, ranges):
        active = get_session_ranges(ranges, at_bar=50, session=SessionType.NY)
        assert all(r.session == SessionType.NY for r in active)

    def test_all_six_visible_at_end(self, ranges):
        active = get_session_ranges(ranges, at_bar=47)
        assert len(active) == 6


# ──────────────────────────────────────────────────────────────────────────────
# last_session_range
# ──────────────────────────────────────────────────────────────────────────────

class TestLastSessionRange:
    @pytest.fixture
    def ranges(self):
        return detect_session_ranges(_hourly_df(periods=48))

    def test_returns_most_recent(self, ranges):
        r = last_session_range(ranges, at_bar=47)
        assert r is not None
        # NY day 2: formed_at=45 is the last completed session
        assert r.session == SessionType.NY
        assert r.formed_at == 45

    def test_returns_none_before_any_session(self, ranges):
        assert last_session_range(ranges, at_bar=5) is None

    def test_session_type_filter(self, ranges):
        r = last_session_range(ranges, at_bar=47, session=SessionType.ASIAN)
        assert r is not None
        assert r.session == SessionType.ASIAN
        assert r.formed_at == 32   # Day 2 Asian

    def test_returns_none_when_no_match_for_type(self, ranges):
        # At bar 7 nothing is formed yet
        assert last_session_range(ranges, at_bar=7, session=SessionType.ASIAN) is None
