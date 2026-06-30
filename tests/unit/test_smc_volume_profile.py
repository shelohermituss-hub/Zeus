"""
Unit tests for zeus.strategy.smc.volume_profile.

Grid design
-----------
A "range anchor" wide bar (high=100, low=0, vol=1) is prepended to all
POC/VA scenarios to fix the price grid regardless of where the dominant
point bars land:

    price_low=0, price_high=100, num_bins=10, bin_width=10
    bin i centre = 0 + (i + 0.5) * 10  →  5, 15, 25, 35, 45, 55, 65, 75, 85, 95

The anchor contributes 0.1 volume per bin — negligible compared with the
test signal bars.
"""
import pytest
import numpy as np
import pandas as pd

from zeus.strategy.smc.volume_profile import (
    VolumeProfile,
    compute_volume_profile,
    detect_volume_profiles,
    get_latest_volume_profile,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

# Grid constants shared across POC and value-area tests
_BINS      = 10
_ANCHOR_H  = 100.0   # wide anchor bar sets range [0, 100]
_ANCHOR_L  = 0.0
_ANCHOR_V  = 1.0     # gives 0.1 vol/bin — background noise only
# Bin centres with this grid: 5, 15, 25, 35, 45, 55, 65, 75, 85, 95
_BIN_W     = (_ANCHOR_H - _ANCHOR_L) / _BINS   # = 10.0
_CENTRES   = [_ANCHOR_L + (i + 0.5) * _BIN_W for i in range(_BINS)]  # 5..95


def _point_bars(prices_and_vols: list[tuple[float, float]],
                anchor: bool = False):
    """
    Build arrays of point bars (high==low) at given (price, volume) pairs.

    When anchor=True, prepend a wide bar [0, 100, vol=1] to fix the price
    grid to bin_width=10 so bin centres land at 5, 15, …, 95.
    """
    highs   = np.array([p for p, _ in prices_and_vols], dtype=float)
    lows    = highs.copy()
    volumes = np.array([v for _, v in prices_and_vols], dtype=float)
    if anchor:
        highs   = np.concatenate([[_ANCHOR_H], highs])
        lows    = np.concatenate([[_ANCHOR_L], lows])
        volumes = np.concatenate([[_ANCHOR_V], volumes])
    return highs, lows, volumes


def _df(highs, lows, volumes, closes=None) -> pd.DataFrame:
    n = len(highs)
    return pd.DataFrame({
        "high":   highs,
        "low":    lows,
        "close":  closes if closes is not None else [(h + lo) / 2 for h, lo in zip(highs, lows)],
        "volume": volumes,
    })


# ──────────────────────────────────────────────────────────────────────────────
# compute_volume_profile — edge cases
# ──────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_arrays_return_none(self):
        result = compute_volume_profile(
            np.array([]), np.array([]), np.array([]), formed_at=0
        )
        assert result is None

    def test_zero_volume_returns_none(self):
        h, lo, v = _point_bars([(50.0, 0.0), (60.0, 0.0)])
        assert compute_volume_profile(h, lo, v) is None

    def test_single_point_bar(self):
        h, lo, v = _point_bars([(55.0, 100.0)])
        result = compute_volume_profile(h, lo, v, formed_at=0)
        assert result is not None
        # All bars at same price → degenerate path
        assert result.poc == pytest.approx(55.0)
        assert result.vah == pytest.approx(55.0)
        assert result.val == pytest.approx(55.0)

    def test_all_bars_at_same_price(self):
        h = np.array([50.0, 50.0, 50.0])
        lo = np.array([50.0, 50.0, 50.0])
        v  = np.array([10.0, 20.0, 30.0])
        result = compute_volume_profile(h, lo, v, formed_at=5)
        assert result is not None
        assert result.poc == pytest.approx(50.0)
        assert result.formed_at == 5

    def test_formed_at_is_preserved(self):
        h, lo, v = _point_bars([(55.0, 10.0)])
        result = compute_volume_profile(h, lo, v, formed_at=42)
        assert result.formed_at == 42


# ──────────────────────────────────────────────────────────────────────────────
# POC detection
# ──────────────────────────────────────────────────────────────────────────────
#
# All tests use anchor=True → grid [0, 100], bin_width=10
# Exact bin centres: _CENTRES = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]

class TestPOCDetection:
    def test_single_dominant_price_becomes_poc(self):
        # 90 vol at bin 5 (centre=55), 10 vol at bin 0 (centre=5)
        h, lo, v = _point_bars([(5.0, 10.0), (55.0, 90.0)], anchor=True)
        result = compute_volume_profile(h, lo, v, num_bins=_BINS, formed_at=1)
        assert result.poc == pytest.approx(55.0)

    def test_poc_at_lowest_bin_centre(self):
        # All dominant volume at bin 0 (centre=5)
        h, lo, v = _point_bars([(5.0, 100.0), (95.0, 1.0)], anchor=True)
        result = compute_volume_profile(h, lo, v, num_bins=_BINS, formed_at=1)
        assert result.poc == pytest.approx(5.0)

    def test_poc_at_highest_bin_centre(self):
        # All dominant volume at bin 9 (centre=95)
        h, lo, v = _point_bars([(5.0, 1.0), (95.0, 100.0)], anchor=True)
        result = compute_volume_profile(h, lo, v, num_bins=_BINS, formed_at=1)
        assert result.poc == pytest.approx(95.0)

    def test_poc_in_middle(self):
        # Dominant at bin 4 (centre=45); flanked by low-volume bars at 5 and 95
        h, lo, v = _point_bars([(5.0, 5.0), (45.0, 100.0), (95.0, 5.0)], anchor=True)
        result = compute_volume_profile(h, lo, v, num_bins=_BINS, formed_at=2)
        assert result.poc == pytest.approx(45.0)

    def test_wide_bar_volume_distributed_across_bins(self):
        # Anchor is already a wide bar; add a heavy point bar at bin 4 (centre=45)
        highs   = np.array([100.0, 45.0])
        lows    = np.array([  0.0, 45.0])
        volumes = np.array([ 10.0, 100.0])
        result  = compute_volume_profile(highs, lows, volumes, num_bins=_BINS, formed_at=1)
        assert result.poc == pytest.approx(45.0)


# ──────────────────────────────────────────────────────────────────────────────
# Value area boundaries
# ──────────────────────────────────────────────────────────────────────────────
#
# Grid is [0, 100], bw=10 (anchor=True).
# VAL = price_low + va_lo_bin * bw       (bottom edge of lowest VA bin)
# VAH = price_low + (va_hi_bin + 1) * bw (top edge of highest VA bin)

class TestValueArea:
    def test_val_le_poc_le_vah(self):
        # Invariant check with anchored grid
        h, lo, v = _point_bars([(5.0, 20.0), (55.0, 100.0), (95.0, 20.0)], anchor=True)
        r = compute_volume_profile(h, lo, v, num_bins=_BINS, formed_at=2)
        assert r.val <= r.poc <= r.vah

    def test_value_area_pct_one_covers_full_range(self):
        # Single uniform wide bar → all bins equal → VA with pct=1.0 covers [0, 100]
        h   = np.array([100.0])
        lo  = np.array([  0.0])
        v   = np.array([ 10.0])
        r = compute_volume_profile(h, lo, v, num_bins=_BINS,
                                   value_area_pct=1.0, formed_at=0)
        assert r.val == pytest.approx(0.0)
        assert r.vah == pytest.approx(100.0)

    def test_single_bin_dominates_va_is_one_bin_wide(self):
        # anchor + heavy point bar at bin 4 (centre=45)
        # Total ≈ 1 + 95 = 96; target = 0.7 * 96 = 67.2
        # Bin 4 alone has ~95.1 ≥ 67.2 → VA = bin 4 only → [40, 50]
        h, lo, v = _point_bars([(5.0, 5.0), (45.0, 95.0)], anchor=True)
        r = compute_volume_profile(h, lo, v, num_bins=_BINS,
                                   value_area_pct=0.70, formed_at=1)
        assert r.val == pytest.approx(40.0)
        assert r.vah == pytest.approx(50.0)

    def test_va_expands_toward_higher_volume_neighbour(self):
        # anchor + bars at bin3 (35, v=10), bin4 (45, v=50 — POC), bin5 (55, v=40)
        # Total ≈ 101; target = 0.8 * 101 = 80.8
        # POC bin4 alone ≈ 50.1 < 80.8; bin5 (40.1) > bin3 (10.1) → expand UP
        # After adding bin5: va_vol ≈ 90.2 ≥ 80.8 → stop
        # VAH = top of bin5 = 60; VAL = bottom of bin4 = 40
        h, lo, v = _point_bars(
            [(35.0, 10.0), (45.0, 50.0), (55.0, 40.0)], anchor=True
        )
        r = compute_volume_profile(h, lo, v, num_bins=_BINS,
                                   value_area_pct=0.80, formed_at=2)
        assert r.vah == pytest.approx(60.0)
        assert r.val == pytest.approx(40.0)

    def test_va_forced_to_expand_downward_at_top_bin(self):
        # POC at bin 9 (centre=95, dominant vol); value_area_pct=1.0 → must expand all the way down
        # anchor + point bars at bin0 (5, v=20) and bin9 (95, v=80)
        h, lo, v = _point_bars([(5.0, 20.0), (95.0, 80.0)], anchor=True)
        r = compute_volume_profile(h, lo, v, num_bins=_BINS,
                                   value_area_pct=1.0, formed_at=1)
        assert r.val == pytest.approx(0.0)
        assert r.vah == pytest.approx(100.0)


# ──────────────────────────────────────────────────────────────────────────────
# VolumeProfile helpers
# ──────────────────────────────────────────────────────────────────────────────

class TestVolumeProfileHelpers:
    @pytest.fixture
    def profile(self):
        return VolumeProfile(poc=50_000.0, vah=52_000.0, val=48_000.0, formed_at=10)

    def test_is_near_poc_within_tolerance(self, profile):
        # 0.3 % of 50_000 = 150
        assert profile.is_near_poc(50_100.0)

    def test_is_near_poc_outside_tolerance(self, profile):
        assert not profile.is_near_poc(50_200.0)   # 200 > 150

    def test_is_near_poc_custom_tolerance(self, profile):
        assert profile.is_near_poc(50_500.0, tolerance_pct=0.01)
        assert not profile.is_near_poc(50_500.0, tolerance_pct=0.001)

    def test_is_in_value_area_inside(self, profile):
        assert profile.is_in_value_area(50_000.0)
        assert profile.is_in_value_area(48_000.0)   # boundary
        assert profile.is_in_value_area(52_000.0)   # boundary

    def test_is_in_value_area_outside(self, profile):
        assert not profile.is_in_value_area(47_999.0)
        assert not profile.is_in_value_area(52_001.0)

    def test_is_above_value_area(self, profile):
        assert profile.is_above_value_area(53_000.0)
        assert not profile.is_above_value_area(51_000.0)

    def test_is_below_value_area(self, profile):
        assert profile.is_below_value_area(47_000.0)
        assert not profile.is_below_value_area(50_000.0)


# ──────────────────────────────────────────────────────────────────────────────
# detect_volume_profiles
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectVolumeProfiles:
    def _make_df(self, n=20) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        highs   = 50_000 + rng.uniform(0, 500, n)
        lows    = 50_000 - rng.uniform(0, 500, n)
        volumes = rng.uniform(1, 100, n)
        return _df(highs, lows, volumes)

    def test_step1_returns_one_profile_per_bar(self):
        df       = self._make_df(10)
        profiles = detect_volume_profiles(df, lookback=5, step=1)
        assert len(profiles) == 10

    def test_formed_at_matches_bar_index(self):
        df       = self._make_df(10)
        profiles = detect_volume_profiles(df, lookback=5, step=1)
        for i, p in enumerate(profiles):
            assert p.formed_at == i

    def test_step_controls_output_count(self):
        df       = self._make_df(20)
        profiles = detect_volume_profiles(df, lookback=5, step=5)
        # Bars 0, 5, 10, 15 → 4 profiles
        assert len(profiles) == 4
        assert [p.formed_at for p in profiles] == [0, 5, 10, 15]

    def test_lookback_limits_window(self):
        # Build 20 bars where bar 0 has an extremely high spike
        highs   = np.full(20, 50_100.0)
        lows    = np.full(20, 49_900.0)
        volumes = np.ones(20)
        # Spike at bar 0 only
        highs[0] = 60_000.0
        df = _df(highs, lows, volumes)
        # At bar 19, lookback=5 → window is bars 15-19, spike not included
        profiles = detect_volume_profiles(df, lookback=5, step=1)
        last = profiles[-1]
        # Window is 49_900–50_100; spike at 60_000 must not influence this
        assert last.vah < 52_000.0

    def test_missing_volume_column_uses_unit_volume(self):
        # DataFrame without 'volume' — should still return profiles
        n   = 10
        df  = pd.DataFrame({
            "high":  np.full(n, 50_100.0),
            "low":   np.full(n, 49_900.0),
            "close": np.full(n, 50_000.0),
        })
        profiles = detect_volume_profiles(df, lookback=5, step=1)
        assert len(profiles) == n
        for p in profiles:
            assert p is not None


# ──────────────────────────────────────────────────────────────────────────────
# get_latest_volume_profile
# ──────────────────────────────────────────────────────────────────────────────

class TestGetLatestVolumeProfile:
    @pytest.fixture
    def profiles(self):
        return [
            VolumeProfile(poc=50_000, vah=51_000, val=49_000, formed_at=5),
            VolumeProfile(poc=50_200, vah=51_200, val=49_200, formed_at=10),
            VolumeProfile(poc=50_400, vah=51_400, val=49_400, formed_at=15),
        ]

    def test_returns_latest_at_bar(self, profiles):
        p = get_latest_volume_profile(profiles, at_bar=20)
        assert p.formed_at == 15

    def test_respects_at_bar_cutoff(self, profiles):
        p = get_latest_volume_profile(profiles, at_bar=10)
        assert p.formed_at == 10

    def test_exactly_at_formed_at_is_included(self, profiles):
        p = get_latest_volume_profile(profiles, at_bar=5)
        assert p is not None
        assert p.formed_at == 5

    def test_before_first_profile_returns_none(self, profiles):
        assert get_latest_volume_profile(profiles, at_bar=4) is None

    def test_empty_list_returns_none(self):
        assert get_latest_volume_profile([], at_bar=100) is None
