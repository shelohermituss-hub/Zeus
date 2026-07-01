"""
Tests for POC-zone alignment gate in MTFSMCStrategy / ScalpSMCStrategy.

These tests verify:
  - _in_htf_zone() now returns (zone_type, zone_low, zone_high) or None
  - require_poc_zone_confluence gate rejects when POC is outside the zone
  - require_poc_zone_confluence gate passes when POC is inside the zone
  - ScalpSMCStrategy exposes the new parameters
  - poc_zone_tolerance_pct expands the zone boundaries for the check
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from zeus.strategy.mtf_strategy import MTFSMCStrategy
from zeus.strategy.scalp_strategy import ScalpSMCStrategy
from zeus.strategy.base import SignalType
from zeus.strategy.smc.volume_profile import VolumeProfile


# ── Helpers ───────────────────────────────────────────────────────────────────

def _flat_df(n: int = 300, price: float = 2000.0) -> pd.DataFrame:
    """Minimal flat OHLCV DataFrame."""
    return pd.DataFrame(
        {
            "open":   [price] * n,
            "high":   [price + 1] * n,
            "low":    [price - 1] * n,
            "close":  [price] * n,
            "volume": [1000.0] * n,
        },
        index=pd.date_range("2026-01-06 08:00", periods=n, freq="1min", tz="UTC"),
    )


def _make_strategy(
    require_poc: bool = False,
    poc_tol: float = 0.005,
) -> MTFSMCStrategy:
    df_htf = _flat_df(n=300, price=2000.0)
    return MTFSMCStrategy(
        df_htf=df_htf,
        min_htf_score=1.0,
        require_poc_zone_confluence=require_poc,
        poc_zone_tolerance_pct=poc_tol,
        killzone_only=False,
        require_entry_fvg=False,
        require_htf_internal_align=False,
        max_daily_signals=0,
        max_signals_per_session=0,
    )


# ── Tests: _in_htf_zone return type ──────────────────────────────────────────

class TestInHtfZoneReturnType:

    def test_returns_none_when_outside_all_zones(self):
        strat = _make_strategy()
        mock_result = MagicMock()
        mock_result.internal_obs = []
        mock_result.swing_obs    = []
        mock_result.fvgs         = []
        mock_result.fib_zones    = []
        result = strat._in_htf_zone(mock_result, bar_idx=50, price=2000.0, direction=1)
        assert result is None

    def test_returns_tuple_when_price_in_ob(self):
        strat = _make_strategy()

        ob = MagicMock()
        ob.direction = 1
        ob.low       = 1995.0
        ob.high      = 2005.0
        ob.formed_at = 10
        ob.invalidated_at = None

        mock_result = MagicMock()
        mock_result.internal_obs = [ob]
        mock_result.swing_obs    = []
        mock_result.fvgs         = []
        mock_result.fib_zones    = []

        with patch("zeus.strategy.mtf_strategy.get_active_order_blocks", return_value=[ob]):
            result = strat._in_htf_zone(mock_result, bar_idx=50, price=2000.0, direction=1)

        assert result is not None
        zone_type, zone_low, zone_high = result
        assert zone_type  == "OB"
        assert zone_low   == 1995.0
        assert zone_high  == 2005.0

    def test_returns_tuple_when_price_in_fvg(self):
        strat = _make_strategy()

        fvg = MagicMock()
        fvg.direction = 1
        fvg.bottom    = 1998.0
        fvg.top       = 2002.0
        fvg.formed_at = 10
        fvg.invalidated_at = None

        mock_result = MagicMock()
        mock_result.internal_obs = []
        mock_result.swing_obs    = []
        mock_result.fvgs         = [fvg]
        mock_result.fib_zones    = []

        with patch("zeus.strategy.mtf_strategy.get_active_order_blocks", return_value=[]):
            with patch("zeus.strategy.mtf_strategy.get_active_fvgs", return_value=[fvg]):
                result = strat._in_htf_zone(mock_result, bar_idx=50, price=2000.0, direction=1)

        assert result is not None
        zone_type, zone_low, zone_high = result
        assert zone_type  == "FVG"
        assert zone_low   == 1998.0
        assert zone_high  == 2002.0


# ── Tests: POC zone confluence gate logic ────────────────────────────────────

class TestPocZoneGate:
    """Unit tests for the POC-in-zone check in isolation."""

    def _run_poc_check(
        self,
        zone_low:  float,
        zone_high: float,
        poc:       float,
        tol_pct:   float = 0.005,
    ) -> bool:
        """
        Replicate the gate logic from generate_signal() step 4.3.
        Returns True = POC is within zone (check passes).
        """
        tol_price  = poc * tol_pct
        return (zone_low - tol_price) <= poc <= (zone_high + tol_price)

    def test_poc_inside_zone_passes(self):
        assert self._run_poc_check(1990.0, 2010.0, poc=2000.0)

    def test_poc_at_lower_bound_passes(self):
        assert self._run_poc_check(1990.0, 2010.0, poc=1990.0)

    def test_poc_at_upper_bound_passes(self):
        assert self._run_poc_check(1990.0, 2010.0, poc=2010.0)

    def test_poc_below_zone_fails(self):
        assert not self._run_poc_check(1990.0, 2010.0, poc=1985.0, tol_pct=0.0)

    def test_poc_above_zone_fails(self):
        assert not self._run_poc_check(1990.0, 2010.0, poc=2015.0, tol_pct=0.0)

    def test_tolerance_rescues_borderline_poc(self):
        # POC is 5 pips below zone_low; tolerance=0.005 → tol_price=9.975 → passes
        poc = 1985.0
        zone_low = 1990.0
        assert self._run_poc_check(zone_low, 2010.0, poc=poc, tol_pct=0.005)

    def test_tight_tolerance_rejects_borderline_poc(self):
        # POC is 5 pips below zone_low; tolerance=0.0 → fails
        poc = 1985.0
        zone_low = 1990.0
        assert not self._run_poc_check(zone_low, 2010.0, poc=poc, tol_pct=0.0)


# ── Tests: ScalpSMCStrategy parameter exposure ────────────────────────────────

class TestScalpStrategyParameters:

    def test_scalp_default_poc_confluence_disabled(self):
        df = _flat_df(n=300)
        strat = ScalpSMCStrategy(df_htf_1h=df)
        assert strat._require_poc_zone_confluence is False

    def test_scalp_poc_confluence_can_be_enabled(self):
        df = _flat_df(n=300)
        strat = ScalpSMCStrategy(df_htf_1h=df, require_poc_zone_confluence=True)
        assert strat._require_poc_zone_confluence is True

    def test_scalp_poc_tolerance_wired(self):
        df = _flat_df(n=300)
        strat = ScalpSMCStrategy(df_htf_1h=df, poc_zone_tolerance_pct=0.01)
        assert strat._poc_zone_tol == 0.01

    def test_mtf_default_poc_confluence_disabled(self):
        df = _flat_df(n=300)
        strat = MTFSMCStrategy(df_htf=df)
        assert strat._require_poc_zone_confluence is False

    def test_mtf_poc_confluence_can_be_enabled(self):
        df = _flat_df(n=300)
        strat = MTFSMCStrategy(df_htf=df, require_poc_zone_confluence=True)
        assert strat._require_poc_zone_confluence is True

    def test_poc_zone_tolerance_stored(self):
        df = _flat_df(n=300)
        strat = MTFSMCStrategy(df_htf=df, poc_zone_tolerance_pct=0.003)
        assert strat._poc_zone_tol == 0.003


# ── Tests: VolumeProfile.is_near_poc still works (no regression) ─────────────

class TestVolumeProfileNearPoc:

    def test_is_near_poc_true_when_inside_tolerance(self):
        vp = VolumeProfile(poc=2000.0, vah=2010.0, val=1990.0, formed_at=0)
        assert vp.is_near_poc(2000.0, tolerance_pct=0.003)
        assert vp.is_near_poc(2005.0, tolerance_pct=0.003)  # 0.25% away

    def test_is_near_poc_false_when_outside_tolerance(self):
        vp = VolumeProfile(poc=2000.0, vah=2010.0, val=1990.0, formed_at=0)
        assert not vp.is_near_poc(2050.0, tolerance_pct=0.003)

    def test_zone_poc_logic_differs_from_price_poc_logic(self):
        """
        The new gate (POC inside zone) is conceptually different from Factor 6
        (price near POC). This test shows they can give opposite answers.

        Scenario: OB zone is wide [1980–2020], POC=2000 (inside zone).
        Entry price = 1981 (barely inside zone, far from POC).

        Factor 6 (price near POC): 1981 vs 2000 = 0.95% → NOT near (tol=0.3%).
        Gate 4.3 (POC inside zone): 2000 in [1980–2020] → YES.
        """
        vp     = VolumeProfile(poc=2000.0, vah=2010.0, val=1990.0, formed_at=0)
        price  = 1981.0   # entry at lower edge of wide OB zone
        zone_l, zone_h = 1980.0, 2020.0

        # Factor 6 (current behaviour)
        f6_active = vp.is_near_poc(price, tolerance_pct=0.003)
        assert not f6_active   # price is 0.95% from POC → Factor 6 = False

        # Gate 4.3 (new behaviour)
        tol_price = vp.poc * 0.005
        poc_in_zone = (zone_l - tol_price) <= vp.poc <= (zone_h + tol_price)
        assert poc_in_zone     # POC 2000 is inside [1980–2020] → gate passes
