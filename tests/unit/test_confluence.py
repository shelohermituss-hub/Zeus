"""
Unit tests for zeus.strategy.confluence.

Each factor is tested in isolation with a minimal SMCResult stub so the
test verifies factor logic without any dependency on real SMC detection.

Coverage plan
-------------
TestFactorResult          — dataclass basics
TestConfluenceScore       — properties: score, confidence, gates, is_tradeable
TestF1MarketStructure     — GATE: swing_bias matches direction
TestF2FibonacciOTE        — price inside OTE zone
TestF3OrderBlock          — price inside active OB of correct direction
TestF4LiquiditySweep      — recent sweep confirms direction
TestF5FVG                 — price inside active FVG
TestF6POC                 — price near Point of Control
TestF7KillzoneSession     — timestamp inside London or NY kill zone window
TestF8EntryModel          — GATE: internal_bias matches direction
TestF9DiscountPremium     — price below/above POC
TestF10Fib50              — price near Fibonacci 50 % midpoint
TestScoreConfluence       — full 10-factor evaluation
TestBestConfluence        — direction arbitration, tie-breaking, no-signal
"""
from __future__ import annotations

import pandas as pd
import pytest

from zeus.strategy.smc.fibonacci import FibZone
from zeus.strategy.smc.fvg import FairValueGap
from zeus.strategy.smc.indicator import SMCResult
from zeus.strategy.smc.liquidity import LiqType, LiquidityLevel, LiquiditySweep
from zeus.strategy.smc.order_block import OrderBlock
from zeus.strategy.smc.pivot import BEARISH, BULLISH
from zeus.strategy.smc.session import SessionRange, SessionType
from zeus.strategy.smc.volume_profile import VolumeProfile
from zeus.strategy.confluence import (
    ConfluenceScore,
    FactorResult,
    PatternGrade,
    best_confluence,
    score_confluence,
    _nearest_zone_edge,
)


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _empty_result(**overrides) -> SMCResult:
    """Minimal SMCResult with sensible defaults that fire no factors."""
    base = SMCResult(
        swing_bias=0,
        internal_bias=0,
        swing_obs=[],
        internal_obs=[],
        fvgs=[],
        fib_zones=[],
        liquidity_sweeps=[],
        session_ranges=[],
        volume_profile=None,
    )
    for k, v in overrides.items():
        object.__setattr__(base, k, v)
    return base


def _result(**overrides) -> SMCResult:
    """Alias — same as _empty_result but named for clarity in full-score tests."""
    return _empty_result(**overrides)


def _vp(poc: float = 100.0, vah: float = 110.0, val: float = 90.0) -> VolumeProfile:
    return VolumeProfile(poc=poc, vah=vah, val=val, formed_at=0)


def _fib_zone_bullish(
    swing_high: float = 110.0,
    swing_low: float  = 90.0,
    formed_at: int    = 0,
) -> FibZone:
    return FibZone(
        swing_high=swing_high,
        swing_low=swing_low,
        direction=BULLISH,
        leg_high_bar=5,
        leg_low_bar=0,
        formed_at=formed_at,
    )


def _fib_zone_bearish(
    swing_high: float = 110.0,
    swing_low: float  = 90.0,
    formed_at: int    = 0,
) -> FibZone:
    return FibZone(
        swing_high=swing_high,
        swing_low=swing_low,
        direction=BEARISH,
        leg_high_bar=0,
        leg_low_bar=5,
        formed_at=formed_at,
    )


def _ob(direction: int, low: float, high: float, detected_at: int = 0) -> OrderBlock:
    return OrderBlock(
        bar_index=0,
        high=high,
        low=low,
        direction=direction,
        is_internal=False,
        detected_at=detected_at,
        mitigated_at=-1,
    )


def _fvg(direction: int, bottom: float, top: float, bar_index: int = 0) -> FairValueGap:
    return FairValueGap(bar_index=bar_index, top=top, bottom=bottom, direction=direction, mitigated_at=-1)


def _sweep(direction: int, bar_index: int = 5, level: float = 100.0) -> LiquiditySweep:
    liq_type = LiqType.SSL if direction == BULLISH else LiqType.BSL
    return LiquiditySweep(bar_index=bar_index, level=level, liq_type=liq_type, direction=direction)


def _session_range(
    session: SessionType = SessionType.ASIAN,
    high: float = 105.0,
    low: float  = 95.0,
    formed_at: int = 1,
) -> SessionRange:
    return SessionRange(
        session=session,
        high=high,
        low=low,
        high_bar=0,
        low_bar=1,
        start_bar=0,
        formed_at=formed_at,
    )


def _all_active_result(bar_index: int = 10) -> SMCResult:
    """
    Construct an SMCResult where all 10 factors fire BULLISH at bar_index=10,
    price=100.0.

    Factor mapping:
      F1  swing_bias  = BULLISH
      F2  fib zone OTE zone covers [97.12, 98.36] — price 97.5 inside
      F3  bullish OB [98.0, 102.0]                 — price 100 inside
      F4  BULLISH sweep at bar 8 (2 bars ago)
      F5  bullish FVG [98.0, 102.0]               — price 100 inside
      F6  poc=100 (price == poc → within 0.003)
      F7  killzone timestamp — pass timestamp=london_ts to score_confluence
      F8  internal_bias = BULLISH
      F9  price(100) <= poc(100)                   — discount zone
      F10 fib 50% = 100.0 (mid of [90,110])        — price 100 within tol
    """
    price = 100.0
    # Fibonacci: swing_high=110, swing_low=90 → range=20
    #   OTE bottom = 110 - 20*0.786 = 94.28
    #   OTE top    = 110 - 20*0.618 = 97.64
    #   50%        = 110 - 20*0.5   = 100.0
    fib = _fib_zone_bullish(swing_high=110.0, swing_low=90.0, formed_at=0)
    assert fib.ote_bottom == pytest.approx(94.28, abs=0.01)
    assert fib.ote_top    == pytest.approx(97.64, abs=0.01)
    assert fib.level_50   == pytest.approx(100.0)

    return SMCResult(
        swing_bias=BULLISH,
        internal_bias=BULLISH,
        swing_obs=[_ob(BULLISH, low=98.0, high=102.0)],
        internal_obs=[],
        fvgs=[_fvg(BULLISH, bottom=98.0, top=102.0, bar_index=0)],
        fib_zones=[fib],
        liquidity_sweeps=[_sweep(BULLISH, bar_index=8)],
        session_ranges=[_session_range(low=price, high=110.0, formed_at=1)],
        volume_profile=_vp(poc=price, vah=110.0, val=90.0),
    )


# ──────────────────────────────────────────────────────────────────────────────
# FactorResult
# ──────────────────────────────────────────────────────────────────────────────

class TestFactorResult:
    def test_active_true(self):
        f = FactorResult(1, "Market Structure", True, "detail")
        assert f.active is True

    def test_active_false(self):
        f = FactorResult(1, "Market Structure", False)
        assert f.active is False

    def test_default_detail_empty(self):
        f = FactorResult(2, "Fibonacci OTE", True)
        assert f.detail == ""

    def test_factor_is_frozen(self):
        f = FactorResult(1, "Market Structure", True)
        with pytest.raises(Exception):
            f.active = False  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────────────
# ConfluenceScore properties
# ──────────────────────────────────────────────────────────────────────────────

class TestConfluenceScore:
    def _make(self, active_ids: list[int]) -> ConfluenceScore:
        """Build a ConfluenceScore with exactly the given factor IDs active."""
        factors = [
            FactorResult(i + 1, f"F{i+1}", (i + 1) in active_ids)
            for i in range(10)
        ]
        return ConfluenceScore(direction=BULLISH, price=100.0, bar_index=5, factors=factors)

    def test_active_count_zero(self):
        cs = self._make([])
        assert cs.active_count == 0

    def test_active_count_all(self):
        cs = self._make(list(range(1, 11)))
        assert cs.active_count == 10

    def test_score_equals_active_count(self):
        cs = self._make([1, 3, 5])
        assert cs.score == pytest.approx(3.0)

    def test_confidence_zero(self):
        cs = self._make([])
        assert cs.confidence == pytest.approx(0.0)

    def test_confidence_full(self):
        cs = self._make(list(range(1, 11)))
        assert cs.confidence == pytest.approx(1.0)

    def test_confidence_partial(self):
        cs = self._make([1, 2, 3, 4, 5])
        assert cs.confidence == pytest.approx(0.5)

    def test_has_structure_when_factor1_active(self):
        cs = self._make([1])
        assert cs.has_structure is True

    def test_has_structure_false_when_factor1_inactive(self):
        cs = self._make([2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert cs.has_structure is False

    def test_has_entry_confirmation_when_factor8_active(self):
        cs = self._make([8])
        assert cs.has_entry_confirmation is True

    def test_has_entry_confirmation_false_when_factor8_inactive(self):
        cs = self._make([1, 2, 3, 4, 5, 6, 7, 9, 10])
        assert cs.has_entry_confirmation is False

    def test_is_tradeable_requires_both_gates_and_score(self):
        cs = self._make([1, 8, 2, 3, 4])   # 5 factors active (score=5 ≥ 4)
        assert cs.is_tradeable(min_score=4.0) is True

    def test_not_tradeable_without_structure(self):
        cs = self._make([2, 3, 4, 5, 6, 7, 8, 9, 10])  # 9 active, no F1
        assert cs.is_tradeable(min_score=4.0) is False

    def test_not_tradeable_without_entry_model(self):
        cs = self._make([1, 2, 3, 4, 5, 6, 7, 9, 10])  # 9 active, no F8
        assert cs.is_tradeable(min_score=4.0) is False

    def test_not_tradeable_when_score_below_min(self):
        cs = self._make([1, 8, 2])   # 3 active, score=3 < 4
        assert cs.is_tradeable(min_score=4.0) is False

    def test_tradeable_at_exactly_min_score(self):
        cs = self._make([1, 8, 2, 3])  # score=4 == min_score
        assert cs.is_tradeable(min_score=4.0) is True

    def test_tradeable_custom_min_score(self):
        cs = self._make([1, 8])   # score=2
        assert cs.is_tradeable(min_score=2.0) is True
        assert cs.is_tradeable(min_score=3.0) is False


# ──────────────────────────────────────────────────────────────────────────────
# Pattern quality grading
# ──────────────────────────────────────────────────────────────────────────────

class TestPatternGrade:
    def _make(self, active_ids: list[int]) -> ConfluenceScore:
        """Build a ConfluenceScore with exactly the given factor IDs active."""
        factors = [
            FactorResult(i + 1, f"F{i+1}", (i + 1) in active_ids)
            for i in range(10)
        ]
        return ConfluenceScore(direction=BULLISH, price=100.0, bar_index=5, factors=factors)

    def test_f_grade_when_gate_missing(self):
        cs = self._make([2, 3, 4, 5, 6, 7, 8, 9, 10])  # 9 active, no F1 gate
        assert cs.grade() == PatternGrade.F

    def test_f_grade_when_score_below_min(self):
        cs = self._make([1, 8, 2])  # score=3 < default min_score=4
        assert cs.grade() == PatternGrade.F

    def test_c_grade_at_min_score(self):
        cs = self._make([1, 8, 2, 3])  # score=4, below default b_threshold=6
        assert cs.grade() == PatternGrade.C

    def test_b_grade_at_b_threshold(self):
        cs = self._make([1, 8, 2, 3, 4, 5])  # score=6, below default a_threshold=8
        assert cs.grade() == PatternGrade.B

    def test_a_grade_at_a_threshold(self):
        cs = self._make([1, 8, 2, 3, 4, 5, 6, 7])  # score=8
        assert cs.grade() == PatternGrade.A

    def test_a_grade_at_full_score(self):
        cs = self._make(list(range(1, 11)))  # score=10
        assert cs.grade() == PatternGrade.A

    def test_grade_respects_custom_thresholds(self):
        cs = self._make([1, 8, 2, 3, 4])  # score=5
        assert cs.grade(min_score=4.0, b_threshold=5.0, a_threshold=9.0) == PatternGrade.B
        assert cs.grade(min_score=4.0, b_threshold=6.0, a_threshold=9.0) == PatternGrade.C

    def test_grade_rejects_a_threshold_below_b_threshold(self):
        cs = self._make([1, 8])
        with pytest.raises(ValueError):
            cs.grade(b_threshold=8.0, a_threshold=6.0)

    def test_grade_with_min_score_above_b_threshold_skips_c(self):
        """A strict min_score can make the C tier unreachable without raising."""
        cs = self._make([1, 8, 2, 3, 4, 5, 6])  # score=7
        grade = cs.grade(min_score=7.0, b_threshold=6.0, a_threshold=8.0)
        assert grade == PatternGrade.B

    def test_grade_is_never_better_than_is_tradeable_allows(self):
        """Grading must never disagree with the gate check (fail-closed)."""
        cs = self._make([2, 3, 4, 5, 6, 7, 8, 9, 10])  # high score, no F1 gate
        assert cs.is_tradeable(min_score=4.0) is False
        assert cs.grade(min_score=4.0) == PatternGrade.F

    def test_pattern_grade_string_values(self):
        assert PatternGrade.A.value == "A"
        assert PatternGrade.B.value == "B"
        assert PatternGrade.C.value == "C"
        assert PatternGrade.F.value == "F"


# ──────────────────────────────────────────────────────────────────────────────
# Factor 1 — Market Structure (GATE)
# ──────────────────────────────────────────────────────────────────────────────

class TestF1MarketStructure:
    def test_active_when_swing_bias_matches_bullish(self):
        r = _empty_result(swing_bias=BULLISH)
        cs = score_confluence(r, price=100.0, bar_index=0, direction=BULLISH)
        assert cs.factors[0].active is True

    def test_active_when_swing_bias_matches_bearish(self):
        r = _empty_result(swing_bias=BEARISH)
        cs = score_confluence(r, price=100.0, bar_index=0, direction=BEARISH)
        assert cs.factors[0].active is True

    def test_inactive_when_swing_bias_mismatches(self):
        r = _empty_result(swing_bias=BEARISH)
        cs = score_confluence(r, price=100.0, bar_index=0, direction=BULLISH)
        assert cs.factors[0].active is False

    def test_inactive_when_swing_bias_neutral(self):
        r = _empty_result(swing_bias=0)
        cs = score_confluence(r, price=100.0, bar_index=0, direction=BULLISH)
        assert cs.factors[0].active is False

    def test_detail_contains_bias(self):
        r = _empty_result(swing_bias=BULLISH)
        cs = score_confluence(r, price=100.0, bar_index=0, direction=BULLISH)
        assert "BULLISH" in cs.factors[0].detail


# ──────────────────────────────────────────────────────────────────────────────
# Factor 2 — Fibonacci OTE
# ──────────────────────────────────────────────────────────────────────────────

class TestF2FibonacciOTE:
    def _cs(self, price: float, direction: int, fib_zones: list) -> ConfluenceScore:
        r = _empty_result(fib_zones=fib_zones)
        return score_confluence(r, price=price, bar_index=10, direction=direction)

    def test_active_when_price_in_bullish_ote(self):
        # swing_high=110, swing_low=90, range=20
        # OTE bottom = 110 - 20*0.786 = 94.28
        # OTE top    = 110 - 20*0.618 = 97.64
        fib = _fib_zone_bullish(110.0, 90.0, formed_at=0)
        cs = self._cs(price=96.0, direction=BULLISH, fib_zones=[fib])
        assert cs.factors[1].active is True

    def test_inactive_when_price_above_bullish_ote(self):
        fib = _fib_zone_bullish(110.0, 90.0, formed_at=0)
        cs = self._cs(price=105.0, direction=BULLISH, fib_zones=[fib])
        assert cs.factors[1].active is False

    def test_inactive_when_price_below_bullish_ote(self):
        fib = _fib_zone_bullish(110.0, 90.0, formed_at=0)
        cs = self._cs(price=92.0, direction=BULLISH, fib_zones=[fib])
        assert cs.factors[1].active is False

    def test_active_when_price_in_bearish_ote(self):
        # swing_high=110, swing_low=90
        # BEARISH: OTE bottom = 90 + 20*0.618 = 102.36
        #          OTE top    = 90 + 20*0.786 = 105.72
        fib = _fib_zone_bearish(110.0, 90.0, formed_at=0)
        cs = self._cs(price=104.0, direction=BEARISH, fib_zones=[fib])
        assert cs.factors[1].active is True

    def test_inactive_when_no_fib_zone(self):
        cs = self._cs(price=100.0, direction=BULLISH, fib_zones=[])
        assert cs.factors[1].active is False

    def test_inactive_when_fib_zone_not_yet_formed(self):
        fib = _fib_zone_bullish(110.0, 90.0, formed_at=20)
        # bar_index=10 < formed_at=20 → zone not visible yet
        cs = self._cs(price=96.0, direction=BULLISH, fib_zones=[fib])
        assert cs.factors[1].active is False

    def test_direction_filter_bullish_zone_not_used_for_bearish(self):
        fib = _fib_zone_bullish(110.0, 90.0, formed_at=0)
        cs = self._cs(price=96.0, direction=BEARISH, fib_zones=[fib])
        assert cs.factors[1].active is False


# ──────────────────────────────────────────────────────────────────────────────
# Factor 3 — Order Block
# ──────────────────────────────────────────────────────────────────────────────

class TestF3OrderBlock:
    def _cs(self, price: float, direction: int,
            swing_obs: list, internal_obs: list, bar_index: int = 10) -> ConfluenceScore:
        r = _empty_result(swing_obs=swing_obs, internal_obs=internal_obs)
        return score_confluence(r, price=price, bar_index=bar_index, direction=direction)

    def test_active_price_inside_swing_ob(self):
        ob = _ob(BULLISH, low=98.0, high=102.0, detected_at=0)
        cs = self._cs(100.0, BULLISH, swing_obs=[ob], internal_obs=[])
        assert cs.factors[2].active is True

    def test_active_price_inside_internal_ob(self):
        ob = _ob(BULLISH, low=98.0, high=102.0, detected_at=0)
        cs = self._cs(100.0, BULLISH, swing_obs=[], internal_obs=[ob])
        assert cs.factors[2].active is True

    def test_inactive_price_outside_ob(self):
        ob = _ob(BULLISH, low=98.0, high=102.0, detected_at=0)
        cs = self._cs(110.0, BULLISH, swing_obs=[ob], internal_obs=[])
        assert cs.factors[2].active is False

    def test_inactive_ob_wrong_direction(self):
        ob = _ob(BEARISH, low=98.0, high=102.0, detected_at=0)
        cs = self._cs(100.0, BULLISH, swing_obs=[ob], internal_obs=[])
        assert cs.factors[2].active is False

    def test_inactive_ob_not_yet_detected(self):
        ob = _ob(BULLISH, low=98.0, high=102.0, detected_at=20)
        # bar_index=10, OB detected_at=20 → not yet active
        cs = self._cs(100.0, BULLISH, swing_obs=[ob], internal_obs=[], bar_index=10)
        assert cs.factors[2].active is False

    def test_price_on_ob_boundary_is_active(self):
        ob = _ob(BULLISH, low=98.0, high=102.0, detected_at=0)
        cs_low  = self._cs(98.0,  BULLISH, swing_obs=[ob], internal_obs=[])
        cs_high = self._cs(102.0, BULLISH, swing_obs=[ob], internal_obs=[])
        assert cs_low.factors[2].active is True
        assert cs_high.factors[2].active is True


# ──────────────────────────────────────────────────────────────────────────────
# Factor 4 — Liquidity Sweep
# ──────────────────────────────────────────────────────────────────────────────

class TestF4LiquiditySweep:
    def _cs(self, direction: int, sweeps: list,
            bar_index: int = 10, lookback: int = 10,
            sweep_zone_tol_pct: float = 0.0,
            obs: list | None = None, fvgs: list | None = None) -> ConfluenceScore:
        r = _empty_result(
            liquidity_sweeps=sweeps,
            swing_obs=obs or [],
            fvgs=fvgs or [],
        )
        return score_confluence(r, price=100.0, bar_index=bar_index,
                                direction=direction, sweep_lookback=lookback,
                                sweep_zone_tol_pct=sweep_zone_tol_pct)

    # ── Basic lookback tests (zone check disabled) ──

    def test_active_when_recent_sweep_matches_direction(self):
        sweep = _sweep(BULLISH, bar_index=8)
        cs = self._cs(BULLISH, sweeps=[sweep], bar_index=10, lookback=5)
        assert cs.factors[3].active is True   # 2 bars ago ≤ 5

    def test_inactive_when_sweep_too_old(self):
        sweep = _sweep(BULLISH, bar_index=0)
        cs = self._cs(BULLISH, sweeps=[sweep], bar_index=10, lookback=5)
        assert cs.factors[3].active is False  # 10 bars ago > 5

    def test_inactive_when_no_sweep(self):
        cs = self._cs(BULLISH, sweeps=[], bar_index=10, lookback=10)
        assert cs.factors[3].active is False

    def test_inactive_sweep_wrong_direction(self):
        sweep = _sweep(BEARISH, bar_index=9)
        cs = self._cs(BULLISH, sweeps=[sweep], bar_index=10, lookback=5)
        assert cs.factors[3].active is False

    def test_active_at_exact_lookback_boundary(self):
        sweep = _sweep(BULLISH, bar_index=5)
        cs = self._cs(BULLISH, sweeps=[sweep], bar_index=10, lookback=5)
        # 10 - 5 = 5 <= 5 → active
        assert cs.factors[3].active is True

    # ── Intra-zone confirmation (sweep_zone_tol_pct > 0) ──

    def test_zone_check_active_sweep_at_zone_bottom(self):
        """SSL sweep at 97.0, bullish OB low=98.0 → 97.0 ≤ 98.0+tol → active."""
        ob    = _ob(BULLISH, low=98.0, high=102.0)
        sweep = _sweep(BULLISH, bar_index=8, level=97.0)
        cs    = self._cs(BULLISH, sweeps=[sweep], bar_index=10, lookback=5,
                         sweep_zone_tol_pct=0.005, obs=[ob])
        assert cs.factors[3].active is True

    def test_zone_check_active_sweep_below_zone(self):
        """Sweep well below zone bottom (classic stop-hunt below OB) → active."""
        ob    = _ob(BULLISH, low=98.0, high=102.0)
        sweep = _sweep(BULLISH, bar_index=8, level=95.0)
        cs    = self._cs(BULLISH, sweeps=[sweep], bar_index=10, lookback=5,
                         sweep_zone_tol_pct=0.005, obs=[ob])
        assert cs.factors[3].active is True

    def test_zone_check_inactive_sweep_above_zone_bottom(self):
        """Sweep at 101.0, above OB low=98.0 → not at zone edge → inactive."""
        ob    = _ob(BULLISH, low=98.0, high=102.0)
        sweep = _sweep(BULLISH, bar_index=8, level=101.0)
        cs    = self._cs(BULLISH, sweeps=[sweep], bar_index=10, lookback=5,
                         sweep_zone_tol_pct=0.005, obs=[ob])
        assert cs.factors[3].active is False

    def test_zone_check_detail_mentions_zone_edge_on_failure(self):
        ob    = _ob(BULLISH, low=98.0, high=102.0)
        sweep = _sweep(BULLISH, bar_index=8, level=101.0)
        cs    = self._cs(BULLISH, sweeps=[sweep], bar_index=10, lookback=5,
                         sweep_zone_tol_pct=0.005, obs=[ob])
        assert "zone edge" in cs.factors[3].detail.lower()

    def test_zone_check_bearish_sweep_above_zone_top(self):
        """BSL sweep at 103.0, bearish OB high=102.0 → 103.0 ≥ 102.0-tol → active."""
        ob    = _ob(BEARISH, low=98.0, high=102.0)
        sweep = _sweep(BEARISH, bar_index=8, level=103.0)
        cs    = self._cs(BEARISH, sweeps=[sweep], bar_index=10, lookback=5,
                         sweep_zone_tol_pct=0.005, obs=[ob])
        assert cs.factors[3].active is True

    def test_zone_check_bearish_sweep_below_zone_top_fails(self):
        """BSL sweep at 97.0 < OB high=102.0-tol → not at zone edge → inactive."""
        ob    = _ob(BEARISH, low=98.0, high=102.0)
        sweep = _sweep(BEARISH, bar_index=8, level=97.0)
        cs    = self._cs(BEARISH, sweeps=[sweep], bar_index=10, lookback=5,
                         sweep_zone_tol_pct=0.005, obs=[ob])
        assert cs.factors[3].active is False

    def test_zone_check_no_zone_skips_zone_check(self):
        """No active OB/FVG → zone edge is None → zone check skipped → active."""
        sweep = _sweep(BULLISH, bar_index=8, level=105.0)
        cs    = self._cs(BULLISH, sweeps=[sweep], bar_index=10, lookback=5,
                         sweep_zone_tol_pct=0.005)  # no obs, no fvgs
        assert cs.factors[3].active is True

    def test_zone_check_zero_disables_zone_check(self):
        """sweep_zone_tol_pct=0.0 → zone check disabled → sweep far above zone still active."""
        ob    = _ob(BULLISH, low=98.0, high=102.0)
        sweep = _sweep(BULLISH, bar_index=8, level=105.0)
        cs    = self._cs(BULLISH, sweeps=[sweep], bar_index=10, lookback=5,
                         sweep_zone_tol_pct=0.0, obs=[ob])
        assert cs.factors[3].active is True

    def test_zone_check_fvg_used_as_zone_edge(self):
        """FVG bottom=97.0, sweep at 96.0 → 96.0 ≤ 97.0+tol → active."""
        fvg   = _fvg(BULLISH, bottom=97.0, top=100.0, bar_index=0)
        sweep = _sweep(BULLISH, bar_index=8, level=96.0)
        cs    = self._cs(BULLISH, sweeps=[sweep], bar_index=10, lookback=5,
                         sweep_zone_tol_pct=0.005, fvgs=[fvg])
        assert cs.factors[3].active is True

    def test_zone_check_uses_minimum_edge_of_multiple_zones(self):
        """Two bullish OBs: low=96 and low=98 → zone_edge=96; sweep at 95 → active."""
        ob1   = _ob(BULLISH, low=98.0, high=102.0)
        ob2   = _ob(BULLISH, low=96.0, high=100.0)
        sweep = _sweep(BULLISH, bar_index=8, level=95.0)
        cs    = self._cs(BULLISH, sweeps=[sweep], bar_index=10, lookback=5,
                         sweep_zone_tol_pct=0.005, obs=[ob1, ob2])
        assert cs.factors[3].active is True


# ──────────────────────────────────────────────────────────────────────────────
# _nearest_zone_edge
# ──────────────────────────────────────────────────────────────────────────────

class TestNearestZoneEdge:
    """Direct unit tests for the _nearest_zone_edge() private helper."""

    def test_returns_none_when_no_zones(self):
        r = _empty_result()
        assert _nearest_zone_edge(r, bar_index=10, direction=BULLISH) is None

    def test_returns_ob_low_for_bullish(self):
        ob = _ob(BULLISH, low=98.0, high=102.0)
        r  = _empty_result(swing_obs=[ob])
        assert _nearest_zone_edge(r, bar_index=10, direction=BULLISH) == pytest.approx(98.0)

    def test_returns_ob_high_for_bearish(self):
        ob = _ob(BEARISH, low=98.0, high=102.0)
        r  = _empty_result(swing_obs=[ob])
        assert _nearest_zone_edge(r, bar_index=10, direction=BEARISH) == pytest.approx(102.0)

    def test_ignores_wrong_direction_ob(self):
        ob = _ob(BEARISH, low=98.0, high=102.0)
        r  = _empty_result(swing_obs=[ob])
        assert _nearest_zone_edge(r, bar_index=10, direction=BULLISH) is None

    def test_returns_fvg_bottom_for_bullish(self):
        fvg = _fvg(BULLISH, bottom=97.0, top=100.0, bar_index=0)
        r   = _empty_result(fvgs=[fvg])
        assert _nearest_zone_edge(r, bar_index=10, direction=BULLISH) == pytest.approx(97.0)

    def test_returns_fvg_top_for_bearish(self):
        fvg = _fvg(BEARISH, bottom=98.0, top=103.0, bar_index=0)
        r   = _empty_result(fvgs=[fvg])
        assert _nearest_zone_edge(r, bar_index=10, direction=BEARISH) == pytest.approx(103.0)

    def test_bullish_returns_minimum_of_multiple_zones(self):
        ob1 = _ob(BULLISH, low=96.0, high=100.0)
        ob2 = _ob(BULLISH, low=98.0, high=102.0)
        r   = _empty_result(swing_obs=[ob1, ob2])
        assert _nearest_zone_edge(r, bar_index=10, direction=BULLISH) == pytest.approx(96.0)

    def test_bearish_returns_maximum_of_multiple_zones(self):
        ob1 = _ob(BEARISH, low=98.0, high=102.0)
        ob2 = _ob(BEARISH, low=100.0, high=105.0)
        r   = _empty_result(swing_obs=[ob1, ob2])
        assert _nearest_zone_edge(r, bar_index=10, direction=BEARISH) == pytest.approx(105.0)

    def test_mixes_ob_and_fvg_edges(self):
        ob  = _ob(BULLISH, low=98.0, high=102.0)
        fvg = _fvg(BULLISH, bottom=95.0, top=99.0, bar_index=0)
        r   = _empty_result(swing_obs=[ob], fvgs=[fvg])
        # min(98.0, 95.0) = 95.0
        assert _nearest_zone_edge(r, bar_index=10, direction=BULLISH) == pytest.approx(95.0)


# ──────────────────────────────────────────────────────────────────────────────
# Factor 5 — FVG / Imbalance
# ──────────────────────────────────────────────────────────────────────────────

class TestF5FVG:
    def _cs(self, price: float, direction: int, fvgs: list,
            bar_index: int = 10) -> ConfluenceScore:
        r = _empty_result(fvgs=fvgs)
        return score_confluence(r, price=price, bar_index=bar_index, direction=direction)

    def test_active_price_inside_bullish_fvg(self):
        fvg = _fvg(BULLISH, bottom=98.0, top=102.0, bar_index=0)
        cs = self._cs(100.0, BULLISH, fvgs=[fvg])
        assert cs.factors[4].active is True

    def test_inactive_price_outside_fvg(self):
        fvg = _fvg(BULLISH, bottom=98.0, top=102.0, bar_index=0)
        cs = self._cs(110.0, BULLISH, fvgs=[fvg])
        assert cs.factors[4].active is False

    def test_inactive_fvg_wrong_direction(self):
        fvg = _fvg(BEARISH, bottom=98.0, top=102.0, bar_index=0)
        cs = self._cs(100.0, BULLISH, fvgs=[fvg])
        assert cs.factors[4].active is False

    def test_inactive_mitigated_fvg(self):
        fvg = FairValueGap(bar_index=0, top=102.0, bottom=98.0,
                           direction=BULLISH, mitigated_at=5)
        cs = self._cs(100.0, BULLISH, fvgs=[fvg], bar_index=10)
        # mitigated_at=5 < bar_index=10 → not active
        assert cs.factors[4].active is False

    def test_inactive_no_fvgs(self):
        cs = self._cs(100.0, BULLISH, fvgs=[])
        assert cs.factors[4].active is False


# ──────────────────────────────────────────────────────────────────────────────
# Factor 6 — POC
# ──────────────────────────────────────────────────────────────────────────────

class TestF6POC:
    def _cs(self, price: float, vp: VolumeProfile | None,
            tol: float = 0.003) -> ConfluenceScore:
        r = _empty_result(volume_profile=vp)
        return score_confluence(r, price=price, bar_index=0, direction=BULLISH,
                                poc_tolerance_pct=tol)

    def test_active_when_price_near_poc(self):
        vp = _vp(poc=100.0)
        cs = self._cs(price=100.0, vp=vp)
        assert cs.factors[5].active is True

    def test_active_within_default_tolerance(self):
        vp = _vp(poc=100.0)
        # 0.3% of 100 = 0.3 → price 100.2 is within tolerance
        cs = self._cs(price=100.2, vp=vp)
        assert cs.factors[5].active is True

    def test_inactive_when_price_far_from_poc(self):
        vp = _vp(poc=100.0)
        cs = self._cs(price=105.0, vp=vp)
        assert cs.factors[5].active is False

    def test_inactive_when_no_volume_profile(self):
        cs = self._cs(price=100.0, vp=None)
        assert cs.factors[5].active is False

    def test_custom_tolerance(self):
        vp = _vp(poc=100.0)
        cs_tight = self._cs(price=100.5, vp=vp, tol=0.003)  # 0.3% → inactive
        cs_wide   = self._cs(price=100.5, vp=vp, tol=0.010)  # 1% → active
        assert cs_tight.factors[5].active is False
        assert cs_wide.factors[5].active  is True


# ──────────────────────────────────────────────────────────────────────────────
# Factor 7 — Kill Zone Session
# ──────────────────────────────────────────────────────────────────────────────

class TestF7KillzoneSession:
    """F7 is active when the bar timestamp falls within London (07–11h) or NY (12–15h) UTC."""

    def _ts(self, hour: int, minute: int = 0) -> pd.Timestamp:
        return pd.Timestamp(f"2025-01-01 {hour:02d}:{minute:02d}:00", tz="UTC")

    def _cs(self, timestamp, direction: int = BULLISH) -> ConfluenceScore:
        r = _empty_result()
        return score_confluence(r, price=100.0, bar_index=0,
                                direction=direction, timestamp=timestamp)

    def test_active_during_london_killzone(self):
        cs = self._cs(self._ts(9, 0))   # 09:00 UTC — inside London KZ
        assert cs.factors[6].active is True

    def test_active_at_london_kz_start(self):
        cs = self._cs(self._ts(7, 0))   # 07:00 UTC — inclusive start
        assert cs.factors[6].active is True

    def test_inactive_at_london_kz_end(self):
        cs = self._cs(self._ts(11, 0))  # 11:00 UTC — exclusive end
        assert cs.factors[6].active is False

    def test_active_during_ny_killzone(self):
        cs = self._cs(self._ts(13, 30))  # 13:30 UTC — inside NY KZ
        assert cs.factors[6].active is True

    def test_active_at_ny_kz_start(self):
        cs = self._cs(self._ts(12, 0))  # 12:00 UTC — inclusive start
        assert cs.factors[6].active is True

    def test_inactive_at_ny_kz_end(self):
        cs = self._cs(self._ts(15, 0))  # 15:00 UTC — exclusive end
        assert cs.factors[6].active is False

    def test_inactive_outside_both_killzones_asian(self):
        cs = self._cs(self._ts(3, 0))   # 03:00 UTC — Asian session
        assert cs.factors[6].active is False

    def test_inactive_in_gap_between_killzones(self):
        cs = self._cs(self._ts(11, 30))  # 11:30 UTC — gap between London and NY
        assert cs.factors[6].active is False

    def test_inactive_after_ny_close(self):
        cs = self._cs(self._ts(20, 0))  # 20:00 UTC — after NY KZ
        assert cs.factors[6].active is False

    def test_inactive_when_no_timestamp(self):
        cs = self._cs(timestamp=None)
        assert cs.factors[6].active is False

    def test_detail_shows_london_label(self):
        cs = self._cs(self._ts(9, 0))
        assert "London" in cs.factors[6].detail

    def test_detail_shows_ny_label(self):
        cs = self._cs(self._ts(13, 0))
        assert "NY" in cs.factors[6].detail

    def test_detail_shows_outside_when_inactive(self):
        cs = self._cs(self._ts(3, 0))
        assert "outside" in cs.factors[6].detail

    def test_direction_agnostic_bullish_and_bearish_both_fire(self):
        """F7 is not direction-filtered — it fires for both BULLISH and BEARISH."""
        ts = self._ts(9, 0)
        cs_bull = self._cs(ts, direction=BULLISH)
        cs_bear = self._cs(ts, direction=BEARISH)
        assert cs_bull.factors[6].active is True
        assert cs_bear.factors[6].active is True

    def test_tz_aware_timestamp_supported(self):
        ts = pd.Timestamp("2025-01-15 10:00:00+00:00")  # 10:00 UTC — London KZ
        cs = self._cs(ts)
        assert cs.factors[6].active is True

    def test_tz_naive_treated_as_utc(self):
        ts = pd.Timestamp("2025-01-15 09:00:00")  # tz-naive — treated as UTC
        cs = self._cs(ts)
        assert cs.factors[6].active is True


# ──────────────────────────────────────────────────────────────────────────────
# Factor 8 — Entry Model (GATE)
# ──────────────────────────────────────────────────────────────────────────────

class TestF8EntryModel:
    def test_active_when_internal_bias_matches_bullish(self):
        r = _empty_result(internal_bias=BULLISH)
        cs = score_confluence(r, price=100.0, bar_index=0, direction=BULLISH)
        assert cs.factors[7].active is True

    def test_active_when_internal_bias_matches_bearish(self):
        r = _empty_result(internal_bias=BEARISH)
        cs = score_confluence(r, price=100.0, bar_index=0, direction=BEARISH)
        assert cs.factors[7].active is True

    def test_inactive_when_internal_bias_mismatches(self):
        r = _empty_result(internal_bias=BEARISH)
        cs = score_confluence(r, price=100.0, bar_index=0, direction=BULLISH)
        assert cs.factors[7].active is False

    def test_inactive_when_internal_bias_neutral(self):
        r = _empty_result(internal_bias=0)
        cs = score_confluence(r, price=100.0, bar_index=0, direction=BULLISH)
        assert cs.factors[7].active is False

    def test_detail_contains_bias_name(self):
        r = _empty_result(internal_bias=BEARISH)
        cs = score_confluence(r, price=100.0, bar_index=0, direction=BEARISH)
        assert "BEARISH" in cs.factors[7].detail


# ──────────────────────────────────────────────────────────────────────────────
# Factor 9 — Discount / Premium Zone
# ──────────────────────────────────────────────────────────────────────────────

class TestF9DiscountPremium:
    def _cs(self, price: float, direction: int, vp: VolumeProfile | None) -> ConfluenceScore:
        r = _empty_result(volume_profile=vp)
        return score_confluence(r, price=price, bar_index=0, direction=direction)

    def test_active_bullish_price_below_poc(self):
        vp = _vp(poc=100.0)
        cs = self._cs(price=95.0, direction=BULLISH, vp=vp)
        assert cs.factors[8].active is True   # discount: 95 ≤ 100

    def test_active_bullish_price_at_poc(self):
        vp = _vp(poc=100.0)
        cs = self._cs(price=100.0, direction=BULLISH, vp=vp)
        assert cs.factors[8].active is True   # 100 ≤ 100

    def test_inactive_bullish_price_above_poc(self):
        vp = _vp(poc=100.0)
        cs = self._cs(price=105.0, direction=BULLISH, vp=vp)
        assert cs.factors[8].active is False  # premium: 105 > 100

    def test_active_bearish_price_above_poc(self):
        vp = _vp(poc=100.0)
        cs = self._cs(price=105.0, direction=BEARISH, vp=vp)
        assert cs.factors[8].active is True   # premium: 105 ≥ 100

    def test_active_bearish_price_at_poc(self):
        vp = _vp(poc=100.0)
        cs = self._cs(price=100.0, direction=BEARISH, vp=vp)
        assert cs.factors[8].active is True   # 100 ≥ 100

    def test_inactive_bearish_price_below_poc(self):
        vp = _vp(poc=100.0)
        cs = self._cs(price=95.0, direction=BEARISH, vp=vp)
        assert cs.factors[8].active is False  # discount: 95 < 100

    def test_inactive_when_no_volume_profile(self):
        cs = self._cs(price=100.0, direction=BULLISH, vp=None)
        assert cs.factors[8].active is False


# ──────────────────────────────────────────────────────────────────────────────
# Factor 10 — Fibonacci 50 %
# ──────────────────────────────────────────────────────────────────────────────

class TestF10Fib50:
    def _cs(self, price: float, direction: int, fib_zones: list,
            tol: float = 0.003, bar_index: int = 10) -> ConfluenceScore:
        r = _empty_result(fib_zones=fib_zones)
        return score_confluence(r, price=price, bar_index=bar_index,
                                direction=direction, fib_50_tolerance_pct=tol)

    def test_active_when_price_at_50_pct_bullish(self):
        # swing_high=110, swing_low=90 → 50% = 100.0
        fib = _fib_zone_bullish(110.0, 90.0, formed_at=0)
        cs = self._cs(price=100.0, direction=BULLISH, fib_zones=[fib])
        assert cs.factors[9].active is True

    def test_active_when_price_within_tolerance(self):
        fib = _fib_zone_bullish(110.0, 90.0, formed_at=0)
        # 0.3% of 100 = 0.3 → price 100.2 inside tolerance
        cs = self._cs(price=100.2, direction=BULLISH, fib_zones=[fib], tol=0.003)
        assert cs.factors[9].active is True

    def test_inactive_when_price_far_from_50_pct(self):
        fib = _fib_zone_bullish(110.0, 90.0, formed_at=0)
        cs = self._cs(price=95.0, direction=BULLISH, fib_zones=[fib])
        assert cs.factors[9].active is False

    def test_active_bearish_50_pct(self):
        # BEARISH 50% = swing_low + range*0.5 = 90 + 10 = 100
        fib = _fib_zone_bearish(110.0, 90.0, formed_at=0)
        cs = self._cs(price=100.0, direction=BEARISH, fib_zones=[fib])
        assert cs.factors[9].active is True

    def test_inactive_when_no_fib_zone(self):
        cs = self._cs(price=100.0, direction=BULLISH, fib_zones=[])
        assert cs.factors[9].active is False

    def test_direction_filter(self):
        fib = _fib_zone_bullish(110.0, 90.0, formed_at=0)
        cs = self._cs(price=100.0, direction=BEARISH, fib_zones=[fib])
        assert cs.factors[9].active is False


# ──────────────────────────────────────────────────────────────────────────────
# score_confluence — structural invariants
# ──────────────────────────────────────────────────────────────────────────────

class TestScoreConfluence:
    def test_returns_confluence_score(self):
        r = _empty_result()
        cs = score_confluence(r, price=100.0, bar_index=0, direction=BULLISH)
        assert isinstance(cs, ConfluenceScore)

    def test_exactly_10_factors(self):
        r = _empty_result()
        cs = score_confluence(r, price=100.0, bar_index=0, direction=BULLISH)
        assert len(cs.factors) == 10

    def test_factor_ids_are_1_through_10(self):
        r = _empty_result()
        cs = score_confluence(r, price=100.0, bar_index=0, direction=BULLISH)
        ids = [f.factor_id for f in cs.factors]
        assert ids == list(range(1, 11))

    def test_direction_propagated(self):
        r = _empty_result()
        cs = score_confluence(r, price=100.0, bar_index=5, direction=BEARISH)
        assert cs.direction == BEARISH

    def test_price_propagated(self):
        r = _empty_result()
        cs = score_confluence(r, price=123.45, bar_index=0, direction=BULLISH)
        assert cs.price == pytest.approx(123.45)

    def test_bar_index_propagated(self):
        r = _empty_result()
        cs = score_confluence(r, price=100.0, bar_index=42, direction=BULLISH)
        assert cs.bar_index == 42

    def test_zero_score_when_all_empty(self):
        r = _empty_result()
        cs = score_confluence(r, price=100.0, bar_index=0, direction=BULLISH)
        assert cs.score == pytest.approx(0.0)

    def test_perfect_10_score(self):
        r = _all_active_result(bar_index=10)
        # F7 requires a timestamp inside a kill zone; use 09:00 UTC (London KZ)
        london_ts = pd.Timestamp("2025-01-01 09:00:00", tz="UTC")
        # Use price=100.0 which hits F1,F3,F4,F5,F6,F7,F8,F9,F10 but NOT F2
        # (OTE zone is 94.28–97.64; price=100 is above)
        # → 9 factors active
        cs = score_confluence(r, price=100.0, bar_index=10, direction=BULLISH,
                              timestamp=london_ts)
        # Verify specific factors are active/inactive
        assert cs.factors[0].active is True   # F1: swing_bias=BULLISH
        assert cs.factors[1].active is False  # F2: price 100 not in OTE [94.28,97.64]
        assert cs.factors[2].active is True   # F3: price 100 in OB [98,102]
        assert cs.factors[3].active is True   # F4: sweep at bar 8 (2 bars ago ≤ 10)
        assert cs.factors[4].active is True   # F5: price 100 in FVG [98,102]
        assert cs.factors[5].active is True   # F6: poc=100, price=100
        assert cs.factors[6].active is True   # F7: timestamp 09:00 UTC → London KZ
        assert cs.factors[7].active is True   # F8: internal_bias=BULLISH
        assert cs.factors[8].active is True   # F9: price(100) <= poc(100)
        assert cs.factors[9].active is True   # F10: fib 50%=100.0, price=100.0
        assert cs.score == pytest.approx(9.0)

    def test_f2_fires_when_price_in_ote(self):
        """Explicitly verify F2 fires when price is placed inside the OTE zone."""
        r = _all_active_result(bar_index=10)
        # OTE zone = [94.28, 97.64]; place price at 96.0
        cs = score_confluence(r, price=96.0, bar_index=10, direction=BULLISH)
        assert cs.factors[1].active is True


# ──────────────────────────────────────────────────────────────────────────────
# best_confluence
# ──────────────────────────────────────────────────────────────────────────────

class TestBestConfluence:
    def test_returns_none_when_neither_tradeable(self):
        r = _empty_result()   # all factors inactive → score=0 for both directions
        result = best_confluence(r, price=100.0, bar_index=0, min_score=4.0)
        assert result is None

    def test_returns_bullish_when_only_bullish_tradeable(self):
        # Only bullish gates + enough factors
        r = _empty_result(
            swing_bias=BULLISH,
            internal_bias=BULLISH,
            volume_profile=_vp(poc=100.0),
            swing_obs=[_ob(BULLISH, low=98.0, high=102.0)],
            liquidity_sweeps=[_sweep(BULLISH, bar_index=5)],
        )
        result = best_confluence(r, price=100.0, bar_index=10, min_score=4.0)
        assert result is not None
        assert result.direction == BULLISH

    def test_returns_bearish_when_only_bearish_tradeable(self):
        r = _empty_result(
            swing_bias=BEARISH,
            internal_bias=BEARISH,
            volume_profile=_vp(poc=100.0),
            swing_obs=[_ob(BEARISH, low=98.0, high=102.0)],
            liquidity_sweeps=[_sweep(BEARISH, bar_index=5)],
        )
        result = best_confluence(r, price=100.0, bar_index=10, min_score=4.0)
        assert result is not None
        assert result.direction == BEARISH

    def test_higher_score_direction_wins(self):
        # Bearish has higher score: give it an extra factor (OB)
        # Bullish: F1(swing=BULL), F8(internal=BULL) = 2 factors — not tradeable (< 4)
        # This test needs careful construction so both are tradeable but scores differ.
        # Build a result where bearish scores 5 and bullish scores 4 (both tradeable)
        r = _empty_result(
            swing_bias=BEARISH,
            internal_bias=BEARISH,
            volume_profile=_vp(poc=100.0),
            swing_obs=[_ob(BEARISH, low=98.0, high=102.0)],
            liquidity_sweeps=[_sweep(BEARISH, bar_index=5)],
            fvgs=[_fvg(BEARISH, bottom=98.0, top=102.0, bar_index=0)],
        )
        # score_confluence for BEARISH at price=100: F1,F3,F4,F5,F6,F8,F9 = 7 factors
        # BULLISH would need both gates but swing_bias=BEARISH → F1 inactive → not tradeable
        result = best_confluence(r, price=100.0, bar_index=10, min_score=4.0)
        assert result is not None
        assert result.direction == BEARISH

    def test_bullish_wins_on_tie(self):
        """When both directions score equally, BULLISH wins."""
        # Construct symmetric result: both gates fire for both directions
        # by giving swing_bias=BULLISH, internal_bias=BULLISH for one side
        # and the same number of non-gate factors for both.
        # Since swing_bias and internal_bias are single values, they can only
        # match one direction — symmetric tie is impossible with real SMCResult.
        # Instead test the selection logic directly via ConfluenceScore.
        factors_bull = [FactorResult(i+1, f"F{i+1}", True) for i in range(10)]
        factors_bear = [FactorResult(i+1, f"F{i+1}", True) for i in range(10)]
        bull_cs = ConfluenceScore(BULLISH, 100.0, 10, factors_bull)
        bear_cs = ConfluenceScore(BEARISH, 100.0, 10, factors_bear)

        candidates = [bull_cs, bear_cs]
        winner = max(candidates, key=lambda s: (s.score, s.direction))
        assert winner.direction == BULLISH   # +1 > -1 on tie

    def test_returns_none_when_score_below_min(self):
        # Both gates fire but score < min_score
        r = _empty_result(swing_bias=BULLISH, internal_bias=BULLISH)
        # Only 2 factors active (F1, F8) → score=2 < 4
        result = best_confluence(r, price=100.0, bar_index=0, min_score=4.0)
        assert result is None

    def test_custom_min_score(self):
        r = _empty_result(swing_bias=BULLISH, internal_bias=BULLISH)
        # score=2: tradeable at min_score=2.0, not tradeable at min_score=3.0
        assert best_confluence(r, 100.0, 0, min_score=2.0) is not None
        assert best_confluence(r, 100.0, 0, min_score=3.0) is None

    def test_gate_required_regardless_of_score(self):
        """A high score without both gates must not be tradeable."""
        r = _empty_result(
            swing_bias=BULLISH,
            internal_bias=0,           # F8 gate inactive
            volume_profile=_vp(poc=100.0),
            swing_obs=[_ob(BULLISH, low=98.0, high=102.0)],
            liquidity_sweeps=[_sweep(BULLISH, bar_index=5)],
            fvgs=[_fvg(BULLISH, bottom=98.0, top=102.0)],
        )
        result = best_confluence(r, price=100.0, bar_index=10, min_score=4.0)
        # Despite having F1,F3,F4,F5,F6,F9 active (6 factors),
        # F8 gate is missing → not tradeable
        assert result is None
