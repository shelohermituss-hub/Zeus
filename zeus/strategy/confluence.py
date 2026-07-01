"""
Multi-factor confluence scoring engine (factors 1–10).

Aggregates all SMC sub-module outputs for a given bar into a per-direction
score. The strategy uses this score to decide whether to open a trade.

Factors and gate roles
----------------------
#  Name                   Gate?  Active when
1  Market Structure        YES   swing_bias matches direction (trend filter)
2  Fibonacci OTE           no    price inside 61.8–78.6 % retracement zone
3  Order Block             no    price inside an active OB of matching direction
4  Liquidity Sweep         no    sweep confirmed direction within lookback bars
5  FVG / Imbalance         no    price inside an active fair-value gap
6  POC                     no    price within tolerance of Point of Control
7  Kill Zone Session       no    bar timestamp within London (07-11h) or NY (12-15h) UTC
8  Entry Model             YES   internal_bias matches direction (trigger)
9  Discount / Premium      no    price below POC for long; above POC for short
10 Fibonacci 50 %          no    price within tolerance of the 50 % midpoint

Gate factors (1 and 8) MUST both be active for is_tradeable() to return True,
regardless of how many other factors fire.  This prevents entering trades
against the trend or without an internal trigger.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from zeus.strategy.smc.fibonacci import get_latest_fib_zone
from zeus.strategy.smc.fvg import get_active_fvgs
from zeus.strategy.smc.indicator import SMCResult
from zeus.strategy.smc.liquidity import last_sweep
from zeus.strategy.smc.order_block import get_active_order_blocks
from zeus.strategy.smc.pivot import BEARISH, BULLISH
from zeus.strategy.smc.session import get_session_ranges, is_in_killzone, killzone_name


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FactorResult:
    """Evaluation result for a single confluence factor."""
    factor_id: int    # 1–10
    name:      str
    active:    bool   # True when this factor aligns with the trade direction
    detail:    str = ""


class PatternGrade(str, Enum):
    """
    Setup quality tier derived from a ConfluenceScore.

    F means not tradeable (a gate factor is missing or the score is below
    min_score) — an F-graded setup must never be sized or entered. A/B/C
    only ever apply to tradeable setups, ranked by how many of the 10
    factors aligned, so downstream code (position sizing, filtering) can
    treat the grade as a strict quality ordering: A > B > C > F.
    """
    A = "A"
    B = "B"
    C = "C"
    F = "F"


@dataclass(frozen=True)
class ConfluenceScore:
    """
    Complete confluence evaluation for one direction at one bar.

    ``factors`` contains exactly 10 FactorResult entries, indexed 0–9
    (factor_id 1–10).  Use named properties rather than raw index access
    where possible to avoid off-by-one errors.
    """
    direction:  int               # BULLISH (+1) or BEARISH (-1)
    price:      float             # price at time of evaluation
    bar_index:  int
    factors:    list[FactorResult]  # len == 10

    @property
    def active_count(self) -> int:
        """Number of factors currently active."""
        return sum(f.active for f in self.factors)

    @property
    def score(self) -> float:
        """Raw active-factor count (0.0–10.0)."""
        return float(self.active_count)

    @property
    def confidence(self) -> float:
        """Fraction of active factors (0.0–1.0)."""
        n = len(self.factors)
        return self.active_count / n if n else 0.0

    @property
    def has_structure(self) -> bool:
        """Factor 1 gate — swing bias matches direction."""
        return self.factors[0].active

    @property
    def has_entry_confirmation(self) -> bool:
        """Factor 8 gate — internal bias matches direction."""
        return self.factors[7].active

    def is_tradeable(self, min_score: float = 4.0) -> bool:
        """
        True when the score meets the minimum AND both gate factors fire.

        Gates (factors 1 and 8) must be active regardless of total score.
        A high-scoring setup without structure or an internal trigger is not
        entered — this enforces the fail-closed principle at entry.
        """
        return (
            self.has_structure
            and self.has_entry_confirmation
            and self.score >= min_score
        )

    def grade(
        self,
        min_score:   float = 4.0,
        b_threshold: float = 6.0,
        a_threshold: float = 8.0,
    ) -> PatternGrade:
        """
        Classify this setup into a quality tier (A/B/C/F).

        F is returned whenever is_tradeable(min_score) is False — grading
        a non-tradeable setup A, B, or C would let downstream code size or
        enter a trade that the gate logic already rejected, so the two
        checks must never disagree (fail-closed).

        Args:
            min_score:   Same threshold passed to is_tradeable() — the floor
                         for any non-F grade. May exceed b_threshold (e.g. a
                         strategy configured with a strict min_score simply
                         never produces a C grade — that tier becomes
                         unreachable, not invalid).
            b_threshold: Minimum score for a B grade.
            a_threshold: Minimum score for an A grade (the top tier). Must
                         be >= b_threshold.
        """
        if a_threshold < b_threshold:
            raise ValueError(
                "a_threshold must be >= b_threshold "
                f"(got b_threshold={b_threshold}, a_threshold={a_threshold})"
            )
        if not self.is_tradeable(min_score):
            return PatternGrade.F
        if self.score >= a_threshold:
            return PatternGrade.A
        if self.score >= b_threshold:
            return PatternGrade.B
        return PatternGrade.C


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def score_confluence(
    result:               SMCResult,
    price:                float,
    bar_index:            int,
    direction:            int,
    timestamp:            pd.Timestamp | None = None,
    poc_tolerance_pct:    float = 0.003,
    fib_50_tolerance_pct: float = 0.003,
    sweep_lookback:       int   = 10,
    sweep_zone_tol_pct:   float = 0.0,
) -> ConfluenceScore:
    """
    Evaluate all 10 confluence factors for one direction at one bar.

    Args:
        result:               Full SMCResult from indicator.analyze().
        price:                Current close price (or entry price candidate).
        bar_index:            Current bar index for historical look-up gating.
        direction:            BULLISH (+1) or BEARISH (-1).
        timestamp:            UTC timestamp of the current bar — used by F7
                              (Kill Zone Session). When None, F7 is inactive.
        poc_tolerance_pct:    ±% band around POC for factor 6.
        fib_50_tolerance_pct: ±% band around Fib 50 % for factor 10.
        sweep_lookback:       Max bars since last sweep to count (factor 4).
        sweep_zone_tol_pct:   ±% tolerance for sweep-vs-zone-edge alignment
                              (factor 4 intra-zone confirmation). 0.0 disables
                              the zone check (backward-compatible default).

    Returns:
        ConfluenceScore with 10 FactorResult entries.
    """
    factors = [
        _f1_market_structure(result, direction),
        _f2_fibonacci_ote(result, price, bar_index, direction),
        _f3_order_block(result, price, bar_index, direction),
        _f4_liquidity_sweep(result, bar_index, direction, sweep_lookback, sweep_zone_tol_pct),
        _f5_fvg(result, price, bar_index, direction),
        _f6_poc(result, price, poc_tolerance_pct),
        _f7_killzone(timestamp),
        _f8_entry_model(result, direction),
        _f9_discount_premium(result, price, direction),
        _f10_fib_50(result, price, bar_index, direction, fib_50_tolerance_pct),
    ]
    return ConfluenceScore(
        direction=direction,
        price=price,
        bar_index=bar_index,
        factors=factors,
    )


def best_confluence(
    result:    SMCResult,
    price:     float,
    bar_index: int,
    min_score: float = 4.0,
    timestamp: pd.Timestamp | None = None,
    **kwargs,
) -> ConfluenceScore | None:
    """
    Evaluate both directions and return the higher-scoring tradeable signal.

    Returns None when neither direction meets min_score + gate requirements.
    On a score tie the BULLISH direction wins (conservative default).

    Args:
        timestamp: UTC timestamp of the current bar forwarded to F7 (Kill Zone).
    """
    bull = score_confluence(result, price, bar_index, BULLISH, timestamp=timestamp, **kwargs)
    bear = score_confluence(result, price, bar_index, BEARISH, timestamp=timestamp, **kwargs)

    candidates = [s for s in (bull, bear) if s.is_tradeable(min_score)]
    if not candidates:
        return None
    # Prefer higher score; BULLISH (+1) wins on tie (BEARISH = -1 < BULLISH = +1)
    return max(candidates, key=lambda s: (s.score, s.direction))


# ──────────────────────────────────────────────────────────────────────────────
# Factor implementations (private)
# ──────────────────────────────────────────────────────────────────────────────

def _bias_name(bias: int) -> str:
    if bias == BULLISH:
        return "BULLISH"
    if bias == BEARISH:
        return "BEARISH"
    return "NONE"


def _f1_market_structure(result: SMCResult, direction: int) -> FactorResult:
    """Factor 1 (GATE) — swing market structure bias matches the trade direction."""
    active = result.swing_bias == direction
    return FactorResult(
        1, "Market Structure", active,
        f"swing_bias={_bias_name(result.swing_bias)}",
    )


def _f2_fibonacci_ote(
    result: SMCResult, price: float, bar_index: int, direction: int,
) -> FactorResult:
    """Factor 2 — price inside the 61.8–78.6 % Fibonacci OTE retracement zone."""
    zone = get_latest_fib_zone(result.fib_zones, bar_index, direction)
    if zone is None:
        return FactorResult(2, "Fibonacci OTE", False, "no fib zone available")
    active = zone.is_in_ote(price)
    return FactorResult(
        2, "Fibonacci OTE", active,
        f"ote=[{zone.ote_bottom:.2f}, {zone.ote_top:.2f}]",
    )


def _f3_order_block(
    result: SMCResult, price: float, bar_index: int, direction: int,
) -> FactorResult:
    """Factor 3 — price is inside an active order block of the correct direction."""
    all_obs = (
        get_active_order_blocks(result.internal_obs, bar_index)
        + get_active_order_blocks(result.swing_obs, bar_index)
    )
    for ob in all_obs:
        if ob.direction == direction and ob.low <= price <= ob.high:
            return FactorResult(
                3, "Order Block", True,
                f"inside OB [{ob.low:.2f}, {ob.high:.2f}]",
            )
    return FactorResult(3, "Order Block", False, "price outside all active OBs")


def _f4_liquidity_sweep(
    result: SMCResult, bar_index: int, direction: int, lookback: int,
    zone_tol_pct: float = 0.0,
) -> FactorResult:
    """
    Factor 4 — a recent sweep confirms the trade direction.

    When zone_tol_pct > 0 the sweep level must also align with the nearest
    active zone boundary (intra-zone confirmation):
    - BULLISH: SSL sweep level ≤ nearest bullish zone bottom + tolerance
    - BEARISH: BSL sweep level ≥ nearest bearish zone top   - tolerance

    This ensures the liquidity grab happened AT the zone, not elsewhere.
    """
    sweep = last_sweep(result.liquidity_sweeps, bar_index, direction)
    if sweep is None:
        return FactorResult(4, "Liquidity Sweep", False, "no sweep found")

    bars_ago = bar_index - sweep.bar_index
    if bars_ago > lookback:
        return FactorResult(
            4, "Liquidity Sweep", False,
            f"last sweep {bars_ago} bar(s) ago at {sweep.level:.2f} (>{lookback})",
        )

    if zone_tol_pct > 0:
        zone_edge = _nearest_zone_edge(result, bar_index, direction)
        if zone_edge is not None:
            tol = zone_edge * zone_tol_pct
            if direction == BULLISH:
                zone_ok = sweep.level <= zone_edge + tol
            else:
                zone_ok = sweep.level >= zone_edge - tol
            if not zone_ok:
                return FactorResult(
                    4, "Liquidity Sweep", False,
                    f"sweep {sweep.level:.2f} not at zone edge {zone_edge:.2f}",
                )

    return FactorResult(
        4, "Liquidity Sweep", True,
        f"sweep {bars_ago} bar(s) ago at {sweep.level:.2f}",
    )


def _nearest_zone_edge(result: SMCResult, bar_index: int, direction: int) -> float | None:
    """
    Return the relevant zone boundary for sweep alignment (factor 4).

    BULLISH: minimum of active bullish OB lows and FVG bottoms.
             A sweep at or below this level confirms the zone as a reversal.
    BEARISH: maximum of active bearish OB highs and FVG tops.
             A sweep at or above this level confirms the zone as a reversal.

    Returns None when no active zone exists for the given direction.
    """
    edges: list[float] = []

    all_obs = (
        get_active_order_blocks(result.internal_obs, bar_index)
        + get_active_order_blocks(result.swing_obs, bar_index)
    )
    for ob in all_obs:
        if ob.direction == direction:
            edges.append(ob.low if direction == BULLISH else ob.high)

    for fvg in get_active_fvgs(result.fvgs, bar_index):
        if fvg.direction == direction:
            edges.append(fvg.bottom if direction == BULLISH else fvg.top)

    if not edges:
        return None
    return min(edges) if direction == BULLISH else max(edges)


def _f5_fvg(
    result: SMCResult, price: float, bar_index: int, direction: int,
) -> FactorResult:
    """Factor 5 — price is inside an active fair-value gap (imbalance zone)."""
    for fvg in get_active_fvgs(result.fvgs, bar_index):
        if fvg.direction == direction and fvg.bottom <= price <= fvg.top:
            return FactorResult(
                5, "FVG / Imbalance", True,
                f"inside FVG [{fvg.bottom:.2f}, {fvg.top:.2f}]",
            )
    return FactorResult(5, "FVG / Imbalance", False, "no active FVG at price")


def _f6_poc(
    result: SMCResult, price: float, tolerance_pct: float,
) -> FactorResult:
    """Factor 6 — price is within tolerance of the volume Point of Control."""
    if result.volume_profile is None:
        return FactorResult(6, "POC", False, "no volume profile")
    active = result.volume_profile.is_near_poc(price, tolerance_pct)
    return FactorResult(
        6, "POC", active,
        f"poc={result.volume_profile.poc:.2f} tol={tolerance_pct:.2%}",
    )


def _f7_killzone(timestamp: pd.Timestamp | None) -> FactorResult:
    """
    Factor 7 — current bar falls within a high-probability kill zone.

    London Kill Zone : 07:00–11:00 UTC  (Asian→London transition)
    NY Kill Zone     : 12:00–15:00 UTC  (London/NY overlap)

    Trading exclusively within these windows avoids low-liquidity Asian
    sessions where false signals are most frequent for XAUUSD.
    When no timestamp is provided the factor is inactive (safe default).
    """
    if timestamp is None:
        return FactorResult(7, "Kill Zone Session", False, "no timestamp")
    kz = killzone_name(timestamp)
    active = kz is not None
    return FactorResult(
        7, "Kill Zone Session", active,
        f"killzone={kz or 'outside'} ({timestamp.strftime('%H:%M')} UTC)",
    )


def _f8_entry_model(result: SMCResult, direction: int) -> FactorResult:
    """Factor 8 (GATE) — internal structure bias confirms the trade direction."""
    active = result.internal_bias == direction
    return FactorResult(
        8, "Entry Model", active,
        f"internal_bias={_bias_name(result.internal_bias)}",
    )


def _f9_discount_premium(
    result: SMCResult, price: float, direction: int,
) -> FactorResult:
    """
    Factor 9 — price is in the structurally correct price zone.

    BULLISH: discount zone (price ≤ POC, i.e. below fair value).
    BEARISH: premium zone (price ≥ POC, i.e. above fair value).
    """
    if result.volume_profile is None:
        return FactorResult(9, "Discount/Premium Zone", False, "no volume profile")
    vp = result.volume_profile
    if direction == BULLISH:
        active = price <= vp.poc
        label  = "discount"
    else:
        active = price >= vp.poc
        label  = "premium"
    return FactorResult(
        9, "Discount/Premium Zone", active,
        f"{label}: price={price:.2f} poc={vp.poc:.2f}",
    )


def _f10_fib_50(
    result: SMCResult, price: float, bar_index: int,
    direction: int, tolerance_pct: float,
) -> FactorResult:
    """Factor 10 — price is within tolerance of the Fibonacci 50 % midpoint."""
    zone = get_latest_fib_zone(result.fib_zones, bar_index, direction)
    if zone is None:
        return FactorResult(10, "Fibonacci 50%", False, "no fib zone available")
    active = zone.is_near_50(price, tolerance_pct)
    return FactorResult(
        10, "Fibonacci 50%", active,
        f"50%={zone.level_50:.2f} tol={tolerance_pct:.2%}",
    )
