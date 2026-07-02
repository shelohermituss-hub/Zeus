"""
Harmonic pattern detection.

Translates the LuxAlgo Harmonic Pattern Detection (Pine Script) logic to Python.
Detects Bat, Gartley, Butterfly, Crab patterns from M15 pivot sequences.

Pattern structure
-----------------
  Bullish : X(low) → A(high) → B(low) → C(high) → D(PRZ, projected low)
  Bearish : X(high) → A(low) → B(high) → C(low) → D(PRZ, projected high)

Fibonacci rules (from LuxAlgo Pine Script)
-------------------------------------------
  Bat       : AB ∈ (0.382, 0.500), BC ∈ (0.382, 0.886), D=0.886·XA, CD ∈ (1.618, 2.618)
  Gartley   : AB ≈ 0.618,          BC ∈ (0.382, 0.886), D=0.786·XA, CD ∈ (1.130, 1.618)
  Butterfly : AB ≈ 0.786,          BC ∈ (0.382, 0.886), D=1.270·XA, CD ∈ (1.618, 2.240)
  Crab      : AB ∈ (0.382, 0.618), BC ∈ (0.382, 0.886), D=1.618·XA, CD ∈ (2.224, 3.618)

PRZ (Potential Reversal Zone)
------------------------------
  Matches Pine Script: D ± 0.382·|D − X|

Causal safety
--------------
  Each HarmonicPattern carries confirmed_at = c_pivot.confirmed_at.
  The strategy must not act on any bar < confirmed_at.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from zeus.strategy.smc.pivot import detect_pivots, PivotPoint


# ── Ratio helpers ─────────────────────────────────────────────────────────────

def _ab_ratio(x: float, a: float, b: float) -> float:
    """Retracement of B within XA: |AB| / |XA|."""
    xa = abs(x - a)
    return abs(b - a) / xa if xa > 1e-10 else 0.0


def _bc_ratio(a: float, b: float, c: float) -> float:
    """Retracement/extension of C within AB: |BC| / |AB|."""
    ab = abs(a - b)
    return abs(c - b) / ab if ab > 1e-10 else 0.0


def _cd_ratio(b: float, c: float, d: float) -> float:
    """Extension of CD within BC: |CD| / |BC|."""
    bc = abs(c - b)
    return abs(d - c) / bc if bc > 1e-10 else 0.0


def _project_d(a: float, x: float, fib: float) -> float:
    """Project D: a + fib*(x−a). Matches Pine Script d_y = a_y + fib*(x_y − a_y)."""
    return a + fib * (x - a)


def _prz(d: float, x: float) -> tuple[float, float]:
    """PRZ = D ± 0.382·|D − X|. Matches Pine Script upper/lower_prz."""
    margin = 0.382 * abs(d - x)
    return d - margin, d + margin


# ── Pattern dataclass ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HarmonicPattern:
    """A detected harmonic pattern waiting for PRZ touch at D."""
    pattern_type: str    # "Bat" | "Gartley" | "Butterfly" | "Crab"
    direction:    str    # "bullish" | "bearish"
    # Pivot bar indices in the M15 DataFrame
    x_bar:        int
    a_bar:        int
    b_bar:        int
    c_bar:        int
    confirmed_at: int    # c_pivot.confirmed_at — earliest safe bar for entry
    # Pivot prices
    x_price:      float
    a_price:      float
    b_price:      float
    c_price:      float
    d_price:      float  # projected D (center of PRZ)
    prz_low:      float
    prz_high:     float
    # Diagnostics
    ab_ratio:     float
    bc_ratio:     float
    cd_ratio:     float


# ── Pattern matching ──────────────────────────────────────────────────────────

def _try_match(
    x: float,
    a: float,
    b: float,
    c: float,
    precision: float = 0.03,
) -> tuple[str, float, float, float] | None:
    """
    Try to match prices (X, A, B, C) to a harmonic pattern.

    Checks patterns in priority order matching Pine Script if/else-if logic:
    Bat → Gartley → Butterfly → Crab.

    Returns (pattern_type, d_price, prz_low, prz_high) or None.
    """
    ab = _ab_ratio(x, a, b)
    bc = _bc_ratio(a, b, c)

    # ── Bat: AB ∈ (0.382, 0.500) ──────────────────────────────────────────
    if 0.382 <= ab <= 0.500 and 0.382 <= bc <= 0.886:
        d  = _project_d(a, x, 0.886)
        cd = _cd_ratio(b, c, d)
        if 1.618 <= cd <= 2.618:
            lo, hi = _prz(d, x)
            return "Bat", d, lo, hi

    # ── Gartley: AB ≈ 0.618 ───────────────────────────────────────────────
    if abs(ab - 0.618) <= precision and 0.382 <= bc <= 0.886:
        d  = _project_d(a, x, 0.786)
        cd = _cd_ratio(b, c, d)
        if 1.130 <= cd <= 1.618:
            lo, hi = _prz(d, x)
            return "Gartley", d, lo, hi

    # ── Butterfly: AB ≈ 0.786 ─────────────────────────────────────────────
    if abs(ab - 0.786) <= precision and 0.382 <= bc <= 0.886:
        d  = _project_d(a, x, 1.270)
        cd = _cd_ratio(b, c, d)
        if 1.618 <= cd <= 2.240:
            lo, hi = _prz(d, x)
            return "Butterfly", d, lo, hi

    # ── Crab: AB ∈ (0.382, 0.618) ─────────────────────────────────────────
    if 0.382 <= ab <= 0.618 and 0.382 <= bc <= 0.886:
        d  = _project_d(a, x, 1.618)
        cd = _cd_ratio(b, c, d)
        if 2.224 <= cd <= 3.618:
            lo, hi = _prz(d, x)
            return "Crab", d, lo, hi

    return None


# ── Public detector ───────────────────────────────────────────────────────────

def detect_harmonics(
    df:         pd.DataFrame,
    pivot_size: int   = 5,
    precision:  float = 0.03,
    long_only:  bool  = True,
) -> list[HarmonicPattern]:
    """
    Detect harmonic patterns from a M15 OHLCV DataFrame.

    Scans every consecutive group of 4 pivots (X, A, B, C) from
    detect_pivots(), which already guarantees alternating HIGH/LOW order.
    Projects D and computes the PRZ for each valid match.

    Parameters
    ----------
    df         : M15 OHLCV DataFrame with DatetimeIndex
    pivot_size : bars required on each side for pivot confirmation (default 5)
    precision  : AB ratio tolerance for Gartley (≈0.618) and Butterfly (≈0.786)
    long_only  : if True, return only bullish patterns (default True)
    """
    pivots = detect_pivots(df["high"], df["low"], size=pivot_size)
    if len(pivots) < 4:
        return []

    patterns: list[HarmonicPattern] = []

    for i in range(len(pivots) - 3):
        px, pa, pb, pc = pivots[i], pivots[i + 1], pivots[i + 2], pivots[i + 3]

        # detect_pivots guarantees alternation; direction from X pivot type
        if px.is_high:
            direction = "bearish"
        else:
            direction = "bullish"

        if long_only and direction != "bullish":
            continue

        x_p, a_p, b_p, c_p = px.level, pa.level, pb.level, pc.level

        result = _try_match(x_p, a_p, b_p, c_p, precision=precision)
        if result is None:
            continue

        ptype, d_price, prz_low, prz_high = result

        # Sanity: for bullish D must be below C (the last swing high)
        #         for bearish D must be above C (the last swing low)
        if direction == "bullish" and d_price >= c_p:
            continue
        if direction == "bearish" and d_price <= c_p:
            continue

        ab = _ab_ratio(x_p, a_p, b_p)
        bc = _bc_ratio(a_p, b_p, c_p)
        cd = _cd_ratio(b_p, c_p, d_price)

        patterns.append(HarmonicPattern(
            pattern_type = ptype,
            direction    = direction,
            x_bar        = px.bar_index,
            a_bar        = pa.bar_index,
            b_bar        = pb.bar_index,
            c_bar        = pc.bar_index,
            confirmed_at = pc.confirmed_at,
            x_price      = x_p,
            a_price      = a_p,
            b_price      = b_p,
            c_price      = c_p,
            d_price      = d_price,
            prz_low      = prz_low,
            prz_high     = prz_high,
            ab_ratio     = ab,
            bc_ratio     = bc,
            cd_ratio     = cd,
        ))

    return patterns
