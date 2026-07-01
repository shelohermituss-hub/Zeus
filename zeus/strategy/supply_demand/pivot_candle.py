"""
Pivot candle detection for Supply & Demand zone analysis.

A pivot candle is the last candle in an accumulation base just before an
impulsive move (rally or drop). It shows which side won the "battle":

    Demand pivot  — small body + long LOWER wick  → buyers took control
    Supply pivot  — small body + long UPPER wick  → sellers took control

The pivot candle defines the precise S&D zone boundaries, enabling a tight
stop-loss (just beyond the wick tip) for a high R:R entry.

Zone levels
-----------
    Demand pivot:
        zone_top    = body_high   (entry: wait for price to return here)
        zone_bottom = body_low    (standard SL reference)
        wick_low    = wick_low    (tight SL: just below the rejection wick)

    Supply pivot:
        zone_bottom = body_low    (entry: wait for price to return here)
        zone_top    = body_high   (standard SL reference)
        wick_high   = wick_high   (tight SL: just above the rejection wick)

Usage
-----
    from zeus.strategy.supply_demand.pivot_candle import (
        analyze_candle, find_last_pivot_candle, scan_for_pivots, PivotSide,
    )

    pc = analyze_candle(ts, o, h, l, c)
    if pc.side == PivotSide.DEMAND and pc.score >= 6:
        print(f"Strong demand pivot — entry zone: {pc.body_low}–{pc.body_high}")
        print(f"Tight SL below: {pc.wick_low}")
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────

_MIN_RANGE   = 1e-8   # avoid div-by-zero on flat candles
MIN_SCORE    = 3.0    # below this → DOJI (no side assigned)
MIN_WICK_R   = 0.25   # rejection wick must cover ≥ 25 % of candle range
DOMINANCE_R  = 1.5    # dominant wick must be ≥ 1.5× the opposite wick


# ── Data types ───────────────────────────────────────────────────────────────

class PivotSide(Enum):
    DEMAND  = "demand"   # long lower wick  → buyers won → demand zone
    SUPPLY  = "supply"   # long upper wick  → sellers won → supply zone
    DOJI    = "doji"     # ambiguous / indecision (both wicks similar)


@dataclass(frozen=True)
class PivotCandle:
    """
    Result of analyzing a single OHLC candle as a potential pivot.

    Attributes
    ----------
    index            : timestamp of the candle
    open/high/low/close : raw OHLC values
    side             : DEMAND, SUPPLY, or DOJI
    score            : 0–10, higher = cleaner pivot (use ≥ 5 for high-quality)
    body_high        : max(open, close)  — top of body
    body_low         : min(open, close)  — bottom of body
    wick_high        : == high           — tip of upper wick
    wick_low         : == low            — tip of lower wick
    upper_wick_ratio : upper wick / candle range
    lower_wick_ratio : lower wick / candle range
    body_ratio       : body / candle range
    """
    index:            pd.Timestamp
    open:             float
    high:             float
    low:              float
    close:            float
    side:             PivotSide
    score:            float
    body_high:        float
    body_low:         float
    wick_high:        float
    wick_low:         float
    upper_wick_ratio: float
    lower_wick_ratio: float
    body_ratio:       float


# ── Internal helpers ─────────────────────────────────────────────────────────

def _decompose(
    o: float, h: float, l: float, c: float,
) -> tuple[float, float, float, float, float]:
    """
    Decompose OHLC into structural components.

    Returns
    -------
    (range, body, upper_wick, lower_wick, body_midpoint_position)
        body_midpoint_position: 0.0 = at candle bottom, 1.0 = at top
    """
    rng   = h - l
    if rng < _MIN_RANGE:
        return rng, 0.0, 0.0, 0.0, 0.5

    body_h = max(o, c)
    body_l = min(o, c)
    body   = body_h - body_l
    upper  = h - body_h
    lower  = body_l - l
    body_mid = ((body_h + body_l) / 2.0 - l) / rng   # 0 = bottom, 1 = top

    return rng, body, upper, lower, body_mid


def _score_demand(
    rng: float, body: float, upper: float, lower: float, body_mid: float,
) -> float:
    """
    Score a candle as a DEMAND pivot (0–10).

    High score = pin bar with long lower wick + small body near the top.

    Scoring breakdown:
        4 pts — long lower wick (≥ 50 % of range = max)
        2 pts — small body (< 15 % = max, > 30 % = 0)
        2 pts — body positioned near the top of the range
        2 pts — tiny upper wick (< 10 % = max)
    """
    if rng < _MIN_RANGE:
        return 0.0

    lower_r = lower / rng
    upper_r = upper / rng
    body_r  = body  / rng

    pts  = 0.0
    pts += min(4.0, lower_r * 8.0)                       # lower wick
    pts += max(0.0, 2.0 - body_r * (2.0 / 0.30))         # small body
    pts += min(2.0, body_mid * 2.0)                       # body near top
    pts += max(0.0, 2.0 - upper_r * (2.0 / 0.10))        # tiny upper wick

    return min(10.0, pts)


def _score_supply(
    rng: float, body: float, upper: float, lower: float, body_mid: float,
) -> float:
    """
    Score a candle as a SUPPLY pivot (0–10).

    High score = pin bar with long upper wick + small body near the bottom.

    Scoring breakdown:
        4 pts — long upper wick (≥ 50 % of range = max)
        2 pts — small body (< 15 % = max, > 30 % = 0)
        2 pts — body positioned near the bottom of the range
        2 pts — tiny lower wick (< 10 % = max)
    """
    if rng < _MIN_RANGE:
        return 0.0

    lower_r = lower / rng
    upper_r = upper / rng
    body_r  = body  / rng

    pts  = 0.0
    pts += min(4.0, upper_r * 8.0)                        # upper wick
    pts += max(0.0, 2.0 - body_r * (2.0 / 0.30))          # small body
    pts += min(2.0, (1.0 - body_mid) * 2.0)               # body near bottom
    pts += max(0.0, 2.0 - lower_r * (2.0 / 0.10))         # tiny lower wick

    return min(10.0, pts)


def _determine_side(
    upper_r: float,
    lower_r: float,
    score_d: float,
    score_s: float,
) -> tuple[PivotSide, float]:
    """
    Decide DEMAND / SUPPLY / DOJI based on wick dominance + score.

    Rules:
      - The dominant wick must be ≥ MIN_WICK_R of range.
      - The dominant wick must be ≥ DOMINANCE_R × the opposite wick.
      - The corresponding score must be ≥ MIN_SCORE.
    """
    lower_dominates = (
        lower_r >= MIN_WICK_R
        and lower_r >= upper_r * DOMINANCE_R
        and score_d >= MIN_SCORE
    )
    upper_dominates = (
        upper_r >= MIN_WICK_R
        and upper_r >= lower_r * DOMINANCE_R
        and score_s >= MIN_SCORE
    )

    if lower_dominates and (not upper_dominates or score_d >= score_s):
        return PivotSide.DEMAND, score_d
    if upper_dominates:
        return PivotSide.SUPPLY, score_s
    return PivotSide.DOJI, max(score_d, score_s)


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_candle(
    index: pd.Timestamp,
    o: float,
    h: float,
    l: float,
    c: float,
) -> PivotCandle:
    """
    Analyze a single OHLC candle and return its pivot classification.

    Parameters
    ----------
    index : timestamp of the candle
    o, h, l, c : open, high, low, close

    Returns
    -------
    PivotCandle with side=DOJI when no clear dominance is detected.
    """
    rng, body, upper, lower, body_mid = _decompose(o, h, l, c)

    body_h = max(o, c)
    body_l = min(o, c)

    if rng < _MIN_RANGE:
        upper_r = lower_r = body_r = 0.0
        score_d = score_s = 0.0
    else:
        upper_r = upper / rng
        lower_r = lower / rng
        body_r  = body  / rng
        score_d = _score_demand(rng, body, upper, lower, body_mid)
        score_s = _score_supply(rng, body, upper, lower, body_mid)

    side, score = _determine_side(upper_r, lower_r, score_d, score_s)

    return PivotCandle(
        index=index,
        open=o, high=h, low=l, close=c,
        side=side,
        score=round(score, 2),
        body_high=body_h,
        body_low=body_l,
        wick_high=h,
        wick_low=l,
        upper_wick_ratio=round(upper_r, 4),
        lower_wick_ratio=round(lower_r, 4),
        body_ratio=round(body_r, 4),
    )


def find_last_pivot_candle(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    side: Optional[PivotSide] = None,
    min_score: float = MIN_SCORE,
) -> Optional[PivotCandle]:
    """
    Find the LAST qualifying pivot candle in df[start_idx : end_idx].

    Scans from end_idx-1 backwards so the result is the most recent pivot
    before an impulse — matching "la dernière bougie avant l'impulsion".

    Parameters
    ----------
    df        : OHLCV DataFrame with columns open/high/low/close
    start_idx : first bar index to consider (inclusive)
    end_idx   : first bar index to exclude (= impulse bar)
    side      : if given, only consider candles of that side
    min_score : minimum pivot score to accept

    Returns
    -------
    PivotCandle or None if no qualifying candle is found.
    """
    for i in range(end_idx - 1, start_idx - 1, -1):
        row = df.iloc[i]
        pc  = analyze_candle(
            df.index[i],
            float(row["open"]), float(row["high"]),
            float(row["low"]),  float(row["close"]),
        )
        if pc.side == PivotSide.DOJI:
            continue
        if pc.score < min_score:
            continue
        if side is not None and pc.side != side:
            continue
        return pc

    return None


def scan_for_pivots(
    df: pd.DataFrame,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    min_score: float = MIN_SCORE,
) -> list[PivotCandle]:
    """
    Return all pivot candles (DEMAND or SUPPLY) in df[start_idx : end_idx].

    Results are sorted chronologically (same order as df).

    Parameters
    ----------
    df        : OHLCV DataFrame
    start_idx : first bar index (inclusive)
    end_idx   : last bar index (exclusive); defaults to len(df)
    min_score : minimum pivot score to include
    """
    if end_idx is None:
        end_idx = len(df)

    result: list[PivotCandle] = []
    for i in range(start_idx, end_idx):
        row = df.iloc[i]
        pc  = analyze_candle(
            df.index[i],
            float(row["open"]), float(row["high"]),
            float(row["low"]),  float(row["close"]),
        )
        if pc.score >= min_score and pc.side != PivotSide.DOJI:
            result.append(pc)

    return result
