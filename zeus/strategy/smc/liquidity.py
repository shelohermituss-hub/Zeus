"""
Liquidity level detection and sweep identification for SMC.

Smart money accumulates orders at obvious highs/lows (liquidity pools).
Before entering, the bot waits for a liquidity sweep (a wick beyond the
level that closes back on the other side) as confirmation that the stop-hunt
is complete.

Terminology
-----------
BSL  — Buy-Side Liquidity  : pool above a swing HIGH (stops of shorts; fuel for longs)
SSL  — Sell-Side Liquidity : pool below a swing LOW  (stops of longs; fuel for shorts)

Sweep detection
---------------
BSL sweep (price grabs above SSL): high > level AND close < level  (wick above, close below)
SSL sweep (price grabs below SSL): low  < level AND close > level  (wick below, close above)

A swept level is removed from the active pool so it cannot trigger twice.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from zeus.strategy.smc.pivot import BEARISH, BULLISH, PivotPoint


class LiqType(IntEnum):
    BSL = 1   # Buy-Side Liquidity  (above swing high)
    SSL = -1  # Sell-Side Liquidity (below swing low)


@dataclass(frozen=True)
class LiquidityLevel:
    """A raw liquidity pool derived from a confirmed pivot."""
    bar_index:  int      # bar index of the originating pivot
    level:      float    # price of the pool
    liq_type:   LiqType  # BSL or SSL
    formed_at:  int      # bar index when the pivot was confirmed (actionable from here)


@dataclass(frozen=True)
class LiquiditySweep:
    """
    A confirmed liquidity sweep event.

    direction = BULLISH when SSL was swept (price grabbed lows → now bullish)
    direction = BEARISH when BSL was swept (price grabbed highs → now bearish)
    """
    bar_index:  int      # bar where the sweep candle closed
    level:      float    # price of the swept pool
    liq_type:   LiqType  # which pool was swept
    direction:  int      # BULLISH (ssl sweep) or BEARISH (bsl sweep)


# ──────────────────────────────────────────────────────────────────────────────
# Detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_liquidity_levels(pivots: list[PivotPoint]) -> list[LiquidityLevel]:
    """
    Build a liquidity pool for every confirmed pivot point.

    Each swing HIGH creates a BSL pool; each swing LOW creates an SSL pool.
    Returns levels in chronological order (oldest first).
    """
    levels: list[LiquidityLevel] = []
    for p in pivots:
        liq_type = LiqType.BSL if p.is_high else LiqType.SSL
        levels.append(LiquidityLevel(
            bar_index=p.bar_index,
            level=p.level,
            liq_type=liq_type,
            formed_at=p.confirmed_at,
        ))
    return levels


def detect_sweeps(
    highs:  list[float],
    lows:   list[float],
    closes: list[float],
    levels: list[LiquidityLevel],
) -> list[LiquiditySweep]:
    """
    Scan price data bar-by-bar and record liquidity sweeps.

    BSL sweep: high > level AND close < level  → BEARISH follow-through expected
    SSL sweep: low  < level AND close > level  → BULLISH follow-through expected

    Each level can only be swept once (first sweep wins, level then consumed).

    Args:
        highs, lows, closes: equal-length price arrays (index 0 = oldest bar)
        levels: list produced by detect_liquidity_levels()

    Returns:
        List of LiquiditySweep events in chronological order.
    """
    n = len(closes)
    swept: set[int] = set()   # bar_index of levels already swept
    sweeps: list[LiquiditySweep] = []

    for bar in range(n):
        h = highs[bar]
        lo = lows[bar]
        c  = closes[bar]

        for lv in levels:
            if lv.bar_index in swept:
                continue
            if lv.formed_at > bar:
                # Level not yet actionable
                continue

            if lv.liq_type == LiqType.BSL and h > lv.level and c < lv.level:
                swept.add(lv.bar_index)
                sweeps.append(LiquiditySweep(
                    bar_index=bar,
                    level=lv.level,
                    liq_type=LiqType.BSL,
                    direction=BEARISH,
                ))
            elif lv.liq_type == LiqType.SSL and lo < lv.level and c > lv.level:
                swept.add(lv.bar_index)
                sweeps.append(LiquiditySweep(
                    bar_index=bar,
                    level=lv.level,
                    liq_type=LiqType.SSL,
                    direction=BULLISH,
                ))

    return sweeps


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_active_levels(
    levels: list[LiquidityLevel],
    at_bar: int,
    sweeps: list[LiquiditySweep] | None = None,
) -> list[LiquidityLevel]:
    """
    Return levels that are actionable at at_bar and have not been swept.

    Args:
        levels:  all known levels
        at_bar:  current bar index
        sweeps:  optional — if provided, swept levels are excluded
    """
    swept_prices: set[float] = set()
    if sweeps:
        swept_prices = {s.level for s in sweeps if s.bar_index <= at_bar}

    return [
        lv for lv in levels
        if lv.formed_at <= at_bar and lv.level not in swept_prices
    ]


def last_sweep(
    sweeps: list[LiquiditySweep],
    at_bar: int,
    direction: int | None = None,
) -> LiquiditySweep | None:
    """
    Return the most recent sweep at or before at_bar, optionally filtered
    by direction (BULLISH = SSL swept, BEARISH = BSL swept).
    """
    candidates = [
        s for s in sweeps
        if s.bar_index <= at_bar
        and (direction is None or s.direction == direction)
    ]
    return candidates[-1] if candidates else None
