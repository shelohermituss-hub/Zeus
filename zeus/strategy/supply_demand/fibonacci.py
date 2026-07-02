"""
Fibonacci retracement levels for S&D zone confluence.

Key concepts from the strategy:
    OTE (Optimal Trade Entry) = the 0.705–0.786 "golden pocket"
        The 0.764 level is the core of the OTE zone. When an S&D zone
        aligns with this range on a higher-timeframe move, the reversal
        probability increases significantly.

    Discount / Premium boundary = the 0.5 (equilibrium) level
        DEMAND entries: look for zones BELOW 0.5 (discount — price is cheap)
        SUPPLY entries: look for zones ABOVE 0.5 (premium — price is expensive)

Usage
-----
    from zeus.strategy.supply_demand.fibonacci import (
        compute_fib_levels, zone_fib_confluence, FibLevels, FibConfluence,
    )

    # Bullish impulse: low → high
    fibs = compute_fib_levels(swing_low=1900.0, swing_high=2100.0)
    print(fibs.ote)            # 1947.2  (2100 − 200×0.764)
    print(fibs.equilibrium)    # 2000.0  (0.5 level)

    # Check if a demand zone is at OTE and in discount
    conf = zone_fib_confluence(demand_zone, fibs)
    print(conf.has_ote)          # True / False
    print(conf.is_in_discount)   # True if zone midpoint < equilibrium
    print(conf.score_bonus)      # 0.0–2.0 — add to zone total score
"""
from __future__ import annotations

from dataclasses import dataclass

from .pivot_candle import PivotSide
from .zone_detector import SDZone

# ── Constants ─────────────────────────────────────────────────────────────────

# The "golden pocket" — OTE range
_OTE_LOW  = 0.705
_OTE_HIGH = 0.786

# All named ratios (for nearest-level lookup)
_NAMED_LEVELS: dict[str, float] = {
    "0.236": 0.236,
    "0.382": 0.382,
    "0.500": 0.500,
    "0.618": 0.618,
    "0.705": 0.705,
    "0.764": 0.764,
    "0.786": 0.786,
}


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FibLevels:
    """
    Fibonacci retracement prices for a directional swing move.

    For a BULLISH swing (is_bullish=True):
        swing_low  = impulse origin  → level_1000
        swing_high = impulse peak    → level_0000
        Retracements count downward from swing_high.
        level_r = swing_high − range × r

    For a BEARISH swing (is_bullish=False):
        swing_high = impulse origin  → level_1000
        swing_low  = impulse trough  → level_0000
        Retracements count upward from swing_low.
        level_r = swing_low + range × r
    """
    swing_low:   float
    swing_high:  float
    is_bullish:  bool
    level_0000:  float   # start of retracement (high for bull, low for bear)
    level_0236:  float
    level_0382:  float
    level_0500:  float   # equilibrium — discount / premium boundary
    level_0618:  float   # golden ratio
    level_0705:  float   # OTE lower bound
    level_0764:  float   # OTE core
    level_0786:  float   # OTE upper bound
    level_1000:  float   # full retracement (swing origin)

    @property
    def ote(self) -> float:
        """The 0.764 OTE level — core of the golden pocket."""
        return self.level_0764

    @property
    def ote_low(self) -> float:
        """Lower price bound of the golden pocket (0.705 / 0.786, whichever is lower)."""
        return min(self.level_0705, self.level_0786)

    @property
    def ote_high(self) -> float:
        """Upper price bound of the golden pocket."""
        return max(self.level_0705, self.level_0786)

    @property
    def equilibrium(self) -> float:
        """The 0.5 level — midpoint of the range."""
        return self.level_0500

    @property
    def swing_range(self) -> float:
        return self.swing_high - self.swing_low


@dataclass(frozen=True)
class FibConfluence:
    """
    Result of checking a detected S&D zone against Fibonacci levels.

    Attributes
    ----------
    has_ote         : zone body overlaps the OTE golden pocket (0.705–0.786)
    is_in_discount  : demand zone midpoint is below the 0.5 equilibrium
    is_in_premium   : supply zone midpoint is above the 0.5 equilibrium
    nearest_level   : name of the fib level closest to the zone midpoint
    nearest_distance: price distance from zone midpoint to that level
    score_bonus     : 0.0–2.0 points to add to the zone's composite score
    """
    has_ote:          bool
    is_in_discount:   bool
    is_in_premium:    bool
    nearest_level:    str
    nearest_distance: float
    score_bonus:      float


# ── Public functions ──────────────────────────────────────────────────────────

def compute_fib_levels(
    swing_low:  float,
    swing_high: float,
    is_bullish: bool = True,
) -> FibLevels:
    """
    Compute Fibonacci retracement levels for a directional swing move.

    Parameters
    ----------
    swing_low   : lowest price of the move
    swing_high  : highest price of the move
    is_bullish  : True  → impulse went low→high (retracements count down)
                  False → impulse went high→low (retracements count up)

    Returns
    -------
    FibLevels with all key retracement prices.

    Examples
    --------
    Bullish swing (1900 → 2100):
        0.764 level = 2100 − (200 × 0.764) = 1947.2
        0.500 level = 2100 − (200 × 0.500) = 2000.0

    Bearish swing (2100 → 1900), is_bullish=False:
        0.764 level = 1900 + (200 × 0.764) = 2052.8
        0.500 level = 1900 + (200 × 0.500) = 2000.0
    """
    rng = swing_high - swing_low

    if is_bullish:
        def _lvl(r: float) -> float:
            return swing_high - rng * r
    else:
        def _lvl(r: float) -> float:
            return swing_low + rng * r

    return FibLevels(
        swing_low  = swing_low,
        swing_high = swing_high,
        is_bullish = is_bullish,
        level_0000 = _lvl(0.000),
        level_0236 = _lvl(0.236),
        level_0382 = _lvl(0.382),
        level_0500 = _lvl(0.500),
        level_0618 = _lvl(0.618),
        level_0705 = _lvl(0.705),
        level_0764 = _lvl(0.764),
        level_0786 = _lvl(0.786),
        level_1000 = _lvl(1.000),
    )


def zone_fib_confluence(
    zone: SDZone,
    fibs: FibLevels,
) -> FibConfluence:
    """
    Check whether a detected S&D zone aligns with key Fibonacci levels.

    OTE check
    ---------
    The zone body overlaps the golden pocket (price range between
    the 0.705 and 0.786 retracement levels). This is the highest-
    probability entry area per the ICT/SMC framework.

    Discount / Premium
    ------------------
    DEMAND zone: midpoint below equilibrium (0.5) → in discount
    SUPPLY zone: midpoint above equilibrium (0.5) → in premium

    Score bonus (0.0–2.0)
    ---------------------
    +1.5  has OTE confluence
    +0.5  at 0.618 (golden ratio) without full OTE
    +0.5  in correct discount / premium zone
    (capped at 2.0)

    Parameters
    ----------
    zone : S&D zone from ZoneDetector
    fibs : Fibonacci levels of the enclosing HTF move

    Returns
    -------
    FibConfluence with confluence flags and score bonus.
    """
    mid = zone.midpoint
    eq  = fibs.equilibrium

    # ── OTE confluence ───────────────────────────────────────────────────────
    # Zone body overlaps the golden pocket [ote_low, ote_high]
    has_ote = (zone.zone_bottom <= fibs.ote_high) and (zone.zone_top >= fibs.ote_low)

    # ── Discount / premium ───────────────────────────────────────────────────
    if zone.side == PivotSide.DEMAND:
        is_in_discount = mid < eq
        is_in_premium  = False
    else:
        is_in_discount = False
        is_in_premium  = mid > eq

    # ── Nearest named level ──────────────────────────────────────────────────
    named_prices = {name: _fib_price(fibs, ratio) for name, ratio in _NAMED_LEVELS.items()}
    nearest, nearest_dist = min(
        ((name, abs(mid - price)) for name, price in named_prices.items()),
        key=lambda x: x[1],
    )

    # ── Score bonus ──────────────────────────────────────────────────────────
    bonus = 0.0
    if has_ote:
        bonus += 1.5
    elif nearest == "0.618":
        bonus += 0.5
    if is_in_discount or is_in_premium:
        bonus += 0.5
    bonus = min(2.0, bonus)

    return FibConfluence(
        has_ote          = has_ote,
        is_in_discount   = is_in_discount,
        is_in_premium    = is_in_premium,
        nearest_level    = nearest,
        nearest_distance = round(nearest_dist, 6),
        score_bonus      = round(bonus, 2),
    )


def fib_level_price(fibs: FibLevels, ratio: float) -> float:
    """
    Compute the price at an arbitrary Fibonacci ratio.

    Useful for custom levels not pre-computed in FibLevels.

    Parameters
    ----------
    fibs  : FibLevels computed by compute_fib_levels
    ratio : fib ratio (0.0 = start of retracement, 1.0 = full retracement)
    """
    return _fib_price(fibs, ratio)


# ── Internal ──────────────────────────────────────────────────────────────────

def _fib_price(fibs: FibLevels, ratio: float) -> float:
    rng = fibs.swing_range
    if fibs.is_bullish:
        return fibs.swing_high - rng * ratio
    return fibs.swing_low + rng * ratio
