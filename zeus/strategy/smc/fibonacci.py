"""
Fibonacci retracement zones for SMC trading setups.

OTE (Optimal Trade Entry) zone = 61.8 % – 78.6 % retracement of an impulsive leg.
The 50 % level is tracked as a standalone high-probability confluence level.

Direction convention
--------------------
BULLISH FibZone (direction = +1):
    Impulsive leg was UP   (swing_low  → swing_high).
    Price is expected to RETRACE downward into the OTE zone before continuing.
    price_at(f) = swing_high − range × f
    OTE zone   = [ price_at(0.786), price_at(0.618) ]   (lower → upper)

BEARISH FibZone (direction = −1):
    Impulsive leg was DOWN (swing_high → swing_low).
    Price is expected to BOUNCE upward into the OTE zone before continuing.
    price_at(f) = swing_low + range × f
    OTE zone   = [ price_at(0.618), price_at(0.786) ]   (lower → upper)
"""
from __future__ import annotations

from dataclasses import dataclass

from zeus.strategy.smc.pivot import PivotPoint, BULLISH, BEARISH

# Key Fibonacci fractions
FIB_236 = 0.236
FIB_382 = 0.382
FIB_500 = 0.500   # golden pocket midpoint
FIB_618 = 0.618   # golden ratio
FIB_705 = 0.705   # common additional confluence
FIB_786 = 0.786   # OTE upper bound

OTE_NEAR  = FIB_618   # 61.8 % — start of OTE zone
OTE_DEEP  = FIB_786   # 78.6 % — end of OTE zone


@dataclass(frozen=True)
class FibZone:
    """
    Fibonacci retracement zone derived from a completed swing leg.

    Used to identify the Optimal Trade Entry (OTE) zone and the 50 % level.
    """
    swing_high:   float
    swing_low:    float
    direction:    int    # BULLISH (+1) = long setup ; BEARISH (-1) = short setup
    leg_high_bar: int    # bar index of the swing HIGH that defines the leg
    leg_low_bar:  int    # bar index of the swing LOW that defines the leg
    formed_at:    int    # bar index when this zone became actionable (pivot confirmed)

    @property
    def _range(self) -> float:
        return self.swing_high - self.swing_low

    def price_at(self, fib_fraction: float) -> float:
        """
        Price at the given retracement fraction.

            fraction = 0.0 → anchor (high for BULLISH, low for BEARISH)
            fraction = 1.0 → target (low for BULLISH, high for BEARISH)
        """
        if self.direction == BULLISH:
            return self.swing_high - self._range * fib_fraction
        return self.swing_low + self._range * fib_fraction

    @property
    def level_50(self) -> float:
        """50 % midpoint — key standalone confluence level (strategy factor 10)."""
        return self.price_at(FIB_500)

    @property
    def ote_bottom(self) -> float:
        """Lower price boundary of the OTE zone."""
        return min(self.price_at(OTE_NEAR), self.price_at(OTE_DEEP))

    @property
    def ote_top(self) -> float:
        """Upper price boundary of the OTE zone."""
        return max(self.price_at(OTE_NEAR), self.price_at(OTE_DEEP))

    def is_in_ote(self, price: float) -> bool:
        """True when price is inside the 61.8 %–78.6 % OTE zone."""
        return self.ote_bottom <= price <= self.ote_top

    def is_near_50(self, price: float, tolerance_pct: float = 0.003) -> bool:
        """
        True when price is within tolerance_pct of the 50 % level.

        Default tolerance = 0.3 % (e.g. ±15 USDT on a 50 000 BTC price).
        """
        target = self.level_50
        if target == 0:
            return False
        return abs(price - target) / target <= tolerance_pct


def detect_fib_zones(pivots: list[PivotPoint]) -> list[FibZone]:
    """
    Derive Fibonacci retracement zones from consecutive pivot pairs.

    Each adjacent pair of pivots (alternating high/low) represents a completed
    swing leg. The zone direction indicates the expected follow-up trade:

        swing_low  → swing_high  (price went UP)  → BULLISH zone (look for LONG)
        swing_high → swing_low   (price went DOWN) → BEARISH zone (look for SHORT)

    Returns zones in chronological order (oldest first).
    """
    zones: list[FibZone] = []

    for i in range(len(pivots) - 1):
        a = pivots[i]
        b = pivots[i + 1]

        if not a.is_high and b.is_high:
            # LOW → HIGH  : bullish impulsive leg, look for LONG on retrace
            zones.append(FibZone(
                swing_high=b.level,
                swing_low=a.level,
                direction=BULLISH,
                leg_high_bar=b.bar_index,
                leg_low_bar=a.bar_index,
                formed_at=b.confirmed_at,
            ))
        elif a.is_high and not b.is_high:
            # HIGH → LOW  : bearish impulsive leg, look for SHORT on bounce
            zones.append(FibZone(
                swing_high=a.level,
                swing_low=b.level,
                direction=BEARISH,
                leg_high_bar=a.bar_index,
                leg_low_bar=b.bar_index,
                formed_at=b.confirmed_at,
            ))

    return zones


def get_latest_fib_zone(
    zones: list[FibZone],
    at_bar: int,
    direction: int,
) -> FibZone | None:
    """
    Return the most recently formed Fibonacci zone of the given direction
    that is already actionable at at_bar.
    """
    candidates = [
        z for z in zones
        if z.direction == direction and z.formed_at <= at_bar
    ]
    return candidates[-1] if candidates else None
