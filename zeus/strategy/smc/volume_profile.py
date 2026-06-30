"""
Volume Profile — Point of Control, Value Area High/Low (factor 6).

Distributes each bar's volume across a price grid proportional to how
much of the bar's high-low range falls inside each bin. This matches the
standard "TPO / market profile" approach used in professional platforms.

Key outputs
-----------
POC (Point of Control) : price bin with the highest accumulated volume.
VAH (Value Area High)  : upper edge of the price zone containing
                         ``value_area_pct`` (default 70 %) of total volume.
VAL (Value Area Low)   : lower edge of that same zone.

The value area is built by expanding outward from the POC bin, greedily
adding whichever neighbouring bin (above or below) carries more volume,
until the cumulative total meets the threshold.

Usage in the strategy
---------------------
At the confluence check for factor 6, the signal logic asks:
  - Is the current price near the POC?           → high-probability reversal level
  - Is the current price inside the value area?  → fair-value zone
  - Is the current price outside the VA?         → potential imbalance / extension
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolumeProfile:
    """
    Volume profile snapshot for a price/volume window.

    All price values represent the *centre* of their respective price bins
    (POC) or the outer *edge* of the outermost bin (VAH / VAL), so the
    relationship val ≤ poc ≤ vah always holds.
    """
    poc:       float   # Price of the highest-volume bin (bin centre)
    vah:       float   # Upper edge of the value area
    val:       float   # Lower edge of the value area
    formed_at: int     # Bar index at which this snapshot was taken

    def is_near_poc(self, price: float, tolerance_pct: float = 0.003) -> bool:
        """True when price is within tolerance_pct of the POC."""
        if self.poc == 0:
            return False
        return abs(price - self.poc) / self.poc <= tolerance_pct

    def is_in_value_area(self, price: float) -> bool:
        """True when price is inside [VAL, VAH]."""
        return self.val <= price <= self.vah

    def is_above_value_area(self, price: float) -> bool:
        """True when price is above the value area (premium zone)."""
        return price > self.vah

    def is_below_value_area(self, price: float) -> bool:
        """True when price is below the value area (discount zone)."""
        return price < self.val


# ──────────────────────────────────────────────────────────────────────────────
# Core computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_volume_profile(
    highs:          np.ndarray,
    lows:           np.ndarray,
    volumes:        np.ndarray,
    num_bins:       int   = 100,
    value_area_pct: float = 0.70,
    formed_at:      int   = -1,
) -> VolumeProfile | None:
    """
    Compute a volume profile over the supplied price/volume arrays.

    Args:
        highs:          Bar high prices (1-D array, same length as others).
        lows:           Bar low prices.
        volumes:        Bar volumes (use np.ones if unavailable).
        num_bins:       Number of price bins in the grid. Higher = finer
                        resolution but slower. 100 is a good default.
        value_area_pct: Fraction of total volume the value area must contain.
                        Standard value: 0.70 (70 %).
        formed_at:      Bar index to stamp on the returned snapshot.

    Returns:
        VolumeProfile, or None when there is no volume (all zeros / empty).
    """
    if len(highs) == 0 or volumes.sum() == 0:
        return None

    price_high = float(highs.max())
    price_low  = float(lows.min())

    # Degenerate case: all bars at exactly the same price
    if price_high == price_low:
        return VolumeProfile(
            poc=price_high,
            vah=price_high,
            val=price_low,
            formed_at=formed_at,
        )

    bin_width  = (price_high - price_low) / num_bins
    bin_vols   = np.zeros(num_bins, dtype=float)

    for j in range(len(highs)):
        h  = float(highs[j])
        lo = float(lows[j])
        v  = float(volumes[j])
        bar_range = h - lo

        if bar_range == 0.0:
            # Point bar — assign all volume to its single bin
            b = min(int((lo - price_low) / bin_width), num_bins - 1)
            bin_vols[b] += v
        else:
            # Distribute volume proportionally across all overlapping bins
            first_b = int((lo - price_low) / bin_width)
            last_b  = min(int((h  - price_low) / bin_width), num_bins - 1)
            for b in range(first_b, last_b + 1):
                overlap_lo = max(lo, price_low + b * bin_width)
                overlap_hi = min(h,  price_low + (b + 1) * bin_width)
                overlap    = max(0.0, overlap_hi - overlap_lo)
                bin_vols[b] += v * overlap / bar_range

    # POC: bin with highest volume; price = bin centre
    poc_bin   = int(np.argmax(bin_vols))
    poc_price = price_low + (poc_bin + 0.5) * bin_width

    # Value area: expand from POC until threshold is reached
    total_vol  = bin_vols.sum()
    target_vol = total_vol * value_area_pct

    va_lo = poc_bin
    va_hi = poc_bin
    va_vol = bin_vols[poc_bin]

    while va_vol < target_vol:
        can_up = va_hi + 1 < num_bins
        can_dn = va_lo - 1 >= 0

        if not can_up and not can_dn:
            break

        vol_up = bin_vols[va_hi + 1] if can_up else -1.0
        vol_dn = bin_vols[va_lo - 1] if can_dn else -1.0

        if vol_up >= vol_dn:
            va_hi  += 1
            va_vol += bin_vols[va_hi]
        else:
            va_lo  -= 1
            va_vol += bin_vols[va_lo]

    vah = price_low + (va_hi + 1) * bin_width   # top edge of highest VA bin
    val = price_low + va_lo * bin_width          # bottom edge of lowest VA bin

    return VolumeProfile(poc=poc_price, vah=vah, val=val, formed_at=formed_at)


# ──────────────────────────────────────────────────────────────────────────────
# Rolling detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_volume_profiles(
    df:             pd.DataFrame,
    lookback:       int   = 200,
    num_bins:       int   = 100,
    value_area_pct: float = 0.70,
    step:           int   = 1,
) -> list[VolumeProfile]:
    """
    Compute rolling volume profiles every ``step`` bars.

    At bar ``i`` the window is ``df[max(0, i-lookback+1) : i+1]``.
    When the DataFrame has no ``volume`` column, unit volume (1.0 per bar)
    is used — this degrades POC to a pure price-range midpoint but keeps
    the module functional with OHLC-only data.

    Args:
        df:             OHLCV (or OHLC) DataFrame.
        lookback:       Rolling window length in bars.
        num_bins:       Price grid resolution.
        value_area_pct: Target fraction of volume for the value area.
        step:           Compute a profile every ``step`` bars. step=1 gives
                        one profile per bar; step=lookback gives non-overlapping
                        windows.

    Returns:
        List of VolumeProfile objects, one per computed bar, oldest first.
    """
    highs   = df["high"].to_numpy(dtype=float)
    lows    = df["low"].to_numpy(dtype=float)
    volumes = (
        df["volume"].to_numpy(dtype=float)
        if "volume" in df.columns
        else np.ones(len(df), dtype=float)
    )

    n        = len(df)
    profiles: list[VolumeProfile] = []

    for i in range(0, n, step):
        start = max(0, i - lookback + 1)
        profile = compute_volume_profile(
            highs[start : i + 1],
            lows[start  : i + 1],
            volumes[start : i + 1],
            num_bins=num_bins,
            value_area_pct=value_area_pct,
            formed_at=i,
        )
        if profile is not None:
            profiles.append(profile)

    return profiles


# ──────────────────────────────────────────────────────────────────────────────
# Query helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_latest_volume_profile(
    profiles: list[VolumeProfile],
    at_bar:   int,
) -> VolumeProfile | None:
    """Return the most recently formed profile at or before ``at_bar``."""
    candidates = [p for p in profiles if p.formed_at <= at_bar]
    return candidates[-1] if candidates else None
