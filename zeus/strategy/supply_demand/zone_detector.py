"""
Supply & Demand zone detector — 5-criteria scoring.

Detection algorithm (per bar):
    1. Find local swing lows (demand) / swing highs (supply) with qualifying
       pivot candles (from pivot_candle.py).
    2. For each pivot, look forward up to bos_max_bars for a BOS
       (close above prior swing high for demand / below for supply).
    3. Build the zone from the pivot candle body + wick extreme for SL.
    4. Score on 5 criteria (0–2 pts each, max 10):
         a. BOS strength    — how decisively the impulse closed beyond the swing
         b. Impulse quality — large bodies, few overlapping wicks (imbalance)
         c. Time in zone    — fewer base candles = more powerful imbalance
         d. Freshness       — starts at 2.0, drops on touch, 0.0 on mitigation
         e. Liquidity sweep — zone formed after a stop raid of prior H/L

Mitigation tracking:
    Call update_zones() on every bar. A zone is mitigated when price closes
    through the wick extreme (full zone negation). Freshness drops on first
    touch even without mitigation.

Usage
-----
    from zeus.strategy.supply_demand.zone_detector import ZoneDetector, SDZone

    detector = ZoneDetector()
    zones    = detector.detect_zones(df_m15)

    for bar_ts, bar in df_m1.iterrows():
        detector.update_zones(zones, bar, bar_ts)
        active = [z for z in zones if not z.is_mitigated and z.score.total >= 6]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .pivot_candle import PivotCandle, PivotSide, analyze_candle

# ── Constants ─────────────────────────────────────────────────────────────────

_EPS = 1e-8

MIN_PIVOT_SCORE        = 4.0
MIN_ZONE_SCORE         = 4.0
SWING_LOOKBACK         = 5     # bars each side to confirm a swing H/L
BOS_LOOKBACK           = 20    # bars back to find the swing level for BOS
BOS_MAX_BARS           = 40    # max bars forward to find bullish BOS after demand pivot
BOS_MAX_BARS_SUPPLY    = 80    # max bars for bearish BOS — bear market corrections are slower
MAX_BASE_CANDLES       = 10    # cap on base width for time scoring
ATR_PERIOD             = 14


# ── Score & zone data types ───────────────────────────────────────────────────

@dataclass(frozen=True)
class ZoneScore:
    """
    5-criteria quality score for an S&D zone.  Each criterion: 0–2 pts.
    Total max = 10.  Zones with total ≥ 6 are considered high quality.
    """
    bos:     float   # BOS strength
    impulse: float   # impulse candle quality
    time:    float   # time (candles) in base
    fresh:   float   # freshness (updated dynamically)
    sweep:   float   # liquidity sweep before zone

    @property
    def total(self) -> float:
        return round(self.bos + self.impulse + self.time + self.fresh + self.sweep, 2)

    def __str__(self) -> str:
        return (
            f"{self.total:.1f}/10 "
            f"[bos={self.bos:.1f} imp={self.impulse:.1f} "
            f"time={self.time:.1f} fresh={self.fresh:.1f} sweep={self.sweep:.1f}]"
        )


@dataclass
class SDZone:
    """
    A detected Supply or Demand zone.

    Zone boundaries
    ---------------
    The primary zone is the pivot candle's BODY:
        zone_top    = body_high
        zone_bottom = body_low

    The wick_extreme is outside the body and defines the tight SL level:
        DEMAND: wick_extreme = wick_low  (SL just below the lower wick)
        SUPPLY: wick_extreme = wick_high (SL just above the upper wick)

    A zone is mitigated when a bar closes through the wick_extreme,
    meaning the institutional order has been fully consumed.
    """
    side:         PivotSide
    zone_top:     float           # body_high of pivot candle
    zone_bottom:  float           # body_low of pivot candle
    wick_extreme: float           # tight SL: wick_low (DEMAND) / wick_high (SUPPLY)
    pivot:        PivotCandle
    pivot_bar:    int             # positional index in the source df
    formed_at:    pd.Timestamp   # timestamp of BOS bar (zone becomes active)
    bos_bar:      int
    bos_level:    float           # swing H/L broken by the impulse
    base_candles: int             # width of accumulation base in candles
    score:        ZoneScore
    is_mitigated: bool                    = field(default=False)
    mitigated_at: Optional[pd.Timestamp] = field(default=None)
    touch_count:  int                     = field(default=0)

    @property
    def midpoint(self) -> float:
        return (self.zone_top + self.zone_bottom) / 2.0

    @property
    def height(self) -> float:
        return self.zone_top - self.zone_bottom

    def price_in_zone(self, low: float, high: float) -> bool:
        """True if the bar's range overlaps the zone body."""
        return low <= self.zone_top and high >= self.zone_bottom

    def price_closed_through(self, close: float) -> bool:
        """True if price closed beyond the wick extreme → zone fully negated."""
        if self.side == PivotSide.DEMAND:
            return close < self.wick_extreme
        return close > self.wick_extreme

    def __repr__(self) -> str:
        state = "MITIGATED" if self.is_mitigated else f"touch={self.touch_count}"
        return (
            f"SDZone({self.side.value.upper()} "
            f"{self.zone_bottom:.5f}–{self.zone_top:.5f} "
            f"SL={self.wick_extreme:.5f} "
            f"score={self.score} "
            f"formed={self.formed_at.strftime('%Y-%m-%d %H:%M')} {state})"
        )


# ── ZoneDetector ──────────────────────────────────────────────────────────────

class ZoneDetector:
    """
    Scans OHLCV data and returns all valid S&D zones sorted by formed_at.

    Parameters
    ----------
    swing_lookback  : bars on each side to confirm a swing H/L (default 5)
    bos_lookback    : bars back to define the swing level to beat (default 20)
    bos_max_bars    : max bars forward from pivot to look for BOS (default 40)
    max_base_candles: base wider than this → time score = 0 (default 10)
    min_pivot_score : minimum pivot candle quality (default 4.0)
    min_zone_score  : minimum total zone score to keep (default 4.0)
    atr_period      : ATR smoothing window (default 14)
    """

    def __init__(
        self,
        swing_lookback:        int   = SWING_LOOKBACK,
        bos_lookback:          int   = BOS_LOOKBACK,
        bos_max_bars:          int   = BOS_MAX_BARS,
        bos_max_bars_supply:   int   = BOS_MAX_BARS_SUPPLY,
        max_base_candles:      int   = MAX_BASE_CANDLES,
        min_pivot_score:       float = MIN_PIVOT_SCORE,
        min_zone_score:        float = MIN_ZONE_SCORE,
        atr_period:            int   = ATR_PERIOD,
    ) -> None:
        self.swing_lookback      = swing_lookback
        self.bos_lookback        = bos_lookback
        self.bos_max_bars        = bos_max_bars
        self.bos_max_bars_supply = bos_max_bars_supply
        self.max_base_candles    = max_base_candles
        self.min_pivot_score     = min_pivot_score
        self.min_zone_score      = min_zone_score
        self.atr_period          = atr_period

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_zones(self, df: pd.DataFrame) -> list[SDZone]:
        """
        Scan the entire df and return all detected S&D zones.

        This is a historical (non-causal) scan: swing detection uses bars
        on both sides of the pivot point. Zones are only "active" after
        their bos_bar, which the strategy must enforce.

        Parameters
        ----------
        df : OHLCV DataFrame with columns open/high/low/close.
             Typically M15 or M30.

        Returns
        -------
        list[SDZone] sorted chronologically by formed_at.
        """
        if df.empty or len(df) < self.swing_lookback * 2 + 2:
            return []

        atr    = _compute_atr(df, self.atr_period)
        zones: list[SDZone] = []
        n      = len(df)
        lo     = self.swing_lookback
        # hi_lim: scan up to n-2 so there is at least 1 bar ahead for BOS search
        # (BOS search itself is bounded by bos_max_bars internally)
        hi_lim = n - 1

        if hi_lim <= lo:
            return zones

        for i in range(lo, hi_lim):
            row = df.iloc[i]
            pc  = analyze_candle(
                df.index[i],
                float(row["open"]), float(row["high"]),
                float(row["low"]),  float(row["close"]),
            )
            if pc.score < self.min_pivot_score:
                continue

            if pc.side == PivotSide.DEMAND and self._is_swing_low(df, i):
                bos_bar, bos_lvl = self._find_bullish_bos(df, i, atr)
                if bos_bar is not None:
                    z = self._build_zone(df, i, pc, bos_bar, bos_lvl, atr, PivotSide.DEMAND)
                    if z and z.score.total >= self.min_zone_score:
                        if not self._is_duplicate(z, zones):
                            zones.append(z)

            elif pc.side == PivotSide.SUPPLY and self._is_swing_high(df, i):
                bos_bar, bos_lvl = self._find_bearish_bos(df, i, atr, self.bos_max_bars_supply)
                if bos_bar is not None:
                    z = self._build_zone(df, i, pc, bos_bar, bos_lvl, atr, PivotSide.SUPPLY)
                    if z and z.score.total >= self.min_zone_score:
                        if not self._is_duplicate(z, zones):
                            zones.append(z)

        return sorted(zones, key=lambda z: z.formed_at)

    def update_zones(
        self,
        zones: list[SDZone],
        bar:   pd.Series,
        ts:    pd.Timestamp,
    ) -> None:
        """
        Update touch_count, freshness, and mitigation state for active zones.

        Call once per bar in the backtest/live loop.

        Freshness rules:
            2.0 → never touched (fresh)
            1.0 → price entered zone but did not close through wick (tested)
            0.0 → price closed through wick extreme (mitigated, zone invalid)
        """
        low   = float(bar["low"])
        high  = float(bar["high"])
        close = float(bar["close"])

        for zone in zones:
            if zone.is_mitigated:
                continue
            if not zone.price_in_zone(low, high):
                continue

            zone.touch_count += 1

            if zone.price_closed_through(close):
                zone.is_mitigated = True
                zone.mitigated_at = ts
                zone.score = ZoneScore(
                    bos=zone.score.bos, impulse=zone.score.impulse,
                    time=zone.score.time, fresh=0.0, sweep=zone.score.sweep,
                )
            elif zone.touch_count == 1:
                # First touch — reduce freshness but keep zone active
                zone.score = ZoneScore(
                    bos=zone.score.bos, impulse=zone.score.impulse,
                    time=zone.score.time, fresh=1.0, sweep=zone.score.sweep,
                )

    # ── Private — swing detection ─────────────────────────────────────────────

    def _is_swing_low(self, df: pd.DataFrame, i: int) -> bool:
        lb    = self.swing_lookback
        lo_i  = float(df.iloc[i]["low"])
        start = max(0, i - lb)
        end   = min(len(df), i + lb + 1)
        return lo_i <= float(df.iloc[start:end]["low"].min())

    def _is_swing_high(self, df: pd.DataFrame, i: int) -> bool:
        lb    = self.swing_lookback
        hi_i  = float(df.iloc[i]["high"])
        start = max(0, i - lb)
        end   = min(len(df), i + lb + 1)
        return hi_i >= float(df.iloc[start:end]["high"].max())

    # ── Private — BOS detection ───────────────────────────────────────────────

    def _find_bullish_bos(
        self,
        df:        pd.DataFrame,
        pivot_bar: int,
        atr:       pd.Series,
    ) -> tuple[Optional[int], Optional[float]]:
        """
        Find the first bar after pivot_bar that closes above the prior swing high.
        Returns (bos_bar, bos_level) or (None, None).
        """
        lb_start  = max(0, pivot_bar - self.bos_lookback)
        bos_level = float(df.iloc[lb_start:pivot_bar]["high"].max())

        end = min(len(df), pivot_bar + self.bos_max_bars + 1)
        for j in range(pivot_bar + 1, end):
            if float(df.iloc[j]["close"]) > bos_level:
                return j, bos_level
        return None, None

    def _find_bearish_bos(
        self,
        df:        pd.DataFrame,
        pivot_bar: int,
        atr:       pd.Series,
        max_bars:  Optional[int] = None,
    ) -> tuple[Optional[int], Optional[float]]:
        """
        Find the first bar after pivot_bar that closes below the prior swing low.
        Returns (bos_bar, bos_level) or (None, None).
        """
        lb_start  = max(0, pivot_bar - self.bos_lookback)
        bos_level = float(df.iloc[lb_start:pivot_bar]["low"].min())

        _max = max_bars if max_bars is not None else self.bos_max_bars
        end = min(len(df), pivot_bar + _max + 1)
        for j in range(pivot_bar + 1, end):
            if float(df.iloc[j]["close"]) < bos_level:
                return j, bos_level
        return None, None

    # ── Private — zone construction ───────────────────────────────────────────

    def _build_zone(
        self,
        df:        pd.DataFrame,
        pivot_bar: int,
        pc:        PivotCandle,
        bos_bar:   int,
        bos_level: float,
        atr:       pd.Series,
        side:      PivotSide,
    ) -> Optional[SDZone]:
        if pc.body_high <= pc.body_low:
            return None

        if side == PivotSide.DEMAND:
            wick_ext = pc.wick_low
        else:
            wick_ext = pc.wick_high

        base_start   = self._find_base_start(df, pivot_bar, atr)
        base_candles = pivot_bar - base_start + 1

        score = ZoneScore(
            bos     = self._score_bos(df, bos_bar, bos_level, side, atr),
            impulse = self._score_impulse(df, pivot_bar, bos_bar),
            time    = self._score_time(base_candles),
            fresh   = 2.0,
            sweep   = self._score_sweep(df, pivot_bar, side),
        )

        return SDZone(
            side         = side,
            zone_top     = pc.body_high,
            zone_bottom  = pc.body_low,
            wick_extreme = wick_ext,
            pivot        = pc,
            pivot_bar    = pivot_bar,
            formed_at    = df.index[bos_bar],
            bos_bar      = bos_bar,
            bos_level    = bos_level,
            base_candles = base_candles,
            score        = score,
        )

    def _find_base_start(
        self,
        df:        pd.DataFrame,
        pivot_bar: int,
        atr:       pd.Series,
    ) -> int:
        """
        Walk backwards from pivot_bar to estimate where the base started.

        The base ends when a candle is significantly larger than the ATR
        (indicating the prior impulse move, not accumulation).
        """
        avg_atr = float(atr.iloc[pivot_bar]) if not pd.isna(atr.iloc[pivot_bar]) else 0.0
        threshold = avg_atr * 1.5 if avg_atr > _EPS else float("inf")

        for i in range(pivot_bar - 1, max(-1, pivot_bar - self.max_base_candles), -1):
            rng = float(df.iloc[i]["high"]) - float(df.iloc[i]["low"])
            if rng > threshold:
                return i + 1   # base starts at bar after the large candle
        return max(0, pivot_bar - self.max_base_candles)

    # ── Private — scoring ─────────────────────────────────────────────────────

    def _score_bos(
        self,
        df:        pd.DataFrame,
        bos_bar:   int,
        bos_level: float,
        side:      PivotSide,
        atr:       pd.Series,
    ) -> float:
        """0–2 pts: how decisively the BOS bar closed beyond the swing level."""
        close = float(df.iloc[bos_bar]["close"])
        atr_v = float(atr.iloc[bos_bar])
        if pd.isna(atr_v) or atr_v < _EPS:
            return 1.0

        dist  = (close - bos_level) if side == PivotSide.DEMAND else (bos_level - close)
        ratio = dist / atr_v
        if ratio >= 0.50: return 2.0
        if ratio >= 0.30: return 1.5
        if ratio >= 0.10: return 1.0
        return 0.5

    def _score_impulse(
        self,
        df:        pd.DataFrame,
        pivot_bar: int,
        bos_bar:   int,
    ) -> float:
        """
        0–2 pts: quality of the impulse candles (pivot_bar+1 → bos_bar).

        Higher average body/range ratio = cleaner, more imbalanced impulse.
        """
        impulse = df.iloc[pivot_bar + 1 : bos_bar + 1]
        if impulse.empty:
            return 1.0
        highs  = impulse["high"].to_numpy(dtype=float)
        lows   = impulse["low"].to_numpy(dtype=float)
        opens  = impulse["open"].to_numpy(dtype=float)
        closes = impulse["close"].to_numpy(dtype=float)
        rng  = highs - lows
        body = np.abs(closes - opens)
        mask = rng >= _EPS
        if not mask.any():
            return 1.0
        avg = float((body[mask] / rng[mask]).mean())
        if avg >= 0.70: return 2.0
        if avg >= 0.55: return 1.5
        if avg >= 0.40: return 1.0
        return 0.5

    def _score_time(self, base_candles: int) -> float:
        """0–2 pts: fewer candles in base = faster institutional move = stronger zone."""
        if base_candles <= 2: return 2.0
        if base_candles <= 4: return 1.5
        if base_candles <= 6: return 1.0
        if base_candles <= 8: return 0.5
        return 0.0

    def _score_sweep(
        self,
        df:        pd.DataFrame,
        pivot_bar: int,
        side:      PivotSide,
        lookback:  int = 20,
    ) -> float:
        """
        0–2 pts: zone formed after a liquidity sweep of prior highs/lows.

        For DEMAND: look for a wick that dipped below recent lows then closed above
                    (stop raid of retail longs → Spring in Wyckoff terms).
        For SUPPLY: look for a wick that exceeded recent highs then closed below
                    (stop raid of retail shorts → Upthrust in Wyckoff terms).
        """
        start = max(0, pivot_bar - lookback)
        window = df.iloc[start:pivot_bar]
        if len(window) < 3:
            return 0.0

        if side == PivotSide.DEMAND:
            ref  = float(window["low"].quantile(0.25))
            lows  = window["low"].to_numpy(dtype=float)
            clos  = window["close"].to_numpy(dtype=float)
            if ((lows < ref) & (clos > ref)).any():
                return 2.0
        else:
            ref   = float(window["high"].quantile(0.75))
            highs = window["high"].to_numpy(dtype=float)
            clos  = window["close"].to_numpy(dtype=float)
            if ((highs > ref) & (clos < ref)).any():
                return 2.0
        return 0.0

    # ── Private — deduplication ───────────────────────────────────────────────

    def _is_duplicate(self, new_zone: SDZone, existing: list[SDZone]) -> bool:
        """
        True if an active zone of the same side with an overlapping price
        level already exists (midpoints within 50% of the smaller zone's height).
        """
        for z in existing:
            if z.side != new_zone.side or z.is_mitigated:
                continue
            dist = abs(z.midpoint - new_zone.midpoint)
            ref  = min(z.height, new_zone.height)
            if ref < _EPS:
                continue
            if dist < ref * 0.5:
                return True
        return False


# ── Module-level utility ──────────────────────────────────────────────────────

def _compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    h, l, c   = df["high"], df["low"], df["close"]
    prev_c    = c.shift(1)
    tr        = pd.concat(
        [h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()
