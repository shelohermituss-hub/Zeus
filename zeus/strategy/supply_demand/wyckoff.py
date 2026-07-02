"""
Wyckoff pattern detector for M1 entry confirmation.

Three-phase sequence detected within a bar window:

    Phase 1 — Accumulation
        Tight consolidation base: candle ranges are small, price stays
        compressed. Institutions build positions without moving price.

    Phase 2 — Manipulation  (Spring for demand / Upthrust for supply)
        A single candle wicks BEYOND the accumulation boundary and closes
        BACK INSIDE it. This sweeps stop-losses and shakes out weak hands.
        Spring  (demand): wick below accum_low,  close back above accum_low
        Upthrust (supply): wick above accum_high, close back below accum_high

    Phase 3 — MSS (Market Structure Shift)
        After the manipulation, a candle closes THROUGH the opposite
        accumulation boundary, confirming institutional direction.
        Demand MSS: close > accum_high  (buyers are in control)
        Supply MSS: close < accum_low   (sellers are in control)

Entry fires on the MSS close (or next bar open).
Stop-loss sits just beyond the manipulation wick extreme for a tight R:R.

Usage
-----
    from zeus.strategy.supply_demand.wyckoff import WyckoffDetector
    from zeus.strategy.supply_demand.pivot_candle import PivotSide

    detector = WyckoffDetector()
    pattern  = detector.detect(m1_df, PivotSide.DEMAND, end_idx=len(m1_df))
    if pattern and pattern.score >= 5.0:
        entry = pattern.entry_price
        sl    = pattern.stop_loss_price
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .pivot_candle import PivotSide

# ── Constants ─────────────────────────────────────────────────────────────────

_MIN_RANGE = 1e-8

DEFAULT_LOOKBACK             = 30
DEFAULT_MIN_ACCUM            = 3
DEFAULT_ACCUM_MULT           = 3.0
DEFAULT_MSS_LOOKBACK         = 10
DEFAULT_MIN_SPRING_SWEEP_PCT = 0.20   # spring wick must sweep ≥ 20 % of accum range
DEFAULT_MIN_MSS_STRENGTH_PCT = 0.10   # MSS close must extend ≥ 10 % beyond accum edge


# ── Data type ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WyckoffPattern:
    """
    Detected Wyckoff accumulation → manipulation → MSS pattern.

    Attributes
    ----------
    side           : DEMAND (Spring→long) or SUPPLY (Upthrust→short)
    accum_high     : upper boundary of the accumulation base
    accum_low      : lower boundary of the accumulation base
    accum_bars     : number of candles that formed the base
    manip_bar      : absolute df index of the Spring / Upthrust candle
    manip_extreme  : wick_low (Spring) or wick_high (Upthrust) — the sweep level
    mss_bar        : absolute df index of the MSS close
    mss_close      : close price of the MSS bar
    score          : 0.0–10.0 pattern quality
    formed_at      : timestamp of the MSS bar (signal bar)
    """
    side:          PivotSide
    accum_high:    float
    accum_low:     float
    accum_bars:    int
    manip_bar:     int
    manip_extreme: float
    mss_bar:       int
    mss_close:     float
    score:         float
    formed_at:     pd.Timestamp

    @property
    def entry_price(self) -> float:
        """Suggested entry: MSS close (enter at next bar open in live trading)."""
        return self.mss_close

    @property
    def stop_loss_price(self) -> float:
        """Tight SL: just beyond the manipulation wick extreme."""
        return self.manip_extreme


# ── Detector ──────────────────────────────────────────────────────────────────

class WyckoffDetector:
    """
    Scan a bar window for a Wyckoff accumulation→manipulation→MSS pattern.

    Parameters
    ----------
    lookback             : max bars to scan backward from end_idx
    min_accum_bars       : minimum candles required in the accumulation base
    accum_range_mult     : accumulation range must be ≤ mult × mean candle range
    mss_lookback         : max bars after Spring/Upthrust to expect the MSS
    min_spring_sweep_pct : minimum spring sweep as a fraction of accum range (default 0.20)
    min_mss_strength_pct : minimum MSS close extension as a fraction of accum range (default 0.10)
    """

    def __init__(
        self,
        lookback:             int   = DEFAULT_LOOKBACK,
        min_accum_bars:       int   = DEFAULT_MIN_ACCUM,
        accum_range_mult:     float = DEFAULT_ACCUM_MULT,
        mss_lookback:         int   = DEFAULT_MSS_LOOKBACK,
        min_spring_sweep_pct: float = DEFAULT_MIN_SPRING_SWEEP_PCT,
        min_mss_strength_pct: float = DEFAULT_MIN_MSS_STRENGTH_PCT,
    ) -> None:
        self.lookback             = lookback
        self.min_accum_bars       = min_accum_bars
        self.accum_range_mult     = accum_range_mult
        self.mss_lookback         = mss_lookback
        self.min_spring_sweep_pct = min_spring_sweep_pct
        self.min_mss_strength_pct = min_mss_strength_pct

    # ── Public ────────────────────────────────────────────────────────────────

    def detect(
        self,
        df:      pd.DataFrame,
        side:    PivotSide,
        end_idx: int,
    ) -> Optional[WyckoffPattern]:
        """
        Detect a Wyckoff pattern in df[0:end_idx].

        Scans forward through the most recent `lookback` bars.
        Returns the first complete accumulation→manipulation→MSS sequence found,
        or None if no valid pattern exists.

        Parameters
        ----------
        df      : OHLCV DataFrame (M1 recommended)
        side    : DEMAND (looking for Spring+long) or SUPPLY (Upthrust+short)
        end_idx : exclusive upper bound — scan df[start:end_idx]

        Returns
        -------
        WyckoffPattern or None.
        """
        if side == PivotSide.DOJI:
            return None

        start   = max(0, end_idx - self.lookback)
        n_slice = end_idx - start

        if n_slice < self.min_accum_bars + 2:
            return None

        opens  = df["open"].iloc[start:end_idx].values.astype(float)
        highs  = df["high"].iloc[start:end_idx].values.astype(float)
        lows   = df["low"].iloc[start:end_idx].values.astype(float)
        closes = df["close"].iloc[start:end_idx].values.astype(float)

        if side == PivotSide.DEMAND:
            result = self._scan_demand(highs, lows, closes, n_slice)
        else:
            result = self._scan_supply(highs, lows, closes, n_slice)

        if result is None:
            return None

        accum_h, accum_l, accum_cnt, si, manip_ext, mi = result

        score = self._score(
            side, accum_h, accum_l, accum_cnt,
            highs, lows, closes, opens, si, mi,
        )

        return WyckoffPattern(
            side          = side,
            accum_high    = round(accum_h, 6),
            accum_low     = round(accum_l, 6),
            accum_bars    = accum_cnt,
            manip_bar     = start + si,
            manip_extreme = round(manip_ext, 6),
            mss_bar       = start + mi,
            mss_close     = round(float(closes[mi]), 6),
            score         = round(score, 2),
            formed_at     = df.index[start + mi],
        )

    # ── Private: scanning ─────────────────────────────────────────────────────

    def _scan_demand(
        self,
        highs:  np.ndarray,
        lows:   np.ndarray,
        closes: np.ndarray,
        n:      int,
    ) -> Optional[tuple]:
        """
        Backward scan (most recent accumulation first) for:
            [accum_end tight bars] → Spring (low<accum_l, close>accum_l)
                                   → MSS   (close>accum_h, must be on bar n-1)

        Scanning backward ensures we return the most recent pattern, and
        requiring mi == n-1 means the MSS fires on the current (latest) bar.

        Returns (accum_h, accum_l, accum_cnt, spring_i, spring_low, mss_i).
        """
        for accum_end in range(n - 2, self.min_accum_bars - 1, -1):
            accum_h, accum_l, ok = self._check_accum(highs, lows, accum_end)
            if not ok:
                continue

            accum_rng  = accum_h - accum_l
            min_sweep  = self.min_spring_sweep_pct * accum_rng
            min_mss    = self.min_mss_strength_pct * accum_rng

            search_end = min(n, accum_end + self.mss_lookback + 2)
            for si in range(accum_end, search_end - 1):
                if lows[si] < accum_l and closes[si] > accum_l:
                    # Spring must sweep the accumulation floor by a meaningful amount
                    if (accum_l - lows[si]) < min_sweep:
                        break  # weak spring — try a different accum window
                    # Spring found — look for MSS on the current bar (n-1)
                    for mi in range(si + 1, min(n, si + self.mss_lookback + 1)):
                        if closes[mi] > accum_h:
                            # MSS must close significantly beyond the accum ceiling
                            if mi == n - 1 and (closes[mi] - accum_h) >= min_mss:
                                return (accum_h, accum_l, accum_end, si, lows[si], mi)
                            break  # stale MSS or strength too weak
                    break  # one spring candidate per accumulation window

        return None

    def _scan_supply(
        self,
        highs:  np.ndarray,
        lows:   np.ndarray,
        closes: np.ndarray,
        n:      int,
    ) -> Optional[tuple]:
        """
        Backward scan (most recent accumulation first) for:
            [accum_end tight bars] → Upthrust (high>accum_h, close<accum_h)
                                   → MSS      (close<accum_l, must be on bar n-1)

        Returns (accum_h, accum_l, accum_cnt, upthrust_i, upthrust_high, mss_i).
        """
        for accum_end in range(n - 2, self.min_accum_bars - 1, -1):
            accum_h, accum_l, ok = self._check_accum(highs, lows, accum_end)
            if not ok:
                continue

            accum_rng  = accum_h - accum_l
            min_sweep  = self.min_spring_sweep_pct * accum_rng
            min_mss    = self.min_mss_strength_pct * accum_rng

            search_end = min(n, accum_end + self.mss_lookback + 2)
            for si in range(accum_end, search_end - 1):
                if highs[si] > accum_h and closes[si] < accum_h:
                    # Upthrust must breach the accumulation ceiling by a meaningful amount
                    if (highs[si] - accum_h) < min_sweep:
                        break  # weak upthrust — try a different accum window
                    for mi in range(si + 1, min(n, si + self.mss_lookback + 1)):
                        if closes[mi] < accum_l:
                            if mi == n - 1 and (accum_l - closes[mi]) >= min_mss:
                                return (accum_h, accum_l, accum_end, si, highs[si], mi)
                            break  # stale MSS or strength too weak
                    break

        return None

    def _check_accum(
        self,
        highs:     np.ndarray,
        lows:      np.ndarray,
        accum_end: int,
    ) -> tuple[float, float, bool]:
        """
        Check whether df[0:accum_end] is a tight accumulation base.

        Tightness condition: overall range ≤ accum_range_mult × mean candle range.

        Returns (accum_high, accum_low, is_tight).
        """
        accum_h   = float(np.max(highs[:accum_end]))
        accum_l   = float(np.min(lows[:accum_end]))
        accum_rng = accum_h - accum_l

        if accum_rng < _MIN_RANGE:
            return accum_h, accum_l, False

        avg_candle = float(np.mean(highs[:accum_end] - lows[:accum_end]))
        if avg_candle < _MIN_RANGE:
            return accum_h, accum_l, False

        return accum_h, accum_l, accum_rng <= self.accum_range_mult * avg_candle

    # ── Private: scoring ──────────────────────────────────────────────────────

    def _score(
        self,
        side:      PivotSide,
        accum_h:   float,
        accum_l:   float,
        accum_cnt: int,
        highs:     np.ndarray,
        lows:      np.ndarray,
        closes:    np.ndarray,
        opens:     np.ndarray,
        si:        int,
        mi:        int,
    ) -> float:
        """
        Score the detected Wyckoff pattern 0–10.

        Scoring breakdown
        -----------------
        2 pts — accumulation depth  (more bars = more trapped liquidity)
        2 pts — manipulation sweep  (wick extension beyond accum boundary)
        2 pts — manipulation candle quality (long wick, tight body)
        2 pts — MSS close strength  (how far the close is past accum edge)
        2 pts — MSS speed           (fewer bars between Spring and MSS = stronger)
        """
        accum_rng = accum_h - accum_l
        score     = 0.0

        # 1. Accumulation depth
        score += min(2.0, accum_cnt / self.min_accum_bars)

        # 2. Manipulation sweep extension
        if accum_rng > _MIN_RANGE:
            if side == PivotSide.DEMAND:
                sweep = max(0.0, accum_l - lows[si])
            else:
                sweep = max(0.0, highs[si] - accum_h)
            score += min(2.0, (sweep / accum_rng) * 4.0)

        # 3. Manipulation candle: long wick, small body
        candle_rng = highs[si] - lows[si]
        if candle_rng > _MIN_RANGE:
            body_r = abs(closes[si] - opens[si]) / candle_rng
            if side == PivotSide.DEMAND:
                wick_r = max(0.0, accum_l - lows[si]) / candle_rng
            else:
                wick_r = max(0.0, highs[si] - accum_h) / candle_rng
            score += min(2.0, wick_r * 3.0 + max(0.0, 0.5 - body_r))

        # 4. MSS close strength
        if accum_rng > _MIN_RANGE:
            if side == PivotSide.DEMAND:
                strength = max(0.0, closes[mi] - accum_h)
            else:
                strength = max(0.0, accum_l - closes[mi])
            score += min(2.0, (strength / accum_rng) * 2.0)

        # 5. MSS speed (1 bar after spring = max; each extra bar costs 0.5 pts)
        bars_to_mss = mi - si
        score += max(0.0, 2.0 - (bars_to_mss - 1) * 0.5)

        return min(10.0, score)
