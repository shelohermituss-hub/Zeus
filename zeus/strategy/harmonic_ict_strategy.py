"""
Harmonic + ICT Hybrid Strategy.

Concept
-------
Combine the precision of harmonic patterns (PRZ = high-probability reversal zone)
with the structural confirmation of ICT Order Blocks.

A bullish signal requires BOTH:
  1. An active harmonic pattern with a PRZ zone (Bat / Gartley / Butterfly / Crab).
  2. A bullish ICT Order Block whose zone overlaps or is within the PRZ.

This double-filter substantially reduces false entries while maintaining
confirmation that institutional order flow is aligned with the pattern.

Entry logic
-----------
  The entry bar must:
    - Reach the PRZ (low ≤ prz_high) AND
    - Touch or enter an overlapping active bullish OB (low ≤ ob.high) AND
    - Close inside BOTH zones (cl ≥ max(prz_low, ob.low)) AND
    - Be a bullish candle (cl > op) [optional]

Stop-loss
---------
  min(prz_low, ob.low) − sl_buffer × ATR

Take-profit
-----------
  entry + risk_reward × risk_distance

Pattern invalidation
--------------------
  Harmonic: invalidated if cl < prz_low before confirmation
  OB:       mitigated if price breaks below ob.low
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from zeus.strategy.smc.harmonic import HarmonicPattern, detect_harmonics
from zeus.strategy.smc.order_block import detect_order_blocks, OrderBlock
from zeus.strategy.smc.pivot import detect_pivots, BULLISH
from zeus.strategy.smc.structure import detect_structure
from zeus.strategy.utils import SignalCooldown, atr_series as _atr_series


# ── Signal dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HarmonicICTSignal:
    """
    Trading signal produced by HarmonicICTStrategy.

    Compatible with sd_simulation.simulate_trade() and compute_metrics().
    """
    direction:    str
    entry_price:  float
    stop_loss:    float
    take_profit:  float
    risk_reward:  float
    formed_at:    pd.Timestamp
    bar_index:    int
    pattern_type: str
    prz_low:      float
    prz_high:     float
    ob_bar:       int
    ob_low:       float
    ob_high:      float
    zone_score:   float = 0.0


# ── Strategy ──────────────────────────────────────────────────────────────────

class HarmonicICTStrategy:
    """
    Harmonic pattern + ICT Order Block hybrid strategy (M15).

    The dual-filter requires both a harmonic PRZ and an overlapping bullish OB.
    This produces fewer but higher-quality signals compared to either strategy alone.

    Parameters
    ----------
    pivot_size_harmonic : pivot confirmation for harmonic detection (default 7)
    pivot_size_ob       : pivot confirmation for OB structure (default 10)
    precision           : AB ratio tolerance for Gartley/Butterfly (default 0.03)
    ob_overlap_pct      : minimum overlap between OB and PRZ as fraction of PRZ
                          width (default 0.0 = any overlap required)
    use_h4_trend        : require H4 EMA50 bullish slope
    h4_ema_span         : H4 EMA span (default 50)
    h4_slope_lb         : bars for H4 slope comparison (default 3)
    risk_reward         : R:R target (default 1.5)
    sl_buffer_atr       : SL buffer in ATR (default 0.20)
    max_prz_age_bars    : cancel harmonic pattern after N bars (default 200)
    max_ob_age          : cancel OB after N bars without retest (default 200)
    signal_cooldown     : min M15 bars between signals (default 12)
    require_bullish_bar : entry bar must close bullish (default True)
    atr_period          : ATR smoothing period (default 14)
    """

    def __init__(
        self,
        pivot_size_harmonic: int   = 7,
        pivot_size_ob:       int   = 10,
        precision:           float = 0.03,
        ob_overlap_pct:      float = 0.0,
        use_h4_trend:        bool  = True,
        h4_ema_span:         int   = 50,
        h4_slope_lb:         int   = 3,
        risk_reward:         float = 1.5,
        sl_buffer_atr:       float = 0.20,
        max_prz_age_bars:    int   = 200,
        max_ob_age:          int   = 200,
        signal_cooldown:     int   = 12,
        require_bullish_bar: bool  = True,
        atr_period:          int   = 14,
    ) -> None:
        self.pivot_size_harmonic = pivot_size_harmonic
        self.pivot_size_ob       = pivot_size_ob
        self.precision           = precision
        self.ob_overlap_pct      = ob_overlap_pct
        self.use_h4_trend        = use_h4_trend
        self.h4_ema_span         = h4_ema_span
        self.h4_slope_lb         = h4_slope_lb
        self.risk_reward         = risk_reward
        self.sl_buffer_atr       = sl_buffer_atr
        self.max_prz_age_bars    = max_prz_age_bars
        self.max_ob_age          = max_ob_age
        self.signal_cooldown     = signal_cooldown
        self.require_bullish_bar = require_bullish_bar
        self.atr_period          = atr_period

    def run(self, m15_df: pd.DataFrame) -> list[HarmonicICTSignal]:
        """Run hybrid strategy over a full M15 OHLCV DataFrame."""
        if len(m15_df) < 2:
            return []

        _highs  = m15_df["high"].to_numpy(dtype=float)
        _lows   = m15_df["low"].to_numpy(dtype=float)
        _closes = m15_df["close"].to_numpy(dtype=float)
        _opens  = m15_df["open"].to_numpy(dtype=float)
        n       = len(m15_df)

        _atr = _atr_series(m15_df, self.atr_period).to_numpy(dtype=float)

        # ── H4 EMA50 trend ────────────────────────────────────────────────
        _h4_trend: np.ndarray | None = None
        if self.use_h4_trend:
            h4_df = m15_df.resample("4h", closed="left", label="left").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last"}
            ).dropna()
            h4_ema     = h4_df["close"].ewm(span=self.h4_ema_span, adjust=False).mean()
            _h4_v      = h4_ema.to_numpy(dtype=float)
            lb         = self.h4_slope_lb
            _h4_p      = np.empty_like(_h4_v)
            _h4_p[:lb] = _h4_v[0]
            _h4_p[lb:] = _h4_v[:-lb]
            _h4_bull   = _h4_v > _h4_p
            _h4_bull[:lb] = False
            m15_to_h4  = np.searchsorted(
                h4_df.index.values, m15_df.index.values, side="right"
            ) - 1
            m15_to_h4  = np.clip(m15_to_h4, 0, len(h4_df) - 1)
            _h4_trend  = _h4_bull[m15_to_h4]

        # ── Harmonic patterns ──────────────────────────────────────────────
        harmonics = detect_harmonics(
            m15_df,
            pivot_size = self.pivot_size_harmonic,
            precision  = self.precision,
            long_only  = True,
        )

        # ── ICT Order Blocks (from BOS swing structure) ────────────────────
        swing_pivots = detect_pivots(
            m15_df["high"], m15_df["low"], size=self.pivot_size_ob
        )
        swing_events = detect_structure(m15_df["close"], swing_pivots, is_internal=False)
        bull_events  = [e for e in swing_events if e.direction == BULLISH]
        bull_obs     = detect_order_blocks(
            m15_df["high"], m15_df["low"], m15_df["close"],
            bull_events, atr_period=200,
        )

        if not harmonics or not bull_obs:
            return []

        # ── Tracking sets ──────────────────────────────────────────────────
        invalidated_pats: set[tuple] = set()
        triggered_pats:  set[tuple] = set()
        triggered_obs:   set[int]   = set()

        def _pat_key(p: HarmonicPattern) -> tuple:
            return (p.pattern_type, p.direction, p.x_bar, p.c_bar)

        # ── Main loop ─────────────────────────────────────────────────────
        signals:  list[HarmonicICTSignal] = []
        cooldown = SignalCooldown(self.signal_cooldown)

        for i in range(4, n):
            cl_i  = float(_closes[i])
            op_i  = float(_opens[i])
            lo_i  = float(_lows[i])
            hi_i  = float(_highs[i])
            atr_i = max(float(_atr[i]), 1e-6)

            h4_bull = bool(_h4_trend[i]) if _h4_trend is not None else True

            if self.use_h4_trend and not h4_bull:
                continue

            if not cooldown.can_signal(i):
                continue

            # ── For each active harmonic pattern ───────────────────────────
            for pat in harmonics:
                pk = _pat_key(pat)
                if pk in triggered_pats or pk in invalidated_pats:
                    continue
                if i < pat.confirmed_at:
                    continue
                if i - pat.confirmed_at > self.max_prz_age_bars:
                    continue

                # Invalidate harmonic if close breaks below PRZ
                if cl_i < pat.prz_low:
                    invalidated_pats.add(pk)
                    continue

                # Bar must touch the PRZ
                if lo_i > pat.prz_high:
                    continue
                if cl_i < pat.prz_low:
                    continue

                # ── Find an overlapping active OB ──────────────────────────
                matching_ob: OrderBlock | None = None
                for ob in bull_obs:
                    if ob.bar_index in triggered_obs:
                        continue
                    if ob.detected_at > i:
                        continue
                    ob_age = i - ob.detected_at
                    if ob_age > self.max_ob_age:
                        continue
                    if ob.mitigated_at != -1 and ob.mitigated_at <= i:
                        continue

                    # Check overlap between OB zone and PRZ zone
                    overlap_lo = max(ob.low, pat.prz_low)
                    overlap_hi = min(ob.high, pat.prz_high)
                    if overlap_lo > overlap_hi:
                        # No overlap — check if OB is fully inside PRZ (or vice versa)
                        # Allow OB to be above PRZ bottom but below PRZ top
                        if ob.high < pat.prz_low or ob.low > pat.prz_high:
                            continue
                    if self.ob_overlap_pct > 0:
                        prz_width = pat.prz_high - pat.prz_low
                        if prz_width > 0:
                            overlap = max(0.0, overlap_hi - overlap_lo) / prz_width
                            if overlap < self.ob_overlap_pct:
                                continue

                    # Bar must touch or enter the OB zone
                    if lo_i > ob.high:
                        continue

                    matching_ob = ob
                    break

                if matching_ob is None:
                    continue

                # Bullish confirmation bar
                if self.require_bullish_bar and cl_i <= op_i:
                    continue

                # ── Build signal ───────────────────────────────────────────
                support_floor = min(pat.prz_low, matching_ob.low)
                sl_buf        = self.sl_buffer_atr * atr_i
                sl            = support_floor - sl_buf
                risk_dist     = cl_i - sl
                if risk_dist <= 0:
                    continue
                tp = cl_i + self.risk_reward * risk_dist

                triggered_pats.add(pk)
                triggered_obs.add(matching_ob.bar_index)
                cooldown.mark(i)

                signals.append(HarmonicICTSignal(
                    direction    = "long",
                    entry_price  = round(cl_i, 2),
                    stop_loss    = round(sl, 2),
                    take_profit  = round(tp, 2),
                    risk_reward  = self.risk_reward,
                    formed_at    = m15_df.index[i],
                    bar_index    = i,
                    pattern_type = pat.pattern_type,
                    prz_low      = round(pat.prz_low, 2),
                    prz_high     = round(pat.prz_high, 2),
                    ob_bar       = matching_ob.bar_index,
                    ob_low       = round(matching_ob.low, 2),
                    ob_high      = round(matching_ob.high, 2),
                ))
                break  # one signal per bar

        return signals
