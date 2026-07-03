"""
Convergence Strategy — OB + FVG stack + LTF Sweep + regime filters.

Multi-confluence setup: every condition must hold simultaneously.

  1. D1 EMA(200) macro regime  — D1 close > D1 EMA200 → long bias only
  2. H4 EMA(50) rising slope   — H4 EMA above value N bars ago
  3. Killzone timing            — London 07-11 UTC or NY 12-15 UTC
  4. OB + FVG stack            — bullish ICT OB (BOS-only) with an unmitigated
                                  bullish FVG whose zone overlaps the OB
  5. LTF sweep (inducement)    — within *ltf_sweep_lookback* bars: low < prev low
                                  and close > prev low (sell-side stops swept)
  6. Bullish confirmation bar  — entry bar close > entry bar open

Entry  : close[i] (simulate_trade fills at next M15 open)
SL     : ob.low − sl_buffer_atr × ATR14
TP     : recalculated from fill price at risk_reward × |fill − SL|
TP1    : partial exit at tp1_r (default 0.6R) — caller passes to simulate_all
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from zeus.strategy.smc.fvg import FairValueGap, detect_fvg, get_active_fvgs
from zeus.strategy.smc.order_block import OrderBlock, detect_order_blocks, get_active_order_blocks
from zeus.strategy.smc.pivot import BULLISH, detect_pivots
from zeus.strategy.smc.session import is_in_killzone
from zeus.strategy.smc.structure import StructureType, detect_structure
from zeus.strategy.smc.ltf_sweep import ltf_liquidity_sweep
from zeus.strategy.utils import SignalCooldown, atr_series as _atr_series


# ── Signal ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ConvergenceSignal:
    """
    Trading signal produced by ConvergenceStrategy.

    Satisfies the sd_simulation.Tradeable structural protocol:
      direction, bar_index, stop_loss, risk_reward, formed_at, zone_score.
    """
    direction:      str           # always "long" for current implementation
    entry_price:    float         # close[i] — next-bar open fill in simulation
    stop_loss:      float         # ob.low − sl_buffer_atr × ATR14
    take_profit:    float         # entry + risk_reward × risk_distance
    risk_reward:    float
    formed_at:      pd.Timestamp
    bar_index:      int           # M15 bar index of the signal
    ob_bar:         int           # OB candle bar index
    ob_high:        float
    ob_low:         float
    fvg_bar:        int           # FVG detection bar index
    fvg_top:        float
    fvg_bottom:     float
    ltf_sweep_bar:  int           # bar where sweep was detected
    zone_score:     float = 5.0   # convergence score 0–10 (1 point per filter met)

    @property
    def risk(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def reward(self) -> float:
        return abs(self.take_profit - self.entry_price)


# ── Strategy ───────────────────────────────────────────────────────────────────

class ConvergenceStrategy:
    """
    M15 convergence strategy: OB + FVG + LTF sweep + regime gates.

    Parameters
    ----------
    pivot_size          : swing pivot confirmation bars (default 10)
    long_only           : only long signals (default True)
    use_d1_ema          : D1 EMA(d1_ema_span) macro filter (default True)
    d1_ema_span         : D1 EMA period (default 200)
    use_h4_trend        : H4 EMA(h4_ema_span) slope filter (default True)
    h4_ema_span         : H4 EMA period (default 50)
    h4_slope_lb         : bars used to measure H4 EMA slope (default 3)
    use_killzone        : restrict entries to London / NY killzone (default True)
    use_ltf_sweep       : require LTF inducement sweep (default True)
    ltf_sweep_lookback  : bars to scan for LTF sweep (default 5)
    require_fvg         : require an unmitigated FVG overlapping the OB (default True)
    require_bullish_bar : entry bar must close bullish (default True)
    min_ob_age          : OB must be ≥ N bars old before qualifying (default 2)
    max_ob_age          : OB expires after N bars without a retest (default 200)
    sl_buffer_atr       : SL buffer below OB low in ATR units (default 0.25)
    atr_period          : ATR period for SL buffer and zone score (default 14)
    risk_reward         : full TP target in R units (default 2.0)
    signal_cooldown     : minimum M15 bars between signals (default 12 = 3 h)
    """

    def __init__(
        self,
        pivot_size:          int   = 10,
        long_only:           bool  = True,
        use_d1_ema:          bool  = True,
        d1_ema_span:         int   = 200,
        use_h4_trend:        bool  = True,
        h4_ema_span:         int   = 50,
        h4_slope_lb:         int   = 3,
        use_killzone:        bool  = True,
        use_ltf_sweep:       bool  = True,
        ltf_sweep_lookback:  int   = 5,
        require_fvg:         bool  = True,
        require_bullish_bar: bool  = True,
        min_ob_age:          int   = 2,
        max_ob_age:          int   = 200,
        sl_buffer_atr:       float = 0.25,
        atr_period:          int   = 14,
        risk_reward:         float = 2.0,
        signal_cooldown:     int   = 12,
    ) -> None:
        self.pivot_size          = pivot_size
        self.long_only           = long_only
        self.use_d1_ema          = use_d1_ema
        self.d1_ema_span         = d1_ema_span
        self.use_h4_trend        = use_h4_trend
        self.h4_ema_span         = h4_ema_span
        self.h4_slope_lb         = h4_slope_lb
        self.use_killzone        = use_killzone
        self.use_ltf_sweep       = use_ltf_sweep
        self.ltf_sweep_lookback  = ltf_sweep_lookback
        self.require_fvg         = require_fvg
        self.require_bullish_bar = require_bullish_bar
        self.min_ob_age          = min_ob_age
        self.max_ob_age          = max_ob_age
        self.sl_buffer_atr       = sl_buffer_atr
        self.atr_period          = atr_period
        self.risk_reward         = risk_reward
        self.signal_cooldown     = signal_cooldown

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def run(self, m15_df: pd.DataFrame) -> list[ConvergenceSignal]:
        """
        Scan a full M15 OHLCV DataFrame and return all convergence signals.

        Args:
            m15_df: DatetimeIndex OHLCV DataFrame (UTC or tz-naive UTC).
                    Must have columns: open, high, low, close.

        Returns:
            Chronologically ordered list of ConvergenceSignal.
        """
        if len(m15_df) < max(self.pivot_size * 2 + 10, 30):
            return []

        highs  = m15_df["high"].to_numpy(dtype=float)
        lows   = m15_df["low"].to_numpy(dtype=float)
        closes = m15_df["close"].to_numpy(dtype=float)
        opens  = m15_df["open"].to_numpy(dtype=float)
        n      = len(m15_df)

        atr = _atr_series(m15_df, self.atr_period).to_numpy(dtype=float)

        # ── D1 EMA(200) regime ─────────────────────────────────────────────
        _d1_bull: np.ndarray | None = None
        if self.use_d1_ema:
            _d1_bull = self._build_d1_regime(m15_df)

        # ── H4 EMA(50) slope ──────────────────────────────────────────────
        _h4_trend: np.ndarray | None = None
        if self.use_h4_trend:
            _h4_trend = self._build_h4_trend(m15_df)

        # ── ICT Order Blocks (BOS-only, swing structure) ───────────────────
        swing_pivots = detect_pivots(m15_df["high"], m15_df["low"], size=self.pivot_size)
        swing_events = detect_structure(m15_df["close"], swing_pivots, is_internal=False)

        # BOS-only filter (continuation setups are cleaner than CHoCH reversals)
        bos_events = [e for e in swing_events if e.structure_type == StructureType.BOS]
        if self.long_only:
            bos_events = [e for e in bos_events if e.direction == BULLISH]

        obs = detect_order_blocks(
            m15_df["high"], m15_df["low"], m15_df["close"],
            bos_events,
            atr_period=200,
        )
        bull_obs = [ob for ob in obs if ob.direction == BULLISH]

        if not bull_obs:
            return []

        # ── FVGs ──────────────────────────────────────────────────────────
        all_fvgs = detect_fvg(
            m15_df["high"], m15_df["low"], m15_df["close"], m15_df["open"],
            auto_threshold=True,
        )
        bull_fvgs = [f for f in all_fvgs if f.direction == BULLISH]

        # ── Main scan loop ─────────────────────────────────────────────────
        signals:       list[ConvergenceSignal] = []
        triggered_obs: set[int]               = set()
        cooldown = SignalCooldown(self.signal_cooldown)

        for i in range(max(self.pivot_size + 5, 10), n):
            ts_i  = m15_df.index[i]
            cl_i  = float(closes[i])
            op_i  = float(opens[i])
            lo_i  = float(lows[i])
            hi_i  = float(highs[i])
            atr_i = max(float(atr[i]), 1e-6)

            # ── Cooldown ──────────────────────────────────────────────────
            if not cooldown.can_signal(i):
                continue

            # ── D1 EMA regime ─────────────────────────────────────────────
            if self.use_d1_ema and _d1_bull is not None:
                if not bool(_d1_bull[i]):
                    continue

            # ── H4 slope ──────────────────────────────────────────────────
            if self.use_h4_trend and _h4_trend is not None:
                if not bool(_h4_trend[i]):
                    continue

            # ── Killzone ──────────────────────────────────────────────────
            if self.use_killzone and not is_in_killzone(ts_i):
                continue

            # ── Confirmation bar ──────────────────────────────────────────
            if self.require_bullish_bar and cl_i <= op_i:
                continue

            # ── LTF sweep ─────────────────────────────────────────────────
            sweep_ok = True
            sweep_bar = i
            if self.use_ltf_sweep:
                swept, sweep_reason = ltf_liquidity_sweep(
                    m15_df, i, BULLISH, lookback=self.ltf_sweep_lookback
                )
                if not swept:
                    continue
                # Find the specific sweep bar
                sweep_bar = self._find_sweep_bar(lows, closes, i, self.ltf_sweep_lookback)

            # ── OB + FVG stack scan ───────────────────────────────────────
            active_bull_obs = get_active_order_blocks(bull_obs, at_bar=i)

            for ob in active_bull_obs:
                if ob.bar_index in triggered_obs:
                    continue

                ob_age = i - ob.detected_at
                if ob_age < self.min_ob_age:
                    continue
                if ob_age > self.max_ob_age:
                    continue

                # Price must be touching or inside the OB zone
                if lo_i > ob.high:
                    continue   # price above OB — not retracing into it
                if cl_i < ob.low:
                    continue   # price closed below OB — mitigated

                # FVG overlapping OB (optional)
                overlapping_fvg = None
                if self.require_fvg:
                    active_fvgs     = get_active_fvgs(bull_fvgs, at_bar=i)
                    overlapping_fvg = self._find_overlapping_fvg(active_fvgs, ob)
                    if overlapping_fvg is None:
                        continue

                # ── All filters passed — build signal ─────────────────────
                sl_buf    = self.sl_buffer_atr * atr_i
                sl        = ob.low - sl_buf
                risk_dist = cl_i - sl
                if risk_dist <= 0:
                    continue

                tp = cl_i + self.risk_reward * risk_dist

                score = self._compute_score(
                    use_d1_ema = self.use_d1_ema,
                    use_h4     = self.use_h4_trend,
                    use_kz     = self.use_killzone,
                    use_sweep  = self.use_ltf_sweep,
                    has_fvg    = self.require_fvg,
                    bull_bar   = self.require_bullish_bar,
                )

                sig = ConvergenceSignal(
                    direction      = "long",
                    entry_price    = cl_i,
                    stop_loss      = sl,
                    take_profit    = tp,
                    risk_reward    = self.risk_reward,
                    formed_at      = pd.Timestamp(ts_i),
                    bar_index      = i,
                    ob_bar         = ob.bar_index,
                    ob_high        = ob.high,
                    ob_low         = ob.low,
                    fvg_bar        = overlapping_fvg.bar_index  if overlapping_fvg else -1,
                    fvg_top        = overlapping_fvg.top        if overlapping_fvg else 0.0,
                    fvg_bottom     = overlapping_fvg.bottom     if overlapping_fvg else 0.0,
                    ltf_sweep_bar  = sweep_bar,
                    zone_score     = score,
                )

                signals.append(sig)
                triggered_obs.add(ob.bar_index)
                cooldown.mark(i)
                break   # one signal per bar

        return signals

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_d1_regime(self, m15_df: pd.DataFrame) -> np.ndarray:
        """Map D1 EMA(d1_ema_span) bullish/bearish state to each M15 bar index."""
        d1_df = m15_df.resample("1D", closed="left", label="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna()

        if len(d1_df) < self.d1_ema_span:
            # Not enough daily bars — assume bullish (no filter)
            return np.ones(len(m15_df), dtype=bool)

        d1_ema    = d1_df["close"].ewm(span=self.d1_ema_span, adjust=False).mean()
        d1_bull   = (d1_df["close"] > d1_ema).to_numpy(dtype=bool)

        # Map each M15 bar to the last completed D1 bar
        m15_to_d1 = (
            np.searchsorted(d1_df.index.values, m15_df.index.values, side="right") - 1
        )
        m15_to_d1 = np.clip(m15_to_d1, 0, len(d1_df) - 1)
        return d1_bull[m15_to_d1]

    def _build_h4_trend(self, m15_df: pd.DataFrame) -> np.ndarray:
        """Map H4 EMA slope (bullish = rising) to each M15 bar index."""
        h4_df = m15_df.resample("4h", closed="left", label="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna()

        h4_ema     = h4_df["close"].ewm(span=self.h4_ema_span, adjust=False).mean()
        h4_v       = h4_ema.to_numpy(dtype=float)
        lb         = self.h4_slope_lb
        h4_prev    = np.empty_like(h4_v)
        h4_prev[:lb] = h4_v[0]
        h4_prev[lb:] = h4_v[:-lb]
        h4_bull    = h4_v > h4_prev
        h4_bull[:lb] = False

        m15_to_h4 = (
            np.searchsorted(h4_df.index.values, m15_df.index.values, side="right") - 1
        )
        m15_to_h4 = np.clip(m15_to_h4, 0, len(h4_df) - 1)
        return h4_bull[m15_to_h4]

    @staticmethod
    def _find_overlapping_fvg(
        active_fvgs: list[FairValueGap],
        ob: OrderBlock,
    ) -> FairValueGap | None:
        """Return the most recent bullish FVG whose zone overlaps the OB zone."""
        candidates = [
            fvg for fvg in active_fvgs
            if fvg.direction == BULLISH
            and fvg.bottom < ob.high   # FVG top is not fully above OB
            and fvg.top > ob.low       # FVG bottom is not fully below OB
        ]
        # Most recently detected FVG wins (strongest recency)
        return candidates[-1] if candidates else None

    @staticmethod
    def _find_sweep_bar(
        lows:    np.ndarray,
        closes:  np.ndarray,
        bar_index: int,
        lookback:  int,
    ) -> int:
        """Return the bar index of the LTF sweep within the lookback window."""
        start = max(1, bar_index - lookback + 1)
        for i in range(start, bar_index + 1):
            if lows[i] < lows[i - 1] and closes[i] > lows[i - 1]:
                return i
        return bar_index

    @staticmethod
    def _compute_score(
        use_d1_ema: bool,
        use_h4:     bool,
        use_kz:     bool,
        use_sweep:  bool,
        has_fvg:    bool,
        bull_bar:   bool,
    ) -> float:
        """Count enabled and satisfied filters; normalize to 0–10 scale."""
        total   = sum([use_d1_ema, use_h4, use_kz, use_sweep, has_fvg, bull_bar])
        enabled = 6   # always 6 possible filters
        return round(total / enabled * 10, 1)
