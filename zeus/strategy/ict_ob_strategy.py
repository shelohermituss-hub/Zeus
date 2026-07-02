"""
ICT Order Block Strategy.

Entry logic
-----------
  Long  : price retraces to a bullish Order Block zone after a bullish BOS/CHoCH.
           Entry bar must touch the OB (low ≤ ob.high) and close inside (cl ≥ ob.low).
           Optional: close must be bullish (close > open).
  Short : mirror logic for bearish OB.
           (long_only=True skips bearish by default.)

Order Block origin
------------------
  Derived from external swing structure (pivot_size_swing).  A bullish BOS/CHoCH
  at bar B means price broke above the swing high at pivot P.  The OB is the bar
  with the lowest parsed_low in [pivot_bar, BOS_bar).  This candle represents
  the last institutional selling before the bullish break — demand zone.

  Internal structure (pivot_size_internal) is used for CHoCH-only setups where
  a smaller-timeframe reversal is sought.

Stop-loss
---------
  ob.low  − sl_buffer × ATR  (bullish)
  ob.high + sl_buffer × ATR  (bearish)

Take-profit
-----------
  entry ± risk_reward × risk_distance

Filters available
-----------------
  use_h4_trend     : H4 EMA50 must be rising for longs
  use_killzone     : entry bar must fall within London (07–11 UTC) or NY (12–15 UTC)
  structure_filter : "both" | "bos_only" | "choch_only"
  min_ob_age       : OB must be at least N bars old before it qualifies as entry zone
  max_ob_age       : OB is invalidated after N bars without a retest
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from zeus.strategy.smc.order_block import detect_order_blocks, OrderBlock
from zeus.strategy.smc.pivot import detect_pivots, BULLISH, BEARISH
from zeus.strategy.smc.session import is_in_killzone
from zeus.strategy.smc.structure import detect_structure, StructureType


# ── ATR helper ────────────────────────────────────────────────────────────────

def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, lo, c = df["high"], df["low"], df["close"]
    tr = pd.concat([
        h - lo,
        (h - c.shift(1)).abs(),
        (lo - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ── Signal dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ICTSignal:
    """
    Trading signal produced by ICTObStrategy.

    Interface-compatible with sd_simulation.simulate_trade() and compute_metrics():
    direction, entry_price, stop_loss, take_profit, risk_reward,
    formed_at, bar_index, zone_score are accessed by the simulation engine.
    """
    direction:      str
    entry_price:    float
    stop_loss:      float
    take_profit:    float
    risk_reward:    float
    formed_at:      pd.Timestamp
    bar_index:      int
    ob_bar:         int
    ob_high:        float
    ob_low:         float
    structure_type: str    # "BOS" | "CHoCH"
    zone_score:     float = 0.0


# ── Strategy ──────────────────────────────────────────────────────────────────

class ICTObStrategy:
    """
    M15 ICT Order Block strategy.

    Parameters
    ----------
    pivot_size_swing    : bars for external swing confirmation (default 10)
    pivot_size_internal : bars for internal structure (default 5)
    long_only           : only bullish setups (default True)
    use_h4_trend        : require H4 EMA50 bullish slope for long entries
    h4_ema_span         : H4 EMA span (default 50)
    h4_slope_lb         : bars for H4 slope comparison (default 3)
    use_killzone        : entry bar must fall in London or NY killzone (UTC)
    structure_filter    : "both" | "bos_only" | "choch_only"
    risk_reward         : R:R target (default 2.0)
    sl_buffer_atr       : SL buffer in ATR beyond OB edge (default 0.20)
    min_ob_age          : OB must be ≥ N bars old to be actionable (default 2)
    max_ob_age          : cancel OB after N bars without retest (default 200)
    signal_cooldown     : min M15 bars between signals (default 12 = 3h)
    require_bullish_bar : entry bar must close bullish for longs (default True)
    atr_period          : ATR smoothing period (default 14)
    """

    def __init__(
        self,
        pivot_size_swing:    int   = 10,
        pivot_size_internal: int   = 5,
        long_only:           bool  = True,
        use_h4_trend:        bool  = True,
        h4_ema_span:         int   = 50,
        h4_slope_lb:         int   = 3,
        use_killzone:        bool  = False,
        structure_filter:    str   = "both",
        risk_reward:         float = 2.0,
        sl_buffer_atr:       float = 0.20,
        min_ob_age:          int   = 2,
        max_ob_age:          int   = 200,
        signal_cooldown:     int   = 12,
        require_bullish_bar: bool  = True,
        atr_period:          int   = 14,
    ) -> None:
        self.pivot_size_swing    = pivot_size_swing
        self.pivot_size_internal = pivot_size_internal
        self.long_only           = long_only
        self.use_h4_trend        = use_h4_trend
        self.h4_ema_span         = h4_ema_span
        self.h4_slope_lb         = h4_slope_lb
        self.use_killzone        = use_killzone
        self.structure_filter    = structure_filter
        self.risk_reward         = risk_reward
        self.sl_buffer_atr       = sl_buffer_atr
        self.min_ob_age          = min_ob_age
        self.max_ob_age          = max_ob_age
        self.signal_cooldown     = signal_cooldown
        self.require_bullish_bar = require_bullish_bar
        self.atr_period          = atr_period

    def run(self, m15_df: pd.DataFrame) -> list[ICTSignal]:
        """
        Run strategy over a full M15 OHLCV DataFrame.

        Returns a chronologically ordered list of ICTSignal objects.
        """
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

        # ── Detect order blocks from swing structure ───────────────────────
        swing_pivots = detect_pivots(m15_df["high"], m15_df["low"], size=self.pivot_size_swing)
        swing_events = detect_structure(m15_df["close"], swing_pivots, is_internal=False)

        # Filter structure events by structure_filter
        if self.structure_filter == "bos_only":
            filtered_events = [e for e in swing_events if e.structure_type == StructureType.BOS]
        elif self.structure_filter == "choch_only":
            filtered_events = [e for e in swing_events if e.structure_type == StructureType.CHOCH]
        else:
            filtered_events = swing_events

        # Limit to bullish events for long-only
        if self.long_only:
            filtered_events = [e for e in filtered_events if e.direction == BULLISH]

        obs = detect_order_blocks(
            m15_df["high"], m15_df["low"], m15_df["close"],
            filtered_events,
            atr_period=200,
        )

        if not obs:
            return []

        # Separate bullish/bearish OBs
        bull_obs = [ob for ob in obs if ob.direction == BULLISH]
        bear_obs = [] if self.long_only else [ob for ob in obs if ob.direction == BEARISH]

        # ── Main loop ─────────────────────────────────────────────────────
        signals:       list[ICTSignal] = []
        triggered_obs: set[int]        = set()   # ob.bar_index keys
        _last_sig_bar: int             = -999

        for i in range(4, n):
            cl_i  = float(_closes[i])
            op_i  = float(_opens[i])
            lo_i  = float(_lows[i])
            hi_i  = float(_highs[i])
            atr_i = max(float(_atr[i]), 1e-6)
            ts_i  = m15_df.index[i]

            h4_bull = bool(_h4_trend[i]) if _h4_trend is not None else True

            # ── Killzone check ────────────────────────────────────────────
            if self.use_killzone and not is_in_killzone(ts_i):
                continue

            # ── Cooldown ──────────────────────────────────────────────────
            if i - _last_sig_bar < self.signal_cooldown:
                continue

            # ── Scan active bullish OBs ───────────────────────────────────
            if self.long_only or True:
                for ob in bull_obs:
                    if ob.bar_index in triggered_obs:
                        continue
                    if ob.detected_at > i:
                        continue
                    ob_age = i - ob.detected_at
                    if ob_age < self.min_ob_age:
                        continue
                    if ob_age > self.max_ob_age:
                        continue
                    # OB already mitigated (price broke below OB low)?
                    if ob.mitigated_at != -1 and ob.mitigated_at <= i:
                        continue

                    if self.use_h4_trend and not h4_bull:
                        continue

                    # PRZ touch: bar dips into OB zone but closes above OB low
                    if lo_i > ob.high:
                        continue
                    if cl_i < ob.low:
                        continue

                    # Confirmation: bullish bar
                    if self.require_bullish_bar and cl_i <= op_i:
                        continue

                    sl_buf    = self.sl_buffer_atr * atr_i
                    sl        = ob.low - sl_buf
                    risk_dist = cl_i - sl
                    if risk_dist <= 0:
                        continue
                    tp = cl_i + self.risk_reward * risk_dist

                    stype = (
                        "BOS" if any(
                            e.bar_index == ob.detected_at and e.structure_type == StructureType.BOS
                            for e in filtered_events
                        )
                        else "CHoCH"
                    )

                    triggered_obs.add(ob.bar_index)
                    _last_sig_bar = i

                    signals.append(ICTSignal(
                        direction      = "long",
                        entry_price    = round(cl_i, 2),
                        stop_loss      = round(sl, 2),
                        take_profit    = round(tp, 2),
                        risk_reward    = self.risk_reward,
                        formed_at      = ts_i,
                        bar_index      = i,
                        ob_bar         = ob.bar_index,
                        ob_high        = round(ob.high, 2),
                        ob_low         = round(ob.low, 2),
                        structure_type = stype,
                    ))
                    break  # one signal per bar

        return signals
