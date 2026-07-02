"""
Fair Value Gap Retest Strategy.

Concept
-------
A Fair Value Gap (FVG) marks a 3-bar momentum imbalance: after a strong
impulse bar the market leaves a gap between bar[-2].high and bar[0].low
(bullish) or bar[-2].low and bar[0].high (bearish).  Institutional players
tend to return to these imbalances before continuing in the original direction.

Signal flow (per M15 bar):
    1. Detect all active unmitigated M15 FVGs via detect_fvg().
    2. Align H4 EMA50 slope → primary directional bias.
    3. When price pulls back INTO an active FVG in the direction of H4 bias,
       apply quality filters A–H (each opt-in), then emit signal.
    4. Each FVG is used at most once.

Quality filters (all opt-in via constructor params)
-----------------------------------------------------
A  require_bullish_bar      : entry bar must close bullish (close > open)
B  fvg_max_penetration_pct  : price must not exceed X% into the FVG from top
C  use_h1_trend             : close must be above H1 EMA span
D  use_m15_bos_filter       : a bullish M15 BOS/CHoCH must exist within N bars
E  min_impulse_atr_mult     : FVG impulse bar must be ≥ X × ATR
F  session_start/end_utc    : restrict to active session hours
G  risk_reward              : R:R target (lower = higher WR)
H  use_sd_confluence        : FVG must overlap with an active M15 demand zone
I  use_d1_regime            : suspend trading when D1 close < D1 EMA (bear regime)
       d1_ema_span             : D1 EMA span for regime filter (default 200)
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from zeus.strategy.smc.fvg import detect_fvg
from zeus.strategy.smc.pivot import BEARISH, BULLISH, detect_pivots
from zeus.strategy.smc.structure import StructureType, detect_structure
from zeus.strategy.utils import SignalCooldown, atr_series as _atr_series


# ── Demand zone (for filter H) ────────────────────────────────────────────────

@dataclass(frozen=True)
class _DemandZone:
    bar_index:    int    # M15 bar of the pivot low
    confirmed_at: int    # bar_index + pivot_size (causal boundary)
    zone_top:     float  # proximal level (bodies near the pivot)
    zone_bottom:  float  # distal level (the pivot low)
    mitigated_at: int    # -1 = still active


def _detect_demand_zones(
    m15_df:     pd.DataFrame,
    pivot_size: int = 10,
) -> list[_DemandZone]:
    """
    Lightweight M15 demand zone detection from pivot lows.

    Zone top    = highest body (max of open/close) in [pivot−1, pivot]
    Zone bottom = pivot low
    Mitigation  = close below zone bottom after confirmation
    """
    pivots = detect_pivots(m15_df["high"], m15_df["low"], size=pivot_size)

    o  = m15_df["open"].to_numpy(dtype=float)
    lo = m15_df["low"].to_numpy(dtype=float)
    c  = m15_df["close"].to_numpy(dtype=float)
    n  = len(m15_df)

    zones: list[_DemandZone] = []

    for p in pivots:
        if p.is_high:
            continue
        bi = p.bar_index
        if bi < 1 or bi + pivot_size >= n:
            continue

        zone_bottom = float(lo[bi])
        zone_top    = max(
            max(float(o[bi]),     float(c[bi])),
            max(float(o[bi - 1]), float(c[bi - 1])),
        )
        if zone_top <= zone_bottom:
            continue

        # Mitigation: first bar (after confirmation) that closes below zone_bottom
        mitigated_at = -1
        for j in range(bi + pivot_size, n):
            if c[j] < zone_bottom:
                mitigated_at = j
                break

        zones.append(_DemandZone(
            bar_index    = bi,
            confirmed_at = bi + pivot_size,
            zone_top     = zone_top,
            zone_bottom  = zone_bottom,
            mitigated_at = mitigated_at,
        ))

    return zones


# ── Signal dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FVGSignal:
    """
    Trading signal produced by FVGRetestStrategy.

    Interface-compatible with sd_simulation.simulate_trade() and compute_metrics():
        direction, entry_price, stop_loss, take_profit, risk_reward,
        formed_at, bar_index, zone_score are accessed by the simulation engine.
    zone_score is set to 0.0 (not applicable for FVG).
    """
    direction:    str
    entry_price:  float
    stop_loss:    float
    take_profit:  float
    risk_reward:  float
    formed_at:    pd.Timestamp
    bar_index:    int
    fvg_top:      float
    fvg_bottom:   float
    fvg_bar:      int
    zone_score:   float = 0.0


# ── Strategy ─────────────────────────────────────────────────────────────────

class FVGRetestStrategy:
    """
    M15 FVG retest entries with optional A–H quality filters.

    Base parameters (from R18-B2)
    ────────────────────────────
    risk_reward          : R:R target (default 3.0)
    min_fvg_atr_mult     : minimum FVG size as ATR fraction (default 0.10)
    max_fvg_age_bars     : discard FVGs older than this many M15 bars (default 96)
    use_h4_trend         : require H4 EMA50 slope agreement (default True)
    h4_ema_span          : H4 EMA span (default 50)
    h4_slope_lb          : H4 bars for slope comparison (default 3)
    signal_cooldown      : minimum M15 bars between signals (default 12 = 3h)
    max_signals_per_day  : daily cap, 0 = unlimited (default 3)
    use_session_filter   : restrict to active session (default True)
    session_start_utc    : first UTC hour inclusive (default 7)
    session_end_utc      : last  UTC hour exclusive (default 21)
    sl_buffer_atr_mult   : SL buffer beyond FVG edge (default 0.10)
    long_only            : skip all bearish setups (default True)
    atr_period           : ATR smoothing period in M15 bars (default 14)

    Quality filters (all default off for backward compatibility)
    ────────────────────────────────────────────────────────────
    A  require_bullish_bar     : close > open on entry bar (default False)
    B  fvg_max_penetration_pct : max fraction of FVG price may penetrate (default 1.0 = off)
    C  use_h1_trend            : close > H1 EMA (default False)
       h1_ema_span             : H1 EMA span (default 21)
    D  use_m15_bos_filter      : require recent bullish M15 structure (default False)
       m15_bos_lookback        : max M15 bars since last bull BOS/CHoCH (default 30)
       m15_pivot_size          : pivot confirmation size for BOS detection (default 5)
    E  min_impulse_atr_mult    : FVG impulse bar ≥ X × ATR (default 0.0 = off)
    H  use_sd_confluence       : FVG must overlap active demand zone (default False)
       sd_pivot_size           : pivot size for demand zone detection (default 10)
       sd_zone_buffer_atr_mult : zone/FVG overlap tolerance in ATR (default 0.30)
    I  use_d1_regime           : skip entries when D1 close < D1 EMA (default False)
       d1_ema_span             : D1 EMA span (default 200)
    """

    def __init__(
        self,
        # ── Base ─────────────────────────────────────────────────────────────
        risk_reward:             float = 3.0,
        min_fvg_atr_mult:        float = 0.10,
        max_fvg_age_bars:        int   = 96,
        use_h4_trend:            bool  = True,
        h4_ema_span:             int   = 50,
        h4_slope_lb:             int   = 3,
        signal_cooldown:         int   = 12,
        max_signals_per_day:     int   = 3,
        use_session_filter:      bool  = True,
        session_start_utc:       int   = 7,
        session_end_utc:         int   = 21,
        sl_buffer_atr_mult:      float = 0.10,
        long_only:               bool  = True,
        atr_period:              int   = 14,
        # ── Filter A ─────────────────────────────────────────────────────────
        require_bullish_bar:     bool  = False,
        # ── Filter B ─────────────────────────────────────────────────────────
        fvg_max_penetration_pct: float = 1.0,
        # ── Filter C ─────────────────────────────────────────────────────────
        use_h1_trend:            bool  = False,
        h1_ema_span:             int   = 21,
        # ── Filter D ─────────────────────────────────────────────────────────
        use_m15_bos_filter:      bool  = False,
        m15_bos_lookback:        int   = 30,
        m15_pivot_size:          int   = 5,
        # ── Filter E ─────────────────────────────────────────────────────────
        min_impulse_atr_mult:    float = 0.0,
        # ── Filter H ─────────────────────────────────────────────────────────
        use_sd_confluence:       bool  = False,
        sd_pivot_size:           int   = 10,
        sd_zone_buffer_atr_mult: float = 0.30,
        # ── Filter I ─────────────────────────────────────────────────────────
        use_d1_regime:           bool  = False,
        d1_ema_span:             int   = 200,
    ) -> None:
        self.risk_reward             = risk_reward
        self.min_fvg_atr_mult        = min_fvg_atr_mult
        self.max_fvg_age_bars        = max_fvg_age_bars
        self.use_h4_trend            = use_h4_trend
        self.h4_ema_span             = h4_ema_span
        self.h4_slope_lb             = h4_slope_lb
        self.signal_cooldown         = signal_cooldown
        self.max_signals_per_day     = max_signals_per_day
        self.use_session_filter      = use_session_filter
        self.session_start_utc       = session_start_utc
        self.session_end_utc         = session_end_utc
        self.sl_buffer_atr_mult      = sl_buffer_atr_mult
        self.long_only               = long_only
        self.atr_period              = atr_period
        self.require_bullish_bar     = require_bullish_bar
        self.fvg_max_penetration_pct = fvg_max_penetration_pct
        self.use_h1_trend            = use_h1_trend
        self.h1_ema_span             = h1_ema_span
        self.use_m15_bos_filter      = use_m15_bos_filter
        self.m15_bos_lookback        = m15_bos_lookback
        self.m15_pivot_size          = m15_pivot_size
        self.min_impulse_atr_mult    = min_impulse_atr_mult
        self.use_sd_confluence       = use_sd_confluence
        self.sd_pivot_size           = sd_pivot_size
        self.sd_zone_buffer_atr_mult = sd_zone_buffer_atr_mult
        self.use_d1_regime           = use_d1_regime
        self.d1_ema_span             = d1_ema_span

    # ── Public ───────────────────────────────────────────────────────────────

    def run(self, m15_df: pd.DataFrame) -> list[FVGSignal]:
        """
        Detect FVG retest signals over a full M15 OHLCV DataFrame.

        Returns a chronological list of FVGSignal objects.
        Pass m15_df as both m1_df and the primary DataFrame to simulate_all().
        """
        if len(m15_df) < 2:
            return []

        # ── FVG detection ──────────────────────────────────────────────────
        fvgs = detect_fvg(
            m15_df["high"], m15_df["low"], m15_df["close"], m15_df["open"]
        )
        if not fvgs:
            return []

        # ── ATR (M15) ─────────────────────────────────────────────────────
        _atr = _atr_series(m15_df, self.atr_period).to_numpy(dtype=float)

        # ── Price arrays ──────────────────────────────────────────────────
        _highs  = m15_df["high"].to_numpy(dtype=float)
        _lows   = m15_df["low"].to_numpy(dtype=float)
        _closes = m15_df["close"].to_numpy(dtype=float)
        _opens  = m15_df["open"].to_numpy(dtype=float)
        _hours  = m15_df.index.hour
        _dates  = np.array(m15_df.index.date)
        n = len(m15_df)

        # ── H4 EMA50 trend ────────────────────────────────────────────────
        _h4_trend: np.ndarray | None = None
        if self.use_h4_trend:
            h4_df = m15_df.resample("4h", closed="left", label="left").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last"}
            ).dropna()
            h4_ema   = h4_df["close"].ewm(span=self.h4_ema_span, adjust=False).mean()
            _h4_v    = h4_ema.to_numpy(dtype=float)
            _lb      = self.h4_slope_lb
            _h4_p    = np.empty_like(_h4_v)
            _h4_p[:_lb] = _h4_v[0]
            _h4_p[_lb:] = _h4_v[:-_lb]
            _h4_bull = _h4_v > _h4_p
            _h4_bull[:_lb] = False
            m15_to_h4 = np.searchsorted(
                h4_df.index.values, m15_df.index.values, side="right"
            ) - 1
            m15_to_h4 = np.clip(m15_to_h4, 0, len(h4_df) - 1)
            _h4_trend = _h4_bull[m15_to_h4]

        # ── Filter C : H1 EMA ─────────────────────────────────────────────
        _h1_ema_m15: np.ndarray | None = None
        if self.use_h1_trend:
            h1_df = m15_df.resample("1h", closed="left", label="left").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last"}
            ).dropna()
            h1_ema     = h1_df["close"].ewm(span=self.h1_ema_span, adjust=False).mean()
            _h1_v      = h1_ema.to_numpy(dtype=float)
            m15_to_h1  = np.searchsorted(
                h1_df.index.values, m15_df.index.values, side="right"
            ) - 1
            m15_to_h1  = np.clip(m15_to_h1, 0, len(h1_df) - 1)
            _h1_ema_m15 = _h1_v[m15_to_h1]

        # ── Filter D : M15 market structure ──────────────────────────────
        _last_bull_struct: np.ndarray | None = None
        _last_bear_struct: np.ndarray | None = None
        if self.use_m15_bos_filter:
            m15_pivots = detect_pivots(
                m15_df["high"], m15_df["low"], size=self.m15_pivot_size
            )
            struct_events = detect_structure(m15_df["close"], m15_pivots, is_internal=True)

            bull_bars = sorted(
                ev.bar_index for ev in struct_events if ev.direction == BULLISH
            )
            bear_bars = sorted(
                ev.bar_index for ev in struct_events if ev.direction == BEARISH
            )

            _last_bull_struct = np.full(n, -999, dtype=int)
            _last_bear_struct = np.full(n, -999, dtype=int)
            last_b = -999
            last_r = -999
            bi_idx = 0
            ri_idx = 0
            for i in range(n):
                while bi_idx < len(bull_bars) and bull_bars[bi_idx] <= i:
                    last_b = bull_bars[bi_idx]
                    bi_idx += 1
                while ri_idx < len(bear_bars) and bear_bars[ri_idx] <= i:
                    last_r = bear_bars[ri_idx]
                    ri_idx += 1
                _last_bull_struct[i] = last_b
                _last_bear_struct[i] = last_r

        # ── Filter H : S&D demand zone confluence ─────────────────────────
        _demand_zones: list[_DemandZone] | None = None
        if self.use_sd_confluence:
            _demand_zones = _detect_demand_zones(m15_df, pivot_size=self.sd_pivot_size)

        # ── Filter I : D1 regime (close > D1 EMA) ────────────────────────
        _d1_regime_m15: np.ndarray | None = None
        if self.use_d1_regime:
            d1_df = m15_df.resample("1D", closed="left", label="left").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last"}
            ).dropna()
            d1_ema    = d1_df["close"].ewm(span=self.d1_ema_span, adjust=False).mean()
            _d1_v     = d1_ema.to_numpy(dtype=float)
            _d1_cl    = d1_df["close"].to_numpy(dtype=float)
            # Map each M15 bar to the *previous completed* D1 bar (causal)
            m15_to_d1 = np.searchsorted(
                d1_df.index.values, m15_df.index.values, side="right"
            ) - 1
            m15_to_d1 = np.clip(m15_to_d1, 0, len(d1_df) - 1)
            # Bull regime: previous D1 close > D1 EMA at that bar
            _d1_regime_m15 = _d1_cl[m15_to_d1] > _d1_v[m15_to_d1]

        # ── Main signal loop ──────────────────────────────────────────────
        signals: list[FVGSignal] = []
        _entered: set[int]  = set()
        cooldown = SignalCooldown(self.signal_cooldown)
        _daily_count: dict  = {}

        for i in range(3, n):
            # Session filter (F)
            if self.use_session_filter and not (
                self.session_start_utc <= int(_hours[i]) < self.session_end_utc
            ):
                continue

            # Daily cap
            date_key = _dates[i]
            if (
                self.max_signals_per_day > 0
                and _daily_count.get(date_key, 0) >= self.max_signals_per_day
            ):
                continue

            # Cooldown
            if not cooldown.can_signal(i):
                continue

            h4_bull  = bool(_h4_trend[i]) if _h4_trend is not None else True
            lo_i     = float(_lows[i])
            hi_i     = float(_highs[i])
            cl_i     = float(_closes[i])
            op_i     = float(_opens[i])
            atr_i    = max(float(_atr[i]), 1e-6)

            # Filter A : bullish entry bar
            if self.require_bullish_bar and cl_i <= op_i:
                continue

            # Filter C : H1 EMA
            if _h1_ema_m15 is not None and cl_i <= float(_h1_ema_m15[i]):
                continue

            # Filter D : M15 structure
            if _last_bull_struct is not None:
                last_bull = int(_last_bull_struct[i])
                last_bear = int(_last_bear_struct[i])
                if last_bull == -999:
                    continue  # no bullish structure yet
                if last_bull <= last_bear:
                    continue  # bearish structure is more recent
                if i - last_bull > self.m15_bos_lookback:
                    continue  # bullish structure too old

            # Filter I : D1 regime
            if _d1_regime_m15 is not None and not bool(_d1_regime_m15[i]):
                continue

            for fvg in fvgs:
                if fvg.bar_index >= i:
                    continue
                if fvg.mitigated_at != -1 and fvg.mitigated_at <= i:
                    continue
                if fvg.bar_index in _entered:
                    continue
                if i - fvg.bar_index > self.max_fvg_age_bars:
                    continue

                fvg_size = fvg.top - fvg.bottom
                if fvg_size < self.min_fvg_atr_mult * atr_i:
                    continue

                # Long-only
                if fvg.direction != BULLISH:
                    continue
                if _h4_trend is not None and not h4_bull:
                    continue

                # Price touching the FVG from above
                if lo_i > fvg.top:
                    continue

                # Filter B : max penetration
                if self.fvg_max_penetration_pct < 1.0 and fvg_size > 0:
                    allowed_low = fvg.top - fvg_size * self.fvg_max_penetration_pct
                    if lo_i < allowed_low:
                        continue

                # Filter E : impulse bar strength
                if self.min_impulse_atr_mult > 0.0:
                    b_idx       = fvg.bar_index - 1
                    impulse_sz  = abs(float(_closes[b_idx]) - float(_opens[b_idx]))
                    if impulse_sz < self.min_impulse_atr_mult * float(_atr[b_idx]):
                        continue

                # Filter H : S&D demand zone confluence
                if _demand_zones is not None:
                    sd_buf     = self.sd_zone_buffer_atr_mult * atr_i
                    confluent  = False
                    for dz in _demand_zones:
                        if dz.confirmed_at > i:
                            continue
                        if dz.mitigated_at != -1 and dz.mitigated_at <= i:
                            continue
                        # Overlap: [dz.bottom−buf, dz.top+buf] ∩ [fvg.bottom, fvg.top]
                        if dz.zone_top + sd_buf >= fvg.bottom \
                                and dz.zone_bottom - sd_buf <= fvg.top:
                            confluent = True
                            break
                    if not confluent:
                        continue

                # ── Signal construction ──────────────────────────────────
                sl_buf    = max(self.sl_buffer_atr_mult * atr_i, fvg_size * 0.10)
                sl        = fvg.bottom - sl_buf
                risk_dist = cl_i - sl
                if risk_dist <= 0:
                    continue
                tp        = cl_i + self.risk_reward * risk_dist

                _entered.add(fvg.bar_index)
                cooldown.mark(i)
                _daily_count[date_key] = _daily_count.get(date_key, 0) + 1

                signals.append(FVGSignal(
                    direction   = "long",
                    entry_price = round(cl_i, 2),
                    stop_loss   = round(sl, 2),
                    take_profit = round(tp, 2),
                    risk_reward = self.risk_reward,
                    formed_at   = m15_df.index[i],
                    bar_index   = i,
                    fvg_top     = round(fvg.top, 2),
                    fvg_bottom  = round(fvg.bottom, 2),
                    fvg_bar     = fvg.bar_index,
                ))

                if (
                    self.max_signals_per_day > 0
                    and _daily_count.get(date_key, 0) >= self.max_signals_per_day
                ):
                    break

        return signals
