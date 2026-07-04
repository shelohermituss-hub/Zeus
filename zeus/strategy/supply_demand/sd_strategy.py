"""
Supply & Demand strategy — main signal generator.

Combines:
    - M15 S&D zone detection  (ZoneDetector)
    - M1 Wyckoff confirmation (WyckoffDetector)
    - Optional Fibonacci OTE  (zone_fib_confluence)

Signal flow (per M1 bar, index i):
    1. Update active zone state (mitigation, freshness, touch count)
    2. For each zone that contains the current price:
       a. Run WyckoffDetector on recent M1 bars
       b. If Wyckoff MSS fires at bar i → score → emit SDSignal

Entry  : Wyckoff mss_close
Stop   : zone.wick_extreme (tight; just beyond manipulation wick)
Target : entry ± risk_reward × |entry − stop|

Break-even (BE) at 1R is an execution concern handled outside this module.
The strategy outputs SL/TP levels only.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .fibonacci import FibConfluence, FibLevels, zone_fib_confluence
from .pivot_candle import PivotSide
from .wyckoff import WyckoffDetector, WyckoffPattern
from .zone_detector import SDZone, ZoneDetector, ZoneScore


def int_iter(bool_arr: np.ndarray):
    """Yield integer indices where bool_arr is True (avoids np.where overhead)."""
    return np.flatnonzero(bool_arr)


def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index on the given OHLCV DataFrame."""
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    tr    = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    up   = high - high.shift(1)
    down = low.shift(1) - low
    dm_plus  = up.where((up > down) & (up > 0),   0.0)
    dm_minus = down.where((down > up) & (down > 0), 0.0)

    atr      = tr.ewm(span=period, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm(span=period,  adjust=False).mean() / atr.replace(0, 1e-10)
    di_minus = 100 * dm_minus.ewm(span=period, adjust=False).mean() / atr.replace(0, 1e-10)
    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, 1e-10)
    return dx.ewm(span=period, adjust=False).mean()


# ── Signal data type ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SDSignal:
    """
    Trading signal produced by the Supply & Demand strategy.

    Attributes
    ----------
    direction        : "long" (demand zone) or "short" (supply zone)
    entry_price      : Wyckoff MSS close — in live, enter at next M1 bar open
    stop_loss        : zone.wick_extreme — tight stop beyond manipulation wick
    take_profit      : entry ± risk_reward × |entry − stop|
    risk_reward      : R:R ratio configured on the strategy
    zone_score       : S&D zone quality 0–10 (at signal time, includes fresh decay)
    wyckoff_score    : Wyckoff pattern quality 0–10
    fib_bonus        : Fibonacci OTE confluence bonus 0–2 (0 if not enabled)
    composite_score  : zone_score + fib_bonus (0–12)
    zone             : the triggering S&D zone
    wyckoff          : the Wyckoff pattern that confirmed entry
    fib              : FibConfluence result (None if Fibonacci disabled)
    formed_at        : M1 timestamp of the MSS bar (signal generation time)
    bar_index        : integer index of the signal bar in m1_df
    """
    direction:       str
    entry_price:     float
    stop_loss:       float
    take_profit:     float
    risk_reward:     float
    zone_score:      float
    wyckoff_score:   float
    fib_bonus:       float
    composite_score: float
    zone:            SDZone
    wyckoff:         WyckoffPattern
    fib:             Optional[FibConfluence]
    formed_at:       pd.Timestamp
    bar_index:       int

    @property
    def risk(self) -> float:
        """Absolute distance from entry to stop-loss (= 1R)."""
        return abs(self.entry_price - self.stop_loss)

    @property
    def reward(self) -> float:
        """Absolute distance from entry to take-profit."""
        return abs(self.take_profit - self.entry_price)


# ── Strategy ──────────────────────────────────────────────────────────────────

class SDStrategy:
    """
    Supply & Demand strategy: M15 zones + M1 Wyckoff entry confirmation.

    Parameters
    ----------
    zone_detector      : ZoneDetector instance (default: ZoneDetector())
    wyckoff_detector   : WyckoffDetector instance (default: WyckoffDetector())
    risk_reward        : R:R target (default 3.0)
    min_zone_score     : minimum S&D zone total score (default 5.0)
    min_wyckoff_score  : minimum Wyckoff pattern score (default 4.0)
    min_composite_score: minimum zone_score + fib_bonus (default 5.0)
    signal_cooldown    : M1 bars before the same zone can re-signal (default 30)
    use_trend_filter     : when True, demand zones require a bullish M15 EMA50 slope
                           and supply zones require a bearish slope (default True)
    trend_slope_lookback : number of M15 bars used to measure the EMA50 slope;
                           larger → smoother, less reactive (default 8)
    use_price_above_ema  : when True, also require M15 close > EMA50 for demand
                           (and close < EMA50 for supply) — harder confirmation
    use_session_filter   : when True, only emit signals during active trading hours
                           (session_start_utc..session_end_utc, default True)
    session_start_utc    : first UTC hour of the active session, inclusive (default 7)
    session_end_utc      : last  UTC hour of the active session, exclusive (default 21)
    max_signals_per_day  : global cap on signals per calendar day; 0 = unlimited
                           (default 2)
    use_adx_filter       : when True, only emit signals when M15 ADX ≥ adx_min
    adx_min              : minimum ADX value to trade (default 20.0)
    adx_period           : ADX smoothing period in M15 bars (default 14)
    min_zone_score_long  : override min_zone_score for demand (long) signals only;
                           None → uses min_zone_score (default None)
    min_zone_score_short : override min_zone_score for supply (short) signals only;
                           None → uses min_zone_score (default None)
    min_wyckoff_score_long  : override min_wyckoff_score for demand signals only;
                              None → uses min_wyckoff_score (default None)
    min_wyckoff_score_short : override min_wyckoff_score for supply signals only;
                              None → uses min_wyckoff_score (default None)
    use_h4_trend_filter  : when True, also require H4 EMA50 slope to agree with the
                           M15 trend (bullish H4 for longs, bearish for shorts)
    h4_trend_slope_lb    : number of H4 bars used to measure the EMA50 slope (default 3)
    use_rsi_filter       : when True, only enter longs when M1 RSI ≤ rsi_oversold
                           and shorts when M1 RSI ≥ rsi_overbought
    rsi_period           : RSI smoothing period in M1 bars (default 14)
    rsi_oversold         : RSI upper bound for long entries (default 35.0)
    rsi_overbought       : RSI lower bound for short entries (default 65.0)
    max_daily_losses     : stop emitting signals for the rest of a calendar day once
                           this many losses have been incurred that day; 0 = unlimited
    """

    def __init__(
        self,
        zone_detector:       Optional[ZoneDetector]    = None,
        wyckoff_detector:    Optional[WyckoffDetector] = None,
        risk_reward:         float = 3.0,
        min_zone_score:      float = 5.0,
        min_wyckoff_score:   float = 4.0,
        min_composite_score: float = 5.0,
        signal_cooldown:     int   = 30,
        use_trend_filter:    bool  = True,
        trend_slope_lookback: int  = 8,
        use_price_above_ema: bool  = True,
        use_session_filter:  bool  = True,
        session_start_utc:   int   = 7,
        session_end_utc:     int   = 21,
        max_signals_per_day: int   = 2,
        use_adx_filter:      bool  = False,
        adx_min:             float = 20.0,
        adx_period:          int   = 14,
        first_signal_per_zone: bool = False,
        min_zone_score_long:    Optional[float] = None,
        min_zone_score_short:   Optional[float] = None,
        min_wyckoff_score_long:  Optional[float] = None,
        min_wyckoff_score_short: Optional[float] = None,
        use_h4_trend_filter:  bool  = False,
        h4_trend_slope_lb:    int   = 3,
        use_rsi_filter:       bool  = False,
        rsi_period:           int   = 14,
        rsi_oversold:         float = 35.0,
        rsi_overbought:       float = 65.0,
        min_sl_pips:          float = 0.0,
        pip_size:             float = 0.0001,
        ema_atr_tolerance:    float = 0.0,
        min_score_product:    float = 0.0,
    ) -> None:
        self._zones    = zone_detector    or ZoneDetector()
        self._wyckoff  = wyckoff_detector or WyckoffDetector()
        self.risk_reward             = risk_reward
        self.min_zone_score          = min_zone_score
        self.min_wyckoff_score       = min_wyckoff_score
        self.min_composite_score     = min_composite_score
        self.signal_cooldown         = signal_cooldown
        self.use_trend_filter        = use_trend_filter
        self.trend_slope_lookback    = trend_slope_lookback
        self.use_price_above_ema     = use_price_above_ema
        self.use_session_filter      = use_session_filter
        self.session_start_utc       = session_start_utc
        self.session_end_utc         = session_end_utc
        self.max_signals_per_day     = max_signals_per_day
        self.use_adx_filter          = use_adx_filter
        self.adx_min                 = adx_min
        self.adx_period              = adx_period
        self.first_signal_per_zone   = first_signal_per_zone
        self.min_zone_score_long     = min_zone_score_long  if min_zone_score_long  is not None else min_zone_score
        self.min_zone_score_short    = min_zone_score_short if min_zone_score_short is not None else min_zone_score
        self.min_wyckoff_score_long  = min_wyckoff_score_long  if min_wyckoff_score_long  is not None else min_wyckoff_score
        self.min_wyckoff_score_short = min_wyckoff_score_short if min_wyckoff_score_short is not None else min_wyckoff_score
        self.use_h4_trend_filter  = use_h4_trend_filter
        self.h4_trend_slope_lb    = h4_trend_slope_lb
        self.use_rsi_filter       = use_rsi_filter
        self.rsi_period           = rsi_period
        self.rsi_oversold         = rsi_oversold
        self.rsi_overbought       = rsi_overbought
        self.min_sl_pips          = min_sl_pips
        self.pip_size             = pip_size
        self.ema_atr_tolerance    = ema_atr_tolerance
        self.min_score_product    = min_score_product

    # ── Public ────────────────────────────────────────────────────────────────

    def run(
        self,
        m15_df:              pd.DataFrame,
        m1_df:               pd.DataFrame,
        htf_fibs:            Optional[FibLevels] = None,
        pre_detected_zones:  Optional[list]      = None,
    ) -> list[SDSignal]:
        """
        Full historical backtest: detect M15 zones → iterate M1 bars for signals.

        Parameters
        ----------
        m15_df   : M15 OHLCV DataFrame — used for zone detection
        m1_df    : M1  OHLCV DataFrame — used for Wyckoff confirmation
        htf_fibs : optional pre-computed FibLevels for OTE confluence scoring;
                   None → Fibonacci disabled (fib_bonus stays 0)

        Returns
        -------
        list[SDSignal] in chronological order.

        Notes
        -----
        Zone detection (detect_zones) is a historical scan that uses bars on
        both sides of each pivot.  The temporal causality constraint is enforced
        by only making a zone visible once its formed_at timestamp has passed on
        the M1 timeline (``zone.formed_at <= m1_bar_timestamp``).
        """
        # Zone detection: accept pre-computed zones (saves ~10s per variant in batch runs)
        if pre_detected_zones is not None:
            all_zones = copy.deepcopy(pre_detected_zones)
        else:
            all_zones = self._zones.detect_zones(m15_df)
        if not all_zones:
            return []

        signals: list[SDSignal] = []
        # (formed_at, side, zone_bottom) → last M1 bar index that generated a signal
        _last_signal: dict[tuple, int] = {}
        # "YYYY-MM-DD" → number of signals emitted that day
        _daily_count: dict[str, int] = {}

        # M15 EMA50: used for slope (trend direction) and price-position checks
        m15_ema50  = m15_df["close"].ewm(span=50, adjust=False).mean()
        _slope_lb  = self.trend_slope_lookback

        # M15 ADX (optional)
        m15_adx: pd.Series | None = None
        if self.use_adx_filter:
            m15_adx = _compute_adx(m15_df, self.adx_period)

        def _trend_is_bullish(ts: pd.Timestamp) -> bool:
            pos = m15_ema50.index.searchsorted(ts, side="right") - 1
            if pos < _slope_lb:
                return False
            return float(m15_ema50.iloc[pos]) > float(m15_ema50.iloc[pos - _slope_lb])

        def _price_above_ema(ts: pd.Timestamp) -> bool:
            pos = m15_ema50.index.searchsorted(ts, side="right") - 1
            if pos < 0:
                return False
            return float(m15_df["close"].iloc[pos]) > float(m15_ema50.iloc[pos])

        def _adx_ok(ts: pd.Timestamp) -> bool:
            if m15_adx is None:
                return True
            pos = m15_adx.index.searchsorted(ts, side="right") - 1
            if pos < 0:
                return False
            return float(m15_adx.iloc[pos]) >= self.adx_min

        # ── Zone geometry arrays ──────────────────────────────────────────────
        _zones_sorted = sorted(all_zones, key=lambda z: z.formed_at)
        n_zones_total = len(_zones_sorted)

        _zt  = np.array([z.zone_top     for z in _zones_sorted], dtype=float)
        _zb  = np.array([z.zone_bottom  for z in _zones_sorted], dtype=float)
        _we  = np.array([z.wick_extreme for z in _zones_sorted], dtype=float)
        _dem = np.array([z.side == PivotSide.DEMAND for z in _zones_sorted], dtype=bool)
        _bnf = np.array([z.score.total - z.score.fresh for z in _zones_sorted], dtype=float)
        # Current freshness and mitigation can differ from defaults when pre_detected_zones
        # carries non-fresh state or tests inject custom fixtures.
        _init_fresh = np.array([z.score.fresh  for z in _zones_sorted], dtype=float)
        _init_mit   = np.array([z.is_mitigated for z in _zones_sorted], dtype=bool)

        # Precompute M1 price columns as plain numpy arrays (avoids per-bar pandas overhead)
        _m1_lows   = m1_df["low"].to_numpy(dtype=float)
        _m1_highs  = m1_df["high"].to_numpy(dtype=float)
        _m1_closes = m1_df["close"].to_numpy(dtype=float)
        _m1_opens  = m1_df["open"].to_numpy(dtype=float)
        _m1_index  = m1_df.index
        n_m1       = len(m1_df)

        # Bar index at which each zone becomes active (first M1 bar with ts > formed_at)
        _zone_formed_ns = np.array(
            [z.formed_at.value for z in _zones_sorted], dtype="datetime64[ns]"
        )
        _zone_active_from = np.searchsorted(m1_df.index.values, _zone_formed_ns, side="right")

        # ── PASS 1: vectorized zone state precomputation ──────────────────────
        # For each zone, find:
        #   first_touch_idx: first M1 bar where price enters the zone body → fresh 2→1
        #   mit_idx:         first M1 bar where close passes through wick   → zone dead
        # These are the only events that affect zone score.
        INF = n_m1  # sentinel: event never happens
        _first_touch_idx = np.full(n_zones_total, INF, dtype=np.int64)
        _mit_idx         = np.full(n_zones_total, INF, dtype=np.int64)

        for j in range(n_zones_total):
            start = int(_zone_active_from[j])
            if start >= n_m1:
                continue
            if _init_mit[j]:
                _mit_idx[j] = -1  # already mitigated before any M1 bar
                continue

            lo_sl = _m1_lows[start:]
            hi_sl = _m1_highs[start:]
            cl_sl = _m1_closes[start:]

            top = _zt[j]; bot = _zb[j]; we = _we[j]
            in_zone = (lo_sl <= top) & (hi_sl >= bot)

            # First bar where price enters zone body
            idx_local = int(np.argmax(in_zone)) if in_zone.any() else -1
            if idx_local < 0:
                continue  # zone never touched within M1 data
            _first_touch_idx[j] = start + idx_local

            # First bar where close passes through wick (mitigation)
            if _dem[j]:
                mit_mask = in_zone & (cl_sl < we)
            else:
                mit_mask = in_zone & (cl_sl > we)
            if mit_mask.any():
                _mit_idx[j] = start + int(np.argmax(mit_mask))

            # Honor pre-existing state (zone already touched before our M1 window)
            if _init_fresh[j] < 2.0 and _first_touch_idx[j] == INF:
                _first_touch_idx[j] = 0  # treat as first-touch before bar 0

        # ── Precompute per-bar arrays to avoid pandas overhead in loop ────────
        _m1_hours   = m1_df.index.hour                        # int array
        _m1_ts_vals = m1_df.index.values                      # int64 nanoseconds

        # Precompute trend as M1-aligned boolean arrays (avoids per-bar searchsorted)
        m1_to_m15 = np.searchsorted(m15_df.index.values, _m1_ts_vals, side="right") - 1
        m1_to_m15 = np.clip(m1_to_m15, 0, len(m15_df) - 1)
        _ema50_vals  = m15_ema50.to_numpy(dtype=float)
        _ema50_prev  = np.empty_like(_ema50_vals)
        _ema50_prev[:_slope_lb] = _ema50_vals[0]
        _ema50_prev[_slope_lb:] = _ema50_vals[:-_slope_lb]
        _trend_bull_m15 = (_ema50_vals > _ema50_prev)
        _trend_bull_m15[:_slope_lb] = False
        _trend_bull = _trend_bull_m15[m1_to_m15]               # M1-aligned
        _close_m15  = m15_df["close"].to_numpy(dtype=float)
        _above_ema  = (_close_m15 > _ema50_vals)[m1_to_m15]    # M1-aligned

        if self.use_adx_filter and m15_adx is not None:
            _adx_vals   = m15_adx.to_numpy(dtype=float)
            _adx_ok_arr = (_adx_vals >= self.adx_min)[m1_to_m15]
        else:
            _adx_ok_arr = None

        # EMA zone arrays (M1-aligned): ATR-tolerance zone or binary check
        if self.use_trend_filter and self.use_price_above_ema:
            if self.ema_atr_tolerance > 0:
                h15  = m15_df["high"].to_numpy(dtype=float)
                l15  = m15_df["low"].to_numpy(dtype=float)
                pc15 = np.concatenate([[_close_m15[0]], _close_m15[:-1]])
                tr15 = np.maximum(h15 - l15, np.maximum(
                       np.abs(h15 - pc15), np.abs(l15 - pc15)))
                _atr15 = pd.Series(tr15).rolling(14, min_periods=1).mean().to_numpy(dtype=float)
                _tol15 = _atr15 * self.ema_atr_tolerance
                _ema_long_ok  = (_close_m15 >= _ema50_vals - _tol15)[m1_to_m15]
                _ema_short_ok = (_close_m15 <= _ema50_vals + _tol15)[m1_to_m15]
            else:
                _ema_long_ok  = _above_ema
                _ema_short_ok = ~_above_ema
        else:
            _ema_long_ok  = None
            _ema_short_ok = None

        # H4 trend filter (optional) — resample M15 to 4h, align to M1
        if self.use_h4_trend_filter:
            h4_df = m15_df.resample("4h", closed="left", label="left").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last"}
            ).dropna()
            h4_ema50   = h4_df["close"].ewm(span=50, adjust=False).mean()
            _h4_ema_v  = h4_ema50.to_numpy(dtype=float)
            _lb_h4     = self.h4_trend_slope_lb
            _h4_ema_p  = np.empty_like(_h4_ema_v)
            _h4_ema_p[:_lb_h4] = _h4_ema_v[0]
            _h4_ema_p[_lb_h4:] = _h4_ema_v[:-_lb_h4]
            _h4_bull_h4 = _h4_ema_v > _h4_ema_p
            _h4_bull_h4[:_lb_h4] = False
            m1_to_h4    = np.searchsorted(h4_df.index.values, _m1_ts_vals, side="right") - 1
            m1_to_h4    = np.clip(m1_to_h4, 0, len(h4_df) - 1)
            _h4_trend_bull: np.ndarray | None = _h4_bull_h4[m1_to_h4]
        else:
            _h4_trend_bull = None

        # RSI filter (optional) — vectorised Wilder EMA on M1 closes
        if self.use_rsi_filter:
            delta    = np.diff(_m1_closes, prepend=_m1_closes[0])
            gains    = np.maximum(delta, 0.0)
            losses   = np.maximum(-delta, 0.0)
            _rsi_alpha = 1.0 / self.rsi_period
            _g_ema   = pd.Series(gains).ewm(alpha=_rsi_alpha, adjust=False).mean().to_numpy()
            _l_ema   = pd.Series(losses).ewm(alpha=_rsi_alpha, adjust=False).mean().to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                _rs  = np.where(_l_ema > 0, _g_ema / _l_ema, 100.0)
            _m1_rsi: np.ndarray | None = 100.0 - 100.0 / (1.0 + _rs)
        else:
            _m1_rsi = None

        # ── Wyckoff cache ─────────────────────────────────────────────────────
        _wy_cache: dict[tuple, object] = {}

        # Precompute date keys to avoid strftime overhead in the hot loop
        _m1_date_keys = np.array(m1_df.index.date)

        # ── PASS 2: iterate M1 bars — O(n_bars) with O(1) zone lookups ───────
        for i in range(n_m1):
            # Session filter
            if self.use_session_filter and not (
                self.session_start_utc <= _m1_hours[i] < self.session_end_utc
            ):
                continue

            date_key = _m1_date_keys[i]

            if self.max_signals_per_day > 0 and _daily_count.get(date_key, 0) >= self.max_signals_per_day:
                continue

            lo = _m1_lows[i]; hi = _m1_highs[i]

            # ── Zone score lookup (O(n_zones) but pure numpy) ─────────────────
            # Zone is "active" at bar i if: active_from[j] <= i < mit_idx[j]
            active = (_zone_active_from <= i) & (i < _mit_idx)
            if not active.any():
                continue

            # Freshness at bar i:
            #   _init_fresh  if zone not yet touched  (i < first_touch_idx)
            #   1.0          if touched but not mitigated
            # _first_touch_idx=0 covers pre-existing "already touched" state
            fresh = np.where(i < _first_touch_idx, _init_fresh, 1.0)

            cur_scores = _bnf + fresh

            # Price in zone body
            in_zone = active & (lo <= _zt) & (hi >= _zb)

            # Zone score filter (asymmetric)
            score_ok = (
                (_dem & (cur_scores >= self.min_zone_score_long))
                | (~_dem & (cur_scores >= self.min_zone_score_short))
            )
            candidates = in_zone & score_ok
            if not candidates.any():
                continue

            # Trend filter
            if self.use_trend_filter:
                bullish = bool(_trend_bull[i])
                if bullish:
                    candidates &= _dem
                else:
                    candidates &= ~_dem
                if _ema_long_ok is not None:
                    if not (bool(_ema_long_ok[i]) if bullish else bool(_ema_short_ok[i])):
                        continue
            if not candidates.any():
                continue

            if _adx_ok_arr is not None and not _adx_ok_arr[i]:
                continue

            # Resolve timestamp once (after all cheap filters pass)
            ts = _m1_index[i]

            # Iterate only over the few candidate zone indices
            for j in int_iter(candidates):
                zone = _zones_sorted[j]
                is_demand = zone.side == PivotSide.DEMAND

                # H4 trend must agree with zone direction
                if _h4_trend_bull is not None:
                    if bool(_h4_trend_bull[i]) != is_demand:
                        continue

                # RSI filter: longs need oversold M1 RSI, shorts need overbought
                if _m1_rsi is not None:
                    rsi_val = float(_m1_rsi[i])
                    if is_demand and rsi_val >= self.rsi_oversold:
                        continue
                    if not is_demand and rsi_val <= self.rsi_overbought:
                        continue

                # Per-zone cooldown
                z_key = (zone.formed_at, zone.side, zone.zone_bottom)
                if i - _last_signal.get(z_key, -(self.signal_cooldown + 1)) < self.signal_cooldown:
                    continue
                if self.first_signal_per_zone and z_key in _last_signal:
                    continue

                # Wyckoff confirmation — cached per (bar_index, direction)
                wy_key = (i, zone.side)
                if wy_key not in _wy_cache:
                    w = self._wyckoff.detect_fast(
                        _m1_highs, _m1_lows, _m1_closes, _m1_opens,
                        _m1_index, zone.side, end_idx=i + 1,
                    )
                    # Cache None/mss_bar mismatch as False; valid patterns stored raw
                    # (threshold check done below so asymmetric long/short scores work)
                    _wy_cache[wy_key] = w if (w is not None and w.mss_bar == i) else False
                wyckoff = _wy_cache[wy_key]
                if wyckoff is False:
                    continue
                min_wy = (self.min_wyckoff_score_long if zone.side == PivotSide.DEMAND
                          else self.min_wyckoff_score_short)
                if wyckoff.score < min_wy:
                    continue

                # Sync zone score to current state before building signal
                cur_fresh = float(fresh[j])
                if zone.score.fresh != cur_fresh:
                    zone.score = ZoneScore(
                        bos=zone.score.bos, impulse=zone.score.impulse,
                        time=zone.score.time, fresh=cur_fresh, sweep=zone.score.sweep,
                    )

                sig = self._build_signal(i, ts, zone, wyckoff, htf_fibs)
                if sig is None:
                    continue
                if sig.composite_score < self.min_composite_score:
                    continue
                if self.min_score_product > 0 and sig.zone_score * sig.wyckoff_score < self.min_score_product:
                    continue

                _last_signal[z_key] = i
                _daily_count[date_key] = _daily_count.get(date_key, 0) + 1
                signals.append(sig)

                if self.max_signals_per_day > 0 and _daily_count[date_key] >= self.max_signals_per_day:
                    break

        return signals

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_signal(
        self,
        bar_index: int,
        ts:        pd.Timestamp,
        zone:      SDZone,
        wyckoff:   WyckoffPattern,
        htf_fibs:  Optional[FibLevels],
    ) -> Optional[SDSignal]:
        entry = wyckoff.mss_close
        sl    = zone.wick_extreme
        risk  = abs(entry - sl)

        if risk < 1e-8:
            return None
        if self.min_sl_pips > 0 and risk < self.min_sl_pips * self.pip_size:
            return None

        if zone.side == PivotSide.DEMAND:
            direction = "long"
            tp = entry + self.risk_reward * risk
        else:
            direction = "short"
            tp = entry - self.risk_reward * risk

        # Optional Fibonacci OTE confluence
        fib_bonus: float = 0.0
        fib_conf: Optional[FibConfluence] = None
        if htf_fibs is not None:
            fib_conf  = zone_fib_confluence(zone, htf_fibs)
            fib_bonus = fib_conf.score_bonus

        composite = round(zone.score.total + fib_bonus, 2)

        return SDSignal(
            direction        = direction,
            entry_price      = round(entry, 6),
            stop_loss        = round(sl, 6),
            take_profit      = round(tp, 6),
            risk_reward      = self.risk_reward,
            zone_score       = round(zone.score.total, 2),
            wyckoff_score    = round(wyckoff.score, 2),
            fib_bonus        = round(fib_bonus, 2),
            composite_score  = composite,
            zone             = zone,
            wyckoff          = wyckoff,
            fib              = fib_conf,
            formed_at        = ts,
            bar_index        = bar_index,
        )
