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

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .fibonacci import FibConfluence, FibLevels, zone_fib_confluence
from .pivot_candle import PivotSide
from .wyckoff import WyckoffDetector, WyckoffPattern
from .zone_detector import SDZone, ZoneDetector


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
        min_zone_score_long:  Optional[float] = None,
        min_zone_score_short: Optional[float] = None,
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

    # ── Public ────────────────────────────────────────────────────────────────

    def run(
        self,
        m15_df:   pd.DataFrame,
        m1_df:    pd.DataFrame,
        htf_fibs: Optional[FibLevels] = None,
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

        # ── Performance: incremental zone accumulation ────────────────────────
        # Sort zones by formation time so we can accumulate with a single pointer
        # instead of scanning all_zones on every M1 bar (O(n_zones) → O(1) amort.)
        _zones_sorted = sorted(all_zones, key=lambda z: z.formed_at)
        _zone_ptr     = 0
        active_zones: list = []

        # ── Performance: per-bar Wyckoff cache ───────────────────────────────
        # WyckoffDetector.detect() result depends only on (bar_index, direction).
        # Cache it so every zone with the same direction on the same bar reuses
        # the result without re-scanning 200 M1 bars.
        # Sentinel value False = "computed but no valid pattern this bar/direction"
        _wy_cache: dict[tuple, object] = {}

        # ── Performance: per-bar trend cache ─────────────────────────────────
        _trend_cache: dict[pd.Timestamp, tuple] = {}

        for i in range(len(m1_df)):
            ts  = m1_df.index[i]
            bar = m1_df.iloc[i]

            # Accumulate zones that have formed by this M1 bar
            while _zone_ptr < len(_zones_sorted) and _zones_sorted[_zone_ptr].formed_at <= ts:
                active_zones.append(_zones_sorted[_zone_ptr])
                _zone_ptr += 1

            # Update mitigation / freshness state for all active zones (always runs)
            self._zones.update_zones(active_zones, bar, ts)

            # Session filter: skip signal scanning outside active trading hours
            if self.use_session_filter:
                if not (self.session_start_utc <= ts.hour < self.session_end_utc):
                    continue

            date_key = ts.strftime("%Y-%m-%d")

            # Trend & EMA position for this bar (computed once, shared across zones)
            if self.use_trend_filter and ts not in _trend_cache:
                _trend_cache[ts] = (_trend_is_bullish(ts), _price_above_ema(ts))

            for zone in active_zones:
                # Global daily signal cap — once hit, skip remaining zones for this bar
                if self.max_signals_per_day > 0 and _daily_count.get(date_key, 0) >= self.max_signals_per_day:
                    break

                if zone.is_mitigated:
                    continue
                # Asymmetric zone score: demand and supply can have different floors
                _min_zsc = self.min_zone_score_long if zone.side == PivotSide.DEMAND else self.min_zone_score_short
                if zone.score.total < _min_zsc:
                    continue
                if not zone.price_in_zone(float(bar["low"]), float(bar["high"])):
                    continue

                # Trend alignment: demand only in uptrend, supply only in downtrend
                if self.use_trend_filter:
                    bullish, above = _trend_cache[ts]
                    if zone.side == PivotSide.DEMAND and not bullish:
                        continue
                    if zone.side == PivotSide.SUPPLY and bullish:
                        continue
                    if self.use_price_above_ema:
                        if zone.side == PivotSide.DEMAND and not above:
                            continue
                        if zone.side == PivotSide.SUPPLY and above:
                            continue

                # ADX regime filter: skip ranging/choppy markets
                if self.use_adx_filter and not _adx_ok(ts):
                    continue

                # Per-zone cooldown: avoid rapid re-entries on the same zone
                z_key = (zone.formed_at, zone.side, zone.zone_bottom)
                if i - _last_signal.get(z_key, -(self.signal_cooldown + 1)) < self.signal_cooldown:
                    continue

                # First-signal-only: once a zone has emitted a signal, skip it forever
                if self.first_signal_per_zone and z_key in _last_signal:
                    continue

                # Wyckoff confirmation — cached per (bar_index, direction)
                wy_key = (i, zone.side)
                if wy_key not in _wy_cache:
                    w = self._wyckoff.detect(m1_df, zone.side, end_idx=i + 1)
                    if (w is None or w.score < self.min_wyckoff_score or w.mss_bar != i):
                        _wy_cache[wy_key] = False  # sentinel: no valid pattern
                    else:
                        _wy_cache[wy_key] = w
                wyckoff = _wy_cache[wy_key]
                if wyckoff is False:
                    continue

                sig = self._build_signal(i, ts, zone, wyckoff, htf_fibs)
                if sig is None:
                    continue
                if sig.composite_score < self.min_composite_score:
                    continue

                _last_signal[z_key] = i
                _daily_count[date_key] = _daily_count.get(date_key, 0) + 1
                signals.append(sig)

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
