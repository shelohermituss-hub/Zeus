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
    2. Align H4 EMA50 slope → directional bias.
    3. When price pulls back INTO an active FVG in the direction of H4 bias:
       entry = M15 close  |  SL = just beyond the far edge of the FVG
       TP   = entry ± risk_reward × |entry − SL|
    4. Each FVG is used at most once.

Key differences from S&D + Wyckoff:
    - No pivot-based zone detection.
    - No Wyckoff accumulation / MSS pattern requirement.
    - FVG = momentum imbalance (not supply/demand zone).
    - H4 EMA slope for bias instead of M15 slope.
    - Higher signal frequency on trending markets.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from zeus.strategy.smc.fvg import detect_fvg
from zeus.strategy.smc.pivot import BEARISH, BULLISH


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
class FVGSignal:
    """
    Trading signal produced by FVGRetestStrategy.

    Interface-compatible with sd_simulation.simulate_trade():
        direction, entry_price, stop_loss, take_profit, risk_reward,
        formed_at, bar_index are accessed by the simulation engine.
    """
    direction:    str            # "long" or "short"
    entry_price:  float          # M15 bar close at signal time
    stop_loss:    float          # just beyond the far FVG edge
    take_profit:  float          # entry ± risk_reward × |entry − sl|
    risk_reward:  float
    formed_at:    pd.Timestamp   # M15 bar timestamp
    bar_index:    int            # M15 bar index (sim enters at bar_index + 1)
    fvg_top:      float
    fvg_bottom:   float
    fvg_bar:      int            # bar index where the FVG was detected


# ── Strategy ─────────────────────────────────────────────────────────────────

class FVGRetestStrategy:
    """
    M15 FVG retest entries filtered by H4 EMA50 trend direction.

    Parameters
    ----------
    risk_reward        : R:R target (default 2.0)
    min_fvg_atr_mult   : minimum FVG size as a multiple of ATR(atr_period);
                         smaller gaps are filtered out (default 0.10)
    max_fvg_age_bars   : discard FVGs older than this many M15 bars (default 96 = 24 h)
    use_h4_trend       : require H4 EMA50 slope to agree with FVG direction (default True)
    h4_ema_span        : H4 EMA span (default 50)
    h4_slope_lb        : H4 bars for EMA slope comparison (default 3)
    signal_cooldown    : minimum M15 bars between two signals (default 8 = 2 h)
    max_signals_per_day: hard daily cap on signals, 0 = unlimited (default 4)
    use_session_filter : only trade session_start_utc..session_end_utc (default True)
    session_start_utc  : first UTC hour of active session, inclusive (default 7)
    session_end_utc    : last  UTC hour of active session, exclusive (default 21)
    sl_buffer_atr_mult : SL placed this many ATR multiples beyond the FVG edge (default 0.10)
    long_only          : if True skip all bearish FVG setups (default False)
    atr_period         : ATR smoothing period in M15 bars (default 14)
    """

    def __init__(
        self,
        risk_reward:          float = 2.0,
        min_fvg_atr_mult:     float = 0.10,
        max_fvg_age_bars:     int   = 96,
        use_h4_trend:         bool  = True,
        h4_ema_span:          int   = 50,
        h4_slope_lb:          int   = 3,
        signal_cooldown:      int   = 8,
        max_signals_per_day:  int   = 4,
        use_session_filter:   bool  = True,
        session_start_utc:    int   = 7,
        session_end_utc:      int   = 21,
        sl_buffer_atr_mult:   float = 0.10,
        long_only:            bool  = False,
        atr_period:           int   = 14,
    ) -> None:
        self.risk_reward          = risk_reward
        self.min_fvg_atr_mult     = min_fvg_atr_mult
        self.max_fvg_age_bars     = max_fvg_age_bars
        self.use_h4_trend         = use_h4_trend
        self.h4_ema_span          = h4_ema_span
        self.h4_slope_lb          = h4_slope_lb
        self.signal_cooldown      = signal_cooldown
        self.max_signals_per_day  = max_signals_per_day
        self.use_session_filter   = use_session_filter
        self.session_start_utc    = session_start_utc
        self.session_end_utc      = session_end_utc
        self.sl_buffer_atr_mult   = sl_buffer_atr_mult
        self.long_only            = long_only
        self.atr_period           = atr_period

    # ── Public ───────────────────────────────────────────────────────────────

    def run(self, m15_df: pd.DataFrame) -> list[FVGSignal]:
        """
        Detect FVG retest signals over a full M15 OHLCV DataFrame.

        Returns a chronological list of FVGSignal objects.
        The bar_index field is a M15 bar index; pass m15_df as the second
        argument to sd_simulation.simulate_all() for a M15-level simulation.
        """
        # ── FVG detection ──────────────────────────────────────────────────
        fvgs = detect_fvg(
            m15_df["high"], m15_df["low"], m15_df["close"], m15_df["open"]
        )
        if not fvgs:
            return []

        # ── ATR (vectorised, M15 bars) ────────────────────────────────────
        _atr = _atr_series(m15_df, self.atr_period).to_numpy(dtype=float)

        # ── H4 EMA50 trend aligned to M15 bars ───────────────────────────
        _h4_trend: np.ndarray | None = None
        if self.use_h4_trend:
            h4_df = m15_df.resample("4h", closed="left", label="left").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last"}
            ).dropna()
            h4_ema    = h4_df["close"].ewm(span=self.h4_ema_span, adjust=False).mean()
            _h4_v     = h4_ema.to_numpy(dtype=float)
            _lb       = self.h4_slope_lb
            _h4_p     = np.empty_like(_h4_v)
            _h4_p[:_lb] = _h4_v[0]
            _h4_p[_lb:] = _h4_v[:-_lb]
            _h4_bull  = _h4_v > _h4_p
            _h4_bull[:_lb] = False
            m15_to_h4 = np.searchsorted(
                h4_df.index.values, m15_df.index.values, side="right"
            ) - 1
            m15_to_h4 = np.clip(m15_to_h4, 0, len(h4_df) - 1)
            _h4_trend = _h4_bull[m15_to_h4]

        # ── Numpy price arrays ────────────────────────────────────────────
        _highs  = m15_df["high"].to_numpy(dtype=float)
        _lows   = m15_df["low"].to_numpy(dtype=float)
        _closes = m15_df["close"].to_numpy(dtype=float)
        _hours  = m15_df.index.hour
        _dates  = np.array(m15_df.index.date)
        n = len(m15_df)

        # ── Main loop ─────────────────────────────────────────────────────
        signals: list[FVGSignal] = []
        _entered: set[int] = set()        # FVG bar_index already used
        _last_sig_bar: int  = -999
        _daily_count: dict  = {}

        for i in range(3, n):
            # Session filter
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
            if i - _last_sig_bar < self.signal_cooldown:
                continue

            h4_bull = bool(_h4_trend[i]) if _h4_trend is not None else True
            lo_i    = float(_lows[i])
            hi_i    = float(_highs[i])
            cl_i    = float(_closes[i])
            atr_i   = max(float(_atr[i]), 1e-6)

            for fvg in fvgs:
                # Causal filter: FVG must be formed before bar i
                if fvg.bar_index >= i:
                    continue
                # Skip mitigated FVGs
                if fvg.mitigated_at != -1 and fvg.mitigated_at <= i:
                    continue
                # One entry per FVG
                if fvg.bar_index in _entered:
                    continue
                # Age filter
                if i - fvg.bar_index > self.max_fvg_age_bars:
                    continue
                # Size filter
                fvg_size = fvg.top - fvg.bottom
                if fvg_size < self.min_fvg_atr_mult * atr_i:
                    continue

                if fvg.direction == BULLISH:
                    # Skip if H4 is bearish (trend misaligned)
                    if _h4_trend is not None and not h4_bull:
                        continue
                    # Retest: price pulls back into the gap from above
                    if lo_i > fvg.top:
                        continue
                    sl_buf    = max(self.sl_buffer_atr_mult * atr_i, fvg_size * 0.10)
                    sl        = fvg.bottom - sl_buf
                    risk_dist = cl_i - sl
                    if risk_dist <= 0:
                        continue
                    tp        = cl_i + self.risk_reward * risk_dist
                    direction = "long"

                else:  # BEARISH
                    if self.long_only:
                        continue
                    # Skip if H4 is bullish (trend misaligned)
                    if _h4_trend is not None and h4_bull:
                        continue
                    # Retest: price rallies back into the gap from below
                    if hi_i < fvg.bottom:
                        continue
                    sl_buf    = max(self.sl_buffer_atr_mult * atr_i, fvg_size * 0.10)
                    sl        = fvg.top + sl_buf
                    risk_dist = sl - cl_i
                    if risk_dist <= 0:
                        continue
                    tp        = cl_i - self.risk_reward * risk_dist
                    direction = "short"

                _entered.add(fvg.bar_index)
                _last_sig_bar = i
                _daily_count[date_key] = _daily_count.get(date_key, 0) + 1

                signals.append(FVGSignal(
                    direction   = direction,
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
