"""
Momentum/breakout scalp strategy — M1-native signal generator.

Detects a directional momentum burst over a short M1 window (a handful of
bars, mostly closing the same direction, range expanded vs ATR, clean
bodies rather than wicky chop) and signals a continuation entry in that
direction. Built for high-frequency scalping: SL sits just beyond the
burst's own extreme, target is a small fixed R:R — no runner, no
multi-day holds.

Deliberately independent from the Wyckoff/S&D signal family: no M15
zones, no accumulation/manipulation/MSS structure. Pure M1 momentum read,
so it can fire far more often than the S&D signals (which need a rare
zone + Wyckoff confluence).

Signal shape matches the `Tradeable` protocol used by
zeus/backtest/sd_simulation.py (direction, bar_index, stop_loss,
risk_reward, formed_at, zone_score) so it plugs directly into the
existing, already-validated simulate_all()/PropFirmGuard machinery
without a parallel simulation engine.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_EPS = 1e-8


@dataclass(frozen=True)
class MomentumScalpSignal:
    """Momentum scalp signal — satisfies the Tradeable protocol."""
    direction:    str          # "long" | "short"
    entry_price:  float        # close of the burst's last bar (fill = next bar open)
    stop_loss:    float        # beyond the burst extreme + ATR buffer
    take_profit:  float        # entry ± risk_reward × risk
    risk_reward:  float
    zone_score:   float        # 0-10 burst quality proxy (range/ATR + body cleanliness)
    wyckoff_score: float = 0.0 # unused, kept for structural parity with SDSignal
    composite_score: float = 0.0
    formed_at:    pd.Timestamp = None
    bar_index:    int = 0

    def __post_init__(self):
        object.__setattr__(self, "composite_score", self.zone_score)

    @property
    def risk(self) -> float:
        return abs(self.entry_price - self.stop_loss)


class MomentumScalpStrategy:
    """
    Scan M1 OHLCV for directional momentum bursts and emit scalp signals.

    Parameters
    ----------
    burst_bars           : number of consecutive M1 bars forming the burst window
    atr_period            : ATR smoothing window (M1 bars)
    min_burst_atr_mult     : window range (high-low) must be >= this many ATR
    min_body_ratio         : average body/range ratio across the window (clean
                             momentum, not wicky chop)
    min_same_direction_frac: fraction of bars in the window that must close
                             in the burst direction. Default 1.0 (unanimous) —
                             with a short window (few bars), any threshold below
                             unanimous lets a strictly-alternating chop sequence
                             slip through a sub-window with an accidental 2:1
                             majority; requiring all bars to agree is the
                             deterministic, unambiguous reading of "burst".
    sl_atr_mult            : SL = burst extreme ± this many ATR (buffer beyond
                             the raw wick so normal noise doesn't clip it)
    risk_reward            : fixed R:R target (small — this is a scalp, not a runner)
    min_sl_pips / pip_size : reject signals whose SL distance is below this floor
    cooldown_bars          : minimum M1 bars between two signals
    session_start_utc/session_end_utc : active session window (UTC hour, end exclusive)
    max_signals_per_day    : 0 = unlimited
    min_zone_score         : minimum burst quality score (0-10) to accept a signal
    """

    def __init__(
        self,
        burst_bars:              int   = 3,
        atr_period:              int   = 14,
        min_burst_atr_mult:      float = 1.2,
        min_body_ratio:          float = 0.50,
        min_same_direction_frac: float = 1.0,
        sl_atr_mult:             float = 0.5,
        risk_reward:             float = 1.5,
        min_sl_pips:             float = 0.0,
        pip_size:                float = 0.0001,
        cooldown_bars:           int   = 15,
        session_start_utc:       int   = 7,
        session_end_utc:         int   = 21,
        max_signals_per_day:     int   = 20,
        min_zone_score:          float = 5.0,
    ) -> None:
        self.burst_bars              = burst_bars
        self.atr_period              = atr_period
        self.min_burst_atr_mult      = min_burst_atr_mult
        self.min_body_ratio          = min_body_ratio
        self.min_same_direction_frac = min_same_direction_frac
        self.sl_atr_mult             = sl_atr_mult
        self.risk_reward             = risk_reward
        self.min_sl_pips             = min_sl_pips
        self.pip_size                = pip_size
        self.cooldown_bars           = cooldown_bars
        self.session_start_utc       = session_start_utc
        self.session_end_utc         = session_end_utc
        self.max_signals_per_day     = max_signals_per_day
        self.min_zone_score          = min_zone_score

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self, m1_df: pd.DataFrame) -> list[MomentumScalpSignal]:
        """Scan the full M1 DataFrame and return signals in chronological order."""
        n = len(m1_df)
        w = self.burst_bars
        if n < w + 2:
            return []

        opens  = m1_df["open"].to_numpy(dtype=float)
        highs  = m1_df["high"].to_numpy(dtype=float)
        lows   = m1_df["low"].to_numpy(dtype=float)
        closes = m1_df["close"].to_numpy(dtype=float)
        atr    = _compute_atr_m1(highs, lows, closes, self.atr_period)
        hours  = m1_df.index.hour
        dates  = m1_df.index.date

        signals: list[MomentumScalpSignal] = []
        last_signal_bar = -(self.cooldown_bars + 1)
        daily_count: dict = {}

        for i in range(w - 1, n - 1):   # need i+1 for the entry fill bar
            if self.session_start_utc <= self.session_end_utc:
                in_session = self.session_start_utc <= hours[i] < self.session_end_utc
            else:
                in_session = hours[i] >= self.session_start_utc or hours[i] < self.session_end_utc
            if not in_session:
                continue

            date_key = dates[i]
            if self.max_signals_per_day > 0 and daily_count.get(date_key, 0) >= self.max_signals_per_day:
                continue

            if i - last_signal_bar < self.cooldown_bars:
                continue

            sig = self._detect_at(i, opens, highs, lows, closes, atr, m1_df.index)
            if sig is None:
                continue

            signals.append(sig)
            last_signal_bar = i
            daily_count[date_key] = daily_count.get(date_key, 0) + 1

        return signals

    # ── Private ───────────────────────────────────────────────────────────────

    def _detect_at(
        self, i: int,
        opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
        atr: np.ndarray, index: pd.DatetimeIndex,
    ) -> MomentumScalpSignal | None:
        w = self.burst_bars
        start = i - w + 1   # window = [start, i] inclusive, w bars

        atr_i = atr[i]
        if atr_i < _EPS:
            return None   # dead/flat market — fail closed, no signal on noise

        up_closes   = 0
        down_closes = 0
        body_ratio_sum = 0.0
        body_ratio_n   = 0
        for k in range(start, i + 1):
            if closes[k] > opens[k]:
                up_closes += 1
            elif closes[k] < opens[k]:
                down_closes += 1
            # flat/doji bars (close == open) count toward neither direction —
            # they are not directional evidence either way.
            rng = highs[k] - lows[k]
            if rng > _EPS:
                body_ratio_sum += abs(closes[k] - opens[k]) / rng
                body_ratio_n   += 1

        if body_ratio_n == 0:
            return None
        avg_body_ratio = body_ratio_sum / body_ratio_n

        if up_closes / w >= self.min_same_direction_frac:
            direction = "long"
        elif down_closes / w >= self.min_same_direction_frac:
            direction = "short"
        else:
            return None   # choppy or flat — no directional dominance

        window_high = float(np.max(highs[start:i + 1]))
        window_low  = float(np.min(lows[start:i + 1]))
        window_rng  = window_high - window_low
        if window_rng < self.min_burst_atr_mult * atr_i:
            return None   # burst not strong enough vs current volatility

        if avg_body_ratio < self.min_body_ratio:
            return None   # too much wick/chop for a clean momentum read

        entry = closes[i]
        if direction == "long":
            sl = window_low - self.sl_atr_mult * atr_i
        else:
            sl = window_high + self.sl_atr_mult * atr_i

        risk = abs(entry - sl)
        if risk < 1e-8:
            return None
        if self.min_sl_pips > 0 and risk < self.min_sl_pips * self.pip_size:
            return None

        tp = entry + self.risk_reward * risk if direction == "long" else entry - self.risk_reward * risk

        # Quality score 0-10: half from burst strength (range/ATR, capped),
        # half from candle cleanliness (body ratio) — same 0-10 scale
        # convention as the S&D zone/Wyckoff scores elsewhere in the project.
        strength_pts = min(5.0, (window_rng / atr_i) * (5.0 / self.min_burst_atr_mult) / 2.0)
        clean_pts    = min(5.0, avg_body_ratio * 5.0)
        score = round(strength_pts + clean_pts, 2)
        if score < self.min_zone_score:
            return None

        return MomentumScalpSignal(
            direction    = direction,
            entry_price  = round(entry, 6),
            stop_loss    = round(sl, 6),
            take_profit  = round(tp, 6),
            risk_reward  = self.risk_reward,
            zone_score   = score,
            formed_at    = index[i],
            bar_index    = i,
        )


def _compute_atr_m1(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
    """True Range rolling mean, min_periods=1 — same convention as zone_detector._compute_atr."""
    n = len(highs)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)

    atr = np.empty(n)
    window_sum = 0.0
    for i in range(n):
        window_sum += tr[i]
        if i >= period:
            window_sum -= tr[i - period]
        cnt = min(i + 1, period)
        atr[i] = window_sum / cnt
    return atr
