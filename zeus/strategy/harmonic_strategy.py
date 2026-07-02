"""
Harmonic Pattern Strategy.

Entry logic
-----------
  Long  (bullish pattern): bar's low touches the PRZ (lo ≤ prz_high) and
                            the bar closes above prz_low (cl ≥ prz_low).
                            Optional: close must be bullish (close > open).
  Short (bearish pattern): bar's high touches PRZ and close ≤ prz_high.
                            (long_only=True skips bearish by default.)

Stop-loss
---------
  prz_low  − sl_buffer × ATR  (bullish)
  prz_high + sl_buffer × ATR  (bearish)

Take-profit
-----------
  entry ± risk_reward × risk_distance

Pattern invalidation
--------------------
  A pattern is cancelled if price closes beyond D before confirmation:
    bullish: close < prz_low  (price breaks through the PRZ downward)
    bearish: close > prz_high (price breaks through upward)
  A pattern expires after max_prz_age_bars bars from confirmed_at.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from zeus.strategy.smc.harmonic import HarmonicPattern, detect_harmonics


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
class HarmonicSignal:
    """
    Trading signal produced by HarmonicStrategy.

    Interface-compatible with sd_simulation.simulate_trade() and compute_metrics():
    direction, entry_price, stop_loss, take_profit, risk_reward,
    formed_at, bar_index, zone_score are accessed by the simulation engine.
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
    d_price:      float
    zone_score:   float = 0.0


# ── Strategy ──────────────────────────────────────────────────────────────────

class HarmonicStrategy:
    """
    M15 harmonic pattern strategy with PRZ confirmation entry.

    Parameters
    ----------
    pivot_size          : bars for pivot confirmation (default 5)
    precision           : AB tolerance for Gartley/Butterfly (default 0.03)
    long_only           : only bullish patterns (default True)
    use_h4_trend        : require H4 EMA50 bullish slope for long entries
    h4_ema_span         : H4 EMA span (default 50)
    h4_slope_lb         : bars for H4 slope comparison (default 3)
    risk_reward         : R:R target (default 2.0)
    sl_buffer_atr       : SL buffer in ATR beyond PRZ edge (default 0.20)
    max_prz_age_bars    : cancel pattern if PRZ not hit within N bars (default 200)
    signal_cooldown     : min M15 bars between signals (default 12 = 3h)
    atr_period          : ATR smoothing period (default 14)
    require_bullish_bar : entry bar must close bullish for longs (default True)
    """

    def __init__(
        self,
        pivot_size:          int   = 5,
        precision:           float = 0.03,
        long_only:           bool  = True,
        use_h4_trend:        bool  = True,
        h4_ema_span:         int   = 50,
        h4_slope_lb:         int   = 3,
        risk_reward:         float = 2.0,
        sl_buffer_atr:       float = 0.20,
        max_prz_age_bars:    int   = 200,
        signal_cooldown:     int   = 12,
        atr_period:          int   = 14,
        require_bullish_bar: bool  = True,
    ) -> None:
        self.pivot_size          = pivot_size
        self.precision           = precision
        self.long_only           = long_only
        self.use_h4_trend        = use_h4_trend
        self.h4_ema_span         = h4_ema_span
        self.h4_slope_lb         = h4_slope_lb
        self.risk_reward         = risk_reward
        self.sl_buffer_atr       = sl_buffer_atr
        self.max_prz_age_bars    = max_prz_age_bars
        self.signal_cooldown     = signal_cooldown
        self.atr_period          = atr_period
        self.require_bullish_bar = require_bullish_bar

    def run(self, m15_df: pd.DataFrame) -> list[HarmonicSignal]:
        """
        Run strategy over a full M15 OHLCV DataFrame.

        Returns a chronologically ordered list of HarmonicSignal objects.
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

        # ── Detect all valid patterns ──────────────────────────────────────
        patterns = detect_harmonics(
            m15_df,
            pivot_size = self.pivot_size,
            precision  = self.precision,
            long_only  = self.long_only,
        )
        if not patterns:
            return []

        # ── Per-pattern invalidation tracking ─────────────────────────────
        # pattern key → bar where pattern was broken (close through PRZ)
        invalidated: set[tuple] = set()
        triggered:   set[tuple] = set()

        def _key(p: HarmonicPattern) -> tuple:
            return (p.pattern_type, p.direction, p.x_bar, p.c_bar)

        # ── Main loop ─────────────────────────────────────────────────────
        signals:      list[HarmonicSignal] = []
        _last_sig_bar: int = -999

        for i in range(4, n):
            cl_i  = float(_closes[i])
            op_i  = float(_opens[i])
            lo_i  = float(_lows[i])
            hi_i  = float(_highs[i])
            atr_i = max(float(_atr[i]), 1e-6)

            h4_bull = bool(_h4_trend[i]) if _h4_trend is not None else True

            for pat in patterns:
                k = _key(pat)
                if k in triggered or k in invalidated:
                    continue
                if i < pat.confirmed_at:
                    continue
                if i - pat.confirmed_at > self.max_prz_age_bars:
                    continue

                if pat.direction == "bullish":
                    # Invalidate if close breaks below PRZ
                    if cl_i < pat.prz_low:
                        invalidated.add(k)
                        continue
                    if self.use_h4_trend and not h4_bull:
                        continue
                    # PRZ touch: low dips into PRZ but close stays above
                    if lo_i > pat.prz_high:
                        continue
                    if cl_i < pat.prz_low:
                        continue
                    # Confirmation: bullish bar
                    if self.require_bullish_bar and cl_i <= op_i:
                        continue

                    if i - _last_sig_bar < self.signal_cooldown:
                        continue

                    sl_buf    = self.sl_buffer_atr * atr_i
                    sl        = pat.prz_low - sl_buf
                    risk_dist = cl_i - sl
                    if risk_dist <= 0:
                        continue
                    tp = cl_i + self.risk_reward * risk_dist

                    triggered.add(k)
                    _last_sig_bar = i

                    signals.append(HarmonicSignal(
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
                        d_price      = round(pat.d_price, 2),
                    ))
                    break  # one signal per bar

        return signals
