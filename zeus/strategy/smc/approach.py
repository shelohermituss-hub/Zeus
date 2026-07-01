"""
Approach quality filter — distinguishes controlled pullbacks from impulsive arrivals.

A zone is only tradeable when price *drifts* into it gradually (pullback).
If price *shoots* into the zone with large candles the zone is likely to be
blown through rather than respected — institutional interest has already been
consumed by the momentum move.

Two independent checks run on the N bars immediately before zone contact:

1. Momentum ratio
   |close_now − close_N_bars_ago| / (ATR × N)
   → if > max_momentum the approach is impulsive → REJECT

2. Spike candle
   max(|close − open|) over lookback / ATR
   → if > max_body_atr a single candle is outsized → REJECT

Typical XAUUSD 1 M calibration (defaults)
------------------------------------------
ATR ≈ 2–3 pips/bar, lookback = 5 bars.
Price moving 7–8 pips in 5 bars hits momentum ≈ 0.6 → borderline.
Price moving 4–5 pips in 5 bars → momentum ≈ 0.35 → clean.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_approach_quality(
    df:              pd.DataFrame,
    bar_index:       int,
    lookback:        int   = 5,
    max_momentum:    float = 0.6,
    max_body_atr:    float = 1.5,
) -> tuple[bool, str, float]:
    """
    Assess whether price arrived at *bar_index* cleanly or impulsively.

    Args:
        df:           LTF OHLCV DataFrame (must have open/high/low/close columns).
        bar_index:    Index of the bar that just touched the zone.
        lookback:     Number of bars to look back for the approach assessment.
        max_momentum: Reject if (total_move / ATR / lookback) exceeds this.
        max_body_atr: Reject if any single bar's body exceeds ATR × this factor.

    Returns:
        (is_clean, reason, momentum_ratio)
        is_clean = False → setup should be rejected.
    """
    if bar_index < lookback:
        return True, "insufficient lookback data", 0.0

    start  = bar_index - lookback
    window = df.iloc[start : bar_index + 1]

    highs  = window["high"].to_numpy(dtype=float)
    lows   = window["low"].to_numpy(dtype=float)
    closes = window["close"].to_numpy(dtype=float)
    opens  = window["open"].to_numpy(dtype=float)

    # ── ATR over the window ────────────────────────────────────────────────
    hl       = highs - lows                          # shape: lookback + 1
    prev_c   = closes[:-1]                           # shape: lookback
    hc       = np.abs(highs[1:] - prev_c)
    lc       = np.abs(lows[1:]  - prev_c)
    tr_rest  = np.maximum(hl[1:], np.maximum(hc, lc))
    tr_full  = np.empty(len(highs))
    tr_full[0] = hl[0]
    tr_full[1:] = tr_rest
    atr = float(tr_full.mean())

    if atr < 1e-10:
        return True, "ATR ≈ 0 — cannot assess approach", 0.0

    # ── Check 1: momentum ratio ────────────────────────────────────────────
    price_move = abs(closes[-1] - closes[0])
    momentum   = price_move / (atr * lookback)

    if momentum > max_momentum:
        return (
            False,
            f"impulsive approach: momentum={momentum:.2f} > max={max_momentum:.2f}",
            momentum,
        )

    # ── Check 2: spike candle ─────────────────────────────────────────────
    bodies     = np.abs(closes - opens)
    max_body   = float(bodies.max())
    body_ratio = max_body / atr

    if body_ratio > max_body_atr:
        return (
            False,
            f"spike candle: body/ATR={body_ratio:.2f} > max={max_body_atr:.2f}",
            momentum,
        )

    return (
        True,
        f"clean approach: momentum={momentum:.2f} body/ATR={body_ratio:.2f}",
        momentum,
    )
