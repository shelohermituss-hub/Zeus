"""
Shared strategy utilities.

SignalCooldown
--------------
  Prevents back-to-back signals by enforcing a minimum bar gap.
  Used identically in every strategy — extracted to avoid copy-paste drift.

_atr_series
-----------
  Common ATR calculation (EWM-smoothed true range).
  Extracted from the 4+ strategy files that duplicated it.
"""
from __future__ import annotations

import pandas as pd


# ── ATR ───────────────────────────────────────────────────────────────────────

def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """EWM-smoothed ATR over a standard OHLCV DataFrame."""
    h, lo, c = df["high"], df["low"], df["close"]
    tr = pd.concat([
        h - lo,
        (h - c.shift(1)).abs(),
        (lo - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ── Cooldown ──────────────────────────────────────────────────────────────────

class SignalCooldown:
    """
    Enforce a minimum gap (in bars) between consecutive signals.

    Usage::

        cd = SignalCooldown(cooldown_bars=12)
        for i in range(n):
            if not cd.can_signal(i):
                continue
            # ... signal logic ...
            cd.mark(i)
    """

    def __init__(self, cooldown_bars: int) -> None:
        if cooldown_bars < 0:
            raise ValueError("cooldown_bars must be >= 0")
        self._cooldown = cooldown_bars
        self._last_bar: int = -(cooldown_bars + 1)

    def can_signal(self, bar_index: int) -> bool:
        """Return True if enough bars have passed since the last signal."""
        return bar_index - self._last_bar >= self._cooldown

    def mark(self, bar_index: int) -> None:
        """Record that a signal was emitted at bar_index."""
        self._last_bar = bar_index

    def reset(self) -> None:
        """Reset cooldown state (e.g. at the start of each backtest window)."""
        self._last_bar = -(self._cooldown + 1)
