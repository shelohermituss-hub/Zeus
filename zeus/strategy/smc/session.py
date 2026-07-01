"""
Session range detection for SMC liquidity analysis (factor 7).

Smart money uses Asian, London, and New York session highs/lows as key
liquidity targets. Price frequently sweeps these levels during the following
session before reversing into the true directional move.

Session windows (UTC)
---------------------
Asian  : 00:00 – 08:00  (8 h)
London : 08:00 – 16:00  (8 h)
NY     : 13:00 – 21:00  (8 h)   [overlaps London 13:00–16:00]

A SessionRange is actionable only after its window closes. The ``formed_at``
field holds the bar index of the first bar that falls OUTSIDE the session —
the strategy must not use a range whose session is still in progress.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import pandas as pd

from zeus.strategy.smc.pivot import BEARISH, BULLISH


class SessionType(IntEnum):
    ASIAN  = 0
    LONDON = 1
    NY     = 2


# UTC hour ranges [start_hour, end_hour)
_SESSION_BOUNDS: dict[SessionType, tuple[int, int]] = {
    SessionType.ASIAN:  (0,  8),
    SessionType.LONDON: (8, 16),
    SessionType.NY:     (13, 21),
}


@dataclass(frozen=True)
class SessionRange:
    """
    Completed H/L range for one trading session.

    Levels can be used as liquidity targets or confirmation zones.
    """
    session:   SessionType
    high:      float
    low:       float
    high_bar:  int   # bar index where session high occurred
    low_bar:   int   # bar index where session low occurred
    start_bar: int   # first bar index inside the session
    formed_at: int   # first bar index AFTER the session ends (range actionable here)

    @property
    def mid(self) -> float:
        """Session midpoint — often a liquidity magnet."""
        return (self.high + self.low) / 2

    def is_near_high(self, price: float, tolerance_pct: float = 0.003) -> bool:
        """True when price is within tolerance_pct of the session high."""
        if self.high == 0:
            return False
        return abs(price - self.high) / self.high <= tolerance_pct

    def is_near_low(self, price: float, tolerance_pct: float = 0.003) -> bool:
        """True when price is within tolerance_pct of the session low."""
        if self.low == 0:
            return False
        return abs(price - self.low) / self.low <= tolerance_pct

    def is_near_mid(self, price: float, tolerance_pct: float = 0.003) -> bool:
        """True when price is within tolerance_pct of the session midpoint."""
        target = self.mid
        if target == 0:
            return False
        return abs(price - target) / target <= tolerance_pct


# ──────────────────────────────────────────────────────────────────────────────
# Detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_session_ranges(
    df: pd.DataFrame,
    sessions: list[SessionType] | None = None,
) -> list[SessionRange]:
    """
    Derive completed session H/L ranges from a datetime-indexed OHLCV DataFrame.

    Each session type is tracked independently. A range is emitted when the
    first bar outside the session window is encountered, or when a new day's
    instance of the same session begins (handles sparse/gapped data).

    Incomplete sessions still active at the end of the data are discarded.

    Args:
        df:       OHLCV DataFrame with a DatetimeIndex (UTC or tz-naive UTC).
                  Required columns: ``high``, ``low``.
        sessions: Which session types to detect. Defaults to all three.

    Returns:
        Completed SessionRange objects in chronological order (by formed_at).

    Raises:
        ValueError: If ``df.index`` is not a DatetimeIndex.
    """
    if sessions is None:
        sessions = list(SessionType)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex")

    # Normalise to tz-naive UTC
    timestamps = df.index
    if timestamps.tz is not None:
        timestamps = timestamps.tz_convert("UTC").tz_localize(None)

    highs = df["high"].to_numpy(dtype=float)
    lows  = df["low"].to_numpy(dtype=float)
    n     = len(df)

    completed: list[SessionRange] = []

    # active[stype] holds in-progress session state
    active: dict[SessionType, dict] = {}

    for bar in range(n):
        hour = timestamps[bar].hour
        date = timestamps[bar].date()

        for stype in sessions:
            start_h, end_h = _SESSION_BOUNDS[stype]
            in_session = start_h <= hour < end_h

            if in_session:
                state = active.get(stype)

                if state is None or state["date"] != date:
                    # New calendar day → finalize the previous session if open
                    # (handles data gaps where no out-of-session bar was seen)
                    if state is not None:
                        completed.append(_build_range(stype, state, formed_at=bar))
                    active[stype] = {
                        "date":      date,
                        "high":      highs[bar],
                        "low":       lows[bar],
                        "high_bar":  bar,
                        "low_bar":   bar,
                        "start_bar": bar,
                    }
                else:
                    # Extend the running session high/low
                    if highs[bar] > state["high"]:
                        state["high"]     = highs[bar]
                        state["high_bar"] = bar
                    if lows[bar] < state["low"]:
                        state["low"]     = lows[bar]
                        state["low_bar"] = bar

            else:
                # Outside session window — finalize if an open session exists
                if stype in active:
                    completed.append(_build_range(stype, active.pop(stype), formed_at=bar))

    # Sessions still active at data-end are incomplete → discarded
    return sorted(completed, key=lambda r: (r.formed_at, r.session))


def _build_range(stype: SessionType, state: dict, formed_at: int) -> SessionRange:
    return SessionRange(
        session=stype,
        high=state["high"],
        low=state["low"],
        high_bar=state["high_bar"],
        low_bar=state["low_bar"],
        start_bar=state["start_bar"],
        formed_at=formed_at,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Query helpers
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# ICT Kill Zones (UTC) — high-probability entry windows for XAUUSD
# ──────────────────────────────────────────────────────────────────────────────

# London Kill Zone: Asian→London transition, institutional order flow
LONDON_KZ: tuple[int, int, int, int] = (7, 0, 11, 0)   # 07:00–11:00 UTC

# New York Kill Zone: London/NY overlap, highest XAUUSD liquidity
NY_KZ: tuple[int, int, int, int] = (12, 0, 15, 0)       # 12:00–15:00 UTC


def _ts_minutes_utc(ts: pd.Timestamp) -> int:
    """Return minutes-since-midnight UTC for *ts*."""
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
    return ts.hour * 60 + ts.minute


def is_in_killzone(ts: pd.Timestamp) -> bool:
    """Return True if *ts* falls within the London or NY kill zone (UTC)."""
    m = _ts_minutes_utc(ts)
    in_london = LONDON_KZ[0] * 60 + LONDON_KZ[1] <= m < LONDON_KZ[2] * 60 + LONDON_KZ[3]
    in_ny     = NY_KZ[0]     * 60 + NY_KZ[1]     <= m < NY_KZ[2]     * 60 + NY_KZ[3]
    return in_london or in_ny


def killzone_name(ts: pd.Timestamp) -> str | None:
    """Return 'London', 'NY', or None if outside both kill zones."""
    m = _ts_minutes_utc(ts)
    if LONDON_KZ[0] * 60 + LONDON_KZ[1] <= m < LONDON_KZ[2] * 60 + LONDON_KZ[3]:
        return "London"
    if NY_KZ[0] * 60 + NY_KZ[1] <= m < NY_KZ[2] * 60 + NY_KZ[3]:
        return "NY"
    return None


def get_session_ranges(
    ranges: list[SessionRange],
    at_bar: int,
    session: SessionType | None = None,
) -> list[SessionRange]:
    """
    Return all completed session ranges that are actionable at at_bar,
    optionally filtered to a single session type.
    """
    return [
        r for r in ranges
        if r.formed_at <= at_bar
        and (session is None or r.session == session)
    ]


def last_session_range(
    ranges: list[SessionRange],
    at_bar: int,
    session: SessionType | None = None,
) -> SessionRange | None:
    """
    Return the most recently completed session range at or before at_bar.
    """
    candidates = get_session_ranges(ranges, at_bar, session)
    return candidates[-1] if candidates else None


# ──────────────────────────────────────────────────────────────────────────────
# Daily bias (Recommendation 2 — 1D alignment gate)
# ──────────────────────────────────────────────────────────────────────────────

def get_daily_bias(df_daily: pd.DataFrame, ltf_ts: pd.Timestamp) -> int:
    """
    Return the directional bias of the last fully closed daily candle.

    Looks up the last daily bar whose open timestamp falls strictly before the
    calendar-day boundary of *ltf_ts*, ensuring only completed candles are used.

    Args:
        df_daily: Daily OHLCV DataFrame with a DatetimeIndex.
        ltf_ts:   Timestamp of the current (lower-timeframe) bar.
                  Must share the same timezone as df_daily.index.

    Returns:
        BULLISH (+1) if close > open,
        BEARISH (-1) if close < open,
        0           if close == open or no prior daily bar exists.
    """
    try:
        day_boundary = ltf_ts.normalize()
        idx = df_daily.index.searchsorted(day_boundary, side="left") - 1
    except TypeError:
        return 0
    if idx < 0:
        return 0
    row   = df_daily.iloc[idx]
    close = float(row["close"])
    open_ = float(row["open"])
    if close > open_:
        return BULLISH
    if close < open_:
        return BEARISH
    return 0
