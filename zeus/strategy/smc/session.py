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

# ──────────────────────────────────────────────────────────────────────────────
# Asian Range Sweep Confirmation (Recommendation 6)
#
# ICT killzone logic:
#   Asian session (00:00–08:00 UTC) forms a range whose high/low represents
#   trapped resting liquidity. During the London or NY kill zone, smart money
#   sweeps the Asian extreme opposite to the intended direction before reversing:
#   - Bullish setup: Asian LOW is swept (bar low < asian_low) → liquidity taken
#   - Bearish setup: Asian HIGH is swept (bar high > asian_high) → liquidity taken
#
# The sweep must occur AFTER the Asian session closes (>= 08:00 UTC) and before
# the current LTF bar.  Without a confirmed sweep the setup is not valid.
# ──────────────────────────────────────────────────────────────────────────────

def _tz_naive_utc(ts: pd.Timestamp) -> pd.Timestamp:
    """Return *ts* as a tz-naive UTC Timestamp."""
    if ts.tzinfo is not None:
        return ts.tz_convert("UTC").replace(tzinfo=None)
    return ts


def _utc_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Return df.index as a tz-naive UTC DatetimeIndex."""
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex")
    if idx.tz is not None:
        return idx.tz_convert("UTC").tz_localize(None)
    return idx


_ASIAN_OPEN_H  = 0   # 00:00 UTC
_ASIAN_CLOSE_H = 8   # 08:00 UTC


def get_asian_range_for_day(
    df: pd.DataFrame,
    ltf_ts: pd.Timestamp,
) -> tuple[float, float] | None:
    """
    Return (asian_high, asian_low) for the Asian session of ltf_ts's UTC calendar day.

    The Asian session window is 00:00–08:00 UTC. This function only returns a
    range when the session is fully closed (ltf_ts >= 08:00 UTC of the same day),
    ensuring strategy code never reads a partially-formed range.

    Args:
        df:      LTF OHLCV DataFrame with a DatetimeIndex (UTC or tz-naive UTC).
                 Requires ``high`` and ``low`` columns.
        ltf_ts:  Current bar timestamp.

    Returns:
        (asian_high, asian_low) or None if the session has not closed yet or
        no bars fall within the Asian window.
    """
    ts_utc   = _tz_naive_utc(ltf_ts)
    day_start = ts_utc.normalize()                            # 00:00 UTC
    asian_end = day_start + pd.Timedelta(hours=_ASIAN_CLOSE_H)

    if ts_utc < asian_end:
        return None   # Asian session still open

    utc_idx    = _utc_index(df)
    asian_mask = (utc_idx >= day_start) & (utc_idx < asian_end)
    asian_bars = df.iloc[asian_mask.nonzero()[0]]

    if len(asian_bars) == 0:
        return None

    return float(asian_bars["high"].max()), float(asian_bars["low"].min())


def asian_range_swept(
    df:        pd.DataFrame,
    bar_index: int,
    ltf_ts:    pd.Timestamp,
    direction: int,
) -> bool:
    """
    Return True if the Asian range extreme aligned with *direction* was swept
    (taken out by a wick) between Asian close (08:00 UTC) and *bar_index* inclusive.

    For a BULLISH setup: a bar's *low* must have gone below the Asian session low.
    For a BEARISH setup: a bar's *high* must have gone above the Asian session high.

    Args:
        df:        LTF OHLCV DataFrame with DatetimeIndex and ``high``/``low`` columns.
        bar_index: Index of the current bar in df (inclusive upper bound for sweep check).
        ltf_ts:    Timestamp of bar_index (used to locate today's Asian range).
        direction: BULLISH (+1) or BEARISH (-1).

    Returns:
        True if the sweep occurred; False if the Asian range is unavailable or
        no sweep was detected.
    """
    asian_range = get_asian_range_for_day(df, ltf_ts)
    if asian_range is None:
        return False

    asian_high, asian_low = asian_range

    ts_utc    = _tz_naive_utc(ltf_ts)
    day_start = ts_utc.normalize()
    asian_end = day_start + pd.Timedelta(hours=_ASIAN_CLOSE_H)

    utc_idx       = _utc_index(df)
    post_asian    = utc_idx >= asian_end

    highs = df["high"].to_numpy(dtype=float)
    lows  = df["low"].to_numpy(dtype=float)

    for i in range(bar_index + 1):
        if not post_asian[i]:
            continue
        if direction == BULLISH and lows[i] < asian_low:
            return True
        if direction == BEARISH and highs[i] > asian_high:
            return True

    return False


def get_weekly_bias(df_daily: pd.DataFrame, ltf_ts: pd.Timestamp) -> int:
    """
    Return the directional bias of the last fully closed weekly candle.

    Resamples *df_daily* to weekly (Monday-open, Sunday-close, label='left') and
    finds the last completed week strictly before the ISO week that contains
    *ltf_ts*.  Only a full week (at least 4 trading days) is used; partial
    first/last weeks are discarded.

    Args:
        df_daily: Daily OHLCV DataFrame with a DatetimeIndex (UTC or tz-naive).
        ltf_ts:   Timestamp of the current LTF bar.

    Returns:
        BULLISH (+1) if weekly close > open,
        BEARISH (-1) if weekly close < open,
        0           if equal or no prior complete week exists.
    """
    if df_daily is None or len(df_daily) == 0:
        return 0

    try:
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in df_daily.columns:
            agg["volume"] = "sum"
        df_weekly = (
            df_daily.resample("W-MON", label="left", closed="left")
            .agg(agg)
            .dropna(subset=["open"])
        )
    except Exception:
        return 0

    if df_weekly.empty:
        return 0

    # Find the Monday that opens the current week
    ts_utc = _tz_naive_utc(ltf_ts)
    current_week_start = ts_utc - pd.Timedelta(days=ts_utc.weekday())
    current_week_start = current_week_start.normalize()

    idx = df_weekly.index.searchsorted(current_week_start, side="left") - 1
    if idx < 0:
        return 0

    row   = df_weekly.iloc[idx]
    close = float(row["close"])
    open_ = float(row["open"])
    if close > open_:
        return BULLISH
    if close < open_:
        return BEARISH
    return 0


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
