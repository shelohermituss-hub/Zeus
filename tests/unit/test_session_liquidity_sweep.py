"""
Unit tests for prev_session_liquidity_swept() and the require_session_sweep gate.
"""
from __future__ import annotations

import pandas as pd
import pytest

from zeus.strategy.smc.session import (
    SessionRange,
    SessionType,
    detect_session_ranges,
    prev_session_liquidity_swept,
)
from zeus.strategy.smc.pivot import BULLISH, BEARISH
from zeus.strategy.scalp_strategy import ScalpSMCStrategy
from zeus.strategy.mtf_strategy import MTFSMCStrategy


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_session_range(
    session:   SessionType = SessionType.ASIAN,
    high:      float = 2010.0,
    low:       float = 1990.0,
    formed_at: int   = 10,
) -> SessionRange:
    return SessionRange(
        session=session,
        high=high,
        low=low,
        high_bar=5,
        low_bar=7,
        start_bar=0,
        formed_at=formed_at,
    )


def _make_df(
    n:      int   = 50,
    price:  float = 2000.0,
    highs:  list[tuple[int, float]] | None = None,
    lows:   list[tuple[int, float]] | None = None,
) -> pd.DataFrame:
    """
    Build a flat LTF OHLCV df; optionally override specific bar highs/lows.
    """
    h = [price + 1.0] * n
    l = [price - 1.0] * n
    if highs:
        for i, v in highs:
            h[i] = v
    if lows:
        for i, v in lows:
            l[i] = v
    return pd.DataFrame(
        {
            "open":   [price] * n,
            "high":   h,
            "low":    l,
            "close":  [price] * n,
            "volume": [1000.0] * n,
        },
        index=pd.date_range("2026-01-06 08:00", periods=n, freq="1min", tz="UTC"),
    )


def _flat_df(n: int = 300, price: float = 2000.0) -> pd.DataFrame:
    return _make_df(n=n, price=price)


# ── Tests: prev_session_liquidity_swept ───────────────────────────────────────

class TestPrevSessionLiquiditySweep:

    def test_no_session_ranges_returns_false(self):
        df = _make_df()
        swept, reason = prev_session_liquidity_swept([], df, bar_index=30, direction=BULLISH)
        assert not swept
        assert "no completed session" in reason

    def test_no_range_before_bar_returns_false(self):
        # Session formed_at=100 but bar_index=50 → not yet available
        sr = _make_session_range(formed_at=100)
        df = _make_df()
        swept, _ = prev_session_liquidity_swept([sr], df, bar_index=50, direction=BULLISH)
        assert not swept

    # ── BULLISH (sell-side sweep) ─────────────────────────────────────────

    def test_bullish_sweep_detected_within_lookback(self):
        """Bar 25 dips below session low (1990) → sell-side sweep confirmed."""
        sr = _make_session_range(low=1990.0, high=2010.0, formed_at=15)
        df = _make_df(lows=[(25, 1988.0)])   # bar 25: low = 1988 < 1990
        swept, reason = prev_session_liquidity_swept(
            [sr], df, bar_index=30, direction=BULLISH, sweep_lookback=20,
        )
        assert swept
        assert "sell-side" in reason
        assert "1988" in reason

    def test_bullish_sweep_exact_boundary_not_swept(self):
        """Bar low == session low is NOT a sweep (must be strictly below)."""
        sr = _make_session_range(low=1990.0, high=2010.0, formed_at=15)
        df = _make_df(lows=[(25, 1990.0)])   # low == session low, not below
        swept, _ = prev_session_liquidity_swept(
            [sr], df, bar_index=30, direction=BULLISH, sweep_lookback=20,
        )
        assert not swept

    def test_bullish_sweep_outside_lookback_not_detected(self):
        """Sweep at bar 10 is outside lookback=5 from bar_index=30."""
        sr = _make_session_range(low=1990.0, formed_at=5)
        df = _make_df(lows=[(10, 1988.0)])
        swept, _ = prev_session_liquidity_swept(
            [sr], df, bar_index=30, direction=BULLISH, sweep_lookback=5,
        )
        assert not swept

    def test_bullish_sweep_before_session_close_not_counted(self):
        """Sweep bar 8 is before formed_at=10 → not counted."""
        sr = _make_session_range(low=1990.0, formed_at=10)
        df = _make_df(lows=[(8, 1988.0)])    # before session closed
        swept, _ = prev_session_liquidity_swept(
            [sr], df, bar_index=30, direction=BULLISH, sweep_lookback=30,
        )
        assert not swept

    # ── BEARISH (buy-side sweep) ──────────────────────────────────────────

    def test_bearish_sweep_detected_within_lookback(self):
        """Bar 25 spikes above session high (2010) → buy-side sweep confirmed."""
        sr = _make_session_range(low=1990.0, high=2010.0, formed_at=15)
        df = _make_df(highs=[(25, 2012.0)])
        swept, reason = prev_session_liquidity_swept(
            [sr], df, bar_index=30, direction=BEARISH, sweep_lookback=20,
        )
        assert swept
        assert "buy-side" in reason
        assert "2012" in reason

    def test_bearish_sweep_exact_boundary_not_swept(self):
        """Bar high == session high is NOT a sweep (must be strictly above)."""
        sr = _make_session_range(high=2010.0, formed_at=15)
        df = _make_df(highs=[(25, 2010.0)])
        swept, _ = prev_session_liquidity_swept(
            [sr], df, bar_index=30, direction=BEARISH, sweep_lookback=20,
        )
        assert not swept

    def test_bearish_no_sweep_returns_false_with_reason(self):
        sr = _make_session_range(high=2010.0, low=1990.0, formed_at=5)
        df = _make_df()   # all highs = 2001, all lows = 1999 → no sweep
        swept, reason = prev_session_liquidity_swept(
            [sr], df, bar_index=30, direction=BEARISH, sweep_lookback=10,
        )
        assert not swept
        assert "no" in reason.lower() or "sweep" in reason.lower()

    # ── Lookback window mechanics ─────────────────────────────────────────

    def test_lookback_window_starts_after_session_close(self):
        """
        When session formed_at > bar_index - sweep_lookback, the effective
        scan window starts at formed_at (not bar_index - lookback).
        """
        sr = _make_session_range(low=1990.0, formed_at=20)
        # Sweep at bar 21 (just after session closed), bar_index=30, lookback=20
        # → effective start = max(20, 30-20+1=11) = 20
        df = _make_df(lows=[(21, 1988.0)])
        swept, _ = prev_session_liquidity_swept(
            [sr], df, bar_index=30, direction=BULLISH, sweep_lookback=20,
        )
        assert swept

    def test_most_recent_session_is_used(self):
        """With two sessions, the most recent one (formed_at=20) is used."""
        old_sr  = _make_session_range(low=1980.0, high=2020.0, formed_at=5)
        new_sr  = _make_session_range(low=1995.0, high=2005.0, formed_at=20)
        # Bar 25 dips to 1993 — below new_sr.low (1995) but NOT below old_sr.low (1980)
        df = _make_df(lows=[(25, 1993.0)])
        swept, reason = prev_session_liquidity_swept(
            [old_sr, new_sr], df, bar_index=30, direction=BULLISH, sweep_lookback=15,
        )
        assert swept   # new_sr.low=1995 was swept
        assert "1993" in reason

    def test_reason_includes_session_name(self):
        sr = _make_session_range(session=SessionType.LONDON, low=1990.0, formed_at=5)
        df = _make_df(lows=[(20, 1988.0)])
        swept, reason = prev_session_liquidity_swept(
            [sr], df, bar_index=25, direction=BULLISH, sweep_lookback=20,
        )
        assert swept
        assert "LONDON" in reason.upper() or "london" in reason.lower()


# ── Tests: detect_session_ranges integration ─────────────────────────────────

class TestDetectAndSweep:

    def test_detect_then_sweep_asian_london_flow(self):
        """
        Build a 1M df spanning Asian + London session.
        Asian: 00:00–08:00 UTC → range [1990, 2010].
        London: bar after 08:00 spikes below 1990 → sweep confirmed.
        """
        # 12h of 1M bars: 00:00–12:00 UTC on 2026-01-06
        idx = pd.date_range("2026-01-06 00:00", periods=720, freq="1min", tz="UTC")
        highs  = [2005.0] * 720
        lows   = [1995.0] * 720
        closes = [2000.0] * 720

        # Asian session: bar 100 has a high of 2010, bar 200 has a low of 1990
        highs[100] = 2010.0
        lows[200]  = 1990.0

        # First London bar (08:00) = bar 480; sweep at bar 490: low = 1988
        lows[490]  = 1988.0

        df = pd.DataFrame(
            {"open": closes, "high": highs, "low": lows, "close": closes, "volume": [1000]*720},
            index=idx,
        )

        session_ranges = detect_session_ranges(df)
        assert len(session_ranges) >= 1   # at least Asian range

        swept, reason = prev_session_liquidity_swept(
            session_ranges, df, bar_index=500, direction=BULLISH, sweep_lookback=30,
        )
        assert swept
        assert "1988" in reason


# ── Tests: MTFSMCStrategy parameter exposure ─────────────────────────────────

class TestMTFSessionSweepParam:

    def test_default_disabled(self):
        df = _flat_df()
        s = MTFSMCStrategy(df_htf=df)
        assert s._require_session_sweep is False

    def test_can_be_enabled(self):
        df = _flat_df()
        s = MTFSMCStrategy(df_htf=df, require_session_sweep=True)
        assert s._require_session_sweep is True

    def test_lookback_stored(self):
        df = _flat_df()
        s = MTFSMCStrategy(df_htf=df, session_sweep_lookback=60)
        assert s._session_sweep_lookback == 60

    def test_session_ranges_cache_starts_none(self):
        df = _flat_df()
        s = MTFSMCStrategy(df_htf=df)
        assert s._session_ranges is None


class TestScalpSessionSweepParam:

    def test_default_disabled(self):
        df = _flat_df()
        s = ScalpSMCStrategy(df_htf_1h=df)
        assert s._require_session_sweep is False

    def test_can_be_enabled(self):
        df = _flat_df()
        s = ScalpSMCStrategy(df_htf_1h=df, require_session_sweep=True)
        assert s._require_session_sweep is True

    def test_lookback_wired(self):
        df = _flat_df()
        s = ScalpSMCStrategy(df_htf_1h=df, session_sweep_lookback=45)
        assert s._session_sweep_lookback == 45
