"""
Unit tests for ltf_liquidity_sweep() — LTF entry sweep confirmation.
"""
from __future__ import annotations

import pandas as pd
import pytest

from zeus.strategy.smc.ltf_sweep import ltf_liquidity_sweep
from zeus.strategy.smc.pivot import BULLISH, BEARISH
from zeus.strategy.scalp_strategy import ScalpSMCStrategy
from zeus.strategy.mtf_strategy import MTFSMCStrategy


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df(ohlc: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Build df from list of (open, high, low, close) tuples."""
    return pd.DataFrame(
        {
            "open":   [r[0] for r in ohlc],
            "high":   [r[1] for r in ohlc],
            "low":    [r[2] for r in ohlc],
            "close":  [r[3] for r in ohlc],
            "volume": [1000.0] * len(ohlc),
        },
        index=pd.date_range("2026-01-06 08:00", periods=len(ohlc), freq="1min", tz="UTC"),
    )


def _flat_df(n: int = 300, price: float = 2000.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open":   [price] * n,
            "high":   [price + 1] * n,
            "low":    [price - 1] * n,
            "close":  [price] * n,
            "volume": [1000.0] * n,
        },
        index=pd.date_range("2026-01-06 08:00", periods=n, freq="1min", tz="UTC"),
    )


# ── Tests: BULLISH sweep (sell-side liquidity taken) ─────────────────────────

class TestBullishSweep:

    def test_clean_sweep_detected(self):
        """
        Bar 1: high=2005, low=1995, close=2002 (prev bar)
        Bar 2: high=2003, low=1993, close=1997  ← low < prev_low (1995), close > prev_low
        """
        df = _make_df([
            (2000, 2005, 1995, 2002),  # bar 0 (prev)
            (2000, 2003, 1993, 1997),  # bar 1: low 1993 < 1995, close 1997 > 1995 → SWEEP
        ])
        confirmed, reason = ltf_liquidity_sweep(df, bar_index=1, direction=BULLISH)
        assert confirmed
        assert "sell-side" in reason
        assert "1993" in reason

    def test_low_equal_prev_low_not_a_sweep(self):
        """Low == prev_low is NOT a sweep — must be strictly below."""
        df = _make_df([
            (2000, 2005, 1995, 2002),
            (2000, 2003, 1995, 1998),  # low == prev_low, not below
        ])
        confirmed, _ = ltf_liquidity_sweep(df, bar_index=1, direction=BULLISH)
        assert not confirmed

    def test_sweep_but_close_below_prev_low_not_confirmed(self):
        """Bar swept the low but closed BELOW prev_low → continuation, not reversal."""
        df = _make_df([
            (2000, 2005, 1995, 2002),
            (2000, 2003, 1990, 1993),  # low < prev_low but close (1993) < prev_low (1995)
        ])
        confirmed, _ = ltf_liquidity_sweep(df, bar_index=1, direction=BULLISH)
        assert not confirmed

    def test_sweep_on_lookback_bar_detected(self):
        """Sweep 2 bars ago (within lookback=3) should be accepted."""
        df = _make_df([
            (2000, 2005, 1995, 2002),  # bar 0
            (2000, 2003, 1990, 1998),  # bar 1: SWEEP (low 1990 < 1995, close 1998 > 1995)
            (2001, 2004, 1996, 2002),  # bar 2: normal
            (2001, 2004, 1997, 2003),  # bar 3: entry bar
        ])
        confirmed, reason = ltf_liquidity_sweep(df, bar_index=3, direction=BULLISH, lookback=3)
        assert confirmed

    def test_sweep_outside_lookback_not_detected(self):
        """Sweep at bar 1 is outside lookback=2 from bar_index=3."""
        df = _make_df([
            (2000, 2005, 1995, 2002),
            (2000, 2003, 1990, 1998),  # SWEEP — bar 1
            (2001, 2004, 1996, 2002),  # bar 2
            (2001, 2004, 1997, 2003),  # bar 3 — entry
        ])
        confirmed, _ = ltf_liquidity_sweep(df, bar_index=3, direction=BULLISH, lookback=2)
        assert not confirmed   # lookback=2 → scans bars 2 and 3 only

    def test_no_sweep_returns_false_with_reason(self):
        df = _make_df([
            (2000, 2005, 1995, 2002),
            (2001, 2006, 1996, 2003),  # high went up, no sweep of low
            (2001, 2006, 1997, 2004),
        ])
        confirmed, reason = ltf_liquidity_sweep(df, bar_index=2, direction=BULLISH)
        assert not confirmed
        assert "no" in reason.lower() or "sell-side" in reason


# ── Tests: BEARISH sweep (buy-side liquidity taken) ──────────────────────────

class TestBearishSweep:

    def test_clean_sweep_detected(self):
        """
        Bar 0: high=2010, low=2000, close=2005
        Bar 1: high=2015, low=2002, close=2008  ← high > prev_high (2010), close < prev_high
        """
        df = _make_df([
            (2005, 2010, 2000, 2005),  # bar 0 (prev)
            (2006, 2015, 2002, 2008),  # bar 1: high 2015 > 2010, close 2008 < 2010 → SWEEP
        ])
        confirmed, reason = ltf_liquidity_sweep(df, bar_index=1, direction=BEARISH)
        assert confirmed
        assert "buy-side" in reason
        assert "2015" in reason

    def test_high_equal_prev_high_not_a_sweep(self):
        df = _make_df([
            (2005, 2010, 2000, 2005),
            (2006, 2010, 2002, 2007),  # high == prev_high, not above
        ])
        confirmed, _ = ltf_liquidity_sweep(df, bar_index=1, direction=BEARISH)
        assert not confirmed

    def test_sweep_but_close_above_prev_high_not_confirmed(self):
        """Swept the high but closed ABOVE prev_high → continuation, not reversal."""
        df = _make_df([
            (2005, 2010, 2000, 2005),
            (2006, 2015, 2002, 2012),  # high 2015 > prev_high 2010, close 2012 > 2010
        ])
        confirmed, _ = ltf_liquidity_sweep(df, bar_index=1, direction=BEARISH)
        assert not confirmed

    def test_sweep_within_lookback_detected(self):
        df = _make_df([
            (2005, 2010, 2000, 2005),  # bar 0
            (2006, 2015, 2002, 2008),  # bar 1: SWEEP
            (2007, 2009, 2001, 2004),  # bar 2: normal
            (2006, 2008, 2000, 2003),  # bar 3: entry
        ])
        confirmed, _ = ltf_liquidity_sweep(df, bar_index=3, direction=BEARISH, lookback=3)
        assert confirmed

    def test_direction_isolation(self):
        """A bullish sweep should NOT trigger a bearish check."""
        df = _make_df([
            (2000, 2005, 1995, 2002),
            (2000, 2003, 1990, 1998),  # bullish sweep: low 1990 < 1995, close 1998 > 1995
        ])
        # Bullish sweep → should pass for BULLISH, fail for BEARISH
        bull_confirmed, _ = ltf_liquidity_sweep(df, bar_index=1, direction=BULLISH)
        bear_confirmed, _ = ltf_liquidity_sweep(df, bar_index=1, direction=BEARISH)
        assert bull_confirmed
        assert not bear_confirmed


# ── Tests: edge cases ─────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_bar_index_zero_returns_false(self):
        df = _make_df([(2000, 2005, 1995, 2002)])
        confirmed, reason = ltf_liquidity_sweep(df, bar_index=0, direction=BULLISH)
        assert not confirmed
        assert "insufficient" in reason

    def test_lookback_one_only_checks_current_bar(self):
        df = _make_df([
            (2000, 2005, 1995, 2002),
            (2000, 2003, 1990, 1998),  # SWEEP at bar 1
            (2001, 2004, 1996, 2002),  # bar 2: no sweep vs bar 1 (low 1996 > low 1990)
        ])
        # lookback=1 → only bar 2 checked against bar 1
        confirmed, _ = ltf_liquidity_sweep(df, bar_index=2, direction=BULLISH, lookback=1)
        assert not confirmed   # bar 2 low (1996) is NOT below bar 1 low (1990)

    def test_returns_tuple_bool_str(self):
        df = _flat_df()
        result = ltf_liquidity_sweep(df, bar_index=5, direction=BULLISH)
        confirmed, reason = result
        assert isinstance(confirmed, bool)
        assert isinstance(reason, str)


# ── Tests: strategy parameter exposure ───────────────────────────────────────

class TestStrategyParams:

    def test_scalp_ltf_sweep_enabled_by_default(self):
        df = _flat_df()
        s = ScalpSMCStrategy(df_htf_1h=df)
        assert s._require_ltf_sweep is True

    def test_scalp_ltf_sweep_lookback_default(self):
        df = _flat_df()
        s = ScalpSMCStrategy(df_htf_1h=df)
        assert s._ltf_sweep_lookback == 3

    def test_scalp_ltf_sweep_can_be_disabled(self):
        df = _flat_df()
        s = ScalpSMCStrategy(df_htf_1h=df, require_ltf_sweep=False)
        assert s._require_ltf_sweep is False

    def test_scalp_ltf_sweep_lookback_configurable(self):
        df = _flat_df()
        s = ScalpSMCStrategy(df_htf_1h=df, ltf_sweep_lookback=5)
        assert s._ltf_sweep_lookback == 5

    def test_mtf_ltf_sweep_disabled_by_default(self):
        df = _flat_df()
        s = MTFSMCStrategy(df_htf=df)
        assert s._require_ltf_sweep is False

    def test_mtf_ltf_sweep_can_be_enabled(self):
        df = _flat_df()
        s = MTFSMCStrategy(df_htf=df, require_ltf_sweep=True)
        assert s._require_ltf_sweep is True
