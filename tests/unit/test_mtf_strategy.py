"""
Unit tests for zeus.strategy.mtf_strategy.MTFSMCStrategy.

Test classes
------------
TestMTFSMCStrategyInit       — parameter storage
TestKillZoneGate             — killzone_only=True rejects bars outside kill zones
TestDailyBiasGateMTF         — df_daily gate rejects signals conflicting with daily bias
TestGenerateSignalMTFSmoke   — end-to-end smoke test (no mocks, synthetic data)
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from zeus.strategy.base import Signal, SignalType
from zeus.strategy.confluence import PatternGrade
from zeus.strategy.mtf_strategy import MTFSMCStrategy
from zeus.strategy.smc.pivot import BEARISH, BULLISH


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _synthetic_ohlcv(n: int = 300, seed: int = 0, start: str | None = None,
                     freq: str = "1h") -> pd.DataFrame:
    """Deterministic OHLCV DataFrame; optionally with a DatetimeIndex."""
    rng    = np.random.default_rng(seed)
    closes = 2000.0 + np.cumsum(rng.normal(0, 2, n))
    highs  = closes + rng.uniform(0.5, 5, n)
    lows   = closes - rng.uniform(0.5, 5, n)
    opens  = np.roll(closes, 1); opens[0] = closes[0]
    vols   = rng.uniform(10, 200, n)
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows,
                       "close": closes, "volume": vols})
    if start is not None:
        df.index = pd.date_range(start, periods=n, freq=freq)
    return df


def _daily_df(opens: list[float], closes: list[float],
              start: str = "2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(opens), freq="D")
    return pd.DataFrame(
        {"open": opens, "high": closes, "low": opens, "close": closes},
        index=idx,
    )


def _make_htf(n: int = 300, seed: int = 1) -> pd.DataFrame:
    return _synthetic_ohlcv(n=n, seed=seed, start="2023-01-01", freq="4h")


# ──────────────────────────────────────────────────────────────────────────────
# Initialisation
# ──────────────────────────────────────────────────────────────────────────────

class TestMTFSMCStrategyInit:
    def test_df_htf_stored(self):
        df_htf = _make_htf()
        s = MTFSMCStrategy(df_htf=df_htf)
        assert s._df_htf is df_htf

    def test_killzone_only_default_true(self):
        s = MTFSMCStrategy(df_htf=_make_htf())
        assert s._killzone_only is True

    def test_killzone_only_can_be_disabled(self):
        s = MTFSMCStrategy(df_htf=_make_htf(), killzone_only=False)
        assert s._killzone_only is False

    def test_df_daily_default_none(self):
        s = MTFSMCStrategy(df_htf=_make_htf())
        assert s._df_daily is None

    def test_df_daily_stored(self):
        df_htf  = _make_htf()
        df_daily = _daily_df([100.0], [110.0])
        s = MTFSMCStrategy(df_htf=df_htf, df_daily=df_daily)
        assert s._df_daily is df_daily

    def test_min_grade_default_c(self):
        s = MTFSMCStrategy(df_htf=_make_htf())
        assert s._min_grade == PatternGrade.C

    def test_sl_pips_default(self):
        s = MTFSMCStrategy(df_htf=_make_htf())
        assert s._sl_pips == pytest.approx(20.0)

    def test_max_sl_pips_default(self):
        s = MTFSMCStrategy(df_htf=_make_htf())
        assert s._max_sl_pips == pytest.approx(30.0)

    def test_min_htf_bars_computed(self):
        s = MTFSMCStrategy(df_htf=_make_htf(), swing_length=50, internal_length=5)
        assert s._min_htf_bars == 100  # max(50,5)*2


# ──────────────────────────────────────────────────────────────────────────────
# Kill Zone gate
# ──────────────────────────────────────────────────────────────────────────────

class TestKillZoneGate:
    """
    When killzone_only=True (default), bars outside London/NY kill zones
    must return Signal(NONE) immediately, before any HTF analysis runs.
    """

    @pytest.fixture
    def strategy(self):
        return MTFSMCStrategy(df_htf=_make_htf(), killzone_only=True)

    def _ltf_outside_kz(self, n: int = 200) -> pd.DataFrame:
        """LTF DataFrame whose last bar falls at 03:00 UTC (outside both KZs)."""
        # Last bar at 03:00 UTC
        start = "2024-01-15 03:00"
        # Step backwards so that bar_index=n-1 lands at 03:00
        idx = pd.date_range(end=start, periods=n, freq="1min")
        rng = np.random.default_rng(99)
        c   = 2000.0 + np.cumsum(rng.normal(0, 1, n))
        return pd.DataFrame(
            {"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": [10.0] * n},
            index=idx,
        )

    def _ltf_inside_london_kz(self, n: int = 200) -> pd.DataFrame:
        """LTF DataFrame whose last bar falls at 09:00 UTC (London KZ)."""
        start = "2024-01-15 09:00"
        idx = pd.date_range(end=start, periods=n, freq="1min")
        rng = np.random.default_rng(77)
        c   = 2000.0 + np.cumsum(rng.normal(0, 1, n))
        return pd.DataFrame(
            {"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": [10.0] * n},
            index=idx,
        )

    def test_bar_outside_kz_returns_none_signal(self, strategy):
        df_ltf = self._ltf_outside_kz()
        sig    = strategy.generate_signal(df_ltf, bar_index=len(df_ltf) - 1)
        assert sig.type == SignalType.NONE

    def test_bar_outside_kz_reason_mentions_killzone(self, strategy):
        df_ltf = self._ltf_outside_kz()
        sig    = strategy.generate_signal(df_ltf, bar_index=len(df_ltf) - 1)
        assert "killzone" in sig.reason.lower()

    def test_bar_outside_kz_confidence_zero(self, strategy):
        df_ltf = self._ltf_outside_kz()
        sig    = strategy.generate_signal(df_ltf, bar_index=len(df_ltf) - 1)
        assert sig.confidence == pytest.approx(0.0)

    def test_killzone_disabled_does_not_filter(self):
        s      = MTFSMCStrategy(df_htf=_make_htf(), killzone_only=False)
        df_ltf = self._ltf_outside_kz()
        # Without the gate the pipeline proceeds (may still return NONE for
        # other reasons, but NOT due to the killzone check)
        sig = s.generate_signal(df_ltf, bar_index=len(df_ltf) - 1)
        # We can only assert it doesn't say "outside killzone"
        assert "outside killzone" not in sig.reason.lower()


# ──────────────────────────────────────────────────────────────────────────────
# Daily bias gate in MTFSMCStrategy
# ──────────────────────────────────────────────────────────────────────────────

class TestDailyBiasGateMTF:
    """
    Verify that the daily bias gate in MTFSMCStrategy rejects trades that
    conflict with the prior day's candle direction.

    Strategy under test: killzone_only=False (to isolate the daily bias gate).
    HTF swing_bias and internal_bias are controlled via mocks.
    """

    def _make_strategy(self, df_daily=None):
        return MTFSMCStrategy(
            df_htf=_make_htf(),
            killzone_only=False,
            swing_length=10,
            internal_length=5,
            df_daily=df_daily,
        )

    def _ltf_with_ts(self, date: str = "2024-01-02", hour: int = 9,
                     n: int = 200) -> pd.DataFrame:
        """LTF DataFrame with DatetimeIndex; last bar at date+hour."""
        end = pd.Timestamp(f"{date} {hour:02d}:00:00")
        idx = pd.date_range(end=end, periods=n, freq="1min")
        rng = np.random.default_rng(55)
        c   = 2000.0 + np.cumsum(rng.normal(0, 1, n))
        return pd.DataFrame(
            {"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": [10.0] * n},
            index=idx,
        )

    def _mock_htf_result(self, direction: int):
        """Return an SMCResult mock with swing_bias=internal_bias=direction."""
        r = MagicMock()
        r.swing_bias    = direction
        r.internal_bias = direction
        return r

    def test_no_df_daily_signal_not_filtered(self):
        s      = self._make_strategy(df_daily=None)
        df_ltf = self._ltf_with_ts()
        with patch.object(s, "_htf_analysis", return_value=self._mock_htf_result(BULLISH)), \
             patch.object(s, "_in_htf_zone", return_value="OB"), \
             patch.object(s, "_ltf_confirms", return_value=True), \
             patch("zeus.strategy.mtf_strategy.best_confluence") as mock_bc:
            mock_bc.return_value = None  # Enough to stop at confluence step
            sig = s.generate_signal(df_ltf, bar_index=len(df_ltf) - 1)
        # Stopped at confluence step, not daily bias
        assert "daily bias" not in sig.reason.lower()

    def test_aligned_daily_does_not_filter_bullish(self):
        """Bullish daily + bullish HTF direction → no rejection at daily gate."""
        df_daily = _daily_df([100.0], [110.0], "2024-01-01")
        s        = self._make_strategy(df_daily=df_daily)
        df_ltf   = self._ltf_with_ts("2024-01-02")
        with patch.object(s, "_htf_analysis", return_value=self._mock_htf_result(BULLISH)), \
             patch.object(s, "_in_htf_zone", return_value="OB"), \
             patch.object(s, "_ltf_confirms", return_value=True), \
             patch("zeus.strategy.mtf_strategy.best_confluence") as mock_bc:
            mock_bc.return_value = None
            sig = s.generate_signal(df_ltf, bar_index=len(df_ltf) - 1)
        assert "daily bias" not in sig.reason.lower()

    def test_conflicting_daily_rejects_bullish_signal(self):
        """Bearish daily + bullish HTF → rejected at daily bias gate."""
        df_daily = _daily_df([110.0], [100.0], "2024-01-01")  # bearish
        s        = self._make_strategy(df_daily=df_daily)
        df_ltf   = self._ltf_with_ts("2024-01-02")
        with patch.object(s, "_htf_analysis", return_value=self._mock_htf_result(BULLISH)):
            sig = s.generate_signal(df_ltf, bar_index=len(df_ltf) - 1)
        assert sig.type == SignalType.NONE
        assert "daily bias" in sig.reason.lower()

    def test_conflicting_daily_rejects_bearish_signal(self):
        """Bullish daily + bearish HTF → rejected at daily bias gate."""
        df_daily = _daily_df([100.0], [110.0], "2024-01-01")  # bullish
        s        = self._make_strategy(df_daily=df_daily)
        df_ltf   = self._ltf_with_ts("2024-01-02")
        with patch.object(s, "_htf_analysis", return_value=self._mock_htf_result(BEARISH)):
            sig = s.generate_signal(df_ltf, bar_index=len(df_ltf) - 1)
        assert sig.type == SignalType.NONE
        assert "daily bias" in sig.reason.lower()

    def test_doji_daily_does_not_reject(self):
        """Neutral daily (doji) → gate inactive, pipeline continues."""
        df_daily = _daily_df([100.0], [100.0], "2024-01-01")  # doji
        s        = self._make_strategy(df_daily=df_daily)
        df_ltf   = self._ltf_with_ts("2024-01-02")
        with patch.object(s, "_htf_analysis", return_value=self._mock_htf_result(BULLISH)), \
             patch.object(s, "_in_htf_zone", return_value="OB"), \
             patch.object(s, "_ltf_confirms", return_value=True), \
             patch("zeus.strategy.mtf_strategy.best_confluence") as mock_bc:
            mock_bc.return_value = None
            sig = s.generate_signal(df_ltf, bar_index=len(df_ltf) - 1)
        assert "daily bias" not in sig.reason.lower()

    def test_rejection_reason_is_descriptive(self):
        df_daily = _daily_df([110.0], [100.0], "2024-01-01")
        s        = self._make_strategy(df_daily=df_daily)
        df_ltf   = self._ltf_with_ts("2024-01-02")
        with patch.object(s, "_htf_analysis", return_value=self._mock_htf_result(BULLISH)):
            sig = s.generate_signal(df_ltf, bar_index=len(df_ltf) - 1)
        assert "bearish" in sig.reason.lower()
        assert "bullish" in sig.reason.lower()


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end smoke test (no mocks)
# ──────────────────────────────────────────────────────────────────────────────

class TestGenerateSignalMTFSmoke:
    """
    Real pipeline with synthetic data. We don't assert on direction — just
    that the strategy runs without crashing and returns well-formed Signals.
    """

    @pytest.fixture
    def df_htf(self):
        return _synthetic_ohlcv(n=300, seed=10, start="2023-01-01", freq="4h")

    @pytest.fixture
    def df_ltf(self):
        # 1M bars starting after the first HTF bar
        return _synthetic_ohlcv(n=500, seed=20, start="2023-01-02", freq="1min")

    @pytest.fixture
    def strategy(self, df_htf):
        return MTFSMCStrategy(
            df_htf=df_htf,
            killzone_only=False,
            swing_length=10,
            internal_length=5,
            atr_period=50,
        )

    def test_returns_signal_instance(self, strategy, df_ltf):
        sig = strategy.generate_signal(df_ltf, bar_index=len(df_ltf) - 1)
        assert isinstance(sig, Signal)

    def test_signal_type_is_valid(self, strategy, df_ltf):
        sig = strategy.generate_signal(df_ltf, bar_index=len(df_ltf) - 1)
        assert sig.type in (SignalType.LONG, SignalType.SHORT, SignalType.NONE)

    def test_confidence_in_unit_interval(self, strategy, df_ltf):
        sig = strategy.generate_signal(df_ltf, bar_index=len(df_ltf) - 1)
        assert 0.0 <= sig.confidence <= 1.0

    def test_bar_index_propagated(self, strategy, df_ltf):
        bar = len(df_ltf) - 1
        sig = strategy.generate_signal(df_ltf, bar_index=bar)
        assert sig.bar_index == bar

    def test_no_crash_on_multiple_bars(self, strategy, df_ltf):
        signals = [
            strategy.generate_signal(df_ltf, bar_index=i)
            for i in range(0, min(50, len(df_ltf)), 10)
        ]
        assert all(isinstance(s, Signal) for s in signals)

    def test_insufficient_htf_data_returns_none(self, df_ltf):
        """Strategy with only a handful of HTF bars → insufficient data guard."""
        df_htf_tiny = _synthetic_ohlcv(n=5, seed=0, start="2023-01-01", freq="4h")
        s = MTFSMCStrategy(df_htf=df_htf_tiny, killzone_only=False,
                           swing_length=10, internal_length=5)
        sig = s.generate_signal(df_ltf, bar_index=len(df_ltf) - 1)
        assert sig.type == SignalType.NONE
        assert "htf" in sig.reason.lower()
