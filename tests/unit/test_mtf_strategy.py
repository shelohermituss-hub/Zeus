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

    def test_sweep_zone_tol_pct_default(self):
        s = MTFSMCStrategy(df_htf=_make_htf())
        assert s._sweep_zone_tol_pct == pytest.approx(0.005)

    def test_sweep_zone_tol_pct_custom(self):
        s = MTFSMCStrategy(df_htf=_make_htf(), sweep_zone_tol_pct=0.01)
        assert s._sweep_zone_tol_pct == pytest.approx(0.01)

    def test_df_mtf_default_none(self):
        s = MTFSMCStrategy(df_htf=_make_htf())
        assert s._df_mtf is None

    def test_df_mtf_stored(self):
        df_htf = _make_htf()
        df_mtf = _synthetic_ohlcv(n=200, seed=5, start="2023-01-01", freq="1h")
        s = MTFSMCStrategy(df_htf=df_htf, df_mtf=df_mtf)
        assert s._df_mtf is df_mtf

    def test_mss_lookback_default(self):
        s = MTFSMCStrategy(df_htf=_make_htf())
        assert s._mss_lookback == 10

    def test_mss_lookback_custom(self):
        s = MTFSMCStrategy(df_htf=_make_htf(), mss_lookback=20)
        assert s._mss_lookback == 20

    def test_require_entry_fvg_default_true(self):
        s = MTFSMCStrategy(df_htf=_make_htf())
        assert s._require_entry_fvg is True

    def test_require_entry_fvg_can_be_disabled(self):
        s = MTFSMCStrategy(df_htf=_make_htf(), require_entry_fvg=False)
        assert s._require_entry_fvg is False

    def test_require_asian_sweep_default_false(self):
        s = MTFSMCStrategy(df_htf=_make_htf())
        assert s._require_asian_sweep is False

    def test_require_asian_sweep_can_be_enabled(self):
        s = MTFSMCStrategy(df_htf=_make_htf(), require_asian_sweep=True)
        assert s._require_asian_sweep is True

    def test_max_daily_signals_default(self):
        s = MTFSMCStrategy(df_htf=_make_htf())
        assert s._max_daily_signals == 2

    def test_max_daily_signals_custom(self):
        s = MTFSMCStrategy(df_htf=_make_htf(), max_daily_signals=3)
        assert s._max_daily_signals == 3

    def test_max_signals_per_session_default(self):
        s = MTFSMCStrategy(df_htf=_make_htf())
        assert s._max_signals_per_session == 1

    def test_max_signals_per_session_custom(self):
        s = MTFSMCStrategy(df_htf=_make_htf(), max_signals_per_session=2)
        assert s._max_signals_per_session == 2

    def test_signal_log_starts_empty(self):
        s = MTFSMCStrategy(df_htf=_make_htf())
        assert s._signal_log == []


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
             patch.object(s, "_ltf_entry_confirmed", return_value=True), \
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
             patch.object(s, "_ltf_entry_confirmed", return_value=True), \
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
             patch.object(s, "_ltf_entry_confirmed", return_value=True), \
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


# ──────────────────────────────────────────────────────────────────────────────
# 1H MSS gate
# ──────────────────────────────────────────────────────────────────────────────

class TestMTFMSSGate:
    """
    Verify that the 1H MSS gate (step 3.6) works correctly.

    killzone_only=False so we isolate the MSS gate.
    HTF swing/internal bias controlled via mocks.
    """

    def _make_mtf_df(self, n: int = 200) -> pd.DataFrame:
        return _synthetic_ohlcv(n=n, seed=7, start="2023-01-01", freq="1h")

    def _ltf_with_ts(self, n: int = 200) -> pd.DataFrame:
        # "2023-02-01" gives ~30 days * 6 bars/day = ~180 HTF bars — well above _min_htf_bars=20
        end = pd.Timestamp("2023-02-01 10:00:00")
        idx = pd.date_range(end=end, periods=n, freq="5min")
        rng = np.random.default_rng(42)
        c   = 2000.0 + np.cumsum(rng.normal(0, 1, n))
        return pd.DataFrame(
            {"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": [10.0] * n},
            index=idx,
        )

    def _mock_htf(self, direction: int):
        r = MagicMock()
        r.swing_bias    = direction
        r.internal_bias = direction
        return r

    def test_no_df_mtf_gate_disabled_passes_through(self):
        """When df_mtf=None the MSS gate is inactive — pipeline continues."""
        s = MTFSMCStrategy(
            df_htf=_make_htf(), df_mtf=None,
            killzone_only=False, swing_length=10, internal_length=5,
        )
        df_ltf = self._ltf_with_ts()
        with patch.object(s, "_htf_analysis", return_value=self._mock_htf(BULLISH)), \
             patch.object(s, "_in_htf_zone", return_value="OB"), \
             patch.object(s, "_ltf_entry_confirmed", return_value=True), \
             patch("zeus.strategy.mtf_strategy.best_confluence") as mock_bc:
            mock_bc.return_value = None
            sig = s.generate_signal(df_ltf, bar_index=len(df_ltf) - 1)
        assert "1h mss" not in sig.reason.lower()

    def test_mss_confirmed_passes_through(self):
        """When MSS is confirmed the gate does not reject."""
        df_mtf = self._make_mtf_df()
        s = MTFSMCStrategy(
            df_htf=_make_htf(), df_mtf=df_mtf,
            killzone_only=False, swing_length=10, internal_length=5,
        )
        df_ltf = self._ltf_with_ts()
        with patch.object(s, "_htf_analysis", return_value=self._mock_htf(BULLISH)), \
             patch.object(s, "_mtf_mss_confirmed", return_value=True), \
             patch.object(s, "_in_htf_zone", return_value="OB"), \
             patch.object(s, "_ltf_entry_confirmed", return_value=True), \
             patch("zeus.strategy.mtf_strategy.best_confluence") as mock_bc:
            mock_bc.return_value = None
            sig = s.generate_signal(df_ltf, bar_index=len(df_ltf) - 1)
        assert "1h mss" not in sig.reason.lower()

    def test_mss_not_confirmed_rejects_signal(self):
        """When MSS returns False the gate rejects with informative reason."""
        df_mtf = self._make_mtf_df()
        s = MTFSMCStrategy(
            df_htf=_make_htf(), df_mtf=df_mtf,
            killzone_only=False, swing_length=10, internal_length=5,
        )
        df_ltf = self._ltf_with_ts()
        with patch.object(s, "_htf_analysis", return_value=self._mock_htf(BULLISH)), \
             patch.object(s, "_mtf_mss_confirmed", return_value=False):
            sig = s.generate_signal(df_ltf, bar_index=len(df_ltf) - 1)
        assert sig.type == SignalType.NONE
        assert "1h mss" in sig.reason.lower()

    def test_mss_not_confirmed_signal_type_none(self):
        df_mtf = self._make_mtf_df()
        s = MTFSMCStrategy(
            df_htf=_make_htf(), df_mtf=df_mtf,
            killzone_only=False, swing_length=10, internal_length=5,
        )
        df_ltf = self._ltf_with_ts()
        with patch.object(s, "_htf_analysis", return_value=self._mock_htf(BEARISH)), \
             patch.object(s, "_mtf_mss_confirmed", return_value=False):
            sig = s.generate_signal(df_ltf, bar_index=len(df_ltf) - 1)
        assert sig.type == SignalType.NONE
        assert sig.confidence == pytest.approx(0.0)

    def test_last_mtf_bar_returns_valid_index(self):
        df_mtf = self._make_mtf_df(n=100)
        s = MTFSMCStrategy(df_htf=_make_htf(), df_mtf=df_mtf, killzone_only=False)
        ts = pd.Timestamp("2023-01-03 10:00:00")
        idx = s._last_mtf_bar(ts)
        assert 0 <= idx < len(df_mtf)

    def test_mtf_mss_confirmed_none_df_returns_true(self):
        s = MTFSMCStrategy(df_htf=_make_htf(), df_mtf=None, killzone_only=False)
        ts = pd.Timestamp("2023-01-03 10:00:00")
        assert s._mtf_mss_confirmed(ts, BULLISH) is True
        assert s._mtf_mss_confirmed(ts, BEARISH) is True


# ──────────────────────────────────────────────────────────────────────────────
# 5M LTF entry trigger (bias + FVG)
# ──────────────────────────────────────────────────────────────────────────────

class TestEntryFVGTrigger:
    """
    Verify _ltf_entry_confirmed() behaviour:
    - With require_entry_fvg=False: only LTF bias matters.
    - With require_entry_fvg=True (default): both bias AND 5M FVG required.
    """

    def _make_ltf(self, n: int = 50, seed: int = 11) -> pd.DataFrame:
        return _synthetic_ohlcv(n=n, seed=seed, start="2023-01-03", freq="5min")

    def test_fvg_not_required_bias_match_returns_true(self):
        """When require_entry_fvg=False, matching LTF bias is sufficient."""
        s = MTFSMCStrategy(
            df_htf=_make_htf(), killzone_only=False,
            swing_length=10, internal_length=5, atr_period=30,
            require_entry_fvg=False,
        )
        df = self._make_ltf()
        bar = len(df) - 1
        # Run real analysis — we only assert that if internal_bias matches it returns True.
        # Use mock to force bias match.
        with patch("zeus.strategy.mtf_strategy.analyze") as mock_analyze:
            mock_result = MagicMock()
            mock_result.internal_bias = BULLISH
            mock_analyze.return_value = mock_result
            result = s._ltf_entry_confirmed(df, bar, BULLISH, float(df["close"].iloc[bar]))
        assert result is True

    def test_fvg_not_required_bias_mismatch_returns_false(self):
        s = MTFSMCStrategy(
            df_htf=_make_htf(), killzone_only=False,
            swing_length=10, internal_length=5, atr_period=30,
            require_entry_fvg=False,
        )
        df = self._make_ltf()
        bar = len(df) - 1
        with patch("zeus.strategy.mtf_strategy.analyze") as mock_analyze:
            mock_result = MagicMock()
            mock_result.internal_bias = BEARISH
            mock_analyze.return_value = mock_result
            result = s._ltf_entry_confirmed(df, bar, BULLISH, float(df["close"].iloc[bar]))
        assert result is False

    def test_fvg_required_no_active_fvg_returns_false(self):
        """When FVG is required but no active FVG exists, entry is rejected."""
        s = MTFSMCStrategy(
            df_htf=_make_htf(), killzone_only=False,
            swing_length=10, internal_length=5, atr_period=30,
            require_entry_fvg=True,
        )
        df = self._make_ltf()
        bar = len(df) - 1
        with patch("zeus.strategy.mtf_strategy.analyze") as mock_analyze, \
             patch("zeus.strategy.mtf_strategy.get_active_fvgs", return_value=[]):
            mock_result = MagicMock()
            mock_result.internal_bias = BULLISH
            mock_analyze.return_value = mock_result
            result = s._ltf_entry_confirmed(df, bar, BULLISH, float(df["close"].iloc[bar]))
        assert result is False

    def test_fvg_required_price_in_fvg_returns_true(self):
        """Price inside an active aligned FVG → entry confirmed."""
        s = MTFSMCStrategy(
            df_htf=_make_htf(), killzone_only=False,
            swing_length=10, internal_length=5, atr_period=30,
            require_entry_fvg=True,
        )
        df = self._make_ltf()
        bar = len(df) - 1
        price = float(df["close"].iloc[bar])

        mock_fvg = MagicMock()
        mock_fvg.direction = BULLISH
        mock_fvg.bottom    = price - 2.0
        mock_fvg.top       = price + 2.0

        with patch("zeus.strategy.mtf_strategy.analyze") as mock_analyze, \
             patch("zeus.strategy.mtf_strategy.get_active_fvgs", return_value=[mock_fvg]):
            mock_result = MagicMock()
            mock_result.internal_bias = BULLISH
            mock_analyze.return_value = mock_result
            result = s._ltf_entry_confirmed(df, bar, BULLISH, price)
        assert result is True

    def test_fvg_required_price_outside_fvg_returns_false(self):
        """FVG exists but price is outside it → entry rejected."""
        s = MTFSMCStrategy(
            df_htf=_make_htf(), killzone_only=False,
            swing_length=10, internal_length=5, atr_period=30,
            require_entry_fvg=True,
        )
        df = self._make_ltf()
        bar = len(df) - 1
        price = float(df["close"].iloc[bar])

        mock_fvg = MagicMock()
        mock_fvg.direction = BULLISH
        mock_fvg.bottom    = price + 50.0   # far above price
        mock_fvg.top       = price + 100.0

        with patch("zeus.strategy.mtf_strategy.analyze") as mock_analyze, \
             patch("zeus.strategy.mtf_strategy.get_active_fvgs", return_value=[mock_fvg]):
            mock_result = MagicMock()
            mock_result.internal_bias = BULLISH
            mock_analyze.return_value = mock_result
            result = s._ltf_entry_confirmed(df, bar, BULLISH, price)
        assert result is False

    def test_fvg_wrong_direction_ignored(self):
        """A bearish FVG doesn't satisfy a bullish entry requirement."""
        s = MTFSMCStrategy(
            df_htf=_make_htf(), killzone_only=False,
            swing_length=10, internal_length=5, atr_period=30,
            require_entry_fvg=True,
        )
        df = self._make_ltf()
        bar = len(df) - 1
        price = float(df["close"].iloc[bar])

        mock_fvg = MagicMock()
        mock_fvg.direction = BEARISH     # wrong direction
        mock_fvg.bottom    = price - 2.0
        mock_fvg.top       = price + 2.0

        with patch("zeus.strategy.mtf_strategy.analyze") as mock_analyze, \
             patch("zeus.strategy.mtf_strategy.get_active_fvgs", return_value=[mock_fvg]):
            mock_result = MagicMock()
            mock_result.internal_bias = BULLISH
            mock_analyze.return_value = mock_result
            result = s._ltf_entry_confirmed(df, bar, BULLISH, price)
        assert result is False

    def test_insufficient_window_returns_false(self):
        """Window smaller than internal_length*2+1 immediately returns False."""
        s = MTFSMCStrategy(
            df_htf=_make_htf(), killzone_only=False,
            swing_length=10, internal_length=5, atr_period=30,
            ltf_lookback=3,   # tiny window
        )
        df = self._make_ltf(n=5)
        result = s._ltf_entry_confirmed(df, 2, BULLISH, 2000.0)
        assert result is False

    def test_generate_signal_uses_ltf_entry_confirmed(self):
        """generate_signal() reaches _ltf_entry_confirmed (not old _ltf_confirms)."""
        s = MTFSMCStrategy(
            df_htf=_make_htf(), killzone_only=False,
            swing_length=10, internal_length=5,
        )
        df_ltf = _synthetic_ohlcv(n=200, seed=30, start="2023-01-02", freq="5min")
        mock_htf = MagicMock()
        mock_htf.swing_bias    = BULLISH
        mock_htf.internal_bias = BULLISH
        # Patch _last_htf_bar so the pipeline passes the HTF-data guard
        with patch.object(s, "_last_htf_bar", return_value=150), \
             patch.object(s, "_htf_analysis", return_value=mock_htf), \
             patch.object(s, "_in_htf_zone", return_value="FVG"), \
             patch("zeus.strategy.mtf_strategy.best_confluence") as mock_bc, \
             patch.object(s, "_ltf_entry_confirmed", return_value=False) as mock_ltf:
            mock_cs = MagicMock()
            mock_cs.active_count = 5
            mock_cs.confidence   = 0.5
            mock_cs.grade.return_value = MagicMock(value="C")
            mock_bc.return_value = mock_cs
            sig = s.generate_signal(df_ltf, bar_index=len(df_ltf) - 1)
        mock_ltf.assert_called_once()
        assert sig.type == SignalType.NONE
        assert "ltf entry" in sig.reason.lower()

# ──────────────────────────────────────────────────────────────────────────────
# Asian range sweep gate (step 3.7)
# ──────────────────────────────────────────────────────────────────────────────

class TestAsianSweepGate:
    """
    Verify that the Asian range sweep gate (require_asian_sweep=True) correctly
    rejects setups where the Asian extreme has not been taken out.

    All tests use killzone_only=False and mock HTF analysis to isolate this gate.
    """

    _ASIAN_HIGH = 2010.0
    _ASIAN_LOW  = 1990.0

    def _make_ltf_with_asian(
        self,
        post_highs: list[float],
        post_lows:  list[float],
        date: str = "2024-01-15",
    ) -> pd.DataFrame:
        """
        Hourly DataFrame: 8 Asian bars (00-07 UTC) + post-Asian bars from 08:00.
        post_highs/post_lows must have the same length.
        """
        asian_h = [self._ASIAN_HIGH] * 8
        asian_l = [self._ASIAN_LOW]  * 8
        highs = asian_h + post_highs
        lows  = asian_l + post_lows
        idx   = pd.date_range(f"{date} 00:00", periods=len(highs), freq="1h")
        closes = [(h + lo) / 2 for h, lo in zip(highs, lows)]
        return pd.DataFrame(
            {"open": closes, "high": highs, "low": lows,
             "close": closes, "volume": [100.0] * len(highs)},
            index=idx,
        )

    def _make_htf_aligned(self) -> pd.DataFrame:
        """300-bar 4H HTF covering well before 2024-01-15."""
        return _synthetic_ohlcv(n=300, seed=1, start="2022-01-01", freq="4h")

    def _mock_htf(self, direction: int):
        r = MagicMock()
        r.swing_bias    = direction
        r.internal_bias = direction
        return r

    def _make_strategy(self, require_asian_sweep: bool = True) -> MTFSMCStrategy:
        return MTFSMCStrategy(
            df_htf=self._make_htf_aligned(),
            killzone_only=False,
            swing_length=10,
            internal_length=5,
            require_asian_sweep=require_asian_sweep,
        )

    # ── Gate disabled ─────────────────────────────────────────────────────

    def test_gate_disabled_no_asian_check(self):
        """When require_asian_sweep=False the gate never fires."""
        s = self._make_strategy(require_asian_sweep=False)
        # Post-Asian: high stays below Asian high → bearish sweep never happened
        df_ltf = self._make_ltf_with_asian(
            post_highs=[2005.0] * 8,
            post_lows=[1995.0]  * 8,
        )
        bar = len(df_ltf) - 1
        with patch.object(s, "_htf_analysis", return_value=self._mock_htf(BEARISH)), \
             patch.object(s, "_in_htf_zone", return_value="OB"), \
             patch.object(s, "_ltf_entry_confirmed", return_value=True), \
             patch("zeus.strategy.mtf_strategy.best_confluence") as mock_bc:
            mock_bc.return_value = None
            sig = s.generate_signal(df_ltf, bar_index=bar)
        assert "asian range" not in sig.reason.lower()

    # ── Gate enabled — bullish ────────────────────────────────────────────

    def test_bullish_sweep_confirmed_passes_gate(self):
        """Asian LOW pierced by a post-Asian bar → gate passes."""
        s = self._make_strategy()
        # One bar dips to 1985 (below Asian low 1990) → bullish sweep
        post_h = [2005.0] * 8
        post_l = [1985.0] + [1995.0] * 7
        df_ltf = self._make_ltf_with_asian(post_highs=post_h, post_lows=post_l)
        bar = len(df_ltf) - 1
        with patch.object(s, "_htf_analysis", return_value=self._mock_htf(BULLISH)), \
             patch.object(s, "_in_htf_zone", return_value="OB"), \
             patch.object(s, "_ltf_entry_confirmed", return_value=True), \
             patch("zeus.strategy.mtf_strategy.best_confluence") as mock_bc:
            mock_bc.return_value = None
            sig = s.generate_signal(df_ltf, bar_index=bar)
        assert "asian range" not in sig.reason.lower()

    def test_bullish_no_sweep_rejects(self):
        """Asian LOW not pierced → gate rejects."""
        s = self._make_strategy()
        # All post-Asian bars stay above Asian low (1990)
        post_h = [2005.0] * 8
        post_l = [1992.0] * 8
        df_ltf = self._make_ltf_with_asian(post_highs=post_h, post_lows=post_l)
        bar = len(df_ltf) - 1
        with patch.object(s, "_htf_analysis", return_value=self._mock_htf(BULLISH)):
            sig = s.generate_signal(df_ltf, bar_index=bar)
        assert sig.type == SignalType.NONE
        assert "asian range" in sig.reason.lower()

    # ── Gate enabled — bearish ────────────────────────────────────────────

    def test_bearish_sweep_confirmed_passes_gate(self):
        """Asian HIGH pierced by a post-Asian bar → gate passes."""
        s = self._make_strategy()
        # One bar spikes to 2015 (above Asian high 2010) → bearish sweep
        post_h = [2015.0] + [2005.0] * 7
        post_l = [1995.0] * 8
        df_ltf = self._make_ltf_with_asian(post_highs=post_h, post_lows=post_l)
        bar = len(df_ltf) - 1
        with patch.object(s, "_htf_analysis", return_value=self._mock_htf(BEARISH)), \
             patch.object(s, "_in_htf_zone", return_value="OB"), \
             patch.object(s, "_ltf_entry_confirmed", return_value=True), \
             patch("zeus.strategy.mtf_strategy.best_confluence") as mock_bc:
            mock_bc.return_value = None
            sig = s.generate_signal(df_ltf, bar_index=bar)
        assert "asian range" not in sig.reason.lower()

    def test_bearish_no_sweep_rejects(self):
        """Asian HIGH not pierced → gate rejects."""
        s = self._make_strategy()
        post_h = [2008.0] * 8   # below Asian high 2010
        post_l = [1995.0] * 8
        df_ltf = self._make_ltf_with_asian(post_highs=post_h, post_lows=post_l)
        bar = len(df_ltf) - 1
        with patch.object(s, "_htf_analysis", return_value=self._mock_htf(BEARISH)):
            sig = s.generate_signal(df_ltf, bar_index=bar)
        assert sig.type == SignalType.NONE
        assert "asian range" in sig.reason.lower()

    # ── Rejection signal shape ────────────────────────────────────────────

    def test_rejection_confidence_zero(self):
        s = self._make_strategy()
        post_h = [2005.0] * 8
        post_l = [1992.0] * 8
        df_ltf = self._make_ltf_with_asian(post_highs=post_h, post_lows=post_l)
        bar = len(df_ltf) - 1
        with patch.object(s, "_htf_analysis", return_value=self._mock_htf(BULLISH)):
            sig = s.generate_signal(df_ltf, bar_index=bar)
        assert sig.confidence == pytest.approx(0.0)

    def test_rejection_bar_index_propagated(self):
        s = self._make_strategy()
        post_h = [2005.0] * 8
        post_l = [1992.0] * 8
        df_ltf = self._make_ltf_with_asian(post_highs=post_h, post_lows=post_l)
        bar = len(df_ltf) - 1
        with patch.object(s, "_htf_analysis", return_value=self._mock_htf(BULLISH)):
            sig = s.generate_signal(df_ltf, bar_index=bar)
        assert sig.bar_index == bar

# ──────────────────────────────────────────────────────────────────────────────
# Daily / session frequency gate (step 7.5)
# ──────────────────────────────────────────────────────────────────────────────

class TestDailyFrequencyLimit:
    """
    Verify that the frequency gate correctly limits signals per day and per session.

    Strategy is configured with:
      killzone_only=False        — to control session manually via signal_log
      max_daily_signals=2        — default
      max_signals_per_session=1  — default

    All quality gates (HTF, zone, confluence, LTF, SL) are bypassed via mocks
    so only the frequency gate is tested.
    """

    # ── Helpers ───────────────────────────────────────────────────────────

    def _make_strategy(self, max_daily: int = 2, max_per_session: int = 1) -> MTFSMCStrategy:
        return MTFSMCStrategy(
            df_htf=_make_htf(),
            killzone_only=False,
            swing_length=10,
            internal_length=5,
            max_daily_signals=max_daily,
            max_signals_per_session=max_per_session,
        )

    def _ltf_at(self, date: str = "2024-01-15", hour: int = 9,
                n: int = 200) -> pd.DataFrame:
        """LTF DataFrame whose last bar lands at UTC {date} {hour:02d}:00."""
        end = pd.Timestamp(f"{date} {hour:02d}:00:00")
        idx = pd.date_range(end=end, periods=n, freq="1min")
        rng = np.random.default_rng(42)
        c   = 2000.0 + np.cumsum(rng.normal(0, 1, n))
        return pd.DataFrame(
            {"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": [10.0] * n},
            index=idx,
        )

    def _mock_htf(self, direction: int = BULLISH):
        r = MagicMock()
        r.swing_bias    = direction
        r.internal_bias = direction
        return r

    def _full_pass_mocks(self, s: MTFSMCStrategy, direction: int = BULLISH):
        """Context manager that makes all quality gates return 'pass'."""
        mock_cs = MagicMock()
        mock_cs.active_count = 5
        mock_cs.confidence   = 0.5
        mock_cs.grade.return_value = MagicMock(value="C")

        return (
            patch.object(s, "_last_htf_bar", return_value=150),
            patch.object(s, "_htf_analysis", return_value=self._mock_htf(direction)),
            patch.object(s, "_in_htf_zone", return_value="OB"),
            patch.object(s, "_ltf_entry_confirmed", return_value=True),
            patch("zeus.strategy.mtf_strategy.best_confluence", return_value=mock_cs),
        )

    # ── No prior signals → always pass ────────────────────────────────────

    def test_first_signal_of_day_passes(self):
        s = self._make_strategy()
        df = self._ltf_at(hour=9)   # London KZ
        with patch.object(s, "_last_htf_bar", return_value=150), \
             patch.object(s, "_htf_analysis", return_value=self._mock_htf()), \
             patch.object(s, "_in_htf_zone", return_value="OB"), \
             patch.object(s, "_ltf_entry_confirmed", return_value=True), \
             patch("zeus.strategy.mtf_strategy.best_confluence") as mock_bc:
            mock_cs = MagicMock()
            mock_cs.active_count = 5
            mock_cs.confidence   = 0.5
            mock_cs.grade.return_value = MagicMock(value="C")
            mock_bc.return_value = mock_cs
            sig = s.generate_signal(df, bar_index=len(df) - 1)
        assert sig.type != SignalType.NONE or "limit" not in sig.reason.lower()

    # ── Session limit ──────────────────────────────────────────────────────

    def test_second_signal_same_session_rejected(self):
        """max_signals_per_session=1: second London signal on same day → rejected."""
        s = self._make_strategy(max_daily=5, max_per_session=1)
        import datetime
        s._signal_log.append((datetime.date(2024, 1, 15), "London"))
        df = self._ltf_at(date="2024-01-15", hour=9)   # London KZ
        with patch.object(s, "_last_htf_bar", return_value=150), \
             patch.object(s, "_htf_analysis", return_value=self._mock_htf()), \
             patch.object(s, "_in_htf_zone", return_value="OB"), \
             patch.object(s, "_ltf_entry_confirmed", return_value=True), \
             patch("zeus.strategy.mtf_strategy.best_confluence") as mock_bc:
            mock_cs = MagicMock()
            mock_cs.active_count = 5
            mock_cs.confidence   = 0.5
            mock_cs.grade.return_value = MagicMock(value="C")
            mock_bc.return_value = mock_cs
            sig = s.generate_signal(df, bar_index=len(df) - 1)
        assert sig.type == SignalType.NONE
        assert "london" in sig.reason.lower()
        assert "limit" in sig.reason.lower()

    def test_second_signal_different_session_passes_session_gate(self):
        """After one London signal, an NY signal is allowed (different session)."""
        s = self._make_strategy(max_daily=5, max_per_session=1)
        import datetime
        s._signal_log.append((datetime.date(2024, 1, 15), "London"))
        df = self._ltf_at(date="2024-01-15", hour=13)   # NY KZ (12-15 UTC)
        with patch.object(s, "_last_htf_bar", return_value=150), \
             patch.object(s, "_htf_analysis", return_value=self._mock_htf()), \
             patch.object(s, "_in_htf_zone", return_value="OB"), \
             patch.object(s, "_ltf_entry_confirmed", return_value=True), \
             patch("zeus.strategy.mtf_strategy.best_confluence") as mock_bc:
            mock_cs = MagicMock()
            mock_cs.active_count = 5
            mock_cs.confidence   = 0.5
            mock_cs.grade.return_value = MagicMock(value="C")
            mock_bc.return_value = mock_cs
            sig = s.generate_signal(df, bar_index=len(df) - 1)
        assert "session limit" not in sig.reason.lower()

    # ── Daily limit ────────────────────────────────────────────────────────

    def test_daily_limit_reached_rejects(self):
        """max_daily_signals=2: third signal on same day → rejected."""
        s = self._make_strategy(max_daily=2, max_per_session=5)
        import datetime
        today = datetime.date(2024, 1, 15)
        s._signal_log.append((today, "London"))
        s._signal_log.append((today, "NY"))
        df = self._ltf_at(date="2024-01-15", hour=13)
        with patch.object(s, "_last_htf_bar", return_value=150), \
             patch.object(s, "_htf_analysis", return_value=self._mock_htf()), \
             patch.object(s, "_in_htf_zone", return_value="OB"), \
             patch.object(s, "_ltf_entry_confirmed", return_value=True), \
             patch("zeus.strategy.mtf_strategy.best_confluence") as mock_bc:
            mock_cs = MagicMock()
            mock_cs.active_count = 5
            mock_cs.confidence   = 0.5
            mock_cs.grade.return_value = MagicMock(value="C")
            mock_bc.return_value = mock_cs
            sig = s.generate_signal(df, bar_index=len(df) - 1)
        assert sig.type == SignalType.NONE
        assert "daily" in sig.reason.lower()
        assert "limit" in sig.reason.lower()

    def test_daily_limit_only_counts_same_day(self):
        """Log entries from yesterday do not count toward today's limit."""
        s = self._make_strategy(max_daily=2, max_per_session=5)
        import datetime
        yesterday = datetime.date(2024, 1, 14)
        s._signal_log.append((yesterday, "London"))
        s._signal_log.append((yesterday, "NY"))
        df = self._ltf_at(date="2024-01-15", hour=9)
        with patch.object(s, "_last_htf_bar", return_value=150), \
             patch.object(s, "_htf_analysis", return_value=self._mock_htf()), \
             patch.object(s, "_in_htf_zone", return_value="OB"), \
             patch.object(s, "_ltf_entry_confirmed", return_value=True), \
             patch("zeus.strategy.mtf_strategy.best_confluence") as mock_bc:
            mock_cs = MagicMock()
            mock_cs.active_count = 5
            mock_cs.confidence   = 0.5
            mock_cs.grade.return_value = MagicMock(value="C")
            mock_bc.return_value = mock_cs
            sig = s.generate_signal(df, bar_index=len(df) - 1)
        assert "daily signal limit" not in sig.reason.lower()

    # ── Limits disabled (= 0) ─────────────────────────────────────────────

    def test_max_daily_zero_disables_daily_limit(self):
        s = self._make_strategy(max_daily=0, max_per_session=0)
        import datetime
        today = datetime.date(2024, 1, 15)
        # Pre-fill 10 signals today — daily limit disabled, should not block
        s._signal_log.extend([(today, "London")] * 10)
        df = self._ltf_at(date="2024-01-15", hour=13)
        with patch.object(s, "_last_htf_bar", return_value=150), \
             patch.object(s, "_htf_analysis", return_value=self._mock_htf()), \
             patch.object(s, "_in_htf_zone", return_value="OB"), \
             patch.object(s, "_ltf_entry_confirmed", return_value=True), \
             patch("zeus.strategy.mtf_strategy.best_confluence") as mock_bc:
            mock_cs = MagicMock()
            mock_cs.active_count = 5
            mock_cs.confidence   = 0.5
            mock_cs.grade.return_value = MagicMock(value="C")
            mock_bc.return_value = mock_cs
            sig = s.generate_signal(df, bar_index=len(df) - 1)
        assert "limit" not in sig.reason.lower()

    # ── Log grows on emission ─────────────────────────────────────────────

    def test_signal_log_grows_on_emission(self):
        """Each emitted signal is appended to _signal_log."""
        s = self._make_strategy(max_daily=5, max_per_session=5)
        df = self._ltf_at(hour=9)
        assert len(s._signal_log) == 0
        with patch.object(s, "_last_htf_bar", return_value=150), \
             patch.object(s, "_htf_analysis", return_value=self._mock_htf()), \
             patch.object(s, "_in_htf_zone", return_value="OB"), \
             patch.object(s, "_ltf_entry_confirmed", return_value=True), \
             patch("zeus.strategy.mtf_strategy.best_confluence") as mock_bc:
            mock_cs = MagicMock()
            mock_cs.active_count = 5
            mock_cs.confidence   = 0.5
            mock_cs.grade.return_value = MagicMock(value="C")
            mock_bc.return_value = mock_cs
            sig = s.generate_signal(df, bar_index=len(df) - 1)
        if sig.type != SignalType.NONE:
            assert len(s._signal_log) == 1

    # ── reset_signal_log ──────────────────────────────────────────────────

    def test_reset_clears_log(self):
        s = self._make_strategy()
        import datetime
        s._signal_log.append((datetime.date(2024, 1, 15), "London"))
        s._signal_log.append((datetime.date(2024, 1, 15), "NY"))
        s.reset_signal_log()
        assert s._signal_log == []

    def test_after_reset_first_signal_passes_gate(self):
        """After reset, the daily limit no longer blocks."""
        s = self._make_strategy(max_daily=2, max_per_session=5)
        import datetime
        today = datetime.date(2024, 1, 15)
        s._signal_log.extend([(today, "London"), (today, "NY")])
        s.reset_signal_log()
        df = self._ltf_at(date="2024-01-15", hour=9)
        with patch.object(s, "_last_htf_bar", return_value=150), \
             patch.object(s, "_htf_analysis", return_value=self._mock_htf()), \
             patch.object(s, "_in_htf_zone", return_value="OB"), \
             patch.object(s, "_ltf_entry_confirmed", return_value=True), \
             patch("zeus.strategy.mtf_strategy.best_confluence") as mock_bc:
            mock_cs = MagicMock()
            mock_cs.active_count = 5
            mock_cs.confidence   = 0.5
            mock_cs.grade.return_value = MagicMock(value="C")
            mock_bc.return_value = mock_cs
            sig = s.generate_signal(df, bar_index=len(df) - 1)
        assert "daily signal limit" not in sig.reason.lower()
