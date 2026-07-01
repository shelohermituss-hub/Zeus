"""
Unit tests for zeus.strategy.smc_strategy.SMCStrategy.

Test classes
------------
TestSMCStrategyInit          — parameter storage, _min_bars
TestSignalFromScore          — static conversion: ConfluenceScore → Signal
TestGenerateSignalGuard      — insufficient-data guard
TestGenerateSignalMocked     — full generate_signal() with analyze / best_confluence mocked
TestGenerateSignalEndToEnd   — real pipeline smoke test (no mocks, synthetic OHLCV)
"""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from zeus.strategy.base import Signal, SignalType
from zeus.strategy.confluence import ConfluenceScore
from zeus.strategy.smc.pivot import BEARISH, BULLISH
from zeus.strategy.smc_strategy import SMCStrategy
from zeus.strategy.confluence import FactorResult


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_score(direction: int, active_ids: list[int], bar_index: int = 5) -> ConfluenceScore:
    """Build a ConfluenceScore with the given factor IDs active."""
    factors = [
        FactorResult(i + 1, f"Factor{i+1}", (i + 1) in active_ids)
        for i in range(10)
    ]
    return ConfluenceScore(direction=direction, price=100.0, bar_index=bar_index, factors=factors)


def _synthetic_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Deterministic OHLCV DataFrame with n bars."""
    rng    = np.random.default_rng(seed)
    closes = 50_000.0 + np.cumsum(rng.normal(0, 50, n))
    highs  = closes + rng.uniform(10, 100, n)
    lows   = closes - rng.uniform(10, 100, n)
    opens  = np.roll(closes, 1)
    opens[0] = closes[0]
    vols   = rng.uniform(1, 100, n)
    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes, "volume": vols})


# ──────────────────────────────────────────────────────────────────────────────
# Initialisation
# ──────────────────────────────────────────────────────────────────────────────

class TestSMCStrategyInit:
    def test_default_swing_length(self):
        s = SMCStrategy()
        assert s.swing_length == 50

    def test_default_internal_length(self):
        s = SMCStrategy()
        assert s.internal_length == 5

    def test_default_min_score(self):
        s = SMCStrategy()
        assert s.min_score == pytest.approx(4.0)

    def test_default_sweep_lookback(self):
        s = SMCStrategy()
        assert s.sweep_lookback == 10

    def test_default_tolerances(self):
        s = SMCStrategy()
        assert s.poc_tolerance_pct    == pytest.approx(0.003)
        assert s.fib_50_tolerance_pct == pytest.approx(0.003)

    def test_default_volume_profile_params(self):
        s = SMCStrategy()
        assert s.vol_num_bins       == 100
        assert s.vol_value_area_pct == pytest.approx(0.70)

    def test_min_bars_uses_swing_length(self):
        # swing_length=50, internal_length=5 → min_bars = max(50,5)*2 = 100
        s = SMCStrategy(swing_length=50, internal_length=5)
        assert s._min_bars == 100

    def test_min_bars_uses_internal_length_when_larger(self):
        # swing_length=10, internal_length=30 → min_bars = max(10,30)*2 = 60
        s = SMCStrategy(swing_length=10, internal_length=30)
        assert s._min_bars == 60

    def test_custom_parameters_stored(self):
        s = SMCStrategy(min_score=6.0, sweep_lookback=5, vol_num_bins=50)
        assert s.min_score      == pytest.approx(6.0)
        assert s.sweep_lookback == 5
        assert s.vol_num_bins   == 50

    def test_is_strategy_subclass(self):
        from zeus.strategy.base import Strategy
        assert isinstance(SMCStrategy(), Strategy)


# ──────────────────────────────────────────────────────────────────────────────
# _signal_from_score static method
# ──────────────────────────────────────────────────────────────────────────────

class TestSignalFromScore:
    def test_bullish_score_returns_long(self):
        cs = _make_score(BULLISH, [1, 2, 3, 4, 5, 6, 7, 8])
        sig = SMCStrategy._signal_from_score(cs, bar_index=5)
        assert sig.type == SignalType.LONG

    def test_bearish_score_returns_short(self):
        cs = _make_score(BEARISH, [1, 2, 3, 4, 5, 6, 7, 8])
        sig = SMCStrategy._signal_from_score(cs, bar_index=5)
        assert sig.type == SignalType.SHORT

    def test_confidence_equals_cs_confidence(self):
        # active_ids has 7 items → confidence = 7/10 = 0.7
        cs = _make_score(BULLISH, [1, 2, 3, 4, 5, 6, 7])
        sig = SMCStrategy._signal_from_score(cs, bar_index=5)
        assert sig.confidence == pytest.approx(0.7)

    def test_confidence_full(self):
        cs = _make_score(BULLISH, list(range(1, 11)))
        sig = SMCStrategy._signal_from_score(cs, bar_index=5)
        assert sig.confidence == pytest.approx(1.0)

    def test_bar_index_propagated(self):
        cs = _make_score(BULLISH, [1, 8])
        sig = SMCStrategy._signal_from_score(cs, bar_index=42)
        assert sig.bar_index == 42

    def test_reason_starts_with_direction_bullish(self):
        cs = _make_score(BULLISH, [1, 8])
        sig = SMCStrategy._signal_from_score(cs, bar_index=0)
        assert sig.reason.startswith("BULLISH")

    def test_reason_starts_with_direction_bearish(self):
        cs = _make_score(BEARISH, [1, 8])
        sig = SMCStrategy._signal_from_score(cs, bar_index=0)
        assert sig.reason.startswith("BEARISH")

    def test_reason_contains_score_fraction(self):
        cs = _make_score(BULLISH, [1, 2, 3, 4, 5])
        sig = SMCStrategy._signal_from_score(cs, bar_index=0)
        assert "5/10" in sig.reason

    def test_reason_includes_active_factor_names(self):
        cs = _make_score(BULLISH, [1, 8])
        sig = SMCStrategy._signal_from_score(cs, bar_index=0)
        assert "Factor1"  in sig.reason
        assert "Factor8"  in sig.reason

    def test_reason_excludes_inactive_factor_names(self):
        # Only factors 1 and 8 are active
        cs = _make_score(BULLISH, [1, 8])
        sig = SMCStrategy._signal_from_score(cs, bar_index=0)
        assert "Factor2"  not in sig.reason
        assert "Factor10" not in sig.reason

    def test_reason_is_string(self):
        cs = _make_score(BULLISH, [1, 8])
        sig = SMCStrategy._signal_from_score(cs, bar_index=0)
        assert isinstance(sig.reason, str)

    def test_reason_includes_grade_c_at_default_min_score(self):
        cs = _make_score(BULLISH, [1, 8, 2, 3])  # score=4
        sig = SMCStrategy._signal_from_score(cs, bar_index=0)
        assert "grade=C" in sig.reason

    def test_reason_includes_grade_a_at_full_score(self):
        cs = _make_score(BULLISH, list(range(1, 11)))  # score=10
        sig = SMCStrategy._signal_from_score(cs, bar_index=0)
        assert "grade=A" in sig.reason

    def test_reason_grade_uses_passed_min_score(self):
        cs = _make_score(BULLISH, [1, 8, 2])  # score=3
        sig = SMCStrategy._signal_from_score(cs, bar_index=0, min_score=3.0)
        assert "grade=C" in sig.reason


# ──────────────────────────────────────────────────────────────────────────────
# generate_signal — insufficient-data guard
# ──────────────────────────────────────────────────────────────────────────────

class TestGenerateSignalGuard:
    def test_returns_none_signal_when_too_few_bars(self):
        s  = SMCStrategy(swing_length=50, internal_length=5)
        df = _synthetic_df(n=10)
        sig = s.generate_signal(df, bar_index=9)
        assert sig.type == SignalType.NONE

    def test_reason_mentions_insufficient_data(self):
        s  = SMCStrategy()
        df = _synthetic_df(n=5)
        sig = s.generate_signal(df, bar_index=4)
        assert "insufficient" in sig.reason.lower()

    def test_confidence_is_zero_when_insufficient(self):
        s  = SMCStrategy()
        df = _synthetic_df(n=5)
        sig = s.generate_signal(df, bar_index=4)
        assert sig.confidence == pytest.approx(0.0)

    def test_bar_index_propagated_when_insufficient(self):
        s  = SMCStrategy()
        df = _synthetic_df(n=5)
        sig = s.generate_signal(df, bar_index=4)
        assert sig.bar_index == 4

    def test_exactly_at_min_bars_does_not_guard(self):
        """At the boundary bar itself, the guard must NOT trigger."""
        s  = SMCStrategy(swing_length=10, internal_length=5)  # min_bars=20
        df = _synthetic_df(n=100)
        # bar_index=19 → window has 20 rows = min_bars → should proceed to analyze
        with patch("zeus.strategy.smc_strategy.analyze") as mock_analyze, \
             patch("zeus.strategy.smc_strategy.best_confluence") as mock_bc:
            mock_analyze.return_value = MagicMock()
            mock_bc.return_value = None
            sig = s.generate_signal(df, bar_index=19)
            assert mock_analyze.called
            assert sig.type == SignalType.NONE   # no confluent signal from mock

    def test_one_below_min_bars_triggers_guard(self):
        s  = SMCStrategy(swing_length=10, internal_length=5)  # min_bars=20
        df = _synthetic_df(n=100)
        # bar_index=18 → window has 19 rows < 20
        with patch("zeus.strategy.smc_strategy.analyze") as mock_analyze:
            sig = s.generate_signal(df, bar_index=18)
            assert not mock_analyze.called
            assert sig.type == SignalType.NONE


# ──────────────────────────────────────────────────────────────────────────────
# generate_signal — mocked pipeline
# ──────────────────────────────────────────────────────────────────────────────

class TestGenerateSignalMocked:
    """Patch analyze() and best_confluence() so tests are fast and deterministic."""

    @pytest.fixture
    def strategy(self):
        return SMCStrategy(
            swing_length=10,
            internal_length=5,
            min_score=4.0,
            sweep_lookback=8,
            poc_tolerance_pct=0.005,
            fib_50_tolerance_pct=0.006,
            vol_num_bins=50,
            vol_value_area_pct=0.80,
        )

    @pytest.fixture
    def df(self):
        return _synthetic_df(n=100)

    def test_returns_none_when_best_confluence_returns_none(self, strategy, df):
        with patch("zeus.strategy.smc_strategy.analyze") as mock_a, \
             patch("zeus.strategy.smc_strategy.best_confluence") as mock_bc:
            mock_a.return_value  = MagicMock()
            mock_bc.return_value = None
            sig = strategy.generate_signal(df, bar_index=50)
            assert sig.type == SignalType.NONE
            assert sig.confidence == pytest.approx(0.0)
            assert "no confluent signal" in sig.reason

    def test_returns_long_on_bullish_score(self, strategy, df):
        cs = _make_score(BULLISH, [1, 2, 3, 4, 5, 6, 7, 8])
        with patch("zeus.strategy.smc_strategy.analyze") as mock_a, \
             patch("zeus.strategy.smc_strategy.best_confluence", return_value=cs):
            mock_a.return_value = MagicMock()
            sig = strategy.generate_signal(df, bar_index=50)
            assert sig.type == SignalType.LONG

    def test_returns_short_on_bearish_score(self, strategy, df):
        cs = _make_score(BEARISH, [1, 2, 3, 4, 5, 6, 7, 8])
        with patch("zeus.strategy.smc_strategy.analyze") as mock_a, \
             patch("zeus.strategy.smc_strategy.best_confluence", return_value=cs):
            mock_a.return_value = MagicMock()
            sig = strategy.generate_signal(df, bar_index=50)
            assert sig.type == SignalType.SHORT

    def test_confidence_derived_from_score(self, strategy, df):
        cs = _make_score(BULLISH, [1, 2, 3, 4, 5, 6])   # 6/10 = 0.6
        with patch("zeus.strategy.smc_strategy.analyze") as mock_a, \
             patch("zeus.strategy.smc_strategy.best_confluence", return_value=cs):
            mock_a.return_value = MagicMock()
            sig = strategy.generate_signal(df, bar_index=50)
            assert sig.confidence == pytest.approx(0.6)

    def test_bar_index_in_signal(self, strategy, df):
        cs = _make_score(BULLISH, [1, 8, 2, 3, 4])
        with patch("zeus.strategy.smc_strategy.analyze") as mock_a, \
             patch("zeus.strategy.smc_strategy.best_confluence", return_value=cs):
            mock_a.return_value = MagicMock()
            sig = strategy.generate_signal(df, bar_index=77)
            assert sig.bar_index == 77

    def test_analyze_called_with_correct_params(self, strategy, df):
        with patch("zeus.strategy.smc_strategy.analyze") as mock_a, \
             patch("zeus.strategy.smc_strategy.best_confluence", return_value=None):
            mock_a.return_value = MagicMock()
            strategy.generate_signal(df, bar_index=50)

            args, kwargs = mock_a.call_args
            assert kwargs.get("swing_length",    args[1] if len(args) > 1 else None) == 10
            assert kwargs.get("internal_length", None)  == 5
            assert kwargs.get("vol_num_bins",    None)  == 50
            assert kwargs.get("vol_value_area_pct", None) == pytest.approx(0.80)

    def test_best_confluence_called_with_correct_params(self, strategy, df):
        dummy_result = MagicMock()
        with patch("zeus.strategy.smc_strategy.analyze", return_value=dummy_result), \
             patch("zeus.strategy.smc_strategy.best_confluence") as mock_bc:
            mock_bc.return_value = None
            strategy.generate_signal(df, bar_index=50)

            _, kwargs = mock_bc.call_args
            assert kwargs.get("min_score",           None) == pytest.approx(4.0)
            assert kwargs.get("poc_tolerance_pct",   None) == pytest.approx(0.005)
            assert kwargs.get("fib_50_tolerance_pct", None) == pytest.approx(0.006)
            assert kwargs.get("sweep_lookback",      None) == 8
            assert "timestamp" in kwargs

    def test_analyze_window_is_sliced_to_bar_index(self, strategy, df):
        """analyze() must only see bars up to and including bar_index."""
        captured_windows: list[pd.DataFrame] = []

        def capture_analyze(window, **_kw):
            captured_windows.append(window)
            return MagicMock()

        with patch("zeus.strategy.smc_strategy.analyze", side_effect=capture_analyze), \
             patch("zeus.strategy.smc_strategy.best_confluence", return_value=None):
            strategy.generate_signal(df, bar_index=40)

        assert len(captured_windows) == 1
        assert len(captured_windows[0]) == 41   # bars 0–40 inclusive

    def test_price_from_close_at_bar_index(self, strategy, df):
        """best_confluence receives the close price at bar_index."""
        expected_price = float(df["close"].iloc[40])
        captured_prices: list[float] = []

        def capture_bc(result, price, bar_index, **_kw):
            captured_prices.append(price)
            return None

        with patch("zeus.strategy.smc_strategy.analyze", return_value=MagicMock()), \
             patch("zeus.strategy.smc_strategy.best_confluence", side_effect=capture_bc):
            strategy.generate_signal(df, bar_index=40)

        assert len(captured_prices) == 1
        assert captured_prices[0] == pytest.approx(expected_price)


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end smoke test (real pipeline, no mocks)
# ──────────────────────────────────────────────────────────────────────────────

class TestGenerateSignalEndToEnd:
    """
    Run the real SMC pipeline on a synthetic DataFrame.

    No assertion on signal direction — the synthetic data is not designed to
    produce a specific confluent signal.  The tests verify:
    - No exception is raised.
    - The returned Signal is well-formed.
    - The signal type is one of LONG / SHORT / NONE.
    """

    @pytest.fixture
    def strategy(self):
        # Short lookbacks so the test runs fast
        return SMCStrategy(swing_length=10, internal_length=3, atr_period=20)

    @pytest.fixture
    def df(self):
        return _synthetic_df(n=200, seed=0)

    def test_no_exception_at_last_bar(self, strategy, df):
        sig = strategy.generate_signal(df, bar_index=len(df) - 1)
        assert sig is not None

    def test_returns_signal_instance(self, strategy, df):
        sig = strategy.generate_signal(df, bar_index=len(df) - 1)
        assert isinstance(sig, Signal)

    def test_signal_type_is_valid(self, strategy, df):
        sig = strategy.generate_signal(df, bar_index=len(df) - 1)
        assert sig.type in (SignalType.LONG, SignalType.SHORT, SignalType.NONE)

    def test_confidence_in_unit_interval(self, strategy, df):
        sig = strategy.generate_signal(df, bar_index=len(df) - 1)
        assert 0.0 <= sig.confidence <= 1.0

    def test_bar_index_correct(self, strategy, df):
        bar = len(df) - 1
        sig = strategy.generate_signal(df, bar_index=bar)
        assert sig.bar_index == bar

    def test_reason_is_nonempty_string(self, strategy, df):
        sig = strategy.generate_signal(df, bar_index=len(df) - 1)
        assert isinstance(sig.reason, str)
        assert len(sig.reason) > 0

    def test_multiple_bars_produce_signals_without_crash(self, strategy, df):
        """Sweep from the first eligible bar to the end."""
        min_bar = strategy._min_bars - 1
        signals = [
            strategy.generate_signal(df, bar_index=i)
            for i in range(min_bar, min(min_bar + 20, len(df)))
        ]
        assert all(isinstance(s, Signal) for s in signals)
        assert all(0.0 <= s.confidence <= 1.0 for s in signals)

    def test_with_datetime_index(self, strategy):
        """DatetimeIndex activates session detection (factor 7) without error."""
        idx = pd.date_range("2024-01-01", periods=200, freq="1h")
        df  = _synthetic_df(n=200, seed=1)
        df.index = idx
        sig = strategy.generate_signal(df, bar_index=199)
        assert isinstance(sig, Signal)

    def test_without_volume_column(self, strategy):
        """Strategy must work with OHLC-only data (volume defaults to ones)."""
        df = _synthetic_df(n=200).drop(columns=["volume"])
        sig = strategy.generate_signal(df, bar_index=199)
        assert isinstance(sig, Signal)


# ──────────────────────────────────────────────────────────────────────────────
# Daily bias gate (Recommendation 2)
# ──────────────────────────────────────────────────────────────────────────────

def _make_daily_df(open_: float, close: float, date: str = "2024-01-01") -> pd.DataFrame:
    """Single-row daily OHLCV DataFrame."""
    idx = pd.DatetimeIndex([pd.Timestamp(date)])
    return pd.DataFrame(
        {"open": [open_], "high": [max(open_, close)],
         "low": [min(open_, close)], "close": [close]},
        index=idx,
    )


class TestDailyBiasGate:
    """
    Verify that df_daily=... filters conflicting signals.

    All tests patch best_confluence() to control the signal direction;
    only the daily bias gate logic is under test here.
    """

    @pytest.fixture
    def df_ltf(self):
        """LTF DataFrame with a DatetimeIndex so timestamp is available."""
        idx = pd.date_range("2024-01-02 00:00", periods=200, freq="1h")
        base = _synthetic_df(n=200, seed=7)
        base.index = idx
        return base

    def test_no_df_daily_never_filters(self, df_ltf):
        """Without df_daily the gate is disabled — signal passes through."""
        cs = _make_score(BULLISH, [1, 2, 3, 4, 5, 8])
        s  = SMCStrategy(swing_length=10, internal_length=3, atr_period=20)
        with patch("zeus.strategy.smc_strategy.analyze") as mock_a, \
             patch("zeus.strategy.smc_strategy.best_confluence", return_value=cs):
            mock_a.return_value = MagicMock()
            sig = s.generate_signal(df_ltf, bar_index=100)
        assert sig.type == SignalType.LONG

    def test_aligned_daily_bias_passes_signal(self, df_ltf):
        """Bullish daily + bullish signal → no rejection."""
        df_daily = _make_daily_df(open_=100.0, close=110.0, date="2024-01-01")
        cs       = _make_score(BULLISH, [1, 2, 3, 4, 5, 8])
        s = SMCStrategy(swing_length=10, internal_length=3, atr_period=20,
                        df_daily=df_daily)
        with patch("zeus.strategy.smc_strategy.analyze") as mock_a, \
             patch("zeus.strategy.smc_strategy.best_confluence", return_value=cs):
            mock_a.return_value = MagicMock()
            sig = s.generate_signal(df_ltf, bar_index=100)
        assert sig.type == SignalType.LONG

    def test_conflicting_daily_bias_rejects_signal(self, df_ltf):
        """Bearish daily + bullish signal → NONE."""
        df_daily = _make_daily_df(open_=110.0, close=100.0, date="2024-01-01")
        cs       = _make_score(BULLISH, [1, 2, 3, 4, 5, 8])
        s = SMCStrategy(swing_length=10, internal_length=3, atr_period=20,
                        df_daily=df_daily)
        with patch("zeus.strategy.smc_strategy.analyze") as mock_a, \
             patch("zeus.strategy.smc_strategy.best_confluence", return_value=cs):
            mock_a.return_value = MagicMock()
            sig = s.generate_signal(df_ltf, bar_index=100)
        assert sig.type == SignalType.NONE

    def test_conflicting_reason_mentions_daily_bias(self, df_ltf):
        """Rejected signal reason mentions 'daily bias'."""
        df_daily = _make_daily_df(open_=110.0, close=100.0, date="2024-01-01")
        cs       = _make_score(BULLISH, [1, 2, 3, 4, 5, 8])
        s = SMCStrategy(swing_length=10, internal_length=3, atr_period=20,
                        df_daily=df_daily)
        with patch("zeus.strategy.smc_strategy.analyze") as mock_a, \
             patch("zeus.strategy.smc_strategy.best_confluence", return_value=cs):
            mock_a.return_value = MagicMock()
            sig = s.generate_signal(df_ltf, bar_index=100)
        assert "daily bias" in sig.reason.lower()

    def test_doji_daily_does_not_filter(self, df_ltf):
        """Neutral daily (doji, close == open) → gate inactive, signal passes."""
        df_daily = _make_daily_df(open_=100.0, close=100.0, date="2024-01-01")
        cs       = _make_score(BULLISH, [1, 2, 3, 4, 5, 8])
        s = SMCStrategy(swing_length=10, internal_length=3, atr_period=20,
                        df_daily=df_daily)
        with patch("zeus.strategy.smc_strategy.analyze") as mock_a, \
             patch("zeus.strategy.smc_strategy.best_confluence", return_value=cs):
            mock_a.return_value = MagicMock()
            sig = s.generate_signal(df_ltf, bar_index=100)
        assert sig.type == SignalType.LONG

    def test_short_signal_aligned_with_bearish_daily_passes(self, df_ltf):
        """Bearish daily + short signal → passes."""
        df_daily = _make_daily_df(open_=110.0, close=100.0, date="2024-01-01")
        cs       = _make_score(BEARISH, [1, 2, 3, 4, 5, 8])
        s = SMCStrategy(swing_length=10, internal_length=3, atr_period=20,
                        df_daily=df_daily)
        with patch("zeus.strategy.smc_strategy.analyze") as mock_a, \
             patch("zeus.strategy.smc_strategy.best_confluence", return_value=cs):
            mock_a.return_value = MagicMock()
            sig = s.generate_signal(df_ltf, bar_index=100)
        assert sig.type == SignalType.SHORT

    def test_no_datetime_index_gate_skipped(self):
        """Without DatetimeIndex timestamp is None → daily bias gate skipped."""
        df_daily = _make_daily_df(open_=110.0, close=100.0, date="2024-01-01")
        df_ltf   = _synthetic_df(n=200, seed=3)  # integer index, no timestamps
        cs       = _make_score(BULLISH, [1, 2, 3, 4, 5, 8])
        s = SMCStrategy(swing_length=10, internal_length=3, atr_period=20,
                        df_daily=df_daily)
        with patch("zeus.strategy.smc_strategy.analyze") as mock_a, \
             patch("zeus.strategy.smc_strategy.best_confluence", return_value=cs):
            mock_a.return_value = MagicMock()
            sig = s.generate_signal(df_ltf, bar_index=100)
        # Gate requires timestamp; without it the signal should pass unfiltered
        assert sig.type == SignalType.LONG

    def test_df_daily_stored_as_attribute(self):
        df_daily = _make_daily_df(100.0, 110.0)
        s = SMCStrategy(df_daily=df_daily)
        assert s.df_daily is df_daily

    def test_default_df_daily_is_none(self):
        assert SMCStrategy().df_daily is None
