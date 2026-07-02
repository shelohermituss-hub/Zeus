"""Unit tests for HarmonicStrategy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zeus.strategy.harmonic_strategy import HarmonicSignal, HarmonicStrategy


def _make_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Synthetic M15 OHLCV — random walk, long enough for pivot detection."""
    rng  = np.random.default_rng(seed)
    idx  = pd.date_range("2025-01-01 00:00", periods=n, freq="15min")
    mid  = 1900.0 + np.cumsum(rng.normal(0, 0.5, n))
    half = np.abs(rng.normal(1.0, 0.3, n))
    opens  = mid - rng.uniform(-half * 0.5, half * 0.5, n)
    highs  = np.maximum(mid + half, opens)
    lows   = np.minimum(mid - half, opens)
    closes = mid + rng.uniform(-half * 0.5, half * 0.5, n)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=idx,
    )


class TestHarmonicStrategyInstantiation:
    def test_defaults(self):
        s = HarmonicStrategy()
        assert s.pivot_size == 5
        assert s.risk_reward == 2.0
        assert s.long_only is True

    def test_custom_params(self):
        s = HarmonicStrategy(pivot_size=7, risk_reward=1.5, use_h4_trend=False)
        assert s.pivot_size == 7
        assert s.risk_reward == 1.5
        assert s.use_h4_trend is False


class TestHarmonicStrategyRun:
    def test_empty_dataframe_returns_no_signals(self):
        s   = HarmonicStrategy()
        df  = pd.DataFrame(
            columns=["open", "high", "low", "close"],
            index=pd.DatetimeIndex([]),
        )
        assert s.run(df) == []

    def test_very_short_dataframe_returns_no_signals(self):
        s  = HarmonicStrategy()
        df = _make_df(n=10)
        assert s.run(df) == []

    def test_run_does_not_crash_on_synthetic_data(self):
        s       = HarmonicStrategy(use_h4_trend=False)
        df      = _make_df(n=300)
        signals = s.run(df)
        assert isinstance(signals, list)

    def test_signals_have_correct_fields(self):
        s       = HarmonicStrategy(use_h4_trend=False)
        df      = _make_df(n=1000, seed=7)
        signals = s.run(df)
        for sig in signals:
            assert isinstance(sig, HarmonicSignal)
            assert sig.direction == "long"
            assert sig.stop_loss < sig.entry_price
            assert sig.take_profit > sig.entry_price
            assert sig.risk_reward == pytest.approx(s.risk_reward)
            assert sig.prz_low <= sig.prz_high
            assert 0 <= sig.bar_index < len(df)

    def test_no_duplicate_bar_indices(self):
        s       = HarmonicStrategy(use_h4_trend=False)
        df      = _make_df(n=1000, seed=11)
        signals = s.run(df)
        bars    = [sig.bar_index for sig in signals]
        assert len(bars) == len(set(bars)), "multiple signals on same bar"

    def test_signals_chronologically_ordered(self):
        s       = HarmonicStrategy(use_h4_trend=False)
        df      = _make_df(n=1000, seed=3)
        signals = s.run(df)
        bars    = [sig.bar_index for sig in signals]
        assert bars == sorted(bars)

    def test_cooldown_enforced(self):
        s       = HarmonicStrategy(use_h4_trend=False, signal_cooldown=12)
        df      = _make_df(n=1000, seed=5)
        signals = s.run(df)
        for a, b in zip(signals, signals[1:]):
            assert b.bar_index - a.bar_index >= 12, (
                f"cooldown violated: {a.bar_index} → {b.bar_index}"
            )

    def test_risk_reward_parameter_respected(self):
        for rr in (1.0, 1.5, 2.0):
            s       = HarmonicStrategy(use_h4_trend=False, risk_reward=rr)
            df      = _make_df(n=1000, seed=99)
            signals = s.run(df)
            for sig in signals:
                assert sig.risk_reward == pytest.approx(rr)
