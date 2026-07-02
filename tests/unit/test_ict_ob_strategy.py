"""Unit tests for ICTObStrategy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zeus.strategy.ict_ob_strategy import ICTObStrategy, ICTSignal


def _make_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Synthetic M15 OHLCV — trending random walk with clear swing structure."""
    rng  = np.random.default_rng(seed)
    idx  = pd.date_range("2025-01-02 07:00", periods=n, freq="15min")
    # Uptrend to ensure bullish BOS events
    trend = np.linspace(1900.0, 2000.0, n)
    noise = np.cumsum(rng.normal(0, 0.3, n))
    mid   = trend + noise
    half  = np.abs(rng.normal(1.5, 0.5, n))
    opens  = mid - rng.uniform(-half * 0.3, half * 0.3, n)
    highs  = np.maximum(mid + half, opens)
    lows   = np.minimum(mid - half, opens)
    closes = mid + rng.uniform(-half * 0.3, half * 0.3, n)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=idx,
    )


class TestICTObStrategyInstantiation:
    def test_defaults(self):
        s = ICTObStrategy()
        assert s.long_only is True
        assert s.risk_reward == 2.0
        assert s.pivot_size_swing == 10

    def test_custom_params(self):
        s = ICTObStrategy(pivot_size_swing=5, risk_reward=1.5, long_only=False)
        assert s.pivot_size_swing == 5
        assert s.risk_reward == 1.5
        assert s.long_only is False


class TestICTObStrategyRun:
    def test_empty_df_returns_no_signals(self):
        s  = ICTObStrategy()
        df = pd.DataFrame(
            columns=["open", "high", "low", "close"],
            index=pd.DatetimeIndex([]),
        )
        assert s.run(df) == []

    def test_tiny_df_returns_no_signals(self):
        s  = ICTObStrategy()
        df = _make_df(n=20)
        assert s.run(df) == []

    def test_run_does_not_crash(self):
        s       = ICTObStrategy(use_h4_trend=False)
        df      = _make_df(n=500)
        signals = s.run(df)
        assert isinstance(signals, list)

    def test_long_only_has_only_long_signals(self):
        s       = ICTObStrategy(use_h4_trend=False, long_only=True)
        df      = _make_df(n=800, seed=3)
        signals = s.run(df)
        for sig in signals:
            assert sig.direction == "long"

    def test_bidirectional_may_include_short_signals(self):
        s       = ICTObStrategy(use_h4_trend=False, long_only=False)
        df      = _make_df(n=1500, seed=7)
        signals = s.run(df)
        directions = {sig.direction for sig in signals}
        # At least long signals must appear in a trending dataset
        assert "long" in directions or len(signals) == 0

    def test_signal_fields_valid_for_long(self):
        s       = ICTObStrategy(use_h4_trend=False, long_only=True)
        df      = _make_df(n=1000, seed=11)
        signals = s.run(df)
        for sig in signals:
            assert isinstance(sig, ICTSignal)
            assert sig.direction == "long"
            assert sig.stop_loss < sig.entry_price
            assert sig.take_profit > sig.entry_price
            assert sig.ob_low <= sig.ob_high
            assert 0 <= sig.bar_index < len(df)

    def test_signal_fields_valid_for_short(self):
        s       = ICTObStrategy(use_h4_trend=False, long_only=False)
        df      = _make_df(n=1000, seed=17)
        signals = [sig for sig in s.run(df) if sig.direction == "short"]
        for sig in signals:
            assert sig.stop_loss > sig.entry_price
            assert sig.take_profit < sig.entry_price

    def test_cooldown_enforced(self):
        s       = ICTObStrategy(use_h4_trend=False, signal_cooldown=12)
        df      = _make_df(n=1000, seed=5)
        signals = s.run(df)
        for a, b in zip(signals, signals[1:]):
            assert b.bar_index - a.bar_index >= 12

    def test_no_duplicate_bar_indices(self):
        s       = ICTObStrategy(use_h4_trend=False)
        df      = _make_df(n=1000, seed=9)
        signals = s.run(df)
        bars    = [sig.bar_index for sig in signals]
        assert len(bars) == len(set(bars))

    def test_risk_reward_respected(self):
        for rr in (1.0, 1.5, 2.0):
            s       = ICTObStrategy(use_h4_trend=False, risk_reward=rr)
            df      = _make_df(n=1000, seed=99)
            signals = s.run(df)
            for sig in signals:
                assert sig.risk_reward == pytest.approx(rr)

    def test_structure_filter_bos_only(self):
        s       = ICTObStrategy(use_h4_trend=False, structure_filter="bos_only")
        df      = _make_df(n=800)
        signals = s.run(df)
        for sig in signals:
            assert sig.structure_type == "BOS"
