"""
Tests for zeus/backtest/walk_forward.py.

Uses a deterministic toy momentum strategy and synthetic OHLCV data —
no real market data, network calls, or randomness are involved, so
results are fully reproducible.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from zeus.backtest.report import BacktestResult
from zeus.backtest.walk_forward import (
    InsufficientDataError,
    InvalidParamGridError,
    WalkForwardOptimizer,
    WalkForwardReport,
    WalkForwardWindow,
)
from zeus.strategy.base import Signal, SignalType, Strategy


# ======================================================================
# Helpers
# ======================================================================

class _MomentumStrategy(Strategy):
    """Deterministic toy strategy: long/short on close-to-close momentum."""

    def __init__(self, threshold: float = 0.0) -> None:
        self._threshold = threshold

    def generate_signal(self, df: pd.DataFrame, bar_index: int) -> Signal:
        if bar_index == 0:
            return Signal(SignalType.NONE, 0.0, "warmup", bar_index)
        prev = float(df["close"].iloc[bar_index - 1])
        curr = float(df["close"].iloc[bar_index])
        change = (curr - prev) / prev if prev else 0.0
        if change > self._threshold:
            return Signal(SignalType.LONG, 1.0, "momentum-up", bar_index)
        if change < -self._threshold:
            return Signal(SignalType.SHORT, 1.0, "momentum-down", bar_index)
        return Signal(SignalType.NONE, 0.0, "flat", bar_index)


def _strategy_factory(params: dict) -> Strategy:
    return _MomentumStrategy(threshold=params["threshold"])


def _make_df(n_bars: int = 100) -> pd.DataFrame:
    """Synthetic oscillating price series — gives momentum thresholds distinct outcomes."""
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="1h", tz="UTC")
    closes = [100.0 + 10.0 * math.sin(i / 5.0) + 0.3 * (i % 7) for i in range(n_bars)]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes, "volume": [1000.0] * n_bars},
        index=idx,
    )


_GRID = {"threshold": [0.0, 0.005, 0.02]}


def _optimizer(**overrides) -> WalkForwardOptimizer:
    kwargs = dict(
        strategy_factory=_strategy_factory,
        param_grid=_GRID,
        train_bars=40,
        test_bars=20,
        objective="net_pnl",
    )
    kwargs.update(overrides)
    return WalkForwardOptimizer(**kwargs)


# ======================================================================
# Construction validation
# ======================================================================

class TestConstruction:
    @pytest.mark.parametrize("train_bars,test_bars", [(0, 10), (10, 0), (-5, 10)])
    def test_rejects_non_positive_window_sizes(self, train_bars, test_bars):
        with pytest.raises(ValueError):
            _optimizer(train_bars=train_bars, test_bars=test_bars)

    def test_rejects_empty_param_grid(self):
        with pytest.raises(InvalidParamGridError):
            _optimizer(param_grid={})

    def test_rejects_param_with_no_values(self):
        with pytest.raises(InvalidParamGridError):
            _optimizer(param_grid={"threshold": []})

    def test_rejects_non_positive_step_bars(self):
        with pytest.raises(ValueError):
            _optimizer(step_bars=0)

    def test_rejects_unknown_string_objective(self):
        with pytest.raises(ValueError):
            _optimizer(objective="not_a_real_metric")

    def test_accepts_callable_objective(self):
        _optimizer(objective=lambda result: result.net_pnl)


# ======================================================================
# Fail-closed behaviour on insufficient data
# ======================================================================

class TestInsufficientData:
    def test_raises_when_fewer_bars_than_one_window(self):
        optimizer = _optimizer(train_bars=40, test_bars=20)
        df = _make_df(50)  # needs 60
        with pytest.raises(InsufficientDataError):
            optimizer.run(df)

    def test_exactly_one_window_worth_of_bars_succeeds(self):
        optimizer = _optimizer(train_bars=40, test_bars=20)
        df = _make_df(60)
        report = optimizer.run(df)
        assert len(report.windows) == 1


# ======================================================================
# Window construction
# ======================================================================

class TestWindows:
    def test_default_step_produces_non_overlapping_test_windows(self):
        optimizer = _optimizer(train_bars=40, test_bars=20)  # step defaults to test_bars
        df = _make_df(100)
        report = optimizer.run(df)

        assert len(report.windows) == 3
        expected_ranges = [(0, 40, 40, 60), (20, 60, 60, 80), (40, 80, 80, 100)]
        for window, (tr_start, tr_end, te_start, te_end) in zip(report.windows, expected_ranges):
            assert (window.train_start, window.train_end, window.test_start, window.test_end) == (
                tr_start, tr_end, te_start, te_end,
            )

    def test_custom_step_bars_changes_window_count(self):
        optimizer = _optimizer(train_bars=40, test_bars=20, step_bars=40)
        df = _make_df(100)
        report = optimizer.run(df)
        assert len(report.windows) == 2
        assert report.windows[0].train_start == 0
        assert report.windows[1].train_start == 40

    def test_best_params_come_from_grid(self):
        optimizer = _optimizer(train_bars=40, test_bars=20)
        df = _make_df(100)
        report = optimizer.run(df)
        for window in report.windows:
            assert window.best_params["threshold"] in _GRID["threshold"]

    def test_out_sample_result_is_backtest_result(self):
        optimizer = _optimizer(train_bars=40, test_bars=20)
        df = _make_df(100)
        report = optimizer.run(df)
        for window in report.windows:
            assert isinstance(window.out_sample_result, BacktestResult)


# ======================================================================
# Report aggregation
# ======================================================================

class TestReport:
    def test_combined_equity_curve_starts_at_one(self):
        optimizer = _optimizer(train_bars=40, test_bars=20)
        df = _make_df(100)
        report = optimizer.run(df)
        curve = report.combined_equity_curve()
        assert curve[0] == pytest.approx(1.0)

    def test_summary_has_expected_keys_and_window_count(self):
        optimizer = _optimizer(train_bars=40, test_bars=20)
        df = _make_df(100)
        report = optimizer.run(df)
        summary = report.summary()

        expected_keys = {
            "objective", "n_windows", "mean_in_sample_score",
            "mean_out_sample_score", "score_degradation_pct",
            "out_of_sample_total_return_pct", "windows",
        }
        assert expected_keys.issubset(summary.keys())
        assert summary["n_windows"] == 3
        assert len(summary["windows"]) == 3
        assert summary["objective"] == "net_pnl"

    def test_score_degradation_zero_when_in_sample_score_zero(self):
        empty_result = BacktestResult(trades=[], equity_curve=[1000.0], initial_balance=1000.0)
        window = WalkForwardWindow(
            window_index=0,
            train_start=0, train_end=10, test_start=10, test_end=20,
            best_params={"threshold": 0.0},
            in_sample_score=0.0,
            out_sample_score=5.0,
            out_sample_result=empty_result,
        )
        report = WalkForwardReport(windows=[window], objective="net_pnl")
        assert report.score_degradation_pct == 0.0


# ======================================================================
# Reproducibility (no randomness anywhere in this strategy/engine path)
# ======================================================================

class TestReproducibility:
    def test_identical_runs_produce_identical_reports(self):
        df = _make_df(100)
        report1 = _optimizer(train_bars=40, test_bars=20).run(df)
        report2 = _optimizer(train_bars=40, test_bars=20).run(df)

        assert report1.summary() == report2.summary()
        assert report1.combined_equity_curve() == report2.combined_equity_curve()
