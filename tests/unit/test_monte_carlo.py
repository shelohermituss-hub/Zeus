"""
Tests for zeus/backtest/monte_carlo.py.

Uses synthetic closed trades wrapped in a BacktestResult — no real market
data or network calls are required.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from zeus.backtest.monte_carlo import (
    DrawdownDistribution,
    InsufficientTradeDataError,
    MonteCarloAnalyzer,
    ReturnConfidenceInterval,
)
from zeus.backtest.report import BacktestResult
from zeus.exchange.connector import OrderResult
from zeus.orders.models import Trade, TradeStatus


# ======================================================================
# Helpers
# ======================================================================

def _order_result() -> OrderResult:
    return OrderResult(
        order_id="o1", symbol="BTC/USDT", side="buy",
        quantity=1.0, filled=1.0, price=100.0, average=100.0,
        status="closed", timestamp=datetime.now(tz=timezone.utc),
    )


def _trade(pnl: float) -> Trade:
    t = Trade(
        trade_id="t1", symbol="BTC/USDT", side="buy",
        quantity=1.0, entry_price=100.0,
        sl_price=99.0, tp_price=102.0,
        entry_order=_order_result(),
    )
    t.status = TradeStatus.CLOSED_TP if pnl >= 0 else TradeStatus.CLOSED_SL
    t.realized_pnl = pnl
    return t


def _result(pnls: list[float], initial_balance: float = 1000.0) -> BacktestResult:
    trades = [_trade(p) for p in pnls]
    return BacktestResult(
        trades=trades,
        equity_curve=[initial_balance],
        initial_balance=initial_balance,
    )


# ======================================================================
# Construction validation
# ======================================================================

class TestConstruction:
    def test_rejects_non_positive_n_simulations(self):
        with pytest.raises(ValueError):
            MonteCarloAnalyzer(n_simulations=0)

    @pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.1])
    def test_rejects_confidence_outside_open_interval(self, confidence):
        with pytest.raises(ValueError):
            MonteCarloAnalyzer(confidence=confidence)

    def test_accepts_valid_confidence_boundaries(self):
        MonteCarloAnalyzer(confidence=0.01)
        MonteCarloAnalyzer(confidence=0.99)


# ======================================================================
# Fail-closed behaviour on insufficient / invalid data
# ======================================================================

class TestFailClosed:
    def test_raises_when_fewer_trades_than_minimum(self):
        analyzer = MonteCarloAnalyzer(min_trades=20, seed=1)
        result = _result([10.0] * 5)
        with pytest.raises(InsufficientTradeDataError):
            analyzer.analyze_drawdowns(result)

    def test_raises_when_initial_balance_not_positive(self):
        analyzer = MonteCarloAnalyzer(min_trades=5, seed=1)
        result = _result([10.0] * 5, initial_balance=0.0)
        with pytest.raises(ValueError):
            analyzer.analyze_drawdowns(result)

    def test_sufficient_trades_does_not_raise(self):
        analyzer = MonteCarloAnalyzer(min_trades=20, seed=1)
        result = _result([10.0] * 20)
        analyzer.analyze_drawdowns(result)  # should not raise


# ======================================================================
# Drawdown distribution
# ======================================================================

class TestDrawdownDistribution:
    def test_returns_drawdown_distribution_with_consistent_ordering(self):
        analyzer = MonteCarloAnalyzer(n_simulations=500, min_trades=20, seed=42)
        pnls = [10.0, -8.0, 12.0, -5.0, 7.0, -15.0, 20.0, -3.0] * 3
        result = _result(pnls)
        dd = analyzer.analyze_drawdowns(result)

        assert isinstance(dd, DrawdownDistribution)
        assert dd.expected_max_drawdown_pct >= 0.0
        assert dd.median_max_drawdown_pct >= 0.0
        assert dd.worst_case_drawdown_pct >= dd.expected_max_drawdown_pct
        assert dd.worst_case_drawdown_pct >= dd.tail_drawdown_pct

    def test_all_winning_trades_have_zero_drawdown(self):
        analyzer = MonteCarloAnalyzer(n_simulations=200, min_trades=20, seed=7)
        result = _result([10.0] * 20)
        dd = analyzer.analyze_drawdowns(result)

        assert dd.expected_max_drawdown_pct == 0.0
        assert dd.worst_case_drawdown_pct == 0.0


# ======================================================================
# Probability of loss
# ======================================================================

class TestProbabilityOfLoss:
    def test_all_losing_trades_always_lose(self):
        analyzer = MonteCarloAnalyzer(n_simulations=200, min_trades=20, seed=3)
        result = _result([-5.0] * 20)
        assert analyzer.probability_of_loss(result) == 1.0

    def test_all_winning_trades_never_lose(self):
        analyzer = MonteCarloAnalyzer(n_simulations=200, min_trades=20, seed=3)
        result = _result([5.0] * 20)
        assert analyzer.probability_of_loss(result) == 0.0


# ======================================================================
# Return confidence interval
# ======================================================================

class TestReturnConfidenceInterval:
    def test_bounds_are_consistent(self):
        analyzer = MonteCarloAnalyzer(n_simulations=500, min_trades=20, seed=11)
        pnls = [10.0, -8.0, 12.0, -5.0, 7.0, -15.0, 20.0, -3.0] * 3
        result = _result(pnls)
        ci = analyzer.return_confidence_interval(result)

        assert isinstance(ci, ReturnConfidenceInterval)
        assert ci.lower_bound_pct <= ci.expected_return_pct <= ci.upper_bound_pct
        assert ci.std_pct >= 0.0

    def test_all_winning_trades_have_strictly_positive_lower_bound(self):
        analyzer = MonteCarloAnalyzer(n_simulations=200, min_trades=20, seed=5)
        result = _result([5.0] * 20)
        ci = analyzer.return_confidence_interval(result)
        assert ci.lower_bound_pct > 0.0


# ======================================================================
# Reproducibility
# ======================================================================

class TestReproducibility:
    def test_same_seed_produces_identical_results(self):
        pnls = [10.0, -8.0, 12.0, -5.0, 7.0, -15.0, 20.0, -3.0] * 3
        result = _result(pnls)

        a1 = MonteCarloAnalyzer(n_simulations=300, min_trades=20, seed=99)
        a2 = MonteCarloAnalyzer(n_simulations=300, min_trades=20, seed=99)

        assert a1.analyze_drawdowns(result) == a2.analyze_drawdowns(result)
        assert a1.return_confidence_interval(result) == a2.return_confidence_interval(result)

    def test_different_seeds_can_produce_different_results(self):
        pnls = [10.0, -8.0, 12.0, -5.0, 7.0, -15.0, 20.0, -3.0] * 3
        result = _result(pnls)

        a1 = MonteCarloAnalyzer(n_simulations=300, min_trades=20, seed=1)
        a2 = MonteCarloAnalyzer(n_simulations=300, min_trades=20, seed=2)

        assert a1.analyze_drawdowns(result) != a2.analyze_drawdowns(result)


# ======================================================================
# analyze() aggregate dict
# ======================================================================

class TestAnalyze:
    def test_returns_dict_with_expected_keys(self):
        analyzer = MonteCarloAnalyzer(n_simulations=200, min_trades=20, seed=13)
        pnls = [10.0, -8.0, 12.0, -5.0, 7.0, -15.0, 20.0, -3.0] * 3
        result = _result(pnls)
        summary = analyzer.analyze(result)

        expected_keys = {
            "n_closed_trades", "n_simulations", "confidence",
            "expected_max_drawdown_pct", "worst_case_drawdown_pct",
            "tail_drawdown_pct", "probability_of_loss",
            "expected_return_pct", "return_lower_bound_pct",
            "return_upper_bound_pct",
        }
        assert expected_keys.issubset(summary.keys())
        assert summary["n_closed_trades"] == 24
        assert summary["n_simulations"] == 200
