"""
Monte Carlo robustness analysis for backtest results.

Resamples the realized P&L of closed trades (with replacement) to answer:
"How sensitive is this strategy's drawdown and return to the exact sequence
in which these same trades happened to occur?" This complements
BacktestResult's single-path metrics, which only describe the one
historical sequence that actually happened.

All percentage fields are fractions (0.05 == 5%), matching the convention
used by zeus.backtest.report.BacktestResult.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zeus.backtest.report import BacktestResult


class InsufficientTradeDataError(ValueError):
    """Too few closed trades for a statistically meaningful resampling estimate."""


@dataclass(frozen=True)
class DrawdownDistribution:
    """Distribution of max drawdown across resampled trade sequences."""
    expected_max_drawdown_pct: float
    median_max_drawdown_pct:   float
    worst_case_drawdown_pct:   float
    tail_drawdown_pct:         float  # drawdown at the configured confidence level


@dataclass(frozen=True)
class ReturnConfidenceInterval:
    """Confidence interval for total return across resampled trade sequences."""
    expected_return_pct: float
    lower_bound_pct:     float
    upper_bound_pct:     float
    std_pct:             float


class MonteCarloAnalyzer:
    """
    Bootstrap-resamples closed-trade outcomes from a BacktestResult.

    Raises InsufficientTradeDataError instead of returning a falsely
    confident estimate when result.n_trades < min_trades — a Monte Carlo
    estimate from a handful of trades is noise, not a risk signal
    (fail-closed: refuse to answer rather than mislead).
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        confidence: float = 0.95,
        min_trades: int = 20,
        seed: int | None = None,
    ) -> None:
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if n_simulations < 1:
            raise ValueError("n_simulations must be positive")
        self._n_simulations = n_simulations
        self._confidence    = confidence
        self._min_trades    = min_trades
        self._rng           = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def analyze(self, result: BacktestResult) -> dict:
        """Run all analyses and return a plain dict (suitable for logging)."""
        dd = self.analyze_drawdowns(result)
        ci = self.return_confidence_interval(result)
        return {
            "n_closed_trades":          result.n_trades,
            "n_simulations":            self._n_simulations,
            "confidence":               self._confidence,
            "expected_max_drawdown_pct": round(dd.expected_max_drawdown_pct * 100, 2),
            "worst_case_drawdown_pct":   round(dd.worst_case_drawdown_pct * 100, 2),
            "tail_drawdown_pct":         round(dd.tail_drawdown_pct * 100, 2),
            "probability_of_loss":       round(self.probability_of_loss(result), 4),
            "expected_return_pct":       round(ci.expected_return_pct * 100, 2),
            "return_lower_bound_pct":    round(ci.lower_bound_pct * 100, 2),
            "return_upper_bound_pct":    round(ci.upper_bound_pct * 100, 2),
        }

    def analyze_drawdowns(self, result: BacktestResult) -> DrawdownDistribution:
        """Estimate the distribution of max drawdown under resampled trade order."""
        equity = self._bootstrap_equity_curves(result)
        running_max = np.maximum.accumulate(equity, axis=1)
        drawdowns = (equity - running_max) / running_max
        max_dd = -drawdowns.min(axis=1)  # positive fractions

        tail_pctl = self._confidence * 100
        return DrawdownDistribution(
            expected_max_drawdown_pct=float(max_dd.mean()),
            median_max_drawdown_pct=float(np.median(max_dd)),
            worst_case_drawdown_pct=float(max_dd.max()),
            tail_drawdown_pct=float(np.percentile(max_dd, tail_pctl)),
        )

    def probability_of_loss(self, result: BacktestResult) -> float:
        """Fraction of resampled trade sequences that end with a net loss."""
        equity = self._bootstrap_equity_curves(result)
        final_returns = equity[:, -1] - 1.0
        return float((final_returns < 0).mean())

    def return_confidence_interval(self, result: BacktestResult) -> ReturnConfidenceInterval:
        """Confidence interval for total return under resampled trade order."""
        equity = self._bootstrap_equity_curves(result)
        final_returns = equity[:, -1] - 1.0

        lower_pctl = (1 - self._confidence) / 2 * 100
        upper_pctl = 100 - lower_pctl
        return ReturnConfidenceInterval(
            expected_return_pct=float(final_returns.mean()),
            lower_bound_pct=float(np.percentile(final_returns, lower_pctl)),
            upper_bound_pct=float(np.percentile(final_returns, upper_pctl)),
            std_pct=float(final_returns.std()),
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _bootstrap_equity_curves(self, result: BacktestResult) -> np.ndarray:
        """Resample trade order with replacement -> shape (n_simulations, n_trades)."""
        if result.n_trades < self._min_trades:
            raise InsufficientTradeDataError(
                f"{result.n_trades} closed trades < minimum {self._min_trades} "
                "required for a statistically meaningful Monte Carlo estimate"
            )
        if result.initial_balance <= 0:
            raise ValueError("initial_balance must be positive")

        returns = np.array(
            [t.realized_pnl for t in result.closed_trades], dtype=float
        ) / result.initial_balance

        idx = self._rng.integers(0, len(returns), size=(self._n_simulations, len(returns)))
        sampled = returns[idx]
        return np.cumprod(1.0 + sampled, axis=1)
