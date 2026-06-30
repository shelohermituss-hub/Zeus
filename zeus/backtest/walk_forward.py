"""
Walk-forward optimization for trading strategies.

Splits historical data into successive (train, test) windows. For each
window, every parameter combination in the grid is backtested on the train
slice; the best-scoring combination is then re-tested, untouched, on the
following test slice. Stitching only the test-slice results together
answers the question a single in-sample backtest cannot: "How would this
strategy have performed on data its parameters were never fit to?"

This is the standard defense against overfitting a strategy to one
historical period — per CLAUDE.md, no strategy should be judged on
in-sample performance alone.
"""
from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from zeus.backtest.engine import BacktestEngine
from zeus.backtest.report import BacktestResult
from zeus.strategy.base import Strategy

Objective = Callable[[BacktestResult], float]

# Built-in objective functions, all oriented so that higher == better.
# max_drawdown_pct is negated since a smaller drawdown is the better outcome.
_OBJECTIVES: dict[str, Objective] = {
    "net_pnl":          lambda r: r.net_pnl,
    "total_return_pct": lambda r: r.total_return_pct,
    "sharpe_ratio":     lambda r: r.sharpe_ratio,
    "profit_factor":    lambda r: r.profit_factor,
    "win_rate":         lambda r: r.win_rate,
    "max_drawdown_pct": lambda r: -r.max_drawdown_pct,
}


class InsufficientDataError(ValueError):
    """Too few bars to form even one complete (train, test) window."""


class InvalidParamGridError(ValueError):
    """param_grid is empty or contains a parameter with no candidate values."""


@dataclass(frozen=True)
class WalkForwardWindow:
    """One (train, test) step of the walk-forward run."""
    window_index:      int
    train_start:        int   # bar index, inclusive
    train_end:          int   # bar index, exclusive
    test_start:         int   # bar index, inclusive
    test_end:            int   # bar index, exclusive
    best_params:        dict
    in_sample_score:    float
    out_sample_score:   float
    out_sample_result:  BacktestResult


@dataclass(frozen=True)
class WalkForwardReport:
    """Aggregate result of a full walk-forward run."""
    windows:   list[WalkForwardWindow]
    objective: str

    @property
    def mean_in_sample_score(self) -> float:
        return sum(w.in_sample_score for w in self.windows) / len(self.windows)

    @property
    def mean_out_sample_score(self) -> float:
        return sum(w.out_sample_score for w in self.windows) / len(self.windows)

    @property
    def score_degradation_pct(self) -> float:
        """
        Fractional drop from in-sample to out-sample score.

        0.0 means out-of-sample performance fully matched in-sample;
        1.0 means it collapsed to zero; negative means out-of-sample
        beat in-sample. A large positive value is the signature of
        overfitting to the train window.
        """
        in_sample = self.mean_in_sample_score
        if in_sample == 0:
            return 0.0
        return (in_sample - self.mean_out_sample_score) / abs(in_sample)

    def combined_equity_curve(self) -> list[float]:
        """
        Stitch each window's out-of-sample equity curve into one continuous,
        normalized curve (starting at 1.0) by chaining period returns.

        Each window's BacktestEngine run starts from a fresh balance, so
        curves are normalized to relative growth before chaining — this
        reflects compounding the strategy's out-of-sample returns only.
        """
        combined = [1.0]
        for window in self.windows:
            curve = window.out_sample_result.equity_curve
            if not curve or curve[0] == 0:
                continue
            base = combined[-1]
            combined.extend(base * (v / curve[0]) for v in curve[1:])
        return combined

    @property
    def out_of_sample_total_return_pct(self) -> float:
        curve = self.combined_equity_curve()
        if len(curve) < 2 or curve[0] == 0:
            return 0.0
        return (curve[-1] - curve[0]) / curve[0]

    def summary(self) -> dict:
        """Return all key metrics as a plain dict (suitable for logging)."""
        return {
            "objective":                    self.objective,
            "n_windows":                    len(self.windows),
            "mean_in_sample_score":         round(self.mean_in_sample_score, 4),
            "mean_out_sample_score":        round(self.mean_out_sample_score, 4),
            "score_degradation_pct":        round(self.score_degradation_pct * 100, 2),
            "out_of_sample_total_return_pct": round(self.out_of_sample_total_return_pct * 100, 2),
            "windows": [
                {
                    "window_index":     w.window_index,
                    "train_range":      [w.train_start, w.train_end],
                    "test_range":       [w.test_start, w.test_end],
                    "best_params":      w.best_params,
                    "in_sample_score":  round(w.in_sample_score, 4),
                    "out_sample_score": round(w.out_sample_score, 4),
                }
                for w in self.windows
            ],
        }


class WalkForwardOptimizer:
    """
    Rolling walk-forward grid-search optimizer.

    For each window of train_bars followed by test_bars, every combination
    in param_grid is backtested on the train slice and scored by `objective`.
    The single best-scoring combination is then backtested, unmodified, on
    the test slice — the result that actually counts.

    Windows roll forward by step_bars (defaults to test_bars, i.e.
    non-overlapping out-of-sample periods) until the data is exhausted.
    """

    def __init__(
        self,
        strategy_factory: Callable[[dict], Strategy],
        param_grid: dict[str, list],
        train_bars: int,
        test_bars: int,
        step_bars: int | None = None,
        objective: str | Objective = "sharpe_ratio",
        engine_kwargs: dict | None = None,
    ) -> None:
        if train_bars <= 0 or test_bars <= 0:
            raise ValueError("train_bars and test_bars must be positive")
        if not param_grid:
            raise InvalidParamGridError("param_grid must contain at least one parameter")
        if any(len(values) == 0 for values in param_grid.values()):
            raise InvalidParamGridError("every parameter in param_grid needs at least one value")

        self._factory      = strategy_factory
        self._param_grid    = param_grid
        self._train_bars    = train_bars
        self._test_bars      = test_bars
        self._step_bars      = step_bars if step_bars is not None else test_bars
        if self._step_bars <= 0:
            raise ValueError("step_bars must be positive")
        self._objective_name = objective if isinstance(objective, str) else objective.__name__
        self._objective      = self._resolve_objective(objective)
        self._engine_kwargs  = engine_kwargs or {}

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(self, df: pd.DataFrame, symbol: str = "BTC/USDT") -> WalkForwardReport:
        """Run the full walk-forward optimization over *df*."""
        n_bars = len(df)
        window_size = self._train_bars + self._test_bars
        if n_bars < window_size:
            raise InsufficientDataError(
                f"{n_bars} bars available < {window_size} required for one "
                f"walk-forward window (train={self._train_bars} + test={self._test_bars})"
            )

        combos = self._param_combinations()
        windows: list[WalkForwardWindow] = []

        train_start = 0
        window_index = 0
        while train_start + window_size <= n_bars:
            train_end = train_start + self._train_bars
            test_end  = train_end + self._test_bars

            train_df = df.iloc[train_start:train_end]
            test_df  = df.iloc[train_end:test_end]

            best_params, in_sample_score = self._optimise(train_df, symbol, combos)
            out_sample_result = self._run_backtest(best_params, test_df, symbol)
            out_sample_score  = self._objective(out_sample_result)

            windows.append(WalkForwardWindow(
                window_index=window_index,
                train_start=train_start,
                train_end=train_end,
                test_start=train_end,
                test_end=test_end,
                best_params=best_params,
                in_sample_score=in_sample_score,
                out_sample_score=out_sample_score,
                out_sample_result=out_sample_result,
            ))

            window_index += 1
            train_start  += self._step_bars

        return WalkForwardReport(windows=windows, objective=self._objective_name)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _resolve_objective(self, objective: str | Objective) -> Objective:
        if callable(objective):
            return objective
        if objective not in _OBJECTIVES:
            raise ValueError(
                f"Unknown objective '{objective}'; choose from "
                f"{sorted(_OBJECTIVES)} or pass a callable"
            )
        return _OBJECTIVES[objective]

    def _param_combinations(self) -> list[dict]:
        keys = sorted(self._param_grid)
        value_lists = [self._param_grid[k] for k in keys]
        return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]

    def _optimise(
        self, train_df: pd.DataFrame, symbol: str, combos: list[dict]
    ) -> tuple[dict, float]:
        """Backtest every combo on train_df, return the best (params, score)."""
        best_params: dict | None = None
        best_score = float("-inf")
        for params in combos:
            result = self._run_backtest(params, train_df, symbol)
            score = self._objective(result)
            if score > best_score:
                best_score, best_params = score, params
        assert best_params is not None  # combos is non-empty by construction
        return best_params, best_score

    def _run_backtest(self, params: dict, df: pd.DataFrame, symbol: str) -> BacktestResult:
        strategy = self._factory(params)
        engine = BacktestEngine(strategy, **self._engine_kwargs)
        return engine.run(df, symbol)
