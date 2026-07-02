"""
Walk-forward adapter for sd_simulation-based strategies.

Problem
-------
WalkForwardOptimizer (zeus/backtest/walk_forward.py) was built around
BacktestEngine + Strategy base class.  The newer strategies (Harmonic,
ICT OB, FVG, Hybrid) expose a different interface:

    signals = strategy.run(m15_df)          # -> list[Signal]
    results, n = simulate_all(signals, ...)  # sd_simulation

SDWalkForwardOptimizer bridges the gap without requiring BacktestEngine.

Usage example::

    from zeus.backtest.sd_engine_adapter import SDWalkForwardOptimizer
    from zeus.strategy.harmonic_strategy import HarmonicStrategy

    def factory(params: dict) -> HarmonicStrategy:
        return HarmonicStrategy(
            pivot_size = params["pivot_size"],
            risk_reward = params["risk_reward"],
        )

    opt = SDWalkForwardOptimizer(
        strategy_factory = factory,
        param_grid = {
            "pivot_size":  [5, 7, 10],
            "risk_reward": [1.0, 1.5, 2.0],
        },
        train_bars = 2000,
        test_bars  = 500,
        objective  = "win_rate",
    )
    report = opt.run(m15_df, m1_df=m1_df)
    print(report.summary())
"""
from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from zeus.backtest.sd_simulation import compute_metrics, simulate_all

# ── Objective functions ────────────────────────────────────────────────────────

_OBJECTIVES: dict[str, Callable[[dict], float]] = {
    "win_rate":    lambda m: m["win_rate"],
    "total_r":     lambda m: m["total_r"],
    "ev":          lambda m: m["total_r"] / m["n_trades"] if m["n_trades"] else 0.0,
    "max_dd_neg":  lambda m: -m["max_dd"],
    "profit_factor": lambda m: (
        sum(r for r in [m.get("total_r", 0)] if r > 0) /
        max(abs(sum(r for r in [m.get("total_r", 0)] if r < 0)), 1e-9)
    ),
}


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SDWindowResult:
    """Out-of-sample result for one walk-forward window."""
    window_index:    int
    train_start:     int
    train_end:       int
    test_start:      int
    test_end:        int
    best_params:     dict
    in_sample_score: float
    out_sample_score: float
    metrics:         dict   # compute_metrics() dict for the test slice


@dataclass
class SDWalkForwardReport:
    """Aggregate result of a full sd_simulation walk-forward run."""
    windows:   list[SDWindowResult]
    objective: str

    @property
    def mean_in_sample(self) -> float:
        return float(np.mean([w.in_sample_score for w in self.windows])) if self.windows else 0.0

    @property
    def mean_out_sample(self) -> float:
        return float(np.mean([w.out_sample_score for w in self.windows])) if self.windows else 0.0

    @property
    def score_degradation_pct(self) -> float:
        """Fraction drop from in-sample to out-sample (0=no drop, 1=collapsed)."""
        ins = self.mean_in_sample
        if ins == 0:
            return 0.0
        return (ins - self.mean_out_sample) / abs(ins)

    def summary(self) -> dict:
        return {
            "objective":            self.objective,
            "n_windows":            len(self.windows),
            "mean_in_sample":       round(self.mean_in_sample, 4),
            "mean_out_sample":      round(self.mean_out_sample, 4),
            "score_degradation_pct": round(self.score_degradation_pct * 100, 2),
            "windows": [
                {
                    "index":          w.window_index,
                    "train":          [w.train_start, w.train_end],
                    "test":           [w.test_start, w.test_end],
                    "best_params":    w.best_params,
                    "in_sample":      round(w.in_sample_score, 4),
                    "out_sample":     round(w.out_sample_score, 4),
                    "win_rate":       round(w.metrics.get("win_rate", 0.0), 2),
                    "total_r":        round(w.metrics.get("total_r", 0.0), 2),
                    "n_signals":      w.metrics.get("n_signals", 0),
                }
                for w in self.windows
            ],
        }

    def print_summary(self) -> None:
        s = self.summary()
        print(f"\n  Walk-Forward ({self.objective}) — {s['n_windows']} windows")
        print(f"  In-sample avg : {s['mean_in_sample']:.4f}")
        print(f"  Out-sample avg: {s['mean_out_sample']:.4f}")
        print(f"  Degradation   : {s['score_degradation_pct']:.1f}%")
        print(f"\n  {'Win':>4}  {'Train':>14}  {'Test':>14}  "
              f"{'InS':>7}  {'OutS':>7}  {'WR':>6}  {'Params'}")
        print("  " + "─" * 80)
        for w in self.windows:
            print(
                f"  {w.window_index:>4}  "
                f"[{w.train_start:>6},{w.train_end:>6}]  "
                f"[{w.test_start:>6},{w.test_end:>6}]  "
                f"{w.in_sample_score:>7.3f}  "
                f"{w.out_sample_score:>7.3f}  "
                f"{w.metrics.get('win_rate', 0.0):>5.1f}%  "
                f"{w.best_params}"
            )
        print()


# ── Optimizer ─────────────────────────────────────────────────────────────────

class SDWalkForwardOptimizer:
    """
    Rolling walk-forward optimizer for sd_simulation-based strategies.

    For each (train, test) window, all param combinations are back-tested on
    the train slice; the winner is applied unmodified to the test slice.

    Args:
        strategy_factory: Callable(params: dict) -> strategy with .run(df) method.
        param_grid:       Dict of param_name → list of candidate values.
        train_bars:       Number of M15 bars in each training slice.
        test_bars:        Number of M15 bars in each test slice.
        step_bars:        Roll-forward step (defaults to test_bars).
        objective:        Metric to maximise: "win_rate" | "total_r" | "ev" |
                          "max_dd_neg" | "profit_factor", or a callable(metrics)->float.
        initial_equity:   Starting equity for each simulation (default 10_000).
        risk_pct:         Risk per trade as fraction of equity (default 0.01).
        spread:           Bid/ask spread in price units (default 0.30).
        slippage_ticks:   Additional slippage beyond spread (default 0.0).
    """

    def __init__(
        self,
        strategy_factory: Callable[[dict], Any],
        param_grid:       dict[str, list],
        train_bars:       int,
        test_bars:        int,
        step_bars:        int | None = None,
        objective:        str | Callable[[dict], float] = "win_rate",
        initial_equity:   float = 10_000.0,
        risk_pct:         float = 0.01,
        spread:           float = 0.30,
        slippage_ticks:   float = 0.0,
    ) -> None:
        if train_bars <= 0 or test_bars <= 0:
            raise ValueError("train_bars and test_bars must be positive")
        if not param_grid:
            raise ValueError("param_grid must contain at least one parameter")
        if any(len(v) == 0 for v in param_grid.values()):
            raise ValueError("every parameter in param_grid needs at least one value")

        self._factory        = strategy_factory
        self._param_grid     = param_grid
        self._train_bars     = train_bars
        self._test_bars      = test_bars
        self._step_bars      = step_bars or test_bars
        self._obj_name       = objective if isinstance(objective, str) else objective.__name__
        self._obj            = self._resolve_obj(objective)
        self._initial_equity = initial_equity
        self._risk_pct       = risk_pct
        self._spread         = spread
        self._slippage       = slippage_ticks

    # ── Public ────────────────────────────────────────────────────────────────

    def run(
        self,
        m15_df: pd.DataFrame,
        m1_df:  pd.DataFrame | None = None,
    ) -> SDWalkForwardReport:
        """
        Run the full walk-forward over *m15_df*.

        If *m1_df* is supplied, SL/TP resolution is done at M1 granularity
        (recommended for realistic loss estimation).
        """
        n = len(m15_df)
        window_size = self._train_bars + self._test_bars
        if n < window_size:
            raise ValueError(
                f"Only {n} bars available; need {window_size} for one window "
                f"(train={self._train_bars} + test={self._test_bars})"
            )

        combos   = self._combos()
        windows: list[SDWindowResult] = []
        train_start = 0
        idx = 0

        while train_start + window_size <= n:
            train_end = train_start + self._train_bars
            test_end  = train_end  + self._test_bars

            train_df = m15_df.iloc[train_start:train_end]
            test_df  = m15_df.iloc[train_end:test_end]

            train_m1 = self._slice_m1(m1_df, train_df) if m1_df is not None else None
            test_m1  = self._slice_m1(m1_df, test_df)  if m1_df is not None else None

            best_params, in_score = self._optimise(train_df, train_m1, combos)
            out_metrics = self._score(best_params, test_df, test_m1)
            out_score   = self._obj(out_metrics)

            windows.append(SDWindowResult(
                window_index=idx,
                train_start=train_start,
                train_end=train_end,
                test_start=train_end,
                test_end=test_end,
                best_params=best_params,
                in_sample_score=in_score,
                out_sample_score=out_score,
                metrics=out_metrics,
            ))
            idx         += 1
            train_start += self._step_bars

        return SDWalkForwardReport(windows=windows, objective=self._obj_name)

    # ── Private ───────────────────────────────────────────────────────────────

    def _resolve_obj(self, obj: str | Callable) -> Callable[[dict], float]:
        if callable(obj):
            return obj
        if obj not in _OBJECTIVES:
            raise ValueError(
                f"Unknown objective '{obj}'; choose from "
                f"{sorted(_OBJECTIVES)} or pass a callable"
            )
        return _OBJECTIVES[obj]

    def _combos(self) -> list[dict]:
        keys = sorted(self._param_grid)
        return [
            dict(zip(keys, vals))
            for vals in itertools.product(*[self._param_grid[k] for k in keys])
        ]

    def _score(
        self,
        params: dict,
        df: pd.DataFrame,
        m1_df: pd.DataFrame | None,
    ) -> dict:
        strategy = self._factory(params)
        signals  = strategy.run(df)
        if not signals:
            return {
                "n_signals": 0, "n_trades": 0, "n_expired": 0,
                "n_wins": 0, "n_losses": 0, "n_scratches": 0,
                "win_rate": 0.0, "total_r": 0.0, "total_usd": 0.0,
                "total_spread": 0.0, "final_eq": self._initial_equity,
                "max_dd": 0.0, "avg_bars_win": 0, "avg_bars_loss": 0,
                "avg_zone_win": 0.0, "avg_zone_loss": 0.0,
            }
        tick_df = m1_df if m1_df is not None else df
        results, n_exp = simulate_all(
            signals, df, tick_df=tick_df,
            risk_pct=self._risk_pct,
            spread=self._spread,
            slippage_ticks=self._slippage,
            initial_equity=self._initial_equity,
        )
        return compute_metrics(results, self._initial_equity, len(signals), n_exp)

    def _optimise(
        self,
        train_df: pd.DataFrame,
        train_m1: pd.DataFrame | None,
        combos:   list[dict],
    ) -> tuple[dict, float]:
        best_params: dict | None = None
        best_score = float("-inf")
        for params in combos:
            m = self._score(params, train_df, train_m1)
            s = self._obj(m)
            if s > best_score:
                best_score  = s
                best_params = params
        assert best_params is not None
        return best_params, best_score

    @staticmethod
    def _slice_m1(
        m1_df: pd.DataFrame,
        m15_slice: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return the M1 rows that fall within the M15 slice's time range."""
        start = m15_slice.index[0]
        end   = m15_slice.index[-1]
        return m1_df.loc[start:end]
