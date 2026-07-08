"""Tests de zeus/strategy/engulfing_rsi.py — port du robot MetaQuotes
"BullishBearish Engulfing RSI.mq5" (détection de pattern + simulation)."""
from __future__ import annotations

import pandas as pd
import pytest

from zeus.strategy.engulfing_rsi import (
    EngulfingRsiSignal,
    EngulfingRsiStrategy,
    simulate_engulfing_rsi,
)


def _mk_df(bars: list[dict], start: str = "2026-01-01 00:00") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(bars), freq="1min")
    return pd.DataFrame(bars, index=idx)


def _trend_bars(n: int, start: float, step: float) -> list[dict]:
    """Tendance régulière (step>0 haussière, step<0 baissière), petites mèches."""
    bars = []
    p = start
    for _ in range(n):
        o, c = p, p + step
        bars.append(dict(open=o, high=max(o, c) + 0.05, low=min(o, c) - 0.05, close=c))
        p = c
    return bars


def _downtrend_then_bull_engulf(n_down: int = 45, body: float = 3.0) -> list[dict]:
    bars = _trend_bars(n_down, start=100.0, step=-0.5)
    prev_close = bars[-1]["close"]
    o_i = prev_close - 0.1
    c_i = o_i + body
    bars.append(dict(open=o_i, high=c_i + 0.05, low=o_i - 0.05, close=c_i))
    return bars


def _uptrend_then_bear_engulf(n_up: int = 45, body: float = 3.0) -> list[dict]:
    bars = _trend_bars(n_up, start=100.0, step=0.5)
    prev_close = bars[-1]["close"]
    o_i = prev_close + 0.1
    c_i = o_i - body
    bars.append(dict(open=o_i, high=o_i + 0.05, low=c_i - 0.05, close=c_i))
    return bars


class TestPatternDetection:
    def test_downtrend_reversal_emits_long_signal(self):
        df = _mk_df(_downtrend_then_bull_engulf())
        strat = EngulfingRsiStrategy()
        signals = strat.run(df)
        assert len(signals) == 1
        assert signals[0].direction == "long"
        assert signals[0].bar_index == 45

    def test_uptrend_reversal_emits_short_signal(self):
        df = _mk_df(_uptrend_then_bear_engulf())
        strat = EngulfingRsiStrategy()
        signals = strat.run(df)
        assert len(signals) == 1
        assert signals[0].direction == "short"

    def test_small_body_not_engulfing_enough_no_signal(self):
        # Body barely bigger than average — RSI condition may pass but the
        # engulfing-strength gate (body > avg_body) should reject it when
        # the body is actually SMALLER than the rolling average.
        bars = _downtrend_then_bull_engulf(body=0.3)   # smaller than the ~0.5 avg body
        df = _mk_df(bars)
        strat = EngulfingRsiStrategy()
        assert strat.run(df) == []

    def test_wrong_trend_context_no_signal(self):
        # Bullish-engulfing candlestick shape appended after an UPTREND
        # (not a downtrend) — mid_oc vs SMA context fails, so no long signal
        # even though the two-candle shape looks engulfing.
        bars = _trend_bars(45, start=100.0, step=0.5)   # uptrend, not down
        prev_close = bars[-1]["close"]
        o_i = prev_close - 0.1
        c_i = o_i + 3.0
        bars.append(dict(open=o_i, high=c_i + 0.05, low=o_i - 0.05, close=c_i))
        df = _mk_df(bars)
        strat = EngulfingRsiStrategy()
        assert strat.run(df) == []

    def test_rsi_not_confirming_no_signal(self):
        # Downtrend reversal, but require an unreachable RSI confirmation
        # threshold (rsi_buy_max=0) — pattern shape is fine, RSI gate blocks it.
        df = _mk_df(_downtrend_then_bull_engulf())
        strat = EngulfingRsiStrategy(rsi_buy_max=0.0)
        assert strat.run(df) == []

    def test_insufficient_data_returns_empty(self):
        df = _mk_df(_trend_bars(5, start=100.0, step=0.1))
        strat = EngulfingRsiStrategy()
        assert strat.run(df) == []


class TestSimulation:
    """Teste chaque branche de sortie de simulate_engulfing_rsi indépendamment
    de la détection — un seul signal synthétique, chemin de prix contrôlé."""

    def _flat_df(self, n: int, price: float = 100.0) -> pd.DataFrame:
        bars = [dict(open=price, high=price, low=price, close=price) for _ in range(n)]
        return _mk_df(bars)

    def test_tp_hit_first_is_a_win(self):
        df = self._flat_df(20)
        df.iloc[5, df.columns.get_loc("high")] = 100.30   # entry_bar=1, tp = 100+0.20
        sig = EngulfingRsiSignal(direction="long", bar_index=0,
                                  formed_at=df.index[0], entry_ref=100.0)
        results = simulate_engulfing_rsi(
            [sig], df, sl_points=200, tp_points=200, point_size=0.001,
            duration_bars=50, lot=0.1, contract_size=100000,
        )
        assert len(results) == 1
        assert results[0].outcome == "win"
        assert results[0].exit_bar == 5

    def test_sl_hit_first_is_a_loss(self):
        df = self._flat_df(20)
        df.iloc[5, df.columns.get_loc("low")] = 99.70   # sl = 100-0.20
        sig = EngulfingRsiSignal(direction="long", bar_index=0,
                                  formed_at=df.index[0], entry_ref=100.0)
        results = simulate_engulfing_rsi(
            [sig], df, sl_points=200, tp_points=200, point_size=0.001,
            duration_bars=50, lot=0.1, contract_size=100000,
        )
        assert len(results) == 1
        assert results[0].outcome == "loss"

    def test_sl_priority_when_both_hit_same_bar(self):
        df = self._flat_df(20)
        df.iloc[5, df.columns.get_loc("low")]  = 99.70
        df.iloc[5, df.columns.get_loc("high")] = 100.30
        sig = EngulfingRsiSignal(direction="long", bar_index=0,
                                  formed_at=df.index[0], entry_ref=100.0)
        results = simulate_engulfing_rsi(
            [sig], df, sl_points=200, tp_points=200, point_size=0.001,
            duration_bars=50, lot=0.1, contract_size=100000,
        )
        assert results[0].outcome == "loss"   # pessimistic: SL checked first

    def test_duration_expiry_closes_at_market(self):
        df = self._flat_df(20)   # never touches SL/TP, RSI stays flat (no cross)
        sig = EngulfingRsiSignal(direction="long", bar_index=0,
                                  formed_at=df.index[0], entry_ref=100.0)
        results = simulate_engulfing_rsi(
            [sig], df, sl_points=200, tp_points=200, point_size=0.001,
            duration_bars=5, lot=0.1, contract_size=100000,
        )
        assert len(results) == 1
        assert results[0].bars_held == 5
        assert results[0].outcome == "scratch"   # flat price → no P&L

    def test_no_exit_before_end_of_data_is_excluded(self):
        df = self._flat_df(6)
        sig = EngulfingRsiSignal(direction="long", bar_index=0,
                                  formed_at=df.index[0], entry_ref=100.0)
        results = simulate_engulfing_rsi(
            [sig], df, sl_points=200, tp_points=200, point_size=0.001,
            duration_bars=50, lot=0.1, contract_size=100000,
        )
        assert results == []   # never closed — expired, excluded like sd_simulation

    def test_short_direction_pnl_sign(self):
        df = self._flat_df(20)
        df.iloc[5, df.columns.get_loc("low")] = 99.70   # tp for a short = entry - 0.20*points... see below
        sig = EngulfingRsiSignal(direction="short", bar_index=0,
                                  formed_at=df.index[0], entry_ref=100.0)
        results = simulate_engulfing_rsi(
            [sig], df, sl_points=200, tp_points=200, point_size=0.001,
            duration_bars=50, lot=0.1, contract_size=100000,
        )
        assert len(results) == 1
        assert results[0].outcome == "win"
        assert results[0].pnl_usd > 0
