"""Tests de zeus/strategy/momentum_scalp.py — détection de burst momentum M1."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zeus.strategy.momentum_scalp import MomentumScalpStrategy


def _quiet_bars(n: int, price: float = 100.0, half_range: float = 0.25) -> list[dict]:
    """Bougies plates (doji) : open==close, petit wick des deux côtés."""
    return [
        dict(open=price, high=price + half_range, low=price - half_range, close=price)
        for _ in range(n)
    ]


def _mk_df(bars: list[dict], start: str = "2026-06-01 10:00") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(bars), freq="1min")
    return pd.DataFrame(bars, index=idx)


def _bull_burst(start_price: float, step: float = 0.7, wick: float = 0.05, n: int = 3) -> list[dict]:
    bars = []
    p = start_price
    for _ in range(n):
        o, c = p, p + step
        bars.append(dict(open=o, high=c + wick, low=o - wick, close=c))
        p = c
    return bars


def _bear_burst(start_price: float, step: float = 0.7, wick: float = 0.05, n: int = 3) -> list[dict]:
    bars = []
    p = start_price
    for _ in range(n):
        o, c = p, p - step
        bars.append(dict(open=o, high=o + wick, low=c - wick, close=c))
        p = c
    return bars


def _wicky_burst(start_price: float, n: int = 3) -> list[dict]:
    """Range large mais corps minuscule — momentum apparent mais pas propre."""
    bars = []
    p = start_price
    for _ in range(n):
        o, c = p, p + 0.05   # tiny body
        bars.append(dict(open=o, high=o + 1.5, low=o - 1.5, close=c))  # huge wicks
        p = c
    return bars


class TestBurstDetection:
    def test_clean_bull_burst_emits_long_signal(self):
        bars = _quiet_bars(10) + _bull_burst(100.0) + _quiet_bars(2)
        df = _mk_df(bars)
        strat = MomentumScalpStrategy()
        signals = strat.run(df)
        assert len(signals) == 1
        sig = signals[0]
        assert sig.direction == "long"
        assert sig.bar_index == 12          # last bar of the 3-bar burst (index 10,11,12)
        assert sig.entry_price == pytest.approx(102.1, abs=1e-6)
        assert sig.stop_loss < 99.95         # below window low, buffered further by ATR
        assert sig.take_profit > sig.entry_price
        assert sig.zone_score >= 5.0

    def test_clean_bear_burst_emits_short_signal(self):
        bars = _quiet_bars(10) + _bear_burst(100.0) + _quiet_bars(2)
        df = _mk_df(bars)
        strat = MomentumScalpStrategy()
        signals = strat.run(df)
        assert len(signals) == 1
        sig = signals[0]
        assert sig.direction == "short"
        assert sig.entry_price == pytest.approx(97.9, abs=1e-6)
        assert sig.stop_loss > 100.05
        assert sig.take_profit < sig.entry_price

    def test_choppy_alternating_bars_no_signal(self):
        # Alternates up/down every bar within the window — no directional dominance.
        bars = _quiet_bars(10)
        p = 100.0
        for k in range(6):
            step = 0.7 if k % 2 == 0 else -0.7
            o, c = p, p + step
            bars.append(dict(open=o, high=max(o, c) + 0.05, low=min(o, c) - 0.05, close=c))
            p = c
        df = _mk_df(bars)
        strat = MomentumScalpStrategy()
        assert strat.run(df) == []

    def test_flat_market_no_signal_fail_closed(self):
        bars = _quiet_bars(20)
        df = _mk_df(bars)
        strat = MomentumScalpStrategy()
        # Must not raise (ATR ~ 0) and must not fabricate a signal from noise.
        assert strat.run(df) == []

    def test_weak_burst_below_atr_threshold_no_signal(self):
        # Directional and clean, but the whole window barely moves vs. prior volatility.
        bars = _quiet_bars(10, half_range=1.0)   # ATR settles around 2.0
        p = 100.0
        for _ in range(3):
            o, c = p, p + 0.3   # tiny step vs ATR ~2.0
            bars.append(dict(open=o, high=c + 0.02, low=o - 0.02, close=c))
            p = c
        df = _mk_df(bars)
        strat = MomentumScalpStrategy()
        assert strat.run(df) == []

    def test_wicky_burst_no_signal(self):
        bars = _quiet_bars(10) + _wicky_burst(100.0) + _quiet_bars(2)
        df = _mk_df(bars)
        strat = MomentumScalpStrategy()
        assert strat.run(df) == []


class TestFilters:
    def test_session_filter_blocks_outside_hours(self):
        bars = _quiet_bars(10) + _bull_burst(100.0) + _quiet_bars(2)
        df = _mk_df(bars, start="2026-06-01 23:00")   # outside default 7-21 UTC
        strat = MomentumScalpStrategy()
        assert strat.run(df) == []

    def test_cooldown_blocks_second_burst_too_soon(self):
        bars = (_quiet_bars(10) + _bull_burst(100.0)
                + _quiet_bars(5) + _bull_burst(110.0))
        df = _mk_df(bars)
        strat = MomentumScalpStrategy(cooldown_bars=15)
        signals = strat.run(df)
        assert len(signals) == 1   # second burst is only 8 bars after the first

    def test_cooldown_allows_second_burst_after_gap(self):
        bars = (_quiet_bars(10) + _bull_burst(100.0)
                + _quiet_bars(20) + _bull_burst(110.0) + _quiet_bars(2))
        df = _mk_df(bars)
        strat = MomentumScalpStrategy(cooldown_bars=15)
        signals = strat.run(df)
        assert len(signals) == 2

    def test_max_signals_per_day_caps_count(self):
        bars = _quiet_bars(10)
        p = 100.0
        for k in range(6):
            bars += _bull_burst(p, step=0.7)
            p += 2.1
            bars += _quiet_bars(20)
        df = _mk_df(bars)
        strat = MomentumScalpStrategy(cooldown_bars=5, max_signals_per_day=2)
        signals = strat.run(df)
        assert len(signals) == 2

    def test_min_zone_score_filters_marginal_bursts(self):
        bars = _quiet_bars(10) + _bull_burst(100.0) + _quiet_bars(2)
        df = _mk_df(bars)
        lenient = MomentumScalpStrategy(min_zone_score=5.0)
        strict  = MomentumScalpStrategy(min_zone_score=9.9)
        assert len(lenient.run(df)) == 1
        assert len(strict.run(df)) == 0

    def test_min_sl_pips_rejects_too_tight_stop(self):
        bars = _quiet_bars(10) + _bull_burst(100.0) + _quiet_bars(2)
        df = _mk_df(bars)
        strat = MomentumScalpStrategy(min_sl_pips=50_000, pip_size=0.0001)
        assert strat.run(df) == []


class TestEdgeCases:
    def test_too_few_bars_returns_empty(self):
        df = _mk_df(_quiet_bars(3))
        strat = MomentumScalpStrategy()
        assert strat.run(df) == []

    def test_signals_are_chronological_and_tradeable_protocol_shape(self):
        bars = (_quiet_bars(10) + _bull_burst(100.0)
                + _quiet_bars(20) + _bear_burst(110.0) + _quiet_bars(2))
        df = _mk_df(bars)
        strat = MomentumScalpStrategy(cooldown_bars=15)
        signals = strat.run(df)
        assert len(signals) == 2
        assert signals[0].formed_at < signals[1].formed_at
        for s in signals:
            # Tradeable protocol fields used by sd_simulation.simulate_all
            assert isinstance(s.direction, str)
            assert isinstance(s.bar_index, int)
            assert isinstance(s.stop_loss, float)
            assert isinstance(s.risk_reward, float)
            assert isinstance(s.formed_at, pd.Timestamp)
            assert isinstance(s.zone_score, float)
