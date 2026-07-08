"""Tests de zeus/strategy/candlestick_patterns.py — un cas positif par
pattern (7 familles portées depuis les "Free Robots" MetaQuotes)."""
from __future__ import annotations

import pandas as pd

from zeus.strategy.candlestick_patterns import PATTERNS, CandlestickRsiStrategy


def _mk_df(bars: list[dict], start: str = "2026-01-01 00:00") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(bars), freq="1min")
    return pd.DataFrame(bars, index=idx)


def _trend_bars(n: int, start: float, step: float) -> list[dict]:
    bars = []
    p = start
    for _ in range(n):
        o, c = p, p + step
        bars.append(dict(open=o, high=max(o, c) + 0.05, low=min(o, c) - 0.05, close=c))
        p = c
    return bars


class TestAllPatternsDetectAtLeastOnce:
    """Smoke test : chaque pattern doit pouvoir se déclencher sur une
    configuration de bougies construite pour satisfaire ses conditions,
    sans lever d'exception (bornes de tableau, indices, NaN)."""

    def test_harami_bullish(self):
        bars = _trend_bars(45, start=100.0, step=-0.5)   # downtrend context
        prev_close = bars[-1]["close"]
        # bar i-1 : long bearish body >> avg
        bars.append(dict(open=prev_close, high=prev_close + 0.05,
                          low=prev_close - 3.2, close=prev_close - 3.0))
        o2, c2 = bars[-1]["open"], bars[-1]["close"]
        # bar i : small bullish body fully inside [c2, o2]
        bars.append(dict(open=c2 + 0.2, high=c2 + 0.9, low=c2 + 0.1, close=c2 + 0.8))
        df = _mk_df(bars)
        strat = CandlestickRsiStrategy("harami")
        sigs = strat.run(df)
        assert any(s.direction == "long" for s in sigs)

    def test_dark_cloud(self):
        bars = _trend_bars(45, start=100.0, step=0.5)   # uptrend context
        prev_close = bars[-1]["close"]
        bars.append(dict(open=prev_close, high=prev_close + 3.2,
                          low=prev_close - 0.05, close=prev_close + 3.0))
        o2, c2, h2 = bars[-1]["open"], bars[-1]["close"], bars[-1]["high"]
        bars.append(dict(open=h2 + 0.1, high=h2 + 0.15, low=o2 - 0.1, close=o2 + 0.5))
        df = _mk_df(bars)
        strat = CandlestickRsiStrategy("dark_cloud_piercing")
        sigs = strat.run(df)
        assert any(s.direction == "short" for s in sigs)

    def test_hammer(self):
        bars = _trend_bars(45, start=100.0, step=-0.5)   # downtrend context
        prev = bars[-1]
        # Hammer: small body in upper third, closes/opens below previous bar
        bars.append(dict(open=prev["close"] - 0.4, high=prev["close"] - 0.35,
                          low=prev["close"] - 2.0, close=prev["close"] - 0.3))
        df = _mk_df(bars)
        strat = CandlestickRsiStrategy("hanging_man_hammer")
        sigs = strat.run(df)
        assert any(s.direction == "long" for s in sigs)

    def test_morning_star(self):
        # Needs a real sustained downtrend (not just a shape match) so RSI
        # is actually oversold enough to confirm — a shallow -0.1 drift
        # left RSI just above the 40 gate despite the pattern matching.
        bars = _trend_bars(45, start=100.0, step=-0.5)
        prev_close = bars[-1]["close"]
        bars.append(dict(open=prev_close + 2.0, high=prev_close + 2.05,
                          low=prev_close - 1.0, close=prev_close - 0.5))
        c3, o3 = bars[-1]["close"], bars[-1]["open"]
        bars.append(dict(open=prev_close - 0.55, high=prev_close - 0.5,
                          low=prev_close - 0.65, close=prev_close - 0.55))
        mid_oc3 = (o3 + c3) / 2.0
        bars.append(dict(open=prev_close - 0.6, high=mid_oc3 + 1.0,
                          low=prev_close - 0.7, close=mid_oc3 + 0.8))
        df = _mk_df(bars)
        strat = CandlestickRsiStrategy("morning_evening_star")
        sigs = strat.run(df)
        assert any(s.direction == "long" for s in sigs)

    def test_three_white_soldiers(self):
        # Same reasoning as test_morning_star: a longer/steeper downtrend
        # is needed for RSI to actually confirm the reversal.
        bars = _trend_bars(60, start=150.0, step=-0.5)
        p = bars[-1]["close"]
        for _ in range(3):
            o, c = p, p + 3.0
            bars.append(dict(open=o, high=c + 0.05, low=o - 0.05, close=c))
            p = c
        df = _mk_df(bars)
        strat = CandlestickRsiStrategy("black_crows_soldiers")
        sigs = strat.run(df)
        assert any(s.direction == "long" for s in sigs)

    def test_meeting_lines_bullish(self):
        bars = _trend_bars(45, start=100.0, step=-0.1)
        prev_close = bars[-1]["close"]
        bars.append(dict(open=prev_close, high=prev_close + 0.05,
                          low=prev_close - 3.0, close=prev_close - 2.9))
        c2 = bars[-1]["close"]
        bars.append(dict(open=c2 - 3.0, high=c2 + 0.05, low=c2 - 3.05, close=c2 + 0.02))
        df = _mk_df(bars)
        strat = CandlestickRsiStrategy("meeting_lines")
        sigs = strat.run(df)
        assert any(s.direction == "long" for s in sigs)

    def test_unknown_pattern_raises(self):
        try:
            CandlestickRsiStrategy("not_a_real_pattern")
            assert False, "should have raised"
        except ValueError:
            pass

    def test_insufficient_data_all_patterns_return_empty(self):
        df = _mk_df(_trend_bars(5, start=100.0, step=0.1))
        for name in PATTERNS:
            strat = CandlestickRsiStrategy(name)
            assert strat.run(df) == []
