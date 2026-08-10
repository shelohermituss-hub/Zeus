"""Tests for zeus.strategy.open_us_orb_strategy.OpenUSORBStrategy."""
from __future__ import annotations

import pandas as pd
import pytest

from zeus.strategy.open_us_orb_strategy import OpenUSORBStrategy


def _bars(start_utc: str, ohlc: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    idx = pd.date_range(start_utc, periods=len(ohlc), freq="1min")
    return pd.DataFrame(
        [{"open": o, "high": h, "low": l, "close": c} for o, h, l, c in ohlc],
        index=idx,
    )


class TestBreakout:
    def test_long_breakout_above_range(self):
        # 13:30-13:34 UTC = 09:30-09:34 ET : construction du range (5 min)
        # sur [1.100, 1.105]. Barre 13:35 (09:35 ET) casse au-dessus.
        ohlc = [
            (1.100, 1.102, 1.099, 1.101),
            (1.101, 1.104, 1.100, 1.102),
            (1.102, 1.105, 1.100, 1.103),
            (1.103, 1.105, 1.101, 1.104),
            (1.104, 1.105, 1.101, 1.103),
            (1.103, 1.110, 1.103, 1.108),   # cassure haussière : close=1.108 > or_high=1.105
        ]
        df = _bars("2026-06-01 13:30", ohlc)
        strat = OpenUSORBStrategy(or_minutes=5, search_minutes=30)
        signals = strat.generate_signals(df)
        assert len(signals) == 1
        sig = signals[0]
        assert sig.direction == "long"
        assert sig.stop_loss == pytest.approx(1.099, abs=1e-6)   # or_low
        assert sig.entry_price == pytest.approx(1.108, abs=1e-6)

    def test_short_breakout_below_range(self):
        ohlc = [
            (1.100, 1.102, 1.099, 1.101),
            (1.101, 1.104, 1.100, 1.102),
            (1.102, 1.105, 1.100, 1.103),
            (1.103, 1.105, 1.101, 1.104),
            (1.104, 1.105, 1.101, 1.103),
            (1.103, 1.103, 1.090, 1.092),   # cassure baissière : close=1.092 < or_low=1.099
        ]
        df = _bars("2026-06-01 13:30", ohlc)
        strat = OpenUSORBStrategy(or_minutes=5, search_minutes=30)
        signals = strat.generate_signals(df)
        assert len(signals) == 1
        assert signals[0].direction == "short"
        assert signals[0].stop_loss == pytest.approx(1.105, abs=1e-6)   # or_high

    def test_no_breakout_no_signal(self):
        ohlc = [
            (1.100, 1.102, 1.099, 1.101),
            (1.101, 1.104, 1.100, 1.102),
            (1.102, 1.105, 1.100, 1.103),
            (1.103, 1.105, 1.101, 1.104),
            (1.104, 1.105, 1.101, 1.103),
            (1.103, 1.104, 1.101, 1.102),   # reste dans le range
        ]
        df = _bars("2026-06-01 13:30", ohlc)
        strat = OpenUSORBStrategy(or_minutes=5, search_minutes=30)
        assert strat.generate_signals(df) == []

    def test_one_signal_per_day(self):
        base = [
            (1.100, 1.102, 1.099, 1.101),
            (1.101, 1.104, 1.100, 1.102),
            (1.102, 1.105, 1.100, 1.103),
            (1.103, 1.105, 1.101, 1.104),
            (1.104, 1.105, 1.101, 1.103),
            (1.103, 1.110, 1.103, 1.108),
            (1.108, 1.120, 1.108, 1.118),   # deuxième cassure encore plus haute — doit être ignorée
        ]
        df = _bars("2026-06-01 13:30", base)
        strat = OpenUSORBStrategy(or_minutes=5, search_minutes=30)
        assert len(strat.generate_signals(df)) == 1

    def test_no_signal_before_window(self):
        # Toute la séquence tombe avant 09:30 ET (13:00-13:05 UTC = 09:00-09:05 ET)
        ohlc = [
            (1.100, 1.102, 1.099, 1.101),
            (1.101, 1.104, 1.100, 1.102),
            (1.102, 1.105, 1.100, 1.103),
            (1.103, 1.105, 1.101, 1.104),
            (1.104, 1.105, 1.101, 1.103),
            (1.103, 1.110, 1.103, 1.108),
        ]
        df = _bars("2026-06-01 13:00", ohlc)
        strat = OpenUSORBStrategy(or_minutes=5, search_minutes=30)
        assert strat.generate_signals(df) == []

    def test_no_signal_on_weekend(self):
        ohlc = [
            (1.100, 1.102, 1.099, 1.101),
            (1.101, 1.104, 1.100, 1.102),
            (1.102, 1.105, 1.100, 1.103),
            (1.103, 1.105, 1.101, 1.104),
            (1.104, 1.105, 1.101, 1.103),
            (1.103, 1.110, 1.103, 1.108),
        ]
        df = _bars("2026-06-06 13:30", ohlc)   # samedi
        strat = OpenUSORBStrategy(or_minutes=5, search_minutes=30)
        assert strat.generate_signals(df) == []

    def test_rejects_tz_aware_index(self):
        df = _bars("2026-06-01 13:30", [(1.1, 1.1, 1.1, 1.1)] * 6)
        df.index = df.index.tz_localize("UTC")
        strat = OpenUSORBStrategy(or_minutes=5, search_minutes=30)
        with pytest.raises(ValueError):
            strat.generate_signals(df)
