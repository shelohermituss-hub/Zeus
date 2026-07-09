"""Tests de ConfluenceScalpStrategy — preset "9-10 confirmations" (A++)
construit sur l'infrastructure SMC existante (mtf_strategy.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zeus.strategy.base import Signal, SignalType
from zeus.strategy.confluence import PatternGrade
from zeus.strategy.confluence_scalp_strategy import ConfluenceScalpStrategy


def _ohlcv(n: int = 200, seed: int = 0, freq: str = "1h",
           start: str = "2026-01-01") -> pd.DataFrame:
    rng    = np.random.default_rng(seed)
    closes = 3000.0 + np.cumsum(rng.normal(0, 1, n))
    highs  = closes + rng.uniform(0.5, 3, n)
    lows   = closes - rng.uniform(0.5, 3, n)
    opens  = np.roll(closes, 1); opens[0] = closes[0]
    vols   = rng.uniform(10, 100, n)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=pd.date_range(start, periods=n, freq=freq),
    )


def _make_strategy(**kwargs) -> ConfluenceScalpStrategy:
    return ConfluenceScalpStrategy(
        df_htf_1h=_ohlcv(200, seed=1, freq="1h"),
        df_mtf_15m=_ohlcv(800, seed=2, freq="15min"),
        df_daily=_ohlcv(60, seed=3, freq="D"),
        **kwargs,
    )


class TestAllConfirmationsEnabled:
    """Vérifie que chacune des 9-10 confirmations documentées est bien
    câblée sur le gate correspondant, activé par défaut (contrairement à
    ScalpSMCStrategy qui les laisse OFF pour la plupart)."""

    def test_ote_fibonacci_enabled(self):
        assert _make_strategy()._require_ote is True

    def test_poc_zone_confluence_enabled(self):
        assert _make_strategy()._require_poc_zone_confluence is True

    def test_weekly_bias_enabled(self):
        assert _make_strategy()._require_weekly_bias is True

    def test_session_sweep_enabled(self):
        assert _make_strategy()._require_session_sweep is True

    def test_asian_sweep_enabled(self):
        assert _make_strategy()._require_asian_sweep is True

    def test_entry_fvg_enabled(self):
        assert _make_strategy()._require_entry_fvg is True

    def test_ltf_sweep_enabled(self):
        assert _make_strategy()._require_ltf_sweep is True

    def test_entry_pattern_enabled(self):
        assert _make_strategy()._require_entry_pattern is True

    def test_choch_candle_enabled(self):
        assert _make_strategy()._require_choch_candle is True

    def test_pd_filter_enabled(self):
        assert _make_strategy()._require_pd_filter is True

    def test_daily_bias_enabled(self):
        assert _make_strategy()._require_daily_bias is True

    def test_clean_approach_enabled(self):
        assert _make_strategy()._require_clean_approach is True

    def test_killzone_only_enabled(self):
        assert _make_strategy()._killzone_only is True

    def test_default_min_grade_is_b(self):
        assert _make_strategy()._min_grade == PatternGrade.B

    def test_custom_min_grade(self):
        s = _make_strategy(min_grade=PatternGrade.A)
        assert s._min_grade == PatternGrade.A


class TestSignalLabel:
    def test_signal_reason_has_conf9_label(self):
        from unittest.mock import patch

        fake_sig = Signal(SignalType.LONG, 0.8, "MTF grade=B score=7/10 zone=OB SL=6.0pips", 10)
        s = _make_strategy()
        with patch.object(s.__class__.__bases__[0], "generate_signal", return_value=fake_sig):
            sig = s.generate_signal(_ohlcv(50, freq="1min"), 10)
        assert "CONF9 grade=" in sig.reason
        assert "MTF grade=" not in sig.reason

    def test_none_signal_label_unchanged(self):
        from unittest.mock import patch

        none_sig = Signal(SignalType.NONE, 0.0, "outside killzone", 5)
        s = _make_strategy()
        with patch.object(s.__class__.__bases__[0], "generate_signal", return_value=none_sig):
            sig = s.generate_signal(_ohlcv(50, freq="1min"), 5)
        assert sig.type == SignalType.NONE
        assert sig.reason == "outside killzone"


class TestFailClosed:
    def test_rejects_outside_killzone(self):
        s = _make_strategy()
        df = _ohlcv(50, freq="1min", start="2026-01-05 03:00")
        sig = s.generate_signal(df, 49)
        assert sig.type == SignalType.NONE

    def test_no_crash_on_random_data(self):
        """Smoke test: with every gate stacked on random OHLC noise, the
        strategy must never raise — and realistically should almost never
        fire (that's the whole point of stacking 9-10 confirmations)."""
        s = _make_strategy()
        df = _ohlcv(300, freq="1min", start="2026-01-05 07:00")
        fired = 0
        for i in range(60, len(df)):
            sig = s.generate_signal(df, i)
            if sig.type != SignalType.NONE:
                fired += 1
        # Not asserting fired == 0 (random data COULD satisfy every gate by
        # chance) — just documents that stacking every gate is extremely
        # restrictive on pure noise.
        assert fired <= 2
