"""Tests des stratégies options binaires : logique des signaux et absence de lookahead."""
import numpy as np
import pandas as pd
import pytest

from zeus.strategy.base import SignalType
from zeus.strategy.binary_options import in_utc_hours, to_signals
from zeus.strategy.binary_options import mean_reversion as mr
from zeus.strategy.binary_options import momentum as mo


def _frame(closes, start="2024-03-04 02:00", opens=None):
    idx = pd.date_range(start, periods=len(closes), freq="1min")
    closes = np.asarray(closes, dtype=float)
    opens = np.r_[closes[0], closes[:-1]] if opens is None else np.asarray(opens, dtype=float)
    return pd.DataFrame({"open": opens, "high": np.maximum(opens, closes),
                         "low": np.minimum(opens, closes), "close": closes}, index=idx)


def _random_walk(n=3000, seed=0, start="2024-03-04 00:00"):
    rng = np.random.default_rng(seed)
    return _frame(1.1 + np.cumsum(rng.normal(0, 1e-4, n)), start=start)


# ── mean reversion ────────────────────────────────────────────────────────────

def test_mean_reversion_spike_up_gives_put():
    closes = [1.0 + 0.0001 * (i % 2) for i in range(30)] + [1.01]
    d = mr.directions(_frame(closes), mr.MeanReversionConfig())
    assert d.iat[-1] == -1 and (d.iloc[:-1] == 0).all()


def test_mean_reversion_spike_down_gives_call():
    closes = [1.0 + 0.0001 * (i % 2) for i in range(30)] + [0.99]
    assert mr.directions(_frame(closes), mr.MeanReversionConfig()).iat[-1] == 1


def test_mean_reversion_respects_session():
    closes = [1.0 + 0.0001 * (i % 2) for i in range(30)] + [1.01]
    d = mr.directions(_frame(closes, start="2024-03-04 12:00"), mr.MeanReversionConfig())
    assert (d == 0).all()


def test_rsi_bounds():
    r = mr.rsi(_random_walk()["close"], 7).dropna()
    assert r.between(0, 100).all()


# ── momentum ──────────────────────────────────────────────────────────────────

def test_momentum_bullish_streak_gives_call():
    closes = [1.0] * 20 + [1.001, 1.002, 1.003]
    d = mo.directions(_frame(closes, start="2024-03-04 09:00"), mo.MomentumConfig(streak=3))
    assert d.iat[-1] == 1 and d.iat[-2] == 0


def test_momentum_bearish_streak_gives_put():
    closes = [1.0] * 20 + [0.999, 0.998, 0.997]
    d = mo.directions(_frame(closes, start="2024-03-04 09:00"), mo.MomentumConfig(streak=3))
    assert d.iat[-1] == -1


def test_momentum_weak_bodies_are_ignored():
    closes = [1.0 + 0.001 * (i % 2) for i in range(20)] + [1.00101, 1.00102, 1.00103]
    d = mo.directions(_frame(closes, start="2024-03-04 09:00"), mo.MomentumConfig(min_body_atr=0.5))
    assert d.iat[-1] == 0


# ── propriétés communes ───────────────────────────────────────────────────────

@pytest.mark.parametrize("module,config", [
    (mr, mr.MeanReversionConfig(session_start_utc=0, session_end_utc=24, bb_std=2.0)),
    (mo, mo.MomentumConfig(session_start_utc=0, session_end_utc=24, min_body_atr=0.3)),
])
def test_no_lookahead(module, config):
    """Le signal d'une bougie ne dépend pas des bougies futures."""
    df = _random_walk()
    full = module.directions(df, config)
    assert full.abs().sum() > 0
    for cut in (500, 1500, 2999):
        partial = module.directions(df.iloc[: cut + 1], config)
        pd.testing.assert_series_equal(partial, full.iloc[: cut + 1])


def test_to_signals_maps_directions():
    d = pd.Series([0, 1, 0, -1], index=pd.date_range("2024-01-01", periods=4, freq="1min"))
    sigs = to_signals(d, "r")
    assert [(s.bar_index, s.type) for s in sigs] == [(1, SignalType.LONG), (3, SignalType.SHORT)]


def test_in_utc_hours_wraps_midnight():
    idx = pd.DatetimeIndex(["2024-01-01 22:00", "2024-01-01 03:00", "2024-01-01 12:00"])
    assert list(in_utc_hours(idx, 21, 6)) == [True, True, False]


@pytest.mark.parametrize("factory", [
    lambda: mr.MeanReversionConfig(rsi_low=60), lambda: mr.MeanReversionConfig(bb_std=0),
    lambda: mo.MomentumConfig(streak=0), lambda: mo.MomentumConfig(session_end_utc=25),
])
def test_invalid_configs_are_refused(factory):
    with pytest.raises(ValueError):
        factory()
