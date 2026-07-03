"""
Unit tests for zeus.strategy.convergence_strategy.ConvergenceStrategy.

Test classes
------------
TestConvergenceSignal     — ConvergenceSignal dataclass properties
TestConvergenceStrategyInit — parameter storage
TestRunGuard              — early-return for insufficient data
TestFilterContributions   — disabling individual filters changes signal count
TestFvgObOverlap          — FVG + OB zone overlap detection logic
TestLtfSweepIntegration   — LTF sweep filter effect on signals
TestTradeable             — ConvergenceSignal satisfies Tradeable protocol
TestEndToEnd              — full pipeline smoke test (synthetic deterministic data)
"""
from __future__ import annotations

from unittest.mock import patch
from typing import Any

import numpy as np
import pandas as pd
import pytest

from zeus.backtest.sd_simulation import Tradeable
from zeus.strategy.convergence_strategy import ConvergenceSignal, ConvergenceStrategy
from zeus.strategy.smc.fvg import FairValueGap
from zeus.strategy.smc.order_block import OrderBlock
from zeus.strategy.smc.pivot import BULLISH, BEARISH


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _m15_df(n: int = 400, seed: int = 7, trend: float = 0.1) -> pd.DataFrame:
    """
    Deterministic trending M15 OHLCV DataFrame with DatetimeIndex.

    trend > 0 creates a mild uptrend so D1 EMA(200) regime is satisfied.
    """
    rng    = np.random.default_rng(seed)
    base   = 1900.0
    closes = base + np.cumsum(rng.normal(trend, 0.8, n))
    highs  = closes + rng.uniform(0.2, 1.5, n)
    lows   = closes - rng.uniform(0.2, 1.5, n)
    opens  = np.roll(closes, 1)
    opens[0] = closes[0]
    vols   = rng.uniform(100, 1000, n)

    idx = pd.date_range("2025-01-06 08:00", periods=n, freq="15min", tz=None)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=idx,
    )


def _make_signal(
    entry: float = 1920.0,
    sl: float = 1910.0,
    tp: float = 1940.0,
    bar_index: int = 50,
) -> ConvergenceSignal:
    return ConvergenceSignal(
        direction     = "long",
        entry_price   = entry,
        stop_loss     = sl,
        take_profit   = tp,
        risk_reward   = 2.0,
        formed_at     = pd.Timestamp("2025-01-06 09:00"),
        bar_index     = bar_index,
        ob_bar        = 45,
        ob_high       = 1925.0,
        ob_low        = 1915.0,
        fvg_bar       = 47,
        fvg_top       = 1922.0,
        fvg_bottom    = 1917.0,
        ltf_sweep_bar = 49,
        zone_score    = 8.3,
    )


# ──────────────────────────────────────────────────────────────────────────────
# ConvergenceSignal
# ──────────────────────────────────────────────────────────────────────────────

class TestConvergenceSignal:
    def test_risk_property(self):
        sig = _make_signal(entry=1920.0, sl=1910.0)
        assert sig.risk == pytest.approx(10.0)

    def test_reward_property(self):
        sig = _make_signal(entry=1920.0, tp=1940.0)
        assert sig.reward == pytest.approx(20.0)

    def test_direction_is_long(self):
        sig = _make_signal()
        assert sig.direction == "long"

    def test_zone_score_default(self):
        sig = _make_signal()
        assert sig.zone_score == pytest.approx(8.3)

    def test_frozen_dataclass(self):
        sig = _make_signal()
        with pytest.raises(Exception):
            sig.direction = "short"  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────────────
# Tradeable protocol
# ──────────────────────────────────────────────────────────────────────────────

class TestTradeable:
    def test_convergence_signal_satisfies_tradeable(self):
        sig = _make_signal()
        assert isinstance(sig, Tradeable), (
            "ConvergenceSignal must satisfy the Tradeable structural protocol"
        )

    def test_required_fields_present(self):
        sig = _make_signal()
        assert hasattr(sig, "direction")
        assert hasattr(sig, "bar_index")
        assert hasattr(sig, "stop_loss")
        assert hasattr(sig, "risk_reward")
        assert hasattr(sig, "formed_at")
        assert hasattr(sig, "zone_score")


# ──────────────────────────────────────────────────────────────────────────────
# Strategy init
# ──────────────────────────────────────────────────────────────────────────────

class TestConvergenceStrategyInit:
    def test_defaults(self):
        s = ConvergenceStrategy()
        assert s.pivot_size         == 10
        assert s.long_only          is True
        assert s.use_d1_ema         is True
        assert s.d1_ema_span        == 200
        assert s.use_h4_trend       is True
        assert s.h4_ema_span        == 50
        assert s.use_killzone       is True
        assert s.use_ltf_sweep      is True
        assert s.ltf_sweep_lookback == 5
        assert s.risk_reward        == pytest.approx(2.0)
        assert s.signal_cooldown    == 12

    def test_custom_params(self):
        s = ConvergenceStrategy(pivot_size=5, risk_reward=1.5, signal_cooldown=6)
        assert s.pivot_size      == 5
        assert s.risk_reward     == pytest.approx(1.5)
        assert s.signal_cooldown == 6


# ──────────────────────────────────────────────────────────────────────────────
# Insufficient data guard
# ──────────────────────────────────────────────────────────────────────────────

class TestRunGuard:
    def test_empty_df_returns_empty(self):
        s   = ConvergenceStrategy()
        df  = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df.index = pd.DatetimeIndex([])
        assert s.run(df) == []

    def test_too_few_bars_returns_empty(self):
        s   = ConvergenceStrategy()
        df  = _m15_df(n=10)
        assert s.run(df) == []

    def test_returns_list(self):
        s   = ConvergenceStrategy()
        df  = _m15_df(n=400)
        res = s.run(df)
        assert isinstance(res, list)

    def test_signals_are_convergence_signal_instances(self):
        s   = ConvergenceStrategy()
        df  = _m15_df(n=400)
        res = s.run(df)
        for sig in res:
            assert isinstance(sig, ConvergenceSignal)


# ──────────────────────────────────────────────────────────────────────────────
# FVG + OB overlap helper
# ──────────────────────────────────────────────────────────────────────────────

class TestFvgObOverlap:
    def _fvg(self, top: float, bottom: float) -> FairValueGap:
        return FairValueGap(bar_index=10, top=top, bottom=bottom, direction=BULLISH)

    def _ob(self, high: float, low: float) -> OrderBlock:
        return OrderBlock(
            bar_index=5, high=high, low=low,
            direction=BULLISH, is_internal=False,
            detected_at=8, mitigated_at=-1,
        )

    def test_full_overlap_found(self):
        ob  = self._ob(high=1920.0, low=1910.0)
        fvg = self._fvg(top=1918.0, bottom=1912.0)   # entirely inside OB
        result = ConvergenceStrategy._find_overlapping_fvg([fvg], ob)
        assert result is not None

    def test_partial_overlap_found(self):
        ob  = self._ob(high=1920.0, low=1910.0)
        fvg = self._fvg(top=1925.0, bottom=1915.0)   # top above OB.high, bottom inside
        result = ConvergenceStrategy._find_overlapping_fvg([fvg], ob)
        assert result is not None

    def test_fvg_above_ob_not_found(self):
        ob  = self._ob(high=1910.0, low=1900.0)
        fvg = self._fvg(top=1925.0, bottom=1915.0)   # entirely above OB
        result = ConvergenceStrategy._find_overlapping_fvg([fvg], ob)
        assert result is None

    def test_fvg_below_ob_not_found(self):
        ob  = self._ob(high=1910.0, low=1900.0)
        fvg = self._fvg(top=1898.0, bottom=1890.0)   # entirely below OB
        result = ConvergenceStrategy._find_overlapping_fvg([fvg], ob)
        assert result is None

    def test_bearish_fvg_not_matched(self):
        ob  = self._ob(high=1920.0, low=1910.0)
        fvg = FairValueGap(bar_index=10, top=1918.0, bottom=1912.0, direction=BEARISH)
        result = ConvergenceStrategy._find_overlapping_fvg([fvg], ob)
        assert result is None

    def test_most_recent_fvg_returned(self):
        ob   = self._ob(high=1920.0, low=1910.0)
        fvg1 = FairValueGap(bar_index=8,  top=1918.0, bottom=1912.0, direction=BULLISH)
        fvg2 = FairValueGap(bar_index=12, top=1916.0, bottom=1911.0, direction=BULLISH)
        result = ConvergenceStrategy._find_overlapping_fvg([fvg1, fvg2], ob)
        assert result is not None
        assert result.bar_index == 12


# ──────────────────────────────────────────────────────────────────────────────
# Filter contribution tests
# ──────────────────────────────────────────────────────────────────────────────

class TestFilterContributions:
    """
    Disabling a restrictive filter should not decrease the signal count
    (it can only stay the same or increase).  We use a fixed dataset and
    verify the monotonicity property between baseline and filter-off variants.
    """
    _df: pd.DataFrame | None = None

    @classmethod
    def _get_df(cls) -> pd.DataFrame:
        if cls._df is None:
            cls._df = _m15_df(n=600, trend=0.15)
        return cls._df

    def _count(self, **kwargs: Any) -> int:
        df  = self._get_df()
        cfg = {
            "pivot_size": 10, "long_only": True, "use_d1_ema": False,
            "use_h4_trend": False, "use_killzone": False, "use_ltf_sweep": False,
            "require_bullish_bar": False, "signal_cooldown": 1,
            "risk_reward": 2.0,
        }
        cfg.update(kwargs)
        return len(ConvergenceStrategy(**cfg).run(df))

    def test_killzone_off_gte_on(self):
        n_on  = self._count(use_killzone=True)
        n_off = self._count(use_killzone=False)
        assert n_off >= n_on

    def test_ltf_sweep_off_gte_on(self):
        n_on  = self._count(use_ltf_sweep=True)
        n_off = self._count(use_ltf_sweep=False)
        assert n_off >= n_on

    def test_bullish_bar_off_gte_on(self):
        n_on  = self._count(require_bullish_bar=True)
        n_off = self._count(require_bullish_bar=False)
        assert n_off >= n_on

    def test_cooldown_increase_reduces_signals(self):
        n_short = self._count(signal_cooldown=1)
        n_long  = self._count(signal_cooldown=48)
        # Longer cooldown must not produce more signals
        assert n_long <= n_short


# ──────────────────────────────────────────────────────────────────────────────
# Signal integrity
# ──────────────────────────────────────────────────────────────────────────────

class TestSignalIntegrity:
    """Every emitted signal must have geometrically correct SL/TP."""

    def test_sl_below_entry_for_longs(self):
        df  = _m15_df(n=600, trend=0.2)
        s   = ConvergenceStrategy(use_d1_ema=False, use_h4_trend=False,
                                   use_killzone=False, use_ltf_sweep=False,
                                   require_bullish_bar=False, signal_cooldown=1)
        for sig in s.run(df):
            assert sig.stop_loss < sig.entry_price, (
                f"SL {sig.stop_loss} must be below entry {sig.entry_price} for long"
            )

    def test_tp_above_entry_for_longs(self):
        df  = _m15_df(n=600, trend=0.2)
        s   = ConvergenceStrategy(use_d1_ema=False, use_h4_trend=False,
                                   use_killzone=False, use_ltf_sweep=False,
                                   require_bullish_bar=False, signal_cooldown=1)
        for sig in s.run(df):
            assert sig.take_profit > sig.entry_price, (
                f"TP {sig.take_profit} must be above entry {sig.entry_price} for long"
            )

    def test_risk_reward_ratio_correct(self):
        df  = _m15_df(n=600, trend=0.2)
        s   = ConvergenceStrategy(use_d1_ema=False, use_h4_trend=False,
                                   use_killzone=False, use_ltf_sweep=False,
                                   require_bullish_bar=False, signal_cooldown=1,
                                   risk_reward=2.0)
        for sig in s.run(df):
            rr = sig.reward / sig.risk
            assert rr == pytest.approx(2.0, rel=0.01), (
                f"Expected RR≈2.0, got {rr:.3f}"
            )

    def test_zone_score_in_range(self):
        df  = _m15_df(n=600, trend=0.2)
        s   = ConvergenceStrategy(use_d1_ema=False, use_h4_trend=False,
                                   use_killzone=False, use_ltf_sweep=False,
                                   require_bullish_bar=False, signal_cooldown=1)
        for sig in s.run(df):
            assert 0.0 <= sig.zone_score <= 10.0

    def test_bar_index_monotone(self):
        df  = _m15_df(n=600, trend=0.2)
        s   = ConvergenceStrategy(use_d1_ema=False, use_h4_trend=False,
                                   use_killzone=False, use_ltf_sweep=False,
                                   require_bullish_bar=False, signal_cooldown=1)
        sigs = s.run(df)
        bars = [sig.bar_index for sig in sigs]
        assert bars == sorted(bars), "Signals must be in chronological order"

    def test_ob_within_signal_bounds(self):
        df  = _m15_df(n=600, trend=0.2)
        s   = ConvergenceStrategy(use_d1_ema=False, use_h4_trend=False,
                                   use_killzone=False, use_ltf_sweep=False,
                                   require_bullish_bar=False, signal_cooldown=1)
        for sig in s.run(df):
            # SL must be at or below OB low (OB low − buffer)
            assert sig.stop_loss <= sig.ob_low + 1e-6, (
                f"SL {sig.stop_loss} must be at/below OB low {sig.ob_low}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Simulation engine compatibility
# ──────────────────────────────────────────────────────────────────────────────

class TestSimulationCompatibility:
    """ConvergenceSignals can be passed directly to simulate_all."""

    def test_simulate_all_accepts_convergence_signals(self):
        from zeus.backtest.sd_simulation import simulate_all

        df   = _m15_df(n=600, trend=0.2)
        s    = ConvergenceStrategy(
            use_d1_ema=False, use_h4_trend=False,
            use_killzone=False, use_ltf_sweep=False,
            require_bullish_bar=False, signal_cooldown=1,
        )
        sigs = s.run(df)

        if not sigs:
            pytest.skip("No signals generated on this synthetic data — increase n or trend.")

        results, n_expired = simulate_all(
            sigs, df, initial_equity=10_000.0, risk_pct=0.01,
            spread=0.30, tp1_r=0.6, tp1_size=0.5,
        )
        assert isinstance(results, list)
        assert isinstance(n_expired, int)

    def test_compute_metrics_runs(self):
        from zeus.backtest.sd_simulation import compute_metrics, simulate_all

        df   = _m15_df(n=600, trend=0.2)
        s    = ConvergenceStrategy(
            use_d1_ema=False, use_h4_trend=False,
            use_killzone=False, use_ltf_sweep=False,
            require_bullish_bar=False, signal_cooldown=1,
        )
        sigs = s.run(df)
        if not sigs:
            pytest.skip("No signals generated.")

        results, n_exp = simulate_all(sigs, df, initial_equity=10_000.0, tp1_r=0.6)
        m = compute_metrics(results, 10_000.0, len(sigs), n_exp)
        assert "win_rate"  in m
        assert "n_trades"  in m
        assert "total_r"   in m
        assert "max_dd_pct" in m
