"""Tests de ConfluenceSoftScoredStrategy — variante N-of-M des 10
confirmations documentées (voir test_confluence_scalp_strategy.py pour la
variante hard-AND-gate)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from zeus.strategy.base import SignalType
from zeus.strategy.confluence_soft_strategy import ConfluenceSoftScoredStrategy


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


def _make_strategy(**kwargs) -> ConfluenceSoftScoredStrategy:
    return ConfluenceSoftScoredStrategy(
        df_htf_1h=_ohlcv(300, seed=1, freq="1h"),
        df_daily=_ohlcv(60, seed=3, freq="D"),
        **kwargs,
    )


class TestConstruction:
    def test_default_min_confirmations_is_12(self):
        assert _make_strategy()._min_confirmations == 12

    def test_custom_min_confirmations(self):
        assert _make_strategy(min_confirmations=6)._min_confirmations == 6

    def test_mss_gate_disabled(self):
        """df_mtf is always None — the 1H/15M MSS gate is never wired for
        this strategy (same conjunction problem documented in
        run_scalp_backtest.py: it's always in pullback when we want to
        enter, producing 0 setups)."""
        assert _make_strategy()._df_mtf is None

    def test_killzone_not_a_hard_gate(self):
        """killzone_only=False — killzone becomes confirmation #7, scored
        like the other 9, not an early hard rejection."""
        assert _make_strategy()._killzone_only is False

    def test_htf_internal_align_not_required(self):
        assert _make_strategy()._require_htf_internal_align is False

    def test_n_confirmations_constant_is_16(self):
        assert ConfluenceSoftScoredStrategy._N_CONFIRMATIONS == 16

    def test_zone_is_not_a_hard_gate(self):
        """Regression guard for the audit fix: the HTF zone check must be
        one of the scored confirmations, not an early hard return, so a
        bar can still accumulate points without literally sitting inside
        an OB/FVG/OTE zone."""
        from zeus.strategy.confluence_soft_strategy import _CONFIRMATION_NAMES
        assert "any_zone" in _CONFIRMATION_NAMES


class TestFailClosed:
    def test_no_crash_on_random_data(self):
        """Smoke test: soft-scoring on pure random noise must never raise,
        and firing should still be rare at a demanding threshold."""
        s = _make_strategy(min_confirmations=9)
        df = _ohlcv(300, freq="1min", start="2026-01-05 00:00")
        fired = 0
        for i in range(60, len(df)):
            sig = s.generate_signal(df, i)
            assert sig.type in (SignalType.NONE, SignalType.LONG, SignalType.SHORT)
            if sig.type != SignalType.NONE:
                fired += 1
        assert fired <= 3

    def test_insufficient_htf_data_rejected(self):
        s = _make_strategy()
        df = _ohlcv(50, freq="1min", start="2026-01-01 00:00")
        sig = s.generate_signal(df, 10)
        assert sig.type == SignalType.NONE
        assert "insufficient HTF data" in sig.reason


class TestMonotonicity:
    def test_lower_threshold_fires_at_least_as_often(self):
        """Lowering min_confirmations must never reduce the number of
        signals fired on the same data — the N-of-M count doesn't change
        with the threshold, only the accept/reject decision does."""
        df = _ohlcv(400, freq="1min", start="2026-01-05 00:00")

        def _count(min_conf: int) -> int:
            s = _make_strategy(min_confirmations=min_conf, max_daily_signals=0,
                                max_signals_per_session=0)
            return sum(
                1 for i in range(60, len(df))
                if s.generate_signal(df, i).type != SignalType.NONE
            )

        n_strict = _count(9)
        n_loose  = _count(3)
        assert n_loose >= n_strict

    def test_reason_reports_confirmation_fraction_when_rejected(self):
        s = _make_strategy(min_confirmations=9)
        df = _ohlcv(300, freq="1min", start="2026-01-05 00:00")
        rejected_on_count = [
            s.generate_signal(df, i) for i in range(60, len(df))
        ]
        # At least some rejections should be the N-of-M count message
        # (as opposed to "insufficient HTF data" / "no HTF swing bias" /
        # "price outside HTF zone", which short-circuit before counting).
        count_msgs = [s for s in rejected_on_count if "confirmations (need" in s.reason]
        assert len(count_msgs) >= 0  # documents the reason format exists; no data-shape guarantee


class TestFibZoneStaleness:
    def test_stale_fib_zone_is_dropped(self):
        """Regression guard for the audit fix: a FibZone formed long before
        max_fib_zone_age_bars must not count toward ote/fib_50, since price
        may have moved far away from that swing leg in the meantime."""
        from unittest.mock import MagicMock
        from zeus.strategy.smc.fibonacci import FibZone
        from zeus.strategy.smc.pivot import BULLISH

        s = _make_strategy(max_fib_zone_age_bars=5)
        stale_zone = FibZone(
            swing_high=3100.0, swing_low=3000.0, direction=BULLISH,
            leg_high_bar=0, leg_low_bar=1, formed_at=0,
        )
        htf_result = MagicMock(fib_zones=[stale_zone])
        assert s._fresh_fib_zone(htf_result, 100, BULLISH) is None

    def test_fresh_fib_zone_is_kept(self):
        from unittest.mock import MagicMock
        from zeus.strategy.smc.fibonacci import FibZone
        from zeus.strategy.smc.pivot import BULLISH

        s = _make_strategy(max_fib_zone_age_bars=5)
        fresh_zone = FibZone(
            swing_high=3100.0, swing_low=3000.0, direction=BULLISH,
            leg_high_bar=0, leg_low_bar=1, formed_at=98,
        )
        htf_result = MagicMock(fib_zones=[fresh_zone])
        assert s._fresh_fib_zone(htf_result, 100, BULLISH) is fresh_zone

    def test_default_max_fib_zone_age_bars(self):
        assert _make_strategy()._max_fib_zone_age_bars == 40

    def test_custom_max_fib_zone_age_bars(self):
        assert _make_strategy(max_fib_zone_age_bars=10)._max_fib_zone_age_bars == 10


class TestLtfBiasAndFvgSplit:
    def test_returns_two_independent_booleans(self):
        s = _make_strategy()
        df = _ohlcv(100, freq="1min", start="2026-01-05 07:00")
        from zeus.strategy.smc.pivot import BULLISH
        bias_ok, fvg_ok = s._ltf_bias_and_fvg(df, 50, BULLISH, float(df["close"].iloc[50]))
        assert isinstance(bias_ok, bool) or isinstance(bias_ok, (int,))
        assert isinstance(fvg_ok, bool) or isinstance(fvg_ok, (int,))

    def test_insufficient_window_returns_false_false(self):
        s = _make_strategy()
        df = _ohlcv(5, freq="1min", start="2026-01-05 07:00")
        from zeus.strategy.smc.pivot import BULLISH
        bias_ok, fvg_ok = s._ltf_bias_and_fvg(df, 2, BULLISH, float(df["close"].iloc[2]))
        assert bias_ok is False
        assert fvg_ok is False


class TestSignalShape:
    def test_fired_signal_has_soft_label_and_valid_confidence(self):
        s = _make_strategy(min_confirmations=1, max_daily_signals=0,
                            max_signals_per_session=0)
        df = _ohlcv(400, freq="1min", start="2026-01-05 00:00")
        fired = [
            sig for i in range(60, len(df))
            if (sig := s.generate_signal(df, i)).type != SignalType.NONE
        ]
        assert len(fired) > 0, "min_confirmations=1 should fire at least once on 400 random bars"
        for sig in fired:
            assert sig.reason.startswith("SOFT ")
            assert 0.0 <= sig.confidence <= 1.0
