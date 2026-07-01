"""
Unit tests for V3 quality filters:
  - require_ote      (gate 4.1 — OTE zone mandatory)
  - require_daily_bias (gate 3.5 strict — neutral day = no trade)
  - ScalpSMCStrategy V3 defaults (score=5, SL=20/30, OTE+daily required)
"""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from zeus.strategy.smc.pivot import BULLISH, BEARISH
from zeus.strategy.base import SignalType
from zeus.strategy.mtf_strategy import MTFSMCStrategy
from zeus.strategy.scalp_strategy import ScalpSMCStrategy


# ── Helpers ───────────────────────────────────────────────────────────────────

def _flat_df(n: int = 300, price: float = 2000.0, freq: str = "1h") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open":   [price] * n,
            "high":   [price + 1] * n,
            "low":    [price - 1] * n,
            "close":  [price] * n,
            "volume": [1000.0] * n,
        },
        index=pd.date_range("2026-01-06 08:00", periods=n, freq=freq, tz="UTC"),
    )


def _make_scalp(**kwargs) -> ScalpSMCStrategy:
    return ScalpSMCStrategy(
        df_htf_1h=_flat_df(200, freq="1h"),
        df_mtf_15m=_flat_df(800, freq="15min"),
        df_daily=_flat_df(60, freq="D"),
        **kwargs,
    )


def _make_mtf(**kwargs) -> MTFSMCStrategy:
    return MTFSMCStrategy(
        df_htf=_flat_df(200, freq="1h"),
        df_daily=_flat_df(60, freq="D"),
        **kwargs,
    )


# ── Tests: ScalpSMCStrategy V3 defaults ──────────────────────────────────────

class TestScalpV3Defaults:

    def test_sl_pips_default_20(self):
        s = _make_scalp()
        assert s._sl_pips == pytest.approx(20.0)

    def test_max_sl_pips_default_30(self):
        s = _make_scalp()
        assert s._max_sl_pips == pytest.approx(30.0)

    def test_min_htf_score_default_5(self):
        s = _make_scalp()
        assert s._min_htf_score == pytest.approx(5.0)

    def test_require_ote_enabled_by_default(self):
        s = _make_scalp()
        assert s._require_ote is True

    def test_require_daily_bias_enabled_by_default(self):
        s = _make_scalp()
        assert s._require_daily_bias is True

    def test_require_ltf_sweep_still_on(self):
        s = _make_scalp()
        assert s._require_ltf_sweep is True

    def test_require_clean_approach_still_on(self):
        s = _make_scalp()
        assert s._require_clean_approach is True

    def test_ote_can_be_disabled(self):
        s = _make_scalp(require_ote=False)
        assert s._require_ote is False

    def test_daily_bias_can_be_disabled(self):
        s = _make_scalp(require_daily_bias=False)
        assert s._require_daily_bias is False

    def test_sl_pips_overridable(self):
        s = _make_scalp(sl_pips=25.0, max_sl_pips=35.0)
        assert s._sl_pips == pytest.approx(25.0)
        assert s._max_sl_pips == pytest.approx(35.0)

    def test_min_score_overridable(self):
        s = _make_scalp(min_htf_score=6.0)
        assert s._min_htf_score == pytest.approx(6.0)


# ── Tests: require_ote gate (gate 4.1) ───────────────────────────────────────

def _fake_htf_result(swing_bias: int = BULLISH):
    """Build a minimal SMCResult with a defined swing bias."""
    from zeus.strategy.smc.indicator import SMCResult
    return SMCResult(swing_bias=swing_bias, internal_bias=swing_bias)


def _patched_signal(s: MTFSMCStrategy, price: float = 2000.0) -> "Signal":
    """Drive generate_signal past HTF-data checks by patching internals."""
    from zeus.strategy.confluence import ConfluenceScore, FactorResult

    cs = ConfluenceScore(
        direction=BULLISH, price=price, bar_index=5,
        factors=[FactorResult(i + 1, f"F{i+1}", True) for i in range(10)],
    )
    htf_result = _fake_htf_result(BULLISH)

    with (
        patch.object(s, "_last_htf_bar", return_value=100),
        patch.object(s, "_htf_analysis", return_value=htf_result),
        patch.object(s, "_mtf_mss_confirmed", return_value=True),
        patch.object(s, "_in_htf_zone", return_value=("OB", price - 15.0, price + 5.0)),
        patch.object(s, "_ltf_entry_confirmed", return_value=True),
        patch("zeus.strategy.mtf_strategy.best_confluence", return_value=cs),
        patch("zeus.strategy.mtf_strategy.get_daily_bias", return_value=BULLISH),
    ):
        df = _flat_df(50, price=price, freq="1min")
        return s.generate_signal(df, bar_index=49)


class TestRequireOTEGate:

    def test_ote_gate_off_passes_non_ote(self):
        """With require_ote=False, non-OTE price is NOT rejected by OTE gate."""
        from zeus.strategy.smc.fibonacci import FibZone

        # OTE for swing 1950→2050 is ~1962–1938 for BULLISH (61.8–78.6% retrace)
        # price=2020 is above OTE → would fail OTE check
        fake_fib = FibZone(
            swing_high=2050.0, swing_low=1950.0,
            direction=BULLISH,
            leg_high_bar=5, leg_low_bar=0, formed_at=5,
        )
        s = _make_mtf(require_ote=False, killzone_only=False)
        with patch("zeus.strategy.mtf_strategy.get_latest_fib_zone", return_value=fake_fib):
            sig = _patched_signal(s, price=2020.0)
        assert "OTE zone" not in sig.reason

    def test_ote_gate_on_rejects_non_ote(self):
        """With require_ote=True, price outside OTE zone must be rejected."""
        from zeus.strategy.smc.fibonacci import FibZone

        fake_fib = FibZone(
            swing_high=2050.0, swing_low=1950.0,
            direction=BULLISH,
            leg_high_bar=5, leg_low_bar=0, formed_at=5,
        )
        s = _make_mtf(require_ote=True, killzone_only=False)
        with patch("zeus.strategy.mtf_strategy.get_latest_fib_zone", return_value=fake_fib):
            sig = _patched_signal(s, price=2020.0)  # well above OTE band
        assert "OTE zone" in sig.reason

    def test_ote_gate_on_no_fib_zone_rejects(self):
        """No fib zone available → OTE gate fails closed."""
        s = _make_mtf(require_ote=True, killzone_only=False)
        with patch("zeus.strategy.mtf_strategy.get_latest_fib_zone", return_value=None):
            sig = _patched_signal(s, price=2000.0)
        assert "OTE zone" in sig.reason

    def test_mtf_ote_disabled_by_default(self):
        s = _make_mtf()
        assert s._require_ote is False


# ── Tests: require_daily_bias gate (gate 3.5 strict) ─────────────────────────

class TestRequireDailyBiasGate:

    def test_neutral_daily_bias_rejected_when_required(self):
        """When daily bias is 0 (neutral) and require_daily_bias=True → reject."""
        s = _make_mtf(require_daily_bias=True, killzone_only=False)
        with patch("zeus.strategy.mtf_strategy.get_daily_bias", return_value=0):
            with (
                patch.object(s, "_last_htf_bar", return_value=100),
                patch.object(s, "_htf_analysis", return_value=_fake_htf_result(BULLISH)),
                patch.object(s, "_mtf_mss_confirmed", return_value=True),
            ):
                df = _flat_df(50, freq="1min")
                sig = s.generate_signal(df, bar_index=49)
        assert "neutral" in sig.reason.lower()

    def test_neutral_daily_bias_allowed_when_not_required(self):
        """With require_daily_bias=False, neutral daily bias is not a rejection reason."""
        s = _make_mtf(require_daily_bias=False, killzone_only=False)
        with patch("zeus.strategy.mtf_strategy.get_daily_bias", return_value=0):
            with (
                patch.object(s, "_last_htf_bar", return_value=100),
                patch.object(s, "_htf_analysis", return_value=_fake_htf_result(BULLISH)),
                patch.object(s, "_mtf_mss_confirmed", return_value=True),
            ):
                df = _flat_df(50, freq="1min")
                sig = s.generate_signal(df, bar_index=49)
        assert "neutral" not in sig.reason.lower()

    def test_no_df_daily_with_require_fails_closed(self):
        """require_daily_bias=True but df_daily=None → fail closed."""
        s = MTFSMCStrategy(
            df_htf=_flat_df(200, freq="1h"),
            require_daily_bias=True,
            killzone_only=False,
        )
        with (
            patch.object(s, "_last_htf_bar", return_value=100),
            patch.object(s, "_htf_analysis", return_value=_fake_htf_result(BULLISH)),
            patch.object(s, "_mtf_mss_confirmed", return_value=True),
        ):
            df = _flat_df(50, freq="1min")
            sig = s.generate_signal(df, bar_index=49)
        assert "df_daily not provided" in sig.reason

    def test_conflicting_daily_bias_always_rejects(self):
        """Conflicting daily bias (db != direction) rejects even when require=False."""
        s = _make_mtf(require_daily_bias=False, killzone_only=False)
        with patch("zeus.strategy.mtf_strategy.get_daily_bias", return_value=BEARISH):
            with (
                patch.object(s, "_last_htf_bar", return_value=100),
                patch.object(s, "_htf_analysis", return_value=_fake_htf_result(BULLISH)),
                patch.object(s, "_mtf_mss_confirmed", return_value=True),
            ):
                df = _flat_df(50, freq="1min")
                sig = s.generate_signal(df, bar_index=49)
        assert "conflicts" in sig.reason.lower()

    def test_mtf_daily_bias_disabled_by_default(self):
        s = MTFSMCStrategy(df_htf=_flat_df(200, freq="1h"))
        assert s._require_daily_bias is False

    def test_scalp_daily_bias_enabled_by_default(self):
        s = _make_scalp()
        assert s._require_daily_bias is True


# ── Tests: require_entry_pattern gate (gate 6.4) ─────────────────────────────

def _df_with_hammer(price: float = 2000.0) -> pd.DataFrame:
    """49-bar flat df followed by a hammer candle at bar_index=49.
    Hammer: lower_wick=7, range=10 → 70% ≥ 60%; close in upper half."""
    flat = _flat_df(49, price=price, freq="1min")
    hammer = pd.DataFrame(
        {
            "open":   [price + 2],
            "high":   [price + 3],
            "low":    [price - 7],
            "close":  [price + 2],
            "volume": [1000.0],
        },
        index=[flat.index[-1] + pd.Timedelta(minutes=1)],
    )
    return pd.concat([flat, hammer])


def _patched_signal_df(s: MTFSMCStrategy, df: pd.DataFrame, price: float = 2000.0):
    """Same as _patched_signal but uses a caller-supplied LTF DataFrame."""
    from zeus.strategy.confluence import ConfluenceScore, FactorResult

    cs = ConfluenceScore(
        direction=BULLISH, price=price, bar_index=5,
        factors=[FactorResult(i + 1, f"F{i+1}", True) for i in range(10)],
    )
    htf_result = _fake_htf_result(BULLISH)

    with (
        patch.object(s, "_last_htf_bar", return_value=100),
        patch.object(s, "_htf_analysis", return_value=htf_result),
        patch.object(s, "_mtf_mss_confirmed", return_value=True),
        patch.object(s, "_in_htf_zone", return_value=("OB", price - 15.0, price + 5.0)),
        patch.object(s, "_ltf_entry_confirmed", return_value=True),
        patch("zeus.strategy.mtf_strategy.best_confluence", return_value=cs),
        patch("zeus.strategy.mtf_strategy.get_daily_bias", return_value=BULLISH),
        patch("zeus.strategy.mtf_strategy.ltf_liquidity_sweep", return_value=(True, "sweep ok")),
    ):
        return s.generate_signal(df, bar_index=49)


class TestRequireEntryPatternGate:

    def test_gate_off_doji_not_rejected(self):
        """With require_entry_pattern=False, a zero-range bar is not rejected by the gate."""
        s = _make_mtf(require_entry_pattern=False, killzone_only=False)
        df = _flat_df(50, freq="1min")
        sig = _patched_signal_df(s, df)
        assert "entry candle pattern" not in sig.reason

    def test_gate_on_doji_rejected(self):
        """With require_entry_pattern=True, a zero-range doji is rejected by gate 6.4."""
        s = _make_mtf(require_entry_pattern=True, killzone_only=False)
        df = _flat_df(50, freq="1min")
        sig = _patched_signal_df(s, df)
        assert "entry candle pattern" in sig.reason

    def test_gate_on_hammer_passes(self):
        """With require_entry_pattern=True, a valid hammer passes gate 6.4."""
        s = _make_mtf(require_entry_pattern=True, killzone_only=False)
        df = _df_with_hammer()
        sig = _patched_signal_df(s, df)
        assert "entry candle pattern" not in sig.reason

    def test_gate_disabled_by_default_mtf(self):
        s = _make_mtf()
        assert s._require_entry_pattern is False

    def test_gate_disabled_by_default_scalp(self):
        s = _make_scalp()
        assert s._require_entry_pattern is False

    def test_custom_wick_ratio_forwarded(self):
        s = _make_mtf(require_entry_pattern=True, min_wick_ratio=0.80)
        assert s._min_wick_ratio == pytest.approx(0.80)
