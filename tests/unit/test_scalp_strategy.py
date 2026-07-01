"""
Unit tests for ScalpSMCStrategy and PartialCloseConfig.scalp().
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zeus.backtest.partial_close import PartialCloseConfig, PartialCloseState
from zeus.strategy.base import Signal, SignalType
from zeus.strategy.scalp_strategy import ScalpSMCStrategy


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

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


def _make_strategy(**kwargs) -> ScalpSMCStrategy:
    return ScalpSMCStrategy(
        df_htf_1h=_ohlcv(200, seed=1, freq="1h"),
        df_mtf_15m=_ohlcv(800, seed=2, freq="15min"),
        df_daily=_ohlcv(60, seed=3, freq="D"),
        **kwargs,
    )


# ──────────────────────────────────────────────────────────────────────────────
# PartialCloseConfig.scalp()
# ──────────────────────────────────────────────────────────────────────────────

class TestPartialCloseScalp:
    def test_three_levels(self):
        cfg = PartialCloseConfig.scalp()
        assert len(cfg.levels) == 3

    def test_first_level_is_be_at_1_5r(self):
        lvl = PartialCloseConfig.scalp().levels[0]
        assert lvl.r_multiple == pytest.approx(1.5)
        assert lvl.close_fraction == pytest.approx(0.0)
        assert lvl.sl_to_r == pytest.approx(0.0)

    def test_second_level_close_60pct_at_2_5r(self):
        lvl = PartialCloseConfig.scalp().levels[1]
        assert lvl.r_multiple == pytest.approx(2.5)
        assert lvl.close_fraction == pytest.approx(0.60)
        assert lvl.sl_to_r is None

    def test_third_level_full_close_at_4r(self):
        lvl = PartialCloseConfig.scalp().levels[2]
        assert lvl.r_multiple == pytest.approx(4.0)
        assert lvl.close_fraction == pytest.approx(1.0)
        assert lvl.sl_to_r is None

    def test_trailing_disabled(self):
        cfg = PartialCloseConfig.scalp()
        assert cfg.trailing.activate_at_r > 100

    def test_be_triggers_at_1_5r(self):
        """Price reaching 1.5R should move SL to breakeven, no quantity closed."""
        cfg   = PartialCloseConfig.scalp()
        state = PartialCloseState(entry_price=3000.0, sl_price=2994.0,
                                  is_long=True, original_qty=1.0)
        # 1R = 6 pips; 1.5R = 9 pips above entry → price = 3009
        events = state.process_bar(bar_high=3009.5, bar_low=2999.0, config=cfg)
        assert state.sl_price == pytest.approx(3000.0, abs=0.01)
        # BE-only level: close_fraction=0 → no quantity closed
        closed_qty = sum(e.qty_closed for e in events if hasattr(e, "qty_closed"))
        assert closed_qty == pytest.approx(0.0)

    def test_full_close_at_4r(self):
        """Price reaching 4R should close 100% of remaining position."""
        cfg   = PartialCloseConfig.scalp()
        state = PartialCloseState(entry_price=3000.0, sl_price=2994.0,
                                  is_long=True, original_qty=1.0)
        # hit 1.5R first → BE
        state.process_bar(bar_high=3009.5, bar_low=2999.0, config=cfg)
        # hit 2.5R → close 60 %
        state.process_bar(bar_high=3015.5, bar_low=3009.0, config=cfg)
        # hit 4R → close 100 % of remainder (40 % of original)
        state.process_bar(bar_high=3025.0, bar_low=3020.0, config=cfg)
        assert state.remaining_qty == pytest.approx(0.0, abs=0.001)


# ──────────────────────────────────────────────────────────────────────────────
# ScalpSMCStrategy — initialisation
# ──────────────────────────────────────────────────────────────────────────────

class TestScalpStrategyInit:
    def test_default_sl_pips(self):
        s = _make_strategy()
        assert s._sl_pips == pytest.approx(6.0)

    def test_default_max_sl_pips(self):
        s = _make_strategy()
        assert s._max_sl_pips == pytest.approx(10.0)

    def test_weekly_bias_disabled(self):
        s = _make_strategy()
        assert s._require_weekly_bias is False

    def test_asian_sweep_disabled(self):
        s = _make_strategy()
        assert s._require_asian_sweep is False

    def test_pd_filter_disabled(self):
        s = _make_strategy()
        assert s._require_pd_filter is False

    def test_default_max_daily_signals(self):
        s = _make_strategy()
        assert s._max_daily_signals == 5

    def test_default_max_per_session(self):
        s = _make_strategy()
        assert s._max_signals_per_session == 2

    def test_custom_sl_pips(self):
        s = _make_strategy(sl_pips=8.0, max_sl_pips=12.0)
        assert s._sl_pips == pytest.approx(8.0)
        assert s._max_sl_pips == pytest.approx(12.0)

    def test_choch_candle_off_by_default(self):
        s = _make_strategy()
        assert s._require_choch_candle is False

    def test_mtf_df_stored(self):
        df_15m = _ohlcv(400, freq="15min")
        s = ScalpSMCStrategy(
            df_htf_1h=_ohlcv(100, freq="1h"),
            df_mtf_15m=df_15m,
        )
        assert s._df_mtf is df_15m


# ──────────────────────────────────────────────────────────────────────────────
# ScalpSMCStrategy — signal label
# ──────────────────────────────────────────────────────────────────────────────

class TestScalpSignalLabel:
    """When a real signal fires, reason must say 'SCALP grade=' not 'MTF grade='."""

    def test_none_signal_passthrough(self):
        s   = _make_strategy(killzone_only=False)
        df  = _ohlcv(50, freq="1min", start="2026-01-04 09:00")
        sig = s.generate_signal(df, 49)
        # NONE signals are passed through unchanged
        assert sig.type == SignalType.NONE

    def test_signal_reason_has_scalp_label(self):
        """Patch parent to emit a LONG signal and verify label replacement."""
        from unittest.mock import patch
        from zeus.strategy.base import Signal, SignalType

        fake_sig = Signal(SignalType.LONG, 0.8, "MTF grade=C score=4/10 zone=OB SL=6.0pips", 10)

        s = _make_strategy()
        with patch.object(s.__class__.__bases__[0], "generate_signal",
                          return_value=fake_sig):
            sig = s.generate_signal(_ohlcv(50, freq="1min"), 10)

        assert "SCALP grade=" in sig.reason
        assert "MTF grade=" not in sig.reason

    def test_none_signal_label_unchanged(self):
        """NONE signals (reason = gate rejection) are NOT relabelled."""
        from unittest.mock import patch
        from zeus.strategy.base import Signal, SignalType

        none_sig = Signal(SignalType.NONE, 0.0, "outside killzone", 5)

        s = _make_strategy()
        with patch.object(s.__class__.__bases__[0], "generate_signal",
                          return_value=none_sig):
            sig = s.generate_signal(_ohlcv(50, freq="1min"), 5)

        assert sig.type == SignalType.NONE
        assert sig.reason == "outside killzone"


# ──────────────────────────────────────────────────────────────────────────────
# ScalpSMCStrategy — kill zone gate
# ──────────────────────────────────────────────────────────────────────────────

class TestScalpKillZoneGate:
    def test_rejects_outside_killzone_by_default(self):
        s  = _make_strategy(killzone_only=True)
        # 03:00 UTC = outside London (07–11) and NY (12–15)
        df = _ohlcv(50, freq="1min", start="2026-01-05 03:00")
        sig = s.generate_signal(df, 49)
        assert sig.type == SignalType.NONE
        assert "killzone" in sig.reason

    def test_allows_outside_killzone_when_disabled(self):
        """With killzone_only=False the gate is skipped (no killzone rejection)."""
        s  = _make_strategy(killzone_only=False)
        df = _ohlcv(50, freq="1min", start="2026-01-05 03:00")
        sig = s.generate_signal(df, 49)
        # Should not be rejected by killzone gate (may fail later gates — that's fine)
        assert "killzone" not in sig.reason


# ──────────────────────────────────────────────────────────────────────────────
# build_timeframes — 15min and m1 keys present
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildTimeframes:
    def test_15min_key_present(self):
        from zeus.backtest.data_loader import build_timeframes
        df_m1 = _ohlcv(500, freq="1min", start="2026-01-01")
        tfs = build_timeframes(df_m1)
        assert "15min" in tfs

    def test_m1_key_present(self):
        from zeus.backtest.data_loader import build_timeframes
        df_m1 = _ohlcv(500, freq="1min", start="2026-01-01")
        tfs = build_timeframes(df_m1)
        assert "m1" in tfs

    def test_15min_has_fewer_bars_than_m1(self):
        from zeus.backtest.data_loader import build_timeframes
        df_m1 = _ohlcv(500, freq="1min", start="2026-01-01")
        tfs = build_timeframes(df_m1)
        assert len(tfs["15min"]) < len(tfs["m1"])

    def test_all_expected_keys_present(self):
        from zeus.backtest.data_loader import build_timeframes
        df_m1 = _ohlcv(500, freq="1min", start="2026-01-01")
        tfs = build_timeframes(df_m1)
        for key in ("m1", "5min", "15min", "1h", "4h", "1d"):
            assert key in tfs, f"Missing key: {key}"
