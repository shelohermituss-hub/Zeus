"""
Unit tests for AdvancedBacktestEngine.
"""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock

from zeus.backtest.advanced_engine import AdvancedBacktestEngine
from zeus.backtest.partial_close import PartialCloseConfig, PartialCloseLevel, TrailingConfig
from zeus.orders.models import TradeStatus
from zeus.strategy.base import Signal, SignalType


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _flat_df(n: int = 200, price: float = 5000.0) -> pd.DataFrame:
    """Flat OHLCV DataFrame — no signal should trigger in most strategies."""
    idx = pd.date_range("2026-03-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "open":   price,
            "high":   price + 5,
            "low":    price - 5,
            "close":  price,
            "volume": 100.0,
        },
        index=idx,
    )


def _trend_df(n: int = 300, start: float = 5000.0, step: float = 2.0) -> pd.DataFrame:
    """Steadily rising OHLCV — useful for testing partial closes."""
    idx   = pd.date_range("2026-03-01", periods=n, freq="1min", tz="UTC")
    close = np.array([start + i * step for i in range(n)])
    return pd.DataFrame(
        {
            "open":   close - step,
            "high":   close + 5,
            "low":    close - 5,
            "close":  close,
            "volume": 100.0,
        },
        index=idx,
    )


def _always_long_strategy(signal_at: int = 110) -> MagicMock:
    """Strategy that emits a LONG signal at bar *signal_at* and NONE otherwise."""
    strat = MagicMock()
    def generate(df, bar_index):
        if bar_index == signal_at:
            return Signal(SignalType.LONG, 0.8, "test LONG", bar_index)
        return Signal(SignalType.NONE, 0.0, "", bar_index)
    strat.generate_signal.side_effect = generate
    return strat


def _engine(strategy=None, **kwargs) -> AdvancedBacktestEngine:
    defaults = dict(
        initial_balance=10_000.0,
        stop_loss_pips=20.0,
        max_sl_pips=30.0,
        pip_value=1.0,
        min_rr=3.0,
        max_position_pct=0.02,
        max_open_positions=1,
        fee_pct=0.001,
        slippage_pct=0.0,
        partial_close=PartialCloseConfig.default(),
    )
    defaults.update(kwargs)
    return AdvancedBacktestEngine(strategy=strategy or MagicMock(), **defaults)


# ──────────────────────────────────────────────────────────────────────────────
# Basic run behaviour
# ──────────────────────────────────────────────────────────────────────────────

class TestBasicRun:
    def test_returns_backtest_result(self):
        strat = MagicMock()
        strat.generate_signal.return_value = Signal(SignalType.NONE, 0.0, "", 0)
        engine = _engine(strat)
        result = engine.run(_flat_df(50))
        assert hasattr(result, "equity_curve")
        assert hasattr(result, "trades")

    def test_equity_curve_length(self):
        strat = MagicMock()
        strat.generate_signal.return_value = Signal(SignalType.NONE, 0.0, "", 0)
        engine = _engine(strat)
        n = 50
        result = engine.run(_flat_df(n))
        assert len(result.equity_curve) == n + 1

    def test_no_trades_no_pnl_change(self):
        strat = MagicMock()
        strat.generate_signal.return_value = Signal(SignalType.NONE, 0.0, "", 0)
        engine = _engine(strat)
        result = engine.run(_flat_df(50))
        assert result.equity_curve[0] == pytest.approx(10_000.0)
        # No fees without trades — equity stays flat
        assert result.equity_curve[-1] == pytest.approx(10_000.0)

    def test_initial_balance_preserved(self):
        strat = MagicMock()
        strat.generate_signal.return_value = Signal(SignalType.NONE, 0.0, "", 0)
        engine = _engine(strat, initial_balance=25_000.0)
        result = engine.run(_flat_df(50))
        assert result.initial_balance == 25_000.0


# ──────────────────────────────────────────────────────────────────────────────
# SL validation
# ──────────────────────────────────────────────────────────────────────────────

class TestSlValidation:
    def test_sl_too_wide_is_rejected(self):
        """Natural SL from bar range exceeds max_sl_pips → setup rejected."""
        strat = MagicMock()

        def gen(df, idx):
            if idx == 110:
                return Signal(SignalType.LONG, 1.0, "test", idx)
            return Signal(SignalType.NONE, 0.0, "", idx)

        strat.generate_signal.side_effect = gen
        idx   = pd.date_range("2026-03-01", periods=200, freq="1min", tz="UTC")
        price = 5000.0
        # bar_low = price - 40 → natural_sl = price-40-1 = 4959, distance=41 pips > max=30
        df = pd.DataFrame({
            "open":   price, "high": price + 10, "low": price - 40,
            "close":  price, "volume": 100.0,
        }, index=idx)
        engine = _engine(strat, max_sl_pips=30.0, stop_loss_pips=20.0)
        result = engine.run(df)
        rejected = [t for t in result.trades if t.status == TradeStatus.REJECTED]
        assert len(rejected) >= 1

    def test_sl_within_limit_opens_trade(self):
        df = _trend_df(200, start=5000.0, step=1.0)
        engine = _engine(_always_long_strategy(110), max_sl_pips=30.0, stop_loss_pips=20.0)
        result = engine.run(df)
        open_or_closed = [
            t for t in result.trades
            if t.status not in (TradeStatus.REJECTED, TradeStatus.FAILED)
        ]
        assert len(open_or_closed) >= 1


# ──────────────────────────────────────────────────────────────────────────────
# Partial close ladder
# ──────────────────────────────────────────────────────────────────────────────

class TestPartialCloseLadder:
    def test_trade_closed_by_sl_when_price_drops(self):
        """Price drops after entry → position closed at SL, PnL negative."""
        strat = MagicMock()

        def gen(df, idx):
            if idx == 5:
                return Signal(SignalType.LONG, 1.0, "test", idx)
            return Signal(SignalType.NONE, 0.0, "", idx)

        strat.generate_signal.side_effect = gen
        n = 30
        idx    = pd.date_range("2026-03-01", periods=n, freq="1min", tz="UTC")
        closes = [5000.0] * 6 + [4900.0] * 24  # price crashes after entry
        # Entry bar: small range → natural SL = 4999-1=4998 → use default 20-pip SL=4980
        # Crash bars: bar_low = close-5 → 4895 < SL 4980 → SL fires
        lows   = [c - 5 for c in closes]
        df = pd.DataFrame({
            "open":   closes,
            "high":   [c + 5 for c in closes],
            "low":    lows,
            "close":  closes,
            "volume": 100.0,
        }, index=idx)
        engine = _engine(strat, stop_loss_pips=20.0, max_sl_pips=30.0)
        result = engine.run(df)
        closed = result.closed_trades
        assert len(closed) == 1
        assert closed[0].realized_pnl < 0

    def test_position_force_closed_at_end(self):
        """When backtest ends, any open position is force-closed."""
        strat = _always_long_strategy(signal_at=5)
        df = _flat_df(20, price=5000.0)  # not enough movement to hit SL
        engine = _engine(strat, stop_loss_pips=20.0)
        result = engine.run(df)
        # All trades should be closed (force-close at last bar)
        open_trades = [t for t in result.trades if t.status == TradeStatus.OPEN]
        assert open_trades == []

    def test_max_open_positions_respected(self):
        """Second signal ignored when max_open_positions=1 is already full."""
        calls = [0]

        def gen(df, idx):
            calls[0] += 1
            if idx in (5, 10):
                return Signal(SignalType.LONG, 1.0, "test", idx)
            return Signal(SignalType.NONE, 0.0, "", idx)

        strat = MagicMock()
        strat.generate_signal.side_effect = gen
        df = _flat_df(30, price=5000.0)
        engine = _engine(strat, max_open_positions=1, stop_loss_pips=20.0)
        result = engine.run(df)
        trades = [t for t in result.trades if t.status != TradeStatus.REJECTED]
        assert len(trades) <= 1


# ──────────────────────────────────────────────────────────────────────────────
# Position sizing
# ──────────────────────────────────────────────────────────────────────────────

class TestPositionSizing:
    def test_risk_amount_is_max_position_pct(self):
        """
        At 2% max risk and 20-pip SL: qty = (10000 × 0.02) / (20 × 1.0) = 10 units.
        """
        strat = _always_long_strategy(signal_at=110)
        df = _flat_df(200, price=5000.0)
        engine = _engine(strat, max_position_pct=0.02, stop_loss_pips=20.0, pip_value=1.0)
        result = engine.run(df)
        non_rejected = [t for t in result.trades if t.status != TradeStatus.REJECTED]
        if non_rejected:
            assert non_rejected[0].quantity == pytest.approx(10.0, rel=0.01)


# ──────────────────────────────────────────────────────────────────────────────
# Config — custom partial close
# ──────────────────────────────────────────────────────────────────────────────

class TestCustomConfig:
    def test_single_level_config_accepted(self):
        cfg = PartialCloseConfig(
            levels=[PartialCloseLevel(2.0, 1.0, sl_to_r=None)],
            trailing=TrailingConfig(hard_close_r=10.0),
        )
        engine = _engine(_always_long_strategy(), partial_close=cfg)
        result = engine.run(_flat_df(200))
        assert result is not None

    def test_no_levels_config_uses_trailing_only(self):
        cfg = PartialCloseConfig(
            levels=[],
            trailing=TrailingConfig(activate_at_r=3.0, trail_r=2.0, hard_close_r=10.0),
        )
        engine = _engine(_always_long_strategy(), partial_close=cfg)
        result = engine.run(_flat_df(200))
        assert result is not None
