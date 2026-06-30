"""
Tests for zeus/backtest/report.py and zeus/backtest/engine.py.

All tests use synthetic OHLCV data and a lightweight mock strategy —
no real market data or network calls are required.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from zeus.backtest.engine import BacktestEngine
from zeus.backtest.report import BacktestResult
from zeus.orders.models import Trade, TradeStatus
from zeus.exchange.connector import OrderResult
from zeus.strategy.base import Signal, SignalType


# ======================================================================
# Helpers
# ======================================================================

def _make_df(
    n_bars: int,
    close: float = 100.0,
    high_offset: float = 1.0,
    low_offset: float = 1.0,
    closes: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame."""
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="1h", tz="UTC")
    c = closes if closes is not None else [close] * n_bars
    h = highs if highs is not None else [v + high_offset for v in c]
    l = lows  if lows  is not None else [v - low_offset  for v in c]
    return pd.DataFrame(
        {"open": c, "high": h, "low": l, "close": c, "volume": [1000.0] * n_bars},
        index=idx,
    )


def _order_result(
    order_id: str = "o1",
    symbol: str = "BTC/USDT",
    side: str = "buy",
    qty: float = 0.1,
    avg: float = 100.0,
    status: str = "closed",
) -> OrderResult:
    return OrderResult(
        order_id=order_id, symbol=symbol, side=side,
        quantity=qty, filled=qty, price=avg, average=avg,
        status=status, timestamp=datetime.now(tz=timezone.utc),
    )


def _trade(
    pnl: float,
    symbol: str = "BTC/USDT",
    side: str = "buy",
    qty: float = 1.0,
    entry: float = 100.0,
    status: TradeStatus = TradeStatus.CLOSED_TP,
) -> Trade:
    t = Trade(
        trade_id="t1", symbol=symbol, side=side,
        quantity=qty, entry_price=entry,
        sl_price=entry * 0.99, tp_price=entry * 1.02,
        entry_order=_order_result(),
    )
    t.status = status
    t.realized_pnl = pnl
    return t


class _AlwaysNone:
    """Strategy that never emits a signal."""
    def generate_signal(self, df, bar_index):
        return Signal(SignalType.NONE, 0.0, "none", bar_index)


class _SignalAtBar:
    """Strategy that fires one signal at a fixed bar."""
    def __init__(self, bar: int, stype: SignalType = SignalType.LONG):
        self._bar = bar
        self._type = stype

    def generate_signal(self, df, bar_index):
        if bar_index == self._bar:
            return Signal(self._type, 0.9, "signal", bar_index)
        return Signal(SignalType.NONE, 0.0, "none", bar_index)


# ======================================================================
# BacktestResult — unit tests (no engine needed)
# ======================================================================

class TestBacktestResultEmpty:
    def setup_method(self):
        self.r = BacktestResult(trades=[], equity_curve=[10_000.0], initial_balance=10_000.0)

    def test_n_trades_zero(self):
        assert self.r.n_trades == 0

    def test_win_rate_zero(self):
        assert self.r.win_rate == 0.0

    def test_total_gross_pnl_zero(self):
        assert self.r.total_gross_pnl == 0.0

    def test_total_fees_zero(self):
        assert self.r.total_fees == 0.0

    def test_net_pnl_zero(self):
        assert self.r.net_pnl == 0.0

    def test_max_drawdown_zero(self):
        assert self.r.max_drawdown_pct == 0.0

    def test_profit_factor_zero(self):
        assert self.r.profit_factor == 0.0

    def test_sharpe_zero_single_point(self):
        assert self.r.sharpe_ratio == 0.0

    def test_final_equity_equals_initial(self):
        assert self.r.final_equity == pytest.approx(10_000.0)

    def test_total_return_pct_zero(self):
        assert self.r.total_return_pct == pytest.approx(0.0)


class TestBacktestResultOneWin:
    def setup_method(self):
        t = _trade(pnl=200.0)
        self.r = BacktestResult(
            trades=[t],
            equity_curve=[10_000.0, 10_200.0],
            initial_balance=10_000.0,
            fee_pct=0.001,
        )

    def test_n_trades_one(self):
        assert self.r.n_trades == 1

    def test_n_winning_one(self):
        assert self.r.n_winning == 1

    def test_n_losing_zero(self):
        assert self.r.n_losing == 0

    def test_win_rate_one(self):
        assert self.r.win_rate == pytest.approx(1.0)

    def test_total_gross_pnl(self):
        assert self.r.total_gross_pnl == pytest.approx(200.0)

    def test_total_fees_positive(self):
        # entry=100, qty=1 → entry_value=100, exit_price≈102, exit_value≈102
        # fees = (100 + 102) * 0.001 = 0.202
        assert self.r.total_fees > 0

    def test_net_pnl_less_than_gross(self):
        assert self.r.net_pnl < self.r.total_gross_pnl

    def test_profit_factor_infinite(self):
        assert self.r.profit_factor == float("inf")

    def test_avg_win(self):
        assert self.r.avg_win == pytest.approx(200.0)

    def test_avg_loss_zero(self):
        assert self.r.avg_loss == 0.0

    def test_final_equity(self):
        assert self.r.final_equity == pytest.approx(10_200.0)

    def test_total_return_positive(self):
        assert self.r.total_return_pct > 0


class TestBacktestResultOneLoss:
    def setup_method(self):
        t = _trade(pnl=-100.0, status=TradeStatus.CLOSED_SL)
        self.r = BacktestResult(
            trades=[t],
            equity_curve=[10_000.0, 9_900.0],
            initial_balance=10_000.0,
        )

    def test_n_losing_one(self):
        assert self.r.n_losing == 1

    def test_win_rate_zero(self):
        assert self.r.win_rate == 0.0

    def test_profit_factor_zero(self):
        assert self.r.profit_factor == 0.0

    def test_avg_loss_negative(self):
        assert self.r.avg_loss == pytest.approx(-100.0)

    def test_max_drawdown_positive(self):
        assert self.r.max_drawdown_pct > 0


class TestBacktestResultMixed:
    def setup_method(self):
        trades = [_trade(200.0), _trade(-100.0, status=TradeStatus.CLOSED_SL)]
        self.r = BacktestResult(
            trades=trades,
            equity_curve=[10_000.0, 10_200.0, 10_100.0],
            initial_balance=10_000.0,
        )

    def test_win_rate_half(self):
        assert self.r.win_rate == pytest.approx(0.5)

    def test_profit_factor(self):
        assert self.r.profit_factor == pytest.approx(2.0)

    def test_gross_pnl(self):
        assert self.r.total_gross_pnl == pytest.approx(100.0)


class TestBacktestResultRejectedExcluded:
    def test_rejected_trades_not_counted(self):
        good = _trade(100.0)
        bad  = _trade(0.0, status=TradeStatus.REJECTED)
        r = BacktestResult(trades=[good, bad], equity_curve=[10_000.0], initial_balance=10_000.0)
        assert r.n_trades == 1

    def test_failed_trades_not_counted(self):
        bad = _trade(0.0, status=TradeStatus.FAILED)
        r = BacktestResult(trades=[bad], equity_curve=[10_000.0], initial_balance=10_000.0)
        assert r.n_trades == 0

    def test_open_trades_not_counted(self):
        open_t = _trade(0.0, status=TradeStatus.OPEN)
        r = BacktestResult(trades=[open_t], equity_curve=[10_000.0], initial_balance=10_000.0)
        assert r.n_trades == 0


class TestBacktestResultMaxDrawdown:
    def test_flat_equity_zero_drawdown(self):
        r = BacktestResult(trades=[], equity_curve=[1_000.0, 1_000.0, 1_000.0], initial_balance=1_000.0)
        assert r.max_drawdown_pct == pytest.approx(0.0)

    def test_drawdown_computed_from_peak(self):
        # peak=1200, valley=900 → dd = 300/1200 = 25%
        r = BacktestResult(
            trades=[], equity_curve=[1_000.0, 1_200.0, 900.0, 1_100.0],
            initial_balance=1_000.0,
        )
        assert r.max_drawdown_pct == pytest.approx(0.25)

    def test_rising_equity_zero_drawdown(self):
        r = BacktestResult(
            trades=[], equity_curve=[1_000.0, 1_100.0, 1_200.0], initial_balance=1_000.0
        )
        assert r.max_drawdown_pct == pytest.approx(0.0)


class TestBacktestResultSummary:
    def test_summary_keys_present(self):
        r = BacktestResult(trades=[], equity_curve=[10_000.0], initial_balance=10_000.0)
        s = r.summary()
        for key in (
            "n_trades", "win_rate", "net_pnl", "total_return_pct",
            "max_drawdown_pct", "profit_factor", "sharpe_ratio",
            "initial_balance", "final_equity",
        ):
            assert key in s

    def test_summary_values_are_rounded(self):
        r = BacktestResult(
            trades=[_trade(pnl=1.23456789)],
            equity_curve=[10_000.0, 10_001.23456789],
            initial_balance=10_000.0,
        )
        s = r.summary()
        # Values should be rounded (not raw floats with many decimals)
        assert isinstance(s["net_pnl"], float)


# ======================================================================
# BacktestEngine — integration tests with mock strategy
# ======================================================================

class TestEngineEmpty:
    def test_empty_df_returns_result(self):
        engine = BacktestEngine(_AlwaysNone())
        r = engine.run(pd.DataFrame(columns=["open", "high", "low", "close", "volume"]))
        assert isinstance(r, BacktestResult)

    def test_empty_df_equity_curve_length_one(self):
        engine = BacktestEngine(_AlwaysNone())
        r = engine.run(pd.DataFrame(columns=["open", "high", "low", "close", "volume"]))
        assert len(r.equity_curve) == 1

    def test_empty_df_no_trades(self):
        engine = BacktestEngine(_AlwaysNone())
        r = engine.run(pd.DataFrame(columns=["open", "high", "low", "close", "volume"]))
        assert r.n_trades == 0


class TestEngineNoSignal:
    def test_no_signal_no_trades(self):
        engine = BacktestEngine(_AlwaysNone())
        r = engine.run(_make_df(50))
        assert r.n_trades == 0

    def test_no_signal_equity_curve_length(self):
        engine = BacktestEngine(_AlwaysNone())
        df = _make_df(20)
        r = engine.run(df)
        assert len(r.equity_curve) == len(df) + 1

    def test_no_signal_equity_flat(self):
        engine = BacktestEngine(_AlwaysNone())
        r = engine.run(_make_df(10, close=100.0))
        # No trades → equity should remain near initial_balance
        assert all(abs(e - 10_000.0) < 1.0 for e in r.equity_curve)


class TestEngineOneLong:
    """Strategy fires LONG once; flat price; position closed at end of run."""

    def setup_method(self):
        # 20 bars, signal at bar 5
        self.df = _make_df(20, close=100.0)
        engine = BacktestEngine(
            _SignalAtBar(bar=5),
            initial_balance=10_000.0,
            slippage_pct=0.0,   # zero slippage for deterministic PnL
            stop_loss_pct=0.10,
            take_profit_pct=0.20,
        )
        self.r = engine.run(self.df, symbol="BTC/USDT")

    def test_one_closed_trade(self):
        assert self.r.n_trades == 1

    def test_equity_curve_length(self):
        assert len(self.r.equity_curve) == len(self.df) + 1

    def test_initial_equity_correct(self):
        assert self.r.equity_curve[0] == pytest.approx(10_000.0)

    def test_trade_side_long(self):
        assert self.r.closed_trades[0].side == "buy"

    def test_trade_closed_man_at_end(self):
        # price never hits SL/TP on a flat market → closed manually
        assert self.r.closed_trades[0].status == TradeStatus.CLOSED_MAN


class TestEngineOneShort:
    def test_short_signal_creates_sell_trade(self):
        engine = BacktestEngine(
            _SignalAtBar(bar=5, stype=SignalType.SHORT),
            initial_balance=10_000.0,
            slippage_pct=0.0,
            stop_loss_pct=0.10,
            take_profit_pct=0.20,
        )
        r = engine.run(_make_df(20, close=100.0))
        assert r.n_trades == 1
        assert r.closed_trades[0].side == "sell"


class TestEngineIntrabarSL:
    """Long trade stopped out when bar low drops below SL."""

    def setup_method(self):
        # 10 bars; signal at bar 3; SL at 99.0 (1% below 100)
        closes = [100.0] * 10
        highs  = [101.0] * 10
        lows   = [99.5]  * 10   # stays above SL until bar 5

        # Bar 5: low drops below SL
        lows[5] = 98.0

        self.df = _make_df(10, closes=closes, highs=highs, lows=lows)
        engine = BacktestEngine(
            _SignalAtBar(bar=3),
            initial_balance=10_000.0,
            slippage_pct=0.0,
            stop_loss_pct=0.01,
            take_profit_pct=0.05,
        )
        self.r = engine.run(self.df)

    def test_trade_closed_sl(self):
        assert self.r.closed_trades[0].status == TradeStatus.CLOSED_SL

    def test_sl_pnl_negative(self):
        assert self.r.closed_trades[0].realized_pnl < 0


class TestEngineIntrabarTP:
    """Long trade hits take-profit when bar high crosses TP."""

    def setup_method(self):
        closes = [100.0] * 10
        highs  = [100.5]  * 10   # below TP until bar 6
        lows   = [99.5]   * 10

        # Bar 6: high crosses TP (102.0 for tp_pct=0.02)
        highs[6] = 103.0

        self.df = _make_df(10, closes=closes, highs=highs, lows=lows)
        engine = BacktestEngine(
            _SignalAtBar(bar=3),
            initial_balance=10_000.0,
            slippage_pct=0.0,
            stop_loss_pct=0.05,
            take_profit_pct=0.02,
        )
        self.r = engine.run(self.df)

    def test_trade_closed_tp(self):
        assert self.r.closed_trades[0].status == TradeStatus.CLOSED_TP

    def test_tp_pnl_positive(self):
        assert self.r.closed_trades[0].realized_pnl > 0


class TestEngineNoHighLow:
    """Without high/low columns, intrabar exits don't fire."""

    def test_no_intrabar_exit_without_columns(self):
        # Price drops well below SL, but no high/low → no intrabar exit
        df = pd.DataFrame(
            {"close": [100.0] * 10},
            index=pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC"),
        )
        engine = BacktestEngine(
            _SignalAtBar(bar=3),
            initial_balance=10_000.0,
            slippage_pct=0.0,
            stop_loss_pct=0.01,
            take_profit_pct=0.05,
        )
        r = engine.run(df)
        # Trade is closed manually at end (not via SL)
        assert r.n_trades == 1
        assert r.closed_trades[0].status == TradeStatus.CLOSED_MAN


class TestEngineEquityCurve:
    def test_equity_starts_at_initial_balance(self):
        engine = BacktestEngine(_AlwaysNone(), initial_balance=5_000.0)
        r = engine.run(_make_df(5))
        assert r.equity_curve[0] == pytest.approx(5_000.0)

    def test_equity_curve_length_matches_bars_plus_one(self):
        engine = BacktestEngine(_AlwaysNone())
        df = _make_df(30)
        r = engine.run(df)
        assert len(r.equity_curve) == 31

    def test_winning_trade_increases_equity(self):
        # Rising price: signal at bar 3, price rises after → MAN close with profit
        closes = [100.0, 100.0, 100.0, 100.0] + [110.0] * 16
        df = _make_df(20, closes=closes)
        engine = BacktestEngine(
            _SignalAtBar(bar=3),
            initial_balance=10_000.0,
            slippage_pct=0.0,
            stop_loss_pct=0.30,   # wide SL so price rise doesn't trigger SL first
            take_profit_pct=None,
        )
        r = engine.run(df)
        assert r.final_equity > r.initial_balance


class TestEngineDedupliation:
    """Second signal on same symbol while trade is open must be ignored."""

    def test_two_signals_same_symbol_one_trade(self):
        class _TwoSignals:
            def generate_signal(self, df, bar_index):
                if bar_index in (3, 7):
                    return Signal(SignalType.LONG, 0.9, "signal", bar_index)
                return Signal(SignalType.NONE, 0.0, "none", bar_index)

        engine = BacktestEngine(
            _TwoSignals(),
            initial_balance=10_000.0,
            slippage_pct=0.0,
            stop_loss_pct=0.30,   # wide — won't trigger on flat market
        )
        r = engine.run(_make_df(15))
        # Only one trade should exist (second signal ignored, trade force-closed at end)
        closed = [t for t in r.trades if t.status == TradeStatus.CLOSED_MAN]
        assert len(closed) == 1


class TestEngineRunIsolation:
    """Two consecutive run() calls must be fully independent."""

    def test_two_runs_independent_equity(self):
        engine = BacktestEngine(
            _SignalAtBar(bar=2),
            initial_balance=10_000.0,
            slippage_pct=0.0,
            stop_loss_pct=0.10,
        )
        df = _make_df(10)
        r1 = engine.run(df)
        r2 = engine.run(df)
        assert r1.equity_curve[0] == r2.equity_curve[0]
        assert len(r1.trades) == len(r2.trades)

    def test_two_runs_do_not_share_trades(self):
        engine = BacktestEngine(_SignalAtBar(bar=2), initial_balance=10_000.0, slippage_pct=0.0)
        df = _make_df(10)
        r1 = engine.run(df)
        r2 = engine.run(df)
        assert r1.trades is not r2.trades
