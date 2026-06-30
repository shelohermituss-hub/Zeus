"""
Tests for zeus/orders/models.py and zeus/orders/executor.py.

All exchange interactions are mocked; no network calls are made.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest

from zeus.exchange.connector import OrderResult
from zeus.orders.executor import OrderExecutor
from zeus.orders.models import Trade, TradeStatus
from zeus.risk.manager import RiskManager
from zeus.risk.models import OrderSide, ValidationResult
from zeus.strategy.base import Signal, SignalType


# ======================================================================
# Helpers
# ======================================================================

def _make_order_result(
    order_id: str = "ord-1",
    symbol: str = "BTC/USDT",
    side: str = "buy",
    quantity: float = 0.1,
    filled: float = 0.1,
    price: float = 50_000.0,
    average: float | None = 50_000.0,
    status: str = "closed",
) -> OrderResult:
    return OrderResult(
        order_id=order_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        filled=filled,
        price=price,
        average=average,
        status=status,
        timestamp=datetime.now(tz=timezone.utc),
    )


def _open_trade(
    symbol: str = "BTC/USDT",
    side: str = "buy",
    quantity: float = 0.1,
    entry_price: float = 50_000.0,
    sl_price: float = 49_500.0,
    tp_price: float | None = 51_000.0,
) -> Trade:
    return Trade(
        trade_id="t-1",
        symbol=symbol,
        side=side,
        quantity=quantity,
        entry_price=entry_price,
        sl_price=sl_price,
        tp_price=tp_price,
        entry_order=_make_order_result(),
        signal_reason="test",
    )


def _make_risk(balance: float = 100_000.0) -> RiskManager:
    rm = RiskManager(
        max_position_pct=0.02,
        stop_loss_pct=0.05,   # wide SL so tests don't trigger SL-distance rejection
        take_profit_pct=0.10,
        max_daily_loss_pct=0.10,
        max_drawdown_pct=0.20,
        max_exposure_pct=0.50,
        max_open_positions=5,
    )
    rm.initialise_capital(balance)
    return rm


def _make_connector(fill_price: float = 50_000.0) -> MagicMock:
    conn = MagicMock()
    conn.place_market_order.return_value = _make_order_result(average=fill_price)
    return conn


def _long_signal(bar_index: int = 0) -> Signal:
    return Signal(SignalType.LONG, 0.8, "test long", bar_index)


def _short_signal(bar_index: int = 0) -> Signal:
    return Signal(SignalType.SHORT, 0.8, "test short", bar_index)


def _none_signal(bar_index: int = 0) -> Signal:
    return Signal(SignalType.NONE, 0.0, "no signal", bar_index)


# ======================================================================
# Trade model tests
# ======================================================================

class TestTradeModel:
    def test_default_status_is_open(self):
        t = _open_trade()
        assert t.status == TradeStatus.OPEN

    def test_is_open_true_when_open(self):
        t = _open_trade()
        assert t.is_open is True

    def test_is_open_false_when_closed(self):
        t = _open_trade()
        t.status = TradeStatus.CLOSED_SL
        assert t.is_open is False

    def test_is_long_buy(self):
        t = _open_trade(side="buy")
        assert t.is_long is True

    def test_is_long_sell(self):
        t = _open_trade(side="sell")
        assert t.is_long is False

    def test_unrealized_pnl_long_profit(self):
        t = _open_trade(side="buy", quantity=1.0, entry_price=50_000.0)
        assert t.unrealized_pnl(51_000.0) == pytest.approx(1_000.0)

    def test_unrealized_pnl_long_loss(self):
        t = _open_trade(side="buy", quantity=1.0, entry_price=50_000.0)
        assert t.unrealized_pnl(49_000.0) == pytest.approx(-1_000.0)

    def test_unrealized_pnl_short_profit(self):
        t = _open_trade(side="sell", quantity=1.0, entry_price=50_000.0)
        assert t.unrealized_pnl(49_000.0) == pytest.approx(1_000.0)

    def test_unrealized_pnl_short_loss(self):
        t = _open_trade(side="sell", quantity=1.0, entry_price=50_000.0)
        assert t.unrealized_pnl(51_000.0) == pytest.approx(-1_000.0)

    def test_unrealized_pnl_at_entry_is_zero(self):
        t = _open_trade(side="buy", quantity=2.0, entry_price=50_000.0)
        assert t.unrealized_pnl(50_000.0) == pytest.approx(0.0)

    def test_default_realized_pnl_is_zero(self):
        t = _open_trade()
        assert t.realized_pnl == 0.0

    def test_exit_order_initially_none(self):
        t = _open_trade()
        assert t.exit_order is None

    def test_closed_at_initially_none(self):
        t = _open_trade()
        assert t.closed_at is None

    def test_opened_at_is_utc(self):
        t = _open_trade()
        assert t.opened_at.tzinfo is not None

    def test_tp_price_can_be_none(self):
        t = _open_trade(tp_price=None)
        assert t.tp_price is None

    def test_trade_status_enum_values(self):
        assert TradeStatus.OPEN.value       == "open"
        assert TradeStatus.CLOSED_SL.value  == "closed_sl"
        assert TradeStatus.CLOSED_TP.value  == "closed_tp"
        assert TradeStatus.CLOSED_MAN.value == "closed_man"
        assert TradeStatus.REJECTED.value   == "rejected"
        assert TradeStatus.FAILED.value     == "failed"


# ======================================================================
# OrderExecutor initialisation
# ======================================================================

class TestOrderExecutorInit:
    def test_open_trades_empty_on_init(self):
        ex = OrderExecutor(_make_connector(), _make_risk())
        assert ex.open_trades == []

    def test_all_trades_empty_on_init(self):
        ex = OrderExecutor(_make_connector(), _make_risk())
        assert ex.all_trades == []

    def test_custom_sl_pct_stored(self):
        ex = OrderExecutor(_make_connector(), _make_risk(), stop_loss_pct=0.02)
        assert ex._sl_pct == 0.02

    def test_custom_tp_pct_stored(self):
        ex = OrderExecutor(_make_connector(), _make_risk(), take_profit_pct=0.04)
        assert ex._tp_pct == 0.04

    def test_tp_pct_none_disables_tp(self):
        ex = OrderExecutor(_make_connector(), _make_risk(), take_profit_pct=None)
        assert ex._tp_pct is None


# ======================================================================
# _bracket_prices
# ======================================================================

class TestBracketPrices:
    def setup_method(self):
        self.ex = OrderExecutor(
            _make_connector(),
            _make_risk(),
            stop_loss_pct=0.01,
            take_profit_pct=0.02,
        )

    def test_long_sl_below_entry(self):
        sl, _ = self.ex._bracket_prices(OrderSide.BUY, 100.0)
        assert sl == pytest.approx(99.0)

    def test_long_tp_above_entry(self):
        _, tp = self.ex._bracket_prices(OrderSide.BUY, 100.0)
        assert tp == pytest.approx(102.0)

    def test_short_sl_above_entry(self):
        sl, _ = self.ex._bracket_prices(OrderSide.SELL, 100.0)
        assert sl == pytest.approx(101.0)

    def test_short_tp_below_entry(self):
        _, tp = self.ex._bracket_prices(OrderSide.SELL, 100.0)
        assert tp == pytest.approx(98.0)

    def test_no_tp_when_tp_pct_none(self):
        ex = OrderExecutor(_make_connector(), _make_risk(), take_profit_pct=None)
        _, tp = ex._bracket_prices(OrderSide.BUY, 100.0)
        assert tp is None


# ======================================================================
# execute() — none signal
# ======================================================================

class TestExecuteNoneSignal:
    def test_none_signal_returns_none(self):
        ex = OrderExecutor(_make_connector(), _make_risk())
        result = ex.execute(_none_signal(), "BTC/USDT", 50_000.0)
        assert result is None

    def test_none_signal_no_order_placed(self):
        conn = _make_connector()
        ex = OrderExecutor(conn, _make_risk())
        ex.execute(_none_signal(), "BTC/USDT", 50_000.0)
        conn.place_market_order.assert_not_called()

    def test_none_signal_no_trade_recorded(self):
        ex = OrderExecutor(_make_connector(), _make_risk())
        ex.execute(_none_signal(), "BTC/USDT", 50_000.0)
        assert ex.all_trades == []


# ======================================================================
# execute() — deduplication
# ======================================================================

class TestExecuteDeduplication:
    def test_second_signal_same_symbol_ignored(self):
        conn = _make_connector(fill_price=50_000.0)
        ex = OrderExecutor(conn, _make_risk())
        first = ex.execute(_long_signal(), "BTC/USDT", 50_000.0)
        assert first is not None and first.is_open

        second = ex.execute(_long_signal(), "BTC/USDT", 50_000.0)
        assert second is None

    def test_second_signal_different_symbol_allowed(self):
        conn = _make_connector(fill_price=50_000.0)
        conn.place_market_order.return_value = _make_order_result(symbol="ETH/USDT", average=2_000.0)
        ex = OrderExecutor(conn, _make_risk(balance=200_000.0))
        first = ex.execute(_long_signal(), "BTC/USDT", 50_000.0)

        # Patch the connector return for the second symbol
        conn.place_market_order.return_value = _make_order_result(
            symbol="ETH/USDT", average=2_000.0
        )
        second = ex.execute(_long_signal(), "ETH/USDT", 2_000.0)
        assert first is not None
        assert second is not None

    def test_after_close_new_signal_allowed(self):
        conn = _make_connector(fill_price=50_000.0)
        ex = OrderExecutor(conn, _make_risk())
        trade = ex.execute(_long_signal(), "BTC/USDT", 50_000.0)
        assert trade is not None

        # Manually close it
        conn.place_market_order.return_value = _make_order_result(side="sell", average=51_000.0)
        ex.close_trade(trade.trade_id, 51_000.0)

        # Re-entry should succeed
        conn.place_market_order.return_value = _make_order_result(average=51_500.0)
        second = ex.execute(_long_signal(), "BTC/USDT", 51_500.0)
        assert second is not None
        assert second.is_open


# ======================================================================
# execute() — successful long
# ======================================================================

class TestExecuteLong:
    def setup_method(self):
        self.conn = _make_connector(fill_price=50_000.0)
        self.risk = _make_risk(balance=100_000.0)
        self.ex = OrderExecutor(
            self.conn, self.risk, stop_loss_pct=0.01, take_profit_pct=0.02
        )
        self.trade = self.ex.execute(_long_signal(), "BTC/USDT", 50_000.0)

    def test_returns_trade(self):
        assert self.trade is not None

    def test_trade_is_open(self):
        assert self.trade.is_open

    def test_trade_status_open(self):
        assert self.trade.status == TradeStatus.OPEN

    def test_trade_side_buy(self):
        assert self.trade.side == "buy"

    def test_trade_symbol(self):
        assert self.trade.symbol == "BTC/USDT"

    def test_trade_entry_price_from_fill(self):
        assert self.trade.entry_price == pytest.approx(50_000.0)

    def test_trade_sl_below_entry(self):
        assert self.trade.sl_price == pytest.approx(49_500.0)

    def test_trade_tp_above_entry(self):
        assert self.trade.tp_price == pytest.approx(51_000.0)

    def test_connector_called_with_buy(self):
        self.conn.place_market_order.assert_called_once()
        _, kwargs = self.conn.place_market_order.call_args
        assert kwargs.get("side") or self.conn.place_market_order.call_args[0][1] == "buy"

    def test_risk_state_positions_incremented(self):
        assert self.risk.state.open_positions == 1

    def test_trade_recorded_in_all_trades(self):
        assert self.trade in self.ex.all_trades

    def test_trade_in_open_trades(self):
        assert self.trade in self.ex.open_trades

    def test_trade_id_is_string(self):
        assert isinstance(self.trade.trade_id, str)


# ======================================================================
# execute() — successful short
# ======================================================================

class TestExecuteShort:
    def setup_method(self):
        conn = _make_connector(fill_price=50_000.0)
        conn.place_market_order.return_value = _make_order_result(side="sell", average=50_000.0)
        self.ex = OrderExecutor(conn, _make_risk(), stop_loss_pct=0.01, take_profit_pct=0.02)
        self.trade = self.ex.execute(_short_signal(), "BTC/USDT", 50_000.0)

    def test_side_is_sell(self):
        assert self.trade.side == "sell"

    def test_sl_above_entry(self):
        assert self.trade.sl_price == pytest.approx(50_500.0)

    def test_tp_below_entry(self):
        assert self.trade.tp_price == pytest.approx(49_000.0)


# ======================================================================
# execute() — risk manager rejection
# ======================================================================

class TestExecuteRejected:
    def test_rejected_trade_recorded_as_rejected(self):
        conn = _make_connector()
        risk = _make_risk(balance=100_000.0)
        risk.set_kill_switch(True)
        ex = OrderExecutor(conn, risk)
        trade = ex.execute(_long_signal(), "BTC/USDT", 50_000.0)
        assert trade is not None
        assert trade.status == TradeStatus.REJECTED

    def test_rejected_trade_no_exchange_call(self):
        conn = _make_connector()
        risk = _make_risk(balance=100_000.0)
        risk.set_kill_switch(True)
        ex = OrderExecutor(conn, risk)
        ex.execute(_long_signal(), "BTC/USDT", 50_000.0)
        conn.place_market_order.assert_not_called()

    def test_rejected_trade_not_in_open_trades(self):
        conn = _make_connector()
        risk = _make_risk(balance=100_000.0)
        risk.set_kill_switch(True)
        ex = OrderExecutor(conn, risk)
        trade = ex.execute(_long_signal(), "BTC/USDT", 50_000.0)
        assert trade not in ex.open_trades

    def test_rejected_trade_in_all_trades(self):
        conn = _make_connector()
        risk = _make_risk(balance=100_000.0)
        risk.set_kill_switch(True)
        ex = OrderExecutor(conn, risk)
        trade = ex.execute(_long_signal(), "BTC/USDT", 50_000.0)
        assert trade in ex.all_trades


# ======================================================================
# execute() — exchange error
# ======================================================================

class TestExecuteFailed:
    def setup_method(self):
        conn = _make_connector()
        conn.place_market_order.side_effect = RuntimeError("exchange down")
        self.ex = OrderExecutor(conn, _make_risk())
        self.trade = self.ex.execute(_long_signal(), "BTC/USDT", 50_000.0)

    def test_failed_trade_recorded(self):
        assert self.trade is not None

    def test_failed_status(self):
        assert self.trade.status == TradeStatus.FAILED

    def test_failed_not_in_open_trades(self):
        assert self.trade not in self.ex.open_trades

    def test_failed_in_all_trades(self):
        assert self.trade in self.ex.all_trades

    def test_risk_state_not_updated_on_failure(self):
        # open_positions should still be 0 since the fill never happened
        assert self.ex._risk.state.open_positions == 0


# ======================================================================
# check_exits() — stop-loss
# ======================================================================

class TestCheckExitsSL:
    def setup_method(self):
        self.conn = _make_connector(fill_price=49_000.0)
        # Exit order returns sell side
        self.conn.place_market_order.return_value = _make_order_result(
            side="sell", average=49_000.0
        )
        self.risk = _make_risk()
        self.ex = OrderExecutor(self.conn, self.risk, stop_loss_pct=0.01)

        # Manually inject an open long trade
        self.conn.place_market_order.return_value = _make_order_result(average=50_000.0)
        self.trade = self.ex.execute(_long_signal(), "BTC/USDT", 50_000.0)
        assert self.trade is not None

        # Now set exit side_effect for exit order
        self.conn.place_market_order.return_value = _make_order_result(
            side="sell", average=49_000.0
        )

    def test_sl_hit_closes_trade(self):
        closed = self.ex.check_exits("BTC/USDT", 49_000.0)
        assert len(closed) == 1

    def test_sl_trade_status(self):
        self.ex.check_exits("BTC/USDT", 49_000.0)
        assert self.trade.status == TradeStatus.CLOSED_SL

    def test_sl_trade_not_in_open(self):
        self.ex.check_exits("BTC/USDT", 49_000.0)
        assert self.trade not in self.ex.open_trades

    def test_sl_realized_pnl_negative(self):
        self.ex.check_exits("BTC/USDT", 49_000.0)
        assert self.trade.realized_pnl < 0

    def test_sl_closed_at_set(self):
        self.ex.check_exits("BTC/USDT", 49_000.0)
        assert self.trade.closed_at is not None

    def test_sl_exit_order_set(self):
        self.ex.check_exits("BTC/USDT", 49_000.0)
        assert self.trade.exit_order is not None

    def test_no_exit_above_sl(self):
        closed = self.ex.check_exits("BTC/USDT", 50_200.0)
        assert closed == []
        assert self.trade.is_open


# ======================================================================
# check_exits() — take-profit
# ======================================================================

class TestCheckExitsTP:
    def setup_method(self):
        conn = _make_connector(fill_price=50_000.0)
        self.conn = conn
        self.risk = _make_risk()
        self.ex = OrderExecutor(conn, self.risk, stop_loss_pct=0.01, take_profit_pct=0.02)

        conn.place_market_order.return_value = _make_order_result(average=50_000.0)
        self.trade = self.ex.execute(_long_signal(), "BTC/USDT", 50_000.0)
        assert self.trade is not None

        conn.place_market_order.return_value = _make_order_result(
            side="sell", average=51_500.0
        )

    def test_tp_hit_closes_trade(self):
        closed = self.ex.check_exits("BTC/USDT", 51_500.0)
        assert len(closed) == 1

    def test_tp_trade_status(self):
        self.ex.check_exits("BTC/USDT", 51_500.0)
        assert self.trade.status == TradeStatus.CLOSED_TP

    def test_tp_realized_pnl_positive(self):
        self.ex.check_exits("BTC/USDT", 51_500.0)
        assert self.trade.realized_pnl > 0

    def test_no_exit_below_tp(self):
        closed = self.ex.check_exits("BTC/USDT", 50_500.0)
        assert closed == []


# ======================================================================
# check_exits() — short positions
# ======================================================================

class TestCheckExitsShort:
    def setup_method(self):
        conn = _make_connector()
        conn.place_market_order.return_value = _make_order_result(side="sell", average=50_000.0)
        self.risk = _make_risk()
        self.ex = OrderExecutor(conn, self.risk, stop_loss_pct=0.01, take_profit_pct=0.02)
        self.trade = self.ex.execute(_short_signal(), "BTC/USDT", 50_000.0)
        assert self.trade is not None
        conn.place_market_order.return_value = _make_order_result(side="buy", average=51_000.0)
        self.conn = conn

    def test_short_sl_above_entry(self):
        # SL at 50_500; price crossing above triggers SL
        closed = self.ex.check_exits("BTC/USDT", 50_600.0)
        assert len(closed) == 1
        assert self.trade.status == TradeStatus.CLOSED_SL

    def test_short_tp_below_entry(self):
        # TP at 49_000; price crossing below triggers TP
        self.conn.place_market_order.return_value = _make_order_result(
            side="buy", average=49_000.0
        )
        closed = self.ex.check_exits("BTC/USDT", 48_500.0)
        assert len(closed) == 1
        assert self.trade.status == TradeStatus.CLOSED_TP


# ======================================================================
# check_exits() — different symbol ignored
# ======================================================================

class TestCheckExitsSymbolFilter:
    def test_different_symbol_not_checked(self):
        conn = _make_connector(fill_price=50_000.0)
        ex = OrderExecutor(conn, _make_risk())
        trade = ex.execute(_long_signal(), "BTC/USDT", 50_000.0)
        assert trade is not None

        closed = ex.check_exits("ETH/USDT", 0.0)  # extreme price
        assert closed == []
        assert trade.is_open


# ======================================================================
# close_trade() — manual close
# ======================================================================

class TestCloseTrade:
    def setup_method(self):
        conn = _make_connector(fill_price=50_000.0)
        self.conn = conn
        self.risk = _make_risk()
        self.ex = OrderExecutor(conn, self.risk, stop_loss_pct=0.01, take_profit_pct=0.02)
        conn.place_market_order.return_value = _make_order_result(average=50_000.0)
        self.trade = self.ex.execute(_long_signal(), "BTC/USDT", 50_000.0)
        assert self.trade is not None
        conn.place_market_order.return_value = _make_order_result(side="sell", average=50_500.0)

    def test_manual_close_returns_trade(self):
        result = self.ex.close_trade(self.trade.trade_id, 50_500.0)
        assert result is self.trade

    def test_manual_close_status(self):
        self.ex.close_trade(self.trade.trade_id, 50_500.0)
        assert self.trade.status == TradeStatus.CLOSED_MAN

    def test_manual_close_pnl_positive(self):
        self.ex.close_trade(self.trade.trade_id, 50_500.0)
        assert self.trade.realized_pnl > 0

    def test_manual_close_not_in_open_trades(self):
        self.ex.close_trade(self.trade.trade_id, 50_500.0)
        assert self.trade not in self.ex.open_trades

    def test_close_unknown_id_returns_none(self):
        result = self.ex.close_trade("nonexistent", 50_000.0)
        assert result is None

    def test_close_already_closed_returns_none(self):
        self.conn.place_market_order.return_value = _make_order_result(
            side="sell", average=50_500.0
        )
        self.ex.close_trade(self.trade.trade_id, 50_500.0)
        result = self.ex.close_trade(self.trade.trade_id, 50_500.0)
        assert result is None


# ======================================================================
# open_trades / all_trades properties
# ======================================================================

class TestTradeRegistries:
    def test_open_trades_excludes_closed(self):
        conn = _make_connector()
        conn.place_market_order.return_value = _make_order_result(average=50_000.0)
        ex = OrderExecutor(conn, _make_risk())
        t = ex.execute(_long_signal(), "BTC/USDT", 50_000.0)
        assert t is not None

        conn.place_market_order.return_value = _make_order_result(side="sell", average=51_000.0)
        ex.close_trade(t.trade_id, 51_000.0)

        assert ex.open_trades == []
        assert t in ex.all_trades

    def test_multiple_open_trades_different_symbols(self):
        conn = _make_connector()
        conn.place_market_order.return_value = _make_order_result(average=50_000.0)
        ex = OrderExecutor(conn, _make_risk(balance=500_000.0))

        t1 = ex.execute(_long_signal(), "BTC/USDT", 50_000.0)
        conn.place_market_order.return_value = _make_order_result(
            symbol="ETH/USDT", average=2_000.0
        )
        t2 = ex.execute(_long_signal(), "ETH/USDT", 2_000.0)

        assert t1 in ex.open_trades
        assert t2 in ex.open_trades
        assert len(ex.open_trades) == 2


# ======================================================================
# Risk state integration
# ======================================================================

class TestRiskStateIntegration:
    def test_balance_decreases_on_open(self):
        conn = _make_connector(fill_price=50_000.0)
        risk = _make_risk(balance=100_000.0)
        ex = OrderExecutor(conn, risk, stop_loss_pct=0.01, take_profit_pct=0.02)
        ex.execute(_long_signal(), "BTC/USDT", 50_000.0)
        assert risk.state.available_balance < 100_000.0

    def test_balance_restored_on_close(self):
        conn = _make_connector(fill_price=50_000.0)
        risk = _make_risk(balance=100_000.0)
        ex = OrderExecutor(conn, risk, stop_loss_pct=0.01, take_profit_pct=0.02)
        trade = ex.execute(_long_signal(), "BTC/USDT", 50_000.0)
        assert trade is not None
        balance_after_open = risk.state.available_balance

        conn.place_market_order.return_value = _make_order_result(side="sell", average=50_000.0)
        ex.close_trade(trade.trade_id, 50_000.0)
        assert risk.state.available_balance > balance_after_open

    def test_positions_zero_after_close(self):
        conn = _make_connector(fill_price=50_000.0)
        risk = _make_risk(balance=100_000.0)
        ex = OrderExecutor(conn, risk, stop_loss_pct=0.01, take_profit_pct=0.02)
        trade = ex.execute(_long_signal(), "BTC/USDT", 50_000.0)
        assert trade is not None
        assert risk.state.open_positions == 1

        conn.place_market_order.return_value = _make_order_result(side="sell", average=50_000.0)
        ex.close_trade(trade.trade_id, 50_000.0)
        assert risk.state.open_positions == 0
