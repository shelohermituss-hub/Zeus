"""
Tests for zeus/exchange/mt5.py.

MetaTrader5 is a Windows-only library not present in CI. All tests patch
zeus.exchange.mt5.mt5 and zeus.exchange.mt5._MT5_AVAILABLE so the connector
can be exercised on any platform without the real MT5 terminal.
"""
from __future__ import annotations

import time
from datetime import timezone
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

import zeus.exchange.mt5 as mt5_module
from zeus.exchange.mt5 import MT5Connector, _units_to_lots


# ======================================================================
# Mock-MT5 factory
# ======================================================================

def _make_mt5() -> MagicMock:
    """Return a MagicMock wired with real-ish MT5 constants and defaults."""
    m = MagicMock()

    # Timeframe constants
    m.TIMEFRAME_M1  = 1
    m.TIMEFRAME_M5  = 5
    m.TIMEFRAME_M15 = 15
    m.TIMEFRAME_M30 = 30
    m.TIMEFRAME_H1  = 16385
    m.TIMEFRAME_H2  = 16386
    m.TIMEFRAME_H4  = 16388
    m.TIMEFRAME_H6  = 16390
    m.TIMEFRAME_H12 = 16396
    m.TIMEFRAME_D1  = 16408
    m.TIMEFRAME_W1  = 32769

    # Trade constants
    m.TRADE_RETCODE_DONE  = 10009
    m.TRADE_ACTION_DEAL   = 1
    m.TRADE_ACTION_PENDING = 5
    m.TRADE_ACTION_REMOVE = 8
    m.ORDER_TYPE_BUY        = 0
    m.ORDER_TYPE_SELL       = 1
    m.ORDER_TYPE_BUY_LIMIT  = 2
    m.ORDER_TYPE_SELL_LIMIT = 3
    m.ORDER_TYPE_BUY_STOP   = 4
    m.ORDER_TIME_GTC        = 1
    m.ORDER_FILLING_IOC     = 1
    m.ORDER_FILLING_RETURN  = 2
    m.ORDER_STATE_FILLED    = 1
    m.ORDER_STATE_CANCELED  = 3

    # Default: initialize succeeds
    m.initialize.return_value = True

    return m


def _sym_info(
    contract_size: float = 100.0,
    volume_min:    float = 0.01,
    volume_max:    float = 100.0,
    volume_step:   float = 0.01,
) -> MagicMock:
    info = MagicMock()
    info.trade_contract_size = contract_size
    info.volume_min  = volume_min
    info.volume_max  = volume_max
    info.volume_step = volume_step
    return info


def _tick(ask: float = 1980.0, bid: float = 1979.5) -> MagicMock:
    t = MagicMock()
    t.ask = ask
    t.bid = bid
    return t


def _order_result(
    retcode:  int   = 10009,
    order:    int   = 123,
    deal:     int   = 456,
    volume:   float = 0.01,
    price:    float = 1980.0,
    comment:  str   = "done",
) -> MagicMock:
    r = MagicMock()
    r.retcode = retcode
    r.order   = order
    r.deal    = deal
    r.volume  = volume
    r.price   = price
    r.comment = comment
    return r


# ======================================================================
# Context manager that patches the mt5 module and _MT5_AVAILABLE flag
# ======================================================================

from contextlib import contextmanager

@contextmanager
def _mt5_patch(mock_mt5: MagicMock):
    with patch.object(mt5_module, "mt5", mock_mt5), \
         patch.object(mt5_module, "_MT5_AVAILABLE", True):
        yield mock_mt5


# ======================================================================
# Helper: build a connected connector
# ======================================================================

def _connector(mock_mt5: MagicMock, **kwargs) -> MT5Connector:
    return MT5Connector(login=12345, password="secret", server="Demo", **kwargs)


# ======================================================================
# TestLotConversion — _units_to_lots utility
# ======================================================================

class TestLotConversion:
    def test_basic_conversion(self):
        # 10 oz gold at contract_size=100 → 0.10 lots
        lots = _units_to_lots(10.0, 100.0, 0.01, 100.0, 0.01)
        assert lots == pytest.approx(0.10)

    def test_clamped_to_minimum(self):
        lots = _units_to_lots(0.001, 100.0, 0.01, 100.0, 0.01)
        assert lots == pytest.approx(0.01)

    def test_clamped_to_maximum(self):
        lots = _units_to_lots(100_000.0, 100.0, 0.01, 100.0, 0.01)
        assert lots == pytest.approx(100.0)

    def test_rounded_to_step(self):
        # 15.5 oz / 100 = 0.155 lots → rounded to step 0.01 → 0.16
        lots = _units_to_lots(15.5, 100.0, 0.01, 100.0, 0.01)
        assert lots == pytest.approx(0.16, abs=1e-9)


# ======================================================================
# TestInitialization
# ======================================================================

class TestInitialization:
    def test_init_calls_mt5_initialize(self):
        mock = _make_mt5()
        with _mt5_patch(mock):
            _connector(mock)
        mock.initialize.assert_called_once_with(
            login=12345, password="secret", server="Demo"
        )

    def test_init_raises_when_mt5_unavailable(self):
        with patch.object(mt5_module, "_MT5_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="not available on this platform"):
                MT5Connector(login=1, password="p", server="s")

    def test_init_raises_when_mt5_initialize_fails(self):
        mock = _make_mt5()
        mock.initialize.return_value = False
        mock.last_error.return_value = (-10006, "connection failed")
        with _mt5_patch(mock):
            with pytest.raises(RuntimeError, match="MT5 initialization failed"):
                _connector(mock)

    def test_custom_deviation_stored(self):
        mock = _make_mt5()
        with _mt5_patch(mock):
            conn = MT5Connector(login=1, password="p", server="s", deviation=50)
        assert conn._deviation == 50


# ======================================================================
# TestFetchOhlcv
# ======================================================================

class TestFetchOhlcv:
    def _rates(self, n: int = 5) -> list:
        """Build a minimal structured-array-like list of rate dicts."""
        base_time = int(time.time()) - n * 3600
        rows = []
        for i in range(n):
            rows.append({
                "time":        base_time + i * 3600,
                "open":        1900.0 + i,
                "high":        1905.0 + i,
                "low":         1895.0 + i,
                "close":       1902.0 + i,
                "tick_volume": 100 + i,
                "spread":      5,
                "real_volume": 0,
            })
        return rows

    def test_returns_dataframe_with_correct_columns(self):
        mock = _make_mt5()
        mock.copy_rates_from_pos.return_value = self._rates(10)
        with _mt5_patch(mock):
            conn = _connector(mock)
            df = conn.fetch_ohlcv("XAUUSD", "1h", limit=10)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 10

    def test_index_is_utc_datetime(self):
        mock = _make_mt5()
        mock.copy_rates_from_pos.return_value = self._rates(3)
        with _mt5_patch(mock):
            conn = _connector(mock)
            df = conn.fetch_ohlcv("XAUUSD", "1h", limit=3)
        assert df.index.tz == timezone.utc

    def test_correct_timeframe_constant_passed(self):
        mock = _make_mt5()
        mock.copy_rates_from_pos.return_value = self._rates(5)
        with _mt5_patch(mock):
            conn = _connector(mock)
            conn.fetch_ohlcv("XAUUSD", "1h", limit=5)
        mock.copy_rates_from_pos.assert_called_once_with("XAUUSD", 16385, 0, 5)

    def test_h4_timeframe_constant(self):
        mock = _make_mt5()
        mock.copy_rates_from_pos.return_value = self._rates(3)
        with _mt5_patch(mock):
            conn = _connector(mock)
            conn.fetch_ohlcv("GBPUSD", "4h", limit=3)
        mock.copy_rates_from_pos.assert_called_once_with("GBPUSD", 16388, 0, 3)

    def test_empty_result_returns_empty_dataframe(self):
        mock = _make_mt5()
        mock.copy_rates_from_pos.return_value = []
        with _mt5_patch(mock):
            conn = _connector(mock)
            df = conn.fetch_ohlcv("XAUUSD", "1h", limit=5)
        assert df.empty
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_none_result_returns_empty_dataframe(self):
        mock = _make_mt5()
        mock.copy_rates_from_pos.return_value = None
        with _mt5_patch(mock):
            conn = _connector(mock)
            df = conn.fetch_ohlcv("XAUUSD", "1h", limit=5)
        assert df.empty

    def test_unsupported_timeframe_raises(self):
        mock = _make_mt5()
        with _mt5_patch(mock):
            conn = _connector(mock)
            with pytest.raises(ValueError, match="Unsupported timeframe"):
                conn.fetch_ohlcv("XAUUSD", "2d", limit=5)


# ======================================================================
# TestFetchBalance
# ======================================================================

class TestFetchBalance:
    def test_returns_margin_free(self):
        mock = _make_mt5()
        acct = MagicMock()
        acct.margin_free = 8_500.0
        mock.account_info.return_value = acct
        with _mt5_patch(mock):
            conn = _connector(mock)
            balance = conn.fetch_balance()
        assert balance == pytest.approx(8_500.0)

    def test_raises_when_account_info_none(self):
        mock = _make_mt5()
        mock.account_info.return_value = None
        with _mt5_patch(mock):
            conn = _connector(mock)
            with pytest.raises(RuntimeError, match="Cannot fetch account info"):
                conn.fetch_balance()


# ======================================================================
# TestPlaceMarketOrder
# ======================================================================

class TestPlaceMarketOrder:
    def _setup(self, mock: MagicMock, side: str = "buy") -> MagicMock:
        mock.symbol_info.return_value       = _sym_info(100.0, 0.01, 100.0, 0.01)
        mock.symbol_info_tick.return_value  = _tick(ask=1980.0, bid=1979.5)
        mock.order_send.return_value        = _order_result(
            retcode=10009, order=123, deal=456, volume=0.10, price=1980.0
        )
        return mock

    def test_buy_order_result_is_closed(self):
        mock = _make_mt5()
        self._setup(mock)
        with _mt5_patch(mock):
            conn = _connector(mock)
            result = conn.place_market_order("XAUUSD", "buy", 10.0)
        assert result.status == "closed"
        assert result.side == "buy"
        assert result.symbol == "XAUUSD"

    def test_buy_uses_ask_price(self):
        mock = _make_mt5()
        self._setup(mock)
        with _mt5_patch(mock):
            conn = _connector(mock)
            conn.place_market_order("XAUUSD", "buy", 10.0)
        request = mock.order_send.call_args[0][0]
        assert request["price"] == pytest.approx(1980.0)
        assert request["type"] == mock.ORDER_TYPE_BUY

    def test_sell_uses_bid_price(self):
        mock = _make_mt5()
        self._setup(mock, side="sell")
        mock.order_send.return_value = _order_result(
            retcode=10009, volume=0.10, price=1979.5
        )
        with _mt5_patch(mock):
            conn = _connector(mock)
            conn.place_market_order("XAUUSD", "sell", 10.0)
        request = mock.order_send.call_args[0][0]
        assert request["price"] == pytest.approx(1979.5)
        assert request["type"] == mock.ORDER_TYPE_SELL

    def test_average_fill_price_set(self):
        mock = _make_mt5()
        self._setup(mock)
        with _mt5_patch(mock):
            conn = _connector(mock)
            result = conn.place_market_order("XAUUSD", "buy", 10.0)
        assert result.average == pytest.approx(1980.0)

    def test_filled_quantity_in_base_units(self):
        mock = _make_mt5()
        self._setup(mock)
        with _mt5_patch(mock):
            conn = _connector(mock)
            result = conn.place_market_order("XAUUSD", "buy", 10.0)
        # 0.10 lots × 100 contract_size = 10.0 units
        assert result.filled == pytest.approx(10.0)

    def test_order_id_is_string(self):
        mock = _make_mt5()
        self._setup(mock)
        with _mt5_patch(mock):
            conn = _connector(mock)
            result = conn.place_market_order("XAUUSD", "buy", 10.0)
        assert result.order_id == "123"

    def test_raises_when_symbol_not_found(self):
        mock = _make_mt5()
        mock.symbol_info.return_value = None
        with _mt5_patch(mock):
            conn = _connector(mock)
            with pytest.raises(RuntimeError, match="not found in MT5"):
                conn.place_market_order("BADSY", "buy", 1.0)

    def test_raises_when_no_tick(self):
        mock = _make_mt5()
        mock.symbol_info.return_value      = _sym_info()
        mock.symbol_info_tick.return_value = None
        with _mt5_patch(mock):
            conn = _connector(mock)
            with pytest.raises(RuntimeError, match="No tick data"):
                conn.place_market_order("XAUUSD", "buy", 1.0)

    def test_raises_on_rejected_order(self):
        mock = _make_mt5()
        mock.symbol_info.return_value      = _sym_info()
        mock.symbol_info_tick.return_value = _tick()
        mock.order_send.return_value = _order_result(retcode=10018, comment="no money")
        with _mt5_patch(mock):
            conn = _connector(mock)
            with pytest.raises(RuntimeError, match="Market order rejected"):
                conn.place_market_order("XAUUSD", "buy", 10.0)

    def test_raises_when_order_send_returns_none(self):
        mock = _make_mt5()
        mock.symbol_info.return_value      = _sym_info()
        mock.symbol_info_tick.return_value = _tick()
        mock.order_send.return_value = None
        with _mt5_patch(mock):
            conn = _connector(mock)
            with pytest.raises(RuntimeError, match="Market order rejected"):
                conn.place_market_order("XAUUSD", "buy", 10.0)

    def test_lot_size_sent_to_order_send(self):
        mock = _make_mt5()
        mock.symbol_info.return_value      = _sym_info(contract_size=100.0, volume_step=0.01)
        mock.symbol_info_tick.return_value = _tick()
        mock.order_send.return_value       = _order_result(volume=0.10)
        with _mt5_patch(mock):
            conn = _connector(mock)
            conn.place_market_order("XAUUSD", "buy", 10.0)
        request = mock.order_send.call_args[0][0]
        assert request["volume"] == pytest.approx(0.10)

    def test_deviation_forwarded(self):
        mock = _make_mt5()
        mock.symbol_info.return_value      = _sym_info()
        mock.symbol_info_tick.return_value = _tick()
        mock.order_send.return_value       = _order_result()
        with _mt5_patch(mock):
            conn = MT5Connector(login=1, password="p", server="s", deviation=30)
            conn.place_market_order("XAUUSD", "buy", 10.0)
        request = mock.order_send.call_args[0][0]
        assert request["deviation"] == 30


# ======================================================================
# TestPlaceLimitOrder
# ======================================================================

class TestPlaceLimitOrder:
    def _setup(self, mock: MagicMock) -> None:
        mock.symbol_info.return_value  = _sym_info()
        mock.order_send.return_value   = _order_result(retcode=10009, order=789)

    def test_limit_order_status_is_open(self):
        mock = _make_mt5()
        self._setup(mock)
        with _mt5_patch(mock):
            conn = _connector(mock)
            result = conn.place_limit_order("GBPUSD", "buy", 10_000.0, 1.25000)
        assert result.status == "open"
        assert result.filled == pytest.approx(0.0)

    def test_limit_price_stored(self):
        mock = _make_mt5()
        self._setup(mock)
        with _mt5_patch(mock):
            conn = _connector(mock)
            result = conn.place_limit_order("GBPUSD", "sell", 10_000.0, 1.26000)
        assert result.price == pytest.approx(1.26000)

    def test_buy_limit_order_type(self):
        mock = _make_mt5()
        self._setup(mock)
        with _mt5_patch(mock):
            conn = _connector(mock)
            conn.place_limit_order("GBPUSD", "buy", 10_000.0, 1.25000)
        request = mock.order_send.call_args[0][0]
        assert request["type"] == mock.ORDER_TYPE_BUY_LIMIT

    def test_sell_limit_order_type(self):
        mock = _make_mt5()
        self._setup(mock)
        with _mt5_patch(mock):
            conn = _connector(mock)
            conn.place_limit_order("GBPUSD", "sell", 10_000.0, 1.26000)
        request = mock.order_send.call_args[0][0]
        assert request["type"] == mock.ORDER_TYPE_SELL_LIMIT

    def test_raises_on_rejected(self):
        mock = _make_mt5()
        mock.symbol_info.return_value = _sym_info()
        mock.order_send.return_value  = _order_result(retcode=10014, comment="invalid price")
        with _mt5_patch(mock):
            conn = _connector(mock)
            with pytest.raises(RuntimeError, match="Limit order rejected"):
                conn.place_limit_order("GBPUSD", "buy", 10_000.0, 0.0)


# ======================================================================
# TestCancelOrder
# ======================================================================

class TestCancelOrder:
    def test_returns_true_on_success(self):
        mock = _make_mt5()
        mock.order_send.return_value = _order_result(retcode=10009)
        with _mt5_patch(mock):
            conn = _connector(mock)
            assert conn.cancel_order("789", "GBPUSD") is True

    def test_returns_false_on_failure(self):
        mock = _make_mt5()
        mock.order_send.return_value = _order_result(retcode=10006)
        with _mt5_patch(mock):
            conn = _connector(mock)
            assert conn.cancel_order("789", "GBPUSD") is False

    def test_returns_false_when_send_returns_none(self):
        mock = _make_mt5()
        mock.order_send.return_value = None
        with _mt5_patch(mock):
            conn = _connector(mock)
            assert conn.cancel_order("789", "GBPUSD") is False

    def test_sends_remove_action(self):
        mock = _make_mt5()
        mock.order_send.return_value = _order_result()
        with _mt5_patch(mock):
            conn = _connector(mock)
            conn.cancel_order("789", "GBPUSD")
        request = mock.order_send.call_args[0][0]
        assert request["action"] == mock.TRADE_ACTION_REMOVE
        assert request["order"] == 789


# ======================================================================
# TestFetchOrder
# ======================================================================

class TestFetchOrder:
    def test_returns_open_status_for_pending_order(self):
        mock = _make_mt5()
        pending = MagicMock()
        pending.ticket = 999
        pending.type   = mock.ORDER_TYPE_BUY_LIMIT
        pending.volume_initial = 0.10
        pending.price_open     = 1950.0
        pending.time_setup     = int(time.time())
        mock.orders_get.return_value = [pending]
        mock.symbol_info.return_value = _sym_info()
        with _mt5_patch(mock):
            conn = _connector(mock)
            result = conn.fetch_order("999", "XAUUSD")
        assert result.status == "open"
        assert result.side == "buy"

    def test_returns_closed_status_for_filled_history(self):
        mock = _make_mt5()
        mock.orders_get.return_value = []
        hist = MagicMock()
        hist.ticket         = 999
        hist.type           = mock.ORDER_TYPE_BUY
        hist.state          = mock.ORDER_STATE_FILLED
        hist.volume_initial = 0.10
        hist.price_open     = 1950.0
        hist.time_setup     = int(time.time())
        mock.history_orders_get.return_value = [hist]
        deal = MagicMock()
        deal.price  = 1951.0
        deal.volume = 0.10
        mock.history_deals_get.return_value = [deal]
        mock.symbol_info.return_value = _sym_info()
        with _mt5_patch(mock):
            conn = _connector(mock)
            result = conn.fetch_order("999", "XAUUSD")
        assert result.status == "closed"
        assert result.average == pytest.approx(1951.0)

    def test_returns_cancelled_status(self):
        mock = _make_mt5()
        mock.orders_get.return_value = []
        hist = MagicMock()
        hist.ticket         = 999
        hist.type           = mock.ORDER_TYPE_BUY_LIMIT
        hist.state          = mock.ORDER_STATE_CANCELED
        hist.volume_initial = 0.10
        hist.price_open     = 1950.0
        hist.time_setup     = int(time.time())
        mock.history_orders_get.return_value = [hist]
        mock.history_deals_get.return_value  = []
        mock.symbol_info.return_value = _sym_info()
        with _mt5_patch(mock):
            conn = _connector(mock)
            result = conn.fetch_order("999", "XAUUSD")
        assert result.status == "cancelled"

    def test_raises_when_order_not_found(self):
        mock = _make_mt5()
        mock.orders_get.return_value         = []
        mock.history_orders_get.return_value = []
        with _mt5_patch(mock):
            conn = _connector(mock)
            with pytest.raises(RuntimeError, match="not found in MT5"):
                conn.fetch_order("999", "XAUUSD")


# ======================================================================
# TestFetchOpenOrders
# ======================================================================

class TestFetchOpenOrders:
    def test_returns_empty_list_when_no_orders(self):
        mock = _make_mt5()
        mock.orders_get.return_value  = []
        mock.symbol_info.return_value = _sym_info()
        with _mt5_patch(mock):
            conn = _connector(mock)
            assert conn.fetch_open_orders("XAUUSD") == []

    def test_returns_empty_list_when_orders_get_returns_none(self):
        mock = _make_mt5()
        mock.orders_get.return_value = None
        with _mt5_patch(mock):
            conn = _connector(mock)
            assert conn.fetch_open_orders("XAUUSD") == []

    def test_returns_open_orders(self):
        mock = _make_mt5()
        o1 = MagicMock()
        o1.ticket = 1; o1.type = mock.ORDER_TYPE_BUY_LIMIT
        o1.volume_initial = 0.10; o1.price_open = 1950.0
        o1.time_setup = int(time.time())
        o2 = MagicMock()
        o2.ticket = 2; o2.type = mock.ORDER_TYPE_SELL_LIMIT
        o2.volume_initial = 0.05; o2.price_open = 2000.0
        o2.time_setup = int(time.time())
        mock.orders_get.return_value  = [o1, o2]
        mock.symbol_info.return_value = _sym_info()
        with _mt5_patch(mock):
            conn = _connector(mock)
            orders = conn.fetch_open_orders("XAUUSD")
        assert len(orders) == 2
        assert orders[0].side == "buy"
        assert orders[1].side == "sell"


# ======================================================================
# TestIsConnected
# ======================================================================

class TestIsConnected:
    def test_returns_true_when_terminal_connected(self):
        mock = _make_mt5()
        info = MagicMock()
        info.connected = True
        mock.terminal_info.return_value = info
        with _mt5_patch(mock):
            conn = _connector(mock)
            assert conn.is_connected() is True

    def test_returns_false_when_terminal_disconnected(self):
        mock = _make_mt5()
        info = MagicMock()
        info.connected = False
        mock.terminal_info.return_value = info
        with _mt5_patch(mock):
            conn = _connector(mock)
            assert conn.is_connected() is False

    def test_returns_false_when_terminal_info_none(self):
        mock = _make_mt5()
        mock.terminal_info.return_value = None
        with _mt5_patch(mock):
            conn = _connector(mock)
            assert conn.is_connected() is False


# ======================================================================
# TestClose
# ======================================================================

class TestClose:
    def test_calls_mt5_shutdown(self):
        mock = _make_mt5()
        with _mt5_patch(mock):
            conn = _connector(mock)
            conn.close()
        mock.shutdown.assert_called_once()


# ======================================================================
# TestLotsFromNotional
# ======================================================================

class TestLotsFromNotional:
    def test_basic_notional_conversion(self):
        mock = _make_mt5()
        mock.symbol_info.return_value      = _sym_info(contract_size=100.0)
        mock.symbol_info_tick.return_value = _tick(ask=2000.0)
        with _mt5_patch(mock):
            conn = _connector(mock)
            # $10,000 / (100 oz * $2000/oz) = 0.05 lots
            lots = conn.lots_from_notional("XAUUSD", 10_000.0)
        assert lots == pytest.approx(0.05)

    def test_raises_when_symbol_not_found(self):
        mock = _make_mt5()
        mock.symbol_info.return_value = None
        with _mt5_patch(mock):
            conn = _connector(mock)
            with pytest.raises(RuntimeError, match="not found in MT5"):
                conn.lots_from_notional("BADSY", 1000.0)

    def test_raises_when_no_tick(self):
        mock = _make_mt5()
        mock.symbol_info.return_value      = _sym_info()
        mock.symbol_info_tick.return_value = None
        with _mt5_patch(mock):
            conn = _connector(mock)
            with pytest.raises(RuntimeError, match="No tick data"):
                conn.lots_from_notional("XAUUSD", 1000.0)
