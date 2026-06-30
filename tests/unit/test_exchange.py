"""
Unit tests for zeus.exchange.

TestOrderResult          — is_filled / is_open properties
TestOhlcvToDf            — shared CCXT-list → DataFrame conversion
TestLiveConnector        — mocked CCXT exchange object
TestLiveConnectorRetry   — retry and backoff behaviour
TestPaperConnector       — pure in-memory simulation
TestPaperConnectorLimits — limit order lifecycle (place / cancel / fill)
TestCreateConnector      — factory creates the right concrete type
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest
import ccxt

from zeus.exchange.connector import OrderResult, ohlcv_to_df
from zeus.exchange.live import LiveConnector, _parse_order
from zeus.exchange.paper import PaperConnector
from zeus.exchange.factory import create_connector
from zeus.config import Settings, Mode


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_TS_MS = 1_700_000_000_000   # 2023-11-14 22:13:20 UTC


def _raw_order(
    order_id: str = "123",
    status: str   = "closed",
    side: str     = "buy",
    amount: float = 0.1,
    filled: float = 0.1,
    price: float  = 50_000.0,
    average: float = 50_100.0,
) -> dict:
    return {
        "id":        order_id,
        "symbol":    "BTC/USDT",
        "side":      side,
        "amount":    amount,
        "filled":    filled,
        "price":     price,
        "average":   average,
        "status":    status,
        "timestamp": _TS_MS,
    }


def _raw_ohlcv(n: int = 3) -> list[list]:
    base = 50_000.0
    return [
        [_TS_MS + i * 3_600_000, base + i, base + i + 10,
         base + i - 10, base + i + 5, 1.0 + i]
        for i in range(n)
    ]


def _mock_exchange(**method_returns) -> MagicMock:
    """Build a MagicMock that returns the given values per method name."""
    ex = MagicMock()
    for method, retval in method_returns.items():
        getattr(ex, method).return_value = retval
    return ex


def _paper(balance: float = 10_000.0, slippage: float = 0.0) -> PaperConnector:
    return PaperConnector(initial_balance=balance, slippage_pct=slippage)


# ──────────────────────────────────────────────────────────────────────────────
# OrderResult
# ──────────────────────────────────────────────────────────────────────────────

class TestOrderResult:
    def _make(self, status: str, filled: float = 0.1) -> OrderResult:
        return OrderResult(
            order_id="1", symbol="BTC/USDT", side="buy",
            quantity=0.1, filled=filled, price=None, average=50_000.0,
            status=status, timestamp=datetime.now(tz=timezone.utc),
        )

    def test_is_filled_when_closed_and_nonzero(self):
        assert self._make("closed", 0.1).is_filled is True

    def test_not_filled_when_open(self):
        assert self._make("open", 0.0).is_filled is False

    def test_not_filled_when_closed_but_zero(self):
        assert self._make("closed", 0.0).is_filled is False

    def test_is_open_when_status_open(self):
        assert self._make("open").is_open is True

    def test_not_open_when_closed(self):
        assert self._make("closed").is_open is False

    def test_raw_excluded_from_equality(self):
        base = self._make("closed")
        with_raw = OrderResult(
            order_id="1", symbol="BTC/USDT", side="buy",
            quantity=0.1, filled=0.1, price=None, average=50_000.0,
            status="closed", timestamp=base.timestamp,
            raw={"extra": "data"},
        )
        assert base == with_raw


# ──────────────────────────────────────────────────────────────────────────────
# ohlcv_to_df
# ──────────────────────────────────────────────────────────────────────────────

class TestOhlcvToDf:
    def test_empty_input_returns_empty_df(self):
        df = ohlcv_to_df([])
        assert df.empty
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_returns_dataframe(self):
        df = ohlcv_to_df(_raw_ohlcv(3))
        assert isinstance(df, pd.DataFrame)

    def test_columns(self):
        df = ohlcv_to_df(_raw_ohlcv(3))
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_row_count(self):
        df = ohlcv_to_df(_raw_ohlcv(5))
        assert len(df) == 5

    def test_index_is_datetime_utc(self):
        df = ohlcv_to_df(_raw_ohlcv(1))
        assert isinstance(df.index, pd.DatetimeIndex)
        assert str(df.index.tz) == "UTC"

    def test_oldest_bar_first(self):
        raw = _raw_ohlcv(3)
        df  = ohlcv_to_df(raw)
        first_ts = pd.Timestamp(_TS_MS, unit="ms", tz="UTC")
        assert df.index[0] == first_ts

    def test_values_are_float(self):
        df = ohlcv_to_df(_raw_ohlcv(2))
        assert df.dtypes["close"] == float


# ──────────────────────────────────────────────────────────────────────────────
# LiveConnector — normal operations
# ──────────────────────────────────────────────────────────────────────────────

class TestLiveConnector:
    @pytest.fixture
    def ex(self):
        return _mock_exchange(
            fetch_ohlcv=_raw_ohlcv(10),
            fetch_balance={"free": {"USDT": 5_000.0, "BTC": 0.5}},
            create_market_order=_raw_order(order_id="m1", status="closed"),
            create_limit_order=_raw_order(order_id="l1", status="open"),
            cancel_order={"id": "l1", "status": "cancelled"},
            fetch_order=_raw_order(order_id="l1", status="closed"),
            fetch_open_orders=[_raw_order(order_id="o1", status="open")],
        )

    @pytest.fixture
    def conn(self, ex) -> LiveConnector:
        return LiveConnector(exchange=ex, base_delay=0)

    def test_fetch_ohlcv_returns_dataframe(self, conn):
        df = conn.fetch_ohlcv("BTC/USDT", "1h")
        assert isinstance(df, pd.DataFrame)

    def test_fetch_ohlcv_has_correct_columns(self, conn):
        df = conn.fetch_ohlcv("BTC/USDT", "1h")
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_fetch_ohlcv_passes_limit(self, ex, conn):
        conn.fetch_ohlcv("BTC/USDT", "1h", limit=200)
        ex.fetch_ohlcv.assert_called_once_with("BTC/USDT", "1h", limit=200)

    def test_fetch_balance_returns_usdt_free(self, conn):
        assert conn.fetch_balance() == pytest.approx(5_000.0)

    def test_fetch_balance_custom_currency(self, ex):
        conn = LiveConnector(exchange=ex, quote_currency="BTC", base_delay=0)
        assert conn.fetch_balance() == pytest.approx(0.5)

    def test_fetch_balance_missing_currency_returns_zero(self, ex):
        conn = LiveConnector(exchange=ex, quote_currency="ETH", base_delay=0)
        assert conn.fetch_balance() == pytest.approx(0.0)

    def test_place_market_order_returns_order_result(self, conn):
        r = conn.place_market_order("BTC/USDT", "buy", 0.1)
        assert isinstance(r, OrderResult)
        assert r.order_id == "m1"

    def test_place_market_order_calls_exchange(self, ex, conn):
        conn.place_market_order("BTC/USDT", "buy", 0.1)
        ex.create_market_order.assert_called_once_with("BTC/USDT", "buy", 0.1)

    def test_place_limit_order_returns_order_result(self, conn):
        r = conn.place_limit_order("BTC/USDT", "sell", 0.1, 55_000.0)
        assert isinstance(r, OrderResult)
        assert r.order_id == "l1"
        assert r.status == "open"

    def test_place_limit_order_calls_exchange(self, ex, conn):
        conn.place_limit_order("BTC/USDT", "sell", 0.1, 55_000.0)
        ex.create_limit_order.assert_called_once_with("BTC/USDT", "sell", 0.1, 55_000.0)

    def test_cancel_order_returns_true_on_success(self, conn):
        assert conn.cancel_order("l1", "BTC/USDT") is True

    def test_cancel_order_returns_false_on_not_found(self, ex, conn):
        ex.cancel_order.side_effect = ccxt.OrderNotFound()
        assert conn.cancel_order("bad-id", "BTC/USDT") is False

    def test_fetch_order_returns_parsed_result(self, conn):
        r = conn.fetch_order("l1", "BTC/USDT")
        assert isinstance(r, OrderResult)
        assert r.order_id == "l1"

    def test_fetch_open_orders_returns_list(self, conn):
        orders = conn.fetch_open_orders("BTC/USDT")
        assert isinstance(orders, list)
        assert len(orders) == 1
        assert orders[0].order_id == "o1"

    def test_is_connected_true_when_load_markets_succeeds(self, conn):
        assert conn.is_connected() is True

    def test_is_connected_false_when_load_markets_raises(self, ex, conn):
        ex.load_markets.side_effect = ccxt.NetworkError("down")
        assert conn.is_connected() is False

    def test_close_calls_exchange_close(self, ex, conn):
        conn.close()
        ex.close.assert_called_once()

    def test_close_does_not_raise_when_exchange_has_no_close(self):
        ex = MagicMock(spec=[])  # no 'close' attribute
        conn = LiveConnector(exchange=ex, base_delay=0)
        conn.close()   # must not raise


# ──────────────────────────────────────────────────────────────────────────────
# LiveConnector — retry behaviour
# ──────────────────────────────────────────────────────────────────────────────

class TestLiveConnectorRetry:
    @patch("zeus.exchange.live.time.sleep")
    def test_retries_on_network_error_then_succeeds(self, mock_sleep):
        ex = MagicMock()
        ex.fetch_ohlcv.side_effect = [
            ccxt.NetworkError("err"),
            _raw_ohlcv(3),
        ]
        conn = LiveConnector(exchange=ex, max_retries=3, base_delay=1.0)
        df = conn.fetch_ohlcv("BTC/USDT", "1h")
        assert len(df) == 3
        assert ex.fetch_ohlcv.call_count == 2
        mock_sleep.assert_called_once_with(1.0)   # attempt 0: 1 * 2^0

    @patch("zeus.exchange.live.time.sleep")
    def test_retries_on_request_timeout(self, mock_sleep):
        ex = MagicMock()
        ex.fetch_ohlcv.side_effect = [
            ccxt.RequestTimeout("timeout"),
            ccxt.RequestTimeout("timeout"),
            _raw_ohlcv(1),
        ]
        conn = LiveConnector(exchange=ex, max_retries=3, base_delay=1.0)
        conn.fetch_ohlcv("BTC/USDT", "1h")
        assert ex.fetch_ohlcv.call_count == 3
        # delays: 1*2^0=1, 1*2^1=2
        assert mock_sleep.call_args_list == [call(1.0), call(2.0)]

    @patch("zeus.exchange.live.time.sleep")
    def test_rate_limit_uses_quadratic_backoff(self, mock_sleep):
        ex = MagicMock()
        ex.fetch_ohlcv.side_effect = [
            ccxt.RateLimitExceeded("rate"),
            _raw_ohlcv(1),
        ]
        conn = LiveConnector(exchange=ex, max_retries=3, base_delay=1.0)
        conn.fetch_ohlcv("BTC/USDT", "1h")
        mock_sleep.assert_called_once_with(1.0)   # 1 * 4^0

    @patch("zeus.exchange.live.time.sleep")
    def test_does_not_retry_auth_error(self, mock_sleep):
        ex = MagicMock()
        ex.fetch_ohlcv.side_effect = ccxt.AuthenticationError("bad key")
        conn = LiveConnector(exchange=ex, max_retries=3, base_delay=1.0)
        with pytest.raises(ccxt.AuthenticationError):
            conn.fetch_ohlcv("BTC/USDT", "1h")
        mock_sleep.assert_not_called()
        assert ex.fetch_ohlcv.call_count == 1   # no retries

    @patch("zeus.exchange.live.time.sleep")
    def test_raises_after_max_retries_exhausted(self, mock_sleep):
        ex = MagicMock()
        ex.fetch_ohlcv.side_effect = ccxt.NetworkError("persistent")
        conn = LiveConnector(exchange=ex, max_retries=2, base_delay=1.0)
        with pytest.raises(ccxt.NetworkError):
            conn.fetch_ohlcv("BTC/USDT", "1h")
        assert ex.fetch_ohlcv.call_count == 3   # initial + 2 retries

    @patch("zeus.exchange.live.time.sleep")
    def test_sleep_delays_match_attempts(self, mock_sleep):
        ex = MagicMock()
        ex.fetch_balance.side_effect = [
            ccxt.NetworkError(),
            ccxt.NetworkError(),
            {"free": {"USDT": 100.0}},
        ]
        conn = LiveConnector(exchange=ex, max_retries=3, base_delay=2.0)
        conn.fetch_balance()
        # delays: 2*2^0=2, 2*2^1=4
        assert mock_sleep.call_args_list == [call(2.0), call(4.0)]


# ──────────────────────────────────────────────────────────────────────────────
# PaperConnector — market orders and balance
# ──────────────────────────────────────────────────────────────────────────────

class TestPaperConnector:
    @pytest.fixture
    def paper(self):
        p = _paper(balance=10_000.0, slippage=0.0)
        p.set_current_price("BTC/USDT", 50_000.0)
        return p

    def test_initial_balance(self):
        assert _paper(5_000.0).fetch_balance() == pytest.approx(5_000.0)

    def test_is_connected_always_true(self, paper):
        assert paper.is_connected() is True

    def test_close_does_not_raise(self, paper):
        paper.close()   # no-op

    def test_set_current_price_stores_price(self):
        p = _paper()
        p.set_current_price("BTC/USDT", 42_000.0)
        # verify it's used by placing an order
        p.place_market_order("BTC/USDT", "sell", 1.0)
        assert p.fetch_balance() == pytest.approx(10_000.0 + 42_000.0)

    def test_set_current_price_rejects_non_positive(self):
        with pytest.raises(ValueError):
            _paper().set_current_price("BTC/USDT", 0.0)

    def test_place_market_buy_deducts_balance(self, paper):
        paper.place_market_order("BTC/USDT", "buy", 0.1)
        assert paper.fetch_balance() == pytest.approx(10_000.0 - 0.1 * 50_000.0)

    def test_place_market_sell_increases_balance(self, paper):
        paper.place_market_order("BTC/USDT", "sell", 0.1)
        assert paper.fetch_balance() == pytest.approx(10_000.0 + 0.1 * 50_000.0)

    def test_slippage_applied_to_buy(self):
        p = _paper(balance=100_000.0, slippage=0.01)
        p.set_current_price("BTC/USDT", 50_000.0)
        r = p.place_market_order("BTC/USDT", "buy", 1.0)
        assert r.average == pytest.approx(50_000.0 * 1.01)

    def test_slippage_applied_to_sell(self):
        p = _paper(slippage=0.01)
        p.set_current_price("BTC/USDT", 50_000.0)
        r = p.place_market_order("BTC/USDT", "sell", 1.0)
        assert r.average == pytest.approx(50_000.0 * 0.99)

    def test_market_order_status_is_closed(self, paper):
        r = paper.place_market_order("BTC/USDT", "buy", 0.01)
        assert r.status == "closed"

    def test_market_order_filled_equals_quantity(self, paper):
        r = paper.place_market_order("BTC/USDT", "buy", 0.05)
        assert r.filled == pytest.approx(0.05)

    def test_market_order_is_filled(self, paper):
        r = paper.place_market_order("BTC/USDT", "buy", 0.01)
        assert r.is_filled is True

    def test_market_order_price_is_none(self, paper):
        r = paper.place_market_order("BTC/USDT", "buy", 0.01)
        assert r.price is None

    def test_market_order_has_timestamp(self, paper):
        r = paper.place_market_order("BTC/USDT", "buy", 0.01)
        assert isinstance(r.timestamp, datetime)

    def test_market_order_id_starts_with_paper(self, paper):
        r = paper.place_market_order("BTC/USDT", "buy", 0.01)
        assert r.order_id.startswith("paper-")

    def test_insufficient_balance_raises(self):
        p = _paper(balance=100.0, slippage=0.0)
        p.set_current_price("BTC/USDT", 50_000.0)
        with pytest.raises(ValueError, match="Insufficient"):
            p.place_market_order("BTC/USDT", "buy", 1.0)

    def test_place_market_order_without_price_raises(self):
        p = _paper()
        with pytest.raises(ValueError, match="No current price"):
            p.place_market_order("BTC/USDT", "buy", 0.1)

    def test_fetch_order_returns_stored_order(self, paper):
        r = paper.place_market_order("BTC/USDT", "buy", 0.01)
        fetched = paper.fetch_order(r.order_id, "BTC/USDT")
        assert fetched.order_id == r.order_id

    def test_fetch_order_raises_for_unknown_id(self, paper):
        with pytest.raises(ValueError, match="not found"):
            paper.fetch_order("unknown-id", "BTC/USDT")

    def test_fetch_ohlcv_raises_without_ccxt_exchange(self):
        p = PaperConnector(initial_balance=10_000.0)
        with pytest.raises(RuntimeError, match="ccxt_exchange"):
            p.fetch_ohlcv("BTC/USDT", "1h")

    def test_fetch_ohlcv_sets_price(self):
        mock_ex = MagicMock()
        mock_ex.fetch_ohlcv.return_value = _raw_ohlcv(3)
        p = PaperConnector(initial_balance=10_000.0, ccxt_exchange=mock_ex, slippage_pct=0.0)
        p.fetch_ohlcv("BTC/USDT", "1h")
        # last close from _raw_ohlcv: base + (n-1) + 5 = 50_000 + 2 + 5 = 50_007
        r = p.place_market_order("BTC/USDT", "sell", 1.0)
        assert r.average == pytest.approx(50_007.0)

    def test_multiple_market_orders_cumulative_balance(self, paper):
        paper.place_market_order("BTC/USDT", "buy",  0.1)   # -5000
        paper.place_market_order("BTC/USDT", "sell", 0.05)  # +2500
        assert paper.fetch_balance() == pytest.approx(10_000.0 - 5_000.0 + 2_500.0)


# ──────────────────────────────────────────────────────────────────────────────
# PaperConnector — limit order lifecycle
# ──────────────────────────────────────────────────────────────────────────────

class TestPaperConnectorLimits:
    @pytest.fixture
    def paper(self):
        p = _paper(balance=10_000.0, slippage=0.0)
        p.set_current_price("BTC/USDT", 50_000.0)
        return p

    def test_limit_order_status_is_open(self, paper):
        r = paper.place_limit_order("BTC/USDT", "buy", 0.1, 48_000.0)
        assert r.status == "open"

    def test_limit_order_filled_is_zero(self, paper):
        r = paper.place_limit_order("BTC/USDT", "buy", 0.1, 48_000.0)
        assert r.filled == pytest.approx(0.0)

    def test_limit_order_average_is_none(self, paper):
        r = paper.place_limit_order("BTC/USDT", "buy", 0.1, 48_000.0)
        assert r.average is None

    def test_limit_order_is_open(self, paper):
        r = paper.place_limit_order("BTC/USDT", "buy", 0.1, 48_000.0)
        assert r.is_open is True

    def test_cancel_open_limit_order(self, paper):
        r = paper.place_limit_order("BTC/USDT", "buy", 0.1, 48_000.0)
        assert paper.cancel_order(r.order_id, "BTC/USDT") is True
        assert paper.fetch_order(r.order_id, "BTC/USDT").status == "cancelled"

    def test_cancel_returns_false_for_unknown_id(self, paper):
        assert paper.cancel_order("no-such-id", "BTC/USDT") is False

    def test_cancel_returns_false_for_already_closed(self, paper):
        r = paper.place_market_order("BTC/USDT", "buy", 0.01)
        assert paper.cancel_order(r.order_id, "BTC/USDT") is False

    def test_fetch_open_orders_returns_only_open(self, paper):
        r1 = paper.place_limit_order("BTC/USDT", "buy",  0.1, 48_000.0)
        r2 = paper.place_limit_order("BTC/USDT", "sell", 0.1, 52_000.0)
        paper.cancel_order(r1.order_id, "BTC/USDT")
        open_orders = paper.fetch_open_orders("BTC/USDT")
        ids = [o.order_id for o in open_orders]
        assert r1.order_id not in ids
        assert r2.order_id in ids

    def test_fill_limit_order_updates_balance(self, paper):
        r = paper.place_limit_order("BTC/USDT", "buy", 0.1, 48_000.0)
        paper.fill_limit_order(r.order_id, fill_price=48_000.0)
        assert paper.fetch_balance() == pytest.approx(10_000.0 - 0.1 * 48_000.0)

    def test_fill_limit_order_status_becomes_closed(self, paper):
        r = paper.place_limit_order("BTC/USDT", "buy", 0.1, 48_000.0)
        filled = paper.fill_limit_order(r.order_id, fill_price=48_000.0)
        assert filled.status == "closed"
        assert filled.is_filled is True

    def test_fill_limit_order_raises_when_not_open(self, paper):
        r = paper.place_limit_order("BTC/USDT", "buy", 0.1, 48_000.0)
        paper.cancel_order(r.order_id, "BTC/USDT")
        with pytest.raises(ValueError, match="not open"):
            paper.fill_limit_order(r.order_id, fill_price=48_000.0)

    def test_fill_limit_order_no_longer_in_open_orders(self, paper):
        r = paper.place_limit_order("BTC/USDT", "buy", 0.1, 48_000.0)
        paper.fill_limit_order(r.order_id, fill_price=48_000.0)
        assert paper.fetch_open_orders("BTC/USDT") == []


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

class TestCreateConnector:
    def _settings(self, mode: str = "paper", balance: float = 5_000.0) -> Settings:
        return Settings(
            mode=mode,
            exchange="binance",
            api_key="key" if mode == "live" else "",
            api_secret="secret" if mode == "live" else "",
            paper_balance=balance,
        )

    def test_paper_mode_creates_paper_connector(self):
        with patch("zeus.exchange.factory.ccxt") as mock_ccxt:
            mock_ccxt.binance.return_value = MagicMock()
            conn = create_connector(self._settings("paper"))
        assert isinstance(conn, PaperConnector)

    def test_paper_connector_uses_configured_balance(self):
        with patch("zeus.exchange.factory.ccxt") as mock_ccxt:
            mock_ccxt.binance.return_value = MagicMock()
            conn = create_connector(self._settings("paper", balance=7_500.0))
        assert conn.fetch_balance() == pytest.approx(7_500.0)

    def test_live_mode_creates_live_connector(self):
        with patch("zeus.exchange.factory.ccxt") as mock_ccxt:
            mock_ccxt.binance.return_value = MagicMock()
            conn = create_connector(self._settings("live"))
        assert isinstance(conn, LiveConnector)

    def test_live_mode_passes_api_key_to_exchange(self):
        with patch("zeus.exchange.factory.ccxt") as mock_ccxt:
            mock_ccxt.binance.return_value = MagicMock()
            create_connector(self._settings("live"))
            _, kwargs = mock_ccxt.binance.call_args
            # called with a dict containing apiKey
            init_dict = mock_ccxt.binance.call_args[0][0]
            assert init_dict.get("apiKey") == "key"
            assert init_dict.get("secret") == "secret"
