"""
In-memory paper trading connector.

Fills market orders immediately at the last known close price (plus
configurable slippage).  Limit orders are recorded as "open" and must be
filled externally (e.g. by the paper trading engine when price crosses the
level).  No real network calls are made for order execution.

Market data (fetch_ohlcv) is delegated to an optional CCXT exchange
(read-only, no authentication required for public endpoints).  When no
CCXT exchange is provided, callers must set prices manually via
set_current_price() — this is the recommended approach for unit tests
and backtests.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from zeus.exchange.connector import ExchangeConnector, OrderResult, ohlcv_to_df
from zeus.utils.logger import logger


class PaperConnector(ExchangeConnector):
    """
    Simulated exchange connector for paper trading and backtesting.

    Args:
        initial_balance: Starting quote-currency balance (e.g. 10 000 USDT).
        ccxt_exchange:   Optional CCXT exchange for public market data.
                         When None, callers must use set_current_price().
        slippage_pct:    Slippage fraction applied to fills.
                         Buys fill at price × (1 + slippage_pct);
                         sells fill at price × (1 − slippage_pct).
    """

    def __init__(
        self,
        initial_balance: float,
        ccxt_exchange:   Any | None = None,
        slippage_pct:    float = 0.0005,
    ) -> None:
        self._balance  = initial_balance
        self._exchange = ccxt_exchange
        self._slippage = slippage_pct
        self._orders:  dict[str, OrderResult] = {}   # order_id → OrderResult
        self._prices:  dict[str, float]       = {}   # symbol  → last close price

    # ------------------------------------------------------------------ #
    # ExchangeConnector interface                                          #
    # ------------------------------------------------------------------ #

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        if self._exchange is None:
            raise RuntimeError(
                "PaperConnector has no ccxt_exchange — "
                "call set_current_price() or pass ccxt_exchange at init."
            )
        raw = self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df  = ohlcv_to_df(raw)
        if not df.empty:
            self._prices[symbol] = float(df["close"].iloc[-1])
        return df

    def fetch_balance(self) -> float:
        return self._balance

    def place_market_order(self, symbol: str, side: str, quantity: float) -> OrderResult:
        price = self._prices.get(symbol)
        if price is None or price <= 0:
            raise ValueError(
                f"No current price for '{symbol}'. "
                "Call fetch_ohlcv() or set_current_price() first."
            )

        fill_price = self._slippage_price(price, side)
        cost       = quantity * fill_price

        if side == "buy" and self._balance < cost:
            raise ValueError(
                f"Insufficient paper balance: need {cost:.4f}, "
                f"have {self._balance:.4f}"
            )

        result = self._record_fill(symbol, side, quantity, fill_price)
        self._apply_fill(side, quantity, fill_price)

        logger.info("Paper market order filled",
                    symbol=symbol, side=side, quantity=quantity,
                    fill_price=fill_price, balance=self._balance)
        return result

    def place_limit_order(
        self, symbol: str, side: str, quantity: float, price: float,
    ) -> OrderResult:
        order_id = self._new_id()
        result   = OrderResult(
            order_id  = order_id,
            symbol    = symbol,
            side      = side,
            quantity  = quantity,
            filled    = 0.0,
            price     = price,
            average   = None,
            status    = "open",
            timestamp = datetime.now(tz=timezone.utc),
        )
        self._orders[order_id] = result
        logger.info("Paper limit order placed (open)",
                    symbol=symbol, side=side, quantity=quantity, price=price)
        return result

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        order = self._orders.get(order_id)
        if order is None or order.status != "open":
            logger.warning("Paper cancel: order not found or not open",
                           order_id=order_id)
            return False
        self._orders[order_id] = _replace(order, status="cancelled")
        logger.info("Paper order cancelled", order_id=order_id)
        return True

    def fetch_order(self, order_id: str, symbol: str) -> OrderResult:
        order = self._orders.get(order_id)
        if order is None:
            raise ValueError(f"Order '{order_id}' not found")
        return order

    def fetch_open_orders(self, symbol: str) -> list[OrderResult]:
        return [
            o for o in self._orders.values()
            if o.symbol == symbol and o.status == "open"
        ]

    def is_connected(self) -> bool:
        return True

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Paper-only API                                                       #
    # ------------------------------------------------------------------ #

    def set_current_price(self, symbol: str, price: float) -> None:
        """
        Manually set the current market price for a symbol.

        Use this in unit tests and backtests instead of calling fetch_ohlcv().
        Market orders placed after this call fill at price ± slippage.
        """
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")
        self._prices[symbol] = price

    def fill_limit_order(self, order_id: str, fill_price: float) -> OrderResult:
        """
        Simulate a limit order fill at fill_price.

        Called by the paper trading engine when price crosses the order level.
        Returns the updated OrderResult.  Raises if order is not open.
        """
        order = self._orders.get(order_id)
        if order is None or order.status != "open":
            raise ValueError(f"Order '{order_id}' is not open")
        filled = _replace(order, filled=order.quantity, average=fill_price, status="closed")
        self._orders[order_id] = filled
        self._apply_fill(order.side, order.quantity, fill_price)
        return filled

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _slippage_price(self, price: float, side: str) -> float:
        if side == "buy":
            return price * (1 + self._slippage)
        return price * (1 - self._slippage)

    def _apply_fill(self, side: str, quantity: float, fill_price: float) -> None:
        if side == "buy":
            self._balance -= quantity * fill_price
        else:
            self._balance += quantity * fill_price

    def _record_fill(
        self, symbol: str, side: str, quantity: float, fill_price: float,
    ) -> OrderResult:
        order_id = self._new_id()
        result   = OrderResult(
            order_id  = order_id,
            symbol    = symbol,
            side      = side,
            quantity  = quantity,
            filled    = quantity,
            price     = None,
            average   = fill_price,
            status    = "closed",
            timestamp = datetime.now(tz=timezone.utc),
        )
        self._orders[order_id] = result
        return result

    @staticmethod
    def _new_id() -> str:
        return f"paper-{uuid.uuid4().hex[:12]}"


def _replace(order: OrderResult, **changes) -> OrderResult:
    """Return a new OrderResult with the given fields replaced."""
    d = {
        "order_id":  order.order_id,
        "symbol":    order.symbol,
        "side":      order.side,
        "quantity":  order.quantity,
        "filled":    order.filled,
        "price":     order.price,
        "average":   order.average,
        "status":    order.status,
        "timestamp": order.timestamp,
        "raw":       order.raw,
    }
    d.update(changes)
    return OrderResult(**d)
