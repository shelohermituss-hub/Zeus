"""
CCXT-backed live exchange connector.

Wraps every API call with exponential-backoff retry logic for transient
network errors.  Permanent errors (auth failures, invalid orders, insufficient
funds) are raised immediately — they indicate configuration mistakes or
strategy bugs, not momentary connectivity issues.

Retry categories
----------------
Transient (2-second base, doubles per attempt):
    NetworkError, RequestTimeout, ExchangeNotAvailable

Rate-limited (4-second base, quadruples per attempt — longer wait):
    RateLimitExceeded

Permanent (no retry):
    AuthenticationError, InsufficientFunds, InvalidOrder, BadSymbol, …
    (anything not in the transient set)
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import ccxt
import pandas as pd

from zeus.exchange.connector import ExchangeConnector, OrderResult, ohlcv_to_df
from zeus.utils.logger import logger


_TRANSIENT = (
    ccxt.NetworkError,
    ccxt.RequestTimeout,
    ccxt.ExchangeNotAvailable,
)


def _parse_order(raw: dict) -> OrderResult:
    """Convert a CCXT order dict to a normalised OrderResult."""
    ts = raw.get("timestamp")
    dt = (
        datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        if ts is not None
        else datetime.now(tz=timezone.utc)
    )
    return OrderResult(
        order_id  = str(raw["id"]),
        symbol    = raw.get("symbol", ""),
        side      = raw.get("side", ""),
        quantity  = float(raw.get("amount") or 0),
        filled    = float(raw.get("filled") or 0),
        price     = float(raw["price"])   if raw.get("price")   else None,
        average   = float(raw["average"]) if raw.get("average") else None,
        status    = raw.get("status", "unknown"),
        timestamp = dt,
        raw       = raw,
    )


class LiveConnector(ExchangeConnector):
    """
    CCXT-backed connector for live trading.

    Inject a pre-configured ccxt.Exchange instance so credentials are
    never handled inside this class.  Use factory.create_connector() to
    build one from Settings.

    Args:
        exchange:       A ccxt.Exchange instance with API keys configured.
        quote_currency: Quote currency to extract from fetch_balance()
                        responses (default "USDT").
        max_retries:    Retry attempts on transient errors (default 3).
        base_delay:     Base sleep in seconds for backoff (default 1.0).
    """

    def __init__(
        self,
        exchange:       Any,
        quote_currency: str   = "USDT",
        max_retries:    int   = 3,
        base_delay:     float = 1.0,
    ) -> None:
        self._exchange       = exchange
        self._quote_currency = quote_currency
        self._max_retries    = max_retries
        self._base_delay     = base_delay

    # ------------------------------------------------------------------ #
    # ExchangeConnector interface                                          #
    # ------------------------------------------------------------------ #

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        raw = self._call(
            lambda: self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        )
        return ohlcv_to_df(raw)

    def fetch_balance(self) -> float:
        raw = self._call(lambda: self._exchange.fetch_balance())
        return float(raw.get("free", {}).get(self._quote_currency, 0.0))

    def place_market_order(self, symbol: str, side: str, quantity: float) -> OrderResult:
        logger.info("Placing market order",
                    symbol=symbol, side=side, quantity=quantity)
        raw    = self._call(
            lambda: self._exchange.create_market_order(symbol, side, quantity)
        )
        result = _parse_order(raw)
        logger.info("Market order placed",
                    order_id=result.order_id, status=result.status,
                    filled=result.filled, average=result.average)
        return result

    def place_limit_order(
        self, symbol: str, side: str, quantity: float, price: float,
    ) -> OrderResult:
        logger.info("Placing limit order",
                    symbol=symbol, side=side, quantity=quantity, price=price)
        raw    = self._call(
            lambda: self._exchange.create_limit_order(symbol, side, quantity, price)
        )
        result = _parse_order(raw)
        logger.info("Limit order placed",
                    order_id=result.order_id, status=result.status)
        return result

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        logger.info("Cancelling order", order_id=order_id, symbol=symbol)
        try:
            self._call(lambda: self._exchange.cancel_order(order_id, symbol))
            return True
        except ccxt.OrderNotFound:
            logger.warning("Order not found for cancellation", order_id=order_id)
            return False

    def fetch_order(self, order_id: str, symbol: str) -> OrderResult:
        raw = self._call(lambda: self._exchange.fetch_order(order_id, symbol))
        return _parse_order(raw)

    def fetch_open_orders(self, symbol: str) -> list[OrderResult]:
        raw_list = self._call(lambda: self._exchange.fetch_open_orders(symbol))
        return [_parse_order(o) for o in raw_list]

    def is_connected(self) -> bool:
        try:
            self._call(lambda: self._exchange.load_markets())
            return True
        except Exception:
            return False

    def close(self) -> None:
        if hasattr(self._exchange, "close"):
            try:
                self._exchange.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _call(self, fn):
        """
        Execute fn with exponential-backoff retry on transient CCXT errors.

        Transient errors sleep for base_delay * 2^attempt seconds.
        Rate-limit errors sleep for base_delay * 4^attempt seconds (longer).
        Any other exception bubbles up immediately without retrying.
        """
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                return fn()

            except ccxt.RateLimitExceeded as exc:
                last_exc = exc
                delay = self._base_delay * (4 ** attempt)
                logger.warning("Rate limit exceeded — backing off",
                               attempt=attempt, delay=delay)
                time.sleep(delay)

            except _TRANSIENT as exc:
                last_exc = exc
                delay = self._base_delay * (2 ** attempt)
                logger.warning("Transient error — retrying",
                               error=str(exc), attempt=attempt, delay=delay)
                time.sleep(delay)

            except Exception:
                raise  # permanent error — bubble immediately

        raise last_exc  # type: ignore[misc]
