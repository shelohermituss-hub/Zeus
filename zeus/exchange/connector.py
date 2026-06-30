"""
Exchange connector interface and shared data structures.

All concrete connectors (live, paper) implement ExchangeConnector so the
execution layer is completely exchange-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OrderResult:
    """
    Normalised representation of an exchange order.

    Works for both live (from CCXT) and paper (simulated) orders.
    The ``raw`` field stores the original exchange response for debugging;
    it is excluded from equality comparison and hashing.
    """
    order_id:  str
    symbol:    str
    side:      str            # "buy" or "sell"
    quantity:  float          # requested amount (base currency)
    filled:    float          # amount actually filled
    price:     float | None   # limit price (None for market orders)
    average:   float | None   # average fill price (None when unfilled)
    status:    str            # "open" | "closed" | "cancelled" | "failed"
    timestamp: datetime
    raw: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def is_filled(self) -> bool:
        """True when the order is fully closed with a non-zero fill."""
        return self.status == "closed" and self.filled > 0

    @property
    def is_open(self) -> bool:
        """True when the order is still waiting in the book."""
        return self.status == "open"


# ──────────────────────────────────────────────────────────────────────────────
# Abstract interface
# ──────────────────────────────────────────────────────────────────────────────

class ExchangeConnector(ABC):
    """
    Abstract base class for all exchange connectors.

    Concrete implementations:
        LiveConnector  — wraps a CCXT exchange for real order execution.
        PaperConnector — simulates fills in-memory for paper trading.

    Both expose an identical interface so the rest of the system (orders,
    backtest, paper) never needs to know which connector is in use.
    """

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol:    str,
        timeframe: str,
        limit:     int = 500,
    ) -> pd.DataFrame:
        """
        Fetch the most recent OHLCV bars for symbol/timeframe.

        Returns a DataFrame with columns [open, high, low, close, volume],
        a DatetimeIndex (UTC), and oldest bar first.
        """
        ...

    @abstractmethod
    def fetch_balance(self) -> float:
        """Return the available quote-currency balance (e.g. USDT)."""
        ...

    @abstractmethod
    def place_market_order(
        self,
        symbol:   str,
        side:     str,      # "buy" or "sell"
        quantity: float,
    ) -> OrderResult:
        """Place a market order. Raises on immediate rejection."""
        ...

    @abstractmethod
    def place_limit_order(
        self,
        symbol:   str,
        side:     str,
        quantity: float,
        price:    float,
    ) -> OrderResult:
        """Place a limit order at the given price."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an open order.

        Returns True if the order was successfully cancelled,
        False if it was not found or already closed.
        """
        ...

    @abstractmethod
    def fetch_order(self, order_id: str, symbol: str) -> OrderResult:
        """Fetch the current state of a single order by ID."""
        ...

    @abstractmethod
    def fetch_open_orders(self, symbol: str) -> list[OrderResult]:
        """Return all currently open orders for symbol."""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Return True when the exchange is reachable and credentials are valid.

        Used by the risk manager's connectivity gate and the monitoring layer.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release resources (HTTP sessions, file handles, etc.)."""
        ...


# ──────────────────────────────────────────────────────────────────────────────
# Shared utility
# ──────────────────────────────────────────────────────────────────────────────

def ohlcv_to_df(raw: list[list]) -> pd.DataFrame:
    """
    Convert a raw CCXT OHLCV list to a DataFrame.

    CCXT format: [[timestamp_ms, open, high, low, close, volume], ...]
    Output: DataFrame[open, high, low, close, volume] with DatetimeIndex (UTC).
    Returns an empty DataFrame with correct columns when raw is empty.
    """
    if not raw:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.index.name = None
    return df[["open", "high", "low", "close", "volume"]].astype(float)
