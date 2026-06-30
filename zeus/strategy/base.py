"""Abstract Strategy interface — all concrete strategies must subclass this."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum

import pandas as pd


class SignalType(IntEnum):
    SHORT = -1
    NONE  =  0
    LONG  = +1


@dataclass(frozen=True)
class Signal:
    """Trading signal emitted by a strategy."""
    type: SignalType
    confidence: float   # 0.0 – 1.0
    reason: str         # human-readable rationale (for logging)
    bar_index: int      # which bar produced this signal


class Strategy(ABC):
    """
    Base class for all trading strategies.

    Concrete classes implement generate_signal(), which receives a slice of
    OHLCV data up to and including the current bar and returns a Signal.

    Keeping this interface minimal ensures that:
    - Strategies are interchangeable (paper / live / backtest use the same API)
    - Each strategy is independently testable with synthetic data
    """

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, bar_index: int) -> Signal:
        """
        Generate a signal at bar_index.

        Args:
            df:        Full OHLCV DataFrame (at least up to bar_index).
            bar_index: Index of the bar being evaluated.

        Returns:
            Signal with type LONG, SHORT, or NONE.
        """
        ...
