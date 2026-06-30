"""
Orders layer data structures.

A Trade represents the full lifecycle of one position: from entry signal
through exit (SL, TP, or manual close).  Trades are mutable — their status,
exit details and PnL are written in-place as events occur.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from zeus.exchange.connector import OrderResult


class TradeStatus(str, Enum):
    OPEN       = "open"        # entry filled, position live
    CLOSED_SL  = "closed_sl"  # stopped out at stop-loss
    CLOSED_TP  = "closed_tp"  # exited at take-profit
    CLOSED_MAN = "closed_man" # manually closed
    REJECTED   = "rejected"   # risk manager blocked before entry
    FAILED     = "failed"     # exchange error during entry


@dataclass
class Trade:
    """
    Full lifecycle record for one position.

    Fields are mutated in-place as the trade progresses (opened → closed).
    The entry_order is always present; exit_order is None until close.
    """
    trade_id:      str
    symbol:        str
    side:          str              # "buy" (long) or "sell" (short)
    quantity:      float            # filled entry quantity
    entry_price:   float            # actual fill price
    sl_price:      float            # stop-loss trigger price
    tp_price:      float | None     # take-profit trigger price; None = no TP
    entry_order:   OrderResult
    signal_reason: str = ""         # Signal.reason for the audit trail

    # Mutable lifecycle fields
    status:        TradeStatus           = TradeStatus.OPEN
    opened_at:     datetime              = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    exit_order:    OrderResult | None    = None
    closed_at:     datetime | None       = None
    realized_pnl:  float                 = 0.0

    @property
    def is_open(self) -> bool:
        return self.status == TradeStatus.OPEN

    @property
    def is_long(self) -> bool:
        return self.side == "buy"

    def unrealized_pnl(self, current_price: float) -> float:
        """P&L if the trade were closed at current_price."""
        if self.is_long:
            return (current_price - self.entry_price) * self.quantity
        return (self.entry_price - current_price) * self.quantity
