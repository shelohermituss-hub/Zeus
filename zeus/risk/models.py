"""
Data structures shared by the risk module.

Keeping models separate from logic allows tests to build fixtures without
importing the full RiskManager (which depends on config).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class OrderSide(str, Enum):
    BUY  = "buy"
    SELL = "sell"


class RejectionReason(str, Enum):
    KILL_SWITCH_ACTIVE      = "kill_switch_active"
    EXCHANGE_DISCONNECTED   = "exchange_disconnected"
    INSUFFICIENT_BALANCE    = "insufficient_balance"
    POSITION_SIZE_TOO_LARGE = "position_size_too_large"
    POSITION_SIZE_ZERO      = "position_size_zero"
    INVALID_STOP_LOSS       = "invalid_stop_loss"
    INVALID_TAKE_PROFIT     = "invalid_take_profit"
    DAILY_LOSS_LIMIT_HIT    = "daily_loss_limit_hit"
    MAX_DRAWDOWN_HIT        = "max_drawdown_hit"
    MAX_EXPOSURE_EXCEEDED   = "max_exposure_exceeded"
    TOO_MANY_OPEN_POSITIONS = "too_many_open_positions"


@dataclass(frozen=True)
class OrderRequest:
    """
    Everything the Risk Manager needs to know about a proposed order.
    Immutable — the strategy emits it, the risk manager validates it.
    """
    symbol: str
    side: OrderSide
    quantity: float          # in base currency (e.g. BTC)
    entry_price: float       # expected fill price
    stop_loss: float         # price level for stop-loss
    take_profit: float | None = None  # optional take-profit level
    signal_reason: str       = ""     # human-readable strategy rationale


@dataclass
class RiskState:
    """
    Mutable session-level state tracked by the Risk Manager.

    Updated after each fill; persisted in memory only (not to disk).
    All monetary values are in the quote currency (e.g. USDT).
    """
    # Balances
    initial_capital: float       = 0.0   # capital at session start
    available_balance: float     = 0.0   # free balance (paper or live)
    equity: float                = 0.0   # total equity including open P&L

    # Daily tracking (reset each calendar day)
    daily_realized_pnl: float    = 0.0
    daily_start_equity: float    = 0.0

    # Drawdown tracking
    peak_equity: float           = 0.0
    max_drawdown_pct: float      = 0.0   # worst drawdown seen this session

    # Position inventory
    open_positions: int          = 0
    total_exposure: float        = 0.0   # sum of open position values in quote currency

    # Kill switch
    kill_switch_active: bool     = False

    # Exchange connectivity
    exchange_connected: bool     = True

    def update_drawdown(self) -> None:
        """Recompute peak_equity and max_drawdown_pct from current equity."""
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        if self.peak_equity > 0:
            dd = (self.peak_equity - self.equity) / self.peak_equity
            if dd > self.max_drawdown_pct:
                self.max_drawdown_pct = dd


@dataclass(frozen=True)
class ValidationResult:
    """Result of a risk check. Immutable once created."""
    approved: bool
    rejection_reason: RejectionReason | None = None
    message: str = ""
    computed_quantity: float = 0.0   # quantity the risk manager allows (may differ from request)

    @classmethod
    def ok(cls, quantity: float) -> "ValidationResult":
        return cls(approved=True, computed_quantity=quantity)

    @classmethod
    def reject(cls, reason: RejectionReason, message: str) -> "ValidationResult":
        return cls(approved=False, rejection_reason=reason, message=message)
