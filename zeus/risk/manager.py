"""
Risk Manager — the single gatekeeper between signal generation and order execution.

Every proposed order MUST pass through RiskManager.validate() before being sent.
A single failing check rejects the entire order (fail-closed principle).

Checks executed in priority order:
    1. Kill switch
    2. Exchange connectivity
    3. Available balance
    4. Position size (absolute and % of capital)
    5. Stop-loss validity
    6. Take-profit validity (when provided)
    7. Daily loss limit
    8. Maximum global drawdown
    9. Total exposure
    10. Maximum open positions

All monetary values are in the quote currency (e.g. USDT).
"""
from __future__ import annotations

import math

from zeus.risk.models import (
    OrderRequest,
    OrderSide,
    RejectionReason,
    RiskState,
    ValidationResult,
)
from zeus.utils.logger import logger


class RiskManager:
    """
    Stateful risk manager. One instance per trading session.

    Parameters come from the application config; state is updated externally
    via update_on_fill() and set_kill_switch() after each execution event.
    """

    def __init__(
        self,
        max_position_pct: float = 0.02,
        stop_loss_pct: float    = 0.01,
        take_profit_pct: float  = 0.02,
        max_daily_loss_pct: float = 0.05,
        max_drawdown_pct: float   = 0.15,
        max_exposure_pct: float   = 0.10,
        max_open_positions: int   = 3,
    ) -> None:
        """
        Args:
            max_position_pct:   Max capital allocated per trade (0.02 = 2 %).
            stop_loss_pct:      Max distance from entry to stop-loss (0.01 = 1 %).
            take_profit_pct:    Min distance from entry to take-profit (0.02 = 2 %).
            max_daily_loss_pct: Halt if today's loss exceeds this fraction of daily_start_equity.
            max_drawdown_pct:   Halt if peak→current drawdown exceeds this fraction.
            max_exposure_pct:   Max total open exposure as fraction of equity.
            max_open_positions: Hard cap on concurrent positions.
        """
        self._max_position_pct    = max_position_pct
        self._stop_loss_pct       = stop_loss_pct
        self._take_profit_pct     = take_profit_pct
        self._max_daily_loss_pct  = max_daily_loss_pct
        self._max_drawdown_pct    = max_drawdown_pct
        self._max_exposure_pct    = max_exposure_pct
        self._max_open_positions  = max_open_positions

        self.state = RiskState()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def validate(self, request: OrderRequest) -> ValidationResult:
        """
        Validate an order request against all risk rules.

        Returns ValidationResult.ok(quantity) if all checks pass,
        ValidationResult.reject() with a reason on the first failure.

        Execution order is intentional:
            1-2: gate checks (kill switch, connectivity) — fastest rejections
            3-4: compute effective quantity, reject if zero
            5:   balance check against the EFFECTIVE (capped) quantity
            6-7: price-level checks (SL, TP)
            8-10: portfolio-level checks (daily loss, drawdown, exposure, positions)
        """
        # 1. Kill switch
        r = self._check_kill_switch(request)
        if not r.approved:
            return self._log_reject(request, r)

        # 2. Exchange connectivity
        r = self._check_exchange_connected(request)
        if not r.approved:
            return self._log_reject(request, r)

        # 3-4. Compute effective quantity (cap to position limit)
        if request.quantity <= 0:
            return self._log_reject(
                request,
                ValidationResult.reject(
                    RejectionReason.POSITION_SIZE_ZERO,
                    f"quantity must be > 0, got {request.quantity}",
                ),
            )
        max_qty = self._max_position_size(request.entry_price)
        if max_qty <= 0:
            return self._log_reject(
                request,
                ValidationResult.reject(
                    RejectionReason.POSITION_SIZE_TOO_LARGE,
                    f"computed max position is {max_qty} — equity too low",
                ),
            )
        effective_qty = min(request.quantity, max_qty)

        # 5. Balance check against the effective (possibly reduced) quantity
        required = effective_qty * request.entry_price
        if self.state.available_balance < required:
            return self._log_reject(
                request,
                ValidationResult.reject(
                    RejectionReason.INSUFFICIENT_BALANCE,
                    f"need {required:.2f}, have {self.state.available_balance:.2f}",
                ),
            )

        # 6-10. Remaining checks (pass request through unchanged)
        for check in (
            self._check_stop_loss,
            self._check_take_profit,
            self._check_daily_loss,
            self._check_max_drawdown,
            self._check_open_positions,
        ):
            r = check(request)
            if not r.approved:
                return self._log_reject(request, r)

        # Exposure check uses effective quantity
        new_exposure = self.state.total_exposure + effective_qty * request.entry_price
        if self.state.equity > 0 and new_exposure / self.state.equity > self._max_exposure_pct:
            return self._log_reject(
                request,
                ValidationResult.reject(
                    RejectionReason.MAX_EXPOSURE_EXCEEDED,
                    f"new exposure {new_exposure / self.state.equity:.2%} would exceed "
                    f"limit {self._max_exposure_pct:.2%}",
                ),
            )

        logger.info(
            "Order approved",
            symbol=request.symbol,
            side=request.side.value,
            quantity=effective_qty,
            entry=request.entry_price,
            stop_loss=request.stop_loss,
        )
        return ValidationResult.ok(quantity=effective_qty)

    def _log_reject(self, request: OrderRequest, result: ValidationResult) -> ValidationResult:
        logger.warning(
            "Order rejected",
            symbol=request.symbol,
            side=request.side.value,
            reason=result.rejection_reason.value if result.rejection_reason else "unknown",
            message=result.message,
        )
        return result

    def set_kill_switch(self, active: bool, reason: str = "") -> None:
        """Activate or deactivate the kill switch."""
        self.state.kill_switch_active = active
        level = logger.error if active else logger.info
        level("Kill switch toggled", active=active, reason=reason)

    def set_exchange_connected(self, connected: bool) -> None:
        """Update exchange connectivity status."""
        self.state.exchange_connected = connected
        if not connected:
            logger.warning("Exchange connection lost — orders will be rejected")

    def initialise_capital(self, balance: float) -> None:
        """Call once at session start with the actual available balance."""
        self.state.available_balance  = balance
        self.state.equity             = balance
        self.state.initial_capital    = balance
        self.state.daily_start_equity = balance
        self.state.peak_equity        = balance

    def update_on_fill(
        self,
        realized_pnl: float,
        position_value: float,
        opened: bool,
    ) -> None:
        """
        Update risk state after a fill is confirmed.

        Args:
            realized_pnl:    P&L of the closed trade (0 when opening).
            position_value:  Notional value of the newly opened position (0 when closing).
            opened:          True when opening a position, False when closing.
        """
        self.state.daily_realized_pnl += realized_pnl
        self.state.equity             += realized_pnl

        if opened:
            self.state.open_positions  += 1
            self.state.total_exposure  += position_value
            self.state.available_balance -= position_value
        else:
            self.state.open_positions  = max(0, self.state.open_positions - 1)
            self.state.total_exposure  = max(0.0, self.state.total_exposure - position_value)
            self.state.available_balance += position_value + realized_pnl

        self.state.update_drawdown()

    def reset_daily_pnl(self) -> None:
        """Reset daily counters. Call at the start of each trading day."""
        self.state.daily_realized_pnl = 0.0
        self.state.daily_start_equity = self.state.equity
        logger.info("Daily P&L reset", equity=self.state.equity)

    # ------------------------------------------------------------------ #
    # Individual checks (private)                                          #
    # ------------------------------------------------------------------ #

    def _check_kill_switch(self, _: OrderRequest) -> ValidationResult:
        if self.state.kill_switch_active:
            return ValidationResult.reject(
                RejectionReason.KILL_SWITCH_ACTIVE,
                "kill switch is active — all orders blocked",
            )
        return ValidationResult.ok(0)

    def _check_exchange_connected(self, _: OrderRequest) -> ValidationResult:
        if not self.state.exchange_connected:
            return ValidationResult.reject(
                RejectionReason.EXCHANGE_DISCONNECTED,
                "exchange is not connected",
            )
        return ValidationResult.ok(0)

    def _check_stop_loss(self, request: OrderRequest) -> ValidationResult:
        entry = request.entry_price
        sl    = request.stop_loss

        if request.side == OrderSide.BUY:
            if sl >= entry:
                return ValidationResult.reject(
                    RejectionReason.INVALID_STOP_LOSS,
                    f"BUY stop-loss {sl} must be below entry {entry}",
                )
            sl_dist_pct = (entry - sl) / entry
        else:
            if sl <= entry:
                return ValidationResult.reject(
                    RejectionReason.INVALID_STOP_LOSS,
                    f"SELL stop-loss {sl} must be above entry {entry}",
                )
            sl_dist_pct = (sl - entry) / entry

        if sl_dist_pct > self._stop_loss_pct * 5:
            # Stop wider than 5× the configured max — almost certainly a mistake
            return ValidationResult.reject(
                RejectionReason.INVALID_STOP_LOSS,
                f"stop-loss distance {sl_dist_pct:.2%} exceeds 5× max "
                f"({self._stop_loss_pct * 5:.2%})",
            )
        return ValidationResult.ok(0)

    def _check_take_profit(self, request: OrderRequest) -> ValidationResult:
        if request.take_profit is None:
            return ValidationResult.ok(0)

        entry = request.entry_price
        tp    = request.take_profit

        if request.side == OrderSide.BUY:
            if tp <= entry:
                return ValidationResult.reject(
                    RejectionReason.INVALID_TAKE_PROFIT,
                    f"BUY take-profit {tp} must be above entry {entry}",
                )
        else:
            if tp >= entry:
                return ValidationResult.reject(
                    RejectionReason.INVALID_TAKE_PROFIT,
                    f"SELL take-profit {tp} must be below entry {entry}",
                )
        return ValidationResult.ok(0)

    def _check_daily_loss(self, _: OrderRequest) -> ValidationResult:
        if self.state.daily_start_equity <= 0:
            return ValidationResult.ok(0)

        loss_pct = -self.state.daily_realized_pnl / self.state.daily_start_equity
        if loss_pct >= self._max_daily_loss_pct:
            return ValidationResult.reject(
                RejectionReason.DAILY_LOSS_LIMIT_HIT,
                f"daily loss {loss_pct:.2%} >= limit {self._max_daily_loss_pct:.2%}",
            )
        return ValidationResult.ok(0)

    def _check_max_drawdown(self, _: OrderRequest) -> ValidationResult:
        if self.state.max_drawdown_pct >= self._max_drawdown_pct:
            return ValidationResult.reject(
                RejectionReason.MAX_DRAWDOWN_HIT,
                f"drawdown {self.state.max_drawdown_pct:.2%} >= limit {self._max_drawdown_pct:.2%}",
            )
        return ValidationResult.ok(0)

    def _check_open_positions(self, _: OrderRequest) -> ValidationResult:
        if self.state.open_positions >= self._max_open_positions:
            return ValidationResult.reject(
                RejectionReason.TOO_MANY_OPEN_POSITIONS,
                f"open positions {self.state.open_positions} >= limit {self._max_open_positions}",
            )
        return ValidationResult.ok(0)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _max_position_size(self, price: float) -> float:
        """Return the maximum allowed quantity in base units."""
        if price <= 0 or self.state.equity <= 0:
            return 0.0
        max_value = self.state.equity * self._max_position_pct
        return max_value / price
