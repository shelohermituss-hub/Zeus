"""
Unit tests for zeus.risk.manager.

Each test covers ONE risk rule in isolation to make failures unambiguous.
State is reset via a fresh RiskManager per test (pytest fixtures handle this).
"""

import pytest

from zeus.risk.manager import RiskManager
from zeus.risk.models import (
    OrderRequest,
    OrderSide,
    RejectionReason,
    ValidationResult,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def rm() -> RiskManager:
    """A RiskManager initialised with 10 000 USDT capital and standard limits."""
    mgr = RiskManager(
        max_position_pct=0.02,
        stop_loss_pct=0.01,
        take_profit_pct=0.02,
        max_daily_loss_pct=0.05,
        max_drawdown_pct=0.15,
        max_exposure_pct=0.10,
        max_open_positions=3,
    )
    mgr.initialise_capital(10_000.0)
    return mgr


def _buy(quantity=0.01, entry=50_000.0, sl=49_500.0, tp=None) -> OrderRequest:
    """Helper — valid BUY request by default."""
    return OrderRequest(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=quantity,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
    )


def _sell(quantity=0.01, entry=50_000.0, sl=50_500.0, tp=None) -> OrderRequest:
    """Helper — valid SELL request by default."""
    return OrderRequest(
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        quantity=quantity,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────────────────────────────────────

class TestHappyPath:
    def test_valid_buy_approved(self, rm):
        result = rm.validate(_buy())
        assert result.approved

    def test_valid_sell_approved(self, rm):
        result = rm.validate(_sell())
        assert result.approved

    def test_approved_result_has_nonzero_quantity(self, rm):
        result = rm.validate(_buy(quantity=0.001))
        assert result.approved
        assert result.computed_quantity > 0

    def test_approved_quantity_capped_at_max_position(self, rm):
        """Requested quantity above the position limit is silently capped."""
        # equity=10000, max_position_pct=0.02 → max_value=200 → max_qty at 50000=0.004
        huge_qty = 1.0
        result = rm.validate(_buy(quantity=huge_qty))
        assert result.approved
        assert result.computed_quantity <= 0.004 + 1e-9


# ──────────────────────────────────────────────────────────────────────────────
# Check 1 — Kill switch
# ──────────────────────────────────────────────────────────────────────────────

class TestKillSwitch:
    def test_kill_switch_blocks_all_orders(self, rm):
        rm.set_kill_switch(True, "manual stop")
        result = rm.validate(_buy())
        assert not result.approved
        assert result.rejection_reason == RejectionReason.KILL_SWITCH_ACTIVE

    def test_kill_switch_can_be_reset(self, rm):
        rm.set_kill_switch(True)
        rm.set_kill_switch(False)
        result = rm.validate(_buy())
        assert result.approved

    def test_kill_switch_is_first_check(self, rm):
        """Even with zero balance, kill switch error must take precedence."""
        rm.state.available_balance = 0.0
        rm.set_kill_switch(True)
        result = rm.validate(_buy())
        assert result.rejection_reason == RejectionReason.KILL_SWITCH_ACTIVE


# ──────────────────────────────────────────────────────────────────────────────
# Check 2 — Exchange connectivity
# ──────────────────────────────────────────────────────────────────────────────

class TestExchangeConnectivity:
    def test_disconnected_exchange_blocks_order(self, rm):
        rm.set_exchange_connected(False)
        result = rm.validate(_buy())
        assert not result.approved
        assert result.rejection_reason == RejectionReason.EXCHANGE_DISCONNECTED

    def test_reconnected_exchange_allows_order(self, rm):
        rm.set_exchange_connected(False)
        rm.set_exchange_connected(True)
        result = rm.validate(_buy())
        assert result.approved


# ──────────────────────────────────────────────────────────────────────────────
# Check 3 — Available balance
# ──────────────────────────────────────────────────────────────────────────────

class TestBalance:
    def test_insufficient_balance_rejected(self, rm):
        rm.state.available_balance = 100.0  # can't afford 0.01 BTC @ 50k
        result = rm.validate(_buy(quantity=0.01, entry=50_000.0))
        assert not result.approved
        assert result.rejection_reason == RejectionReason.INSUFFICIENT_BALANCE

    def test_exact_balance_approved(self, rm):
        """Having exactly enough balance should pass the balance check."""
        rm.state.available_balance = 500.0  # 0.01 * 50_000 = 500
        result = rm.validate(_buy(quantity=0.01, entry=50_000.0))
        assert result.approved


# ──────────────────────────────────────────────────────────────────────────────
# Check 4 — Position size
# ──────────────────────────────────────────────────────────────────────────────

class TestPositionSize:
    def test_zero_quantity_rejected(self, rm):
        result = rm.validate(_buy(quantity=0.0))
        assert not result.approved
        assert result.rejection_reason == RejectionReason.POSITION_SIZE_ZERO

    def test_negative_quantity_rejected(self, rm):
        result = rm.validate(_buy(quantity=-1.0))
        assert not result.approved
        assert result.rejection_reason == RejectionReason.POSITION_SIZE_ZERO


# ──────────────────────────────────────────────────────────────────────────────
# Check 5 — Stop-loss validity
# ──────────────────────────────────────────────────────────────────────────────

class TestStopLoss:
    def test_buy_sl_above_entry_rejected(self, rm):
        result = rm.validate(_buy(entry=50_000.0, sl=51_000.0))
        assert not result.approved
        assert result.rejection_reason == RejectionReason.INVALID_STOP_LOSS

    def test_buy_sl_equal_entry_rejected(self, rm):
        result = rm.validate(_buy(entry=50_000.0, sl=50_000.0))
        assert not result.approved
        assert result.rejection_reason == RejectionReason.INVALID_STOP_LOSS

    def test_sell_sl_below_entry_rejected(self, rm):
        result = rm.validate(_sell(entry=50_000.0, sl=49_000.0))
        assert not result.approved
        assert result.rejection_reason == RejectionReason.INVALID_STOP_LOSS

    def test_buy_sl_too_wide_rejected(self, rm):
        """Stop wider than 5× the configured max should be rejected."""
        # max_stop_loss_pct=0.01, 5× = 0.05, so 10% stop must be rejected
        result = rm.validate(_buy(entry=50_000.0, sl=45_000.0))  # 10% below
        assert not result.approved
        assert result.rejection_reason == RejectionReason.INVALID_STOP_LOSS

    def test_valid_buy_sl_approved(self, rm):
        result = rm.validate(_buy(entry=50_000.0, sl=49_500.0))  # 1% below
        assert result.approved


# ──────────────────────────────────────────────────────────────────────────────
# Check 6 — Take-profit validity
# ──────────────────────────────────────────────────────────────────────────────

class TestTakeProfit:
    def test_buy_tp_below_entry_rejected(self, rm):
        result = rm.validate(_buy(entry=50_000.0, sl=49_500.0, tp=49_000.0))
        assert not result.approved
        assert result.rejection_reason == RejectionReason.INVALID_TAKE_PROFIT

    def test_buy_tp_equal_entry_rejected(self, rm):
        result = rm.validate(_buy(entry=50_000.0, sl=49_500.0, tp=50_000.0))
        assert not result.approved
        assert result.rejection_reason == RejectionReason.INVALID_TAKE_PROFIT

    def test_sell_tp_above_entry_rejected(self, rm):
        result = rm.validate(_sell(entry=50_000.0, sl=50_500.0, tp=51_000.0))
        assert not result.approved
        assert result.rejection_reason == RejectionReason.INVALID_TAKE_PROFIT

    def test_absent_take_profit_accepted(self, rm):
        """tp=None must not trigger a rejection."""
        result = rm.validate(_buy(entry=50_000.0, sl=49_500.0, tp=None))
        assert result.approved

    def test_valid_buy_tp_approved(self, rm):
        result = rm.validate(_buy(entry=50_000.0, sl=49_500.0, tp=51_000.0))
        assert result.approved


# ──────────────────────────────────────────────────────────────────────────────
# Check 7 — Daily loss limit
# ──────────────────────────────────────────────────────────────────────────────

class TestDailyLossLimit:
    def test_daily_loss_at_limit_rejected(self, rm):
        # max_daily_loss_pct=5 %, daily_start_equity=10 000 → limit=-500
        rm.state.daily_realized_pnl = -500.0
        result = rm.validate(_buy())
        assert not result.approved
        assert result.rejection_reason == RejectionReason.DAILY_LOSS_LIMIT_HIT

    def test_daily_loss_below_limit_approved(self, rm):
        rm.state.daily_realized_pnl = -499.0
        result = rm.validate(_buy())
        assert result.approved

    def test_reset_daily_pnl_allows_trading(self, rm):
        rm.state.daily_realized_pnl = -600.0
        rm.reset_daily_pnl()
        result = rm.validate(_buy())
        assert result.approved


# ──────────────────────────────────────────────────────────────────────────────
# Check 8 — Max global drawdown
# ──────────────────────────────────────────────────────────────────────────────

class TestMaxDrawdown:
    def test_drawdown_at_limit_rejected(self, rm):
        rm.state.max_drawdown_pct = 0.15  # at the configured 15% limit
        result = rm.validate(_buy())
        assert not result.approved
        assert result.rejection_reason == RejectionReason.MAX_DRAWDOWN_HIT

    def test_drawdown_below_limit_approved(self, rm):
        rm.state.max_drawdown_pct = 0.14
        result = rm.validate(_buy())
        assert result.approved

    def test_update_on_fill_tracks_drawdown(self, rm):
        """A loss fill must update the drawdown tracking."""
        rm.update_on_fill(realized_pnl=-1_500.0, position_value=0.0, opened=False)
        # equity dropped from 10_000 to 8_500 → drawdown = 15%
        assert rm.state.max_drawdown_pct == pytest.approx(0.15)


# ──────────────────────────────────────────────────────────────────────────────
# Check 9 — Maximum exposure
# ──────────────────────────────────────────────────────────────────────────────

class TestMaxExposure:
    def test_new_order_exceeds_exposure_limit_rejected(self, rm):
        # max_exposure_pct=10%, equity=10000 → max_exposure=1000 USDT
        # position cap: max_qty = 10000*0.02/50000 = 0.004 BTC → effective_cost = 200 USDT
        # to exceed 1000 USDT total, pre-existing exposure must be > 800 USDT
        rm.state.total_exposure = 850.0   # 850 + 200 = 1050 USDT → 10.5% > 10%
        result = rm.validate(_buy(quantity=0.01, entry=50_000.0))
        assert not result.approved
        assert result.rejection_reason == RejectionReason.MAX_EXPOSURE_EXCEEDED

    def test_exposure_within_limit_approved(self, rm):
        rm.state.total_exposure = 0.0
        result = rm.validate(_buy(quantity=0.01, entry=50_000.0))
        assert result.approved


# ──────────────────────────────────────────────────────────────────────────────
# Check 10 — Max open positions
# ──────────────────────────────────────────────────────────────────────────────

class TestMaxOpenPositions:
    def test_at_max_positions_rejected(self, rm):
        rm.state.open_positions = 3  # at the limit
        result = rm.validate(_buy())
        assert not result.approved
        assert result.rejection_reason == RejectionReason.TOO_MANY_OPEN_POSITIONS

    def test_below_max_positions_approved(self, rm):
        rm.state.open_positions = 2
        result = rm.validate(_buy())
        assert result.approved


# ──────────────────────────────────────────────────────────────────────────────
# State management
# ──────────────────────────────────────────────────────────────────────────────

class TestStateManagement:
    def test_update_on_fill_open_increments_positions(self, rm):
        rm.update_on_fill(realized_pnl=0.0, position_value=500.0, opened=True)
        assert rm.state.open_positions == 1
        assert rm.state.total_exposure == pytest.approx(500.0)

    def test_update_on_fill_close_decrements_positions(self, rm):
        rm.update_on_fill(realized_pnl=0.0, position_value=500.0, opened=True)
        rm.update_on_fill(realized_pnl=50.0, position_value=500.0, opened=False)
        assert rm.state.open_positions == 0
        assert rm.state.total_exposure == pytest.approx(0.0)

    def test_open_positions_never_negative(self, rm):
        """Closing more than open positions must not produce negative count."""
        rm.update_on_fill(realized_pnl=0.0, position_value=0.0, opened=False)
        assert rm.state.open_positions == 0

    def test_equity_tracks_pnl(self, rm):
        rm.update_on_fill(realized_pnl=100.0, position_value=0.0, opened=False)
        assert rm.state.equity == pytest.approx(10_100.0)

    def test_peak_equity_not_decreased_on_gain(self, rm):
        rm.update_on_fill(realized_pnl=500.0, position_value=0.0, opened=False)
        assert rm.state.peak_equity == pytest.approx(10_500.0)
