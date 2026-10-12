"""Tests du risk manager options binaires : chaque garde-fou refuse bien le trade."""
from datetime import datetime

import pytest

from zeus.risk.binary_risk import BinaryRejection, BinaryRiskConfig, BinaryRiskManager

T0 = datetime(2024, 3, 4, 10, 0)


def _manager(balance: float = 1000.0, **overrides) -> BinaryRiskManager:
    return BinaryRiskManager(BinaryRiskConfig(**overrides), balance)


def _lose(rm: BinaryRiskManager, times: int = 1) -> None:
    for _ in range(times):
        stake = rm.evaluate(T0, broker_connected=True).stake
        rm.record_open(stake)
        rm.record_close(stake, -stake)


def test_approves_stake_as_fraction_of_balance():
    decision = _manager(1000.0, stake_fraction=0.01).evaluate(T0, broker_connected=True)
    assert decision.approved and decision.stake == 10.0


def test_stake_is_capped_by_max_stake():
    assert _manager(1_000_000.0, max_stake=50.0).evaluate(T0, True).stake == 50.0


def test_rejects_when_stake_below_broker_minimum():
    decision = _manager(50.0, stake_fraction=0.01, min_stake=1.0).evaluate(T0, True)
    assert decision.rejection is BinaryRejection.STAKE_BELOW_MINIMUM


@pytest.mark.parametrize("connected", [False, None])
def test_rejects_when_broker_not_confirmed_connected(connected):
    assert _manager().evaluate(T0, connected).rejection is BinaryRejection.BROKER_DISCONNECTED


def test_kill_switch_blocks_everything():
    rm = _manager()
    rm.kill_switch = True
    assert rm.evaluate(T0, True).rejection is BinaryRejection.KILL_SWITCH_ACTIVE


def test_consecutive_losses_block_until_next_day():
    rm = _manager(max_consecutive_losses=3, max_daily_loss_fraction=0.5)
    _lose(rm, 3)
    assert rm.evaluate(T0, True).rejection is BinaryRejection.CONSECUTIVE_LOSSES_HIT
    assert rm.evaluate(datetime(2024, 3, 5, 10, 0), True).approved


def test_win_resets_consecutive_losses():
    rm = _manager(max_consecutive_losses=2, max_daily_loss_fraction=0.5)
    _lose(rm)
    rm.record_open(10.0)
    rm.record_close(10.0, 9.2)
    _lose(rm)
    assert rm.evaluate(T0, True).approved


def test_daily_loss_limit():
    rm = _manager(1000.0, stake_fraction=0.05, max_daily_loss_fraction=0.05,
                  max_consecutive_losses=10, max_exposure_fraction=0.1)
    _lose(rm, 1)   # -50 → 950 = limite de 5 %
    assert rm.evaluate(T0, True).rejection is BinaryRejection.DAILY_LOSS_LIMIT_HIT


def test_max_drawdown_persists_across_days():
    rm = _manager(1000.0, stake_fraction=0.05, max_daily_loss_fraction=0.5,
                  max_drawdown_fraction=0.1, max_consecutive_losses=10,
                  max_exposure_fraction=0.1)
    _lose(rm, 3)   # 1000 → ~857 < 900
    assert rm.evaluate(datetime(2024, 3, 6, 10, 0), True).rejection is BinaryRejection.MAX_DRAWDOWN_HIT


def test_too_many_open_trades():
    rm = _manager(max_open_trades=1)
    rm.record_open(rm.evaluate(T0, True).stake)
    assert rm.evaluate(T0, True).rejection is BinaryRejection.TOO_MANY_OPEN_TRADES


def test_max_exposure():
    rm = _manager(1000.0, stake_fraction=0.02, max_open_trades=5, max_exposure_fraction=0.03)
    rm.record_open(rm.evaluate(T0, True).stake)
    assert rm.evaluate(T0, True).rejection is BinaryRejection.MAX_EXPOSURE_EXCEEDED


def test_tie_does_not_change_balance_or_streak():
    rm = _manager(max_consecutive_losses=2, max_daily_loss_fraction=0.5)
    _lose(rm)
    rm.record_open(10.0)
    rm.record_close(10.0, 0.0)
    assert rm.balance == pytest.approx(990.0)
    _lose(rm)
    assert rm.evaluate(T0, True).rejection is BinaryRejection.CONSECUTIVE_LOSSES_HIT


@pytest.mark.parametrize("balance", [0.0, -5.0, float("nan"), float("inf")])
def test_invalid_initial_balance_is_refused(balance):
    with pytest.raises(ValueError):
        BinaryRiskManager(BinaryRiskConfig(), balance)


@pytest.mark.parametrize("kwargs", [
    {"stake_fraction": 0.0}, {"stake_fraction": 0.2}, {"min_stake": 0.0},
    {"min_stake": 10.0, "max_stake": 5.0}, {"max_drawdown_fraction": 1.0},
    {"max_consecutive_losses": 0}, {"max_open_trades": 0},
])
def test_invalid_config_is_refused(kwargs):
    with pytest.raises(ValueError):
        BinaryRiskConfig(**kwargs)
