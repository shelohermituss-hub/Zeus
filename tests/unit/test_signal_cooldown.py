"""Unit tests for SignalCooldown utility (zeus.strategy.utils)."""
from zeus.strategy.utils import SignalCooldown


def test_can_signal_initially():
    cd = SignalCooldown(cooldown_bars=5)
    assert cd.can_signal(0) is True
    assert cd.can_signal(10) is True


def test_cannot_signal_immediately_after_mark():
    cd = SignalCooldown(cooldown_bars=5)
    cd.mark(10)
    assert cd.can_signal(10) is False
    assert cd.can_signal(11) is False
    assert cd.can_signal(14) is False


def test_can_signal_exactly_at_cooldown_boundary():
    cd = SignalCooldown(cooldown_bars=5)
    cd.mark(10)
    assert cd.can_signal(15) is True  # 15 - 10 == 5 >= 5


def test_cannot_signal_one_before_boundary():
    cd = SignalCooldown(cooldown_bars=5)
    cd.mark(10)
    assert cd.can_signal(14) is False  # 14 - 10 == 4 < 5


def test_reset_allows_immediate_signal():
    cd = SignalCooldown(cooldown_bars=5)
    cd.mark(10)
    assert cd.can_signal(12) is False
    cd.reset()
    assert cd.can_signal(12) is True


def test_zero_cooldown_always_allows():
    cd = SignalCooldown(cooldown_bars=0)
    cd.mark(10)
    assert cd.can_signal(10) is True   # same bar still allowed
    assert cd.can_signal(11) is True   # next bar allowed


def test_mark_updates_last_bar():
    cd = SignalCooldown(cooldown_bars=3)
    cd.mark(5)
    assert cd.can_signal(7) is False  # 7-5=2 < 3
    cd.mark(7)
    assert cd.can_signal(9) is False  # 9-7=2 < 3
    assert cd.can_signal(10) is True  # 10-7=3 >= 3


def test_cooldown_bars_one():
    cd = SignalCooldown(cooldown_bars=1)
    cd.mark(10)
    assert cd.can_signal(10) is False
    assert cd.can_signal(11) is True
