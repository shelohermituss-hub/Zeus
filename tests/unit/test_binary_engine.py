"""Tests du moteur de backtest options binaires sur des séries synthétiques déterministes."""
import pandas as pd
import pytest

from zeus.backtest.binary_engine import (
    BinaryBacktestConfig, BinaryOutcome, run_binary_backtest,
)
from zeus.risk.binary_risk import BinaryRiskConfig, BinaryRiskManager
from zeus.strategy.base import Signal, SignalType


def _bars(closes: list[float], start: str = "2024-03-04 10:00") -> pd.DataFrame:
    """Bougies M1 où open = close précédent."""
    idx = pd.date_range(start, periods=len(closes), freq="1min")
    opens = [closes[0]] + closes[:-1]
    return pd.DataFrame({"open": opens, "high": closes, "low": closes, "close": closes}, index=idx)


def _run(df, signals, risk_kwargs=None, **cfg):
    risk = BinaryRiskManager(BinaryRiskConfig(**(risk_kwargs or {})), 1000.0)
    return run_binary_backtest(df, signals, BinaryBacktestConfig(**cfg), risk)


def _sig(i, kind=SignalType.LONG):
    return Signal(type=kind, confidence=1.0, reason="test", bar_index=i)


def test_call_win_pays_payout_on_next_bar_entry():
    df = _bars([1.0, 1.0, 1.1, 1.2, 1.3])
    # signal sur bougie 0 → entrée à l'open de la bougie 1 (1.0), expiry 2 min → close bougie 2 (1.1)
    res = _run(df, [_sig(0)], expiry_minutes=2)
    trade = res.trades[0]
    assert trade.entry_price == 1.0 and trade.exit_price == 1.1
    assert trade.outcome is BinaryOutcome.WIN
    assert trade.pnl == pytest.approx(10.0 * 0.92)
    assert res.final_balance == pytest.approx(1009.2)


def test_put_loses_when_price_rises():
    res = _run(_bars([1.0, 1.0, 1.1, 1.2]), [_sig(0, SignalType.SHORT)], expiry_minutes=2)
    assert res.trades[0].outcome is BinaryOutcome.LOSS
    assert res.final_balance == pytest.approx(990.0)


def test_tie_refunds_stake():
    res = _run(_bars([1.0, 1.0, 1.0, 1.0]), [_sig(0)], expiry_minutes=2)
    assert res.trades[0].outcome is BinaryOutcome.TIE and res.final_balance == 1000.0


def test_slippage_is_always_adverse():
    df = _bars([1.0, 1.0, 1.0005, 1.0])
    win = _run(df, [_sig(0)], expiry_minutes=2)
    lost = _run(df, [_sig(0)], expiry_minutes=2, entry_slippage=0.001)
    assert win.trades[0].outcome is BinaryOutcome.WIN
    assert lost.trades[0].entry_price == pytest.approx(1.001)
    assert lost.trades[0].outcome is BinaryOutcome.LOSS


def test_signal_without_enough_future_bars_is_skipped():
    res = _run(_bars([1.0, 1.0, 1.1]), [_sig(1)], expiry_minutes=5)
    assert res.trades == [] and res.skipped_gaps == 1


def test_missing_expiry_bar_skips_trade():
    df = _bars([1.0] * 10).drop(pd.Timestamp("2024-03-04 10:05"))
    res = _run(df, [_sig(0)], expiry_minutes=5)
    assert res.trades == [] and res.skipped_gaps == 1


def test_duplicated_timestamp_skips_trade():
    """Non-régression : heure répétée au changement d'heure → horodatages en double."""
    df = _bars([1.0] * 10)
    df = pd.concat([df, df.iloc[[3]]]).sort_index()
    res = _run(df, [_sig(0)], expiry_minutes=3)   # expiration sur 10:03, dupliquée
    assert res.trades == [] and res.skipped_gaps == 1


def test_missing_intermediate_bar_does_not_change_outcome():
    df = _bars([1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2]).drop(pd.Timestamp("2024-03-04 10:03"))
    res = _run(df, [_sig(0)], expiry_minutes=5)
    assert res.trades[0].outcome is BinaryOutcome.WIN


def test_overlapping_signal_rejected_by_open_trade_limit():
    df = _bars([1.0 + i * 0.01 for i in range(20)])
    res = _run(df, [_sig(0), _sig(2), _sig(10)], expiry_minutes=5)
    assert len(res.trades) == 2
    assert res.rejections == {"too_many_open_trades": 1}


def test_none_signals_are_ignored():
    res = _run(_bars([1.0] * 5), [_sig(0, SignalType.NONE)], expiry_minutes=2)
    assert res.trades == [] and res.skipped_gaps == 0


def test_summary_statistics():
    # alterne gain/perte : 1 CALL gagnant, 1 CALL perdant, 1 CALL perdant
    closes = [1.0, 1.0, 1.1, 1.1, 1.0, 1.0, 0.9, 0.9]
    res = _run(_bars(closes), [_sig(0), _sig(2), _sig(4)], expiry_minutes=2)
    s = res.summary()
    assert (s["wins"], s["losses"]) == (1, 2)
    assert s["max_consecutive_losses"] == 2
    assert s["breakeven_win_rate"] == pytest.approx(1 / 1.92, abs=1e-4)
    assert 0 < s["max_drawdown"] < 0.03
    assert res.win_rate_lower_bound < res.win_rate


@pytest.mark.parametrize("kwargs", [
    {"payout": 0.0}, {"payout": 1.5}, {"expiry_minutes": 0},
    {"latency_bars": 0}, {"entry_slippage": -1.0},
    {"expiry_minutes": 5, "bar_minutes": 2},
])
def test_invalid_config_is_refused(kwargs):
    with pytest.raises(ValueError):
        BinaryBacktestConfig(**kwargs)
