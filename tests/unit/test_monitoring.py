"""
Tests for zeus/monitoring/metrics.py and zeus/monitoring/alerts.py.

No exchange or network calls.  All inputs are synthetic.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from zeus.monitoring.alerts import Alert, AlertConfig, AlertManager
from zeus.monitoring.metrics import MetricsSnapshot, SessionMetrics
from zeus.orders.models import Trade, TradeStatus
from zeus.exchange.connector import OrderResult
from zeus.risk.models import RiskState


# ======================================================================
# Helpers
# ======================================================================

def _order() -> OrderResult:
    return OrderResult(
        order_id="o1", symbol="BTC/USDT", side="buy",
        quantity=0.1, filled=0.1, price=100.0, average=100.0,
        status="closed", timestamp=datetime.now(tz=timezone.utc),
    )


def _trade(
    pnl: float,
    status: TradeStatus = TradeStatus.CLOSED_TP,
    side: str = "buy",
    quantity: float = 1.0,
    entry: float = 100.0,
) -> Trade:
    t = Trade(
        trade_id="t1",
        symbol="BTC/USDT",
        side=side,
        quantity=quantity,
        entry_price=entry,
        sl_price=entry * 0.99,
        tp_price=entry * 1.02,
        entry_order=_order(),
        signal_reason="test",
    )
    t.status = status
    t.realized_pnl = pnl
    return t


def _snapshot(
    drawdown_pct: float = 0.0,
    max_drawdown_pct: float = 0.0,
    error_rate: float = 0.0,
    avg_latency_ms: float = 100.0,
    p99_latency_ms: float = 200.0,
    win_rate: float = 0.5,
    n_trades: int = 20,
    n_winning: int = 10,
    n_losing: int = 10,
    realized_pnl: float = 0.0,
    unrealized_pnl: float = 0.0,
    current_equity: float = 10_000.0,
    initial_balance: float = 10_000.0,
    peak_equity: float = 10_000.0,
    total_ticks: int = 100,
    api_errors: int = 0,
) -> MetricsSnapshot:
    return MetricsSnapshot(
        timestamp=datetime.now(tz=timezone.utc),
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        total_pnl=realized_pnl + unrealized_pnl,
        n_trades=n_trades,
        n_winning=n_winning,
        n_losing=n_losing,
        win_rate=win_rate,
        avg_win=50.0,
        avg_loss=-50.0,
        profit_factor=1.0,
        initial_balance=initial_balance,
        current_equity=current_equity,
        peak_equity=peak_equity,
        drawdown_pct=drawdown_pct,
        max_drawdown_pct=max_drawdown_pct,
        total_return_pct=(current_equity - initial_balance) / initial_balance,
        total_ticks=total_ticks,
        api_errors=api_errors,
        error_rate=error_rate,
        avg_latency_ms=avg_latency_ms,
        p99_latency_ms=p99_latency_ms,
    )


def _risk_state(
    kill_switch: bool = False,
    daily_pnl: float = 0.0,
    daily_start_equity: float = 10_000.0,
) -> RiskState:
    s = RiskState()
    s.kill_switch_active    = kill_switch
    s.daily_realized_pnl   = daily_pnl
    s.daily_start_equity   = daily_start_equity
    return s


# ======================================================================
# MetricsSnapshot
# ======================================================================

class TestMetricsSnapshotFrozen:
    def test_is_frozen(self):
        s = _snapshot()
        with pytest.raises((AttributeError, TypeError)):
            s.n_trades = 99  # type: ignore[misc]

    def test_to_dict_has_expected_keys(self):
        s = _snapshot()
        d = s.to_dict()
        for key in (
            "timestamp", "realized_pnl", "n_trades", "win_rate",
            "drawdown_pct", "max_drawdown_pct", "total_return_pct",
            "api_errors", "avg_latency_ms",
        ):
            assert key in d

    def test_to_dict_win_rate_is_percentage(self):
        s = _snapshot(win_rate=0.75)
        assert s.to_dict()["win_rate"] == pytest.approx(75.0)

    def test_to_dict_profit_factor_inf_as_string(self):
        s = MetricsSnapshot(
            timestamp=datetime.now(tz=timezone.utc),
            realized_pnl=100.0, unrealized_pnl=0.0, total_pnl=100.0,
            n_trades=1, n_winning=1, n_losing=0, win_rate=1.0,
            avg_win=100.0, avg_loss=0.0, profit_factor=float("inf"),
            initial_balance=10_000.0, current_equity=10_100.0,
            peak_equity=10_100.0, drawdown_pct=0.0, max_drawdown_pct=0.0,
            total_return_pct=0.01, total_ticks=10, api_errors=0,
            error_rate=0.0, avg_latency_ms=50.0, p99_latency_ms=100.0,
        )
        assert s.to_dict()["profit_factor"] == "inf"


# ======================================================================
# SessionMetrics — initialisation
# ======================================================================

class TestSessionMetricsInit:
    def test_initial_snapshot_no_trades(self):
        m = SessionMetrics(10_000.0)
        s = m.snapshot()
        assert s.n_trades == 0

    def test_initial_win_rate_zero(self):
        m = SessionMetrics(10_000.0)
        assert m.snapshot().win_rate == 0.0

    def test_initial_equity_equals_balance(self):
        m = SessionMetrics(10_000.0)
        assert m.snapshot().current_equity == pytest.approx(10_000.0)

    def test_initial_drawdown_zero(self):
        m = SessionMetrics(10_000.0)
        assert m.snapshot().drawdown_pct == pytest.approx(0.0)

    def test_initial_ticks_zero(self):
        m = SessionMetrics(10_000.0)
        assert m.snapshot().total_ticks == 0

    def test_initial_api_errors_zero(self):
        m = SessionMetrics(10_000.0)
        assert m.snapshot().api_errors == 0

    def test_initial_latency_zero(self):
        m = SessionMetrics(10_000.0)
        s = m.snapshot()
        assert s.avg_latency_ms == 0.0
        assert s.p99_latency_ms == 0.0


# ======================================================================
# record_trade
# ======================================================================

class TestRecordTrade:
    def test_closed_trade_counted(self):
        m = SessionMetrics(10_000.0)
        m.record_trade(_trade(pnl=100.0))
        assert m.snapshot().n_trades == 1

    def test_open_trade_ignored(self):
        m = SessionMetrics(10_000.0)
        t = _trade(pnl=0.0, status=TradeStatus.OPEN)
        m.record_trade(t)
        assert m.snapshot().n_trades == 0

    def test_rejected_trade_ignored(self):
        m = SessionMetrics(10_000.0)
        m.record_trade(_trade(pnl=0.0, status=TradeStatus.REJECTED))
        assert m.snapshot().n_trades == 0

    def test_failed_trade_ignored(self):
        m = SessionMetrics(10_000.0)
        m.record_trade(_trade(pnl=0.0, status=TradeStatus.FAILED))
        assert m.snapshot().n_trades == 0

    def test_win_counted(self):
        m = SessionMetrics(10_000.0)
        m.record_trade(_trade(pnl=100.0))
        assert m.snapshot().n_winning == 1

    def test_loss_counted(self):
        m = SessionMetrics(10_000.0)
        m.record_trade(_trade(pnl=-50.0, status=TradeStatus.CLOSED_SL))
        assert m.snapshot().n_losing == 1

    def test_realized_pnl_summed(self):
        m = SessionMetrics(10_000.0)
        m.record_trade(_trade(pnl=100.0))
        m.record_trade(_trade(pnl=-30.0, status=TradeStatus.CLOSED_SL))
        assert m.snapshot().realized_pnl == pytest.approx(70.0)

    def test_win_rate_computed(self):
        m = SessionMetrics(10_000.0)
        m.record_trade(_trade(pnl=100.0))
        m.record_trade(_trade(pnl=200.0))
        m.record_trade(_trade(pnl=-50.0, status=TradeStatus.CLOSED_SL))
        assert m.snapshot().win_rate == pytest.approx(2 / 3)

    def test_avg_win(self):
        m = SessionMetrics(10_000.0)
        m.record_trade(_trade(pnl=100.0))
        m.record_trade(_trade(pnl=200.0))
        assert m.snapshot().avg_win == pytest.approx(150.0)

    def test_avg_loss(self):
        m = SessionMetrics(10_000.0)
        m.record_trade(_trade(pnl=-50.0, status=TradeStatus.CLOSED_SL))
        m.record_trade(_trade(pnl=-150.0, status=TradeStatus.CLOSED_SL))
        assert m.snapshot().avg_loss == pytest.approx(-100.0)

    def test_profit_factor(self):
        m = SessionMetrics(10_000.0)
        m.record_trade(_trade(pnl=200.0))
        m.record_trade(_trade(pnl=-100.0, status=TradeStatus.CLOSED_SL))
        assert m.snapshot().profit_factor == pytest.approx(2.0)

    def test_profit_factor_infinite_when_no_losses(self):
        m = SessionMetrics(10_000.0)
        m.record_trade(_trade(pnl=100.0))
        assert m.snapshot().profit_factor == float("inf")

    def test_profit_factor_zero_when_no_wins(self):
        m = SessionMetrics(10_000.0)
        m.record_trade(_trade(pnl=-100.0, status=TradeStatus.CLOSED_SL))
        assert m.snapshot().profit_factor == 0.0

    def test_multiple_trade_statuses_all_counted(self):
        m = SessionMetrics(10_000.0)
        m.record_trade(_trade(pnl=100.0, status=TradeStatus.CLOSED_TP))
        m.record_trade(_trade(pnl=-50.0, status=TradeStatus.CLOSED_SL))
        m.record_trade(_trade(pnl=20.0,  status=TradeStatus.CLOSED_MAN))
        assert m.snapshot().n_trades == 3


# ======================================================================
# record_tick
# ======================================================================

class TestRecordTick:
    def test_tick_count_increments(self):
        m = SessionMetrics(10_000.0)
        m.record_tick(100.0, 10_000.0)
        m.record_tick(120.0, 10_000.0)
        assert m.snapshot().total_ticks == 2

    def test_avg_latency(self):
        m = SessionMetrics(10_000.0)
        m.record_tick(100.0, 10_000.0)
        m.record_tick(200.0, 10_000.0)
        assert m.snapshot().avg_latency_ms == pytest.approx(150.0)

    def test_p99_latency_single_value(self):
        m = SessionMetrics(10_000.0)
        m.record_tick(500.0, 10_000.0)
        assert m.snapshot().p99_latency_ms == pytest.approx(500.0)

    def test_p99_latency_many_values(self):
        m = SessionMetrics(10_000.0)
        for i in range(100):
            m.record_tick(float(i + 1), 10_000.0)
        # p99 of 1..100 at index int(0.99*100)=99 → sorted list index 99 = 100
        assert m.snapshot().p99_latency_ms == pytest.approx(100.0)

    def test_drawdown_computed_from_equity(self):
        m = SessionMetrics(10_000.0)
        m.record_tick(50.0, 9_000.0)   # equity drops 10% from peak 10_000
        s = m.snapshot()
        assert s.max_drawdown_pct == pytest.approx(0.10)

    def test_rising_equity_zero_drawdown(self):
        m = SessionMetrics(10_000.0)
        m.record_tick(50.0, 11_000.0)
        m.record_tick(50.0, 12_000.0)
        assert m.snapshot().max_drawdown_pct == pytest.approx(0.0)

    def test_peak_tracked_across_ticks(self):
        m = SessionMetrics(10_000.0)
        m.record_tick(50.0, 11_000.0)   # peak = 11_000
        m.record_tick(50.0,  9_000.0)   # dd = 2_000/11_000
        s = m.snapshot()
        assert s.peak_equity == pytest.approx(11_000.0)
        assert s.max_drawdown_pct == pytest.approx(2_000.0 / 11_000.0)

    def test_last_equity_updated(self):
        m = SessionMetrics(10_000.0)
        m.record_tick(50.0, 10_500.0)
        assert m.snapshot().current_equity == pytest.approx(10_500.0)

    def test_total_return_pct(self):
        m = SessionMetrics(10_000.0)
        m.record_tick(50.0, 11_000.0)
        assert m.snapshot().total_return_pct == pytest.approx(0.10)


# ======================================================================
# record_api_error
# ======================================================================

class TestRecordApiError:
    def test_error_count_increments(self):
        m = SessionMetrics(10_000.0)
        m.record_api_error()
        m.record_api_error()
        assert m.snapshot().api_errors == 2

    def test_error_rate_zero_when_no_ticks(self):
        m = SessionMetrics(10_000.0)
        m.record_api_error()
        assert m.snapshot().error_rate == 0.0   # no ticks → rate undefined → 0

    def test_error_rate_computed(self):
        m = SessionMetrics(10_000.0)
        for _ in range(10):
            m.record_tick(50.0, 10_000.0)
        m.record_api_error()
        assert m.snapshot().error_rate == pytest.approx(0.1)


# ======================================================================
# snapshot — unrealized_pnl parameter
# ======================================================================

class TestSnapshotUnrealizedPnl:
    def test_unrealized_added_to_equity(self):
        m = SessionMetrics(10_000.0)
        s = m.snapshot(unrealized_pnl=500.0)
        assert s.current_equity == pytest.approx(10_500.0)

    def test_total_pnl_includes_unrealized(self):
        m = SessionMetrics(10_000.0)
        m.record_trade(_trade(pnl=100.0))
        s = m.snapshot(unrealized_pnl=50.0)
        assert s.total_pnl == pytest.approx(150.0)

    def test_drawdown_reflects_unrealized_drop(self):
        m = SessionMetrics(10_000.0)
        m.record_tick(50.0, 10_000.0)
        # Unrealised loss of 500 → equity 9_500 → dd = 500/10_000 = 5%
        s = m.snapshot(unrealized_pnl=-500.0)
        assert s.drawdown_pct == pytest.approx(0.05)


# ======================================================================
# AlertManager — no alerts
# ======================================================================

class TestAlertManagerClean:
    def test_all_clear_returns_empty(self):
        am = AlertManager()
        alerts = am.check(_snapshot())
        assert alerts == []

    def test_all_clear_with_risk_state(self):
        am = AlertManager()
        alerts = am.check(_snapshot(), _risk_state())
        assert alerts == []


# ======================================================================
# AlertManager — drawdown
# ======================================================================

class TestAlertManagerDrawdown:
    def test_no_alert_below_warn(self):
        am = AlertManager(AlertConfig(drawdown_warn_pct=0.05))
        alerts = am.check(_snapshot(drawdown_pct=0.04))
        codes = [a.code for a in alerts]
        assert "DRAWDOWN_HIGH" not in codes
        assert "DRAWDOWN_CRITICAL" not in codes

    def test_warning_at_warn_threshold(self):
        am = AlertManager(AlertConfig(drawdown_warn_pct=0.05, drawdown_error_pct=0.10))
        alerts = am.check(_snapshot(drawdown_pct=0.05))
        codes = [a.code for a in alerts]
        assert "DRAWDOWN_HIGH" in codes

    def test_warning_above_warn_below_error(self):
        am = AlertManager(AlertConfig(drawdown_warn_pct=0.05, drawdown_error_pct=0.10))
        alerts = am.check(_snapshot(drawdown_pct=0.07))
        codes = [a.code for a in alerts]
        assert "DRAWDOWN_HIGH" in codes
        assert "DRAWDOWN_CRITICAL" not in codes

    def test_error_at_error_threshold(self):
        am = AlertManager(AlertConfig(drawdown_warn_pct=0.05, drawdown_error_pct=0.10))
        alerts = am.check(_snapshot(drawdown_pct=0.10))
        codes = [a.code for a in alerts]
        assert "DRAWDOWN_CRITICAL" in codes
        assert "DRAWDOWN_HIGH" not in codes   # error supersedes warning

    def test_error_level_correct(self):
        am = AlertManager(AlertConfig(drawdown_error_pct=0.10))
        alerts = am.check(_snapshot(drawdown_pct=0.15))
        crit = next(a for a in alerts if a.code == "DRAWDOWN_CRITICAL")
        assert crit.level == "ERROR"

    def test_warning_level_correct(self):
        am = AlertManager(AlertConfig(drawdown_warn_pct=0.05, drawdown_error_pct=0.20))
        alerts = am.check(_snapshot(drawdown_pct=0.06))
        warn = next(a for a in alerts if a.code == "DRAWDOWN_HIGH")
        assert warn.level == "WARNING"

    def test_context_contains_drawdown(self):
        am = AlertManager(AlertConfig(drawdown_warn_pct=0.05))
        alerts = am.check(_snapshot(drawdown_pct=0.08))
        alert = next(a for a in alerts if "DRAWDOWN" in a.code)
        assert "drawdown_pct" in alert.context


# ======================================================================
# AlertManager — API error rate
# ======================================================================

class TestAlertManagerErrorRate:
    def test_no_alert_below_threshold(self):
        am = AlertManager(AlertConfig(error_rate_warn_pct=0.10))
        alerts = am.check(_snapshot(error_rate=0.05))
        assert not any(a.code == "HIGH_ERROR_RATE" for a in alerts)

    def test_alert_at_threshold(self):
        am = AlertManager(AlertConfig(error_rate_warn_pct=0.10))
        alerts = am.check(_snapshot(error_rate=0.10, api_errors=10, total_ticks=100))
        assert any(a.code == "HIGH_ERROR_RATE" for a in alerts)

    def test_alert_is_warning(self):
        am = AlertManager(AlertConfig(error_rate_warn_pct=0.10))
        alerts = am.check(_snapshot(error_rate=0.20))
        err_alert = next(a for a in alerts if a.code == "HIGH_ERROR_RATE")
        assert err_alert.level == "WARNING"


# ======================================================================
# AlertManager — latency
# ======================================================================

class TestAlertManagerLatency:
    def test_no_alert_below_threshold(self):
        am = AlertManager(AlertConfig(latency_warn_ms=5_000))
        alerts = am.check(_snapshot(avg_latency_ms=1_000))
        assert not any(a.code == "HIGH_LATENCY" for a in alerts)

    def test_alert_above_threshold(self):
        am = AlertManager(AlertConfig(latency_warn_ms=5_000))
        alerts = am.check(_snapshot(avg_latency_ms=6_000))
        assert any(a.code == "HIGH_LATENCY" for a in alerts)

    def test_alert_is_warning(self):
        am = AlertManager(AlertConfig(latency_warn_ms=1_000))
        alerts = am.check(_snapshot(avg_latency_ms=2_000))
        lat = next(a for a in alerts if a.code == "HIGH_LATENCY")
        assert lat.level == "WARNING"


# ======================================================================
# AlertManager — win rate
# ======================================================================

class TestAlertManagerWinRate:
    def test_no_alert_before_min_trades(self):
        am = AlertManager(AlertConfig(min_win_rate=0.40, min_trades_for_win_rate=10))
        alerts = am.check(_snapshot(win_rate=0.20, n_trades=5))
        assert not any(a.code == "LOW_WIN_RATE" for a in alerts)

    def test_alert_after_min_trades(self):
        am = AlertManager(AlertConfig(min_win_rate=0.40, min_trades_for_win_rate=10))
        alerts = am.check(_snapshot(win_rate=0.20, n_trades=10))
        assert any(a.code == "LOW_WIN_RATE" for a in alerts)

    def test_no_alert_when_win_rate_acceptable(self):
        am = AlertManager(AlertConfig(min_win_rate=0.30, min_trades_for_win_rate=10))
        alerts = am.check(_snapshot(win_rate=0.50, n_trades=20))
        assert not any(a.code == "LOW_WIN_RATE" for a in alerts)


# ======================================================================
# AlertManager — kill switch
# ======================================================================

class TestAlertManagerKillSwitch:
    def test_kill_switch_active_triggers_error(self):
        am = AlertManager()
        alerts = am.check(_snapshot(), _risk_state(kill_switch=True))
        codes = [a.code for a in alerts]
        assert "KILL_SWITCH_ACTIVE" in codes

    def test_kill_switch_error_level(self):
        am = AlertManager()
        alerts = am.check(_snapshot(), _risk_state(kill_switch=True))
        ks = next(a for a in alerts if a.code == "KILL_SWITCH_ACTIVE")
        assert ks.level == "ERROR"

    def test_no_kill_switch_alert_when_off(self):
        am = AlertManager()
        alerts = am.check(_snapshot(), _risk_state(kill_switch=False))
        assert not any(a.code == "KILL_SWITCH_ACTIVE" for a in alerts)

    def test_no_kill_switch_alert_without_risk_state(self):
        am = AlertManager()
        alerts = am.check(_snapshot())
        assert not any(a.code == "KILL_SWITCH_ACTIVE" for a in alerts)


# ======================================================================
# AlertManager — daily loss
# ======================================================================

class TestAlertManagerDailyLoss:
    def test_daily_loss_alert_triggered(self):
        am = AlertManager(AlertConfig(daily_loss_warn_pct=0.03))
        rs = _risk_state(daily_pnl=-400.0, daily_start_equity=10_000.0)
        alerts = am.check(_snapshot(), rs)
        assert any(a.code == "DAILY_LOSS_HIGH" for a in alerts)

    def test_daily_loss_no_alert_below_threshold(self):
        am = AlertManager(AlertConfig(daily_loss_warn_pct=0.05))
        rs = _risk_state(daily_pnl=-100.0, daily_start_equity=10_000.0)
        alerts = am.check(_snapshot(), rs)
        assert not any(a.code == "DAILY_LOSS_HIGH" for a in alerts)

    def test_daily_loss_no_alert_without_risk_state(self):
        am = AlertManager(AlertConfig(daily_loss_warn_pct=0.01))
        alerts = am.check(_snapshot())
        assert not any(a.code == "DAILY_LOSS_HIGH" for a in alerts)

    def test_daily_gain_no_alert(self):
        am = AlertManager(AlertConfig(daily_loss_warn_pct=0.03))
        rs = _risk_state(daily_pnl=+500.0, daily_start_equity=10_000.0)
        alerts = am.check(_snapshot(), rs)
        assert not any(a.code == "DAILY_LOSS_HIGH" for a in alerts)


# ======================================================================
# AlertManager — multiple simultaneous alerts
# ======================================================================

class TestAlertManagerMultiple:
    def test_multiple_conditions_return_all_alerts(self):
        am = AlertManager(AlertConfig(
            drawdown_warn_pct=0.05,
            error_rate_warn_pct=0.10,
        ))
        snap = _snapshot(drawdown_pct=0.06, error_rate=0.15)
        rs   = _risk_state(kill_switch=True)
        alerts = am.check(snap, rs)
        codes = [a.code for a in alerts]
        assert "DRAWDOWN_HIGH" in codes
        assert "HIGH_ERROR_RATE" in codes
        assert "KILL_SWITCH_ACTIVE" in codes


# ======================================================================
# AlertManager — check_and_log
# ======================================================================

class TestCheckAndLog:
    def test_returns_same_alerts_as_check(self):
        am = AlertManager(AlertConfig(drawdown_warn_pct=0.05))
        snap = _snapshot(drawdown_pct=0.06)
        assert am.check_and_log(snap) == am.check(snap)

    def test_empty_when_all_clear(self):
        am = AlertManager()
        assert am.check_and_log(_snapshot()) == []


# ======================================================================
# AlertConfig defaults
# ======================================================================

class TestAlertConfigDefaults:
    def test_drawdown_warn_5_pct(self):
        assert AlertConfig().drawdown_warn_pct == pytest.approx(0.05)

    def test_drawdown_error_10_pct(self):
        assert AlertConfig().drawdown_error_pct == pytest.approx(0.10)

    def test_error_rate_warn_10_pct(self):
        assert AlertConfig().error_rate_warn_pct == pytest.approx(0.10)

    def test_min_win_rate_30_pct(self):
        assert AlertConfig().min_win_rate == pytest.approx(0.30)

    def test_min_trades_10(self):
        assert AlertConfig().min_trades_for_win_rate == 10
