"""
Integration tests — multiple real components running together.

External I/O is the only thing mocked:
  - fetch_ohlcv (prevents real network calls in PaperEngine tests)
  - time.sleep  (prevents real delays in loop tests)

Everything else runs as real instances:
  SMCStrategy, BacktestEngine, PaperEngine, RiskManager,
  OrderExecutor, PaperConnector, SessionMetrics, AlertManager.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from zeus.backtest.engine import BacktestEngine
from zeus.backtest.report import BacktestResult
from zeus.exchange.connector import OrderResult
from zeus.exchange.paper import PaperConnector
from zeus.monitoring.alerts import AlertConfig, AlertManager
from zeus.monitoring.metrics import SessionMetrics
from zeus.orders.executor import OrderExecutor
from zeus.orders.models import Trade, TradeStatus
from zeus.paper.engine import PaperEngine
from zeus.risk.manager import RiskManager
from zeus.strategy.base import Signal, SignalType, Strategy
from zeus.strategy.smc_strategy import SMCStrategy


# ======================================================================
# Shared helpers
# ======================================================================

def _ohlcv(n_bars: int = 150, seed: int = 42) -> pd.DataFrame:
    """
    Deterministic synthetic OHLCV with alternating trend phases and realistic noise.

    Uses UTC-aware DatetimeIndex so PaperEngine daily-reset logic is exercised.
    Alternating 20-bar up/down phases give the SMC analyser enough structure
    events to fire confluent signals with low min_score thresholds.
    """
    rng = np.random.default_rng(seed)
    closes = np.empty(n_bars)
    closes[0] = 30_000.0
    for i in range(1, n_bars):
        drift = 80.0 if (i // 20) % 2 == 0 else -60.0   # 20-bar up / 20-bar down
        closes[i] = max(closes[i - 1] + drift + rng.normal(0, 120.0), 1_000.0)

    spread = np.abs(rng.normal(60, 25, n_bars))
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open":   closes + rng.normal(0, 30, n_bars),
            "high":   closes + spread,
            "low":    np.maximum(closes - spread, 1.0),
            "close":  closes,
            "volume": rng.uniform(500, 5_000, n_bars),
        },
        index=idx,
    )


def _fast_strategy(**kwargs) -> SMCStrategy:
    """
    SMCStrategy tuned for fast test execution.

    swing_length=10 reduces _min_bars to 20 (default is 100).
    min_score=2.0 requires only the two gate factors — maximises signal frequency.
    """
    defaults = dict(swing_length=10, internal_length=3, atr_period=20, min_score=2.0)
    defaults.update(kwargs)
    return SMCStrategy(**defaults)


def _executor_pipeline(
    balance: float = 10_000.0,
    price: float = 100.0,
    sl_pct: float = 0.10,
    tp_pct: float = 0.20,
    max_position_pct: float = 0.02,
    max_open: int = 3,
    symbol: str = "BTC/USDT",
) -> tuple[PaperConnector, RiskManager, OrderExecutor]:
    """Build a real PaperConnector + RiskManager + OrderExecutor triple."""
    connector = PaperConnector(initial_balance=balance, slippage_pct=0.0)
    risk = RiskManager(
        max_position_pct=max_position_pct,
        stop_loss_pct=sl_pct,
        take_profit_pct=tp_pct,
        max_exposure_pct=0.95,   # near-full so exposure check doesn't bind in tests
        max_open_positions=max_open,
    )
    risk.initialise_capital(balance)
    executor = OrderExecutor(
        connector=connector,
        risk_manager=risk,
        stop_loss_pct=sl_pct,
        take_profit_pct=tp_pct,
    )
    connector.set_current_price(symbol, price)
    return connector, risk, executor


def _signal(kind: SignalType = SignalType.LONG, bar_index: int = 0) -> Signal:
    return Signal(kind, 0.9, "integration-test", bar_index)


def _closed_trade(pnl: float, side: str = "buy") -> Trade:
    """Minimal Trade in a closed state for injection into SessionMetrics."""
    now = datetime.now(tz=timezone.utc)
    order = OrderResult(
        order_id="test",
        symbol="BTC/USDT",
        side=side,
        quantity=0.01,
        filled=0.01,
        price=None,
        average=100_000.0,
        status="closed",
        timestamp=now,
    )
    t = Trade(
        trade_id=f"it_{abs(int(pnl * 100))}",
        symbol="BTC/USDT",
        side=side,
        quantity=0.01,
        entry_price=100_000.0,
        sl_price=99_000.0,
        tp_price=102_000.0,
        entry_order=order,
    )
    t.status = TradeStatus.CLOSED_TP if pnl >= 0 else TradeStatus.CLOSED_SL
    t.realized_pnl = pnl
    return t


# ======================================================================
# 1. BacktestEngine + SMCStrategy
# ======================================================================

class TestBacktestPipeline:
    """End-to-end backtest with no mocks whatsoever."""

    def test_run_returns_backtest_result(self):
        engine = BacktestEngine(strategy=_fast_strategy(), initial_balance=10_000.0)
        result = engine.run(_ohlcv())
        assert isinstance(result, BacktestResult)

    def test_equity_curve_length_is_n_bars_plus_one(self):
        df = _ohlcv(n_bars=60)
        engine = BacktestEngine(strategy=_fast_strategy())
        result = engine.run(df)
        assert len(result.equity_curve) == len(df) + 1

    def test_equity_curve_starts_at_initial_balance(self):
        engine = BacktestEngine(strategy=_fast_strategy(), initial_balance=8_000.0)
        result = engine.run(_ohlcv())
        assert result.equity_curve[0] == pytest.approx(8_000.0)

    def test_no_open_trades_after_run(self):
        """All positions are force-closed at the last bar."""
        engine = BacktestEngine(strategy=_fast_strategy(min_score=2.0))
        result = engine.run(_ohlcv())
        open_count = sum(1 for t in result.trades if t.is_open)
        assert open_count == 0

    def test_win_rate_bounded(self):
        result = BacktestEngine(strategy=_fast_strategy()).run(_ohlcv())
        assert 0.0 <= result.win_rate <= 1.0

    def test_two_consecutive_runs_are_independent(self):
        """run() must not carry state between calls."""
        engine = BacktestEngine(strategy=_fast_strategy(), initial_balance=10_000.0)
        r1 = engine.run(_ohlcv(seed=1))
        r2 = engine.run(_ohlcv(seed=2))

        assert r1.equity_curve[0] == pytest.approx(10_000.0)
        assert r2.equity_curve[0] == pytest.approx(10_000.0)
        assert len(r1.equity_curve) == 151   # 150 bars + 1
        assert len(r2.equity_curve) == 151

    def test_stricter_score_never_produces_more_trades(self):
        """
        Raising min_score makes signals harder to fire.
        The trade count with a strict threshold must be ≤ permissive.
        """
        df = _ohlcv()
        trades_low = BacktestEngine(
            strategy=_fast_strategy(min_score=2.0)
        ).run(df).n_trades
        trades_high = BacktestEngine(
            strategy=_fast_strategy(min_score=10.0)
        ).run(df).n_trades
        assert trades_high <= trades_low

    def test_result_metrics_are_self_consistent(self):
        """n_winning + n_losing must sum to n_trades."""
        result = BacktestEngine(strategy=_fast_strategy()).run(_ohlcv())
        assert result.n_winning + result.n_losing == result.n_trades


# ======================================================================
# 2. PaperEngine + SMCStrategy + SessionMetrics (fetch_ohlcv mocked)
# ======================================================================

class TestPaperEnginePipeline:
    """
    Real PaperEngine + SMCStrategy + SessionMetrics running together.
    Only fetch_ohlcv and time.sleep are mocked to prevent I/O.
    """

    def _engine(self, df: pd.DataFrame | None = None, **kwargs) -> PaperEngine:
        if df is None:
            df = _ohlcv()
        market = MagicMock()
        market.fetch_ohlcv.return_value = df
        return PaperEngine(
            strategy=_fast_strategy(),
            market_connector=market,
            poll_interval=0.0,
            slippage_pct=0.0,
            **kwargs,
        )

    @patch("zeus.paper.engine.time.sleep")
    def test_successful_ticks_increment_metrics(self, _sleep):
        engine = self._engine()
        engine.run(max_ticks=5)
        assert engine.metrics.snapshot().total_ticks == 5

    @patch("zeus.paper.engine.time.sleep")
    def test_metrics_equity_reflects_initial_balance(self, _sleep):
        engine = self._engine(initial_balance=25_000.0)
        engine.run(max_ticks=1)
        snap = engine.metrics.snapshot()
        assert snap.current_equity == pytest.approx(25_000.0, rel=0.01)

    @patch("zeus.paper.engine.time.sleep")
    def test_skipped_tick_not_counted(self, _sleep):
        """Empty DataFrame causes early return — tick must not be counted."""
        engine = self._engine(df=pd.DataFrame())
        engine.run(max_ticks=3)
        assert engine.metrics.snapshot().total_ticks == 0

    @patch("zeus.paper.engine.time.sleep")
    def test_fetch_exception_counted_as_api_error(self, _sleep):
        market = MagicMock()
        market.fetch_ohlcv.side_effect = RuntimeError("timeout")
        engine = PaperEngine(
            strategy=_fast_strategy(),
            market_connector=market,
            poll_interval=0.0,
        )
        engine.run(max_ticks=2)
        assert engine.metrics.snapshot().api_errors == 2

    @patch("zeus.paper.engine.time.sleep")
    def test_kill_switch_blocks_entries_loop_continues(self, _sleep):
        """Kill switch must block orders, not the tick loop."""
        engine = self._engine()
        engine.set_kill_switch(True, "integration test")
        engine.run(max_ticks=4)
        assert len(engine.executor.open_trades) == 0
        assert engine.metrics.snapshot().total_ticks == 4

    @patch("zeus.paper.engine.time.sleep")
    def test_stop_from_strategy_halts_loop(self, _sleep):
        """A strategy that calls engine.stop() must exit the loop after one tick."""
        df = _ohlcv()
        market = MagicMock()
        market.fetch_ohlcv.return_value = df

        engine = PaperEngine(
            strategy=_fast_strategy(),
            market_connector=market,
            poll_interval=0.0,
        )

        class _StopOnFirst:
            def __init__(self):
                self._done = False
            def generate_signal(self, df, bar_index):
                if not self._done:
                    self._done = True
                    engine.stop()
                return Signal(SignalType.NONE, 0.0, "stop", bar_index)

        engine._strategy = _StopOnFirst()
        engine.run(max_ticks=100)

        assert engine.metrics.snapshot().total_ticks == 1

    @patch("zeus.paper.engine.time.sleep")
    def test_closed_trade_recorded_in_metrics(self, _sleep):
        """An SL hit on tick 2 must appear in metrics as a closed trade."""
        prices = [100.0, 80.0]
        call_count = [0]

        market = MagicMock()
        def _fetch(*a, **kw):
            p = prices[min(call_count[0], 1)]
            call_count[0] += 1
            n = 50
            idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
            return pd.DataFrame(
                {"open": [p]*n, "high": [p+1]*n,
                 "low":  [p-1]*n, "close": [p]*n, "volume": [100.0]*n},
                index=idx,
            )
        market.fetch_ohlcv.side_effect = _fetch

        class _LongOnce:
            _fired = False
            def generate_signal(self, df, bar_index):
                if not self._fired:
                    self._fired = True
                    return Signal(SignalType.LONG, 0.9, "long", bar_index)
                return Signal(SignalType.NONE, 0.0, "none", bar_index)

        engine = PaperEngine(
            strategy=_LongOnce(),
            market_connector=market,
            initial_balance=10_000.0,
            stop_loss_pct=0.10,     # SL at 90
            poll_interval=0.0,
            slippage_pct=0.0,
        )
        engine.run(max_ticks=2)

        snap = engine.metrics.snapshot()
        assert snap.n_trades == 1
        assert snap.n_losing == 1


# ======================================================================
# 3. RiskManager + OrderExecutor + PaperConnector
# ======================================================================

class TestRiskExecutorPipeline:
    """
    Full order lifecycle through the real risk + execution stack.
    No strategy involved — signals are constructed directly.
    """

    def test_long_entry_decreases_available_balance(self):
        connector, risk, executor = _executor_pipeline()
        before = risk.state.available_balance
        executor.execute(_signal(SignalType.LONG), "BTC/USDT", 100.0)
        assert risk.state.available_balance < before

    def test_short_entry_registers_as_open_position(self):
        connector, risk, executor = _executor_pipeline()
        executor.execute(_signal(SignalType.SHORT), "BTC/USDT", 100.0)
        assert risk.state.open_positions == 1
        assert executor.open_trades[0].side == "sell"

    def test_kill_switch_blocks_entry(self):
        connector, risk, executor = _executor_pipeline()
        risk.set_kill_switch(True, "test")
        result = executor.execute(_signal(SignalType.LONG), "BTC/USDT", 100.0)
        assert result is not None
        assert result.status == TradeStatus.REJECTED
        assert risk.state.open_positions == 0

    def test_max_positions_blocks_extra_entry(self):
        """Opening trades on three symbols fills the cap; a fourth is rejected."""
        connector, risk, executor = _executor_pipeline(max_open=3)
        for sym in ("BTC/USDT", "ETH/USDT", "SOL/USDT"):
            connector.set_current_price(sym, 100.0)
            t = executor.execute(_signal(SignalType.LONG), sym, 100.0)
            assert t is not None and t.is_open, f"Expected open trade for {sym}"

        connector.set_current_price("DOGE/USDT", 100.0)
        result = executor.execute(_signal(SignalType.LONG), "DOGE/USDT", 100.0)
        assert result is not None
        assert result.status == TradeStatus.REJECTED
        assert risk.state.open_positions == 3

    def test_one_position_per_symbol_enforced(self):
        """Duplicate entry on same symbol must be silently blocked."""
        connector, risk, executor = _executor_pipeline()
        executor.execute(_signal(SignalType.LONG), "BTC/USDT", 100.0)
        result = executor.execute(_signal(SignalType.LONG), "BTC/USDT", 100.0)
        assert result is None
        assert risk.state.open_positions == 1

    def test_sl_hit_closes_trade_and_frees_positions(self):
        """Price drops below SL → trade closes, position count returns to zero."""
        connector, risk, executor = _executor_pipeline(sl_pct=0.10)
        executor.execute(_signal(SignalType.LONG), "BTC/USDT", 100.0)
        assert risk.state.open_positions == 1

        # Price drops below SL (entry=100, SL=90 → trigger at 80)
        connector.set_current_price("BTC/USDT", 80.0)
        closed = executor.check_exits("BTC/USDT", 80.0)

        assert len(closed) == 1
        assert closed[0].status == TradeStatus.CLOSED_SL
        assert closed[0].realized_pnl < 0
        assert risk.state.open_positions == 0

    def test_tp_hit_closes_trade_with_profit(self):
        """Price rises above TP → trade closes with positive realized P&L."""
        connector, risk, executor = _executor_pipeline(sl_pct=0.30, tp_pct=0.20)
        executor.execute(_signal(SignalType.LONG), "BTC/USDT", 100.0)

        # Price surpasses TP (entry=100, TP=120 → trigger at 130)
        connector.set_current_price("BTC/USDT", 130.0)
        closed = executor.check_exits("BTC/USDT", 130.0)

        assert len(closed) == 1
        assert closed[0].status == TradeStatus.CLOSED_TP
        assert closed[0].realized_pnl > 0
        assert risk.state.open_positions == 0

    def test_balance_recovers_after_tp_close(self):
        """Available balance after TP should exceed balance right after entry."""
        connector, risk, executor = _executor_pipeline(sl_pct=0.30, tp_pct=0.20)
        executor.execute(_signal(SignalType.LONG), "BTC/USDT", 100.0)
        balance_after_entry = risk.state.available_balance

        connector.set_current_price("BTC/USDT", 130.0)
        executor.check_exits("BTC/USDT", 130.0)

        assert risk.state.available_balance > balance_after_entry


# ======================================================================
# 4. SessionMetrics + AlertManager
# ======================================================================

class TestMonitoringPipeline:
    """
    SessionMetrics and AlertManager working together with real data.
    No mocks — all data is injected via the public API.
    """

    def test_healthy_metrics_produce_no_alerts(self):
        m = SessionMetrics(10_000.0)
        for i in range(5):
            m.record_tick(50.0, 10_000.0 + i * 10)   # equity rising
        snap = m.snapshot()

        alerts = AlertManager().check(snap)
        assert alerts == []

    def test_drawdown_warning_fires_when_equity_drops(self):
        cfg = AlertConfig(drawdown_warn_pct=0.05, drawdown_error_pct=0.15)
        m = SessionMetrics(10_000.0)
        m.record_tick(50.0, 10_000.0)   # peak established
        m.record_tick(50.0, 9_400.0)    # 6 % drawdown → above warn threshold

        snap = m.snapshot()
        alerts = AlertManager(cfg).check(snap)
        codes = {a.code for a in alerts}
        assert "DRAWDOWN_HIGH" in codes

    def test_critical_drawdown_fires_error_level_alert(self):
        cfg = AlertConfig(drawdown_error_pct=0.10)
        m = SessionMetrics(10_000.0)
        m.record_tick(50.0, 10_000.0)
        m.record_tick(50.0, 8_000.0)    # 20 % drawdown → above error threshold

        snap = m.snapshot()
        alerts = AlertManager(cfg).check(snap)
        error_alerts = [a for a in alerts if a.level == "ERROR" and a.code == "DRAWDOWN_CRITICAL"]
        assert len(error_alerts) == 1

    def test_closed_trades_update_n_trades_and_win_rate(self):
        m = SessionMetrics(10_000.0)
        m.record_trade(_closed_trade(+100.0))   # win
        m.record_trade(_closed_trade(-50.0))    # loss
        m.record_trade(_closed_trade(+80.0))    # win

        snap = m.snapshot()
        assert snap.n_trades == 3
        assert snap.n_winning == 2
        assert snap.n_losing == 1
        assert snap.win_rate == pytest.approx(2 / 3)

    def test_low_win_rate_alert_fires_after_min_trades(self):
        cfg = AlertConfig(min_win_rate=0.50, min_trades_for_win_rate=3)
        m = SessionMetrics(10_000.0)
        m.record_tick(50.0, 10_000.0)
        for _ in range(3):
            m.record_trade(_closed_trade(-10.0))   # all losses

        snap = m.snapshot()
        alerts = AlertManager(cfg).check(snap)
        codes = {a.code for a in alerts}
        assert "LOW_WIN_RATE" in codes

    def test_high_error_rate_alert_fires(self):
        cfg = AlertConfig(error_rate_warn_pct=0.20)
        m = SessionMetrics(10_000.0)
        for _ in range(8):
            m.record_tick(50.0, 10_000.0)          # 8 successful ticks
        for _ in range(3):
            m.record_api_error()                    # 3 errors → 37.5 % rate

        snap = m.snapshot()
        alerts = AlertManager(cfg).check(snap)
        codes = {a.code for a in alerts}
        assert "HIGH_ERROR_RATE" in codes

    def test_kill_switch_alert_fires_via_risk_state(self):
        m = SessionMetrics(10_000.0)
        m.record_tick(50.0, 10_000.0)
        snap = m.snapshot()

        risk = RiskManager()
        risk.initialise_capital(10_000.0)
        risk.set_kill_switch(True, "integration test")

        alerts = AlertManager().check(snap, risk.state)
        codes = {a.code for a in alerts}
        assert "KILL_SWITCH_ACTIVE" in codes

    def test_log_summary_does_not_raise(self):
        """log_summary() must complete without exceptions."""
        m = SessionMetrics(10_000.0)
        m.record_tick(40.0, 10_200.0)
        m.record_trade(_closed_trade(+200.0))
        m.log_summary()  # must not raise
