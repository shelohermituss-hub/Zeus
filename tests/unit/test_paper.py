"""
Tests for zeus/paper/engine.py.

All network calls are mocked.  time.sleep is patched to avoid real delays.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from zeus.orders.models import TradeStatus
from zeus.paper.engine import PaperEngine
from zeus.strategy.base import Signal, SignalType


# ======================================================================
# Helpers
# ======================================================================

def _make_df(n_bars: int = 100, close: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open":   [close] * n_bars,
            "high":   [close + 1.0] * n_bars,
            "low":    [close - 1.0] * n_bars,
            "close":  [close] * n_bars,
            "volume": [1_000.0] * n_bars,
        },
        index=idx,
    )


def _market(df: pd.DataFrame | None = None, n_bars: int = 100, close: float = 100.0):
    conn = MagicMock()
    conn.fetch_ohlcv.return_value = df if df is not None else _make_df(n_bars, close)
    return conn


class _AlwaysNone:
    def generate_signal(self, df, bar_index):
        return Signal(SignalType.NONE, 0.0, "none", bar_index)


class _AlwaysLong:
    def generate_signal(self, df, bar_index):
        return Signal(SignalType.LONG, 0.9, "long", bar_index)


class _AlwaysShort:
    def generate_signal(self, df, bar_index):
        return Signal(SignalType.SHORT, 0.9, "short", bar_index)


def _engine(
    strategy=None,
    market=None,
    balance: float = 10_000.0,
    sl_pct: float = 0.10,   # wide SL so accidental closes don't happen
    tp_pct: float | None = 0.20,
    poll_interval: float = 0.0,
    slippage_pct: float = 0.0,  # zero slippage for deterministic test prices
) -> PaperEngine:
    return PaperEngine(
        strategy=strategy or _AlwaysNone(),
        market_connector=market or _market(),
        initial_balance=balance,
        stop_loss_pct=sl_pct,
        take_profit_pct=tp_pct,
        poll_interval=poll_interval,
        slippage_pct=slippage_pct,
    )


# ======================================================================
# Initialisation
# ======================================================================

class TestPaperEngineInit:
    def test_executor_initially_empty(self):
        e = _engine()
        assert e.executor.open_trades == []

    def test_risk_balance_set(self):
        e = _engine(balance=50_000.0)
        assert e.risk.state.available_balance == pytest.approx(50_000.0)

    def test_running_false_before_run(self):
        e = _engine()
        assert e._running is False

    def test_kill_switch_off_by_default(self):
        e = _engine()
        assert e.risk.state.kill_switch_active is False

    def test_last_day_none_before_run(self):
        e = _engine()
        assert e._last_day is None


# ======================================================================
# stop()
# ======================================================================

class TestStop:
    def test_stop_sets_running_false(self):
        e = _engine()
        e._running = True
        e.stop()
        assert e._running is False

    def test_stop_before_run_is_safe(self):
        e = _engine()
        e.stop()   # must not raise


# ======================================================================
# set_kill_switch / set_exchange_connected
# ======================================================================

class TestControls:
    def test_set_kill_switch_activates(self):
        e = _engine()
        e.set_kill_switch(True, "test")
        assert e.risk.state.kill_switch_active is True

    def test_set_kill_switch_deactivates(self):
        e = _engine()
        e.set_kill_switch(True)
        e.set_kill_switch(False)
        assert e.risk.state.kill_switch_active is False

    def test_set_exchange_disconnected(self):
        e = _engine()
        e.set_exchange_connected(False)
        assert e.risk.state.exchange_connected is False

    def test_set_exchange_reconnected(self):
        e = _engine()
        e.set_exchange_connected(False)
        e.set_exchange_connected(True)
        assert e.risk.state.exchange_connected is True


# ======================================================================
# status()
# ======================================================================

class TestStatus:
    def test_status_keys(self):
        e = _engine()
        s = e.status()
        for key in (
            "equity", "available_balance", "open_positions",
            "open_trade_ids", "closed_trades", "kill_switch",
        ):
            assert key in s

    def test_status_initial_equity(self):
        e = _engine(balance=10_000.0)
        assert e.status()["equity"] == pytest.approx(10_000.0)

    def test_status_no_open_trades_initially(self):
        e = _engine()
        assert e.status()["open_positions"] == 0
        assert e.status()["open_trade_ids"] == []

    def test_status_kill_switch_reflected(self):
        e = _engine()
        e.set_kill_switch(True)
        assert e.status()["kill_switch"] is True


# ======================================================================
# run() — loop control
# ======================================================================

class TestRunLoopControl:
    @patch("zeus.paper.engine.time.sleep")
    def test_max_ticks_zero_no_ticks(self, mock_sleep):
        market = _market()
        e = _engine(market=market)
        e.run(max_ticks=0)
        market.fetch_ohlcv.assert_not_called()

    @patch("zeus.paper.engine.time.sleep")
    def test_max_ticks_one_calls_fetch_once(self, mock_sleep):
        market = _market()
        e = _engine(market=market)
        e.run(max_ticks=1)
        market.fetch_ohlcv.assert_called_once()

    @patch("zeus.paper.engine.time.sleep")
    def test_max_ticks_three_calls_fetch_three_times(self, mock_sleep):
        market = _market()
        e = _engine(market=market)
        e.run(max_ticks=3)
        assert market.fetch_ohlcv.call_count == 3

    @patch("zeus.paper.engine.time.sleep")
    def test_sleep_called_between_ticks(self, mock_sleep):
        market = _market()
        e = PaperEngine(
            strategy=_AlwaysNone(),
            market_connector=market,
            poll_interval=5.0,
        )
        e.run(max_ticks=3)
        # sleep called twice (between tick 0→1 and 1→2, not after last)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(5.0)

    @patch("zeus.paper.engine.time.sleep")
    def test_running_false_after_run(self, mock_sleep):
        e = _engine()
        e.run(max_ticks=2)
        assert e._running is False

    @patch("zeus.paper.engine.time.sleep")
    def test_stop_in_strategy_halts_loop(self, mock_sleep):
        """Strategy calls engine.stop() → loop exits early."""
        market = _market()
        engine = _engine(market=market)

        class _StopAfterFirst:
            def generate_signal(self, df, bar_index):
                engine.stop()
                return Signal(SignalType.NONE, 0.0, "stop", bar_index)

        engine._strategy = _StopAfterFirst()
        engine.run(max_ticks=10)
        # Only one fetch should have occurred
        assert market.fetch_ohlcv.call_count == 1


# ======================================================================
# _tick() — data guards
# ======================================================================

class TestTickDataGuards:
    @patch("zeus.paper.engine.time.sleep")
    def test_empty_df_tick_skipped(self, mock_sleep):
        market = _market(df=pd.DataFrame())
        e = _engine(market=market)
        e.run(max_ticks=1)
        # No trade should exist — tick was skipped
        assert e.executor.all_trades == []

    @patch("zeus.paper.engine.time.sleep")
    def test_single_bar_df_tick_skipped(self, mock_sleep):
        market = _market(df=_make_df(1))
        e = _engine(market=market)
        e.run(max_ticks=1)
        assert e.executor.all_trades == []

    @patch("zeus.paper.engine.time.sleep")
    def test_fetch_exception_loop_continues(self, mock_sleep):
        """fetch_ohlcv raises on tick 1, returns data on tick 2."""
        market = MagicMock()
        market.fetch_ohlcv.side_effect = [
            RuntimeError("network error"),
            _make_df(100),
        ]
        e = _engine(strategy=_AlwaysNone(), market=market)
        # Should not raise even though tick 1 fails
        e.run(max_ticks=2)
        assert market.fetch_ohlcv.call_count == 2


# ======================================================================
# _tick() — bar_index
# ======================================================================

class TestTickBarIndex:
    @patch("zeus.paper.engine.time.sleep")
    def test_signal_called_with_penultimate_bar(self, mock_sleep):
        """generate_signal must receive bar_index = len(df) - 2."""
        seen_indices: list[int] = []

        class _RecordBarIndex:
            def generate_signal(self, df, bar_index):
                seen_indices.append(bar_index)
                return Signal(SignalType.NONE, 0.0, "none", bar_index)

        df = _make_df(50)
        market = _market(df=df)
        e = PaperEngine(
            strategy=_RecordBarIndex(),
            market_connector=market,
            poll_interval=0.0,
        )
        e.run(max_ticks=1)
        assert seen_indices == [48]   # len(50) - 2


# ======================================================================
# _tick() — LONG signal → trade opened
# ======================================================================

class TestTickLongSignal:
    def setup_method(self):
        market = _market(close=100.0)
        self.engine = PaperEngine(
            strategy=_AlwaysLong(),
            market_connector=market,
            initial_balance=10_000.0,
            stop_loss_pct=0.10,
            take_profit_pct=0.20,
            poll_interval=0.0,
            slippage_pct=0.0,
        )
        with patch("zeus.paper.engine.time.sleep"):
            self.engine.run(max_ticks=1)

    def test_one_open_trade(self):
        assert len(self.engine.executor.open_trades) == 1

    def test_trade_side_buy(self):
        assert self.engine.executor.open_trades[0].side == "buy"

    def test_status_open_positions_one(self):
        assert self.engine.status()["open_positions"] == 1

    def test_trade_entry_price(self):
        trade = self.engine.executor.open_trades[0]
        assert trade.entry_price == pytest.approx(100.0)

    def test_sl_below_entry(self):
        trade = self.engine.executor.open_trades[0]
        assert trade.sl_price == pytest.approx(90.0)  # 100 * (1 - 0.10)

    def test_tp_above_entry(self):
        trade = self.engine.executor.open_trades[0]
        assert trade.tp_price == pytest.approx(120.0)  # 100 * (1 + 0.20)


# ======================================================================
# _tick() — SHORT signal
# ======================================================================

class TestTickShortSignal:
    @patch("zeus.paper.engine.time.sleep")
    def test_short_trade_side_sell(self, mock_sleep):
        engine = PaperEngine(
            strategy=_AlwaysShort(),
            market_connector=_market(close=100.0),
            initial_balance=10_000.0,
            stop_loss_pct=0.10,
            poll_interval=0.0,
        )
        engine.run(max_ticks=1)
        assert engine.executor.open_trades[0].side == "sell"


# ======================================================================
# _tick() — deduplication
# ======================================================================

class TestTickDeduplication:
    @patch("zeus.paper.engine.time.sleep")
    def test_second_tick_does_not_double_open(self, mock_sleep):
        engine = PaperEngine(
            strategy=_AlwaysLong(),
            market_connector=_market(),
            initial_balance=10_000.0,
            stop_loss_pct=0.10,
            poll_interval=0.0,
        )
        engine.run(max_ticks=5)
        # Second signal ignored — only one open position
        assert len(engine.executor.open_trades) == 1


# ======================================================================
# _tick() — exit detection
# ======================================================================

class TestTickExitDetection:
    @patch("zeus.paper.engine.time.sleep")
    def test_sl_detected_on_next_tick(self, mock_sleep):
        """
        Tick 0: price=100, LONG signal → trade opened, SL at 90.
        Tick 1: price drops to 80 → SL crossed → trade closed.
        """
        prices = [100.0, 80.0]
        tick = [0]

        market = MagicMock()
        def _ohlcv(*args, **kwargs):
            p = prices[min(tick[0], 1)]
            tick[0] += 1
            return _make_df(100, close=p)
        market.fetch_ohlcv.side_effect = _ohlcv

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
            stop_loss_pct=0.10,   # SL at 90.0
            poll_interval=0.0,
        )
        engine.run(max_ticks=2)

        closed = [t for t in engine.executor.all_trades if not t.is_open]
        assert len(closed) == 1
        assert closed[0].status == TradeStatus.CLOSED_SL

    @patch("zeus.paper.engine.time.sleep")
    def test_tp_detected_on_next_tick(self, mock_sleep):
        """
        Tick 0: price=100, LONG signal, TP at 120.
        Tick 1: price rises to 130 → TP hit.
        """
        prices = [100.0, 130.0]
        tick = [0]

        market = MagicMock()
        def _ohlcv(*args, **kwargs):
            p = prices[min(tick[0], 1)]
            tick[0] += 1
            return _make_df(100, close=p)
        market.fetch_ohlcv.side_effect = _ohlcv

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
            stop_loss_pct=0.30,
            take_profit_pct=0.20,   # TP at 120.0
            poll_interval=0.0,
        )
        engine.run(max_ticks=2)

        closed = [t for t in engine.executor.all_trades if not t.is_open]
        assert len(closed) == 1
        assert closed[0].status == TradeStatus.CLOSED_TP


# ======================================================================
# _tick() — kill switch blocks entries
# ======================================================================

class TestTickKillSwitch:
    @patch("zeus.paper.engine.time.sleep")
    def test_kill_switch_prevents_entry(self, mock_sleep):
        engine = PaperEngine(
            strategy=_AlwaysLong(),
            market_connector=_market(),
            initial_balance=10_000.0,
            stop_loss_pct=0.10,
            poll_interval=0.0,
        )
        engine.set_kill_switch(True, "test")
        engine.run(max_ticks=2)

        open_trades = engine.executor.open_trades
        assert len(open_trades) == 0


# ======================================================================
# Daily reset
# ======================================================================

class TestDailyReset:
    def _run_daily_reset(self, engine, iso_day: str) -> None:
        """Call engine._check_daily_reset() with datetime patched to *iso_day*."""
        from datetime import date
        fake_now = MagicMock()
        fake_now.date.return_value = date.fromisoformat(iso_day)
        with patch("zeus.paper.engine.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            engine._check_daily_reset()

    def test_daily_reset_triggered_on_day_change(self):
        from datetime import date
        engine = _engine()
        engine._last_day = date(2024, 1, 1)
        engine._risk.state.daily_realized_pnl = -300.0

        self._run_daily_reset(engine, "2024-01-02")

        assert engine._risk.state.daily_realized_pnl == 0.0

    def test_same_day_no_reset(self):
        from datetime import date
        engine = _engine()
        engine._last_day = date(2024, 1, 1)
        engine._risk.state.daily_realized_pnl = -300.0

        self._run_daily_reset(engine, "2024-01-01")

        assert engine._risk.state.daily_realized_pnl == pytest.approx(-300.0)


# ======================================================================
# Fetch parameters forwarded
# ======================================================================

class TestFetchParameters:
    @patch("zeus.paper.engine.time.sleep")
    def test_fetch_called_with_correct_args(self, mock_sleep):
        market = _market()
        engine = PaperEngine(
            strategy=_AlwaysNone(),
            market_connector=market,
            symbol="ETH/USDT",
            timeframe="15m",
            ohlcv_limit=200,
            poll_interval=0.0,
        )
        engine.run(max_ticks=1)
        market.fetch_ohlcv.assert_called_once_with("ETH/USDT", "15m", 200)


# ======================================================================
# Equity tracking
# ======================================================================

class TestEquityTracking:
    @patch("zeus.paper.engine.time.sleep")
    def test_balance_decreases_after_entry(self, mock_sleep):
        engine = PaperEngine(
            strategy=_AlwaysLong(),
            market_connector=_market(close=100.0),
            initial_balance=10_000.0,
            stop_loss_pct=0.10,
            poll_interval=0.0,
        )
        engine.run(max_ticks=1)
        # Available balance should be less than initial (position locked capital)
        assert engine.risk.state.available_balance < 10_000.0

    @patch("zeus.paper.engine.time.sleep")
    def test_equity_restored_after_exit(self, mock_sleep):
        prices = [100.0, 130.0]
        tick = [0]

        market = MagicMock()
        def _ohlcv(*args, **kwargs):
            p = prices[min(tick[0], 1)]
            tick[0] += 1
            return _make_df(100, close=p)
        market.fetch_ohlcv.side_effect = _ohlcv

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
            stop_loss_pct=0.30,
            take_profit_pct=0.20,
            poll_interval=0.0,
        )
        engine.run(max_ticks=2)
        # After TP close, no open positions
        assert engine.risk.state.open_positions == 0
