"""
Paper trading engine — real-time simulation using live market data.

The loop fetches OHLCV data on each tick, generates a signal from the last
complete bar, and routes it through the full risk + execution pipeline using
a PaperConnector.  No real orders are ever sent.

Differences from BacktestEngine that are intentional and must be kept:
  - Signals are generated on the PENULTIMATE bar (last complete candle),
    not the forming bar.
  - Between ticks the engine sleeps poll_interval seconds.
  - Daily P&L is reset automatically at UTC midnight.
  - Exceptions in a single tick are caught and logged; the loop continues.
"""
from __future__ import annotations

import time
from datetime import date, datetime, timezone

from zeus.exchange.connector import ExchangeConnector
from zeus.exchange.paper import PaperConnector
from zeus.monitoring.alerts import AlertConfig, AlertManager
from zeus.monitoring.metrics import SessionMetrics
from zeus.monitoring.telegram import TelegramNotifier
from zeus.orders.executor import OrderExecutor
from zeus.orders.models import TradeStatus
from zeus.paper.signal_store import SignalStore
from zeus.risk.manager import RiskManager
from zeus.strategy.base import Strategy
from zeus.utils.logger import logger


class PaperEngine:
    """
    Real-time paper trading loop.

    Args:
        strategy:           Any Strategy subclass (identical to the live engine).
        market_connector:   Live or paper connector used only for fetch_ohlcv().
        initial_balance:    Starting paper balance in quote currency.
        symbol:             Trading pair (e.g. "BTC/USDT").
        timeframe:          CCXT timeframe string (e.g. "1h", "15m").
        ohlcv_limit:        Bars to fetch per tick (must cover strategy warm-up).
        poll_interval:      Seconds to sleep between ticks.
        stop_loss_pct:      SL distance from entry (e.g. 0.01 = 1 %).
        take_profit_pct:    TP distance; None disables TP.
        max_position_pct:   Max capital per trade as fraction of equity.
        max_open_positions: Hard cap on concurrent open positions.
        fee_pct:            Fee per side — tracked in status, not deducted live.
        notifier:           Optional TelegramNotifier for ERROR-level alerts.
                            None (default) yields a disabled no-op notifier.
    """

    def __init__(
        self,
        strategy: Strategy,
        market_connector: ExchangeConnector,
        initial_balance: float = 10_000.0,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        ohlcv_limit: int = 500,
        poll_interval: float = 60.0,
        stop_loss_pct: float = 0.01,
        take_profit_pct: float | None = 0.02,
        max_position_pct: float = 0.02,
        max_open_positions: int = 3,
        fee_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        alert_config: AlertConfig | None = None,
        notifier: TelegramNotifier | None = None,
        signal_store_path: str = "",
    ) -> None:
        self._strategy       = strategy
        self._market         = market_connector
        self._symbol         = symbol
        self._timeframe      = timeframe
        self._ohlcv_limit    = ohlcv_limit
        self._poll_interval  = poll_interval
        self._fee_pct        = fee_pct
        self._running        = False
        self._last_day: date | None = None

        self._metrics       = SessionMetrics(initial_balance)
        self._alerts        = AlertManager(alert_config)
        self._notifier      = notifier or TelegramNotifier(bot_token="", chat_id="")
        self._signal_store: SignalStore | None = (
            SignalStore(signal_store_path) if signal_store_path else None
        )

        self._paper = PaperConnector(initial_balance=initial_balance, slippage_pct=slippage_pct)
        self._risk  = RiskManager(
            max_position_pct=max_position_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct if take_profit_pct is not None else 0.0,
            max_open_positions=max_open_positions,
        )
        self._risk.initialise_capital(initial_balance)
        self._executor = OrderExecutor(
            connector=self._paper,
            risk_manager=self._risk,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def metrics(self) -> SessionMetrics:
        """Read-only access to the session metrics accumulator."""
        return self._metrics

    @property
    def executor(self) -> OrderExecutor:
        """Read-only access to the executor (trades, open positions)."""
        return self._executor

    @property
    def risk(self) -> RiskManager:
        """Read-only access to the risk manager (state, kill switch)."""
        return self._risk

    def stop(self) -> None:
        """Signal the loop to exit after the current tick completes."""
        self._running = False

    def set_kill_switch(self, active: bool, reason: str = "") -> None:
        """Activate / deactivate the kill switch on the underlying RiskManager."""
        self._risk.set_kill_switch(active, reason)

    def set_exchange_connected(self, connected: bool) -> None:
        """Update exchange connectivity status in the RiskManager."""
        self._risk.set_exchange_connected(connected)

    def status(self) -> dict:
        """
        Return a snapshot of the current paper trading state.

        Equity here excludes unrealised P&L (we don't have a current price
        outside a tick); add unrealised manually if needed.
        """
        closed_count = sum(
            1 for t in self._executor.all_trades
            if t.status.value.startswith("closed")
        )
        equity = self._risk.state.available_balance + self._risk.state.total_exposure
        return {
            "equity":            round(equity, 2),
            "available_balance": round(self._risk.state.available_balance, 2),
            "open_positions":    self._risk.state.open_positions,
            "open_trade_ids":    [t.trade_id for t in self._executor.open_trades],
            "closed_trades":     closed_count,
            "kill_switch":       self._risk.state.kill_switch_active,
        }

    def run(self, max_ticks: int | None = None) -> None:
        """
        Start the paper trading loop.

        Blocks until:
          - *max_ticks* is reached (useful for tests / finite runs)
          - stop() is called from another thread or from within the strategy
          - KeyboardInterrupt (Ctrl-C)

        Args:
            max_ticks: Maximum number of ticks to process; None = run forever.
        """
        tick = 0
        self._running = True
        logger.info(
            "Paper trading started",
            symbol=self._symbol,
            timeframe=self._timeframe,
            balance=self._risk.state.available_balance,
        )

        try:
            while self._running:
                if max_ticks is not None and tick >= max_ticks:
                    break

                self._tick()
                tick += 1

                if self._running and (max_ticks is None or tick < max_ticks):
                    time.sleep(self._poll_interval)

        except KeyboardInterrupt:
            logger.info("Paper trading stopped by user (KeyboardInterrupt)")
        finally:
            self._running = False
            if self._signal_store is not None:
                self._signal_store.close()
            logger.info(
                "Paper trading stopped",
                ticks=tick,
                open_positions=self._risk.state.open_positions,
            )

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _tick(self) -> None:
        """Execute one data fetch → exit check → signal → entry cycle."""
        t0 = time.monotonic()
        try:
            self._check_daily_reset()

            df = self._market.fetch_ohlcv(
                self._symbol, self._timeframe, self._ohlcv_limit
            )
            if df is None or df.empty:
                logger.warning("Empty OHLCV — tick skipped", symbol=self._symbol)
                return
            if len(df) < 2:
                logger.warning(
                    "Insufficient bars for signal — tick skipped",
                    symbol=self._symbol,
                    bars=len(df),
                )
                return

            close = float(df["close"].iloc[-1])
            self._paper.set_current_price(self._symbol, close)

            # Exit check before new entries
            closed_trades = self._executor.check_exits(self._symbol, close)
            for t in closed_trades:
                self._metrics.record_trade(t)
                logger.info(
                    "Trade exited",
                    trade_id=t.trade_id,
                    symbol=self._symbol,
                    status=t.status.value,
                    pnl=round(t.realized_pnl, 2),
                )

            # Signal on the last COMPLETE bar (penultimate row)
            bar_index = len(df) - 2
            signal = self._strategy.generate_signal(df, bar_index)

            new_trade = self._executor.execute(signal, self._symbol, close)
            if new_trade is not None and new_trade.is_open:
                logger.info(
                    "Trade opened",
                    trade_id=new_trade.trade_id,
                    symbol=self._symbol,
                    side=new_trade.side,
                    entry=new_trade.entry_price,
                    sl=new_trade.sl_price,
                    tp=new_trade.tp_price,
                )
                if self._signal_store is not None:
                    try:
                        self._signal_store.record_signal(
                            signal,
                            strategy_name=type(self._strategy).__name__,
                            symbol=self._symbol,
                        )
                    except Exception as exc:
                        logger.warning("SignalStore record failed", error=str(exc))

            # Equity snapshot with unrealised P&L
            open_pnl = sum(
                t.unrealized_pnl(close) for t in self._executor.open_trades
            )
            equity = (
                self._risk.state.available_balance
                + self._risk.state.total_exposure
                + open_pnl
            )

            latency_ms = (time.monotonic() - t0) * 1000
            self._metrics.record_tick(latency_ms, equity)
            triggered = self._alerts.check_and_log(self._metrics.snapshot(), self._risk.state)
            for alert in triggered:
                if alert.level == "ERROR":
                    self._notifier.notify_alert(alert)

            logger.info(
                "Tick complete",
                symbol=self._symbol,
                close=close,
                equity=round(equity, 2),
                open_positions=self._risk.state.open_positions,
                signal=signal.type.name,
            )

        except Exception as exc:
            self._metrics.record_api_error()
            logger.error("Tick error", symbol=self._symbol, error=str(exc))

    def _check_daily_reset(self) -> None:
        """Reset daily P&L counters when the UTC date rolls over."""
        today = datetime.now(tz=timezone.utc).date()
        if self._last_day is not None and self._last_day != today:
            self._risk.reset_daily_pnl()
            logger.info("Daily P&L reset", date=str(today))
        self._last_day = today
