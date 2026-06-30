"""
Backtest engine — replays OHLCV data bar by bar through the full trading pipeline.

Pipeline per bar:
    1. Intrabar SL/TP check (bar high/low).
    2. Strategy signal at bar close.
    3. OrderExecutor.execute() → risk-validated entry.
    4. Equity snapshot appended to the curve.

After the last bar, any remaining open positions are closed at the last close price.

Fees (fee_pct per side) are not deducted from live balance; they are tracked
separately in BacktestResult so the raw equity curve and fee impact are both visible.
"""
from __future__ import annotations

import pandas as pd

from zeus.backtest.report import BacktestResult
from zeus.exchange.paper import PaperConnector
from zeus.orders.executor import OrderExecutor
from zeus.risk.manager import RiskManager
from zeus.strategy.base import Strategy
from zeus.utils.logger import logger


class BacktestEngine:
    """
    Stateless driver — creates fresh connectors, risk state, and executor on every run().

    Args:
        strategy:         Any Strategy subclass.
        initial_balance:  Starting capital in quote currency.
        fee_pct:          Exchange fee applied per side (e.g. 0.001 = 0.1 %).
        slippage_pct:     Market-order slippage applied by PaperConnector.
        stop_loss_pct:    SL distance from entry as a fraction (e.g. 0.01 = 1 %).
        take_profit_pct:  TP distance from entry; None = no TP.
        max_position_pct: Maximum capital per trade as a fraction of equity.
        max_open_positions: Hard cap on concurrent open trades.
    """

    def __init__(
        self,
        strategy: Strategy,
        initial_balance: float = 10_000.0,
        fee_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        stop_loss_pct: float = 0.01,
        take_profit_pct: float | None = 0.02,
        max_position_pct: float = 0.02,
        max_open_positions: int = 3,
    ) -> None:
        self._strategy          = strategy
        self._initial_balance   = initial_balance
        self._fee_pct           = fee_pct
        self._slippage_pct      = slippage_pct
        self._sl_pct            = stop_loss_pct
        self._tp_pct            = take_profit_pct
        self._max_position_pct  = max_position_pct
        self._max_open_positions = max_open_positions

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(self, df: pd.DataFrame, symbol: str = "BTC/USDT") -> BacktestResult:
        """
        Execute the backtest on *df* for *symbol*.

        Args:
            df:     OHLCV DataFrame with at least a 'close' column.
                    'high' and 'low' columns enable intrabar SL/TP checking.
            symbol: Trading pair label forwarded to the executor.

        Returns:
            BacktestResult with all trades and the equity curve.
        """
        connector, risk, executor = self._build_components()
        equity_curve: list[float] = [self._initial_balance]

        for bar_index in range(len(df)):
            close = float(df["close"].iloc[bar_index])
            connector.set_current_price(symbol, close)

            # 1. Intrabar exits (uses bar high/low when available)
            self._process_intrabar_exits(executor, connector, df, bar_index, symbol)

            # 2. Signal → entry
            signal = self._strategy.generate_signal(df, bar_index)
            executor.execute(signal, symbol, close)

            # 3. Equity snapshot (realized cash + locked exposure + unrealised P&L)
            open_pnl = sum(t.unrealized_pnl(close) for t in executor.open_trades)
            equity = risk.state.available_balance + risk.state.total_exposure + open_pnl
            equity_curve.append(equity)

        # Force-close any open positions at the last close
        if len(df) > 0:
            last_close = float(df["close"].iloc[-1])
            connector.set_current_price(symbol, last_close)
            for trade in list(executor.open_trades):
                executor.close_trade(trade.trade_id, last_close)

        logger.info(
            "Backtest complete",
            symbol=symbol,
            bars=len(df),
            trades=len([t for t in executor.all_trades if t.status.value.startswith("closed")]),
            final_equity=round(equity_curve[-1], 2),
        )

        return BacktestResult(
            trades=executor.all_trades,
            equity_curve=equity_curve,
            initial_balance=self._initial_balance,
            fee_pct=self._fee_pct,
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _build_components(
        self,
    ) -> tuple[PaperConnector, RiskManager, OrderExecutor]:
        """Instantiate fresh connectors for an isolated run."""
        connector = PaperConnector(
            initial_balance=self._initial_balance,
            slippage_pct=self._slippage_pct,
        )
        risk = RiskManager(
            max_position_pct=self._max_position_pct,
            stop_loss_pct=self._sl_pct,
            take_profit_pct=self._tp_pct if self._tp_pct is not None else 0.0,
            max_daily_loss_pct=0.99,   # disabled — backtest runs across days
            max_drawdown_pct=0.99,     # disabled — let equity go negative
            max_exposure_pct=0.95,     # near-full exposure allowed
            max_open_positions=self._max_open_positions,
        )
        risk.initialise_capital(self._initial_balance)
        executor = OrderExecutor(
            connector=connector,
            risk_manager=risk,
            stop_loss_pct=self._sl_pct,
            take_profit_pct=self._tp_pct,
        )
        return connector, risk, executor

    def _process_intrabar_exits(
        self,
        executor: OrderExecutor,
        connector: PaperConnector,
        df: pd.DataFrame,
        bar_index: int,
        symbol: str,
    ) -> None:
        """
        Check whether the bar's high or low crossed any open trade's SL/TP.

        For a long position we assume worst-case ordering: if both SL (low) and
        TP (high) are breached in the same bar, the SL is triggered first.
        """
        if "high" not in df.columns or "low" not in df.columns:
            return

        bar_high = float(df["high"].iloc[bar_index])
        bar_low  = float(df["low"].iloc[bar_index])

        for trade in list(executor.open_trades):
            if trade.symbol != symbol or not trade.is_open:
                continue

            exit_price: float | None = None

            if trade.is_long:
                if bar_low <= trade.sl_price:
                    exit_price = trade.sl_price
                elif trade.tp_price is not None and bar_high >= trade.tp_price:
                    exit_price = trade.tp_price
            else:
                if bar_high >= trade.sl_price:
                    exit_price = trade.sl_price
                elif trade.tp_price is not None and bar_low <= trade.tp_price:
                    exit_price = trade.tp_price

            if exit_price is not None:
                connector.set_current_price(symbol, exit_price)
                executor.check_exits(symbol, exit_price)
