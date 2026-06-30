"""
Order executor — bridges strategy signals and exchange execution.

Responsibilities:
  - Translate a Signal into an OrderRequest (compute side, SL, TP, quantity).
  - Pass the request through RiskManager.validate() before touching the exchange.
  - Record every Trade in an in-memory ledger.
  - Poll open trades for SL / TP breaches via check_exits().
  - Support manual close via close_trade().

One position per symbol is enforced: a second signal on the same symbol while
a trade is open is silently ignored (the risk manager would likely block it
anyway, but we gate earlier for clarity).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from zeus.exchange.connector import ExchangeConnector
from zeus.orders.models import Trade, TradeStatus
from zeus.risk.manager import RiskManager
from zeus.risk.models import OrderRequest, OrderSide
from zeus.strategy.base import Signal, SignalType
from zeus.utils.logger import logger


class OrderExecutor:
    """
    Stateful executor.  One instance per trading session.

    Args:
        connector:       Live or paper exchange connector.
        risk_manager:    Initialised RiskManager (capital must be set before use).
        stop_loss_pct:   Fraction of entry price for SL distance (e.g. 0.01 = 1 %).
        take_profit_pct: Fraction of entry price for TP distance; None disables TP.
    """

    def __init__(
        self,
        connector: ExchangeConnector,
        risk_manager: RiskManager,
        stop_loss_pct: float = 0.01,
        take_profit_pct: float | None = 0.02,
    ) -> None:
        self._connector       = connector
        self._risk            = risk_manager
        self._sl_pct          = stop_loss_pct
        self._tp_pct          = take_profit_pct

        # trade_id → Trade
        self._trades: dict[str, Trade] = {}

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def open_trades(self) -> list[Trade]:
        """All currently open trades."""
        return [t for t in self._trades.values() if t.is_open]

    @property
    def all_trades(self) -> list[Trade]:
        """All trades (open and closed) in insertion order."""
        return list(self._trades.values())

    def execute(self, signal: Signal, symbol: str, price: float) -> Trade | None:
        """
        Attempt to open a position based on *signal*.

        Returns the new Trade on success, None if:
          - signal type is NONE
          - a position is already open on *symbol*
          - the risk manager rejects the request
          - the exchange raises an error during order placement

        Args:
            signal: Signal produced by a Strategy.
            symbol: Trading pair (e.g. "BTC/USDT").
            price:  Current market price used for sizing and bracket calculation.
        """
        if signal.type == SignalType.NONE:
            return None

        if self._has_open_trade(symbol):
            logger.debug(
                "Signal ignored — position already open",
                symbol=symbol,
                signal=signal.type.name,
            )
            return None

        side = OrderSide.BUY if signal.type == SignalType.LONG else OrderSide.SELL
        sl_price, tp_price = self._bracket_prices(side, price)

        # Propose full equity allocation — risk manager will cap it
        equity = self._risk.state.equity
        proposed_qty = equity / price if price > 0 else 0.0

        request = OrderRequest(
            symbol=symbol,
            side=side,
            quantity=proposed_qty,
            entry_price=price,
            stop_loss=sl_price,
            take_profit=tp_price,
            signal_reason=signal.reason,
        )

        validation = self._risk.validate(request)
        if not validation.approved:
            trade = self._record_rejected(signal, symbol, side, price, sl_price, tp_price, request)
            return trade

        try:
            order = self._connector.place_market_order(
                symbol=symbol,
                side=side.value,
                quantity=validation.computed_quantity,
            )
        except Exception as exc:
            logger.error(
                "Exchange error during entry",
                symbol=symbol,
                side=side.value,
                error=str(exc),
            )
            return self._record_failed(signal, symbol, side, price, sl_price, tp_price, request)

        position_value = validation.computed_quantity * price
        self._risk.update_on_fill(
            realized_pnl=0.0,
            position_value=position_value,
            opened=True,
        )

        fill_price = order.average if order.average is not None else price
        trade = Trade(
            trade_id=self._new_trade_id(),
            symbol=symbol,
            side=side.value,
            quantity=validation.computed_quantity,
            entry_price=fill_price,
            sl_price=sl_price,
            tp_price=tp_price,
            entry_order=order,
            signal_reason=signal.reason,
            status=TradeStatus.OPEN,
        )
        self._trades[trade.trade_id] = trade

        logger.info(
            "Trade opened",
            trade_id=trade.trade_id,
            symbol=symbol,
            side=side.value,
            quantity=trade.quantity,
            entry=fill_price,
            sl=sl_price,
            tp=tp_price,
        )
        return trade

    def check_exits(self, symbol: str, current_price: float) -> list[Trade]:
        """
        Check all open trades on *symbol* for SL / TP breaches.

        Returns the list of trades that were closed by this call.
        Trades are mutated in-place (status, exit_order, closed_at, realized_pnl).
        """
        closed: list[Trade] = []
        for trade in self.open_trades:
            if trade.symbol != symbol:
                continue
            exit_status = self._check_exit_condition(trade, current_price)
            if exit_status is not None:
                self._close_trade(trade, current_price, exit_status)
                closed.append(trade)
        return closed

    def close_trade(self, trade_id: str, current_price: float) -> Trade | None:
        """
        Manually close a trade at *current_price*.

        Returns the Trade (now CLOSED_MAN) or None if trade_id is unknown / already closed.
        """
        trade = self._trades.get(trade_id)
        if trade is None or not trade.is_open:
            return None
        self._close_trade(trade, current_price, TradeStatus.CLOSED_MAN)
        return trade

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _bracket_prices(
        self, side: OrderSide, entry: float
    ) -> tuple[float, float | None]:
        """Return (sl_price, tp_price) for a given side and entry."""
        if side == OrderSide.BUY:
            sl = entry * (1.0 - self._sl_pct)
            tp = (entry * (1.0 + self._tp_pct)) if self._tp_pct is not None else None
        else:
            sl = entry * (1.0 + self._sl_pct)
            tp = (entry * (1.0 - self._tp_pct)) if self._tp_pct is not None else None
        return sl, tp

    def _has_open_trade(self, symbol: str) -> bool:
        return any(t.symbol == symbol and t.is_open for t in self._trades.values())

    def _check_exit_condition(
        self, trade: Trade, price: float
    ) -> TradeStatus | None:
        """Return the exit status if *price* triggers an exit, else None."""
        if trade.is_long:
            if price <= trade.sl_price:
                return TradeStatus.CLOSED_SL
            if trade.tp_price is not None and price >= trade.tp_price:
                return TradeStatus.CLOSED_TP
        else:
            if price >= trade.sl_price:
                return TradeStatus.CLOSED_SL
            if trade.tp_price is not None and price <= trade.tp_price:
                return TradeStatus.CLOSED_TP
        return None

    def _close_trade(
        self, trade: Trade, exit_price: float, status: TradeStatus
    ) -> None:
        """Place exit order, compute PnL, update risk state, mutate trade."""
        try:
            exit_order = self._connector.place_market_order(
                symbol=trade.symbol,
                side="sell" if trade.is_long else "buy",
                quantity=trade.quantity,
            )
            fill_price = exit_order.average if exit_order.average is not None else exit_price
        except Exception as exc:
            logger.error(
                "Exchange error during exit",
                trade_id=trade.trade_id,
                symbol=trade.symbol,
                error=str(exc),
            )
            exit_order = None
            fill_price = exit_price

        pnl = trade.unrealized_pnl(fill_price)
        position_value = trade.quantity * trade.entry_price

        self._risk.update_on_fill(
            realized_pnl=pnl,
            position_value=position_value,
            opened=False,
        )

        trade.status       = status
        trade.exit_order   = exit_order
        trade.closed_at    = datetime.now(tz=timezone.utc)
        trade.realized_pnl = pnl

        logger.info(
            "Trade closed",
            trade_id=trade.trade_id,
            symbol=trade.symbol,
            status=status.value,
            exit_price=fill_price,
            pnl=pnl,
        )

    def _record_rejected(
        self,
        signal: Signal,
        symbol: str,
        side: OrderSide,
        price: float,
        sl_price: float,
        tp_price: float | None,
        request: OrderRequest,
    ) -> Trade:
        from zeus.exchange.connector import OrderResult

        dummy_order = OrderResult(
            order_id="rejected",
            symbol=symbol,
            side=side.value,
            quantity=request.quantity,
            filled=0.0,
            price=price,
            average=None,
            status="rejected",
            timestamp=datetime.now(tz=timezone.utc),
        )
        trade = Trade(
            trade_id=self._new_trade_id(),
            symbol=symbol,
            side=side.value,
            quantity=request.quantity,
            entry_price=price,
            sl_price=sl_price,
            tp_price=tp_price,
            entry_order=dummy_order,
            signal_reason=signal.reason,
            status=TradeStatus.REJECTED,
        )
        self._trades[trade.trade_id] = trade
        return trade

    def _record_failed(
        self,
        signal: Signal,
        symbol: str,
        side: OrderSide,
        price: float,
        sl_price: float,
        tp_price: float | None,
        request: OrderRequest,
    ) -> Trade:
        from zeus.exchange.connector import OrderResult

        dummy_order = OrderResult(
            order_id="failed",
            symbol=symbol,
            side=side.value,
            quantity=request.quantity,
            filled=0.0,
            price=price,
            average=None,
            status="failed",
            timestamp=datetime.now(tz=timezone.utc),
        )
        trade = Trade(
            trade_id=self._new_trade_id(),
            symbol=symbol,
            side=side.value,
            quantity=request.quantity,
            entry_price=price,
            sl_price=sl_price,
            tp_price=tp_price,
            entry_order=dummy_order,
            signal_reason=signal.reason,
            status=TradeStatus.FAILED,
        )
        self._trades[trade.trade_id] = trade
        return trade

    @staticmethod
    def _new_trade_id() -> str:
        return str(uuid.uuid4())
