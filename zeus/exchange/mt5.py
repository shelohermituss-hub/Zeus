"""
MetaTrader 5 exchange connector.

Provides OHLCV data and order execution through the MetaTrader5 Python
library. The library is Windows-only and requires a running MT5 terminal.
On Linux/CI the import silently fails; instantiation raises RuntimeError
so the rest of the codebase can import this module safely on any platform.

Quantity convention
-------------------
All quantities flowing in/out of this connector are in **base currency units**
(e.g. troy ounces for XAUUSD, contract units for US30), consistent with the
rest of the system.  Lot conversion (base_units / contract_size) happens
internally in the order methods and is invisible to callers.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from zeus.exchange.connector import ExchangeConnector, OrderResult
from zeus.utils.logger import logger

try:
    import MetaTrader5 as mt5  # type: ignore[import]
    _MT5_AVAILABLE = True
except ImportError:
    mt5 = None  # type: ignore[assignment]
    _MT5_AVAILABLE = False


# Map CCXT-style timeframe strings to MT5 TIMEFRAME attribute names.
# Resolved via getattr(mt5, name) at call time so the real constants are used
# when MT5 is available and mock values are used in tests.
_TIMEFRAME_ATTRS: dict[str, str] = {
    "1m":  "TIMEFRAME_M1",
    "5m":  "TIMEFRAME_M5",
    "15m": "TIMEFRAME_M15",
    "30m": "TIMEFRAME_M30",
    "1h":  "TIMEFRAME_H1",
    "2h":  "TIMEFRAME_H2",
    "4h":  "TIMEFRAME_H4",
    "6h":  "TIMEFRAME_H6",
    "12h": "TIMEFRAME_H12",
    "1d":  "TIMEFRAME_D1",
    "1w":  "TIMEFRAME_W1",
}


def _resolve_timeframe(timeframe: str):
    """Return the MT5 TIMEFRAME constant for a CCXT-style timeframe string."""
    attr = _TIMEFRAME_ATTRS.get(timeframe)
    if attr is None:
        raise ValueError(
            f"Unsupported timeframe '{timeframe}' for MT5 connector. "
            f"Supported: {sorted(_TIMEFRAME_ATTRS)}"
        )
    return getattr(mt5, attr)


def _lots_to_units(lots: float, contract_size: float) -> float:
    return lots * contract_size


def _units_to_lots(
    units: float,
    contract_size: float,
    volume_min: float,
    volume_max: float,
    volume_step: float,
) -> float:
    """Convert base-currency units to lots, clamped and rounded to symbol limits."""
    lots = units / contract_size
    lots = max(volume_min, min(volume_max, lots))
    if volume_step > 0:
        lots = round(lots / volume_step) * volume_step
    return lots


class MT5Connector(ExchangeConnector):
    """
    MetaTrader 5 connector for Forex and CFD instruments.

    Requires a running MT5 terminal and the MetaTrader5 Python package
    (Windows only). Raises RuntimeError at instantiation on unsupported
    platforms rather than at import time, so the rest of the codebase
    stays importable everywhere.

    Args:
        login:    MT5 account number.
        password: MT5 account password.
        server:   MT5 broker server name (e.g. "ICMarketsEU-Demo").
        deviation: Maximum price deviation in points for market orders.
    """

    def __init__(
        self,
        login:     int,
        password:  str,
        server:    str,
        deviation: int = 20,
    ) -> None:
        if not _MT5_AVAILABLE:
            raise RuntimeError(
                "MetaTrader5 library is not available on this platform. "
                "MT5 connector requires Windows with MetaTrader5 installed."
            )

        self._deviation = deviation

        if not mt5.initialize(login=login, password=password, server=server):
            error = mt5.last_error()
            raise RuntimeError(f"MT5 initialization failed: {error}")

        logger.info("MT5 connector initialized", login=login, server=server)

    # ------------------------------------------------------------------ #
    # ExchangeConnector interface                                          #
    # ------------------------------------------------------------------ #

    def fetch_ohlcv(
        self,
        symbol:    str,
        timeframe: str,
        limit:     int = 500,
    ) -> pd.DataFrame:
        tf = _resolve_timeframe(timeframe)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, limit)

        if rates is None or len(rates) == 0:
            logger.warning("MT5 returned no OHLCV data", symbol=symbol, timeframe=timeframe)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rates)
        df.index = pd.to_datetime(df["time"], unit="s", utc=True)
        df.index.name = None
        return (
            df[["open", "high", "low", "close", "tick_volume"]]
            .rename(columns={"tick_volume": "volume"})
            .astype(float)
        )

    def fetch_balance(self) -> float:
        info = mt5.account_info()
        if info is None:
            raise RuntimeError("Cannot fetch account info from MT5")
        return float(info.margin_free)

    def place_market_order(
        self,
        symbol:   str,
        side:     str,
        quantity: float,
    ) -> OrderResult:
        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            raise RuntimeError(f"Symbol {symbol!r} not found in MT5")

        lots = _units_to_lots(
            quantity,
            sym_info.trade_contract_size,
            sym_info.volume_min,
            sym_info.volume_max,
            sym_info.volume_step,
        )

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"No tick data for {symbol!r}")

        order_type = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL
        price = tick.ask if side == "buy" else tick.bid

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       lots,
            "type":         order_type,
            "price":        price,
            "deviation":    self._deviation,
            "magic":        0,
            "comment":      "zeus",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        logger.info("Placing MT5 market order",
                    symbol=symbol, side=side, quantity=quantity, lots=lots)

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = result.retcode if result is not None else "None"
            comment = result.comment if result is not None else "no response"
            raise RuntimeError(
                f"Market order rejected: retcode={retcode}, {comment}"
            )

        filled_units = _lots_to_units(result.volume, sym_info.trade_contract_size)

        logger.info("MT5 market order filled",
                    order_id=result.order, deal=result.deal,
                    filled_lots=result.volume, price=result.price)

        return OrderResult(
            order_id  = str(result.order),
            symbol    = symbol,
            side      = side,
            quantity  = quantity,
            filled    = filled_units,
            price     = None,
            average   = float(result.price),
            status    = "closed",
            timestamp = datetime.now(timezone.utc),
            raw       = {
                "retcode": result.retcode,
                "order":   result.order,
                "deal":    result.deal,
                "volume":  result.volume,
                "price":   result.price,
                "comment": result.comment,
            },
        )

    def place_limit_order(
        self,
        symbol:   str,
        side:     str,
        quantity: float,
        price:    float,
    ) -> OrderResult:
        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            raise RuntimeError(f"Symbol {symbol!r} not found in MT5")

        lots = _units_to_lots(
            quantity,
            sym_info.trade_contract_size,
            sym_info.volume_min,
            sym_info.volume_max,
            sym_info.volume_step,
        )

        order_type = (
            mt5.ORDER_TYPE_BUY_LIMIT if side == "buy" else mt5.ORDER_TYPE_SELL_LIMIT
        )

        request = {
            "action":       mt5.TRADE_ACTION_PENDING,
            "symbol":       symbol,
            "volume":       lots,
            "type":         order_type,
            "price":        price,
            "deviation":    self._deviation,
            "magic":        0,
            "comment":      "zeus",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }

        logger.info("Placing MT5 limit order",
                    symbol=symbol, side=side, quantity=quantity, price=price)

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = result.retcode if result is not None else "None"
            comment = result.comment if result is not None else "no response"
            raise RuntimeError(
                f"Limit order rejected: retcode={retcode}, {comment}"
            )

        logger.info("MT5 limit order placed", order_id=result.order)

        return OrderResult(
            order_id  = str(result.order),
            symbol    = symbol,
            side      = side,
            quantity  = quantity,
            filled    = 0.0,
            price     = price,
            average   = None,
            status    = "open",
            timestamp = datetime.now(timezone.utc),
            raw       = {
                "retcode": result.retcode,
                "order":   result.order,
                "comment": result.comment,
            },
        )

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order":  int(order_id),
        }
        result = mt5.order_send(request)
        if result is None:
            logger.warning("MT5 cancel returned None", order_id=order_id)
            return False
        success = result.retcode == mt5.TRADE_RETCODE_DONE
        if not success:
            logger.warning("MT5 cancel failed",
                           order_id=order_id, retcode=result.retcode)
        return success

    def fetch_order(self, order_id: str, symbol: str) -> OrderResult:
        ticket = int(order_id)

        # Pending orders
        pending = mt5.orders_get(ticket=ticket)
        if pending:
            o = pending[0]
            sym_info = mt5.symbol_info(symbol)
            contract_size = sym_info.trade_contract_size if sym_info else 1.0
            side = "buy" if o.type in (mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_BUY_LIMIT) else "sell"
            return OrderResult(
                order_id  = order_id,
                symbol    = symbol,
                side      = side,
                quantity  = _lots_to_units(o.volume_initial, contract_size),
                filled    = 0.0,
                price     = float(o.price_open),
                average   = None,
                status    = "open",
                timestamp = datetime.fromtimestamp(o.time_setup, tz=timezone.utc),
                raw       = {"ticket": o.ticket, "type": o.type},
            )

        # Historical orders (filled or cancelled)
        history = mt5.history_orders_get(ticket=ticket)
        if history:
            o = history[0]
            sym_info = mt5.symbol_info(symbol)
            contract_size = sym_info.trade_contract_size if sym_info else 1.0
            side = "buy" if o.type in (mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_BUY_LIMIT) else "sell"

            if o.state == mt5.ORDER_STATE_FILLED:
                status = "closed"
            elif o.state == mt5.ORDER_STATE_CANCELED:
                status = "cancelled"
            else:
                status = "open"

            deals = mt5.history_deals_get(order=ticket)
            average = float(deals[0].price) if deals else None
            filled  = _lots_to_units(deals[0].volume, contract_size) if deals else 0.0

            return OrderResult(
                order_id  = order_id,
                symbol    = symbol,
                side      = side,
                quantity  = _lots_to_units(o.volume_initial, contract_size),
                filled    = filled,
                price     = float(o.price_open) if o.price_open else None,
                average   = average,
                status    = status,
                timestamp = datetime.fromtimestamp(o.time_setup, tz=timezone.utc),
                raw       = {"ticket": o.ticket, "type": o.type, "state": o.state},
            )

        raise RuntimeError(f"Order {order_id!r} not found in MT5")

    def fetch_open_orders(self, symbol: str) -> list[OrderResult]:
        orders = mt5.orders_get(symbol=symbol)
        if not orders:
            return []

        sym_info = mt5.symbol_info(symbol)
        contract_size = sym_info.trade_contract_size if sym_info else 1.0

        results = []
        for o in orders:
            buy_types = (mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP)
            side = "buy" if o.type in buy_types else "sell"
            results.append(OrderResult(
                order_id  = str(o.ticket),
                symbol    = symbol,
                side      = side,
                quantity  = _lots_to_units(o.volume_initial, contract_size),
                filled    = 0.0,
                price     = float(o.price_open),
                average   = None,
                status    = "open",
                timestamp = datetime.fromtimestamp(o.time_setup, tz=timezone.utc),
                raw       = {"ticket": o.ticket, "type": o.type},
            ))
        return results

    def is_connected(self) -> bool:
        info = mt5.terminal_info()
        if info is None:
            return False
        return bool(info.connected)

    def close(self) -> None:
        mt5.shutdown()
        logger.info("MT5 connector closed")

    # ------------------------------------------------------------------ #
    # MT5-specific utilities                                               #
    # ------------------------------------------------------------------ #

    def lots_from_notional(self, symbol: str, notional_usd: float) -> float:
        """
        Compute the lot size for a given notional USD value.

        Useful for RiskManager integration: converts the max_position_value
        computed in USD into the correct lot size for the symbol.
        """
        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            raise RuntimeError(f"Symbol {symbol!r} not found in MT5")

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"No tick data for {symbol!r}")

        price = float(tick.ask)
        contract_size = sym_info.trade_contract_size
        value_per_lot = contract_size * price

        lots = notional_usd / value_per_lot
        return _units_to_lots(
            lots * contract_size,
            contract_size,
            sym_info.volume_min,
            sym_info.volume_max,
            sym_info.volume_step,
        )
