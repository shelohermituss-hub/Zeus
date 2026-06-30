"""
Exchange connector factory.

Builds the correct ExchangeConnector from Settings so application
start-up code never has to import ccxt or MetaTrader5 directly or
touch credentials.

API keys come from environment variables via Settings — they are never
logged or hard-coded here.
"""
from __future__ import annotations

import ccxt

from zeus.config import ConnectorType, Settings
from zeus.exchange.connector import ExchangeConnector
from zeus.exchange.live import LiveConnector
from zeus.exchange.paper import PaperConnector


def create_connector(settings: Settings) -> ExchangeConnector:
    """
    Build the full (execution + market-data) connector for the given mode.

    Paper mode:
        Returns a PaperConnector pre-loaded with the configured
        paper_balance. For CCXT, a read-only exchange is attached for
        market-data calls.

    Live mode:
        Returns a LiveConnector wrapping a fully-authenticated CCXT
        exchange. Requires ZEUS_API_KEY and ZEUS_API_SECRET.

    MT5 mode is not supported through this function — use
    create_market_connector() for the market-data side and PaperConnector
    for paper-trading execution.
    """
    exchange_cls = getattr(ccxt, settings.exchange)

    if settings.is_paper:
        ccxt_ex = exchange_cls({"enableRateLimit": True})
        return PaperConnector(
            initial_balance=settings.paper_balance,
            ccxt_exchange=ccxt_ex,
        )

    ccxt_ex = exchange_cls({
        "apiKey":          settings.api_key,
        "secret":          settings.api_secret,
        "enableRateLimit": True,
    })
    return LiveConnector(exchange=ccxt_ex)


def create_market_connector(settings: Settings) -> ExchangeConnector:
    """
    Build the market-data connector based on the configured connector type.

    In paper mode the returned connector is used only for fetch_ohlcv.
    In live mode it is used for both market data and order execution
    (MT5 handles both natively; CCXT uses LiveConnector).

    Args:
        settings: Validated Settings instance.

    Returns:
        ExchangeConnector ready for market-data calls.
    """
    if settings.connector == ConnectorType.MT5:
        from zeus.exchange.mt5 import MT5Connector
        return MT5Connector(
            login=settings.mt5_login,
            password=settings.mt5_password,
            server=settings.mt5_server,
        )

    # Default: CCXT — read-only (no API keys needed for public endpoints)
    exchange_cls = getattr(ccxt, settings.exchange)
    ccxt_ex = exchange_cls({"enableRateLimit": True})
    return LiveConnector(exchange=ccxt_ex)
