"""
Exchange connector factory.

Builds the correct ExchangeConnector from Settings so application
start-up code never has to import ccxt directly or touch credentials.

API keys come from environment variables via Settings — they are never
logged or hard-coded here.
"""
from __future__ import annotations

import ccxt

from zeus.config import Settings
from zeus.exchange.connector import ExchangeConnector
from zeus.exchange.live import LiveConnector
from zeus.exchange.paper import PaperConnector


def create_connector(settings: Settings) -> ExchangeConnector:
    """
    Build the appropriate ExchangeConnector from application Settings.

    Paper mode:
        Returns a PaperConnector pre-loaded with the configured
        paper_balance.  A read-only CCXT exchange (no API keys) is
        attached for market-data calls (fetch_ohlcv).

    Live mode:
        Returns a LiveConnector wrapping a fully-authenticated CCXT
        exchange.  Requires ZEUS_API_KEY and ZEUS_API_SECRET to be set.

    Args:
        settings: Validated Settings instance (from config.get_settings()).

    Returns:
        ExchangeConnector ready for use by the execution layer.
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
