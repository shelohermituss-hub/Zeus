"""
Run a Zeus backtest using historical OHLCV pulled directly from a running
MT5 terminal.

This is NOT a native MT5 Expert Advisor. Zeus stays 100% Python so the
exact same strategy and risk code runs in backtest, paper, and live —
a non-negotiable rule in CLAUDE.md ("le moteur de décision doit être
identique au live"). Porting the strategy to MQL5 would create a second
implementation that can silently diverge from the Python one.

Instead, this script only swaps the data source: BacktestEngine (fees,
slippage, intrabar SL/TP already modelled) replays bars fetched live
from the MT5 terminal via MT5Connector, instead of CCXT history.

Requirements:
    - Windows host running a connected MT5 terminal.
    - MetaTrader5 package installed: pip install -e ".[mt5]"
    - ZEUS_CONNECTOR=mt5 and MT5 credentials set (see zeus/main.py header
      for the full environment variable list).

Usage:
    ZEUS_CONNECTOR=mt5 ZEUS_SYMBOL=XAUUSD ZEUS_TIMEFRAME=1h \
    ZEUS_MT5_LOGIN=12345678 ZEUS_MT5_PASSWORD=secret ZEUS_MT5_SERVER=Demo-Server \
    ZEUS_OHLCV_LIMIT=5000 \
    python -m zeus.backtest.run_mt5

Bar count is bounded by ZEUS_OHLCV_LIMIT and by how much history the MT5
terminal/broker keeps available — increase the terminal's "Max bars in
chart" setting if fewer bars are returned than requested.
"""
from __future__ import annotations

import sys

from zeus.backtest.engine import BacktestEngine
from zeus.config import ConnectorType, get_settings
from zeus.exchange.factory import create_market_connector
from zeus.strategy.smc_strategy import SMCStrategy
from zeus.utils.logger import logger, setup_logger


def main() -> None:
    settings = get_settings()
    setup_logger(settings.log_level.value)

    if settings.connector != ConnectorType.MT5:
        logger.error(
            "ZEUS_CONNECTOR must be 'mt5' to run an MT5-sourced backtest",
            connector=settings.connector.value,
        )
        sys.exit(1)

    market = create_market_connector(settings)
    symbol = settings.mt5_symbol

    try:
        df = market.fetch_ohlcv(symbol, settings.timeframe, settings.ohlcv_limit)
        if df is None or df.empty:
            logger.error(
                "MT5 returned no historical data — check symbol, timeframe, "
                "and terminal history depth",
                symbol=symbol,
                timeframe=settings.timeframe,
            )
            sys.exit(1)

        logger.info(
            "Fetched MT5 history for backtest",
            symbol=symbol,
            timeframe=settings.timeframe,
            bars=len(df),
        )

        strategy = SMCStrategy(min_score=settings.smc_min_score)
        engine = BacktestEngine(
            strategy=strategy,
            initial_balance=settings.paper_balance,
            stop_loss_pct=settings.stop_loss_pct,
            take_profit_pct=settings.take_profit_pct,
            max_position_pct=settings.max_position_pct,
            max_open_positions=settings.max_open_positions,
        )
        result = engine.run(df, symbol=symbol)

        logger.info("MT5 backtest complete", **result.summary())
    finally:
        market.close()


if __name__ == "__main__":
    main()
