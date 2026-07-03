"""
Zeus trading bot — CLI entry point.

Reads all configuration from environment variables (or a .env file).
Only PAPER mode is supported until live paper-trading validation is complete.

Usage:
    python -m zeus.main
    ZEUS_SYMBOL=XAUUSD ZEUS_TIMEFRAME=1h ZEUS_CONNECTOR=mt5 zeus

Key environment variables (full list in zeus/config.py):
    ZEUS_MODE                 paper | live        (default: paper)
    ZEUS_CONNECTOR            ccxt | mt5          (default: ccxt)
    ZEUS_EXCHANGE             ccxt exchange id    (default: binance, ccxt only)
    ZEUS_MT5_LOGIN            MT5 account number  (mt5 only)
    ZEUS_MT5_PASSWORD         MT5 password        (mt5 only)
    ZEUS_MT5_SERVER           MT5 broker server   (mt5 only)
    ZEUS_SYMBOL               trading pair        (default: BTC/USDT)
    ZEUS_TIMEFRAME            OHLCV timeframe     (default: 1h)
    ZEUS_PAPER_BALANCE        starting capital    (default: 10000)
    ZEUS_STOP_LOSS_PCT        SL distance         (default: 0.01)
    ZEUS_TAKE_PROFIT_PCT      TP distance         (default: 0.02)
    ZEUS_MAX_POSITION_PCT     max size / equity   (default: 0.02)
    ZEUS_MAX_OPEN_POSITIONS   concurrent trades   (default: 3)
    ZEUS_POLL_INTERVAL_SECONDS seconds per tick   (default: 60)
    ZEUS_OHLCV_LIMIT          bars per fetch      (default: 500)
    ZEUS_SMC_MIN_SCORE        confluence min      (default: 4.0)
    ZEUS_LOG_LEVEL            DEBUG/INFO/WARNING  (default: INFO)
    ZEUS_TELEGRAM_BOT_TOKEN   Telegram bot token  (optional; unset = disabled)
    ZEUS_TELEGRAM_CHAT_ID     Telegram chat id    (optional; unset = disabled)
"""
from __future__ import annotations

import signal
import sys

from zeus.config import ConnectorType, StrategyType, get_settings
from zeus.exchange.factory import create_market_connector
from zeus.monitoring.telegram import TelegramNotifier
from zeus.paper.engine import PaperEngine
from zeus.strategy.base import Strategy
from zeus.strategy.smc_strategy import SMCStrategy
from zeus.strategy.supply_demand.sd_strategy import SDStrategy
from zeus.utils.logger import logger, setup_logger


def _make_strategy(settings) -> Strategy:
    """Instantiate the strategy requested by ZEUS_STRATEGY env var."""
    stype = settings.strategy
    if stype == StrategyType.SMC:
        return SMCStrategy(min_score=settings.smc_min_score)
    if stype == StrategyType.SD:
        return SDStrategy()
    raise ValueError(f"Unknown strategy type: {stype}")


def main() -> None:
    settings = get_settings()
    setup_logger(settings.log_level.value)

    if settings.is_live:
        logger.error(
            "Live mode is not yet enabled — "
            "complete paper trading validation before deploying live.",
            mode=settings.mode.value,
        )
        sys.exit(1)

    logger.info(
        "Zeus paper trading bot starting",
        mode=settings.mode.value,
        connector=settings.connector.value,
        symbol=settings.symbol,
        timeframe=settings.timeframe,
        balance=settings.paper_balance,
        stop_loss_pct=settings.stop_loss_pct,
        take_profit_pct=settings.take_profit_pct,
        poll_interval_s=settings.poll_interval_seconds,
    )

    market_connector = create_market_connector(settings)
    strategy = _make_strategy(settings)
    logger.info("Strategy selected", strategy=settings.strategy.value)

    # MT5 requires symbol names without slash ("XAUUSD", not "XAU/USD")
    symbol = (
        settings.mt5_symbol
        if settings.connector == ConnectorType.MT5
        else settings.symbol
    )

    notifier = TelegramNotifier(
        bot_token=settings.telegram_bot_token,
        chat_id=settings.telegram_chat_id,
    )
    if notifier.enabled:
        logger.info("Telegram notifications enabled")

    engine = PaperEngine(
        strategy=strategy,
        market_connector=market_connector,
        initial_balance=settings.paper_balance,
        symbol=symbol,
        timeframe=settings.timeframe,
        ohlcv_limit=settings.ohlcv_limit,
        poll_interval=settings.poll_interval_seconds,
        stop_loss_pct=settings.stop_loss_pct,
        take_profit_pct=settings.take_profit_pct,
        max_position_pct=settings.max_position_pct,
        max_open_positions=settings.max_open_positions,
        notifier=notifier,
    )

    def _shutdown(sig, frame):
        logger.info("Shutdown signal received — stopping after current tick", signal=sig)
        engine.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        engine.run()
    finally:
        engine.metrics.log_summary()
        logger.info("Zeus paper trading bot stopped")
        market_connector.close()


if __name__ == "__main__":
    main()
