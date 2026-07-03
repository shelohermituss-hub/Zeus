"""
Entry point for SD strategy V_FINAL_B paper trading.

Usage
-----
    python -m zeus.paper.run_sd_paper

Environment variables
---------------------
    ZEUS_TELEGRAM_BOT_TOKEN   Telegram bot token (optional)
    ZEUS_TELEGRAM_CHAT_ID     Telegram chat ID (optional)
    ZEUS_INITIAL_BALANCE      Paper account balance (default 10000)
    ZEUS_RISK_PCT             Risk per trade as a fraction (default 0.01 = 1%)
    ZEUS_POLL_INTERVAL        Seconds between ticks (default 60)

MT5 connection
--------------
    The engine requires MetaTrader 5 to be running locally.
    Set ZEUS_MT5_LOGIN, ZEUS_MT5_PASSWORD, ZEUS_MT5_SERVER in environment.
"""
from __future__ import annotations

import os
import sys

from zeus.exchange.mt5 import MT5Connector
from zeus.monitoring.telegram import TelegramNotifier
from zeus.paper.sd_paper_engine import SDPaperEngine
from zeus.utils.logger import logger


def main() -> None:
    bot_token       = os.environ.get("ZEUS_TELEGRAM_BOT_TOKEN", "")
    chat_id         = os.environ.get("ZEUS_TELEGRAM_CHAT_ID", "")
    initial_balance = float(os.environ.get("ZEUS_INITIAL_BALANCE", "10000"))
    risk_pct        = float(os.environ.get("ZEUS_RISK_PCT", "0.01"))
    poll_interval   = float(os.environ.get("ZEUS_POLL_INTERVAL", "60"))

    mt5_login    = os.environ.get("ZEUS_MT5_LOGIN")
    mt5_password = os.environ.get("ZEUS_MT5_PASSWORD")
    mt5_server   = os.environ.get("ZEUS_MT5_SERVER")

    if not all([mt5_login, mt5_password, mt5_server]):
        logger.error(
            "MT5 credentials missing — set ZEUS_MT5_LOGIN, "
            "ZEUS_MT5_PASSWORD, ZEUS_MT5_SERVER"
        )
        sys.exit(1)

    logger.info(
        "Starting SD paper trading",
        server=mt5_server,
        balance=initial_balance,
        risk_pct=risk_pct,
    )

    notifier  = TelegramNotifier(bot_token=bot_token, chat_id=chat_id)
    connector = MT5Connector(
        login=int(mt5_login),
        password=mt5_password,
        server=mt5_server,
    )

    engine = SDPaperEngine(
        connector       = connector,
        symbol          = "XAUUSD",
        initial_balance = initial_balance,
        risk_pct        = risk_pct,
        poll_interval   = poll_interval,
        notifier        = notifier,
    )

    engine.run()


if __name__ == "__main__":
    main()
