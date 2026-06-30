"""
Structured logging for Zeus.
- Development: human-readable coloured output on stderr
- Production (log_level INFO+): JSON records appended to data/logs/zeus.log
"""

import sys
from pathlib import Path

from loguru import logger


def setup_logger(log_level: str = "INFO") -> None:
    logger.remove()

    # Console — always on
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>"
        ),
        colorize=True,
    )

    # File — JSON, rotate at 10 MB, keep 14 days
    log_path = Path("data/logs/zeus.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_path),
        level=log_level,
        serialize=True,          # JSON output
        rotation="10 MB",
        retention="14 days",
        compression="gz",
    )

    logger.info("Logger initialised", level=log_level)


__all__ = ["logger", "setup_logger"]
