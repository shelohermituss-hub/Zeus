"""
Central configuration loaded from environment variables (or a .env file).
All values are validated at startup — the bot refuses to run with bad config.
"""

from enum import Enum
from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Mode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ZEUS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Execution mode
    mode: Mode = Mode.PAPER

    # Exchange
    exchange: str = "binance"
    api_key: str = ""
    api_secret: str = ""
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"

    # Risk
    max_position_pct: float = Field(default=0.02, gt=0, le=1)
    stop_loss_pct: float = Field(default=0.01, gt=0, le=1)
    take_profit_pct: float = Field(default=0.02, gt=0, le=1)
    max_daily_loss_pct: float = Field(default=0.05, gt=0, le=1)

    # Paper trading
    paper_balance: float = Field(default=10_000.0, gt=0)

    # Engine
    ohlcv_limit:          int   = Field(default=500,  ge=100)
    poll_interval_seconds: float = Field(default=60.0, gt=0)
    max_open_positions:   int   = Field(default=3,    ge=1)

    # Strategy
    smc_min_score: float = Field(default=4.0, ge=1.0, le=10.0)

    # Logging
    log_level: LogLevel = LogLevel.INFO

    @field_validator("timeframe")
    @classmethod
    def valid_timeframe(cls, v: str) -> str:
        allowed = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"}
        if v not in allowed:
            raise ValueError(f"timeframe must be one of {allowed}, got '{v}'")
        return v

    @model_validator(mode="after")
    def live_mode_requires_credentials(self) -> "Settings":
        if self.mode == Mode.LIVE and (not self.api_key or not self.api_secret):
            raise ValueError("ZEUS_API_KEY and ZEUS_API_SECRET are required in live mode")
        return self

    @property
    def is_live(self) -> bool:
        return self.mode == Mode.LIVE

    @property
    def is_paper(self) -> bool:
        return self.mode == Mode.PAPER


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton settings instance. Cached after first call."""
    return Settings()
