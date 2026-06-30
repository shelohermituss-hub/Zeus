"""Tests for zeus.config — validation logic."""

import pytest
from pydantic import ValidationError

from zeus.config import Mode, Settings


def make_settings(**overrides) -> Settings:
    """Build a Settings instance with test defaults, applying overrides."""
    defaults = dict(
        mode="paper",
        exchange="binance",
        api_key="",
        api_secret="",
        symbol="BTC/USDT",
        timeframe="1h",
        max_position_pct=0.02,
        stop_loss_pct=0.01,
        take_profit_pct=0.02,
        max_daily_loss_pct=0.05,
        paper_balance=10_000.0,
        log_level="INFO",
    )
    defaults.update(overrides)
    return Settings(**defaults)


class TestBasicDefaults:
    def test_paper_mode_is_default(self):
        s = make_settings()
        assert s.mode == Mode.PAPER
        assert s.is_paper
        assert not s.is_live

    def test_paper_balance_positive(self):
        s = make_settings(paper_balance=5_000.0)
        assert s.paper_balance == 5_000.0


class TestLiveModeValidation:
    def test_live_mode_without_credentials_raises(self):
        with pytest.raises(ValidationError, match="ZEUS_API_KEY and ZEUS_API_SECRET are required"):
            make_settings(mode="live", api_key="", api_secret="")

    def test_live_mode_with_credentials_ok(self):
        s = make_settings(mode="live", api_key="key123", api_secret="secret456")
        assert s.is_live

    def test_live_mode_missing_secret_raises(self):
        with pytest.raises(ValidationError):
            make_settings(mode="live", api_key="key123", api_secret="")


class TestRiskParameterValidation:
    def test_max_position_pct_zero_raises(self):
        with pytest.raises(ValidationError):
            make_settings(max_position_pct=0.0)

    def test_max_position_pct_above_one_raises(self):
        with pytest.raises(ValidationError):
            make_settings(max_position_pct=1.01)

    def test_stop_loss_must_be_positive(self):
        with pytest.raises(ValidationError):
            make_settings(stop_loss_pct=-0.01)

    def test_paper_balance_must_be_positive(self):
        with pytest.raises(ValidationError):
            make_settings(paper_balance=0.0)


class TestTimeframeValidation:
    def test_valid_timeframes_accepted(self):
        for tf in ("1m", "5m", "15m", "1h", "4h", "1d"):
            s = make_settings(timeframe=tf)
            assert s.timeframe == tf

    def test_invalid_timeframe_raises(self):
        with pytest.raises(ValidationError, match="timeframe must be one of"):
            make_settings(timeframe="2d")
