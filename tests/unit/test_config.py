"""Tests for zeus.config — validation logic."""

import pytest
from pydantic import ValidationError

from zeus.config import ConnectorType, Mode, Settings


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


class TestEngineDefaults:
    def test_ohlcv_limit_default_500(self):
        s = make_settings()
        assert s.ohlcv_limit == 500

    def test_ohlcv_limit_minimum_100(self):
        with pytest.raises(ValidationError):
            make_settings(ohlcv_limit=99)

    def test_ohlcv_limit_100_is_valid(self):
        s = make_settings(ohlcv_limit=100)
        assert s.ohlcv_limit == 100

    def test_poll_interval_default_60(self):
        s = make_settings()
        assert s.poll_interval_seconds == pytest.approx(60.0)

    def test_poll_interval_zero_raises(self):
        with pytest.raises(ValidationError):
            make_settings(poll_interval_seconds=0.0)

    def test_poll_interval_negative_raises(self):
        with pytest.raises(ValidationError):
            make_settings(poll_interval_seconds=-1.0)

    def test_max_open_positions_default_3(self):
        s = make_settings()
        assert s.max_open_positions == 3

    def test_max_open_positions_zero_raises(self):
        with pytest.raises(ValidationError):
            make_settings(max_open_positions=0)

    def test_max_open_positions_1_is_valid(self):
        s = make_settings(max_open_positions=1)
        assert s.max_open_positions == 1


class TestStrategyDefaults:
    def test_smc_min_score_default_4(self):
        s = make_settings()
        assert s.smc_min_score == pytest.approx(4.0)

    def test_smc_min_score_below_1_raises(self):
        with pytest.raises(ValidationError):
            make_settings(smc_min_score=0.5)

    def test_smc_min_score_above_10_raises(self):
        with pytest.raises(ValidationError):
            make_settings(smc_min_score=10.1)

    def test_smc_min_score_10_is_valid(self):
        s = make_settings(smc_min_score=10.0)
        assert s.smc_min_score == pytest.approx(10.0)


class TestTelegramDefaults:
    def test_telegram_disabled_by_default(self):
        s = make_settings()
        assert s.telegram_bot_token == ""
        assert s.telegram_chat_id == ""

    def test_telegram_fields_accept_overrides(self):
        s = make_settings(telegram_bot_token="123:ABC", telegram_chat_id="-100200300")
        assert s.telegram_bot_token == "123:ABC"
        assert s.telegram_chat_id == "-100200300"


class TestConnectorType:
    def test_default_connector_is_ccxt(self):
        s = make_settings()
        assert s.connector == ConnectorType.CCXT

    def test_mt5_connector_accepted(self):
        s = make_settings(connector="mt5")
        assert s.connector == ConnectorType.MT5

    def test_invalid_connector_raises(self):
        with pytest.raises(ValidationError):
            make_settings(connector="fix")

    def test_mt5_symbol_strips_slash(self):
        s = make_settings(symbol="XAU/USD")
        assert s.mt5_symbol == "XAUUSD"

    def test_mt5_symbol_no_slash_unchanged(self):
        s = make_settings(symbol="GBPUSD")
        assert s.mt5_symbol == "GBPUSD"

    def test_mt5_symbol_us30(self):
        s = make_settings(symbol="US30")
        assert s.mt5_symbol == "US30"


class TestMT5Credentials:
    def test_mt5_live_without_credentials_raises(self):
        with pytest.raises(
            ValidationError,
            match="ZEUS_MT5_LOGIN, ZEUS_MT5_PASSWORD, ZEUS_MT5_SERVER",
        ):
            make_settings(
                mode="live",
                connector="mt5",
                mt5_login=0,
                mt5_password="",
                mt5_server="",
            )

    def test_mt5_live_with_credentials_ok(self):
        s = make_settings(
            mode="live",
            connector="mt5",
            mt5_login=123456,
            mt5_password="secret",
            mt5_server="Demo-Server",
        )
        assert s.is_live
        assert s.mt5_login == 123456

    def test_mt5_paper_without_credentials_ok(self):
        # Paper mode never needs credentials
        s = make_settings(connector="mt5", mode="paper")
        assert s.is_paper

    def test_ccxt_live_still_requires_api_keys(self):
        with pytest.raises(ValidationError, match="ZEUS_API_KEY and ZEUS_API_SECRET"):
            make_settings(mode="live", connector="ccxt", api_key="", api_secret="")
