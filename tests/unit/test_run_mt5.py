"""
Tests for zeus/backtest/run_mt5.py — MT5-sourced backtest CLI.

All MT5/network calls, the strategy, and the backtest engine are mocked.
These tests verify wiring and fail-closed behaviour, not BacktestEngine's
internal correctness (covered by its own test file).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from zeus.backtest.run_mt5 import main
from zeus.config import ConnectorType


# ======================================================================
# Helpers
# ======================================================================

def _settings(connector=ConnectorType.MT5, **overrides) -> MagicMock:
    """Return a MagicMock that looks like a fully-populated Settings object."""
    s = MagicMock()
    s.connector           = connector
    s.mt5_symbol          = "XAUUSD"
    s.timeframe           = "1h"
    s.ohlcv_limit          = 500
    s.paper_balance        = 10_000.0
    s.stop_loss_pct        = 0.01
    s.take_profit_pct      = 0.02
    s.max_position_pct     = 0.02
    s.max_open_positions   = 3
    s.smc_min_score        = 4.0
    s.log_level.value      = "INFO"
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def _df(n_bars: int = 50) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open":   [100.0] * n_bars,
            "high":   [101.0] * n_bars,
            "low":    [99.0] * n_bars,
            "close":  [100.0] * n_bars,
            "volume": [1.0] * n_bars,
        },
        index=idx,
    )


# ======================================================================
# Connector type guard (fail closed if not MT5)
# ======================================================================

class TestConnectorTypeGuard:
    def test_exits_with_code_1_when_connector_not_mt5(self):
        with patch("zeus.backtest.run_mt5.get_settings", return_value=_settings(connector=ConnectorType.CCXT)), \
             patch("zeus.backtest.run_mt5.setup_logger"), \
             pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_market_connector_not_built_when_connector_not_mt5(self):
        with patch("zeus.backtest.run_mt5.get_settings", return_value=_settings(connector=ConnectorType.CCXT)), \
             patch("zeus.backtest.run_mt5.setup_logger"), \
             patch("zeus.backtest.run_mt5.create_market_connector") as mock_factory, \
             pytest.raises(SystemExit):
            main()
        mock_factory.assert_not_called()


# ======================================================================
# Data fetch
# ======================================================================

class TestDataFetch:
    def test_fetch_ohlcv_called_with_mt5_symbol_timeframe_limit(self):
        market = MagicMock()
        market.fetch_ohlcv.return_value = _df()
        settings = _settings(timeframe="15m", ohlcv_limit=200)
        with patch("zeus.backtest.run_mt5.get_settings", return_value=settings), \
             patch("zeus.backtest.run_mt5.setup_logger"), \
             patch("zeus.backtest.run_mt5.create_market_connector", return_value=market), \
             patch("zeus.backtest.run_mt5.SMCStrategy"), \
             patch("zeus.backtest.run_mt5.BacktestEngine"):
            main()
        market.fetch_ohlcv.assert_called_once_with("XAUUSD", "15m", 200)

    def test_exits_with_code_1_on_empty_dataframe(self):
        market = MagicMock()
        market.fetch_ohlcv.return_value = pd.DataFrame()
        with patch("zeus.backtest.run_mt5.get_settings", return_value=_settings()), \
             patch("zeus.backtest.run_mt5.setup_logger"), \
             patch("zeus.backtest.run_mt5.create_market_connector", return_value=market), \
             pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_exits_with_code_1_on_none_dataframe(self):
        market = MagicMock()
        market.fetch_ohlcv.return_value = None
        with patch("zeus.backtest.run_mt5.get_settings", return_value=_settings()), \
             patch("zeus.backtest.run_mt5.setup_logger"), \
             patch("zeus.backtest.run_mt5.create_market_connector", return_value=market), \
             pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_connector_closed_after_empty_dataframe_exit(self):
        """Cleanup must run even on the fail-closed empty-data exit path."""
        market = MagicMock()
        market.fetch_ohlcv.return_value = pd.DataFrame()
        with patch("zeus.backtest.run_mt5.get_settings", return_value=_settings()), \
             patch("zeus.backtest.run_mt5.setup_logger"), \
             patch("zeus.backtest.run_mt5.create_market_connector", return_value=market), \
             pytest.raises(SystemExit):
            main()
        market.close.assert_called_once()


# ======================================================================
# Engine wiring — settings forwarded to BacktestEngine
# ======================================================================

class TestEngineWiring:
    def _engine_kwargs(self, **setting_overrides) -> dict:
        market = MagicMock()
        market.fetch_ohlcv.return_value = _df()
        settings = _settings(**setting_overrides)
        with patch("zeus.backtest.run_mt5.get_settings", return_value=settings), \
             patch("zeus.backtest.run_mt5.setup_logger"), \
             patch("zeus.backtest.run_mt5.create_market_connector", return_value=market), \
             patch("zeus.backtest.run_mt5.SMCStrategy"), \
             patch("zeus.backtest.run_mt5.BacktestEngine") as MockEngine:
            MockEngine.return_value = MagicMock()
            main()
            return MockEngine.call_args.kwargs

    def test_initial_balance_forwarded(self):
        kw = self._engine_kwargs(paper_balance=25_000.0)
        assert kw["initial_balance"] == pytest.approx(25_000.0)

    def test_stop_loss_pct_forwarded(self):
        kw = self._engine_kwargs(stop_loss_pct=0.05)
        assert kw["stop_loss_pct"] == pytest.approx(0.05)

    def test_take_profit_pct_forwarded(self):
        kw = self._engine_kwargs(take_profit_pct=0.08)
        assert kw["take_profit_pct"] == pytest.approx(0.08)

    def test_max_position_pct_forwarded(self):
        kw = self._engine_kwargs(max_position_pct=0.05)
        assert kw["max_position_pct"] == pytest.approx(0.05)

    def test_max_open_positions_forwarded(self):
        kw = self._engine_kwargs(max_open_positions=7)
        assert kw["max_open_positions"] == 7

    def test_engine_run_called_with_fetched_df_and_symbol(self):
        market = MagicMock()
        df = _df()
        market.fetch_ohlcv.return_value = df
        mock_engine = MagicMock()
        with patch("zeus.backtest.run_mt5.get_settings", return_value=_settings()), \
             patch("zeus.backtest.run_mt5.setup_logger"), \
             patch("zeus.backtest.run_mt5.create_market_connector", return_value=market), \
             patch("zeus.backtest.run_mt5.SMCStrategy"), \
             patch("zeus.backtest.run_mt5.BacktestEngine", return_value=mock_engine):
            main()
        mock_engine.run.assert_called_once_with(df, symbol="XAUUSD")

    def test_strategy_min_score_forwarded(self):
        with patch("zeus.backtest.run_mt5.get_settings", return_value=_settings(smc_min_score=6.0)), \
             patch("zeus.backtest.run_mt5.setup_logger"), \
             patch("zeus.backtest.run_mt5.create_market_connector", return_value=MagicMock(fetch_ohlcv=MagicMock(return_value=_df()))), \
             patch("zeus.backtest.run_mt5.SMCStrategy") as MockStrategy, \
             patch("zeus.backtest.run_mt5.BacktestEngine"):
            main()
        MockStrategy.assert_called_once_with(min_score=6.0)


# ======================================================================
# Cleanup guarantees
# ======================================================================

class TestCleanup:
    def test_connector_closed_on_clean_run(self):
        market = MagicMock()
        market.fetch_ohlcv.return_value = _df()
        with patch("zeus.backtest.run_mt5.get_settings", return_value=_settings()), \
             patch("zeus.backtest.run_mt5.setup_logger"), \
             patch("zeus.backtest.run_mt5.create_market_connector", return_value=market), \
             patch("zeus.backtest.run_mt5.SMCStrategy"), \
             patch("zeus.backtest.run_mt5.BacktestEngine"):
            main()
        market.close.assert_called_once()

    def test_connector_closed_when_engine_run_raises(self):
        """market.close() must run even if BacktestEngine.run() throws."""
        market = MagicMock()
        market.fetch_ohlcv.return_value = _df()
        mock_engine = MagicMock()
        mock_engine.run.side_effect = RuntimeError("boom")
        with patch("zeus.backtest.run_mt5.get_settings", return_value=_settings()), \
             patch("zeus.backtest.run_mt5.setup_logger"), \
             patch("zeus.backtest.run_mt5.create_market_connector", return_value=market), \
             patch("zeus.backtest.run_mt5.SMCStrategy"), \
             patch("zeus.backtest.run_mt5.BacktestEngine", return_value=mock_engine), \
             pytest.raises(RuntimeError):
            main()
        market.close.assert_called_once()

    def test_connector_closed_when_fetch_ohlcv_raises(self):
        """market.close() must run even if fetch_ohlcv() throws."""
        market = MagicMock()
        market.fetch_ohlcv.side_effect = RuntimeError("connection lost")
        with patch("zeus.backtest.run_mt5.get_settings", return_value=_settings()), \
             patch("zeus.backtest.run_mt5.setup_logger"), \
             patch("zeus.backtest.run_mt5.create_market_connector", return_value=market), \
             pytest.raises(RuntimeError):
            main()
        market.close.assert_called_once()
