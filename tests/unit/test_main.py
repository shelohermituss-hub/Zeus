"""
Tests for zeus/main.py — CLI entry point.

All network calls, exchange construction, and engine I/O are mocked.
The tests verify wiring and policy (live-mode refusal, cleanup guarantees),
not the behaviour of individual components (covered by their own test files).
"""
from __future__ import annotations

import signal as _signal

import pytest
from unittest.mock import MagicMock, patch

from zeus.main import main


# ======================================================================
# Helpers
# ======================================================================

def _settings(is_live: bool = False, **overrides) -> MagicMock:
    """Return a MagicMock that looks like a fully-populated Settings object."""
    s = MagicMock()
    s.is_live  = is_live
    s.is_paper = not is_live
    s.mode.value = "live" if is_live else "paper"
    s.log_level.value     = "INFO"
    s.exchange            = "binance"
    s.symbol              = "BTC/USDT"
    s.timeframe           = "1h"
    s.paper_balance       = 10_000.0
    s.ohlcv_limit         = 500
    s.poll_interval_seconds = 60.0
    s.stop_loss_pct       = 0.01
    s.take_profit_pct     = 0.02
    s.max_position_pct    = 0.02
    s.max_open_positions  = 3
    s.smc_min_score       = 4.0
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def _run(settings=None, mock_engine=None):
    """
    Call main() with all external dependencies mocked.

    Returns (mock_engine, mock_connector).
    Raises whatever main() raises (for exception-path tests).
    """
    if settings is None:
        settings = _settings()
    if mock_engine is None:
        mock_engine = MagicMock()
    mock_connector = MagicMock()

    with patch("zeus.main.get_settings", return_value=settings), \
         patch("zeus.main.setup_logger"), \
         patch("zeus.main.ccxt"), \
         patch("zeus.main.LiveConnector", return_value=mock_connector), \
         patch("zeus.main.SMCStrategy"), \
         patch("zeus.main.PaperEngine", return_value=mock_engine), \
         patch("zeus.main.signal"):
        main()

    return mock_engine, mock_connector


# ======================================================================
# Live mode refusal
# ======================================================================

class TestLiveModeRefusal:
    def test_exits_with_code_1(self):
        with patch("zeus.main.get_settings", return_value=_settings(is_live=True)), \
             patch("zeus.main.setup_logger"), \
             pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_engine_not_created_in_live_mode(self):
        with patch("zeus.main.get_settings", return_value=_settings(is_live=True)), \
             patch("zeus.main.setup_logger"), \
             patch("zeus.main.PaperEngine") as MockEngine, \
             pytest.raises(SystemExit):
            main()
        MockEngine.assert_not_called()


# ======================================================================
# Paper mode — core lifecycle
# ======================================================================

class TestPaperModeLifecycle:
    def test_engine_run_called(self):
        mock_engine, _ = _run()
        mock_engine.run.assert_called_once_with()

    def test_summary_logged_on_clean_exit(self):
        mock_engine, _ = _run()
        mock_engine.metrics.log_summary.assert_called_once()

    def test_connector_closed_on_clean_exit(self):
        _, mock_connector = _run()
        mock_connector.close.assert_called_once()

    def test_summary_logged_when_run_raises(self):
        """log_summary() must run even if engine.run() throws."""
        mock_engine = MagicMock()
        mock_engine.run.side_effect = RuntimeError("unexpected crash")
        with pytest.raises(RuntimeError):
            _run(mock_engine=mock_engine)
        mock_engine.metrics.log_summary.assert_called_once()

    def test_connector_closed_when_run_raises(self):
        """market_connector.close() must run even if engine.run() throws."""
        mock_engine = MagicMock()
        mock_engine.run.side_effect = RuntimeError("unexpected crash")
        mock_connector = MagicMock()

        with patch("zeus.main.get_settings", return_value=_settings()), \
             patch("zeus.main.setup_logger"), \
             patch("zeus.main.ccxt"), \
             patch("zeus.main.LiveConnector", return_value=mock_connector), \
             patch("zeus.main.SMCStrategy"), \
             patch("zeus.main.PaperEngine", return_value=mock_engine), \
             patch("zeus.main.signal"), \
             pytest.raises(RuntimeError):
            main()

        mock_connector.close.assert_called_once()


# ======================================================================
# Paper mode — settings forwarded to engine
# ======================================================================

class TestEngineWiring:
    def _engine_kwargs(self, **setting_overrides) -> dict:
        """Return the keyword arguments passed to PaperEngine constructor."""
        s = _settings(**setting_overrides)
        mock_connector = MagicMock()

        with patch("zeus.main.get_settings", return_value=s), \
             patch("zeus.main.setup_logger"), \
             patch("zeus.main.ccxt"), \
             patch("zeus.main.LiveConnector", return_value=mock_connector), \
             patch("zeus.main.SMCStrategy"), \
             patch("zeus.main.PaperEngine") as MockEngine, \
             patch("zeus.main.signal"):
            MockEngine.return_value = MagicMock()
            main()
            return MockEngine.call_args.kwargs

    def test_symbol_forwarded(self):
        kw = self._engine_kwargs(symbol="ETH/USDT")
        assert kw["symbol"] == "ETH/USDT"

    def test_timeframe_forwarded(self):
        kw = self._engine_kwargs(timeframe="15m")
        assert kw["timeframe"] == "15m"

    def test_initial_balance_forwarded(self):
        kw = self._engine_kwargs(paper_balance=25_000.0)
        assert kw["initial_balance"] == pytest.approx(25_000.0)

    def test_stop_loss_pct_forwarded(self):
        kw = self._engine_kwargs(stop_loss_pct=0.05)
        assert kw["stop_loss_pct"] == pytest.approx(0.05)

    def test_take_profit_pct_forwarded(self):
        kw = self._engine_kwargs(take_profit_pct=0.08)
        assert kw["take_profit_pct"] == pytest.approx(0.08)

    def test_poll_interval_forwarded(self):
        kw = self._engine_kwargs(poll_interval_seconds=30.0)
        assert kw["poll_interval"] == pytest.approx(30.0)

    def test_ohlcv_limit_forwarded(self):
        kw = self._engine_kwargs(ohlcv_limit=200)
        assert kw["ohlcv_limit"] == 200

    def test_max_open_positions_forwarded(self):
        kw = self._engine_kwargs(max_open_positions=5)
        assert kw["max_open_positions"] == 5


# ======================================================================
# Signal handling
# ======================================================================

class TestSignalHandling:
    def test_sigint_and_sigterm_registered(self):
        mock_engine = MagicMock()
        mock_connector = MagicMock()

        with patch("zeus.main.get_settings", return_value=_settings()), \
             patch("zeus.main.setup_logger"), \
             patch("zeus.main.ccxt"), \
             patch("zeus.main.LiveConnector", return_value=mock_connector), \
             patch("zeus.main.SMCStrategy"), \
             patch("zeus.main.PaperEngine", return_value=mock_engine), \
             patch("zeus.main.signal") as mock_sig:
            main()

        registered = {c.args[0] for c in mock_sig.signal.call_args_list}
        assert mock_sig.SIGINT  in registered
        assert mock_sig.SIGTERM in registered

    def test_shutdown_handler_stops_engine(self):
        """_shutdown() must call engine.stop()."""
        mock_engine = MagicMock()
        mock_connector = MagicMock()
        captured: dict[int, object] = {}

        def _capture(sig, handler):
            captured[sig] = handler

        with patch("zeus.main.get_settings", return_value=_settings()), \
             patch("zeus.main.setup_logger"), \
             patch("zeus.main.ccxt"), \
             patch("zeus.main.LiveConnector", return_value=mock_connector), \
             patch("zeus.main.SMCStrategy"), \
             patch("zeus.main.PaperEngine", return_value=mock_engine), \
             patch("zeus.main.signal") as mock_sig:
            mock_sig.SIGINT  = _signal.SIGINT
            mock_sig.SIGTERM = _signal.SIGTERM
            mock_sig.signal.side_effect = _capture
            main()

        # Fire the captured SIGINT handler
        captured[_signal.SIGINT](_signal.SIGINT, None)
        mock_engine.stop.assert_called_once()
