"""Tests du lanceur CLI P11 (zeus/live/run_p11.py) — surcharges et garde-fous."""
from __future__ import annotations

import pytest

from zeus.live.p11_config import ConfigError
from zeus.live.run_p11 import _parse_args, build_engine


class _StubConnector:
    def fetch_ohlcv(self, symbol, timeframe, limit):
        return None


def _engine(argv):
    args = _parse_args(argv)
    return build_engine(args, connector=_StubConnector())


class TestOverrides:
    def test_default_is_paper_all_pairs(self, monkeypatch):
        e = _engine([])
        assert e.config.mode == "paper"
        assert len(e.config.enabled_symbols) == 11

    def test_risk_override(self):
        e = _engine(["--risk", "0.005"])
        assert e.config.propfirm.risk_per_trade_pct == 0.005
        assert e.risk_usd == pytest.approx(500.0)

    def test_pairs_override(self):
        e = _engine(["--pairs", "XAUUSD,EURUSD"])
        assert sorted(e.config.enabled_symbols) == ["EURUSD", "XAUUSD"]


class TestGuardrails:
    def test_risk_above_validated_max_rejected(self):
        with pytest.raises(ConfigError, match="hors bornes"):
            _engine(["--risk", "0.02"])

    def test_unknown_pair_rejected(self):
        with pytest.raises(ConfigError, match="inconnus"):
            _engine(["--pairs", "BTCUSD"])

    def test_live_requires_double_opt_in(self):
        # YAML livré = paper → --mode live seul doit être refusé
        with pytest.raises(ConfigError, match="double confirmation"):
            _engine(["--mode", "live"])
