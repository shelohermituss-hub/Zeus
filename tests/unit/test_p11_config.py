"""Tests du chargement/validation de la config P11-V4 (zeus/live/p11_config.py)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from zeus.live.p11_config import ConfigError, load_p11_config

_REPO_YAML = Path(__file__).parent.parent.parent / "config" / "p11_v4.yaml"


@pytest.fixture()
def base_cfg() -> dict:
    return yaml.safe_load(_REPO_YAML.read_text())


def _write(tmp_path: Path, cfg: dict) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


# ── Le YAML livré est la référence : il DOIT charger ─────────────────────────

class TestShippedConfig:
    def test_shipped_yaml_loads(self):
        cfg = load_p11_config(_REPO_YAML)
        assert cfg.mode == "paper"
        assert cfg.account_balance == 100_000
        assert len(cfg.enabled_symbols) == 11

    def test_shipped_values_match_backtest(self):
        cfg = load_p11_config(_REPO_YAML)
        assert cfg.propfirm.risk_per_trade_pct == 0.006
        assert cfg.propfirm.max_daily_loss_pct == 0.03
        assert cfg.propfirm.max_total_dd_pct == 0.06
        xau = cfg.groups["xau_v4"]
        assert xau.signals["min_wyckoff_score_long"] == 5.9
        assert xau.signals["min_wyckoff_score_short"] == 8.5
        assert xau.exits.runner_rr == 20.0
        assert xau.exits.tp3_r == 8.0
        fx = cfg.groups["forex_ws75"]
        assert fx.signals["min_wyckoff_score_long"] == 7.5
        assert fx.exits.tp1_r == 1.5
        assert fx.exits.runner_rr == 10.0

    def test_broker_symbol_mapping(self):
        cfg = load_p11_config(_REPO_YAML)
        assert cfg.symbols["GRXEUR"].broker_symbol == "GER40"
        assert cfg.symbols["XAUUSD"].broker_symbol == "XAUUSD"

    def test_default_mode_is_paper_not_live(self):
        # Sécurité : la config livrée ne démarre JAMAIS en live
        assert load_p11_config(_REPO_YAML).mode == "paper"


# ── Fail closed : toute config invalide est refusée ──────────────────────────

class TestFailClosed:
    def test_missing_file(self, tmp_path):
        with pytest.raises(ConfigError, match="introuvable"):
            load_p11_config(tmp_path / "absent.yaml")

    def test_bad_mode(self, tmp_path, base_cfg):
        base_cfg["mode"] = "yolo"
        with pytest.raises(ConfigError, match="mode"):
            load_p11_config(_write(tmp_path, base_cfg))

    def test_risk_above_daily_limit_rejected(self, tmp_path, base_cfg):
        base_cfg["risk"]["risk_per_trade_pct"] = 0.05   # > max_daily_loss_pct
        with pytest.raises(ConfigError, match="risk"):
            load_p11_config(_write(tmp_path, base_cfg))

    def test_missing_risk_field(self, tmp_path, base_cfg):
        del base_cfg["risk"]["max_total_dd_pct"]
        with pytest.raises(ConfigError, match="max_total_dd_pct"):
            load_p11_config(_write(tmp_path, base_cfg))

    def test_unknown_group_rejected(self, tmp_path, base_cfg):
        base_cfg["symbols"]["XAUUSD"]["group"] = "inexistant"
        with pytest.raises(ConfigError, match="groupe inconnu"):
            load_p11_config(_write(tmp_path, base_cfg))

    def test_no_enabled_symbol_rejected(self, tmp_path, base_cfg):
        for s in base_cfg["symbols"].values():
            s["enabled"] = False
        with pytest.raises(ConfigError, match="aucun symbole"):
            load_p11_config(_write(tmp_path, base_cfg))

    def test_bad_exits_order_rejected(self, tmp_path, base_cfg):
        base_cfg["strategy_groups"]["xau_v4"]["exits"]["tp2_r"] = 0.5  # < tp1_r
        with pytest.raises(ConfigError, match="tp1_r < tp2_r"):
            load_p11_config(_write(tmp_path, base_cfg))

    def test_runner_below_tp2_rejected(self, tmp_path, base_cfg):
        base_cfg["strategy_groups"]["forex_ws75"]["exits"]["runner_rr"] = 2.0
        with pytest.raises(ConfigError, match="runner_rr"):
            load_p11_config(_write(tmp_path, base_cfg))

    def test_negative_pip_rejected(self, tmp_path, base_cfg):
        base_cfg["symbols"]["EURUSD"]["pip_size"] = -1
        with pytest.raises(ConfigError, match="pip_size"):
            load_p11_config(_write(tmp_path, base_cfg))

    def test_poll_too_fast_rejected(self, tmp_path, base_cfg):
        base_cfg["execution"]["poll_seconds"] = 1
        with pytest.raises(ConfigError, match="poll_seconds"):
            load_p11_config(_write(tmp_path, base_cfg))

    def test_bad_session_rejected(self, tmp_path, base_cfg):
        base_cfg["strategy_groups"]["xau_v4"]["signals"]["session_end_utc"] = 3
        with pytest.raises(ConfigError, match="session"):
            load_p11_config(_write(tmp_path, base_cfg))


# ── Personnalisation utilisateur ─────────────────────────────────────────────

class TestUserOverrides:
    def test_disable_pair(self, tmp_path, base_cfg):
        base_cfg["symbols"]["EURJPY"]["enabled"] = False
        cfg = load_p11_config(_write(tmp_path, base_cfg))
        assert "EURJPY" not in cfg.enabled_symbols
        assert len(cfg.enabled_symbols) == 10

    def test_change_risk(self, tmp_path, base_cfg):
        base_cfg["risk"]["risk_per_trade_pct"] = 0.005
        cfg = load_p11_config(_write(tmp_path, base_cfg))
        assert cfg.propfirm.risk_per_trade_pct == 0.005

    def test_change_runner_rr(self, tmp_path, base_cfg):
        base_cfg["strategy_groups"]["xau_v4"]["exits"]["runner_rr"] = 15.0
        cfg = load_p11_config(_write(tmp_path, base_cfg))
        assert cfg.groups["xau_v4"].exits.runner_rr == 15.0
