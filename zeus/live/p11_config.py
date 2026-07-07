"""
Chargement et validation de la configuration P11-V4 (config/p11_v4.yaml).

Principe fail-closed : toute valeur manquante, incohérente ou hors bornes
lève ConfigError et le bot refuse de démarrer.  Les valeurs par défaut du
YAML livré sont les paramètres validés par backtest — ce module ne fournit
AUCUN défaut implicite pour les champs critiques.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from zeus.risk.propfirm import PropFirmConfig


class ConfigError(ValueError):
    """Configuration invalide — le bot ne doit pas démarrer."""


# ── Structures ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExitsConfig:
    tp1_r:              float
    tp1_close_pct:      float
    tp2_r:              float
    tp2_cumulative_pct: float
    runner_rr:          float
    tp3_r:              float = 0.0     # 0 = pas de TP3 (groupe forex)
    tp3_cumulative_pct: float = 0.0

    def validate(self, where: str) -> None:
        if not 0 < self.tp1_r < self.tp2_r:
            raise ConfigError(f"{where}: exige 0 < tp1_r < tp2_r")
        if self.tp3_r and not self.tp2_r < self.tp3_r < self.runner_rr:
            raise ConfigError(f"{where}: exige tp2_r < tp3_r < runner_rr")
        if self.runner_rr <= self.tp2_r:
            raise ConfigError(f"{where}: runner_rr doit dépasser tp2_r")
        for name, v in [("tp1_close_pct", self.tp1_close_pct),
                        ("tp2_cumulative_pct", self.tp2_cumulative_pct),
                        ("tp3_cumulative_pct", self.tp3_cumulative_pct)]:
            if not 0.0 <= v <= 1.0:
                raise ConfigError(f"{where}: {name} doit être dans [0, 1]")
        if self.tp2_cumulative_pct < self.tp1_close_pct:
            raise ConfigError(f"{where}: tp2_cumulative_pct < tp1_close_pct")


@dataclass(frozen=True)
class GroupConfig:
    name:           str
    zone_timeframe: str
    wyckoff:        dict
    signals:        dict
    exits:          ExitsConfig
    caps:           dict

    def validate(self) -> None:
        if self.zone_timeframe not in ("5m", "15m", "30m", "1h"):
            raise ConfigError(f"groupe {self.name}: zone_timeframe invalide "
                              f"({self.zone_timeframe})")
        for key in ("min_wyckoff_score_long", "min_wyckoff_score_short"):
            if key not in self.signals:
                raise ConfigError(f"groupe {self.name}: signals.{key} manquant")
        s0 = self.signals.get("session_start_utc")
        s1 = self.signals.get("session_end_utc")
        if s0 is None or s1 is None or not (0 <= s0 < s1 <= 24):
            raise ConfigError(f"groupe {self.name}: session UTC invalide ({s0}-{s1})")
        self.exits.validate(f"groupe {self.name}.exits")
        for key in ("max_daily_losses", "max_monthly_losses"):
            if self.caps.get(key, -1) < 0:
                raise ConfigError(f"groupe {self.name}: caps.{key} manquant ou négatif")


@dataclass(frozen=True)
class SymbolConfig:
    symbol:        str
    enabled:       bool
    group:         str
    pip_size:      float
    broker_symbol: str

    def validate(self, groups: dict[str, GroupConfig]) -> None:
        if self.group not in groups:
            raise ConfigError(f"symbole {self.symbol}: groupe inconnu '{self.group}'")
        if self.pip_size <= 0:
            raise ConfigError(f"symbole {self.symbol}: pip_size doit être > 0")
        if not self.broker_symbol:
            raise ConfigError(f"symbole {self.symbol}: broker_symbol vide")


@dataclass(frozen=True)
class ExecutionConfig:
    poll_seconds:        int
    history_days:        int
    magic_number:        int
    max_slippage_points: int
    dry_run:             bool

    def validate(self) -> None:
        if self.poll_seconds < 5:
            raise ConfigError("execution.poll_seconds doit être >= 5")
        if self.history_days < 7:
            raise ConfigError("execution.history_days doit être >= 7 (warm-up zones)")
        if self.magic_number <= 0:
            raise ConfigError("execution.magic_number doit être > 0")


@dataclass(frozen=True)
class P11Config:
    mode:            str                      # "paper" | "live"
    account_balance: float
    propfirm:        PropFirmConfig
    groups:          dict[str, GroupConfig]
    symbols:         dict[str, SymbolConfig]  # uniquement les enabled
    execution:       ExecutionConfig
    telegram_enabled: bool
    log_level:       str

    @property
    def enabled_symbols(self) -> list[str]:
        return list(self.symbols.keys())


# ── Chargement ────────────────────────────────────────────────────────────────

def load_p11_config(path: str | Path) -> P11Config:
    """Charge et valide config/p11_v4.yaml.  Lève ConfigError si invalide."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"fichier de configuration introuvable : {path}")

    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML invalide dans {path} : {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} : structure YAML racine invalide")

    def require(key: str) -> object:
        if key not in raw:
            raise ConfigError(f"{path} : section '{key}' manquante")
        return raw[key]

    mode = str(require("mode")).lower()
    if mode not in ("paper", "live"):
        raise ConfigError(f"mode invalide : '{mode}' (attendu paper|live)")

    balance = float(require("account_balance"))
    if balance <= 0:
        raise ConfigError("account_balance doit être > 0")

    # Guard propfirm — la validation des bornes est déléguée à PropFirmConfig
    r = require("risk")
    try:
        propfirm = PropFirmConfig(
            max_daily_loss_pct   = float(r["max_daily_loss_pct"]),
            max_total_dd_pct     = float(r["max_total_dd_pct"]),
            risk_per_trade_pct   = float(r["risk_per_trade_pct"]),
            max_losses_per_day   = int(r["max_losses_per_day"]),
            flat_hour_utc        = int(r["flat_hour_utc"]),
            profit_target_pct    = float(r["profit_target_pct"]),
            min_trading_days     = int(r["min_trading_days"]),
        )
    except KeyError as e:
        raise ConfigError(f"risk.{e.args[0]} manquant") from e
    except ValueError as e:
        raise ConfigError(f"risk invalide : {e}") from e

    # Groupes de stratégie
    groups: dict[str, GroupConfig] = {}
    for gname, g in dict(require("strategy_groups")).items():
        try:
            exits = ExitsConfig(
                tp1_r              = float(g["exits"]["tp1_r"]),
                tp1_close_pct      = float(g["exits"]["tp1_close_pct"]),
                tp2_r              = float(g["exits"]["tp2_r"]),
                tp2_cumulative_pct = float(g["exits"]["tp2_cumulative_pct"]),
                tp3_r              = float(g["exits"].get("tp3_r", 0.0)),
                tp3_cumulative_pct = float(g["exits"].get("tp3_cumulative_pct", 0.0)),
                runner_rr          = float(g["exits"]["runner_rr"]),
            )
            grp = GroupConfig(
                name           = gname,
                zone_timeframe = str(g["zone_timeframe"]),
                wyckoff        = dict(g["wyckoff"]),
                signals        = dict(g["signals"]),
                exits          = exits,
                caps           = dict(g["caps"]),
            )
        except KeyError as e:
            raise ConfigError(f"strategy_groups.{gname} : champ {e.args[0]} manquant") from e
        grp.validate()
        groups[gname] = grp

    if not groups:
        raise ConfigError("strategy_groups : aucun groupe défini")

    # Symboles — seuls les enabled sont retenus
    symbols: dict[str, SymbolConfig] = {}
    for sname, s in dict(require("symbols")).items():
        try:
            sc = SymbolConfig(
                symbol        = sname,
                enabled       = bool(s["enabled"]),
                group         = str(s["group"]),
                pip_size      = float(s["pip_size"]),
                broker_symbol = str(s.get("broker_symbol", sname)),
            )
        except KeyError as e:
            raise ConfigError(f"symbols.{sname} : champ {e.args[0]} manquant") from e
        sc.validate(groups)
        if sc.enabled:
            symbols[sname] = sc

    if not symbols:
        raise ConfigError("symbols : aucun symbole activé — rien à trader")

    # Exécution
    e = require("execution")
    try:
        execution = ExecutionConfig(
            poll_seconds        = int(e["poll_seconds"]),
            history_days        = int(e["history_days"]),
            magic_number        = int(e["magic_number"]),
            max_slippage_points = int(e["max_slippage_points"]),
            dry_run             = bool(e["dry_run"]),
        )
    except KeyError as ke:
        raise ConfigError(f"execution.{ke.args[0]} manquant") from ke
    execution.validate()

    mon = raw.get("monitoring", {})

    return P11Config(
        mode             = mode,
        account_balance  = balance,
        propfirm         = propfirm,
        groups           = groups,
        symbols          = symbols,
        execution        = execution,
        telegram_enabled = bool(mon.get("telegram_enabled", False)),
        log_level        = str(mon.get("log_level", "INFO")),
    )
