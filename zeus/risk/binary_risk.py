"""
Risk manager dédié aux options binaires (payout fixe, perte bornée à la mise).

Différences avec le risk manager forex :
- Pas de stop-loss ni de take-profit : la perte maximale d'un trade est sa mise.
- La taille de position est la mise, une fraction du solde bornée par min/max.
- La série de pertes consécutives est un signal de rupture d'edge : elle bloque
  les nouveaux trades jusqu'au jour suivant.

Fail closed : toute valeur incohérente (solde non fini, mise hors bornes,
connexion inconnue) refuse le trade.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum


class BinaryRejection(str, Enum):
    KILL_SWITCH_ACTIVE     = "kill_switch_active"
    BROKER_DISCONNECTED    = "broker_disconnected"
    INVALID_BALANCE        = "invalid_balance"
    STAKE_BELOW_MINIMUM    = "stake_below_minimum"
    INSUFFICIENT_BALANCE   = "insufficient_balance"
    DAILY_LOSS_LIMIT_HIT   = "daily_loss_limit_hit"
    MAX_DRAWDOWN_HIT       = "max_drawdown_hit"
    CONSECUTIVE_LOSSES_HIT = "consecutive_losses_hit"
    TOO_MANY_OPEN_TRADES   = "too_many_open_trades"
    MAX_EXPOSURE_EXCEEDED  = "max_exposure_exceeded"


@dataclass(frozen=True)
class BinaryRiskConfig:
    stake_fraction: float = 0.01          # mise = 1 % du solde
    min_stake: float = 1.0                # minimum broker (Pocket Option : 1 $)
    max_stake: float = 100.0
    max_daily_loss_fraction: float = 0.05     # vs solde de début de journée
    max_drawdown_fraction: float = 0.20       # vs plus haut historique du solde
    max_consecutive_losses: int = 5
    max_open_trades: int = 1
    max_exposure_fraction: float = 0.03       # somme des mises ouvertes / solde

    def __post_init__(self) -> None:
        if not 0 < self.stake_fraction <= 0.05:
            raise ValueError("stake_fraction doit être dans ]0, 0.05]")
        if not 0 < self.min_stake <= self.max_stake:
            raise ValueError("il faut 0 < min_stake <= max_stake")
        for name in ("max_daily_loss_fraction", "max_drawdown_fraction", "max_exposure_fraction"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} doit être dans ]0, 1[")
        if self.max_consecutive_losses < 1 or self.max_open_trades < 1:
            raise ValueError("max_consecutive_losses et max_open_trades doivent être >= 1")


@dataclass(frozen=True)
class BinaryRiskDecision:
    approved: bool
    stake: float = 0.0
    rejection: BinaryRejection | None = None


class BinaryRiskManager:
    """État de risque d'une session ; une instance par compte."""

    def __init__(self, config: BinaryRiskConfig, initial_balance: float) -> None:
        if not math.isfinite(initial_balance) or initial_balance <= 0:
            raise ValueError("initial_balance doit être fini et > 0")
        self.config = config
        self.balance = initial_balance
        self.peak_balance = initial_balance
        self.kill_switch = False
        self._day: date | None = None
        self._day_start_balance = initial_balance
        self._consecutive_losses = 0
        self._open_stakes: list[float] = []

    # -- décision ---------------------------------------------------------

    def evaluate(self, now: datetime, broker_connected: bool) -> BinaryRiskDecision:
        self._roll_day(now)
        rejection = self._first_violation(broker_connected)
        if rejection is not None:
            return BinaryRiskDecision(False, rejection=rejection)
        stake = round(min(self.balance * self.config.stake_fraction, self.config.max_stake), 2)
        if stake < self.config.min_stake:
            return BinaryRiskDecision(False, rejection=BinaryRejection.STAKE_BELOW_MINIMUM)
        if stake > self.balance - sum(self._open_stakes):
            return BinaryRiskDecision(False, rejection=BinaryRejection.INSUFFICIENT_BALANCE)
        if sum(self._open_stakes) + stake > self.balance * self.config.max_exposure_fraction:
            return BinaryRiskDecision(False, rejection=BinaryRejection.MAX_EXPOSURE_EXCEEDED)
        return BinaryRiskDecision(True, stake=stake)

    def _first_violation(self, broker_connected: bool) -> BinaryRejection | None:
        cfg = self.config
        if self.kill_switch:
            return BinaryRejection.KILL_SWITCH_ACTIVE
        if broker_connected is not True:
            return BinaryRejection.BROKER_DISCONNECTED
        if not math.isfinite(self.balance) or self.balance <= 0:
            return BinaryRejection.INVALID_BALANCE
        if self.balance <= self._day_start_balance * (1 - cfg.max_daily_loss_fraction):
            return BinaryRejection.DAILY_LOSS_LIMIT_HIT
        if self.balance <= self.peak_balance * (1 - cfg.max_drawdown_fraction):
            return BinaryRejection.MAX_DRAWDOWN_HIT
        if self._consecutive_losses >= cfg.max_consecutive_losses:
            return BinaryRejection.CONSECUTIVE_LOSSES_HIT
        if len(self._open_stakes) >= cfg.max_open_trades:
            return BinaryRejection.TOO_MANY_OPEN_TRADES
        return None

    def _roll_day(self, now: datetime) -> None:
        if now.date() != self._day:
            self._day = now.date()
            self._day_start_balance = self.balance
            self._consecutive_losses = 0

    # -- cycle de vie d'un trade -------------------------------------------

    def record_open(self, stake: float) -> None:
        self._open_stakes.append(stake)

    def record_close(self, stake: float, pnl: float) -> None:
        """pnl : +stake*payout (gain), -stake (perte), 0 (égalité remboursée)."""
        self._open_stakes.remove(stake)
        self.balance += pnl
        self.peak_balance = max(self.peak_balance, self.balance)
        if pnl < 0:
            self._consecutive_losses += 1
        elif pnl > 0:
            self._consecutive_losses = 0
