"""
Moteur de backtest pour options binaires à expiration fixe (type Pocket Option).

Règles simulées :
- Le signal est calculé sur la clôture de la bougie ``bar_index``.
- Entrée à l'ouverture de la bougie ``bar_index + latency_bars`` (latence de clic),
  dégradée de ``entry_slippage`` (en prix, toujours contre le trader).
- Sortie à la clôture de la bougie qui termine l'expiration.
- Gain = mise * payout ; perte = mise ; prix de sortie == prix d'entrée → mise remboursée.
- Si la bougie d'entrée ou celle d'expiration manque (week-end, trou de données),
  le trade est annulé et compté dans ``skipped_gaps`` : jamais de résultat inventé.
  Les bougies intermédiaires n'influencent pas le résultat d'une binaire.
- Chaque trade passe par le BinaryRiskManager avant d'être ouvert.

Seuil de rentabilité : win rate > 1 / (1 + payout) (52,08 % à 92 % de payout).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import pandas as pd

from zeus.risk.binary_risk import BinaryRiskManager
from zeus.strategy.base import Signal, SignalType


class BinaryOutcome(str, Enum):
    WIN  = "win"
    LOSS = "loss"
    TIE  = "tie"


@dataclass(frozen=True)
class BinaryBacktestConfig:
    payout: float = 0.92
    expiry_minutes: int = 5
    bar_minutes: int = 1
    latency_bars: int = 1
    entry_slippage: float = 0.0   # en unités de prix

    def __post_init__(self) -> None:
        if not 0 < self.payout <= 1:
            raise ValueError("payout doit être dans ]0, 1]")
        if self.expiry_minutes <= 0 or self.bar_minutes <= 0:
            raise ValueError("expiry_minutes et bar_minutes doivent être > 0")
        if self.expiry_minutes % self.bar_minutes:
            raise ValueError("expiry_minutes doit être un multiple de bar_minutes")
        if self.latency_bars < 1:
            raise ValueError("latency_bars doit être >= 1 (on ne trade pas une bougie déjà close)")
        if self.entry_slippage < 0:
            raise ValueError("entry_slippage doit être >= 0")


@dataclass(frozen=True)
class BinaryTrade:
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    expiry_time: pd.Timestamp
    direction: SignalType
    entry_price: float
    exit_price: float
    stake: float
    outcome: BinaryOutcome
    pnl: float
    reason: str


@dataclass
class BinaryBacktestResult:
    trades: list[BinaryTrade]
    initial_balance: float
    final_balance: float
    payout: float
    rejections: Counter = field(default_factory=Counter)
    skipped_gaps: int = 0

    @property
    def wins(self) -> int:
        return sum(t.outcome is BinaryOutcome.WIN for t in self.trades)

    @property
    def losses(self) -> int:
        return sum(t.outcome is BinaryOutcome.LOSS for t in self.trades)

    @property
    def win_rate(self) -> float:
        decided = self.wins + self.losses
        return self.wins / decided if decided else 0.0

    @property
    def breakeven_win_rate(self) -> float:
        return 1 / (1 + self.payout)

    @property
    def win_rate_lower_bound(self) -> float:
        """Borne basse de Wilson à 95 % : l'edge n'est crédible que si elle dépasse le seuil."""
        n = self.wins + self.losses
        if n == 0:
            return 0.0
        z, p = 1.96, self.wins / n
        centre = p + z * z / (2 * n)
        margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return (centre - margin) / (1 + z * z / n)

    @property
    def max_drawdown(self) -> float:
        balance = peak = self.initial_balance
        worst = 0.0
        for t in self.trades:
            balance += t.pnl
            peak = max(peak, balance)
            worst = max(worst, (peak - balance) / peak)
        return worst

    @property
    def max_consecutive_losses(self) -> int:
        run = best = 0
        for t in self.trades:
            if t.outcome is BinaryOutcome.LOSS:
                run += 1
                best = max(best, run)
            elif t.outcome is BinaryOutcome.WIN:
                run = 0
        return best

    def summary(self) -> dict:
        return {
            "trades": len(self.trades),
            "wins": self.wins,
            "losses": self.losses,
            "ties": len(self.trades) - self.wins - self.losses,
            "win_rate": round(self.win_rate, 4),
            "win_rate_lower_95": round(self.win_rate_lower_bound, 4),
            "breakeven_win_rate": round(self.breakeven_win_rate, 4),
            "pnl": round(self.final_balance - self.initial_balance, 2),
            "max_drawdown": round(self.max_drawdown, 4),
            "max_consecutive_losses": self.max_consecutive_losses,
            "skipped_gaps": self.skipped_gaps,
            "rejections": dict(self.rejections),
        }


def run_binary_backtest(
    df: pd.DataFrame,
    signals: Iterable[Signal],
    config: BinaryBacktestConfig,
    risk: BinaryRiskManager,
) -> BinaryBacktestResult:
    """Simule les signaux dans l'ordre chronologique ; ``df`` est un OHLC à pas régulier."""
    initial_balance = risk.balance
    bar = pd.Timedelta(minutes=config.bar_minutes)
    expiry = pd.Timedelta(minutes=config.expiry_minutes)
    index = df.index
    opens, closes = df["open"].to_numpy(), df["close"].to_numpy()

    trades: list[BinaryTrade] = []
    pending: list[BinaryTrade] = []
    rejections: Counter = Counter()
    skipped_gaps = 0

    for signal in sorted(signals, key=lambda s: s.bar_index):
        if signal.type is SignalType.NONE:
            continue
        signal_time = index[signal.bar_index]
        entry_time = signal_time + bar * config.latency_bars
        last_bar_time = entry_time + expiry - bar
        if entry_time not in index or last_bar_time not in index:
            skipped_gaps += 1
            continue

        _settle(pending, trades, risk, until=entry_time)
        decision = risk.evaluate(entry_time.to_pydatetime(), broker_connected=True)
        if not decision.approved:
            rejections[decision.rejection.value] += 1
            continue

        direction = int(signal.type)
        entry_price = opens[index.get_loc(entry_time)] + direction * config.entry_slippage
        exit_price = closes[index.get_loc(last_bar_time)]
        outcome = _outcome(direction, entry_price, exit_price)
        pnl = {BinaryOutcome.WIN: decision.stake * config.payout,
               BinaryOutcome.LOSS: -decision.stake,
               BinaryOutcome.TIE: 0.0}[outcome]
        risk.record_open(decision.stake)
        pending.append(BinaryTrade(
            signal_time=signal_time, entry_time=entry_time, expiry_time=entry_time + expiry,
            direction=signal.type, entry_price=float(entry_price), exit_price=float(exit_price),
            stake=decision.stake, outcome=outcome, pnl=round(pnl, 2), reason=signal.reason,
        ))

    _settle(pending, trades, risk, until=None)
    return BinaryBacktestResult(
        trades=trades, initial_balance=initial_balance, final_balance=risk.balance,
        payout=config.payout, rejections=rejections, skipped_gaps=skipped_gaps,
    )


def _outcome(direction: int, entry_price: float, exit_price: float) -> BinaryOutcome:
    move = (exit_price - entry_price) * direction
    if move > 0:
        return BinaryOutcome.WIN
    if move < 0:
        return BinaryOutcome.LOSS
    return BinaryOutcome.TIE


def _settle(pending: list[BinaryTrade], trades: list[BinaryTrade],
            risk: BinaryRiskManager, until: pd.Timestamp | None) -> None:
    """Clôture (dans l'ordre d'expiration) les trades expirés à ``until`` (tous si None)."""
    for trade in sorted(pending, key=lambda t: t.expiry_time):
        if until is not None and trade.expiry_time > until:
            break
        pending.remove(trade)
        risk.record_close(trade.stake, trade.pnl)
        trades.append(trade)
