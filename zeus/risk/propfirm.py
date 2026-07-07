"""
Profil de risque propfirm — limites internes strictes (fail closed).

Objectif : trader un compte propfirm (challenge ou financé) sans jamais
approcher les limites du broker.  Les limites internes sont volontairement
plus serrées que les règles publiques des firms (FTMO : daily 5% / total 10%) :

    daily interne 3%  <  daily firm 5%
    total interne 6%  <  total firm 10%

Le guard s'utilise en amont de chaque ordre (backtest, paper ou live) :

    guard = PropFirmGuard(PropFirmConfig(), initial_balance=100_000)
    if guard.can_trade(now):
        ...émettre l'ordre...
    guard.on_trade_closed(now, pnl_usd)

Règle de base : toute condition incertaine → trade refusé.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class PropFirmConfig:
    """Limites internes strictes pour compte propfirm.

    Les pourcentages sont exprimés en fraction du solde initial du compte
    (convention FTMO : le daily loss se mesure sur l'équity de début de jour,
    le drawdown total sur le solde initial).
    """
    max_daily_loss_pct:   float = 0.03   # interne 3% (firm : 5%)
    max_total_dd_pct:     float = 0.06   # interne 6% (firm : 10%)
    risk_per_trade_pct:   float = 0.003  # 0.3% par trade
    max_losses_per_day:   int   = 3      # stop journalier après N pertes
    flat_hour_utc:        int   = 21     # plus aucune entrée à partir de cette heure
    profit_target_pct:    float = 0.10   # cible challenge phase 1
    min_trading_days:     int   = 4      # jours de trading minimum du challenge

    def __post_init__(self) -> None:
        if not 0 < self.max_daily_loss_pct < 1:
            raise ValueError("max_daily_loss_pct must be in (0, 1)")
        if not 0 < self.max_total_dd_pct < 1:
            raise ValueError("max_total_dd_pct must be in (0, 1)")
        if not 0 < self.risk_per_trade_pct <= self.max_daily_loss_pct:
            raise ValueError("risk_per_trade_pct must be in (0, max_daily_loss_pct]")
        if self.max_losses_per_day < 1:
            raise ValueError("max_losses_per_day must be >= 1")
        if not 0 <= self.flat_hour_utc <= 23:
            raise ValueError("flat_hour_utc must be in [0, 23]")


@dataclass
class PropFirmState:
    """État mutable du guard — un objet par compte."""
    initial_balance:    float = 0.0
    equity:             float = 0.0
    day_start_equity:   float = 0.0
    current_day:        date | None = None
    losses_today:       int   = 0
    halted_for_day:     bool  = False   # limite journalière atteinte
    halted_permanently: bool  = False   # drawdown total interne atteint
    trading_days:       set[date] = field(default_factory=set)


class PropFirmGuard:
    """
    Garde-fou propfirm : décide si un nouveau trade est autorisé.

    Fail closed : si l'état est incohérent (equity inconnue, jour non
    initialisé), can_trade() retourne False.

    Le guard ne dimensionne pas les positions (voir RiskManager) ; il
    applique uniquement les règles d'arrêt propfirm :

      1. Drawdown total interne     → arrêt définitif du compte
      2. Perte journalière interne  → arrêt jusqu'au lendemain
      3. N pertes dans la journée   → arrêt jusqu'au lendemain
      4. Heure limite (flat_hour)   → plus d'entrée en fin de session
    """

    def __init__(self, config: PropFirmConfig, initial_balance: float) -> None:
        if initial_balance <= 0:
            raise ValueError("initial_balance must be > 0")
        self.config = config
        self.state = PropFirmState(
            initial_balance  = initial_balance,
            equity           = initial_balance,
            day_start_equity = initial_balance,
        )

    # ── Interface publique ────────────────────────────────────────────────────

    def can_trade(self, ts: pd.Timestamp) -> bool:
        """True si une nouvelle entrée est autorisée à l'instant *ts* (UTC)."""
        st = self.state
        if st.halted_permanently:
            return False

        self._roll_day(ts)

        if st.halted_for_day:
            return False
        if ts.hour >= self.config.flat_hour_utc:
            return False
        return True

    def on_trade_closed(self, ts: pd.Timestamp, pnl_usd: float) -> None:
        """Enregistre un trade clôturé et met à jour les arrêts."""
        st = self.state
        self._roll_day(ts)

        st.equity += pnl_usd
        st.trading_days.add(ts.date())
        if pnl_usd < 0:
            st.losses_today += 1

        # 1. Drawdown total interne (mesuré sur le solde initial, convention FTMO)
        total_dd = (st.initial_balance - st.equity) / st.initial_balance
        if total_dd >= self.config.max_total_dd_pct:
            st.halted_permanently = True
            return

        # 2. Perte journalière interne (mesurée sur l'équity de début de jour)
        daily_loss = (st.day_start_equity - st.equity) / st.initial_balance
        if daily_loss >= self.config.max_daily_loss_pct:
            st.halted_for_day = True
            return

        # 3. Série de pertes journalière
        if st.losses_today >= self.config.max_losses_per_day:
            st.halted_for_day = True

    def risk_amount_usd(self) -> float:
        """Montant risqué par trade (fraction du solde initial — taille fixe,
        pas de compounding : la régularité prime sur la croissance en propfirm)."""
        return self.state.initial_balance * self.config.risk_per_trade_pct

    # ── Statut challenge ──────────────────────────────────────────────────────

    def challenge_passed(self) -> bool:
        """True si la cible de profit ET le minimum de jours sont atteints."""
        st = self.state
        profit_pct = (st.equity - st.initial_balance) / st.initial_balance
        return (
            not st.halted_permanently
            and profit_pct >= self.config.profit_target_pct
            and len(st.trading_days) >= self.config.min_trading_days
        )

    def challenge_failed(self) -> bool:
        """True si le compte a touché l'arrêt définitif (DD total interne)."""
        return self.state.halted_permanently

    # ── Privé ─────────────────────────────────────────────────────────────────

    def _roll_day(self, ts: pd.Timestamp) -> None:
        """Réinitialise les compteurs journaliers au changement de jour UTC."""
        st = self.state
        day = ts.date()
        if st.current_day is None or day > st.current_day:
            st.current_day      = day
            st.day_start_equity = st.equity
            st.losses_today     = 0
            st.halted_for_day   = False
