"""
Moteur multi-symboles P11-V4 — boucle live/paper pilotée par config/p11_v4.yaml.

Flux par tick, pour chaque symbole activé :
    1. Récupère les M1 récents via le connecteur (MT5 en live, mock en test).
    2. Vérifie les sorties du trade ouvert (échelle TP / BE / SL / runner).
    3. Resample vers le timeframe de zones du groupe, exécute SDStrategy.
    4. Nouveau signal sur la dernière barre M1 close →
         caps par symbole/direction → PropFirmGuard → dimensionnement → entrée.

Règles de sécurité (fail closed) :
    - un seul trade ouvert par symbole ;
    - aucune entrée si le guard refuse (limite journalière, DD, heure) ;
    - risque fixe en % du solde initial (pas de compounding) ;
    - toute erreur de données → tick ignoré, jamais d'ordre à l'aveugle.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd

from zeus.backtest.data_loader import resample_ohlcv
from zeus.exchange.connector import ExchangeConnector
from zeus.live.order_router import OrderRouter, PaperRouter
from zeus.live.p11_config import ExitsConfig, GroupConfig, P11Config, SymbolConfig
from zeus.risk.propfirm import PropFirmGuard
from zeus.strategy.supply_demand.sd_strategy import SDSignal, SDStrategy
from zeus.strategy.supply_demand.wyckoff import WyckoffDetector
from zeus.strategy.supply_demand.zone_detector import ZoneDetector
from zeus.utils.logger import logger

_M1_LIMIT = 3_000          # barres M1 par fetch (~2 jours) — zones + Wyckoff lookback
_ZONE_TF_PANDAS = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h"}


@dataclass
class P11Trade:
    """Trade ouvert géré par l'échelle de sorties du groupe."""
    symbol:         str
    direction:      str                 # "long" | "short"
    entry_price:    float
    sl:             float               # stop courant (peut monter à BE)
    sl_dist:        float               # distance initiale entrée→SL (= 1R)
    exits:          ExitsConfig
    opened_at:      pd.Timestamp
    remaining:      float = 1.0         # fraction de position restante
    realized_r:     float = 0.0         # R encaissé par les sorties partielles
    tp1_done:       bool = False
    tp2_done:       bool = False
    tp3_done:       bool = False
    lots:           float = 0.0         # taille broker (0 en paper)
    ticket:         int = 0             # ticket de position MT5 (0 en paper)

    def r_at(self, price: float) -> float:
        sign = 1.0 if self.direction == "long" else -1.0
        return sign * (price - self.entry_price) / self.sl_dist

    def level(self, r: float) -> float:
        sign = 1.0 if self.direction == "long" else -1.0
        return self.entry_price + sign * r * self.sl_dist


@dataclass
class SymbolState:
    cfg:            SymbolConfig
    group:          GroupConfig
    strategy:       SDStrategy
    open_trade:     Optional[P11Trade] = None
    seen_signals:   set = field(default_factory=set)
    # pertes par direction — clés "YYYY-MM-DD|long" / "YYYY-MM|short"
    day_losses:     dict = field(default_factory=dict)
    month_losses:   dict = field(default_factory=dict)


def _build_strategy(group: GroupConfig) -> SDStrategy:
    return SDStrategy(
        zone_detector    = ZoneDetector(),
        wyckoff_detector = WyckoffDetector(**group.wyckoff),
        risk_reward      = group.exits.runner_rr,
        **{k: v for k, v in group.signals.items()},
    )


class P11Engine:
    """Boucle de trading du portefeuille P11-V4."""

    def __init__(
        self,
        config:    P11Config,
        connector: ExchangeConnector,
        notifier=None,
        router:    Optional[OrderRouter] = None,
    ) -> None:
        self.config    = config
        self.connector = connector
        self.notifier  = notifier
        self.router    = router if router is not None else PaperRouter()
        self._running  = False

        self.guard = PropFirmGuard(config.propfirm, initial_balance=config.account_balance)
        self.risk_usd = self.guard.risk_amount_usd()

        self.states: dict[str, SymbolState] = {}
        for name, sc in config.symbols.items():
            group = config.groups[sc.group]
            self.states[name] = SymbolState(
                cfg=sc, group=group, strategy=_build_strategy(group),
            )

        self.closed_trades: list[dict] = []

    # ── Boucle publique ───────────────────────────────────────────────────────

    def stop(self) -> None:
        self._running = False

    def run(self, max_ticks: Optional[int] = None) -> None:
        self._running = True
        logger.info(
            "P11 engine started",
            mode=self.config.mode,
            symbols=",".join(self.states),
            risk_usd=self.risk_usd,
        )
        tick = 0
        try:
            while self._running:
                if max_ticks is not None and tick >= max_ticks:
                    break
                self.tick()
                tick += 1
                if self._running and (max_ticks is None or tick < max_ticks):
                    time.sleep(self.config.execution.poll_seconds)
        except KeyboardInterrupt:
            logger.info("P11 engine stopped by user")
        finally:
            self._running = False

    # ── Tick ──────────────────────────────────────────────────────────────────

    def tick(self, now: Optional[pd.Timestamp] = None) -> None:
        """Un cycle complet sur tous les symboles.  `now` injectable en test."""
        for name, st in self.states.items():
            try:
                self._tick_symbol(name, st, now)
            except Exception as exc:               # fail closed : jamais d'ordre sur erreur
                logger.error("Tick error", symbol=name, error=str(exc))

    def _tick_symbol(self, name: str, st: SymbolState, now: Optional[pd.Timestamp]) -> None:
        m1 = self.connector.fetch_ohlcv(st.cfg.broker_symbol, "1m", _M1_LIMIT)
        if m1 is None or len(m1) < 300:
            logger.warning("Insufficient M1 data — symbol skipped", symbol=name)
            return

        ts_now = now if now is not None else m1.index[-1]

        # 1. Sorties du trade ouvert
        if st.open_trade is not None:
            self._manage_exits(name, st, m1, ts_now)

        # 2. Détection de signal
        if st.open_trade is not None:
            return
        zone_tf = _ZONE_TF_PANDAS[st.group.zone_timeframe]
        zone_df = resample_ohlcv(m1, zone_tf)
        if len(zone_df) < 60:
            return
        signals = st.strategy.run(zone_df, m1)
        if not signals:
            return

        last_complete_ts = m1.index[-2]
        sig = next(
            (s for s in reversed(signals)
             if s.formed_at == last_complete_ts and s.formed_at not in st.seen_signals),
            None,
        )
        if sig is None:
            return
        st.seen_signals.add(sig.formed_at)

        # 3. Caps par symbole/direction (identiques au backtest)
        day_key   = f"{sig.formed_at.strftime('%Y-%m-%d')}|{sig.direction}"
        month_key = f"{sig.formed_at.strftime('%Y-%m')}|{sig.direction}"
        if st.day_losses.get(day_key, 0) >= st.group.caps["max_daily_losses"]:
            logger.info("Signal refusé — cap journalier symbole", symbol=name)
            return
        if st.month_losses.get(month_key, 0) >= st.group.caps["max_monthly_losses"]:
            logger.info("Signal refusé — cap mensuel symbole", symbol=name)
            return

        # 4. Guard propfirm (portefeuille)
        if not self.guard.can_trade(ts_now):
            logger.info("Signal refusé — guard propfirm", symbol=name, ts=str(ts_now))
            return

        self._enter(name, st, sig, m1)

    # ── Entrée ────────────────────────────────────────────────────────────────

    def _enter(self, name: str, st: SymbolState, sig: SDSignal, m1: pd.DataFrame) -> None:
        entry_price = float(m1["close"].iloc[-1])   # marché au tick suivant le signal
        sl = float(sig.stop_loss)
        sl_dist = abs(entry_price - sl)
        if sl_dist <= 0:
            return
        # SL du mauvais côté après slippage d'entrée → refus (fail closed)
        if (sig.direction == "long") != (sl < entry_price):
            logger.warning("SL incohérent — entrée refusée", symbol=name)
            return

        fill = self.router.open_position(
            st.cfg.broker_symbol, sig.direction, sl, self.risk_usd, entry_price,
        )
        if fill is None:
            logger.warning("Ordre refusé par le routeur — pas de trade", symbol=name)
            return
        entry_price = fill.price
        sl_dist = abs(entry_price - sl)
        if sl_dist <= 0:
            return

        trade = P11Trade(
            symbol=name, direction=sig.direction,
            entry_price=entry_price, sl=sl, sl_dist=sl_dist,
            exits=st.group.exits, opened_at=sig.formed_at,
            lots=fill.lots, ticket=fill.ticket,
        )
        st.open_trade = trade
        logger.info(
            "ENTRÉE", symbol=name, direction=sig.direction,
            entry=round(entry_price, 5), sl=round(sl, 5),
            risk_usd=self.risk_usd, wyckoff=round(sig.wyckoff_score, 2),
        )
        if self.notifier and self.notifier.enabled:
            self.notifier.send(
                f"🟢 ENTRÉE {name} {sig.direction} @ {entry_price:.5f} "
                f"SL {sl:.5f} (risque {self.risk_usd:.0f}$)"
            )

    # ── Sorties ───────────────────────────────────────────────────────────────

    def _manage_exits(self, name: str, st: SymbolState, m1: pd.DataFrame,
                      ts_now: pd.Timestamp) -> None:
        tr = st.open_trade
        assert tr is not None
        bar = m1.iloc[-1]
        hi, lo = float(bar["high"]), float(bar["low"])
        is_long = tr.direction == "long"
        ex = tr.exits

        def hit(level: float) -> bool:
            return (hi >= level) if is_long else (lo <= level)

        def sl_hit() -> bool:
            return (lo <= tr.sl) if is_long else (hi >= tr.sl)

        # SL / BE d'abord (résolution pessimiste, comme le backtest)
        if sl_hit():
            # En live, le SL attaché côté broker a déjà fermé le restant.
            r_exit = tr.r_at(tr.sl)
            tr.realized_r += tr.remaining * r_exit
            tr.remaining = 0.0
            self._close(name, st, ts_now, reason="SL/BE")
            return

        # Échelle de TP — chaque niveau ne se déclenche qu'une fois
        if not tr.tp1_done and hit(tr.level(ex.tp1_r)):
            tr.tp1_done = True
            if ex.tp1_close_pct > 0:
                closed = ex.tp1_close_pct
                tr.realized_r += closed * ex.tp1_r
                tr.remaining -= closed
                self.router.partial_close(st.cfg.broker_symbol, tr.ticket,
                                          tr.direction, tr.lots * closed)
            tr.sl = tr.entry_price          # break-even
            self.router.modify_sl(st.cfg.broker_symbol, tr.ticket, tr.sl)
            logger.info("TP1 — SL→BE", symbol=name, closed_pct=ex.tp1_close_pct)

        if not tr.tp2_done and hit(tr.level(ex.tp2_r)):
            tr.tp2_done = True
            target_closed = ex.tp2_cumulative_pct
            closed = max(0.0, target_closed - (1.0 - tr.remaining))
            tr.realized_r += closed * ex.tp2_r
            tr.remaining -= closed
            if closed > 0:
                self.router.partial_close(st.cfg.broker_symbol, tr.ticket,
                                          tr.direction, tr.lots * closed)
            # Le reliquat (runner) verrouille le gain du palier TP1 au lieu de
            # rester au BE — aligné sur zeus/backtest/sd_simulation.py, la
            # référence utilisée pour valider les performances du portefeuille.
            tr.sl = tr.level(ex.tp1_r)
            self.router.modify_sl(st.cfg.broker_symbol, tr.ticket, tr.sl)
            logger.info("TP2", symbol=name, closed_pct=closed)

        if ex.tp3_r > 0 and not tr.tp3_done and hit(tr.level(ex.tp3_r)):
            tr.tp3_done = True
            target_closed = ex.tp3_cumulative_pct
            closed = max(0.0, target_closed - (1.0 - tr.remaining))
            tr.realized_r += closed * ex.tp3_r
            tr.remaining -= closed
            if closed > 0:
                self.router.partial_close(st.cfg.broker_symbol, tr.ticket,
                                          tr.direction, tr.lots * closed)
            # Idem : le reliquat verrouille le gain du palier TP2 après TP3.
            tr.sl = tr.level(ex.tp2_r)
            self.router.modify_sl(st.cfg.broker_symbol, tr.ticket, tr.sl)
            logger.info("TP3", symbol=name, closed_pct=closed)

        if hit(tr.level(ex.runner_rr)):
            tr.realized_r += tr.remaining * ex.runner_rr
            if tr.remaining > 0:
                self.router.close_position(st.cfg.broker_symbol, tr.ticket,
                                           tr.direction, tr.lots * tr.remaining)
            tr.remaining = 0.0
            self._close(name, st, ts_now, reason="RUNNER")

    def _close(self, name: str, st: SymbolState, ts_now: pd.Timestamp, reason: str) -> None:
        tr = st.open_trade
        assert tr is not None
        pnl_r = tr.realized_r
        pnl_usd = pnl_r * self.risk_usd
        outcome = "win" if pnl_r > 0 else ("loss" if pnl_r < 0 else "scratch")

        self.guard.on_trade_closed(ts_now, pnl_usd)

        if outcome == "loss":
            day_key = f"{tr.opened_at.strftime('%Y-%m-%d')}|{tr.direction}"
            month_key = f"{tr.opened_at.strftime('%Y-%m')}|{tr.direction}"
            st.day_losses[day_key] = st.day_losses.get(day_key, 0) + 1
            st.month_losses[month_key] = st.month_losses.get(month_key, 0) + 1

        self.closed_trades.append(dict(
            symbol=name, direction=tr.direction, opened_at=str(tr.opened_at),
            closed_at=str(ts_now), pnl_r=round(pnl_r, 3),
            pnl_usd=round(pnl_usd, 2), reason=reason, outcome=outcome,
        ))
        logger.info(
            "SORTIE", symbol=name, reason=reason,
            pnl_r=round(pnl_r, 3), pnl_usd=round(pnl_usd, 2),
            equity=round(self.guard.state.equity, 2),
            halted_day=self.guard.state.halted_for_day,
            halted_perm=self.guard.state.halted_permanently,
        )
        if self.notifier and self.notifier.enabled:
            emoji = "✅" if pnl_usd > 0 else "🔻"
            self.notifier.send(
                f"{emoji} SORTIE {name} {tr.direction} [{reason}] "
                f"{pnl_usd:+.0f}$ ({pnl_r:+.2f}R)"
            )
        st.open_trade = None
