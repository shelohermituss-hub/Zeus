"""
Open US strategy — sweep puis retournement sur l'ouverture actions US (9h30 ET).

Réutilise WyckoffDetector (zeus/strategy/supply_demand/wyckoff.py) **sans
aucune modification** : le pattern accumulation -> manipulation (spring /
upthrust) -> MSS EST la logique "sweep puis retournement" demandée. La seule
chose ajoutée ici est un filtre temporel : on n'accepte un signal que si sa
barre de MSS tombe dans la fenêtre d'ouverture US.

Pourquoi ce choix (parmi ORB / sweep-retournement / continuation) :
WyckoffDetector est la seule des trois logiques déjà validée dans ce dépôt
(utilisée en production par SDStrategy) — la réutiliser ici respecte le
principe du projet ("ne jamais dupliquer une logique déjà validée").

Gestion du fuseau horaire
--------------------------
Contrairement à la version MQL5 (mt5/ZeusOpenMomentum), qui doit recalculer
la règle DST US à la main (MetaTrader n'a pas de fuseaux horaires nommés),
ici on utilise directement pandas tz-aware avec "America/New_York" : la
conversion gère nativement les transitions DST US, donc pas de logique DST
à maintenir séparément.

Un seul signal par jour (le premier valide dans la fenêtre) — cohérent avec
la version MQL5 et avec l'idée qu'une seule opportunité d'ouverture existe
par session.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from zeus.strategy.supply_demand.pivot_candle import PivotSide
from zeus.strategy.supply_demand.wyckoff import WyckoffDetector, WyckoffPattern

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_SESSION_START   = dt.time(9, 30)   # ouverture cash US (heure de New York)
DEFAULT_WINDOW_MINUTES  = 90               # durée pendant laquelle un nouveau signal est accepté
DEFAULT_MIN_SCORE       = 4.0              # même seuil que SDStrategy (MIN_WYCKOFF_SCORE)
DEFAULT_RISK_REWARD     = 3.0


@dataclass(frozen=True)
class OpenUSSignal:
    """
    Signal "Open US" — satisfait le Protocol Tradeable de sd_simulation.py
    (direction, bar_index, stop_loss, risk_reward, formed_at, zone_score)
    sans modification du moteur de simulation existant.

    zone_score reprend le score du pattern Wyckoff (pas de zone S&D ici —
    nommage conservé uniquement pour la compatibilité structurelle avec
    simulate_trade/simulate_all).
    """
    direction:    str            # "long" ou "short"
    bar_index:    int            # index M1 de la barre de MSS
    entry_price:  float          # wyckoff.mss_close (le fill réel = open barre suivante, géré par le moteur)
    stop_loss:    float          # wyckoff.manip_extreme
    take_profit:  float
    risk_reward:  float
    zone_score:   float          # = wyckoff.score (0-10)
    formed_at:    pd.Timestamp
    wyckoff:      WyckoffPattern


class OpenUSStrategy:
    """
    Génère des OpenUSSignal en scannant uniquement les barres M1 tombant
    dans la fenêtre [session_start, session_start + window_minutes) heure
    de New York, un seul signal accepté par jour (le premier valide).
    """

    def __init__(
        self,
        detector:        Optional[WyckoffDetector] = None,
        session_start:   dt.time = DEFAULT_SESSION_START,
        window_minutes:  int     = DEFAULT_WINDOW_MINUTES,
        min_score:       float   = DEFAULT_MIN_SCORE,
        risk_reward:     float   = DEFAULT_RISK_REWARD,
    ) -> None:
        self.detector       = detector or WyckoffDetector()
        self.session_start  = session_start
        self.window_minutes = window_minutes
        self.min_score       = min_score
        self.risk_reward     = risk_reward

    def _in_window(self, ny_time: dt.time) -> bool:
        start_minutes = self.session_start.hour * 60 + self.session_start.minute
        end_minutes    = start_minutes + self.window_minutes
        cur_minutes    = ny_time.hour * 60 + ny_time.minute
        return start_minutes <= cur_minutes < end_minutes

    def generate_signals(self, m1_df: pd.DataFrame) -> list[OpenUSSignal]:
        """
        Scanne m1_df (index UTC tz-naive, colonnes open/high/low/close) et
        retourne la liste des signaux Open US détectés.
        """
        if m1_df.index.tz is not None:
            raise ValueError("m1_df doit avoir un index tz-naive (UTC), comme data_loader.parse_histdata_csv")

        ny_index = m1_df.index.tz_localize("UTC").tz_convert("America/New_York")

        h_arr  = m1_df["high"].values.astype(float)
        l_arr  = m1_df["low"].values.astype(float)
        c_arr  = m1_df["close"].values.astype(float)
        o_arr  = m1_df["open"].values.astype(float)

        signals: list[OpenUSSignal] = []
        last_signal_day: Optional[dt.date] = None

        for i in range(len(m1_df)):
            ny_ts = ny_index[i]
            if ny_ts.weekday() >= 5:
                continue
            if not self._in_window(ny_ts.time()):
                continue
            if last_signal_day == ny_ts.date():
                continue   # déjà un signal ce jour-là

            end_idx = i + 1
            pattern = self._best_pattern(h_arr, l_arr, c_arr, o_arr, m1_df.index, end_idx)
            if pattern is None or pattern.score < self.min_score:
                continue

            direction = "long" if pattern.side == PivotSide.DEMAND else "short"
            entry     = pattern.entry_price
            sl        = pattern.stop_loss_price
            sl_dist   = abs(entry - sl)
            if sl_dist <= 0:
                continue
            tp = entry + (self.risk_reward * sl_dist if direction == "long" else -self.risk_reward * sl_dist)

            signals.append(OpenUSSignal(
                direction   = direction,
                bar_index   = i,
                entry_price = entry,
                stop_loss   = sl,
                take_profit = tp,
                risk_reward = self.risk_reward,
                zone_score  = pattern.score,
                formed_at   = m1_df.index[i],
                wyckoff     = pattern,
            ))
            last_signal_day = ny_ts.date()

        return signals

    def _best_pattern(self, h_arr, l_arr, c_arr, o_arr, df_index, end_idx) -> Optional[WyckoffPattern]:
        """Essaie DEMAND puis SUPPLY, retourne le meilleur score si les deux existent."""
        demand = self.detector.detect_fast(h_arr, l_arr, c_arr, o_arr, df_index, PivotSide.DEMAND, end_idx)
        supply = self.detector.detect_fast(h_arr, l_arr, c_arr, o_arr, df_index, PivotSide.SUPPLY, end_idx)
        if demand and supply:
            return demand if demand.score >= supply.score else supply
        return demand or supply
