"""
Open US ORB (Opening Range Breakout) — approche alternative à
open_us_strategy.OpenUSStrategy (sweep + retournement).

Logique volontairement différente pour tester si un edge existe dans une
direction structurellement opposée à Wyckoff : au lieu d'attendre un
retournement APRÈS un sweep, on suit la CASSURE du range construit sur les
`or_minutes` premières minutes après l'ouverture US (continuation directe,
pas de confirmation de retournement).

Un seul signal par jour (première cassure valide dans la fenêtre de
recherche). Satisfait le même Protocol Tradeable que OpenUSSignal — donc
réutilise directement simulate_all/compute_metrics/print_report sans aucune
modification du moteur de simulation.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional

import pandas as pd

DEFAULT_SESSION_START  = dt.time(9, 30)
DEFAULT_OR_MINUTES     = 15     # durée de construction du range
DEFAULT_SEARCH_MINUTES = 90     # durée pendant laquelle on cherche une cassure après la fin du range
DEFAULT_MIN_RANGE_PIPS = 0.0    # filtre range trop plat (en unités de prix, pas pips — XAUUSD)
DEFAULT_RISK_REWARD    = 2.0


@dataclass(frozen=True)
class ORBSignal:
    direction:    str
    bar_index:    int
    entry_price:  float
    stop_loss:    float
    take_profit:  float
    risk_reward:  float
    zone_score:   float          # constante (pas de scoring ORB) — requis par le Protocol Tradeable
    formed_at:    pd.Timestamp
    or_high:      float
    or_low:       float


class OpenUSORBStrategy:
    def __init__(
        self,
        session_start:   dt.time = DEFAULT_SESSION_START,
        or_minutes:      int     = DEFAULT_OR_MINUTES,
        search_minutes:  int     = DEFAULT_SEARCH_MINUTES,
        min_range_price: float   = DEFAULT_MIN_RANGE_PIPS,
        risk_reward:     float   = DEFAULT_RISK_REWARD,
    ) -> None:
        self.session_start   = session_start
        self.or_minutes      = or_minutes
        self.search_minutes  = search_minutes
        self.min_range_price = min_range_price
        self.risk_reward     = risk_reward

    def generate_signals(self, m1_df: pd.DataFrame) -> list[ORBSignal]:
        if m1_df.index.tz is not None:
            raise ValueError("m1_df doit avoir un index tz-naive (UTC)")

        ny_index = m1_df.index.tz_localize("UTC").tz_convert("America/New_York")
        start_min = self.session_start.hour * 60 + self.session_start.minute
        or_end_min = start_min + self.or_minutes
        search_end_min = or_end_min + self.search_minutes

        highs = m1_df["high"].values.astype(float)
        lows  = m1_df["low"].values.astype(float)
        closes = m1_df["close"].values.astype(float)

        signals: list[ORBSignal] = []
        cur_day: Optional[dt.date] = None
        or_high = or_low = 0.0
        or_built = False
        day_done = False

        for i in range(len(m1_df)):
            ny_ts = ny_index[i]
            if ny_ts.weekday() >= 5:
                continue
            cur_min = ny_ts.hour * 60 + ny_ts.minute

            if ny_ts.date() != cur_day:
                cur_day = ny_ts.date()
                or_high = or_low = 0.0
                or_built = False
                day_done = False

            if day_done or cur_min < start_min:
                continue

            if cur_min < or_end_min:
                if not or_built:
                    or_high, or_low = highs[i], lows[i]
                    or_built = True
                else:
                    or_high = max(or_high, highs[i])
                    or_low  = min(or_low, lows[i])
                continue

            if not or_built or cur_min >= search_end_min:
                day_done = True
                continue

            rng = or_high - or_low
            if rng < self.min_range_price:
                continue

            direction = None
            if closes[i] > or_high:
                direction = "long"
            elif closes[i] < or_low:
                direction = "short"
            if direction is None:
                continue

            entry = closes[i]
            sl    = or_low if direction == "long" else or_high
            sl_dist = abs(entry - sl)
            if sl_dist <= 0:
                continue
            tp = entry + (self.risk_reward * sl_dist if direction == "long" else -self.risk_reward * sl_dist)

            signals.append(ORBSignal(
                direction=direction, bar_index=i, entry_price=entry, stop_loss=sl,
                take_profit=tp, risk_reward=self.risk_reward, zone_score=5.0,
                formed_at=m1_df.index[i], or_high=or_high, or_low=or_low,
            ))
            day_done = True

        return signals
