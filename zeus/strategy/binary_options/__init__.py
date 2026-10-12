"""
Stratégies d'options binaires à expiration fixe.

Chaque stratégie expose ``directions(df, config) -> pd.Series`` : +1 (CALL),
-1 (PUT) ou 0 par bougie, calculé uniquement avec les données jusqu'à la
clôture de cette bougie. Le même appel sert au backtest (série complète) et au
live/paper (dernière valeur), donc le moteur de décision est identique.
"""
from __future__ import annotations

import pandas as pd

from zeus.strategy.base import Signal, SignalType


def to_signals(directions: pd.Series, reason: str) -> list[Signal]:
    """Convertit une série de directions en Signals pour le moteur de backtest."""
    positions = directions.to_numpy().nonzero()[0]
    return [
        Signal(type=SignalType(int(directions.iat[i])), confidence=1.0, reason=reason, bar_index=int(i))
        for i in positions
    ]


def in_utc_hours(index: pd.DatetimeIndex, start_hour: int, end_hour: int) -> pd.Series:
    """True si l'heure UTC de la bougie est dans [start_hour, end_hour[ (fenêtre pouvant passer minuit)."""
    hours = pd.Series(index.hour, index=index)
    if start_hour <= end_hour:
        return (hours >= start_hour) & (hours < end_hour)
    return (hours >= start_hour) | (hours < end_hour)
