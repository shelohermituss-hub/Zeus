"""
Retour à la moyenne sur excès court terme (M1).

Idée : dans un marché calme (session asiatique par défaut), une clôture hors
d'une bande de Bollinger large avec un RSI court extrême est souvent suivie
d'un retour partiel vers la moyenne dans les minutes suivantes.

- Clôture > bande haute et RSI > rsi_high → PUT (-1)
- Clôture < bande basse et RSI < rsi_low  → CALL (+1)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from zeus.strategy.binary_options import in_utc_hours


@dataclass(frozen=True)
class MeanReversionConfig:
    bb_period: int = 20
    bb_std: float = 2.5
    rsi_period: int = 7
    rsi_low: float = 20.0
    rsi_high: float = 80.0
    session_start_utc: int = 0
    session_end_utc: int = 6

    def __post_init__(self) -> None:
        if self.bb_period < 2 or self.rsi_period < 2 or self.bb_std <= 0:
            raise ValueError("bb_period, rsi_period >= 2 et bb_std > 0 requis")
        if not 0 < self.rsi_low < 50 < self.rsi_high < 100:
            raise ValueError("il faut 0 < rsi_low < 50 < rsi_high < 100")
        if not (0 <= self.session_start_utc <= 23 and 0 <= self.session_end_utc <= 24):
            raise ValueError("heures de session hors de [0, 24]")


def rsi(close: pd.Series, period: int) -> pd.Series:
    """RSI de Wilder ; NaN tant que la fenêtre n'est pas remplie."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def directions(df: pd.DataFrame, config: MeanReversionConfig) -> pd.Series:
    close = df["close"]
    mid = close.rolling(config.bb_period).mean()
    width = close.rolling(config.bb_period).std() * config.bb_std
    r = rsi(close, config.rsi_period)
    session = in_utc_hours(df.index, config.session_start_utc, config.session_end_utc)

    put = (close > mid + width) & (r > config.rsi_high) & session
    call = (close < mid - width) & (r < config.rsi_low) & session
    return call.astype(int) - put.astype(int)
