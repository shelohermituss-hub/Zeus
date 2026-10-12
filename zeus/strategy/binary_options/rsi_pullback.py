"""
Pullback RSI(2) dans la tendance (Connors).

Idée : dans une tendance (clôture au-dessus/au-dessous d'une EMA longue), un
RSI très court en excès contre la tendance marque un repli qui reprend souvent.
``trend_period = 0`` désactive le filtre de tendance (retour à la moyenne pur).

- CALL : RSI < rsi_low  (et clôture > EMA si filtre)
- PUT  : RSI > rsi_high (et clôture < EMA si filtre)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from zeus.strategy.binary_options.mean_reversion import rsi


@dataclass(frozen=True)
class RsiPullbackConfig:
    rsi_period: int = 2
    rsi_low: float = 5.0
    rsi_high: float = 95.0
    trend_period: int = 200

    def __post_init__(self) -> None:
        if self.rsi_period < 2 or self.trend_period < 0 or self.trend_period == 1:
            raise ValueError("rsi_period >= 2 et trend_period = 0 ou >= 2 requis")
        if not 0 < self.rsi_low < 50 < self.rsi_high < 100:
            raise ValueError("il faut 0 < rsi_low < 50 < rsi_high < 100")


def directions(df: pd.DataFrame, config: RsiPullbackConfig) -> pd.Series:
    close = df["close"]
    r = rsi(close, config.rsi_period)
    call, put = r < config.rsi_low, r > config.rsi_high
    if config.trend_period:
        ema = close.ewm(span=config.trend_period, min_periods=config.trend_period, adjust=False).mean()
        call, put = call & (close > ema), put & (close < ema)
    return call.astype(int) - put.astype(int)
