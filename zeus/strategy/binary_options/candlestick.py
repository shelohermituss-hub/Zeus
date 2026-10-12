"""
Patterns de chandeliers japonais (module zeus.strategy.candlestick_patterns) pour binaires.

Le pattern donne la direction ; deux filtres optionnels :
- ``use_rsi`` : confirmation RSI d'origine des robots MetaQuotes (RSI 37, 40/60) ;
- ``band_std`` > 0 : le pattern doit se former à un excès — plus bas sous la bande de
  Bollinger basse (CALL) ou plus haut au-dessus de la bande haute (PUT).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from zeus.strategy.candlestick_patterns import PATTERNS, CandlestickRsiStrategy


@dataclass(frozen=True)
class CandlestickConfig:
    pattern: str = "engulfing"
    use_rsi: bool = False
    band_std: float = 0.0
    band_period: int = 20

    def __post_init__(self) -> None:
        if self.pattern not in PATTERNS:
            raise ValueError(f"pattern inconnu : {self.pattern}")
        if self.band_std < 0 or self.band_period < 2:
            raise ValueError("band_std >= 0 et band_period >= 2 requis")


def directions(df: pd.DataFrame, config: CandlestickConfig) -> pd.Series:
    detector = (CandlestickRsiStrategy(config.pattern) if config.use_rsi
                else CandlestickRsiStrategy(config.pattern, rsi_buy_max=101.0, rsi_sell_min=-1.0))
    d = pd.Series(0, index=df.index)
    for s in detector.run(df):
        d.iat[s.bar_index] = 1 if s.direction == "long" else -1
    if config.band_std > 0:
        mid = df["close"].rolling(config.band_period).mean()
        width = df["close"].rolling(config.band_period).std() * config.band_std
        at_low, at_high = df["low"] < mid - width, df["high"] > mid + width
        d = d.where(((d == 1) & at_low) | ((d == -1) & at_high), 0)
    return d
