"""
Étirement Keltner : clôture à plus de ``atr_mult`` ATR de l'EMA → retour vers l'EMA.

- Clôture > EMA + k·ATR → PUT ; clôture < EMA − k·ATR → CALL.
- ``require_rejection`` : la bougie doit déjà se retourner (mèche opposée ≥ corps),
  confirmation classique d'un rejet.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from zeus.strategy.utils import atr_series


@dataclass(frozen=True)
class KeltnerStretchConfig:
    ema_period: int = 20
    atr_period: int = 14
    atr_mult: float = 2.0
    require_rejection: bool = False

    def __post_init__(self) -> None:
        if self.ema_period < 2 or self.atr_period < 2 or self.atr_mult <= 0:
            raise ValueError("ema_period, atr_period >= 2 et atr_mult > 0 requis")


def directions(df: pd.DataFrame, config: KeltnerStretchConfig) -> pd.Series:
    o, h, lo, c = df["open"], df["high"], df["low"], df["close"]
    ema = c.ewm(span=config.ema_period, min_periods=config.ema_period, adjust=False).mean()
    band = config.atr_mult * atr_series(df, config.atr_period)
    put, call = c > ema + band, c < ema - band
    if config.require_rejection:
        body = (c - o).abs()
        put &= (h - pd.concat([o, c], axis=1).max(axis=1)) >= body
        call &= (pd.concat([o, c], axis=1).min(axis=1) - lo) >= body
    return call.astype(int) - put.astype(int)
