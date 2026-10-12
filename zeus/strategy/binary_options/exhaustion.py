"""
Épuisement : après ``streak`` bougies consécutives de même couleur, parier sur l'inverse.

- ``streak`` bougies haussières → PUT ; baissières → CALL.
- ``min_move_atr`` : le mouvement cumulé de la série doit dépasser ce multiple d'ATR.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from zeus.strategy.utils import atr_series


@dataclass(frozen=True)
class ExhaustionConfig:
    streak: int = 5
    min_move_atr: float = 0.0
    atr_period: int = 14

    def __post_init__(self) -> None:
        if self.streak < 2 or self.atr_period < 2 or self.min_move_atr < 0:
            raise ValueError("streak >= 2, atr_period >= 2 et min_move_atr >= 0 requis")


def directions(df: pd.DataFrame, config: ExhaustionConfig) -> pd.Series:
    sign = pd.Series(np.sign(df["close"] - df["open"]), index=df.index)
    window = sign.rolling(config.streak)
    move = (df["close"] - df["open"].shift(config.streak - 1)).abs()
    big = move >= config.min_move_atr * atr_series(df, config.atr_period)
    up = (window.min() == 1) & big
    down = (window.max() == -1) & big
    return down.astype(int) - up.astype(int)
