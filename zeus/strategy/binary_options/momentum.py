"""
Continuation de momentum (M1).

Idée : pendant les sessions liquides (Londres / New York), une série de
bougies de même couleur avec des corps significatifs (en ATR) traduit un flux
directionnel qui se prolonge sur quelques minutes.

- ``streak`` bougies haussières consécutives, corps >= min_body_atr * ATR → CALL (+1)
- même chose baissier → PUT (-1)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from zeus.strategy.binary_options import in_utc_hours
from zeus.strategy.utils import atr_series


@dataclass(frozen=True)
class MomentumConfig:
    streak: int = 3
    min_body_atr: float = 0.5
    atr_period: int = 14
    session_start_utc: int = 7
    session_end_utc: int = 16

    def __post_init__(self) -> None:
        if self.streak < 1 or self.atr_period < 2 or self.min_body_atr < 0:
            raise ValueError("streak >= 1, atr_period >= 2 et min_body_atr >= 0 requis")
        if not (0 <= self.session_start_utc <= 23 and 0 <= self.session_end_utc <= 24):
            raise ValueError("heures de session hors de [0, 24]")


def directions(df: pd.DataFrame, config: MomentumConfig) -> pd.Series:
    body = df["close"] - df["open"]
    strong = body.abs() >= config.min_body_atr * atr_series(df, config.atr_period)
    sign = pd.Series(np.sign(body), index=df.index).where(strong, 0)
    window = sign.rolling(config.streak)
    up = window.min() == 1
    down = window.max() == -1
    session = in_utc_hours(df.index, config.session_start_utc, config.session_end_utc)
    return (up & session).astype(int) - (down & session).astype(int)
