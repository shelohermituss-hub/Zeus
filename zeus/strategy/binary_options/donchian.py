"""
Cassure du canal de Donchian (plus haut / plus bas des ``period`` bougies précédentes).

- mode "breakout" : clôture au-dessus du plus haut → CALL (continuation).
- mode "fade"     : clôture au-dessus du plus haut → PUT (fausse cassure).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DonchianConfig:
    period: int = 20
    mode: str = "breakout"

    def __post_init__(self) -> None:
        if self.period < 2 or self.mode not in ("breakout", "fade"):
            raise ValueError("period >= 2 et mode in {breakout, fade} requis")


def directions(df: pd.DataFrame, config: DonchianConfig) -> pd.Series:
    upper = df["high"].rolling(config.period).max().shift(1)
    lower = df["low"].rolling(config.period).min().shift(1)
    up = (df["close"] > upper).astype(int)
    down = (df["close"] < lower).astype(int)
    d = up - down
    return d if config.mode == "breakout" else -d
