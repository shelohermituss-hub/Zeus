"""
Outils d'évaluation partagés par les études options binaires.

- Chargement d'un fichier M1 et agrégation au timeframe voulu.
- Taille du pip par instrument (pour exprimer l'écart d'exécution).
- Agrégation de résultats de plusieurs jeux (paire × période) en statistiques
  poolées : win rate, borne basse de Wilson, part des jeux au-dessus du seuil.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from zeus.backtest.binary_engine import wilson_lower_bound
from zeus.backtest.data_loader import parse_m1_csv, resample_ohlcv


def pip_size(symbol: str) -> float:
    symbol = symbol.lower()
    if symbol.startswith("xau"):
        return 0.1
    return 0.01 if "jpy" in symbol else 0.0001


def load_timeframe(path: str | Path, timeframe_minutes: int) -> pd.DataFrame:
    """M1 → timeframe ; les minutes en double (heure répétée au changement d'heure) sont écartées."""
    df = parse_m1_csv(path)
    df = df[~df.index.duplicated(keep=False)]
    return df if timeframe_minutes == 1 else resample_ohlcv(df, f"{timeframe_minutes}min")


@dataclass(frozen=True)
class Tally:
    """Comptage d'un jeu : gains et pertes (égalités exclues)."""
    wins: int
    losses: int

    @property
    def win_rate(self) -> float:
        n = self.wins + self.losses
        return self.wins / n if n else 0.0


def pool(tallies: dict[str, Tally], breakeven: float) -> dict:
    wins = sum(t.wins for t in tallies.values())
    n = wins + sum(t.losses for t in tallies.values())
    active = [t for t in tallies.values() if t.wins + t.losses]
    return {
        "trades": n,
        "win_rate": round(wins / n, 4) if n else 0.0,
        "win_rate_lower_95": round(wilson_lower_bound(wins, n), 4),
        "share_above_breakeven": round(sum(t.win_rate > breakeven for t in active) / len(active), 3) if active else 0.0,
        "per_dataset_win_rate": {k: round(t.win_rate, 4) for k, t in tallies.items() if t.wins + t.losses},
    }
