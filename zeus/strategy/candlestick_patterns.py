"""
Détecteurs de patterns de bougies japonaises — port fidèle des 7 familles
de "Free Robots" MetaQuotes envoyées par l'utilisateur (Engulfing, Harami,
Dark Cloud/Piercing Line, Hanging Man/Hammer, Morning/Evening Star & Doji,
Three Black Crows/White Soldiers, Meeting Lines).

Chaque détecteur reproduit exactement CheckPattern() de son fichier .mq5
source. La confirmation par oscillateur (RSI/CCI/MFI/Stochastique) et la
simulation (SL/TP bracket, sortie par croisement, expiration en barres)
sont génériques et partagées — voir engulfing_rsi.py pour ces briques.

NB (trouvaille d'audit) : le fichier source "DarkCloud PiercingLine*.mq5"
contient un bug de syntaxe réel — le bloc Piercing Line est écrit
    if(<condition>) return(true);
    { ExtPatternDetected=true; ... }
Le `{...}` n'est PAS le corps du `if` (qui se termine à son `;`) : c'est un
bloc INCONDITIONNEL qui s'exécute à chaque fois que la condition est
FAUSSE, déclenchant un signal Buy à tort ; et quand la condition est VRAIE,
la fonction retourne avant même de positionner ExtSignalOpen, donc un vrai
Piercing Line ne déclenche RIEN. Porté ici, c'est la logique CORRIGÉE
(la condition déclenche bien le signal) qui est implémentée — porter le
bug tel quel n'aurait aucun sens.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from zeus.strategy.engulfing_rsi import _avg_body, _rsi, _sma

_EPS = 1e-8


@dataclass(frozen=True)
class PatternSignal:
    direction: str
    bar_index: int
    formed_at: pd.Timestamp
    pattern:   str


class _Arrays:
    """Tableaux OHLC + indicateurs précalculés, partagés par tous les détecteurs."""

    def __init__(self, df: pd.DataFrame, avg_body_period: int, ma_period: int):
        self.o = df["open"].to_numpy(dtype=float)
        self.h = df["high"].to_numpy(dtype=float)
        self.l = df["low"].to_numpy(dtype=float)
        self.c = df["close"].to_numpy(dtype=float)
        self.avg_body = _avg_body(self.o, self.c, avg_body_period)
        self.sma      = _sma(self.c, ma_period) if ma_period > 0 else None
        self.mid_range = (self.h + self.l) / 2.0
        self.mid_oc    = (self.o + self.c) / 2.0
        self.n = len(df)


# ── Détecteurs de pattern : renvoient "long", "short" ou None pour la barre i ──
# i = MQL5 index 1 (dernière bougie close) ; i-1 = index 2 ; i-2 = index 3.

def _engulfing(a: _Arrays, i: int) -> str | None:
    p = i - 1
    if (a.o[p] < a.c[p] and (a.o[i] - a.c[i]) > a.avg_body[i]
            and a.c[i] < a.o[p] and a.mid_oc[p] > a.sma[p] and a.o[i] > a.c[p]):
        return "short"
    if (a.o[p] > a.c[p] and (a.c[i] - a.o[i]) > a.avg_body[i]
            and a.c[i] > a.o[p] and a.mid_oc[p] < a.sma[p] and a.o[i] < a.c[p]):
        return "long"
    return None


def _harami(a: _Arrays, i: int) -> str | None:
    p = i - 1
    if (a.c[i] < a.o[i] and (a.c[p] - a.o[p]) > a.avg_body[i]
            and a.c[i] > a.o[p] and a.o[i] < a.c[p] and a.mid_range[p] > a.sma[p]):
        return "short"
    if (a.c[i] > a.o[i] and (a.o[p] - a.c[p]) > a.avg_body[i]
            and a.c[i] < a.o[p] and a.o[i] > a.c[p] and a.mid_range[p] < a.sma[p]):
        return "long"
    return None


def _dark_cloud_piercing(a: _Arrays, i: int) -> str | None:
    p = i - 1
    if ((a.c[p] - a.o[p]) > a.avg_body[i] and a.c[i] < a.c[p] and a.c[i] > a.o[p]
            and a.mid_oc[p] > a.sma[p] and a.o[i] > a.h[p]):
        return "short"
    if ((a.c[i] - a.o[i]) > a.avg_body[i] and (a.o[p] - a.c[p]) > a.avg_body[i]
            and a.c[i] > a.c[p] and a.c[i] < a.o[p] and a.mid_oc[p] < a.sma[p]
            and a.o[i] < a.l[p]):
        return "long"
    return None


def _hanging_man_hammer(a: _Arrays, i: int) -> str | None:
    p = i - 1
    upper_third = a.h[i] - (a.h[i] - a.l[i]) / 3.0
    body_in_upper_third = min(a.o[i], a.c[i]) > upper_third
    if (a.mid_range[i] > a.sma[p] and body_in_upper_third
            and a.c[i] > a.c[p] and a.o[i] > a.o[p]):
        return "short"
    if (a.mid_range[i] < a.sma[p] and body_in_upper_third
            and a.c[i] < a.c[p] and a.o[i] < a.o[p]):
        return "long"
    return None


def _morning_evening_star_doji(a: _Arrays, i: int) -> str | None:
    p, pp = i - 1, i - 2
    mid_oc3 = (a.o[pp] + a.c[pp]) / 2.0
    # Evening Doji
    if ((a.c[pp] - a.o[pp]) > a.avg_body[i] and abs(a.c[p] - a.o[p]) < a.avg_body[i] * 0.1
            and a.c[p] > a.c[pp] and a.o[p] > a.o[pp] and a.o[i] < a.c[p] and a.c[i] < a.c[p]):
        return "short"
    # Evening Star
    if ((a.c[pp] - a.o[pp]) > a.avg_body[i] and abs(a.c[p] - a.o[p]) < a.avg_body[i] * 0.5
            and a.c[p] > a.c[pp] and a.o[p] > a.o[pp] and a.c[i] < mid_oc3):
        return "short"
    # Morning Doji
    if ((a.o[pp] - a.c[pp]) > a.avg_body[i] and abs(a.c[p] - a.o[p]) < a.avg_body[i] * 0.1
            and a.c[p] < a.c[pp] and a.o[p] < a.o[pp] and a.o[i] > a.c[p] and a.c[i] > a.c[p]):
        return "long"
    # Morning Star
    if ((a.o[pp] - a.c[pp]) > a.avg_body[i] and abs(a.c[p] - a.o[p]) < a.avg_body[i] * 0.5
            and a.c[p] < a.c[pp] and a.o[p] < a.o[pp] and a.c[i] > mid_oc3):
        return "long"
    return None


def _black_crows_white_soldiers(a: _Arrays, i: int) -> str | None:
    p, pp = i - 1, i - 2
    if ((a.o[pp] - a.c[pp]) > a.avg_body[i] and (a.o[p] - a.c[p]) > a.avg_body[i]
            and (a.o[i] - a.c[i]) > a.avg_body[i]
            and a.mid_range[p] < a.mid_range[pp] and a.mid_range[i] < a.mid_range[p]):
        return "short"
    if ((a.c[pp] - a.o[pp]) > a.avg_body[i] and (a.c[p] - a.o[p]) > a.avg_body[i]
            and (a.c[i] - a.o[i]) > a.avg_body[i]
            and a.mid_range[p] > a.mid_range[pp] and a.mid_range[i] > a.mid_range[p]):
        return "long"
    return None


def _meeting_lines(a: _Arrays, i: int) -> str | None:
    p = i - 1
    if ((a.c[p] - a.o[p]) > a.avg_body[i] and (a.o[i] - a.c[i]) > a.avg_body[i]
            and abs(a.c[i] - a.c[p]) < 0.1 * a.avg_body[i]):
        return "short"
    if ((a.o[p] - a.c[p]) > a.avg_body[i] and (a.c[i] - a.o[i]) > a.avg_body[i]
            and abs(a.c[i] - a.c[p]) < 0.1 * a.avg_body[i]):
        return "long"
    return None


PATTERNS: dict[str, dict] = {
    "engulfing":              dict(fn=_engulfing,               needs_ma=True,  n_bars=2),
    "harami":                 dict(fn=_harami,                  needs_ma=True,  n_bars=2),
    "dark_cloud_piercing":    dict(fn=_dark_cloud_piercing,      needs_ma=True,  n_bars=2),
    "hanging_man_hammer":     dict(fn=_hanging_man_hammer,       needs_ma=True,  n_bars=2),
    "morning_evening_star":   dict(fn=_morning_evening_star_doji, needs_ma=False, n_bars=3),
    "black_crows_soldiers":   dict(fn=_black_crows_white_soldiers, needs_ma=False, n_bars=3),
    "meeting_lines":          dict(fn=_meeting_lines,            needs_ma=False, n_bars=2),
}


class CandlestickRsiStrategy:
    """Détecteur générique : pattern (au choix dans PATTERNS) + confirmation RSI."""

    def __init__(
        self,
        pattern:          str,
        avg_body_period:  int   = 12,
        ma_period:        int   = 5,
        rsi_period:       int   = 37,
        rsi_buy_max:      float = 40.0,
        rsi_sell_min:     float = 60.0,
    ) -> None:
        if pattern not in PATTERNS:
            raise ValueError(f"pattern inconnu : {pattern} (choix : {list(PATTERNS)})")
        self.pattern_name = pattern
        self._spec        = PATTERNS[pattern]
        self.avg_body_period = avg_body_period
        self.ma_period        = ma_period if self._spec["needs_ma"] else 0
        self.rsi_period       = rsi_period
        self.rsi_buy_max      = rsi_buy_max
        self.rsi_sell_min     = rsi_sell_min

    def run(self, df: pd.DataFrame) -> list[PatternSignal]:
        n_bars_needed = self._spec["n_bars"]
        min_needed = max(self.avg_body_period, self.ma_period, self.rsi_period) + n_bars_needed
        if len(df) < min_needed:
            return []

        a   = _Arrays(df, self.avg_body_period, self.ma_period)
        rsi = _rsi(a.c, self.rsi_period)
        fn  = self._spec["fn"]

        signals: list[PatternSignal] = []
        start = min_needed - 1
        for i in range(start, a.n):
            if np.isnan(a.avg_body[i]) or (a.sma is not None and np.isnan(a.sma[i - 1])):
                continue
            direction = fn(a, i)
            if direction is None:
                continue
            if direction == "long" and rsi[i] < self.rsi_buy_max:
                signals.append(PatternSignal("long", i, df.index[i], self.pattern_name))
            elif direction == "short" and rsi[i] > self.rsi_sell_min:
                signals.append(PatternSignal("short", i, df.index[i], self.pattern_name))

        return signals
