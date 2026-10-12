"""
Recherche options binaires sur les patterns de chandeliers japonais.

7 patterns × confirmation RSI (oui/non) × position (aucune, hors Bollinger 2σ)
× M5/M15/H1 × expiration 1-3 bougies. Même protocole que run_binary_search
(rollover exclu, sélection 2023-2024, validation 2025, contrôle 2026).

Usage
-----
    python -m zeus.backtest.research.run_binary_candlestick_search
"""
from __future__ import annotations

import itertools

from zeus.backtest.research import run_binary_search
from zeus.strategy.binary_options import candlestick
from zeus.strategy.candlestick_patterns import PATTERNS

_OUT = run_binary_search._ROOT / "data" / "results" / "binary_candlestick_search.json"


def variants() -> list[tuple[str, object, object]]:
    return [
        (f"{p}_rsi{int(rsi)}_band{band}", candlestick,
         candlestick.CandlestickConfig(pattern=p, use_rsi=rsi, band_std=band))
        for p, rsi, band in itertools.product(PATTERNS, [False, True], [0.0, 2.0])
    ]


if __name__ == "__main__":
    run_binary_search.main(variants, _OUT)
