"""
Vérification croisée — Open US (Wyckoff + ORB) sur GBPUSD, 2023-2025.

But : les deux sweeps XAUUSD (run_open_us_sweep.py / run_open_us_orb_sweep.py)
et le test de régime de volatilité n'ont trouvé aucun edge robuste sur
l'or. Avant de conclure que le concept "Open US" n'a pas d'edge exploitable
en général (plutôt que spécifiquement sur XAUUSD), on le teste sur un autre
instrument très liquide à l'ouverture US — GBPUSD, qui bénéficie du
chevauchement Londres/New York — avec une 3e année indépendante (2023,
jamais utilisée dans les tests précédents).

Paramètres adaptés à l'échelle de prix forex (spread ~1.5 pip, seuils de
range en pips plutôt qu'en $/oz) mais sinon mêmes stratégies, sans
modification de leur logique.

Usage
-----
    python -m zeus.backtest.run_open_us_gbpusd_check
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv
from zeus.backtest.sd_simulation import compute_metrics, simulate_all
from zeus.strategy.open_us_orb_strategy import OpenUSORBStrategy
from zeus.strategy.open_us_strategy import OpenUSStrategy
from zeus.strategy.supply_demand.wyckoff import WyckoffDetector

_ROOT   = Path(__file__).parent.parent.parent
_M1_DIR = _ROOT / "data" / "historical" / "gbpusd" / "m1"

INITIAL_BALANCE = 10_000.0
RISK_PCT        = 0.01
SPREAD_GBPUSD   = 0.00015   # ~1.5 pip, spread CFD retail typique

PERIODS = {
    "2023": "DAT_MT_GBPUSD_M1_2023.csv",
    "2024": "DAT_MT_GBPUSD_M1_2024.csv",
    "2025": "DAT_MT_GBPUSD_M1_2025.csv",
}


def _load(period: str) -> pd.DataFrame:
    return parse_histdata_csv(_M1_DIR / PERIODS[period])


def _run(name: str, signals_by_period: dict[str, list], dfs: dict[str, pd.DataFrame]) -> None:
    print(f"\n  {name}")
    total = 0.0
    for period, df in dfs.items():
        sigs = signals_by_period[period]
        results, n_expired = simulate_all(
            sigs, df, risk_pct=RISK_PCT, spread=SPREAD_GBPUSD, initial_equity=INITIAL_BALANCE,
        )
        m = compute_metrics(results, INITIAL_BALANCE, len(sigs), n_expired)
        total += m["total_r"]
        print(f"    {period}: R={m['total_r']:+.1f}  n={m['n_trades']:>4}  WR={m['win_rate']:.1f}%")
    print(f"    TOTAL R = {total:+.1f}")


def main() -> None:
    dfs = {p: _load(p) for p in PERIODS}
    for p, df in dfs.items():
        print(f"  {p}: {len(df):,} barres M1")

    # ── Wyckoff (sweep + retournement) ──────────────────────────────────────
    for window in [90, 120]:
        for min_score in [0.0, 5.0]:
            for rr in [2.0, 3.0]:
                sigs_by_period = {}
                for period, df in dfs.items():
                    strat = OpenUSStrategy(
                        detector=WyckoffDetector(), window_minutes=window,
                        min_score=min_score, risk_reward=rr,
                    )
                    sigs_by_period[period] = strat.generate_signals(df)
                _run(f"Wyckoff window={window} min_score={min_score} RR={rr}", sigs_by_period, dfs)

    # ── ORB (breakout) ───────────────────────────────────────────────────────
    for or_min in [15, 30]:
        for search_min in [60, 90]:
            for min_rng in [0.0010, 0.0020]:
                for rr in [1.5, 2.0, 3.0]:
                    sigs_by_period = {}
                    for period, df in dfs.items():
                        strat = OpenUSORBStrategy(
                            or_minutes=or_min, search_minutes=search_min,
                            min_range_price=min_rng, risk_reward=rr,
                        )
                        sigs_by_period[period] = strat.generate_signals(df)
                    _run(f"ORB or={or_min} search={search_min} min_rng={min_rng} RR={rr}",
                         sigs_by_period, dfs)


if __name__ == "__main__":
    main()
