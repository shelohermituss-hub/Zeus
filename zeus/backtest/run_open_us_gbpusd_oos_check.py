"""
Validation hors-échantillon — le seul candidat "robuste" trouvé dans toute
l'exploration Open US : Wyckoff sur GBPUSD, min_score=0, RR=3.0, positif sur
2023 (+3 à +5R), 2024 (+8 à +12R) et 2025 (+6 à +9R) selon la fenêtre
(voir run_open_us_gbpusd_check.py).

Ces trois années ont toutes servi à la RECHERCHE de configuration (parmi
~900 combinaisons testées au total dans cette exploration, XAUUSD + GBPUSD
confondus) — un résultat "positif sur toutes les périodes vues" n'est pas
une preuve d'edge tant qu'il n'est pas confirmé sur des données JAMAIS
vues pendant la recherche. GBPUSD juin 2026 (téléchargé mais jamais utilisé
dans aucun sweep précédent) sert ici de test hors-échantillon.

Résultat (voir sortie du script) : l'edge NE SE CONFIRME PAS hors-
échantillon — conclusion consignée dans README_OPEN_US_FINDINGS.md.

Usage
-----
    python -m zeus.backtest.run_open_us_gbpusd_oos_check
"""
from __future__ import annotations

from pathlib import Path

from zeus.backtest.data_loader import parse_histdata_csv
from zeus.backtest.sd_simulation import compute_metrics, simulate_all
from zeus.strategy.open_us_strategy import OpenUSStrategy
from zeus.strategy.supply_demand.wyckoff import WyckoffDetector

_M1_DIR = Path(__file__).parent.parent.parent / "data" / "historical" / "gbpusd" / "m1"
SPREAD_GBPUSD = 0.00015


def main() -> None:
    df = parse_histdata_csv(_M1_DIR / "DAT_MT_GBPUSD_M1_202606.csv")
    print(f"GBPUSD 2026-06 (hors-échantillon) : {len(df):,} barres, {df.index[0]} -> {df.index[-1]}")

    for window in [90, 120]:
        strat = OpenUSStrategy(detector=WyckoffDetector(), window_minutes=window,
                                min_score=0.0, risk_reward=3.0)
        signals = strat.generate_signals(df)
        results, n_expired = simulate_all(
            signals, df, risk_pct=0.01, spread=SPREAD_GBPUSD, initial_equity=10_000.0,
        )
        m = compute_metrics(results, 10_000.0, len(signals), n_expired)
        print(f"  window={window}: R={m['total_r']:+.1f}  n={m['n_trades']}  "
              f"WR={m['win_rate']:.1f}%  (in-sample 2023-2025 WR ≈ 25.6-26.2%)")


if __name__ == "__main__":
    main()
