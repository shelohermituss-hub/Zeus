"""
Momentum scalp — backtest brut XAUUSD 2025 (étape 2 du plan).

Objectif : vérifier si zeus/strategy/momentum_scalp.py a un edge réel AVANT
tout habillage propfirm ou inclusion portefeuille. Pas de guard ici, pas de
caps par direction — juste le signal + la simulation d'exécution déjà
validée (sd_simulation.simulate_all), à risque fixe simple.

Usage
-----
    python -m zeus.backtest.run_momentum_scalp_backtest
"""
from __future__ import annotations

from pathlib import Path

from zeus.backtest.data_loader import parse_m1_csv
from zeus.backtest.sd_simulation import compute_metrics, simulate_all
from zeus.strategy.momentum_scalp import MomentumScalpStrategy

_DATA = Path(__file__).parent.parent.parent / "data" / "historical" / "xauusd" / "m1"
BALANCE  = 100_000.0
RISK_PCT = 0.006          # même risque fixe que le reste du projet, pour comparaison directe
SPREAD   = 0.30           # USD/oz — même hypothèse que le reste des backtests XAUUSD


def main() -> None:
    path = _DATA / "DAT_MT_XAUUSD_M1_2025.csv"
    m1 = parse_m1_csv(path)
    m1 = m1[~m1.index.duplicated(keep="first")]
    print(f"XAUUSD 2025 : {len(m1):,} barres M1")

    strategy = MomentumScalpStrategy()
    signals = strategy.run(m1)
    print(f"Signaux générés : {len(signals)}  ({len(signals) / 252:.2f}/jour de bourse)")
    if not signals:
        print("Aucun signal — rien à simuler.")
        return

    results, n_expired = simulate_all(
        signals, m1, risk_pct=RISK_PCT, spread=SPREAD, initial_equity=BALANCE,
    )
    metrics = compute_metrics(results, initial_equity=BALANCE,
                               n_signals=len(signals), n_expired=n_expired)

    print(f"\nTrades simulés   : {metrics['n_trades']}  (expirés en fin de data : {n_expired})")
    print(f"Winrate          : {metrics['win_rate']:.1f}%")
    print(f"R total          : {metrics['total_r']:+.2f}")
    print(f"PnL              : {metrics['total_usd']:+,.0f}$  ({metrics['total_usd']/BALANCE*100:+.1f}%)")
    print(f"Coût spread total: {metrics['total_spread']:,.0f}$")
    print(f"Bars tenus (W/L) : {metrics['avg_bars_win']:.1f} / {metrics['avg_bars_loss']:.1f}")
    print(f"DD max (equity)  : {metrics['max_dd']:.2f}%")


if __name__ == "__main__":
    main()
