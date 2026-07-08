"""
Engulfing + RSI — backtest brut GBPUSD 2025 (port du robot MetaQuotes
"BullishBearish Engulfing RSI.mq5" envoyé par l'utilisateur).

Paramètres EA d'origine inchangés (SL/TP=200 points, durée=10 barres,
RSI 37, corps moyen 12 barres, MA tendance 5 barres) — testés sur M1 tels
quels, sans aucun réglage, pour une première mesure honnête de l'edge.

Usage
-----
    python -m zeus.backtest.run_engulfing_rsi_backtest
"""
from __future__ import annotations

from pathlib import Path

from zeus.backtest.data_loader import parse_m1_csv
from zeus.strategy.engulfing_rsi import EngulfingRsiStrategy, simulate_engulfing_rsi

_DATA = Path(__file__).parent.parent.parent / "data" / "historical" / "gbpusd" / "m1"

# Défauts EXACTS du fichier .mq5 d'origine.
SL_POINTS      = 200
TP_POINTS      = 200
DURATION_BARS  = 10
POINT_SIZE     = 0.00001   # GBPUSD 5 chiffres
LOT            = 0.1
CONTRACT_SIZE  = 100_000


def main() -> None:
    path = _DATA / "DAT_MT_GBPUSD_M1_2025.csv"
    m1 = parse_m1_csv(path)
    m1 = m1[~m1.index.duplicated(keep="first")]
    print(f"GBPUSD 2025 : {len(m1):,} barres M1")

    strategy = EngulfingRsiStrategy()
    signals = strategy.run(m1)
    print(f"Signaux générés : {len(signals)}  ({len(signals) / 252:.2f}/jour de bourse)")
    if not signals:
        print("Aucun signal — rien à simuler.")
        return

    results = simulate_engulfing_rsi(
        signals, m1,
        sl_points=SL_POINTS, tp_points=TP_POINTS, point_size=POINT_SIZE,
        duration_bars=DURATION_BARS, lot=LOT, contract_size=CONTRACT_SIZE,
    )

    n = len(results)
    n_expired = len(signals) - n
    wins   = [r for r in results if r.outcome == "win"]
    losses = [r for r in results if r.outcome == "loss"]
    scratches = [r for r in results if r.outcome == "scratch"]
    decided = len(wins) + len(losses)
    wr = len(wins) / decided * 100 if decided else 0.0
    total_pnl = sum(r.pnl_usd for r in results)
    avg_bars = sum(r.bars_held for r in results) / n if n else 0

    print(f"\nTrades simulés   : {n}  (expirés en fin de data : {n_expired})")
    print(f"  gagnants={len(wins)}  perdants={len(losses)}  scratch/RSI={len(scratches)}")
    print(f"Winrate (W/(W+L)): {wr:.1f}%")
    print(f"PnL total        : {total_pnl:+,.2f}$  (lot fixe {LOT}, sans risque%/equity)")
    print(f"Durée moy. (barres): {avg_bars:.1f}")


if __name__ == "__main__":
    main()
