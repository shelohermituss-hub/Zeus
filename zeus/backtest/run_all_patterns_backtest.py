"""
Batch de validation brute — les 7 familles de patterns de bougies
(MetaQuotes "Free Robots") + confirmation RSI, sur GBPUSD 2025.

Mêmes défauts EA d'origine pour tous (SL/TP=200 points, durée=10 barres,
corps moyen 12 barres, MA tendance 5 barres, RSI 37/40/60), aucun réglage
fin — une première mesure honnête, comme les étapes précédentes.

Usage
-----
    python -m zeus.backtest.run_all_patterns_backtest
"""
from __future__ import annotations

from pathlib import Path

from zeus.backtest.data_loader import parse_m1_csv
from zeus.strategy.candlestick_patterns import PATTERNS, CandlestickRsiStrategy
from zeus.strategy.engulfing_rsi import simulate_engulfing_rsi

_DATA = Path(__file__).parent.parent.parent / "data" / "historical" / "gbpusd" / "m1"

SL_POINTS     = 200
TP_POINTS     = 200
DURATION_BARS = 10
POINT_SIZE    = 0.00001
LOT           = 0.1
CONTRACT_SIZE = 100_000


def main() -> None:
    path = _DATA / "DAT_MT_GBPUSD_M1_2025.csv"
    m1 = parse_m1_csv(path)
    m1 = m1[~m1.index.duplicated(keep="first")]
    print(f"GBPUSD 2025 : {len(m1):,} barres M1\n")

    print(f"{'Pattern':<24}{'Signaux':>9}{'Trades':>9}{'WR%':>8}{'PnL$':>12}{'Barres moy.':>13}")
    print("-" * 75)

    for name in PATTERNS:
        strategy = CandlestickRsiStrategy(name)
        signals = strategy.run(m1)
        if not signals:
            print(f"{name:<24}{'0':>9}{'—':>9}{'—':>8}{'—':>12}{'—':>13}")
            continue

        results = simulate_engulfing_rsi(
            signals, m1,
            sl_points=SL_POINTS, tp_points=TP_POINTS, point_size=POINT_SIZE,
            duration_bars=DURATION_BARS, lot=LOT, contract_size=CONTRACT_SIZE,
        )
        n = len(results)
        wins = sum(1 for r in results if r.outcome == "win")
        losses = sum(1 for r in results if r.outcome == "loss")
        decided = wins + losses
        wr = wins / decided * 100 if decided else 0.0
        pnl = sum(r.pnl_usd for r in results)
        avg_bars = sum(r.bars_held for r in results) / n if n else 0

        print(f"{name:<24}{len(signals):>9}{n:>9}{wr:>7.1f}%{pnl:>+11.2f}${avg_bars:>12.1f}")


if __name__ == "__main__":
    main()
