"""
Backtest — stratégie Open US (sweep puis retournement) sur XAUUSD.

Réutilise sans modification :
    - zeus.strategy.open_us_strategy.OpenUSStrategy (génération des signaux)
    - zeus.backtest.sd_simulation.simulate_all/compute_metrics/print_report
      (moteur de simulation générique, déjà utilisé par tous les run_sd_*.py)

Teste plusieurs régimes de marché séparément (2024 complet, 2025 complet,
2026 H1), conformément à la règle du projet "tester plusieurs périodes de
marché" — pas de conclusion tirée d'un seul échantillon.

Usage
-----
    python -m zeus.backtest.run_open_us_backtest
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv
from zeus.backtest.sd_simulation import compute_metrics, print_report, simulate_all
from zeus.strategy.open_us_strategy import OpenUSStrategy

_ROOT   = Path(__file__).parent.parent.parent
_M1_DIR = _ROOT / "data" / "historical" / "xauusd" / "m1"

INITIAL_BALANCE = 10_000.0
RISK_PCT        = 0.01
RISK_REWARD     = 3.0
MIN_SCORE       = 4.0            # même seuil que SDStrategy (MIN_WYCKOFF_SCORE)
WINDOW_MINUTES  = 90             # fenêtre après 9h30 ET pendant laquelle un signal est accepté


def _load(*filenames: str) -> pd.DataFrame:
    frames = [parse_histdata_csv(_M1_DIR / f) for f in filenames]
    return pd.concat(frames).sort_index()


def run_period(label: str, df: pd.DataFrame) -> dict:
    strategy = OpenUSStrategy(
        window_minutes = WINDOW_MINUTES,
        min_score      = MIN_SCORE,
        risk_reward    = RISK_REWARD,
    )
    signals = strategy.generate_signals(df)
    results, n_expired = simulate_all(
        signals, df,
        risk_pct       = RISK_PCT,
        initial_equity = INITIAL_BALANCE,
    )
    print_report(results, INITIAL_BALANCE, len(signals), n_expired, df, label=label)
    return compute_metrics(results, INITIAL_BALANCE, len(signals), n_expired)


def main() -> None:
    periods = [
        ("2024 (complet)",  ["DAT_MT_XAUUSD_M1_2024.csv"]),
        ("2025 (complet)",  ["DAT_MT_XAUUSD_M1_2025.csv"]),
        ("2026 H1",         [f"DAT_MT_XAUUSD_M1_202601.csv", f"DAT_MT_XAUUSD_M1_202602.csv",
                              f"DAT_MT_XAUUSD_M1_202603.csv", f"DAT_MT_XAUUSD_M1_202604.csv",
                              f"DAT_MT_XAUUSD_M1_202605.csv", f"DAT_MT_XAUUSD_M1_202606.csv"]),
    ]

    summary = []
    for label, files in periods:
        df = _load(*files)
        m  = run_period(label, df)
        summary.append((label, m))
        print()

    print("=" * 62)
    print("  Résumé multi-périodes")
    print("=" * 62)
    print(f"  {'Période':<16} {'Signaux':>8} {'Trades':>7} {'WR %':>7} {'Total R':>9} {'P&L $':>10} {'MaxDD %':>8}")
    for label, m in summary:
        print(f"  {label:<16} {m['n_signals']:>8} {m['n_trades']:>7} "
              f"{m['win_rate']:>6.1f}% {m['total_r']:>+8.2f}R {m['total_usd']:>+9.2f}  {m['max_dd']:>6.1f}%")
    print("=" * 62)


if __name__ == "__main__":
    main()
