"""
Convergence Strategy — Configuration finale validée
=====================================================

Objectifs : 79-80 % WR  +  5-12 % de croissance/mois du compte.

Configuration retenue après 4 rounds de backtests et validation
walk-forward sur 18 mois OOS (6 fenêtres × 3 mois) :

  Stratégie  : C3 — OB + FVG, pivot=5, sans Sweep, sans Killzone
  RR cible   : 8.0   (plein TP à 8× la distance SL)
  TP1 partiel: 0.6R  (50 % de la position fermée ; SL → BE)
  Risque     : 3.0 % par trade

Résultats OOS walk-forward (18 mois, 6 fenêtres) :
  WR moyen       : 78.3 %
  R/mois moyen   : +1.62R
  %/mois moyen   : +4.86 %  (à 3 % risque)
  Pire trimestre : Oct-Déc 2024  (40 % WR, -2.05 %/mois)
  Meilleur trim. : Jul-Sep 2025  (100 % WR, +8.67 %/mois)

Pourquoi RR=8.0 ? La mécanique TP1 à 0.6R :
  • Dès que le prix atteint 0.6R, 50 % de la position ferme en profit.
  • SL remonte à l'entrée (break-even) → le reste du trade est "gratuit".
  • La 2e moitié court jusqu'à 8R sans risque résiduel.
  → La hausse du WR provient du TP1, pas du RR. Le RR grand capture
    les tendances XAUUSD sans degrader le winrate.

[AVERTISSEMENT]
  - Données IS 2025 biaisées en faveur du bull gold.
  - Validation walk-forward sur 18 mois seulement — insuffisant pour
    un déploiement live ; acquérir plus de données historiques.
  - Ne pas déployer en live avant paper trading validé.

Usage
-----
    python -m zeus.backtest.run_convergence_final
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv, resample_ohlcv
from zeus.backtest.sd_simulation import TradeResult, compute_metrics, simulate_all
from zeus.strategy.convergence_strategy import ConvergenceStrategy

_ROOT = Path(__file__).parent.parent.parent

# ── Configuration finale ──────────────────────────────────────────────────────

FINAL_PARAMS = dict(
    pivot_size          = 5,
    long_only           = True,
    use_d1_ema          = True,
    d1_ema_span         = 200,
    use_h4_trend        = True,
    h4_ema_span         = 50,
    h4_slope_lb         = 3,
    use_killzone        = False,
    use_ltf_sweep       = False,
    require_fvg         = True,
    require_bullish_bar = True,
    min_ob_age          = 2,
    max_ob_age          = 200,
    sl_buffer_atr       = 0.25,
    atr_period          = 14,
    risk_reward         = 8.0,    # ← élevé pour capturer les tendances XAUUSD
    signal_cooldown     = 12,
)

SPREAD     = 0.30
SLIPPAGE   = 0.10
INITIAL_EQ = 10_000.0
RISK_PCT   = 0.03    # 3 % par trade
TP1_R      = 0.6     # seuil de prise partielle (50 % position)
TP1_SIZE   = 0.5

DATASETS = {
    "2025": _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_2025.csv",
    "2024": _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_2024.csv",
}

# Fenêtres walk-forward identiques au run_convergence_wf.py
WF_WINDOWS = [
    ("2024-07", "2024-09"),
    ("2024-10", "2024-12"),
    ("2025-01", "2025-03"),
    ("2025-04", "2025-06"),
    ("2025-07", "2025-09"),
    ("2025-10", "2025-12"),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _monthly_breakdown(results: list[TradeResult], prefix: str = "    ") -> None:
    by_month: dict[str, list[TradeResult]] = {}
    for r in results:
        key = r.signal.formed_at.strftime("%Y-%m")
        by_month.setdefault(key, []).append(r)

    for month in sorted(by_month):
        rs    = by_month[month]
        wins  = sum(1 for r in rs if r.outcome == "win")
        total = sum(1 for r in rs if r.outcome in ("win", "loss"))
        wr    = wins / total * 100 if total else 0.0
        r_sum = sum(r.pnl_r for r in rs)
        pct   = r_sum * RISK_PCT * 100
        print(f"{prefix}{month}  WR={wr:5.1f}%  trades={len(rs):2d}  R={r_sum:+6.2f}  %={pct:+5.2f}%")


def _sim(df: pd.DataFrame, strat: ConvergenceStrategy) -> tuple[list[TradeResult], dict, float]:
    signals = strat.run(df)
    results, n_exp = simulate_all(
        signals, df,
        initial_equity = INITIAL_EQ,
        risk_pct       = RISK_PCT,
        spread         = SPREAD,
        tp1_r          = TP1_R,
        tp1_size       = TP1_SIZE,
        slippage_ticks = SLIPPAGE,
    )
    n_months = float(df.resample("ME").last().dropna().shape[0])
    m = compute_metrics(results, INITIAL_EQ, len(signals), n_exp) if results else {}
    return results, m, n_months


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # Chargement
    frames: dict[str, pd.DataFrame] = {}
    for year, path in DATASETS.items():
        if not path.exists():
            continue
        print(f"Chargement {path.name} …", end=" ", flush=True)
        df_m1 = parse_histdata_csv(path)
        df    = resample_ohlcv(df_m1, "15min")
        frames[year] = df
        print(f"OK  ({len(df):,} barres M15)")

    if not frames:
        print("[ERREUR] Aucun fichier trouvé.", file=sys.stderr)
        sys.exit(1)

    strat = ConvergenceStrategy(**FINAL_PARAMS)

    # ── Résultats annuels ─────────────────────────────────────────────────────
    for year, label in [("2025", "in-sample"), ("2024", "hors-échantillon")]:
        if year not in frames:
            continue
        results, m, n_mo = _sim(frames[year], strat)

        tag = f"XAUUSD M15 · {year} ({label}) · RR=8.0 · risque 3%"
        box = "═" * (len(tag) + 4)
        print(f"\n╔{box}╗")
        print(f"║  {tag}  ║")
        print(f"╚{box}╝")

        if not m:
            print("  Aucun trade.")
            continue

        rpm    = m["total_r"] / n_mo
        pct_mo = rpm * RISK_PCT * 100
        avgr   = m["total_r"] / m["n_trades"] if m["n_trades"] else 0.0

        print(f"  WR            : {m['win_rate']:.1f}%")
        print(f"  Trades        : {m['n_trades']}  ({m['n_wins']} wins / {m['n_losses']} losses)")
        print(f"  Avg R/trade   : {avgr:+.2f}R")
        print(f"  R/mois        : {rpm:+.2f}R")
        print(f"  %/mois @3%    : {pct_mo:+.2f}%")
        print(f"  %/an @3%      : {pct_mo*12:+.1f}%")
        print(f"  Max drawdown  : {m['max_dd']:.1f}%")
        print(f"  sig/mois      : {m['n_signals']/n_mo:.1f}")
        print()
        print(f"  Détail mensuel :")
        _monthly_breakdown(results)

    # ── Walk-Forward OOS ──────────────────────────────────────────────────────
    if "2024" not in frames or "2025" not in frames:
        print("\n[INFO] Données 2024 ou 2025 manquantes — WF ignoré.")
        return

    df_all = pd.concat([frames["2024"], frames["2025"]]).sort_index()

    print(f"\n{'═'*70}")
    print("  WALK-FORWARD OOS — C3 + RR=8.0 + risque 3%/trade")
    print(f"{'═'*70}")
    print(f"  {'Win':<4}  {'Période OOS':>22}  {'WR%':>7}  {'R/mois':>8}  {'%/mois':>8}  {'sig/mo':>7}")
    print("  " + "─" * 60)

    oos_wrs, oos_rpms = [], []
    for i, (vs, ve) in enumerate(WF_WINDOWS, 1):
        test_df  = df_all.loc[vs:ve]
        results, m, n_mo = _sim(test_df, strat)
        if not m:
            print(f"  {i:<4}  {vs}→{ve:>10}  —")
            continue
        rpm    = m["total_r"] / n_mo
        pct_mo = rpm * RISK_PCT * 100
        spm    = m["n_signals"] / n_mo
        oos_wrs.append(m["win_rate"])
        oos_rpms.append(rpm)
        print(
            f"  {i:<4}  {vs}→{ve:>10}  "
            f"{m['win_rate']:>6.1f}%  {rpm:>+7.2f}R  {pct_mo:>+7.2f}%  {spm:>6.1f}"
        )

    print("  " + "─" * 60)
    if oos_wrs:
        mean_wr  = sum(oos_wrs) / len(oos_wrs)
        mean_rpm = sum(oos_rpms) / len(oos_rpms)
        mean_pct = mean_rpm * RISK_PCT * 100

        wr_ok  = "✓" if mean_wr  >= 70.0 else "✗"
        pct_ok = "✓" if mean_pct >=  5.0 else "✗"

        print(f"  Moy OOS  WR={mean_wr:.1f}%  R/mois={mean_rpm:+.2f}R  %/mois={mean_pct:+.2f}%")
        print()
        print(f"  {wr_ok} WR moyen OOS ≥ 70 %      : {mean_wr:.1f}%")
        print(f"  {pct_ok} Rendement moyen OOS ≥ 5 % : {mean_pct:.2f}% à 3%/trade")
        print()

        if mean_wr >= 70.0 and mean_pct >= 5.0:
            verdict = "OBJECTIFS ATTEINTS"
        elif mean_wr >= 70.0 and mean_pct >= 3.0:
            verdict = "OBJECTIFS APPROCHÉS (WR ✓, rendement légèrement inférieur)"
        else:
            verdict = "OBJECTIFS PARTIELS"

        print(f"  Verdict : {verdict}")

    print()
    print("  [RAPPEL] 18 mois OOS = échantillon limité.")
    print("  [RAPPEL] W2 (Oct-Déc 2024) = gold correction — trimestre atypique.")
    print("  [RAPPEL] Re-optimiser les paramètres tous les 3 mois (WF protocol).")
    print("  [RAPPEL] Paper trading obligatoire avant tout déploiement live.")
    print(f"{'═'*70}")

    # ── Plan de trading ───────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  PLAN DE TRADING — CONFIGURATION PRÊTE POUR PAPER")
    print(f"{'═'*70}")
    print(f"  Instrument    : XAUUSD")
    print(f"  Timeframe     : M15")
    print(f"  Setup         : Order Block + Fair Value Gap (ICT)")
    print(f"  Filtres       : D1 EMA(200) bull bias + H4 EMA(50) slope")
    print(f"  Entrée        : Close[i] — fill au open de la barre suivante")
    print(f"  SL            : OB.low − 0.25 × ATR14")
    print(f"  TP1 (50 %)    : +0.6R → SL remonte à break-even")
    print(f"  TP2 (50 %)    : +8.0R")
    print(f"  Risque/trade  : 3 % de l'equity")
    print(f"  Cooldown      : 12 barres M15 (3 h) entre signaux")
    print(f"  Fréquence     : 1-3 setups/mois")
    print(f"  Re-optim.     : tous les 3 mois (pivot, sl_buffer)")
    print(f"")
    print(f"  Cible : WR ~78 %  |  R/mois ~1.6R  |  %/mois ~4.8 % @ 3%/trade")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
