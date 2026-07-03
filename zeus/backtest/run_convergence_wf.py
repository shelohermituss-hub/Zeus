"""
Convergence C3 — Walk-Forward Validation
==========================================

Question : la WR de 80 % (OB+FVG, sans LTF sweep) tient-elle lorsque les
paramètres sont choisis sur une fenêtre IS et testés OOS sans retouche ?

Protocole
---------
  Données   : XAUUSD M15 · Jan 2024 – Déc 2025 (24 mois combinés)
  Fenêtre   : train 6 mois  →  test 3 mois (pas = 3 mois)
  Fenêtres  : 6 fenêtres OOS couvrant 18 mois hors-échantillon

  Grille d'optimisation (IS) :
    pivot_size    ∈ [3, 5, 7]
    sl_buffer_atr ∈ [0.15, 0.25, 0.40]
    → 9 combinaisons × 6 fenêtres = 54 backtests au total

  Objectif IS : win_rate (maximiser)

Deux séries comparées
---------------------
  OPT  — paramètres optimisés sur IS à chaque fenêtre
  FIX  — paramètres fixes C3 (pivot=5, sl_buf=0.25) sans re-optimisation

Tableau de sortie par fenêtre :
  Win  Train            Test             Best params   IS_WR   OOS_WR  R/mois  DD%  sig/mois

Usage
-----
    python -m zeus.backtest.run_convergence_wf
"""
from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv, resample_ohlcv
from zeus.backtest.sd_simulation import TradeResult, compute_metrics, simulate_all
from zeus.strategy.convergence_strategy import ConvergenceStrategy

_ROOT = Path(__file__).parent.parent.parent

# ── Données ───────────────────────────────────────────────────────────────────

M1_FILES = [
    _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_2024.csv",
    _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_2025.csv",
]

# ── Coûts simulation ─────────────────────────────────────────────────────────

SPREAD     = 0.30
SLIPPAGE   = 0.10
INITIAL_EQ = 10_000.0
RISK_PCT   = 0.01
TP1_R      = 0.6
TP1_SIZE   = 0.5

# ── Base C3 (OB+FVG, sans Sweep ni Killzone) ─────────────────────────────────

C3_BASE = dict(
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
    atr_period          = 14,
    risk_reward         = 2.0,
    signal_cooldown     = 12,
)

# Paramètres fixes pour la série FIX
C3_FIXED = dict(**C3_BASE, pivot_size=5, sl_buffer_atr=0.25)

# Grille d'optimisation
PARAM_GRID = {
    "pivot_size":    [3, 5, 7],
    "sl_buffer_atr": [0.15, 0.25, 0.40],
}

# ── Fenêtres walk-forward ─────────────────────────────────────────────────────

TRAIN_MONTHS = 6
TEST_MONTHS  = 3
STEP_MONTHS  = 3


def _build_windows(df: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame, str, str]]:
    """
    Génère les paires (train_df, test_df, train_label, test_label).

    Le premier train commence au premier mois complet des données.
    """
    # Mois disponibles (début de chaque mois calendaire)
    month_starts = (
        df.resample("MS").first().dropna(subset=["close"]).index.tolist()
    )
    windows = []
    i = 0
    while True:
        train_start_m = month_starts[i] if i < len(month_starts) else None
        train_end_idx = i + TRAIN_MONTHS
        test_end_idx  = train_end_idx + TEST_MONTHS

        if train_end_idx >= len(month_starts) or test_end_idx > len(month_starts):
            break

        train_start = month_starts[i]
        train_end   = month_starts[train_end_idx]
        test_start  = train_end
        test_end    = month_starts[test_end_idx] if test_end_idx < len(month_starts) else df.index[-1]

        train_df = df.loc[train_start : test_start - pd.Timedelta(minutes=15)]
        test_df  = df.loc[test_start  : test_end  - pd.Timedelta(minutes=15)]

        t_lbl = f"{train_start.strftime('%Y-%m')}→{(test_start - pd.Timedelta(days=1)).strftime('%Y-%m')}"
        o_lbl = f"{test_start.strftime('%Y-%m')}→{(test_end - pd.Timedelta(days=1)).strftime('%Y-%m')}"

        windows.append((train_df, test_df, t_lbl, o_lbl))
        i += STEP_MONTHS

    return windows


# ── Backtest helpers ──────────────────────────────────────────────────────────

def _run(df: pd.DataFrame, params: dict) -> dict:
    """Lance C3 avec *params* sur *df*, retourne les métriques."""
    cfg      = {**C3_BASE, **params}
    strategy = ConvergenceStrategy(**cfg)
    signals  = strategy.run(df)

    if not signals:
        return _empty_metrics()

    results, n_exp = simulate_all(
        signals, df,
        initial_equity = INITIAL_EQ,
        risk_pct       = RISK_PCT,
        spread         = SPREAD,
        tp1_r          = TP1_R,
        tp1_size       = TP1_SIZE,
        slippage_ticks = SLIPPAGE,
    )
    m = compute_metrics(results, INITIAL_EQ, len(signals), n_exp)
    return m


def _empty_metrics() -> dict:
    return dict(
        n_signals=0, n_trades=0, n_wins=0, n_losses=0,
        win_rate=0.0, total_r=0.0, max_dd=0.0,
    )


def _optimise(train_df: pd.DataFrame) -> tuple[dict, float, dict]:
    """
    Parcourt la grille sur train_df.
    Retourne (best_params, best_win_rate, all_metrics).
    """
    keys   = sorted(PARAM_GRID)
    combos = [dict(zip(keys, v)) for v in itertools.product(*[PARAM_GRID[k] for k in keys])]

    best_params: dict = combos[0]
    best_wr     = -1.0
    best_m: dict = {}

    for params in combos:
        m = _run(train_df, params)
        wr = m["win_rate"] if m["n_trades"] >= 3 else 0.0
        if wr > best_wr:
            best_wr, best_params, best_m = wr, params, m

    return best_params, best_wr, best_m


# ── Affichage ─────────────────────────────────────────────────────────────────

_COL = dict(win=4, train=18, test=16, params=26, is_wr=8, oos_wr=8, rpm=8, dd=7, spm=9)
_LINE = "─" * (sum(_COL.values()) + len(_COL) * 2 + 4)

_HDR = (
    f"  {'Win':<{_COL['win']}}  {'Train (IS)':<{_COL['train']}}"
    f"  {'Test (OOS)':<{_COL['test']}}  {'Params optimisés':<{_COL['params']}}"
    f"  {'IS WR':>{_COL['is_wr']}}  {'OOS WR':>{_COL['oos_wr']}}"
    f"  {'R/mois':>{_COL['rpm']}}  {'DD%':>{_COL['dd']}}  {'sig/mois':>{_COL['spm']}}"
)


def _fmt_row(
    win_n: int, train_lbl: str, test_lbl: str,
    params_str: str, is_wr: float, m: dict, n_test_months: float,
) -> str:
    wr_s   = f"{m['win_rate']:.1f}%" if m["n_trades"] else "  —   "
    is_wr_s = f"{is_wr:.1f}%"
    rpm    = m["total_r"] / n_test_months if n_test_months else 0.0
    rpm_s  = f"{rpm:+.2f}R" if m["n_trades"] else "  —  "
    dd_s   = f"{m['max_dd']:.1f}%" if m["n_trades"] else "  —  "
    spm    = m["n_signals"] / n_test_months if n_test_months else 0.0
    spm_s  = f"{spm:.1f}" if m["n_signals"] else "0"

    return (
        f"  {win_n:<{_COL['win']}}  {train_lbl:<{_COL['train']}}"
        f"  {test_lbl:<{_COL['test']}}  {params_str:<{_COL['params']}}"
        f"  {is_wr_s:>{_COL['is_wr']}}  {wr_s:>{_COL['oos_wr']}}"
        f"  {rpm_s:>{_COL['rpm']}}  {dd_s:>{_COL['dd']}}  {spm_s:>{_COL['spm']}}"
    )


def _params_str(p: dict) -> str:
    return f"pivot={p.get('pivot_size','?')}  sl_buf={p.get('sl_buffer_atr','?')}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Chargement et concaténation
    parts = []
    for f in M1_FILES:
        if not f.exists():
            print(f"[SKIP] {f.name} introuvable")
            continue
        print(f"Chargement {f.name} …", end=" ", flush=True)
        parts.append(parse_histdata_csv(f))
        print("OK")

    if not parts:
        print("[ERREUR] Aucun fichier de données.", file=sys.stderr)
        sys.exit(1)

    df_m1  = pd.concat(parts).sort_index()
    df_m1  = df_m1[~df_m1.index.duplicated(keep="first")]
    df_m15 = resample_ohlcv(df_m1, "15min")
    print(f"Total : {len(df_m15):,} barres M15  ({df_m15.index[0]:%Y-%m-%d} → {df_m15.index[-1]:%Y-%m-%d})\n")

    windows = _build_windows(df_m15)
    if not windows:
        print("[ERREUR] Pas assez de mois pour construire des fenêtres.", file=sys.stderr)
        sys.exit(1)

    print(f"Fenêtres : {len(windows)}  (train={TRAIN_MONTHS} mois → test={TEST_MONTHS} mois, pas={STEP_MONTHS} mois)")
    print(f"Grille   : {sum(len(v) for v in PARAM_GRID.values())} valeurs → "
          f"{len(list(itertools.product(*PARAM_GRID.values())))} combinaisons\n")

    # ── Série OPT ─────────────────────────────────────────────────────────────
    title = "SÉRIE OPT — paramètres re-optimisés à chaque fenêtre"
    print(f"{'─'*len(title)}")
    print(title)
    print(f"{'─'*len(title)}")
    print(_HDR)
    print(_LINE)

    opt_oos_wrs: list[float] = []
    opt_oos_rs:  list[float] = []
    opt_results_all: list[TradeResult] = []

    for i, (train_df, test_df, t_lbl, o_lbl) in enumerate(windows, 1):
        n_test_mo = TEST_MONTHS

        best_params, is_wr, _ = _optimise(train_df)
        oos_m = _run(test_df, best_params)

        print(_fmt_row(i, t_lbl, o_lbl, _params_str(best_params), is_wr, oos_m, n_test_mo))

        if oos_m["n_trades"]:
            opt_oos_wrs.append(oos_m["win_rate"])
            opt_oos_rs.append(oos_m["total_r"] / n_test_mo)

    print(_LINE)
    if opt_oos_wrs:
        mean_wr  = sum(opt_oos_wrs) / len(opt_oos_wrs)
        mean_rpm = sum(opt_oos_rs)  / len(opt_oos_rs)
        print(f"  Moyenne OOS  —  WR={mean_wr:.1f}%  R/mois={mean_rpm:+.2f}R")
    else:
        print("  Aucun trade OOS généré sur toutes les fenêtres.")

    # ── Série FIX ─────────────────────────────────────────────────────────────
    print()
    title_fix = "SÉRIE FIX — paramètres figés C3 (pivot=5, sl_buf=0.25), pas d'optimisation"
    print(f"{'─'*len(title_fix)}")
    print(title_fix)
    print(f"{'─'*len(title_fix)}")
    print(_HDR)
    print(_LINE)

    fix_oos_wrs: list[float] = []
    fix_oos_rs:  list[float] = []

    for i, (train_df, test_df, t_lbl, o_lbl) in enumerate(windows, 1):
        n_test_mo = TEST_MONTHS
        fix_params = {"pivot_size": 5, "sl_buffer_atr": 0.25}

        # IS score avec paramètres fixes (pour comparer la dégradation)
        is_m  = _run(train_df, fix_params)
        is_wr = is_m["win_rate"] if is_m["n_trades"] >= 3 else 0.0

        oos_m = _run(test_df, fix_params)
        print(_fmt_row(i, t_lbl, o_lbl, "pivot=5  sl_buf=0.25", is_wr, oos_m, n_test_mo))

        if oos_m["n_trades"]:
            fix_oos_wrs.append(oos_m["win_rate"])
            fix_oos_rs.append(oos_m["total_r"] / n_test_mo)

    print(_LINE)
    if fix_oos_wrs:
        mean_wr  = sum(fix_oos_wrs) / len(fix_oos_wrs)
        mean_rpm = sum(fix_oos_rs)  / len(fix_oos_rs)
        print(f"  Moyenne OOS  —  WR={mean_wr:.1f}%  R/mois={mean_rpm:+.2f}R")

    # ── Diagnostic final ──────────────────────────────────────────────────────
    print()
    print("══════════════════════════════════════════════════════════════════")
    print("  DIAGNOSTIC WALK-FORWARD")
    print("══════════════════════════════════════════════════════════════════")
    print(f"  Données testées OOS : {len(windows) * TEST_MONTHS} mois  "
          f"({len(windows)} fenêtres × {TEST_MONTHS} mois)")
    print()
    if opt_oos_wrs and fix_oos_wrs:
        opt_wr = sum(opt_oos_wrs) / len(opt_oos_wrs)
        fix_wr = sum(fix_oos_wrs) / len(fix_oos_wrs)
        opt_rm = sum(opt_oos_rs)  / len(opt_oos_rs)
        fix_rm = sum(fix_oos_rs)  / len(fix_oos_rs)
        print(f"  OPT  WR OOS moyen : {opt_wr:.1f}%   R/mois OOS : {opt_rm:+.2f}R")
        print(f"  FIX  WR OOS moyen : {fix_wr:.1f}%   R/mois OOS : {fix_rm:+.2f}R")
        print()
        if opt_wr >= 60:
            verdict = "ROBUSTE — WR OOS ≥ 60 %, la structure OB+FVG tient hors-échantillon."
        elif opt_wr >= 50:
            verdict = "FRAGILE — WR OOS entre 50-60 %, edge présent mais insuffisant sans optimisation continue."
        else:
            verdict = "NON ROBUSTE — WR OOS < 50 %, la stratégie perd de l'argent en dehors de sa période d'entraînement."
        print(f"  Verdict : {verdict}")
    print()
    print("  [RAPPEL] Résultats OOS sur seulement 18 mois = faible échantillon.")
    print("  Augmenter les données historiques avant tout déploiement live.")
    print("══════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
