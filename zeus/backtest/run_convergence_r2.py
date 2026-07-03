"""
Convergence Strategy — Round 2 Backtest
========================================

Applique les corrections identifiées en Round 1 :
  - Killzone OFF  (trop restrictif, ne filtre pas les mauvais trades)
  - Pivot=5       (plus d'OBs, plus de signaux sans dégrader la WR)
  - Test require_fvg ON vs OFF (contribution isolée de chaque filtre)

Résultats sur une année complète (XAUUSD M15 2025 = 12 mois d'affilée).
Vérification hors-échantillon sur 2024.

Tableau de sortie par variante :
  Var  Description            WR%    avgR   R/mois   DD%   sig/mois  Total_R

Spreads & coûts modèle réaliste :
  SPREAD          0.30 USD/oz
  SLIPPAGE_TICKS  0.10 USD (SL exits uniquement, F-09)
  TP1             0.6R partial (50 % position) → SL → BE
  RISK_PCT        1 % par trade

Usage
-----
    python -m zeus.backtest.run_convergence_r2
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv, resample_ohlcv
from zeus.backtest.sd_simulation import TradeResult, compute_metrics, simulate_all
from zeus.strategy.convergence_strategy import ConvergenceStrategy

_ROOT = Path(__file__).parent.parent.parent

# ── Données ───────────────────────────────────────────────────────────────────

DATASETS = {
    "2025": _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_2025.csv",
    "2024": _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_2024.csv",
}

SPREAD         = 0.30    # USD/oz
SLIPPAGE       = 0.10    # USD (exits SL uniquement)
INITIAL_EQ     = 10_000.0
RISK_PCT       = 0.01
TP1_R          = 0.6
TP1_SIZE       = 0.5
MONTHS_IN_YEAR = 12.0

# ── Variantes Round 2 ─────────────────────────────────────────────────────────

@dataclass
class Variant:
    code:   str
    label:  str
    kwargs: dict


# Base commune Round 2 : killzone OFF, pivot=5, D1/H4 ON
_BASE = dict(
    pivot_size          = 5,
    long_only           = True,
    use_d1_ema          = True,
    d1_ema_span         = 200,
    use_h4_trend        = True,
    h4_ema_span         = 50,
    h4_slope_lb         = 3,
    use_killzone        = False,      # ← R1 fix #1
    use_ltf_sweep       = True,
    ltf_sweep_lookback  = 5,
    require_fvg         = True,
    require_bullish_bar = True,
    min_ob_age          = 2,
    max_ob_age          = 200,
    sl_buffer_atr       = 0.25,
    atr_period          = 14,
    risk_reward         = 2.0,
    signal_cooldown     = 12,
)

VARIANTS: list[Variant] = [
    # ── Référence Round 2 ────────────────────────────────────────────────────
    Variant("C1", "OB+FVG+Sweep  pivot=5",   {**_BASE}),

    # ── Ablations : isole la contribution de chaque filtre ────────────────────
    Variant("C2", "OB+Sweep (sans FVG)",      {**_BASE, "require_fvg": False}),
    Variant("C3", "OB+FVG  (sans Sweep)",     {**_BASE, "use_ltf_sweep": False}),
    Variant("C4", "OB seul",                  {**_BASE, "require_fvg": False,
                                                         "use_ltf_sweep": False}),

    # ── Pivot encore plus petit ────────────────────────────────────────────────
    Variant("C5", "OB+FVG+Sweep  pivot=3",   {**_BASE, "pivot_size": 3}),
    Variant("C6", "OB seul        pivot=3",   {**_BASE, "pivot_size": 3,
                                                         "require_fvg": False,
                                                         "use_ltf_sweep": False}),

    # ── Cooldown plus court (signaux plus fréquents) ──────────────────────────
    Variant("C7", "OB+FVG+Sweep  cool=6",    {**_BASE, "signal_cooldown": 6}),

    # ── RR différents ─────────────────────────────────────────────────────────
    Variant("C8", "OB+FVG+Sweep  RR=1.5",    {**_BASE, "risk_reward": 1.5}),
    Variant("C9", "OB+FVG+Sweep  RR=3.0",    {**_BASE, "risk_reward": 3.0}),

    # ── Killzone RE-activé sur base pivot=5 ───────────────────────────────────
    Variant("C10","OB+FVG+KZ     pivot=5",   {**_BASE, "use_killzone": True}),
]


# ── Helpers affichage ─────────────────────────────────────────────────────────

W  = dict(code=5, label=28, wr=7, avgr=7, rpm=8, dd=6, spm=10, tr=9)

_SEP  = "─" * (sum(W.values()) + len(W) * 2 + 2)
_HDR  = (
    f"  {'Var':<{W['code']}}  {'Description':<{W['label']}}"
    f"  {'WR%':>{W['wr']}}  {'avgR':>{W['avgr']}}"
    f"  {'R/mois':>{W['rpm']}}  {'DD%':>{W['dd']}}"
    f"  {'sig/mois':>{W['spm']}}  {'TotalR':>{W['tr']}}"
)


def _row(code: str, label: str, m: dict, n_months: float) -> str:
    wr   = m["win_rate"]
    avgr = m["total_r"] / m["n_trades"] if m["n_trades"] else 0.0
    rpm  = m["total_r"] / n_months
    dd   = m["max_dd"]
    spm  = m["n_signals"] / n_months
    tr   = m["total_r"]

    wr_s  = f"{wr:.1f}%"
    avgr_s = f"{avgr:+.2f}R"
    rpm_s  = f"{rpm:+.2f}R"
    dd_s   = f"{dd:.1f}%"
    spm_s  = f"{spm:.1f}"
    tr_s   = f"{tr:+.1f}R"

    return (
        f"  {code:<{W['code']}}  {label:<{W['label']}}"
        f"  {wr_s:>{W['wr']}}  {avgr_s:>{W['avgr']}}"
        f"  {rpm_s:>{W['rpm']}}  {dd_s:>{W['dd']}}"
        f"  {spm_s:>{W['spm']}}  {tr_s:>{W['tr']}}"
    )


def _monthly_breakdown(results: list[TradeResult]) -> str:
    """One line per month: mois  WR%  trades  R"""
    by_month: dict[str, list[TradeResult]] = {}
    for r in results:
        key = r.signal.formed_at.strftime("%Y-%m")
        by_month.setdefault(key, []).append(r)

    lines = []
    for month in sorted(by_month):
        rs    = by_month[month]
        wins  = sum(1 for r in rs if r.outcome == "win")
        total = sum(1 for r in rs if r.outcome in ("win", "loss"))
        wr    = wins / total * 100 if total else 0.0
        r_sum = sum(r.pnl_r for r in rs)
        lines.append(f"      {month}  WR={wr:5.1f}%  trades={len(rs):2d}  R={r_sum:+5.1f}")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def _run_year(year: str, df_m15: pd.DataFrame, n_months: float) -> None:
    tag = f"XAUUSD M15 · {year} ({int(n_months)} mois)"
    box = "═" * (len(tag) + 4)
    print(f"\n╔{box}╗")
    print(f"║  {tag}  ║")
    print(f"╚{box}╝")
    print(_HDR)
    print(_SEP)

    for var in VARIANTS:
        strategy = ConvergenceStrategy(**var.kwargs)
        signals  = strategy.run(df_m15)

        results, n_expired = simulate_all(
            signals,
            df_m15,
            initial_equity = INITIAL_EQ,
            risk_pct       = RISK_PCT,
            spread         = SPREAD,
            tp1_r          = TP1_R,
            tp1_size       = TP1_SIZE,
            slippage_ticks = SLIPPAGE,
        )

        if not results:
            # Ligne vide mais présente pour comparaison
            print(
                f"  {var.code:<{W['code']}}  {var.label:<{W['label']}}"
                f"  {'—':>{W['wr']}}  {'—':>{W['avgr']}}"
                f"  {'—':>{W['rpm']}}  {'—':>{W['dd']}}"
                f"  {'0.0':>{W['spm']}}  {'0.0R':>{W['tr']}}"
            )
            continue

        m = compute_metrics(results, INITIAL_EQ, len(signals), n_expired)
        print(_row(var.code, var.label, m, n_months))

        # Détail mensuel uniquement pour C1 (référence)
        if var.code == "C1" and results:
            print(_monthly_breakdown(results))

    print(_SEP)
    print(
        f"  Paramètres : spread={SPREAD} USD/oz  "
        f"slippage={SLIPPAGE} USD  TP1={TP1_R}R  risk={RISK_PCT*100:.0f}%/trade"
    )
    print("  [AVERTISSEMENT] Biais d'optimisation in-sample présent.")
    print("  Validation walk-forward requise avant tout déploiement live.")


def main() -> None:
    loaded: dict[str, pd.DataFrame] = {}

    for year, path in DATASETS.items():
        if not path.exists():
            print(f"[SKIP] Fichier introuvable : {path.name}")
            continue
        print(f"Chargement {path.name} …", end=" ", flush=True)
        df_m1  = parse_histdata_csv(path)
        df_m15 = resample_ohlcv(df_m1, "15min")
        loaded[year] = df_m15
        print(f"OK  ({len(df_m15):,} barres M15)")

    if not loaded:
        print("[ERREUR] Aucun fichier de données trouvé.", file=sys.stderr)
        sys.exit(1)

    # ── 2025 — in-sample ──────────────────────────────────────────────────────
    if "2025" in loaded:
        df = loaded["2025"]
        # Comptage des mois réels dans les données
        n_mo = df.resample("ME").last().dropna().shape[0]
        _run_year("2025 (in-sample)", df, float(n_mo))

    # ── 2024 — hors-échantillon ────────────────────────────────────────────────
    if "2024" in loaded:
        df = loaded["2024"]
        n_mo = df.resample("ME").last().dropna().shape[0]
        _run_year("2024 (hors-échantillon)", df, float(n_mo))


if __name__ == "__main__":
    main()
