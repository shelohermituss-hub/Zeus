"""
Convergence Strategy — Multi-Market Portfolio Backtest
=======================================================

Objectif : 79-80 % WR  +  5-12 R/mois.

C3 (OB+FVG, pivot=5, sans Sweep) tourne en parallèle sur 4 instruments
(XAUUSD, GBPUSD, USDCHF, CADJPY) pour multiplier la fréquence de signaux
et atteindre la cible de rentabilité.

Architecture :
  • Chaque instrument simulé indépendamment avec son propre spread.
  • Risque identique (% de l'equity initiale) sur chaque marché.
  • R/mois portefeuille  = somme des R/mois individuels.
  • Tableau de résultats à 1 %, 2 % et 3 % de risque par trade.

Données 2025 (12 mois) — toutes disponibles pour les 4 instruments.
Données 2024 (12 mois) — uniquement XAUUSD disponible (référence OOS).

Usage
-----
    python -m zeus.backtest.run_convergence_multimarket
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

# ── Paramètres C3 (meilleure variante identifiée) ────────────────────────────

C3_PARAMS = dict(
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
    risk_reward         = 2.0,
    signal_cooldown     = 12,
)

# ── Configuration des instruments ────────────────────────────────────────────
# Spread  : coût typique en unités de prix (long : entry += spread).
# Slippage: friction supplémentaire sur les sorties SL (ordre stop).

@dataclass
class InstrumentConfig:
    symbol:    str
    spread:    float   # en unités de prix
    slippage:  float   # idem
    files_2025: list[Path]
    files_2024: list[Path]   # vide si données absentes


INSTRUMENTS: list[InstrumentConfig] = [
    InstrumentConfig(
        symbol   = "XAUUSD",
        spread   = 0.30,      # 0.30 USD/oz — spread typique CFD gold
        slippage = 0.10,
        files_2025 = [_ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_2025.csv"],
        files_2024 = [_ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_2024.csv"],
    ),
    InstrumentConfig(
        symbol   = "GBPUSD",
        spread   = 0.00020,   # 2 pips — spread typique GBPUSD
        slippage = 0.00003,
        files_2025 = [_ROOT / "data" / "historical" / "gbpusd" / "m1" / "DAT_MT_GBPUSD_M1_2025.csv"],
        files_2024 = [],
    ),
    InstrumentConfig(
        symbol   = "USDCHF",
        spread   = 0.00020,   # 2 pips — spread typique USDCHF
        slippage = 0.00003,
        files_2025 = [_ROOT / "data" / "historical" / "usdchf" / "m1" / "DAT_MT_USDCHF_M1_2025.csv"],
        files_2024 = [],
    ),
    InstrumentConfig(
        symbol   = "CADJPY",
        spread   = 0.030,     # 3 pips JPY — spread typique CADJPY
        slippage = 0.010,
        files_2025 = [_ROOT / "data" / "historical" / "cadjpy" / "m1" / "DAT_MT_CADJPY_M1_2025.csv"],
        files_2024 = [],
    ),
]

INITIAL_EQ     = 10_000.0
TP1_R          = 0.6
TP1_SIZE       = 0.5
RISK_LEVELS    = [0.01, 0.02, 0.03]   # 1 %, 2 %, 3 %
MONTHS_IN_YEAR = 12.0


# ── Chargement données ────────────────────────────────────────────────────────

def _load_df(paths: list[Path]) -> pd.DataFrame | None:
    """Charge et concatène une liste de fichiers M1, retourne M15 resampleé."""
    frames = []
    for p in paths:
        if not p.exists():
            continue
        df_m1 = parse_histdata_csv(p)
        frames.append(df_m1)
    if not frames:
        return None
    df_m1_all = pd.concat(frames).sort_index()
    return resample_ohlcv(df_m1_all, "15min")


# ── Simulation d'un instrument ────────────────────────────────────────────────

@dataclass
class InstrumentResult:
    symbol:    str
    n_months:  float
    n_signals: int
    n_expired: int
    results:   list[TradeResult]
    metrics:   dict
    spread:    float
    slippage:  float


def _run_instrument(
    cfg: InstrumentConfig, df_m15: pd.DataFrame, risk_pct: float
) -> InstrumentResult:
    strategy = ConvergenceStrategy(**C3_PARAMS)
    signals  = strategy.run(df_m15)

    results, n_expired = simulate_all(
        signals,
        df_m15,
        initial_equity = INITIAL_EQ,
        risk_pct       = risk_pct,
        spread         = cfg.spread,
        tp1_r          = TP1_R,
        tp1_size       = TP1_SIZE,
        slippage_ticks = cfg.slippage,
    )

    n_months = float(df_m15.resample("ME").last().dropna().shape[0])
    metrics  = compute_metrics(results, INITIAL_EQ, len(signals), n_expired) if results else {}

    return InstrumentResult(
        symbol    = cfg.symbol,
        n_months  = n_months,
        n_signals = len(signals),
        n_expired = n_expired,
        results   = results,
        metrics   = metrics,
        spread    = cfg.spread,
        slippage  = cfg.slippage,
    )


# ── Affichage ─────────────────────────────────────────────────────────────────

_W = dict(sym=8, wr=7, avgr=7, rpm=9, dd=6, spm=9, tr=9)
_SEP = "─" * (sum(_W.values()) + len(_W) * 2 + 4)
_HDR = (
    f"  {'Sym':<{_W['sym']}}  {'WR%':>{_W['wr']}}  {'avgR':>{_W['avgr']}}"
    f"  {'R/mois':>{_W['rpm']}}  {'DD%':>{_W['dd']}}"
    f"  {'sig/mois':>{_W['spm']}}  {'TotalR':>{_W['tr']}}"
)


def _row(ir: InstrumentResult) -> str:
    m = ir.metrics
    if not m:
        return f"  {ir.symbol:<{_W['sym']}}  {'—':>{_W['wr']}}  {'—':>{_W['avgr']}}  {'—':>{_W['rpm']}}  {'—':>{_W['dd']}}  {'0.0':>{_W['spm']}}  {'—':>{_W['tr']}}"

    wr   = m["win_rate"]
    avgr = m["total_r"] / m["n_trades"] if m["n_trades"] else 0.0
    rpm  = m["total_r"] / ir.n_months
    dd   = m["max_dd"]
    spm  = m["n_signals"] / ir.n_months
    tr   = m["total_r"]

    return (
        f"  {ir.symbol:<{_W['sym']}}"
        f"  {wr:.1f}%{'':<{_W['wr']-5}}"
        f"  {avgr:+.2f}R{'':<{_W['avgr']-5}}"
        f"  {rpm:+.2f}R{'':<{_W['rpm']-5}}"
        f"  {dd:.1f}%{'':<{_W['dd']-4}}"
        f"  {spm:.1f}{'':<{_W['spm']-3}}"
        f"  {tr:+.1f}R"
    )


def _monthly_detail(ir: InstrumentResult) -> str:
    """Détail mensuel pour un instrument."""
    by_month: dict[str, list[TradeResult]] = {}
    for r in ir.results:
        key = r.signal.formed_at.strftime("%Y-%m")
        by_month.setdefault(key, []).append(r)

    lines = []
    for month in sorted(by_month):
        rs    = by_month[month]
        wins  = sum(1 for r in rs if r.outcome == "win")
        total = sum(1 for r in rs if r.outcome in ("win", "loss"))
        wr    = wins / total * 100 if total else 0.0
        r_sum = sum(r.pnl_r for r in rs)
        lines.append(f"      {month}  WR={wr:5.1f}%  trades={len(rs):2d}  R={r_sum:+5.2f}")
    return "\n".join(lines)


def _print_portfolio_block(
    inst_results: list[InstrumentResult], risk_pct: float
) -> None:
    """Affiche tableau par instrument + agrégat portefeuille."""
    pct_label = f"{risk_pct*100:.0f}%"

    # Calcul agrégats portefeuille
    all_metrics = [ir.metrics for ir in inst_results if ir.metrics]
    if not all_metrics:
        print("  Aucun signal généré.\n")
        return

    total_trades  = sum(m["n_trades"]  for m in all_metrics)
    total_wins    = sum(m["n_wins"]    for m in all_metrics)
    total_r       = sum(m["total_r"]   for m in all_metrics)
    total_signals = sum(ir.n_signals   for ir in inst_results)
    n_months_ref  = inst_results[0].n_months  # même période pour tous

    port_wr       = total_wins / total_trades * 100 if total_trades else 0.0
    port_rpm      = total_r / n_months_ref
    port_avgr     = total_r / total_trades if total_trades else 0.0
    port_spm      = total_signals / n_months_ref

    # DD portefeuille = somme des DD individuels (hypothèse indépendance)
    port_dd_ind   = sum(m["max_dd"] for m in all_metrics)
    # Retour % mensuel au niveau risque sélectionné
    monthly_pct   = port_rpm * risk_pct * 100

    box_title = f"Portefeuille C3 · 4 instruments · risque {pct_label}/trade"
    box = "═" * (len(box_title) + 4)
    print(f"\n╔{box}╗")
    print(f"║  {box_title}  ║")
    print(f"╚{box}╝")
    print(_HDR)
    print(_SEP)

    for ir in inst_results:
        print(_row(ir))

    print(_SEP)

    # Ligne portefeuille
    print(
        f"  {'PORTFOLIO':<{_W['sym']}}"
        f"  {port_wr:.1f}%{'':<{_W['wr']-5}}"
        f"  {port_avgr:+.2f}R{'':<{_W['avgr']-5}}"
        f"  {port_rpm:+.2f}R{'':<{_W['rpm']-5}}"
        f"  {port_dd_ind:.1f}%{'':<{_W['dd']-4}}"
        f"  {port_spm:.1f}{'':<{_W['spm']-3}}"
        f"  {total_r:+.1f}R"
    )
    print(_SEP)
    print(
        f"  → Risque {pct_label}/trade  "
        f"Gain mensuel estimé : {port_rpm:+.2f}R  "
        f"≈ {monthly_pct:+.2f}%/mois de croissance compte"
    )


def _print_risk_summary(
    inst_results_by_risk: dict[float, list[InstrumentResult]],
) -> None:
    """Comparatif risque : tableau 1 % / 2 % / 3 %."""
    print("\n" + "═" * 68)
    print("  SYNTHÈSE PAR NIVEAU DE RISQUE  (portefeuille 4 instruments)")
    print("═" * 68)
    print(f"  {'Risque':>7}  {'WR%':>6}  {'R/mois':>8}  {'%/mois':>8}  {'DD% ind.':>9}")
    print("  " + "─" * 46)

    n_months_ref = inst_results_by_risk[RISK_LEVELS[0]][0].n_months

    for risk_pct in RISK_LEVELS:
        inst_results = inst_results_by_risk[risk_pct]
        all_m = [ir.metrics for ir in inst_results if ir.metrics]
        if not all_m:
            continue

        total_trades = sum(m["n_trades"] for m in all_m)
        total_wins   = sum(m["n_wins"]   for m in all_m)
        total_r      = sum(m["total_r"]  for m in all_m)
        port_wr      = total_wins / total_trades * 100 if total_trades else 0.0
        port_rpm     = total_r / n_months_ref
        port_dd      = sum(m["max_dd"]   for m in all_m)
        monthly_pct  = port_rpm * risk_pct * 100

        target_marker = " ← OBJECTIF ✓" if monthly_pct >= 5.0 else ""
        print(
            f"  {risk_pct*100:.0f}%{'':<5}  {port_wr:.1f}%  "
            f"{port_rpm:>+7.2f}R  {monthly_pct:>+7.2f}%  "
            f"{port_dd:>8.1f}%{target_marker}"
        )

    print("  " + "─" * 46)
    print("  [NOTE] DD% ind. = somme des DD individuels (estimation conservative).")
    print("         En pratique, corrélations < 1 → DD réel souvent inférieur.")
    print("═" * 68)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Chargement des données ────────────────────────────────────────────────
    print("Chargement des données…")
    dfs_2025: dict[str, pd.DataFrame] = {}
    dfs_2024: dict[str, pd.DataFrame] = {}

    for cfg in INSTRUMENTS:
        df25 = _load_df(cfg.files_2025)
        if df25 is not None:
            dfs_2025[cfg.symbol] = df25
            n = len(df25)
            print(f"  {cfg.symbol} 2025 : {n:,} barres M15  OK")
        else:
            print(f"  {cfg.symbol} 2025 : données absentes — ignoré")

        df24 = _load_df(cfg.files_2024)
        if df24 is not None:
            dfs_2024[cfg.symbol] = df24

    if not dfs_2025:
        print("[ERREUR] Aucun fichier 2025 trouvé.", file=sys.stderr)
        sys.exit(1)

    # ── Backtests 2025 (portefeuille complet) ─────────────────────────────────
    inst_results_by_risk: dict[float, list[InstrumentResult]] = {}

    for risk_pct in RISK_LEVELS:
        inst_results: list[InstrumentResult] = []
        for cfg in INSTRUMENTS:
            if cfg.symbol not in dfs_2025:
                continue
            ir = _run_instrument(cfg, dfs_2025[cfg.symbol], risk_pct)
            inst_results.append(ir)
        inst_results_by_risk[risk_pct] = inst_results

    # Affichage détaillé pour risque 2 % (niveau cible recommandé)
    _print_portfolio_block(inst_results_by_risk[0.02], risk_pct=0.02)

    # Détail mensuel XAUUSD uniquement (référence)
    for ir in inst_results_by_risk[0.02]:
        if ir.symbol == "XAUUSD" and ir.results:
            print(f"\n  Détail mensuel XAUUSD (risque 2 %) :")
            print(_monthly_detail(ir))

    # Synthèse multi-risque
    _print_risk_summary(inst_results_by_risk)

    # ── XAUUSD OOS 2024 ───────────────────────────────────────────────────────
    if "XAUUSD" in dfs_2024:
        print("\n" + "═" * 68)
        print("  XAUUSD 2024 — hors-échantillon (données OOS disponibles)")
        print("═" * 68)
        xau_cfg = next(c for c in INSTRUMENTS if c.symbol == "XAUUSD")
        for risk_pct in RISK_LEVELS:
            ir_oos = _run_instrument(xau_cfg, dfs_2024["XAUUSD"], risk_pct)
            m = ir_oos.metrics
            if not m:
                continue
            rpm = m["total_r"] / ir_oos.n_months
            monthly_pct = rpm * risk_pct * 100
            print(
                f"  risque {risk_pct*100:.0f}%  →  WR={m['win_rate']:.1f}%  "
                f"R/mois={rpm:+.2f}R  %/mois={monthly_pct:+.2f}%  "
                f"DD={m['max_dd']:.1f}%"
            )
        print("═" * 68)

    # ── Diagnostic final ──────────────────────────────────────────────────────
    risk_2pct_results = inst_results_by_risk[0.02]
    all_m = [ir.metrics for ir in risk_2pct_results if ir.metrics]
    if all_m:
        n_months_ref  = risk_2pct_results[0].n_months
        total_trades  = sum(m["n_trades"] for m in all_m)
        total_wins    = sum(m["n_wins"]   for m in all_m)
        total_r       = sum(m["total_r"]  for m in all_m)
        port_wr       = total_wins / total_trades * 100 if total_trades else 0.0
        port_rpm      = total_r / n_months_ref
        monthly_pct   = port_rpm * 0.02 * 100

        print("\n" + "═" * 68)
        print("  DIAGNOSTIC PORTEFEUILLE")
        print("═" * 68)
        print(f"  Instruments actifs    : {len(all_m)}")
        print(f"  Données               : 2025 (12 mois, in-sample)")
        print(f"  Paramètres C3         : OB+FVG  pivot=5  sl_buf=0.25  RR=2.0")
        print(f"")
        print(f"  WR portefeuille       : {port_wr:.1f}%")
        print(f"  R/mois portefeuille   : {port_rpm:+.2f}R")
        print(f"")
        print(f"  À 2 % risque/trade    : {monthly_pct:+.2f}%/mois  ({monthly_pct*12:+.1f}%/an)")

        target_ok = port_wr >= 75.0 and monthly_pct >= 5.0
        verdict = "OBJECTIFS ATTEINTS" if target_ok else "OBJECTIFS PARTIELS"
        wr_ok    = "✓" if port_wr >= 75.0 else "✗"
        rpm_ok   = "✓" if monthly_pct >= 5.0 else "✗"

        print(f"")
        print(f"  {wr_ok} WR ≥ 75 %            : {port_wr:.1f}%")
        print(f"  {rpm_ok} Rendement ≥ 5 %/mois  : {monthly_pct:.2f}% (à 2 %/trade)")
        print(f"")
        print(f"  Verdict : {verdict}")
        print(f"")
        print(f"  [RAPPEL] Données 2025 in-sample uniquement pour GBPUSD/USDCHF/CADJPY.")
        print(f"  Validation walk-forward multi-instrument requise avant déploiement.")
        print("═" * 68)


if __name__ == "__main__":
    main()
