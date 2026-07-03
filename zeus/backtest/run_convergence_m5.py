"""
Convergence Strategy — XAUUSD M5 Backtest
==========================================

Objectif : atteindre 5-12R/mois en descendant d'une unité de temps.

Sur M5, le moteur détecte 3-5× plus d'Order Blocks et de FVGs qu'en M15.
Si le WR reste ≥ 70 %, la fréquence accrue débloque la cible de rendement.

Variants testés :
  MA  → OB+FVG  pivot=5  sl_buf=0.25  RR=2.0   (C3 direct port vers M5)
  MB  → OB+FVG  pivot=3  sl_buf=0.25  RR=2.0   (plus d'OBs)
  MC  → OB+FVG  pivot=5  sl_buf=0.40  RR=2.0   (SL plus large → moins de stops)
  MD  → OB+FVG  pivot=5  sl_buf=0.25  RR=3.0   (RR élevé)
  ME  → OB+FVG  pivot=3  sl_buf=0.40  RR=3.0   (fréquence + RR élevé)
  MF  → OB+FVG  pivot=5  cool=4       RR=2.0   (cooldown réduit à 20 min)

Données : XAUUSD 2025 (in-sample), XAUUSD 2024 (OOS).

Usage
-----
    python -m zeus.backtest.run_convergence_m5
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

DATASETS = {
    "2025": _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_2025.csv",
    "2024": _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_2024.csv",
}

SPREAD     = 0.30
SLIPPAGE   = 0.10
INITIAL_EQ = 10_000.0
TP1_R      = 0.6
TP1_SIZE   = 0.5

# ── Variantes M5 ─────────────────────────────────────────────────────────────

@dataclass
class Variant:
    code:   str
    label:  str
    kwargs: dict


_BASE = dict(
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
    signal_cooldown     = 12,   # 12 × M5 = 60 min
)

VARIANTS: list[Variant] = [
    Variant("MA", "OB+FVG  p=5  sl=0.25  RR=2  cool=12", {**_BASE, "pivot_size": 5, "sl_buffer_atr": 0.25, "risk_reward": 2.0}),
    Variant("MB", "OB+FVG  p=3  sl=0.25  RR=2  cool=12", {**_BASE, "pivot_size": 3, "sl_buffer_atr": 0.25, "risk_reward": 2.0}),
    Variant("MC", "OB+FVG  p=5  sl=0.40  RR=2  cool=12", {**_BASE, "pivot_size": 5, "sl_buffer_atr": 0.40, "risk_reward": 2.0}),
    Variant("MD", "OB+FVG  p=5  sl=0.25  RR=3  cool=12", {**_BASE, "pivot_size": 5, "sl_buffer_atr": 0.25, "risk_reward": 3.0}),
    Variant("ME", "OB+FVG  p=3  sl=0.40  RR=3  cool=12", {**_BASE, "pivot_size": 3, "sl_buffer_atr": 0.40, "risk_reward": 3.0}),
    Variant("MF", "OB+FVG  p=5  sl=0.25  RR=2  cool=4",  {**_BASE, "pivot_size": 5, "sl_buffer_atr": 0.25, "risk_reward": 2.0, "signal_cooldown": 4}),
]

# ── Affichage ─────────────────────────────────────────────────────────────────

W = dict(code=4, label=33, wr=7, avgr=7, rpm=8, dd=6, spm=9, tr=8)
_SEP = "─" * (sum(W.values()) + len(W) * 2 + 2)
_HDR = (
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
    return (
        f"  {code:<{W['code']}}  {label:<{W['label']}}"
        f"  {wr:.1f}%{'':<{W['wr']-5}}"
        f"  {avgr:+.2f}R{'':<{W['avgr']-5}}"
        f"  {rpm:+.2f}R{'':<{W['rpm']-5}}"
        f"  {dd:.1f}%{'':<{W['dd']-4}}"
        f"  {spm:.1f}{'':<{W['spm']-3}}"
        f"  {tr:+.1f}R"
    )


def _monthly_breakdown(results: list[TradeResult]) -> str:
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
        lines.append(f"      {month}  WR={wr:5.1f}%  trades={len(rs):2d}  R={r_sum:+5.2f}")
    return "\n".join(lines)


def _run_year(year: str, df_m5: pd.DataFrame, n_months: float, risk_pct: float = 0.01) -> list[dict]:
    """Retourne la liste des métriques par variante pour réutilisation."""
    tag = f"XAUUSD M5 · {year} ({int(n_months)} mois) · risque {risk_pct*100:.0f}%"
    box = "═" * (len(tag) + 4)
    print(f"\n╔{box}╗")
    print(f"║  {tag}  ║")
    print(f"╚{box}╝")
    print(_HDR)
    print(_SEP)

    all_metrics = []

    for var in VARIANTS:
        strategy = ConvergenceStrategy(**var.kwargs)
        signals  = strategy.run(df_m5)

        results, n_expired = simulate_all(
            signals,
            df_m5,
            initial_equity = INITIAL_EQ,
            risk_pct       = risk_pct,
            spread         = SPREAD,
            tp1_r          = TP1_R,
            tp1_size       = TP1_SIZE,
            slippage_ticks = SLIPPAGE,
        )

        if not results:
            print(
                f"  {var.code:<{W['code']}}  {var.label:<{W['label']}}"
                f"  {'—':>{W['wr']}}  {'—':>{W['avgr']}}"
                f"  {'—':>{W['rpm']}}  {'—':>{W['dd']}}"
                f"  {len(signals)/n_months:.1f}{'':<{W['spm']-3}}"
                f"  {'0.0R':>{W['tr']}}"
            )
            all_metrics.append({})
            continue

        m = compute_metrics(results, INITIAL_EQ, len(signals), n_expired)
        print(_row(var.code, var.label, m, n_months))
        all_metrics.append(m)

        if var.code == "MA" and results:
            print(_monthly_breakdown(results))

    print(_SEP)
    print(f"  Spread={SPREAD} USD/oz  Slip={SLIPPAGE} USD  TP1={TP1_R}R  Risque={risk_pct*100:.0f}%/trade")
    return all_metrics


def main() -> None:
    loaded: dict[str, pd.DataFrame] = {}

    for year, path in DATASETS.items():
        if not path.exists():
            print(f"[SKIP] {path.name}")
            continue
        print(f"Chargement {path.name} …", end=" ", flush=True)
        df_m1 = parse_histdata_csv(path)
        df_m5 = resample_ohlcv(df_m1, "5min")
        loaded[year] = df_m5
        print(f"OK  ({len(df_m5):,} barres M5)")

    if not loaded:
        print("[ERREUR] Aucun fichier trouvé.", file=sys.stderr)
        sys.exit(1)

    # ── 2025 IS — 1 % et 2 % risque ──────────────────────────────────────────
    if "2025" in loaded:
        df  = loaded["2025"]
        n_mo = float(df.resample("ME").last().dropna().shape[0])
        metrics_1pct = _run_year("2025 IS", df, n_mo, risk_pct=0.01)
        metrics_2pct = _run_year("2025 IS", df, n_mo, risk_pct=0.02)

    # ── 2024 OOS — meilleur variant identifié (MA) ────────────────────────────
    if "2024" in loaded:
        df24 = loaded["2024"]
        n_mo24 = float(df24.resample("ME").last().dropna().shape[0])
        print(f"\n{'═'*68}")
        print(f"  XAUUSD M5 · 2024 (OOS) — comparatif risque 1 %/2 %")
        print(f"{'═'*68}")
        for risk_pct in [0.01, 0.02]:
            for var in VARIANTS:
                strategy = ConvergenceStrategy(**var.kwargs)
                signals  = strategy.run(df24)
                results, n_exp = simulate_all(
                    signals, df24,
                    initial_equity=INITIAL_EQ, risk_pct=risk_pct,
                    spread=SPREAD, tp1_r=TP1_R, tp1_size=TP1_SIZE,
                    slippage_ticks=SLIPPAGE,
                )
                if not results:
                    print(f"  {var.code} risque={risk_pct*100:.0f}%  →  aucun trade")
                    continue
                m   = compute_metrics(results, INITIAL_EQ, len(signals), n_exp)
                rpm = m["total_r"] / n_mo24
                print(
                    f"  {var.code} risque={risk_pct*100:.0f}%  →  "
                    f"WR={m['win_rate']:.1f}%  R/mois={rpm:+.2f}R  "
                    f"DD={m['max_dd']:.1f}%  sig/mois={len(signals)/n_mo24:.1f}"
                )

    # ── Synthèse cible ────────────────────────────────────────────────────────
    if "2025" in loaded:
        df  = loaded["2025"]
        n_mo = float(df.resample("ME").last().dropna().shape[0])

        print("\n" + "═" * 68)
        print("  SYNTHÈSE — Chemin vers 5R/mois  (données IS 2025)")
        print("═" * 68)
        print(f"  {'Var':<4}  {'WR%':>6}  {'R/mois@1%':>10}  {'R/mois@2%':>10}  {'sig/mois':>9}  {'DD@2%':>7}")
        print("  " + "─" * 52)

        for var in VARIANTS:
            strategy = ConvergenceStrategy(**var.kwargs)
            signals  = strategy.run(df)

            r1, _ = simulate_all(signals, df, initial_equity=INITIAL_EQ,
                                  risk_pct=0.01, spread=SPREAD, tp1_r=TP1_R,
                                  tp1_size=TP1_SIZE, slippage_ticks=SLIPPAGE)
            r2, _ = simulate_all(signals, df, initial_equity=INITIAL_EQ,
                                  risk_pct=0.02, spread=SPREAD, tp1_r=TP1_R,
                                  tp1_size=TP1_SIZE, slippage_ticks=SLIPPAGE)

            if not r1:
                continue

            m1 = compute_metrics(r1, INITIAL_EQ, len(signals), 0)
            m2 = compute_metrics(r2, INITIAL_EQ, len(signals), 0)

            rpm1 = m1["total_r"] / n_mo
            rpm2 = m2["total_r"] / n_mo
            spm  = len(signals) / n_mo

            target = " ✓ OBJECTIF" if m1["win_rate"] >= 70.0 and rpm2 >= 5.0 else ""
            print(
                f"  {var.code:<4}  {m1['win_rate']:>5.1f}%  "
                f"{rpm1:>+9.2f}R  {rpm2:>+9.2f}R  "
                f"{spm:>8.1f}  {m2['max_dd']:>6.1f}%{target}"
            )

        print("  " + "─" * 52)
        print("  Cible : WR ≥ 70 %  ET  R/mois@2% ≥ 5R")
        print("  [!] IS 2025 uniquement — valider en walk-forward avant déploiement.")
        print("═" * 68)


if __name__ == "__main__":
    main()
