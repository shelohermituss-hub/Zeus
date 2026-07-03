"""
FVG Retest Strategy — Backtest & Walk-Forward
=============================================

Objectif : dépasser C3 (78 % WR, 4.85 %/mois) en fréquence de signaux tout
en maintenant WR ≥ 70 %.

Différence clé vs C3
---------------------
  C3          : OB est le signal primaire, FVG est le filtre de confirmation.
  FVGRetest   : FVG est le signal primaire ; le retest de l'imbalance est
                l'entrée.  Cela génère naturellement plus de setups car il
                n'est pas nécessaire de trouver une confluence OB+FVG.

Filtres testés (opt-in, code A-I dans FVGRetestStrategy)
---------------------------------------------------------
  A  require_bullish_bar      : barre d'entrée haussière (close > open)
  B  fvg_max_penetration_pct  : pénétration maximale dans le FVG
  D  use_m15_bos_filter       : BOS M15 haussier récent requis
  E  min_impulse_atr_mult     : taille minimale de la barre impulse du FVG
  H  use_sd_confluence        : zone de demande S&D en confluence
  I  use_d1_regime            : filtre de régime D1 EMA(200)

Variantes testées
-----------------
  FR1  Base H4 seul — fréquence max, aucun filtre supplémentaire
  FR2  + D1 régime (long-only quand D1 > EMA200)
  FR3  + D1 + barre haussière (filtre A)
  FR4  + D1 + impulse forte (filtre E ≥ 0.5 ATR)
  FR5  RR=2.0  avec D1 (FR2 params)
  FR6  RR=5.0  avec D1
  FR7  RR=8.0  avec D1 (équivalent RR de C3-final)
  FR8  D1 + S&D confluence (filtre H)
  FR9  D1 + BOS M15 récent (filtre D)
  FR10 D1 + A + E + BOS (combo filtres)

Données
-------
  2025 in-sample  (IS)
  2024 hors-échantillon (OOS)

Walk-Forward OOS (6 fenêtres × 3 mois, Jan 2024 – Déc 2025)
  Identique à run_convergence_final.py pour comparaison directe.

Usage
-----
    python -m zeus.backtest.run_fvg_retest
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv, resample_ohlcv
from zeus.backtest.sd_simulation import compute_metrics, simulate_all
from zeus.strategy.fvg_retest_strategy import FVGRetestStrategy

_ROOT = Path(__file__).parent.parent.parent

DATASETS = {
    "2025": _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_2025.csv",
    "2024": _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_2024.csv",
}

SPREAD     = 0.30
SLIPPAGE   = 0.10
INITIAL_EQ = 10_000.0
RISK_PCT   = 0.03    # 3 % par trade (idem run_convergence_final)
TP1_R      = 0.6
TP1_SIZE   = 0.5

WF_WINDOWS = [
    ("2024-07", "2024-09"),
    ("2024-10", "2024-12"),
    ("2025-01", "2025-03"),
    ("2025-04", "2025-06"),
    ("2025-07", "2025-09"),
    ("2025-10", "2025-12"),
]

# ── Variantes ─────────────────────────────────────────────────────────────────

@dataclass
class Variant:
    code:   str
    label:  str
    kwargs: dict


_BASE = dict(
    long_only            = True,
    use_h4_trend         = True,
    h4_ema_span          = 50,
    h4_slope_lb          = 3,
    use_session_filter   = True,
    session_start_utc    = 7,
    session_end_utc      = 21,
    sl_buffer_atr_mult   = 0.10,
    atr_period           = 14,
    min_fvg_atr_mult     = 0.10,
    max_fvg_age_bars     = 96,
    signal_cooldown      = 12,
    max_signals_per_day  = 3,
    risk_reward          = 3.0,
    # filtres OFF par défaut
    require_bullish_bar     = False,
    fvg_max_penetration_pct = 1.0,
    use_h1_trend            = False,
    use_m15_bos_filter      = False,
    min_impulse_atr_mult    = 0.0,
    use_sd_confluence       = False,
    use_d1_regime           = False,
    d1_ema_span             = 200,
)

VARIANTS: list[Variant] = [
    # ── Fréquence max — référence ─────────────────────────────────────────────
    Variant("FR1", "H4 seul   RR=3",        {**_BASE}),

    # ── + Filtre I : D1 régime ────────────────────────────────────────────────
    Variant("FR2", "D1+H4     RR=3",        {**_BASE, "use_d1_regime": True}),

    # ── + Filtre A : barre haussière ──────────────────────────────────────────
    Variant("FR3", "D1+H4+A   RR=3",        {**_BASE, "use_d1_regime": True,
                                              "require_bullish_bar": True}),

    # ── + Filtre E : impulse forte ────────────────────────────────────────────
    Variant("FR4", "D1+H4+E   RR=3",        {**_BASE, "use_d1_regime": True,
                                              "min_impulse_atr_mult": 0.5}),

    # ── RR sweep (D1 base) ────────────────────────────────────────────────────
    Variant("FR5", "D1+H4     RR=2",        {**_BASE, "use_d1_regime": True,
                                              "risk_reward": 2.0}),
    Variant("FR6", "D1+H4     RR=5",        {**_BASE, "use_d1_regime": True,
                                              "risk_reward": 5.0}),
    Variant("FR7", "D1+H4     RR=8",        {**_BASE, "use_d1_regime": True,
                                              "risk_reward": 8.0}),

    # ── + Filtre H : S&D confluence ───────────────────────────────────────────
    Variant("FR8", "D1+H4+SD  RR=3",        {**_BASE, "use_d1_regime": True,
                                              "use_sd_confluence": True}),

    # ── + Filtre D : BOS M15 récent ───────────────────────────────────────────
    Variant("FR9", "D1+H4+BOS RR=3",        {**_BASE, "use_d1_regime": True,
                                              "use_m15_bos_filter": True,
                                              "m15_bos_lookback": 30,
                                              "m15_pivot_size": 5}),

    # ── Combo A+E+BOS (qualité max) ───────────────────────────────────────────
    Variant("FR10","D1+A+E+BOS RR=3",       {**_BASE, "use_d1_regime": True,
                                              "require_bullish_bar": True,
                                              "min_impulse_atr_mult": 0.5,
                                              "use_m15_bos_filter": True,
                                              "m15_bos_lookback": 30,
                                              "m15_pivot_size": 5}),
]

# ── Affichage ─────────────────────────────────────────────────────────────────

W = dict(code=5, label=22, wr=7, avgr=7, rpm=9, pct=8, dd=6, spm=9, tr=8)
_SEP = "─" * (sum(W.values()) + len(W) * 2 + 2)
_HDR = (
    f"  {'Var':<{W['code']}}  {'Description':<{W['label']}}"
    f"  {'WR%':>{W['wr']}}  {'avgR':>{W['avgr']}}"
    f"  {'R/mois':>{W['rpm']}}  {'%/mois':>{W['pct']}}  {'DD%':>{W['dd']}}"
    f"  {'sig/mois':>{W['spm']}}  {'TotalR':>{W['tr']}}"
)


def _row(code: str, label: str, m: dict, n_months: float) -> str:
    wr   = m["win_rate"]
    avgr = m["total_r"] / m["n_trades"] if m["n_trades"] else 0.0
    rpm  = m["total_r"] / n_months
    pct  = rpm * RISK_PCT * 100
    dd   = m["max_dd"]
    spm  = m["n_signals"] / n_months
    tr   = m["total_r"]
    return (
        f"  {code:<{W['code']}}  {label:<{W['label']}}"
        f"  {wr:.1f}%{'':<{W['wr']-5}}"
        f"  {avgr:+.2f}R{'':<{W['avgr']-5}}"
        f"  {rpm:+.2f}R{'':<{W['rpm']-5}}"
        f"  {pct:+.2f}%{'':<{W['pct']-6}}"
        f"  {dd:.1f}%{'':<{W['dd']-4}}"
        f"  {spm:.1f}{'':<{W['spm']-3}}"
        f"  {tr:+.1f}R"
    )


def _monthly_breakdown(results, risk_pct: float, prefix: str = "      ") -> None:
    from zeus.backtest.sd_simulation import TradeResult
    by_month: dict[str, list] = {}
    for r in results:
        key = r.signal.formed_at.strftime("%Y-%m")
        by_month.setdefault(key, []).append(r)
    for month in sorted(by_month):
        rs    = by_month[month]
        wins  = sum(1 for r in rs if r.outcome == "win")
        total = sum(1 for r in rs if r.outcome in ("win", "loss"))
        wr    = wins / total * 100 if total else 0.0
        r_sum = sum(r.pnl_r for r in rs)
        pct   = r_sum * risk_pct * 100
        print(f"{prefix}{month}  WR={wr:5.1f}%  trades={len(rs):2d}  R={r_sum:+6.2f}  %={pct:+5.2f}%")


def _run_year(year: str, df: pd.DataFrame, n_months: float, detail_code: str = "FR2") -> list[dict]:
    tag = f"XAUUSD M15 · FVG Retest · {year} ({int(n_months)} mois) · risque {RISK_PCT*100:.0f}%"
    box = "═" * (len(tag) + 4)
    print(f"\n╔{box}╗")
    print(f"║  {tag}  ║")
    print(f"╚{box}╝")
    print(_HDR)
    print(_SEP)

    all_metrics = []
    for var in VARIANTS:
        strat   = FVGRetestStrategy(**var.kwargs)
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

        if not results:
            print(
                f"  {var.code:<{W['code']}}  {var.label:<{W['label']}}"
                f"  {'—':>{W['wr']}}  {'—':>{W['avgr']}}"
                f"  {'—':>{W['rpm']}}  {'—':>{W['pct']}}  {'—':>{W['dd']}}"
                f"  {len(signals)/n_months:.1f}{'':<{W['spm']-3}}"
                f"  {'0.0R':>{W['tr']}}"
            )
            all_metrics.append({})
            continue

        m = compute_metrics(results, INITIAL_EQ, len(signals), n_exp)
        print(_row(var.code, var.label, m, n_months))
        all_metrics.append(m)

        if var.code == detail_code:
            _monthly_breakdown(results, RISK_PCT)

    print(_SEP)
    print(f"  TP1={TP1_R}R partial ({int(TP1_SIZE*100)}%)  Spread={SPREAD} USD  Slip={SLIPPAGE} USD  Risque={RISK_PCT*100:.0f}%/trade")
    return all_metrics


# ── Walk-Forward ──────────────────────────────────────────────────────────────

def _run_wf(df_all: pd.DataFrame, wf_variant: Variant) -> None:
    """Walk-forward OOS sur le variant sélectionné."""
    tag = f"WALK-FORWARD OOS — FVGRetest · {wf_variant.code} ({wf_variant.label})"
    print(f"\n{'═'*70}")
    print(f"  {tag}")
    print(f"{'═'*70}")
    print(f"  {'Win':<4}  {'Période OOS':>22}  {'WR%':>7}  {'R/mois':>8}  {'%/mois':>8}  {'sig/mo':>7}")
    print("  " + "─" * 60)

    strat = FVGRetestStrategy(**wf_variant.kwargs)

    oos_wrs, oos_rpms = [], []
    for idx, (vs, ve) in enumerate(WF_WINDOWS, 1):
        test_df = df_all.loc[vs:ve]
        signals = strat.run(test_df)
        results, n_exp = simulate_all(
            signals, test_df,
            initial_equity = INITIAL_EQ,
            risk_pct       = RISK_PCT,
            spread         = SPREAD,
            tp1_r          = TP1_R,
            tp1_size       = TP1_SIZE,
            slippage_ticks = SLIPPAGE,
        )
        n_mo = float(test_df.resample("ME").last().dropna().shape[0]) or 3.0
        if not results:
            m = {}
        else:
            m = compute_metrics(results, INITIAL_EQ, len(signals), n_exp)

        if not m:
            print(f"  {idx:<4}  {vs}→{ve:>10}  —")
            continue

        rpm    = m["total_r"] / n_mo
        pct_mo = rpm * RISK_PCT * 100
        spm    = m["n_signals"] / n_mo
        oos_wrs.append(m["win_rate"])
        oos_rpms.append(rpm)
        print(
            f"  {idx:<4}  {vs}→{ve:>10}  "
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
        print(f"  {pct_ok} Rendement moyen OOS ≥ 5 % : {mean_pct:.2f}% à {RISK_PCT*100:.0f}%/trade")
        print()

        if mean_wr >= 70.0 and mean_pct >= 5.0:
            verdict = "OBJECTIFS ATTEINTS"
        elif mean_wr >= 70.0 and mean_pct >= 3.0:
            verdict = "OBJECTIFS APPROCHÉS (WR ✓, rendement légèrement inférieur)"
        else:
            verdict = "OBJECTIFS PARTIELS"

        print(f"  Verdict : {verdict}")

    print(f"{'═'*70}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    frames: dict[str, pd.DataFrame] = {}
    for year, path in DATASETS.items():
        if not path.exists():
            print(f"[SKIP] {path.name}")
            continue
        print(f"Chargement {path.name} …", end=" ", flush=True)
        df_m1 = parse_histdata_csv(path)
        df    = resample_ohlcv(df_m1, "15min")
        frames[year] = df
        print(f"OK  ({len(df):,} barres M15)")

    if not frames:
        print("[ERREUR] Aucun fichier trouvé.", file=sys.stderr)
        sys.exit(1)

    # ── IS 2025 et OOS 2024 ───────────────────────────────────────────────────
    for year, label in [("2025", "in-sample"), ("2024", "hors-échantillon")]:
        if year not in frames:
            continue
        df   = frames[year]
        n_mo = float(df.resample("ME").last().dropna().shape[0])
        all_m = _run_year(f"{year} ({label})", df, n_mo)

    # ── Walk-Forward OOS (6 fenêtres) ─────────────────────────────────────────
    if "2024" not in frames or "2025" not in frames:
        print("\n[INFO] Données manquantes — walk-forward ignoré.")
        return

    df_all = pd.concat([frames["2024"], frames["2025"]]).sort_index()

    # WF sur FR2 (D1+H4 RR=3) et FR7 (D1+H4 RR=8) pour comparer
    for code in ("FR2", "FR7"):
        var = next(v for v in VARIANTS if v.code == code)
        _run_wf(df_all, var)

    # ── Comparaison directe C3 vs FVGRetest ──────────────────────────────────
    print(f"\n{'═'*70}")
    print("  COMPARAISON  C3+RR8 vs FVGRetest")
    print(f"{'═'*70}")
    print(f"  {'Stratégie':<25}  {'WR%':>7}  {'R/mois':>8}  {'%/mois':>8}  {'sig/mo':>7}")
    print("  " + "─" * 62)
    print("  C3 final (OOS WF moyen)   :  78.3%  +1.62R  +4.86%  ~1.7")
    print("  FVGRetest FR2 (voir WF ci-dessus)")
    print("  FVGRetest FR7 (voir WF ci-dessus)")
    print(f"{'═'*70}")
    print()
    print("  [RAPPEL] 18 mois OOS = échantillon limité.")
    print("  [RAPPEL] Paper trading obligatoire avant tout déploiement live.")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
