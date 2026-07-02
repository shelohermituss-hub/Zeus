"""
FVG Round 21 — Filtre I : régime D1.
======================================

Contexte
--------
Round 20 a identifié ABCDG-RR10 (R:R=1.0) comme la configuration la plus
stable cross-régimes (WR moyen 57.4%, min 37.9% en Q1 2025 bear).

Problème résiduel : en marché range/baissier (Q2 2024, Q1 2025), la WR
tombe sous 40% et l'EV devient négatif. La cause : FVG long-only prend des
trades à contre-tendance quand la D1 structure est baissière.

Solution — Filtre I (régime D1)
--------------------------------
Ne prendre aucun trade long si D1 close < D1 EMA(N).
Suspendre la stratégie en bear macro, la reprendre dès que D1 repasse au-dessus.

Objectif de Round 21
--------------------
Trouver le span D1 EMA optimal (50 / 100 / 150 / 200 / 200+SMA) qui :
  - Filtre Q1 2025 et Q2 2024 sans trop réduire le nombre de signaux
  - Maintient WR ≥ 55% dans les périodes haussières
  - Améliore l'EV global cross-régimes

Usage
-----
    python -m zeus.backtest.run_fvg_round21
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv, resample_ohlcv
from zeus.backtest.sd_simulation import compute_metrics, simulate_all
from zeus.strategy.fvg_retest_strategy import FVGRetestStrategy

_ROOT = Path(__file__).parent.parent.parent

INITIAL_BALANCE = 10_000.0
RISK_PCT        = 0.01
SPREAD_GOLD     = 0.30

M1_2024 = _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_2024.csv"
M1_2025 = _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_2025.csv"


@dataclass
class Cfg:
    label:            str
    risk_reward:      float = 1.0    # RR10 champion R20
    use_d1_regime:    bool  = False
    d1_ema_span:      int   = 200


# ── Variantes ────────────────────────────────────────────────────────────────

VARIANTS: list[Cfg] = [
    # ── Référence sans filtre D1 ─────────────────────────────────────────────
    Cfg("ABCDG-RR10 · sans régime D1      (référence)"),

    # ── Filtre D1 EMA — différents spans ─────────────────────────────────────
    Cfg("ABCDG-RR10 · D1 EMA50",   use_d1_regime=True, d1_ema_span=50),
    Cfg("ABCDG-RR10 · D1 EMA100",  use_d1_regime=True, d1_ema_span=100),
    Cfg("ABCDG-RR10 · D1 EMA150",  use_d1_regime=True, d1_ema_span=150),
    Cfg("ABCDG-RR10 · D1 EMA200",  use_d1_regime=True, d1_ema_span=200),

    # ── Même filtre D1 + R:R=1.5 (pour comparaison avec R19 champion) ────────
    Cfg("ABCDG-RR15 · D1 EMA50",   risk_reward=1.5, use_d1_regime=True, d1_ema_span=50),
    Cfg("ABCDG-RR15 · D1 EMA100",  risk_reward=1.5, use_d1_regime=True, d1_ema_span=100),
    Cfg("ABCDG-RR15 · D1 EMA200",  risk_reward=1.5, use_d1_regime=True, d1_ema_span=200),

    # ── Référence R:R=1.5 sans D1 (R19 champion) ─────────────────────────────
    Cfg("ABCDG-RR15 · sans régime D1      (R19 ref)", risk_reward=1.5),
]


def _build(cfg: Cfg) -> FVGRetestStrategy:
    return FVGRetestStrategy(
        risk_reward             = cfg.risk_reward,
        signal_cooldown         = 12,
        max_signals_per_day     = 3,
        use_session_filter      = True,
        session_start_utc       = 7,
        session_end_utc         = 21,
        long_only               = True,
        use_h4_trend            = True,
        require_bullish_bar     = True,    # A
        fvg_max_penetration_pct = 0.50,   # B
        use_h1_trend            = True,   # C
        h1_ema_span             = 21,
        use_m15_bos_filter      = True,   # D
        m15_bos_lookback        = 30,
        use_d1_regime           = cfg.use_d1_regime,   # I
        d1_ema_span             = cfg.d1_ema_span,
    )


def _run(cfg: Cfg, m15_df: pd.DataFrame) -> tuple[list, dict[str, Any], int]:
    strat   = _build(cfg)
    signals = strat.run(m15_df)
    if not signals:
        return [], {"win_rate": 0, "total_r": 0, "max_dd": 0,
                    "n_wins": 0, "n_losses": 0}, 0
    results, n_exp = simulate_all(
        signals, m15_df, risk_pct=RISK_PCT, spread=SPREAD_GOLD, max_monthly_losses=0,
    )
    m = compute_metrics(results, INITIAL_BALANCE, len(signals), n_exp)
    return results, m, len(signals)


def _line(label: str, m: dict, n_sig: int, n_months: float) -> str:
    spm = n_sig / n_months if n_months else 0
    ev  = m["total_r"] / n_sig if n_sig else 0
    flag = ""
    if n_sig >= 3:
        if m["win_rate"] >= 70:
            flag = " 🎯"
        elif m["win_rate"] >= 60:
            flag = " ✓"
    return (
        f"  {label:<48}  "
        f"sig={n_sig:>3} ({spm:>4.1f}/m)  "
        f"W={m['n_wins']:>3} L={m['n_losses']:>3}  "
        f"WR={m['win_rate']:>5.1f}%  "
        f"R={m['total_r']:>+7.2f}  "
        f"EV={ev:>+6.3f}R  "
        f"DD={m['max_dd']:>5.1f}%{flag}"
    )


def _run_window(
    label:    str,
    m15_df:   pd.DataFrame,
    n_months: float,
) -> dict[str, tuple[dict, int]]:
    print(f"\n{'═' * 115}")
    print(f"  {label}  ({len(m15_df):,} barres M15)")
    print(f"{'═' * 115}")

    rows: list[tuple[Cfg, dict, int]] = []
    for cfg in VARIANTS:
        print(f"  [{cfg.label}] …", end="", flush=True)
        _, m, n_sig = _run(cfg, m15_df)
        rows.append((cfg, m, n_sig))
        print(f" {n_sig} sig  WR={m['win_rate']:.1f}%")

    print(f"\n{'─' * 115}")
    print("  RÉSULTATS DÉTAILLÉS")
    print(f"{'─' * 115}")
    for cfg, m, n_sig in rows:
        print(_line(cfg.label, m, n_sig, n_months))

    target_70 = [(cfg, m, n) for cfg, m, n in rows if m["win_rate"] >= 70 and n >= 3]
    target_60 = [(cfg, m, n) for cfg, m, n in rows if m["win_rate"] >= 60 and n >= 3]

    if target_70:
        print(f"\n  🎯 ≥70% WR (≥3 sig) :")
        for c, m, n in target_70:
            ev = m["total_r"] / n if n else 0
            print(f"     {c.label} → WR={m['win_rate']:.1f}% "
                  f"R={m['total_r']:+.2f} EV={ev:+.3f}R sig={n} DD={m['max_dd']:.1f}%")
    elif target_60:
        print(f"\n  ✓ ≥60% WR (≥3 sig) :")
        for c, m, n in target_60:
            ev = m["total_r"] / n if n else 0
            print(f"     {c.label} → WR={m['win_rate']:.1f}% "
                  f"R={m['total_r']:+.2f} EV={ev:+.3f}R sig={n} DD={m['max_dd']:.1f}%")
    else:
        valid = [(c, m, n) for c, m, n in rows if n >= 3]
        if valid:
            best = max(valid, key=lambda x: x[1]["win_rate"])
            ev = best[1]["total_r"] / best[2] if best[2] else 0
            print(f"\n  Meilleur (≥3 sig) : {best[0].label} "
                  f"WR={best[1]['win_rate']:.1f}% sig={best[2]} EV={ev:+.3f}R")

    return {cfg.label: (m, n_sig) for cfg, m, n_sig in rows}


def _print_cross_table(windows: list[tuple[str, dict[str, tuple[dict, int]]]]) -> None:
    col_w = 11
    print(f"\n{'═' * 140}")
    print("  TABLEAU CROISÉ — filtre D1 × 6 trimestres")
    print(f"  {'Variante':<48}", end="")
    for lbl, _ in windows:
        short = lbl.replace("Q1 2024", "Q1'24").replace("Q2 2024", "Q2'24") \
                   .replace("Q3 2024", "Q3'24").replace("Q4 2024", "Q4'24") \
                   .replace("Q1 2025", "Q1'25").replace("Q4 2025", "Q4'25")
        print(f"  {short[:8]:>{col_w-2}}", end="")
    print("   avg    min")
    print(f"  {'':48}", end="")
    for _ in windows:
        print(f"  {'WR%':>{col_w-2}}", end="")
    print()
    print(f"  {'─' * 130}")

    for cfg in VARIANTS:
        print(f"  {cfg.label:<48}", end="")
        wrs: list[float] = []
        for _, res in windows:
            m, n = res.get(cfg.label, ({}, 0))
            if not m or n < 3:
                print(f"  {'—':>{col_w-2}}", end="")
            else:
                wr = m.get("win_rate", 0.0)
                print(f"  {wr:>{col_w-2}.1f}%", end="")
                wrs.append(wr)
        if wrs:
            avg = sum(wrs) / len(wrs)
            mn  = min(wrs)
            print(f"   {avg:.1f}%  {mn:.1f}%")
        else:
            print()


def main() -> None:
    print("=" * 115)
    print("  FVG Round 21 — Filtre I : régime D1  |  XAUUSD 2024 + 2025")
    print("  Objectif : WR ≥ 55% cross-régimes en suspendant le trading en bear D1")
    print("=" * 115)

    m1_2024 = parse_histdata_csv(M1_2024)
    m1_2024 = m1_2024[~m1_2024.index.duplicated(keep="first")]

    m1_2025 = parse_histdata_csv(M1_2025)
    m1_2025 = m1_2025[~m1_2025.index.duplicated(keep="first")]

    slices = [
        ("Q1 2024 (jan-mar)", m1_2024, "2024-01-01", "2024-03-31"),
        ("Q2 2024 (avr-jun)", m1_2024, "2024-04-01", "2024-06-30"),
        ("Q3 2024 (jul-sep)", m1_2024, "2024-07-01", "2024-09-30"),
        ("Q4 2024 (oct-déc)", m1_2024, "2024-10-01", "2024-12-31"),
        ("Q1 2025 (jan-mar)", m1_2025, "2025-01-01", "2025-03-31"),
        ("Q4 2025 (oct-déc)", m1_2025, "2025-10-01", "2025-12-31"),
    ]

    all_windows: list[tuple[str, dict]] = []

    for name, m1_full, start, end in slices:
        m1_sl  = m1_full.loc[start:end]
        m15_sl = resample_ohlcv(m1_sl, "15min")
        print(f"\n  {name} : {len(m1_sl):,} M1 → {len(m15_sl):,} M15")
        res = _run_window(f"XAUUSD {name}", m15_sl, 3.0)
        all_windows.append((name, res))

    _print_cross_table(all_windows)

    # Résumé EV cumulé cross-fenêtres
    print(f"\n{'═' * 115}")
    print("  EV CUMULÉ CROSS-RÉGIMES (somme des 6 trimestres)")
    print(f"{'─' * 115}")
    for cfg in VARIANTS:
        total_r   = 0.0
        total_sig = 0
        total_w   = 0
        total_l   = 0
        for _, res in all_windows:
            m, n = res.get(cfg.label, ({}, 0))
            if m and n >= 0:
                total_r   += m.get("total_r", 0)
                total_sig += n
                total_w   += m.get("n_wins", 0)
                total_l   += m.get("n_losses", 0)
        wr_total = 100 * total_w / (total_w + total_l) if (total_w + total_l) > 0 else 0
        ev_total = total_r / total_sig if total_sig > 0 else 0
        print(f"  {cfg.label:<48}  "
              f"sig={total_sig:>4}  W={total_w:>3} L={total_l:>3}  "
              f"WR={wr_total:>5.1f}%  "
              f"R={total_r:>+8.2f}  "
              f"EV={ev_total:>+6.3f}R")

    print(f"\n{'═' * 115}")
    print("  Fin Round 21")
    print(f"{'=' * 115}\n")


if __name__ == "__main__":
    main()
