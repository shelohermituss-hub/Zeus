"""
Harmonic Pattern Strategy — Round 1 Exploration.
=================================================

Objectif
--------
Premier backtest de la stratégie de patterns harmoniques (Bat, Gartley,
Butterfly, Crab) sur XAUUSD M15. On cherche à établir une baseline :
combien de patterns détectés, quels types, quel WR initial.

Paramètres testés
-----------------
  - pivot_size      : 5 / 7 / 10   (sensibilité des swings)
  - precision       : 0.03 / 0.05  (tolérance ratio AB)
  - risk_reward     : 1.0 / 1.5 / 2.0

Données : Q1 2025 (exploration) + Q4 2025 (validation out-of-sample)
         + Q4 2024 (validation supplémentaire)

Usage
-----
    python -m zeus.backtest.run_harmonic_round1
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv, resample_ohlcv
from zeus.backtest.sd_simulation import compute_metrics, simulate_all
from zeus.strategy.harmonic_strategy import HarmonicStrategy
from zeus.strategy.smc.harmonic import detect_harmonics

_ROOT = Path(__file__).parent.parent.parent

INITIAL_BALANCE = 10_000.0
RISK_PCT        = 0.01
SPREAD_GOLD     = 0.30

M1_2024 = _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_2024.csv"
M1_2025 = _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_2025.csv"


@dataclass
class Cfg:
    label:        str
    pivot_size:   int   = 5
    precision:    float = 0.03
    risk_reward:  float = 2.0
    use_h4_trend: bool  = True


VARIANTS: list[Cfg] = [
    # ── Baseline ────────────────────────────────────────────────────────────
    Cfg("BASE · pivot5 prec3% RR2.0"),
    Cfg("BASE · pivot5 prec3% RR1.5",  risk_reward=1.5),
    Cfg("BASE · pivot5 prec3% RR1.0",  risk_reward=1.0),

    # ── Pivot size ───────────────────────────────────────────────────────────
    Cfg("PIVOT7 · pivot7 prec3% RR2.0",   pivot_size=7),
    Cfg("PIVOT7 · pivot7 prec3% RR1.5",   pivot_size=7, risk_reward=1.5),
    Cfg("PIVOT7 · pivot7 prec3% RR1.0",   pivot_size=7, risk_reward=1.0),
    Cfg("PIVOT10 · pivot10 prec3% RR2.0", pivot_size=10),
    Cfg("PIVOT10 · pivot10 prec3% RR1.5", pivot_size=10, risk_reward=1.5),
    Cfg("PIVOT10 · pivot10 prec3% RR1.0", pivot_size=10, risk_reward=1.0),

    # ── Precision ────────────────────────────────────────────────────────────
    Cfg("PREC5 · pivot5 prec5% RR2.0",   precision=0.05),
    Cfg("PREC5 · pivot5 prec5% RR1.5",   precision=0.05, risk_reward=1.5),
    Cfg("PREC5 · pivot5 prec5% RR1.0",   precision=0.05, risk_reward=1.0),

    # ── Sans filtre H4 ───────────────────────────────────────────────────────
    Cfg("NOH4 · pivot7 prec3% RR2.0 noH4",  pivot_size=7, use_h4_trend=False),
    Cfg("NOH4 · pivot10 prec3% RR1.5 noH4", pivot_size=10, use_h4_trend=False, risk_reward=1.5),
]


def _build(cfg: Cfg) -> HarmonicStrategy:
    return HarmonicStrategy(
        pivot_size          = cfg.pivot_size,
        precision           = cfg.precision,
        long_only           = True,
        use_h4_trend        = cfg.use_h4_trend,
        risk_reward         = cfg.risk_reward,
        sl_buffer_atr       = 0.20,
        max_prz_age_bars    = 200,
        signal_cooldown     = 12,
        require_bullish_bar = True,
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
        if m["win_rate"] >= 60:
            flag = " 🎯"
        elif m["win_rate"] >= 50:
            flag = " ✓"
    return (
        f"  {label:<46}  "
        f"sig={n_sig:>3} ({spm:>4.1f}/m)  "
        f"W={m['n_wins']:>3} L={m['n_losses']:>3}  "
        f"WR={m['win_rate']:>5.1f}%  "
        f"R={m['total_r']:>+7.2f}  "
        f"EV={ev:>+6.3f}R  "
        f"DD={m['max_dd']:>5.1f}%{flag}"
    )


def _pattern_breakdown(m15_df: pd.DataFrame, pivot_size: int = 5, precision: float = 0.03) -> None:
    """Print detected pattern count by type."""
    pats = detect_harmonics(m15_df, pivot_size=pivot_size, precision=precision, long_only=True)
    counts: dict[str, int] = {}
    for p in pats:
        counts[p.pattern_type] = counts.get(p.pattern_type, 0) + 1
    total = sum(counts.values())
    parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"  → Patterns détectés (pivot={pivot_size} prec={precision:.0%}): "
          f"total={total}  [{parts}]")


def _run_window(
    label:    str,
    m15_df:   pd.DataFrame,
    n_months: float,
) -> dict[str, tuple[dict, int]]:
    print(f"\n{'═' * 112}")
    print(f"  {label}  ({len(m15_df):,} barres M15)")
    print(f"{'═' * 112}")

    # Pattern breakdown for reference pivot sizes
    for ps in [5, 7, 10]:
        _pattern_breakdown(m15_df, pivot_size=ps)

    print()
    rows: list[tuple[Cfg, dict, int]] = []
    for cfg in VARIANTS:
        print(f"  [{cfg.label}] …", end="", flush=True)
        _, m, n_sig = _run(cfg, m15_df)
        rows.append((cfg, m, n_sig))
        print(f" {n_sig} sig  WR={m['win_rate']:.1f}%")

    print(f"\n{'─' * 112}")
    print("  RÉSULTATS DÉTAILLÉS")
    print(f"{'─' * 112}")
    for cfg, m, n_sig in rows:
        print(_line(cfg.label, m, n_sig, n_months))

    target_60 = [(cfg, m, n) for cfg, m, n in rows if m["win_rate"] >= 60 and n >= 3]
    target_50 = [(cfg, m, n) for cfg, m, n in rows if m["win_rate"] >= 50 and n >= 3]

    if target_60:
        print(f"\n  🎯 ≥60% WR (≥3 sig) :")
        for c, m, n in target_60:
            ev = m["total_r"] / n if n else 0
            print(f"     {c.label} → WR={m['win_rate']:.1f}% "
                  f"R={m['total_r']:+.2f} EV={ev:+.3f}R sig={n} DD={m['max_dd']:.1f}%")
    elif target_50:
        print(f"\n  ✓ ≥50% WR (≥3 sig) :")
        for c, m, n in target_50:
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
        else:
            print("\n  Aucune variante avec ≥3 signaux.")

    return {cfg.label: (m, n_sig) for cfg, m, n_sig in rows}


def _print_cross_table(windows: list[tuple[str, dict]]) -> None:
    col_w = 11
    print(f"\n{'═' * 120}")
    print("  TABLEAU CROISÉ — Harmonic R1 × 3 fenêtres")
    print(f"  {'Variante':<46}", end="")
    for lbl, _ in windows:
        short = lbl[:8]
        print(f"  {short:>{col_w-2}}", end="")
    print("   avg    min")
    print(f"  {'':46}", end="")
    for _ in windows:
        print(f"  {'WR%':>{col_w-2}}", end="")
    print()
    print(f"  {'─' * 110}")

    for cfg in VARIANTS:
        print(f"  {cfg.label:<46}", end="")
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
    print("=" * 112)
    print("  Harmonic Pattern Strategy — Round 1 Exploration  |  XAUUSD M15")
    print("  Patterns : Bat · Gartley · Butterfly · Crab  |  Long-only")
    print("=" * 112)

    m1_2024 = parse_histdata_csv(M1_2024)
    m1_2024 = m1_2024[~m1_2024.index.duplicated(keep="first")]

    m1_2025 = parse_histdata_csv(M1_2025)
    m1_2025 = m1_2025[~m1_2025.index.duplicated(keep="first")]

    slices = [
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

    print(f"\n{'═' * 112}")
    print("  Fin Harmonic Round 1")
    print(f"{'=' * 112}\n")


if __name__ == "__main__":
    main()
