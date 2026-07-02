"""
ICT Order Block Strategy — Round 1 Exploration.
================================================

Objectif
--------
Premier backtest de la stratégie ICT Order Block sur XAUUSD M15.
Baseline : combien d'OBs générés, quels types de structure (BOS vs CHoCH),
quel WR initial et EV par variante.

Paramètres testés
-----------------
  - pivot_size_swing   : 5 / 10 / 15  (confirmation swing externe)
  - structure_filter   : "both" / "bos_only" / "choch_only"
  - use_killzone       : True / False
  - risk_reward        : 1.0 / 1.5 / 2.0

Données : Q4 2024 + Q1 2025 + Q4 2025 (mêmes fenêtres que Harmonic R1)

Usage
-----
    python -m zeus.backtest.run_ict_round1
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv, resample_ohlcv
from zeus.backtest.sd_simulation import compute_metrics, simulate_all
from zeus.strategy.ict_ob_strategy import ICTObStrategy

_ROOT = Path(__file__).parent.parent.parent

INITIAL_BALANCE = 10_000.0
RISK_PCT        = 0.01
SPREAD_GOLD     = 0.30

M1_2024 = _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_2024.csv"
M1_2025 = _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_2025.csv"


@dataclass
class Cfg:
    label:               str
    pivot_size_swing:    int   = 10
    structure_filter:    str   = "both"
    use_killzone:        bool  = False
    use_h4_trend:        bool  = True
    risk_reward:         float = 2.0
    min_ob_age:          int   = 2
    max_ob_age:          int   = 200
    require_bullish_bar: bool  = True


VARIANTS: list[Cfg] = [
    # ── Baseline pivot10 ────────────────────────────────────────────────────────
    Cfg("BASE · pivot10 both RR2.0"),
    Cfg("BASE · pivot10 both RR1.5",              risk_reward=1.5),
    Cfg("BASE · pivot10 both RR1.0",              risk_reward=1.0),

    # ── Structure filter ─────────────────────────────────────────────────────────
    Cfg("BOS  · pivot10 bos RR2.0",               structure_filter="bos_only"),
    Cfg("BOS  · pivot10 bos RR1.5",               structure_filter="bos_only", risk_reward=1.5),
    Cfg("BOS  · pivot10 bos RR1.0",               structure_filter="bos_only", risk_reward=1.0),
    Cfg("CHOCH· pivot10 choch RR2.0",             structure_filter="choch_only"),
    Cfg("CHOCH· pivot10 choch RR1.5",             structure_filter="choch_only", risk_reward=1.5),
    Cfg("CHOCH· pivot10 choch RR1.0",             structure_filter="choch_only", risk_reward=1.0),

    # ── Pivot size ───────────────────────────────────────────────────────────────
    Cfg("PIVOT5  · pivot5 both RR2.0",            pivot_size_swing=5),
    Cfg("PIVOT5  · pivot5 both RR1.5",            pivot_size_swing=5, risk_reward=1.5),
    Cfg("PIVOT5  · pivot5 both RR1.0",            pivot_size_swing=5, risk_reward=1.0),
    Cfg("PIVOT15 · pivot15 both RR2.0",           pivot_size_swing=15),
    Cfg("PIVOT15 · pivot15 both RR1.5",           pivot_size_swing=15, risk_reward=1.5),
    Cfg("PIVOT15 · pivot15 both RR1.0",           pivot_size_swing=15, risk_reward=1.0),

    # ── Killzone filter ──────────────────────────────────────────────────────────
    Cfg("KZ · pivot10 both RR2.0 KZ",             use_killzone=True),
    Cfg("KZ · pivot10 both RR1.5 KZ",             use_killzone=True, risk_reward=1.5),
    Cfg("KZ · pivot10 both RR1.0 KZ",             use_killzone=True, risk_reward=1.0),
    Cfg("KZ · pivot10 choch RR1.5 KZ",            use_killzone=True, structure_filter="choch_only", risk_reward=1.5),
    Cfg("KZ · pivot5  choch RR1.5 KZ",            use_killzone=True, structure_filter="choch_only", risk_reward=1.5, pivot_size_swing=5),
]


def _build(cfg: Cfg) -> ICTObStrategy:
    return ICTObStrategy(
        pivot_size_swing    = cfg.pivot_size_swing,
        pivot_size_internal = 5,
        long_only           = True,
        use_h4_trend        = cfg.use_h4_trend,
        use_killzone        = cfg.use_killzone,
        structure_filter    = cfg.structure_filter,
        risk_reward         = cfg.risk_reward,
        sl_buffer_atr       = 0.20,
        min_ob_age          = cfg.min_ob_age,
        max_ob_age          = cfg.max_ob_age,
        signal_cooldown     = 12,
        require_bullish_bar = cfg.require_bullish_bar,
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
        f"  {label:<50}  "
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
    print(f"\n{'═' * 116}")
    print(f"  {label}  ({len(m15_df):,} barres M15)")
    print(f"{'═' * 116}")

    rows: list[tuple[Cfg, dict, int]] = []
    for cfg in VARIANTS:
        print(f"  [{cfg.label}] …", end="", flush=True)
        _, m, n_sig = _run(cfg, m15_df)
        rows.append((cfg, m, n_sig))
        print(f" {n_sig} sig  WR={m['win_rate']:.1f}%")

    print(f"\n{'─' * 116}")
    print("  RÉSULTATS DÉTAILLÉS")
    print(f"{'─' * 116}")
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
    print(f"\n{'═' * 124}")
    print("  TABLEAU CROISÉ — ICT OB R1 × 3 fenêtres")
    print(f"  {'Variante':<50}", end="")
    for lbl, _ in windows:
        short = lbl[:8]
        print(f"  {short:>{col_w-2}}", end="")
    print("   avg    min")
    print(f"  {'':50}", end="")
    for _ in windows:
        print(f"  {'WR%':>{col_w-2}}", end="")
    print()
    print(f"  {'─' * 116}")

    for cfg in VARIANTS:
        print(f"  {cfg.label:<50}", end="")
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
    print("=" * 116)
    print("  ICT Order Block Strategy — Round 1 Exploration  |  XAUUSD M15")
    print("  Structure : BOS / CHoCH  |  Order Block retest  |  Long-only")
    print("=" * 116)

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

    print(f"\n{'═' * 116}")
    print("  Fin ICT OB Round 1")
    print(f"{'=' * 116}\n")


if __name__ == "__main__":
    main()
