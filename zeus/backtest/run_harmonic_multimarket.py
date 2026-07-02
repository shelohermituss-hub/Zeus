"""
Harmonic Pattern Strategy — Multi-Market Robustness Test.
==========================================================

Objectif
--------
Valider la robustesse de la stratégie harmonique sur GBPUSD et USDCHF 2025.
Les mêmes paramètres gagnants de Round 1 (PIVOT7 RR1.0 / RR1.5) sont testés
sur de nouveaux marchés pour confirmer que l'edge n'est pas spécifique à XAUUSD.

Marchés testés
--------------
  - XAUUSD 2025 (référence, Q1 + Q4)
  - GBPUSD 2025 (Q1 + Q4)
  - USDCHF 2025 (Q1 + Q4)

Paramètres  : PIVOT7 / PIVOT5 × RR1.0 / RR1.5 / RR2.0 (sans filtre H4 trend)

Usage
-----
    python -m zeus.backtest.run_harmonic_multimarket
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv, resample_ohlcv
from zeus.backtest.sd_simulation import compute_metrics, simulate_all
from zeus.strategy.harmonic_strategy import HarmonicStrategy

_ROOT = Path(__file__).parent.parent.parent

INITIAL_BALANCE = 10_000.0
RISK_PCT        = 0.01

# Spread per unit (pip-equivalent bid/ask cost)
SPREAD = {
    "XAUUSD": 0.30,    # 30 cents / oz
    "GBPUSD": 0.00020, # 2 pips (0.0002)
    "USDCHF": 0.00020, # 2 pips
    "CADJPY": 0.008,   # 0.8 pip (0.008 JPY)
}

_DATA: dict[str, Path] = {
    "XAUUSD": _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_2025.csv",
    "GBPUSD": _ROOT / "data" / "historical" / "gbpusd" / "m1" / "DAT_MT_GBPUSD_M1_2025.csv",
    "USDCHF": _ROOT / "data" / "historical" / "usdchf" / "m1" / "DAT_MT_USDCHF_M1_2025.csv",
    "CADJPY": _ROOT / "data" / "historical" / "cadjpy" / "m1" / "DAT_MT_CADJPY_M1_2025.csv",
}


@dataclass
class Cfg:
    label:        str
    pivot_size:   int   = 7
    precision:    float = 0.03
    risk_reward:  float = 1.0
    use_h4_trend: bool  = False   # disabled for cross-market (H4 EMA less reliable)


VARIANTS: list[Cfg] = [
    Cfg("PIVOT5  RR1.0 noH4",  pivot_size=5,  risk_reward=1.0),
    Cfg("PIVOT5  RR1.5 noH4",  pivot_size=5,  risk_reward=1.5),
    Cfg("PIVOT5  RR2.0 noH4",  pivot_size=5,  risk_reward=2.0),
    Cfg("PIVOT7  RR1.0 noH4",  pivot_size=7,  risk_reward=1.0),
    Cfg("PIVOT7  RR1.5 noH4",  pivot_size=7,  risk_reward=1.5),
    Cfg("PIVOT7  RR2.0 noH4",  pivot_size=7,  risk_reward=2.0),
    Cfg("PIVOT10 RR1.0 noH4",  pivot_size=10, risk_reward=1.0),
    Cfg("PIVOT10 RR1.5 noH4",  pivot_size=10, risk_reward=1.5),
    Cfg("PIVOT10 RR2.0 noH4",  pivot_size=10, risk_reward=2.0),
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


def _run(
    cfg: Cfg,
    m15_df: pd.DataFrame,
    spread: float,
    m1_df: pd.DataFrame | None = None,
) -> tuple[list, dict[str, Any], int]:
    strat   = _build(cfg)
    signals = strat.run(m15_df)
    if not signals:
        return [], {"win_rate": 0, "total_r": 0, "max_dd": 0,
                    "n_wins": 0, "n_losses": 0}, 0
    results, n_exp = simulate_all(
        signals, m15_df, risk_pct=RISK_PCT, spread=spread, max_monthly_losses=0,
        initial_equity=INITIAL_BALANCE, tick_df=m1_df,
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
        f"  {label:<28}  "
        f"sig={n_sig:>3} ({spm:>4.1f}/m)  "
        f"W={m['n_wins']:>3} L={m['n_losses']:>3}  "
        f"WR={m['win_rate']:>5.1f}%  "
        f"R={m['total_r']:>+7.2f}  "
        f"EV={ev:>+6.3f}R  "
        f"DD={m['max_dd']:>5.1f}%{flag}"
    )


def _run_window(
    label:    str,
    symbol:   str,
    m15_df:   pd.DataFrame,
    n_months: float,
    m1_df:    pd.DataFrame | None = None,
) -> dict[str, tuple[dict, int]]:
    spread = SPREAD[symbol]
    print(f"\n  {'─' * 80}")
    print(f"  {label}  ({len(m15_df):,} M15 bars)  spread={spread}")
    print(f"  {'─' * 80}")
    rows: list[tuple[Cfg, dict, int]] = []
    for cfg in VARIANTS:
        _, m, n_sig = _run(cfg, m15_df, spread, m1_df=m1_df)
        rows.append((cfg, m, n_sig))

    for cfg, m, n_sig in rows:
        print(_line(cfg.label, m, n_sig, n_months))

    return {cfg.label: (m, n_sig) for cfg, m, n_sig in rows}


def _print_cross_table(all_results: dict[str, list[tuple[str, dict]]]) -> None:
    """Print WR cross-table: rows=variants, cols=market×window."""
    headers = []
    for sym, windows in all_results.items():
        for wlabel, _ in windows:
            short = f"{sym[:3]}-{wlabel[:2]}"
            headers.append((sym, wlabel, short))

    col_w = 9
    print(f"\n{'═' * 120}")
    print("  TABLEAU CROISÉ — Harmonic Multi-Market × Windows")
    print(f"  {'Variante':<28}", end="")
    for _, _, h in headers:
        print(f"  {h:>{col_w-2}}", end="")
    print("   avg    min")
    print(f"  {'─' * 112}")

    for cfg in VARIANTS:
        print(f"  {cfg.label:<28}", end="")
        wrs: list[float] = []
        for sym, wlabel, _ in headers:
            res = all_results[sym]
            window_dict = next((d for lbl, d in res if lbl == wlabel), None)
            if window_dict is None:
                print(f"  {'—':>{col_w-2}}", end="")
                continue
            m, n = window_dict.get(cfg.label, ({}, 0))
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
    print("=" * 100)
    print("  Harmonic Pattern Strategy — Multi-Market Robustness  |  M15  |  2025")
    print("  Markets : XAUUSD · GBPUSD · USDCHF · CADJPY  |  Long-only  |  No H4 filter")
    print("=" * 100)

    # Load and cache all data
    m1_data: dict[str, pd.DataFrame] = {}
    for sym, path in _DATA.items():
        if not path.exists():
            print(f"  ⚠ Missing data: {path}")
            continue
        df = parse_histdata_csv(path)
        df = df[~df.index.duplicated(keep="first")]
        m1_data[sym] = df
        print(f"  Loaded {sym}: {len(df):,} M1 bars")

    windows_def = [
        ("Q1 2025", "2025-01-01", "2025-03-31", 3.0),
        ("Q4 2025", "2025-10-01", "2025-12-31", 3.0),
    ]

    all_results: dict[str, list[tuple[str, dict]]] = {}

    for sym in ["XAUUSD", "GBPUSD", "USDCHF", "CADJPY"]:
        if sym not in m1_data:
            continue
        m1_full = m1_data[sym]
        all_results[sym] = []

        print(f"\n{'═' * 100}")
        print(f"  {sym}")
        print(f"{'═' * 100}")

        for wname, start, end, n_months in windows_def:
            m1_sl  = m1_full.loc[start:end]
            if len(m1_sl) == 0:
                print(f"  {wname}: no data")
                continue
            m15_sl = resample_ohlcv(m1_sl, "15min")
            label  = f"{wname} ({len(m15_sl):,} M15)"
            res    = _run_window(label, sym, m15_sl, n_months, m1_df=m1_sl)
            all_results[sym].append((wname, res))

    _print_cross_table(all_results)

    print(f"\n{'═' * 100}")
    print("  Fin Harmonic Multi-Market")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    main()
