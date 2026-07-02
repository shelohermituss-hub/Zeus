"""
ICT Order Block Strategy — Multi-Market Robustness Test.
=========================================================

Objectif
--------
Valider la robustesse de la stratégie ICT OB sur GBPUSD et USDCHF 2025.
Config gagnante : BOS · pivot10 · RR1.0 (testée sur XAUUSD → avg WR=56.5%, min=50%)

Marchés : XAUUSD · GBPUSD · USDCHF  |  Q1 2025 + Q4 2025

Usage
-----
    python -m zeus.backtest.run_ict_multimarket
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv, resample_ohlcv
from zeus.backtest.sd_simulation import compute_metrics, simulate_all
from zeus.strategy.ict_ob_strategy import ICTObStrategy

_ROOT = Path(__file__).parent.parent.parent

INITIAL_BALANCE = 10_000.0
RISK_PCT        = 0.01

SPREAD = {
    "XAUUSD": 0.30,
    "GBPUSD": 0.00020,
    "USDCHF": 0.00020,
}

_DATA: dict[str, Path] = {
    "XAUUSD": _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_2025.csv",
    "GBPUSD": _ROOT / "data" / "historical" / "gbpusd" / "m1" / "DAT_MT_GBPUSD_M1_2025.csv",
    "USDCHF": _ROOT / "data" / "historical" / "usdchf" / "m1" / "DAT_MT_USDCHF_M1_2025.csv",
}


@dataclass
class Cfg:
    label:            str
    pivot_size_swing: int   = 10
    structure_filter: str   = "bos_only"
    use_h4_trend:     bool  = False
    risk_reward:      float = 1.0


VARIANTS: list[Cfg] = [
    # Core BOS configs
    Cfg("BOS pivot10 RR1.0",   pivot_size_swing=10, risk_reward=1.0),
    Cfg("BOS pivot10 RR1.5",   pivot_size_swing=10, risk_reward=1.5),
    Cfg("BOS pivot10 RR2.0",   pivot_size_swing=10, risk_reward=2.0),
    Cfg("BOS pivot5  RR1.0",   pivot_size_swing=5,  risk_reward=1.0),
    Cfg("BOS pivot5  RR1.5",   pivot_size_swing=5,  risk_reward=1.5),
    Cfg("BOS pivot15 RR1.0",   pivot_size_swing=15, risk_reward=1.0),
    Cfg("BOS pivot15 RR1.5",   pivot_size_swing=15, risk_reward=1.5),
    # Both structure
    Cfg("BOTH pivot10 RR1.0",  structure_filter="both",       risk_reward=1.0),
    Cfg("BOTH pivot10 RR1.5",  structure_filter="both",       risk_reward=1.5),
    Cfg("CHOCH pivot10 RR1.0", structure_filter="choch_only", risk_reward=1.0),
]


def _build(cfg: Cfg) -> ICTObStrategy:
    return ICTObStrategy(
        pivot_size_swing    = cfg.pivot_size_swing,
        long_only           = True,
        use_h4_trend        = cfg.use_h4_trend,
        use_killzone        = False,
        structure_filter    = cfg.structure_filter,
        risk_reward         = cfg.risk_reward,
        sl_buffer_atr       = 0.20,
        min_ob_age          = 2,
        max_ob_age          = 200,
        signal_cooldown     = 12,
        require_bullish_bar = True,
    )


def _run(cfg: Cfg, m15_df: pd.DataFrame, spread: float) -> tuple[list, dict[str, Any], int]:
    strat   = _build(cfg)
    signals = strat.run(m15_df)
    if not signals:
        return [], {"win_rate": 0, "total_r": 0, "max_dd": 0,
                    "n_wins": 0, "n_losses": 0}, 0
    results, n_exp = simulate_all(
        signals, m15_df, risk_pct=RISK_PCT, spread=spread, max_monthly_losses=0,
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
) -> dict[str, tuple[dict, int]]:
    spread = SPREAD[symbol]
    print(f"\n  {'─' * 80}")
    print(f"  {label}  ({len(m15_df):,} M15)  spread={spread}")
    print(f"  {'─' * 80}")
    rows: list[tuple[Cfg, dict, int]] = []
    for cfg in VARIANTS:
        _, m, n_sig = _run(cfg, m15_df, spread)
        rows.append((cfg, m, n_sig))
        print(_line(cfg.label, m, n_sig, n_months))
    return {cfg.label: (m, n_sig) for cfg, m, n_sig in rows}


def _print_cross_table(all_results: dict[str, list[tuple[str, dict]]]) -> None:
    headers = []
    for sym in ["XAUUSD", "GBPUSD", "USDCHF"]:
        if sym not in all_results:
            continue
        for wlabel, _ in all_results[sym]:
            short = f"{sym[:3]}-{wlabel[:2]}"
            headers.append((sym, wlabel, short))

    col_w = 9
    print(f"\n{'═' * 110}")
    print("  TABLEAU CROISÉ — ICT OB Multi-Market × Windows")
    print(f"  {'Variante':<28}", end="")
    for _, _, h in headers:
        print(f"  {h:>{col_w-2}}", end="")
    print("   avg    min")
    print(f"  {'─' * 102}")

    for cfg in VARIANTS:
        print(f"  {cfg.label:<28}", end="")
        wrs: list[float] = []
        for sym, wlabel, _ in headers:
            if sym not in all_results:
                print(f"  {'—':>{col_w-2}}", end="")
                continue
            window_dict = next((d for lbl, d in all_results[sym] if lbl == wlabel), None)
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
    print("  ICT Order Block Strategy — Multi-Market Robustness  |  M15  |  2025")
    print("  Markets : XAUUSD · GBPUSD · USDCHF  |  Long-only  |  No H4 filter")
    print("=" * 100)

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

    for sym in ["XAUUSD", "GBPUSD", "USDCHF"]:
        if sym not in m1_data:
            continue
        m1_full = m1_data[sym]
        all_results[sym] = []

        print(f"\n{'═' * 100}")
        print(f"  {sym}")
        print(f"{'═' * 100}")

        for wname, start, end, n_months in windows_def:
            m1_sl = m1_full.loc[start:end]
            if len(m1_sl) == 0:
                print(f"  {wname}: no data")
                continue
            m15_sl = resample_ohlcv(m1_sl, "15min")
            res    = _run_window(f"{sym} {wname}", sym, m15_sl, n_months)
            all_results[sym].append((wname, res))

    _print_cross_table(all_results)

    print(f"\n{'═' * 100}")
    print("  Fin ICT OB Multi-Market")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    main()
