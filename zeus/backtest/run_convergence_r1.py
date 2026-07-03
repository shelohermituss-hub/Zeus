"""
Convergence Strategy — Round 1 Backtest
========================================

Tests the OB + FVG stack + LTF Sweep + regime-filtered strategy across
multiple XAUUSD quarterly periods to establish a baseline WR profile.

Dataset: XAUUSD M1 2025 (full year) resampled to M15.

Variants tested
---------------
  V1  — baseline   : all 6 filters ON, TP1=0.6R, RR=2.0
  V2  — no_d1      : D1 EMA(200) filter OFF (measures its contribution)
  V3  — no_h4      : H4 EMA(50) slope filter OFF
  V4  — no_kz      : killzone filter OFF
  V5  — no_sweep   : LTF sweep filter OFF
  V6  — no_tp1     : TP1 disabled (straight to RR=2.0, no partial exit)
  V7  — rr15       : RR=1.5 (shorter TP, higher WR expectation)
  V8  — pivot5     : smaller pivot (5 bars) for more OBs, more signals

Metrics reported per variant per quarter:
  WR%, n_trades, R/month, total_R, max_DD%, avg_score

IMPORTANT: Results carry optimisation bias — same data used for design
           and testing. Walk-forward validation is required before
           trusting these numbers for live deployment.

Usage
-----
    python -m zeus.backtest.run_convergence_r1
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv, resample_ohlcv
from zeus.backtest.sd_simulation import (
    TradeResult,
    compute_metrics,
    simulate_all,
)
from zeus.strategy.convergence_strategy import ConvergenceStrategy

_ROOT = Path(__file__).parent.parent.parent

# ── Data ──────────────────────────────────────────────────────────────────────

M1_FILE = _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_2025.csv"

SPREAD          = 0.30   # USD/oz — typical XAUUSD CFD active-session spread
SLIPPAGE_TICKS  = 0.10   # F-09: pessimistic stop-order fill slippage
INITIAL_EQUITY  = 10_000.0
RISK_PCT        = 0.01   # 1% per trade

# Quarters for cross-period stability check
QUARTERS: list[tuple[str, str, str]] = [
    ("Q1-2025", "2025-01-01", "2025-04-01"),
    ("Q2-2025", "2025-04-01", "2025-07-01"),
    ("Q3-2025", "2025-07-01", "2025-10-01"),
    ("Q4-2025", "2025-10-01", "2026-01-01"),
]

# ── Variant configs ────────────────────────────────────────────────────────────

@dataclass
class VariantConfig:
    name:        str
    label:       str
    kwargs:      dict   # ConvergenceStrategy kwargs overrides


BASE = dict(
    pivot_size          = 10,
    long_only           = True,
    use_d1_ema          = True,
    d1_ema_span         = 200,
    use_h4_trend        = True,
    h4_ema_span         = 50,
    h4_slope_lb         = 3,
    use_killzone        = True,
    use_ltf_sweep       = True,
    ltf_sweep_lookback  = 5,
    require_bullish_bar = True,
    min_ob_age          = 2,
    max_ob_age          = 200,
    sl_buffer_atr       = 0.25,
    atr_period          = 14,
    risk_reward         = 2.0,
    signal_cooldown     = 12,
)

VARIANTS: list[VariantConfig] = [
    VariantConfig("V1",  "Baseline (all filters)",         {**BASE}),
    VariantConfig("V2",  "No D1 EMA(200)",                 {**BASE, "use_d1_ema": False}),
    VariantConfig("V3",  "No H4 EMA slope",                {**BASE, "use_h4_trend": False}),
    VariantConfig("V4",  "No killzone",                    {**BASE, "use_killzone": False}),
    VariantConfig("V5",  "No LTF sweep",                   {**BASE, "use_ltf_sweep": False}),
    VariantConfig("V6",  "No TP1 (straight RR=2.0)",       {**BASE}),   # tp1_r=0 in simulate_all
    VariantConfig("V7",  "RR=1.5",                         {**BASE, "risk_reward": 1.5}),
    VariantConfig("V8",  "Pivot size=5 (more OBs)",        {**BASE, "pivot_size": 5}),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt(m: dict) -> str:
    wr  = f"{m['win_rate']:.1f}%"
    n   = m["n_trades"]
    r   = f"{m['total_r']:+.1f}R"
    rpm = f"{m.get('r_per_month', 0.0):+.1f}R/mo"
    dd  = f"{m['max_dd']:.1f}%"
    return f"WR={wr:>6}  n={n:>3}  {r:>7}  {rpm:>9}  DD={dd:>5}"


def _r_per_month(results: list[TradeResult], period_start: str, period_end: str) -> float:
    """Estimate R/month over the backtest period."""
    total_r = sum(r.pnl_r for r in results)
    start   = pd.Timestamp(period_start)
    end     = pd.Timestamp(period_end)
    months  = max((end - start).days / 30.44, 1e-3)
    return total_r / months


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not M1_FILE.exists():
        print(f"[ERROR] Data file not found: {M1_FILE}", file=sys.stderr)
        print("  Download XAUUSD M1 2025 from HistData.com and place it at the path above.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loading {M1_FILE.name} …", end=" ", flush=True)
    df_m1 = parse_histdata_csv(M1_FILE)
    df_m15 = resample_ohlcv(df_m1, "15min")
    print(f"done  ({len(df_m15):,} M15 bars)\n")

    # ── Run all variants × all quarters ───────────────────────────────────────
    header = f"{'Variant':<30}  {'Period':<10}  Result"
    print(header)
    print("─" * len(header))

    summary: list[dict] = []

    for var in VARIANTS:
        for q_label, q_start, q_end in QUARTERS:
            m15_slice = df_m15.loc[q_start:q_end]
            if len(m15_slice) < 200:
                continue

            strategy = ConvergenceStrategy(**var.kwargs)
            signals  = strategy.run(m15_slice)

            tp1_r = 0.6 if var.name != "V6" else 0.0

            results, n_expired = simulate_all(
                signals,
                m15_slice,
                initial_equity = INITIAL_EQUITY,
                risk_pct       = RISK_PCT,
                spread         = SPREAD,
                tp1_r          = tp1_r,
                tp1_size       = 0.5,
                slippage_ticks = SLIPPAGE_TICKS,
            )

            m = compute_metrics(results, INITIAL_EQUITY, len(signals), n_expired)
            m["r_per_month"] = _r_per_month(results, q_start, q_end)

            row = dict(variant=var.name, label=var.label, quarter=q_label, **m)
            summary.append(row)

            print(f"  {var.name:<4} {var.label:<24}  {q_label:<10}  {_fmt(m)}")

    # ── Aggregate: mean WR per variant ────────────────────────────────────────
    print("\n── Summary: mean across quarters ─────────────────────────────────────")
    print(f"{'Variant':<6}  {'Label':<28}  {'WR%':>6}  {'R/mo':>8}  {'n/Q':>5}  {'MinWR':>7}")
    print("─" * 70)

    for var in VARIANTS:
        rows = [r for r in summary if r["variant"] == var.name]
        if not rows:
            continue
        wrs    = [r["win_rate"]     for r in rows if r["n_trades"] >= 3]
        rpms   = [r["r_per_month"]  for r in rows if r["n_trades"] >= 3]
        ns     = [r["n_trades"]     for r in rows]
        avg_wr = sum(wrs)  / len(wrs)  if wrs  else 0.0
        avg_rm = sum(rpms) / len(rpms) if rpms else 0.0
        avg_n  = sum(ns)   / len(ns)   if ns   else 0
        min_wr = min(wrs)  if wrs  else 0.0
        print(
            f"  {var.name:<4}  {var.label:<28}"
            f"  {avg_wr:>5.1f}%  {avg_rm:>+7.1f}R  {avg_n:>5.1f}  {min_wr:>6.1f}%"
        )

    print("\n[WARN] These results carry in-sample optimisation bias.")
    print("       Run SDWalkForwardOptimizer before trusting for live deployment.")


if __name__ == "__main__":
    main()
