"""
S&D Strategy — Round 23: TP1 Partial Exit on V123-LONG-CAP1 Champion
======================================================================

Current champion V123-LONG-CAP1:
  trend_slope_lookback=3, min_wyckoff_score_short=99.0, max_daily_losses=1
  All other params: V96 (zone≥5.0, wy≥5.9 long, R:R=1.5, M15/M1, slope+EMA, session 07-21h)

Combined results: 66.2% WR · +52.50R · max DD 3.1% — PASS across all 4 periods.

Round 23 question: does a partial TP1 exit at 1.0R improve the champion?

Rationale:
  - Take 50% off at +1R (lock in partial profit)
  - Remaining 50% runs to +1.5R target
  - Effective pnl_r if TP1 but TP2 not reached: 50% × 1.0R = +0.50R
  - Effective pnl_r if both TP1 and TP2 reached: 50% × 1.0R + 50% × 1.5R = +1.25R
  - Full loss if SL hit before TP1: -1.0R (same as before)
  - This should reduce variance and DD at the cost of some total R

Variants tested:
  - Champion baseline (no TP1)
  - TP1 @0.75R (early exit, very conservative)
  - TP1 @1.0R  (the classic break-even variant)
  - TP1 @1.25R (late partial, closer to main TP)
  All with 50% exit at TP1, remainder to 1.5R.

Also tested: V123-LONG-CAP1 with R:R=2.0 + TP1@1.0R
  (Use partial exit to protect the longer TP2 journey)

Usage
-----
    python -m zeus.backtest.run_sd_round23
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv, resample_ohlcv
from zeus.backtest.sd_simulation import (
    TradeResult,
    compute_metrics,
    print_monthly_breakdown,
    simulate_all,
)
from zeus.strategy.supply_demand.sd_strategy import SDStrategy
from zeus.strategy.supply_demand.wyckoff import WyckoffDetector
from zeus.strategy.supply_demand.zone_detector import ZoneDetector

_ROOT = Path(__file__).parent.parent.parent
_M1   = _ROOT / "data" / "historical" / "xauusd" / "m1"

INITIAL_BALANCE = 10_000.0
RISK_PCT        = 0.01
SPREAD          = 0.30

_V96 = dict(
    risk_reward          = 1.5,
    min_zone_score       = 5.0,
    min_wyckoff_score    = 5.9,
    min_composite_score  = 5.0,
    signal_cooldown      = 10,
    use_trend_filter     = True,
    trend_slope_lookback = 3,      # V123 (was 6 for V96)
    use_price_above_ema  = True,
    use_session_filter   = True,
    session_start_utc    = 7,
    session_end_utc      = 21,
    max_signals_per_day  = 6,
    use_adx_filter       = False,
    use_h4_trend_filter  = False,
    use_rsi_filter       = False,
    min_wyckoff_score_short = 99.0,  # long-only
)

_WY = dict(
    lookback             = 200,
    max_accum_bars       = 20,
    accum_range_mult     = 6.0,
    mss_lookback         = 60,
    min_spring_sweep_pct = 0.05,
    min_mss_strength_pct = 0.03,
)

PERIODS = [
    ("2024 Full Year (IS-A)", [_M1 / "DAT_MT_XAUUSD_M1_2024.csv"]),
    ("2025 Full Year (IS-B)", [_M1 / "DAT_MT_XAUUSD_M1_2025.csv"]),
    ("2026 Jan–Mar  (OOS)  ", [
        _M1 / "DAT_MT_XAUUSD_M1_202601.csv",
        _M1 / "DAT_MT_XAUUSD_M1_202602.csv",
        _M1 / "DAT_MT_XAUUSD_M1_202603.csv",
    ]),
    ("2026 Apr–Jun  (IS)   ", [
        _M1 / "DAT_MT_XAUUSD_M1_202604.csv",
        _M1 / "DAT_MT_XAUUSD_M1_202605.csv",
        _M1 / "DAT_MT_XAUUSD_M1_202606.csv",
    ]),
]


def _build_strategy(**overrides) -> SDStrategy:
    p = {**_V96, **overrides}
    return SDStrategy(
        zone_detector    = ZoneDetector(),
        wyckoff_detector = WyckoffDetector(**_WY),
        **p,
    )


def _load(files: list[Path]) -> pd.DataFrame:
    frames = [parse_histdata_csv(f) for f in files]
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="first")]


def _run_variant(
    m1_df:       pd.DataFrame,
    tp1_r:       float = 0.0,
    tp1_size:    float = 0.5,
    risk_reward: float = 1.5,
) -> tuple[list[TradeResult], dict[str, Any], int]:
    zone_df = resample_ohlcv(m1_df, "15min")
    strat   = _build_strategy(risk_reward=risk_reward)
    signals = strat.run(zone_df, m1_df)
    res, n_exp = simulate_all(
        signals, m1_df,
        risk_pct           = RISK_PCT,
        spread             = SPREAD,
        max_monthly_losses = 4,
        max_daily_losses   = 1,
        tp1_r              = tp1_r,
        tp1_size           = tp1_size,
    )
    m = compute_metrics(res, INITIAL_BALANCE, len(signals), n_exp)
    return res, m, len(signals)


def _run_all_periods(
    tp1_r:       float = 0.0,
    tp1_size:    float = 0.5,
    risk_reward: float = 1.5,
) -> list[tuple[str, dict, int]]:
    rows: list[tuple[str, dict, int]] = []
    for period_label, files in PERIODS:
        m1 = _load(files)
        res, m, n_sig = _run_variant(m1, tp1_r=tp1_r, tp1_size=tp1_size,
                                     risk_reward=risk_reward)
        rows.append((period_label, m, n_sig))
    return rows


def _summary_table(title: str, rows: list[tuple[str, dict, int]]) -> tuple[float, float, float, float]:
    print(f"\n  {title}")
    print(f"  {'Period':<30}  {'Sig':>4}  {'W':>3} {'L':>3}  {'WR%':>5}  {'R':>7}  {'P&L$':>8}  {'DD%':>5}")
    print("  " + "─" * 74)
    total_w = total_l = total_sig = total_r = 0.0
    min_r = float("inf")
    max_dd = 0.0
    all_pos_r = True
    for lbl, m, n in rows:
        print(
            f"  {lbl:<30}  {n:>4}  "
            f"{m['n_wins']:>3} {m['n_losses']:>3}  "
            f"{m['win_rate']:>5.1f}%  "
            f"{m['total_r']:>+7.2f}  "
            f"{m['total_usd']:>+8.0f}$  "
            f"{m['max_dd']:>4.1f}%"
        )
        total_w   += m['n_wins']
        total_l   += m['n_losses']
        total_sig += n
        total_r   += m['total_r']
        if n > 0:
            min_r  = min(min_r, m['total_r'])
            max_dd = max(max_dd, m['max_dd'])
            if m['total_r'] <= 0:
                all_pos_r = False
    decided    = total_w + total_l
    overall_wr = total_w / decided * 100 if decided else 0.0
    if min_r == float("inf"):
        min_r = 0.0
    print("  " + "─" * 74)
    print(f"  {'TOTAL / COMBINED':<30}  {int(total_sig):>4}  "
          f"{int(total_w):>3} {int(total_l):>3}  {overall_wr:>5.1f}%  {total_r:>+7.2f}")
    verdict = "PASS ✓" if overall_wr >= 60 and all_pos_r else "FAIL ✗"
    print(f"  VERDICT: {verdict}  (WR ≥ 60% combined + R > 0 every period)")
    return total_r, min_r, overall_wr, max_dd


def main() -> None:
    print("=" * 100)
    print("  S&D Strategy — Round 23 · XAUUSD")
    print("  TP1 Partial Exit on V123-LONG-CAP1 champion")
    print("=" * 100)

    variants = [
        # (label, tp1_r, tp1_size, risk_reward)
        ("Champion (no TP1)          ", 0.0,  0.5, 1.5),
        ("TP1@0.75R 50% → rest 1.5R ", 0.75, 0.5, 1.5),
        ("TP1@1.0R  50% → rest 1.5R ", 1.0,  0.5, 1.5),
        ("TP1@1.25R 50% → rest 1.5R ", 1.25, 0.5, 1.5),
        ("TP1@1.0R  33% → rest 1.5R ", 1.0,  0.33, 1.5),
        ("TP1@1.0R  66% → rest 1.5R ", 1.0,  0.66, 1.5),
        # Extend TP2 with TP1 safety net
        ("R:R=2.0 + TP1@1.0R 50%    ", 1.0,  0.5, 2.0),
        ("R:R=2.5 + TP1@1.0R 50%    ", 1.0,  0.5, 2.5),
        ("R:R=2.0 + TP1@1.5R 50%    ", 1.5,  0.5, 2.0),
    ]

    all_results: list[tuple[str, float, float, float, float]] = []

    for label, tp1_r, tp1_size, rr in variants:
        print(f"\n  Running {label.strip()} …", end="", flush=True)
        rows = _run_all_periods(tp1_r=tp1_r, tp1_size=tp1_size, risk_reward=rr)
        tr, mr, wr, dd = _summary_table(label, rows)
        all_results.append((label, tr, mr, wr, dd))
        print()

    # ── Summary ──
    print(f"\n{'=' * 100}")
    print("  ROUND 23 — Head-to-head comparison")
    print(f"{'─' * 100}")
    print(f"\n  {'Config':<35}  {'TotR':>7}  {'MinR':>7}  {'OvWR':>5}  {'MaxDD':>6}")
    print("  " + "─" * 65)
    for name, tr, mr, wr, dd in all_results:
        ok   = mr > 0 and wr >= 60
        flag = " ←" if ok else ""
        print(f"  {name:<35}  {tr:>+7.2f}  {mr:>+7.2f}  {wr:>5.1f}%  {dd:>5.1f}%{flag}")

    print(f"\n  ← PASS: WR ≥ 60% combined + R > 0 every period")

    passing = [(n, tr, mr, wr, dd) for n, tr, mr, wr, dd in all_results if mr > 0 and wr >= 60]
    if passing:
        best = max(passing, key=lambda x: x[1])
        print(f"\n  BEST PASSING CONFIG: {best[0].strip()}")
        print(f"    TotR={best[1]:+.2f}  MinR={best[2]:+.2f}  OvWR={best[3]:.1f}%  MaxDD={best[4]:.1f}%")
    else:
        best = max(all_results, key=lambda x: x[1])
        print(f"\n  No config passes. Best R: {best[0].strip()} → {best[1]:+.2f}")

    print(f"\n{'=' * 100}")
    print("  Fin du Round 23")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    main()
