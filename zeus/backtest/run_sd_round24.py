"""
S&D Strategy — Round 24 · XAUUSD
4-Tier Exit Structure on V123-LONG-CAP1 Champion
=================================================

Starting from the Round 23 best config (R:R=2.5 + TP1@1.0R 50% → +55.75R),
this round tests a deeper 4-tier structure designed to capture gold's extended
runner potential:

  TP1 @ 1R  → nothing closed, SL → BE     (protect against reversal)
  TP2 @ 3R  → close 60 %                  (lock main profits)
  TP3 @ ?R  → close to 85 % cumul         (variants: 6R / 7R / 8R)
  Runner    → 15 % runs to full RR target  (variants: 20R / 25R)

Also tested: momentum confirmation at TP2 — when the bar closing above TP2
is strongly bullish (close in upper 40 % of bar range), reduce close to 40 %
instead of 60 %, leaving more position to ride TP3 and runner.

Pass criteria: WR ≥ 60 % combined AND R > 0 every period.

Usage
-----
    python -m zeus.backtest.run_sd_round24
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv, resample_ohlcv
from zeus.backtest.sd_simulation import (
    TradeResult,
    compute_metrics,
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
    risk_reward             = 1.5,      # overridden per variant via dataclasses.replace
    min_zone_score          = 5.0,
    min_wyckoff_score       = 5.9,
    min_composite_score     = 5.0,
    signal_cooldown         = 10,
    use_trend_filter        = True,
    trend_slope_lookback    = 3,
    use_price_above_ema     = True,
    use_session_filter      = True,
    session_start_utc       = 7,
    session_end_utc         = 21,
    max_signals_per_day     = 6,
    use_adx_filter          = False,
    use_h4_trend_filter     = False,
    use_rsi_filter          = False,
    min_wyckoff_score_short = 99.0,     # long-only
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

# ── Variant definitions ───────────────────────────────────────────────────────
#
# Each variant is a dict of kwargs forwarded to simulate_all **plus** a special
# key "runner_rr" that overrides signal.risk_reward (the final runner target).
#
# Structure of every 4-tier variant:
#   TP1 @ tp1_r  → tp1_size = 0.0  (BE only — nothing closed)
#   TP2 @ tp2_r  → cumulative 60 % closed
#   TP3 @ tp3_r  → cumulative 85 % closed  (25 % more at TP3)
#   Runner       → 15 % at runner_rr

_BASE_4T = dict(
    tp1_r              = 1.0,
    tp1_size           = 0.0,   # BE only — no partial close at TP1
    tp2_r              = 3.0,
    tp2_cumulative_pct = 0.60,
    tp3_cumulative_pct = 0.85,
)

VARIANTS: list[tuple[str, dict]] = [
    # ── Round 23 champion (reference) ────────────────────────────────────────
    ("R23-Champion (RR2.5 TP1@1R 50%)", dict(
        tp1_r    = 1.0,
        tp1_size = 0.5,
        runner_rr = 2.5,
    )),

    # ── 4-tier · TP3 sweep · runner @20R ─────────────────────────────────────
    ("4T · TP2@3R(60%) · TP3@6R(85%) · Run@20R", dict(
        **_BASE_4T,
        tp3_r     = 6.0,
        runner_rr = 20.0,
    )),
    ("4T · TP2@3R(60%) · TP3@7R(85%) · Run@20R", dict(
        **_BASE_4T,
        tp3_r     = 7.0,
        runner_rr = 20.0,
    )),
    ("4T · TP2@3R(60%) · TP3@8R(85%) · Run@20R", dict(
        **_BASE_4T,
        tp3_r     = 8.0,
        runner_rr = 20.0,
    )),

    # ── 4-tier · TP3@6R · runner @25R ────────────────────────────────────────
    ("4T · TP2@3R(60%) · TP3@6R(85%) · Run@25R", dict(
        **_BASE_4T,
        tp3_r     = 6.0,
        runner_rr = 25.0,
    )),

    # ── Momentum at TP2: close 40 % if bar is strongly bullish, else 60 % ───
    # TP3@6R, runner @20R
    ("4T · MOM-TP2(40/60%) · TP3@6R(85%) · Run@20R", dict(
        **_BASE_4T,
        tp3_r                = 6.0,
        runner_rr            = 20.0,
        momentum_tp2         = True,
        momentum_strong_pct  = 0.40,
    )),
    # TP3@7R, runner @20R
    ("4T · MOM-TP2(40/60%) · TP3@7R(85%) · Run@20R", dict(
        **_BASE_4T,
        tp3_r                = 7.0,
        runner_rr            = 20.0,
        momentum_tp2         = True,
        momentum_strong_pct  = 0.40,
    )),
]


def _build_strategy(runner_rr: float) -> SDStrategy:
    return SDStrategy(
        zone_detector    = ZoneDetector(),
        wyckoff_detector = WyckoffDetector(**_WY),
        risk_reward      = runner_rr,  # full RR drives the signal object; overridden below
        **{k: v for k, v in _V96.items() if k != "risk_reward"},
    )


def _load(files: list[Path]) -> pd.DataFrame:
    frames = [parse_histdata_csv(f) for f in files if f.exists()]
    if not frames:
        raise FileNotFoundError(f"No data files found for {files}")
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="first")]


def _run_variant(
    m1_df: pd.DataFrame,
    runner_rr: float,
    **sim_kwargs: Any,
) -> tuple[list[TradeResult], dict, int]:
    zone_df  = resample_ohlcv(m1_df, "15min")
    strategy = _build_strategy(runner_rr)
    signals  = strategy.run(zone_df, m1_df)

    # Override each signal's risk_reward to the runner target so the
    # simulation engine drives price towards the correct final TP level.
    signals = [dataclasses.replace(s, risk_reward=runner_rr) for s in signals]

    res, n_exp = simulate_all(
        signals, m1_df,
        risk_pct           = RISK_PCT,
        spread             = SPREAD,
        max_monthly_losses = 4,
        max_daily_losses   = 1,
        **sim_kwargs,
    )
    m = compute_metrics(res, INITIAL_BALANCE, len(signals), n_exp)
    return res, m, len(signals)


def _run_all_periods(runner_rr: float, **sim_kwargs: Any) -> list[tuple[str, dict, int]]:
    rows: list[tuple[str, dict, int]] = []
    for period_label, files in PERIODS:
        m1 = _load(files)
        _, m, n_sig = _run_variant(m1, runner_rr, **sim_kwargs)
        rows.append((period_label, m, n_sig))
    return rows


def _summary_table(title: str, rows: list[tuple[str, dict, int]]) -> tuple[float, float, float, float]:
    print(f"\n  {title}")
    print(f"  {'Period':<30}  {'Sig':>4}  {'W':>3} {'L':>3}  "
          f"{'WR%':>5}  {'R':>7}  {'P&L$':>8}  {'DD%':>5}")
    print("  " + "─" * 76)
    total_w = total_l = total_sig = total_r = 0.0
    min_r   = float("inf")
    max_dd  = 0.0
    all_pos = True
    for lbl, m, n in rows:
        print(
            f"  {lbl:<30}  {n:>4}  "
            f"{m['n_wins']:>3} {m['n_losses']:>3}  "
            f"{m['win_rate']:>5.1f}%  "
            f"{m['total_r']:>+7.2f}  "
            f"{m['total_usd']:>+8.0f}$  "
            f"{m['max_dd']:>4.1f}%"
        )
        total_w   += m["n_wins"]
        total_l   += m["n_losses"]
        total_sig += n
        total_r   += m["total_r"]
        if n > 0:
            min_r  = min(min_r, m["total_r"])
            max_dd = max(max_dd, m["max_dd"])
            if m["total_r"] <= 0:
                all_pos = False
    decided    = total_w + total_l
    overall_wr = total_w / decided * 100 if decided else 0.0
    if min_r == float("inf"):
        min_r = 0.0
    print("  " + "─" * 76)
    print(f"  {'TOTAL / COMBINED':<30}  {int(total_sig):>4}  "
          f"{int(total_w):>3} {int(total_l):>3}  {overall_wr:>5.1f}%  {total_r:>+7.2f}")
    verdict = "PASS ✓" if overall_wr >= 60 and all_pos else "FAIL ✗"
    print(f"  VERDICT: {verdict}  (WR ≥ 60 % + R > 0 every period)")
    return total_r, min_r, overall_wr, max_dd


def main() -> None:
    print("=" * 100)
    print("  S&D Strategy — Round 24 · XAUUSD")
    print("  4-Tier Exit Structure vs V123-LONG-CAP1 Champion")
    print("=" * 100)

    all_results: list[tuple[str, float, float, float, float]] = []

    for label, params in VARIANTS:
        kw        = dict(params)                   # copy — never mutate VARIANTS
        runner_rr = kw.pop("runner_rr", 2.5)

        print(f"\n  Running {label.strip()} …", end="", flush=True)
        rows = _run_all_periods(runner_rr=runner_rr, **kw)
        tr, mr, wr, dd = _summary_table(label, rows)
        all_results.append((label, tr, mr, wr, dd))
        print()

    # ── Head-to-head ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print("  ROUND 24 — Head-to-head comparison")
    print(f"{'─' * 100}")
    print(f"\n  {'Config':<50}  {'TotR':>7}  {'MinR':>7}  {'OvWR':>5}  {'MaxDD':>6}")
    print("  " + "─" * 78)
    for name, tr, mr, wr, dd in all_results:
        ok   = mr > 0 and wr >= 60
        flag = " ←" if ok else ""
        print(f"  {name:<50}  {tr:>+7.2f}  {mr:>+7.2f}  {wr:>5.1f}%  {dd:>5.1f}%{flag}")

    print(f"\n  ← PASS: WR ≥ 60 % combined + R > 0 every period")

    passing = [(n, tr, mr, wr, dd) for n, tr, mr, wr, dd in all_results if mr > 0 and wr >= 60]
    if passing:
        best = max(passing, key=lambda x: x[1])
        print(f"\n  BEST PASSING CONFIG: {best[0].strip()}")
        print(f"    TotR={best[1]:+.2f}  MinR={best[2]:+.2f}  OvWR={best[3]:.1f}%  MaxDD={best[4]:.1f}%")
    else:
        best = max(all_results, key=lambda x: x[1])
        print(f"\n  No config passes. Best R: {best[0].strip()} → {best[1]:+.2f}")

    print(f"\n{'=' * 100}")
    print("  Fin du Round 24")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    main()
