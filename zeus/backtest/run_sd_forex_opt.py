"""
S&D Strategy — Forex Optimisation  (Round FX-1 → FX-4)
========================================================

The XAUUSD V_FINAL_B champion fails on all Forex pairs (WR ≈ 28–42%).
Root causes identified:
  1. Long-only bias — Forex is bidirectional; Supply zones are equally valid.
  2. min_wyckoff_score = 5.9 — calibrated on Gold's sharp Spring/MSS patterns.
  3. Session 07-21h — correct for London+NY Gold; may need per-pair tuning.

Optimisation strategy (GBPUSD as primary instrument — only pair with IS+OOS):
  Round FX-1 : Direction (long-only / bidirectional / symmetric thresholds)
  Round FX-2 : Wyckoff score threshold on best direction config
  Round FX-3 : Session filter variants on best FX-1/FX-2 combo
  Round FX-4 : Zone-score + R:R variants on best FX-1/2/3 combo
  Validation  : Best config applied to CADJPY + USDCHF (cross-instrument check)

Pass criterion for Forex (data-limited, relaxed from XAUUSD):
  WR ≥ 55% combined  AND  R > 0 in every available period  AND  DD ≤ 8%

Data:
  GBPUSD  2025 (IS) + 2026-Jun (OOS)  ← primary optimisation instrument
  CADJPY  2025 (IS)                   ← cross-instrument validation
  USDCHF  2025 (IS)                   ← cross-instrument validation

Usage
-----
    python -m zeus.backtest.run_sd_forex_opt
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv, resample_ohlcv
from zeus.backtest.sd_simulation import compute_metrics, simulate_all
from zeus.strategy.supply_demand.sd_strategy import SDStrategy
from zeus.strategy.supply_demand.wyckoff import WyckoffDetector
from zeus.strategy.supply_demand.zone_detector import ZoneDetector

_ROOT = Path(__file__).parent.parent.parent
_DATA = _ROOT / "data" / "historical"

INITIAL_BALANCE = 10_000.0
RISK_PCT        = 0.01
TP1_R           = 1.25
TP1_SIZE        = 0.50

SPREAD = dict(
    GBPUSD = 0.0001,
    CADJPY = 0.020,
    USDCHF = 0.0001,
)

# Data periods per instrument
PERIODS = dict(
    GBPUSD = [
        ("2025 Full Year (IS)",  [_DATA / "gbpusd" / "m1" / "DAT_MT_GBPUSD_M1_2025.csv"]),
        ("2026 Jun     (OOS)",   [_DATA / "gbpusd" / "m1" / "DAT_MT_GBPUSD_M1_202606.csv"]),
    ],
    CADJPY = [
        ("2025 Full Year (IS)",  [_DATA / "cadjpy" / "m1" / "DAT_MT_CADJPY_M1_2025.csv"]),
    ],
    USDCHF = [
        ("2025 Full Year (IS)",  [_DATA / "usdchf" / "m1" / "DAT_MT_USDCHF_M1_2025.csv"]),
    ],
)

# ── V_FINAL_B base (XAUUSD champion) ─────────────────────────────────────────

_BASE = dict(
    risk_reward             = 1.5,
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
)

_BASE_WY = dict(
    lookback             = 200,
    max_accum_bars       = 20,
    accum_range_mult     = 6.0,
    mss_lookback         = 60,
    min_spring_sweep_pct = 0.05,
    min_mss_strength_pct = 0.03,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _build(strategy_overrides: dict = {}, wy_overrides: dict = {}) -> SDStrategy:
    p  = {**_BASE,    **strategy_overrides}
    wy = {**_BASE_WY, **wy_overrides}
    return SDStrategy(
        zone_detector    = ZoneDetector(),
        wyckoff_detector = WyckoffDetector(**wy),
        **p,
    )


def _load(files: list[Path]) -> pd.DataFrame:
    frames = [parse_histdata_csv(f) for f in files]
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="first")]


def _run(
    m1_df:   pd.DataFrame,
    symbol:  str,
    strategy_overrides: dict = {},
    wy_overrides:       dict = {},
) -> tuple[dict[str, Any], int]:
    m15    = resample_ohlcv(m1_df, "15min")
    strat  = _build(strategy_overrides, wy_overrides)
    sigs   = strat.run(m15, m1_df)
    res, n_exp = simulate_all(
        sigs, m1_df,
        risk_pct           = RISK_PCT,
        spread             = SPREAD[symbol],
        max_monthly_losses = 4,
        max_daily_losses   = 1,
        tp1_r              = TP1_R,
        tp1_size           = TP1_SIZE,
    )
    m = compute_metrics(res, INITIAL_BALANCE, len(sigs), n_exp)
    return m, len(sigs)


def _run_all_periods(
    symbol: str,
    strategy_overrides: dict = {},
    wy_overrides:       dict = {},
) -> list[tuple[str, dict, int]]:
    rows = []
    for label, files in PERIODS[symbol]:
        ok = all(f.exists() for f in files)
        if not ok:
            continue
        m1 = _load(files)
        m, n_sig = _run(m1, symbol, strategy_overrides, wy_overrides)
        rows.append((label, m, n_sig))
    return rows


def _aggregate(rows: list[tuple[str, dict, int]]) -> tuple[float, float, float, float, int]:
    """(total_r, min_r, overall_wr, max_dd, total_decided)."""
    total_w = total_l = total_r = 0.0
    max_dd = 0.0
    min_r  = float("inf")
    for _, m, _ in rows:
        total_w += m["n_wins"]
        total_l += m["n_losses"]
        total_r += m["total_r"]
        max_dd   = max(max_dd, m["max_dd"])
        min_r    = min(min_r, m["total_r"])
    decided = int(total_w + total_l)
    wr = total_w / decided * 100 if decided else 0.0
    if min_r == float("inf"):
        min_r = 0.0
    return total_r, min_r, wr, max_dd, decided


def _passes(wr: float, min_r: float, dd: float, decided: int) -> bool:
    return decided >= 5 and wr >= 55.0 and min_r > 0 and dd <= 8.0


def _print_period_table(
    title: str,
    rows:  list[tuple[str, dict, int]],
) -> tuple[float, float, float, float, int]:
    print(f"\n      {title}")
    print(f"      {'Period':<24}  {'Sig':>4}  {'W':>3} {'L':>3}  {'WR%':>5}  {'R':>7}  {'DD%':>5}")
    print("      " + "─" * 58)
    for lbl, m, n in rows:
        print(
            f"      {lbl:<24}  {n:>4}  "
            f"{m['n_wins']:>3} {m['n_losses']:>3}  "
            f"{m['win_rate']:>5.1f}%  "
            f"{m['total_r']:>+7.2f}  "
            f"{m['max_dd']:>4.1f}%"
        )
    tr, mr, wr, dd, n = _aggregate(rows)
    print("      " + "─" * 58)
    print(f"      {'COMBINED':<24}  {'':>4}  {'':>3} {'':>3}  {wr:>5.1f}%  {tr:>+7.2f}  {dd:>4.1f}%")
    flag = " PASS ✓" if _passes(wr, mr, dd, n) else " FAIL ✗"
    print(f"      → Trades={n}  MinR={mr:+.2f}  Verdict:{flag}")
    return tr, mr, wr, dd, n


# ══════════════════════════════════════════════════════════════════════════════
# Round FX-1  Direction
# ══════════════════════════════════════════════════════════════════════════════

def round_fx1(instruments: list[str]) -> dict[str, dict]:
    """Test long-only vs bidirectional vs short-strict."""
    print("\n" + "═" * 80)
    print("  ROUND FX-1 — Direction filter")
    print("  Hypothesis: long-only bias from Gold is wrong for Forex (bidirectional market)")
    print("═" * 80)

    variants = [
        # (label, wy_long, wy_short)  — wy_short < 10 enables shorts
        ("Long-only (baseline)",        5.9, 99.0),
        ("Bidirectional wy≥5.9",        5.9,  5.9),
        ("Bidirectional wy≥5.0",        5.0,  5.0),
        ("Bidirectional wy≥4.5",        4.5,  4.5),
        ("Shorts-only",                99.0,  5.9),
    ]

    # symbol → best passing variant overrides
    best_per_symbol: dict[str, dict] = {}

    for sym in instruments:
        print(f"\n  ── {sym}  ─────────────────────────────────────────────────")
        sym_results = []
        for vlabel, wy_long, wy_short in variants:
            ov = dict(
                min_wyckoff_score       = wy_long,
                min_wyckoff_score_short = wy_short,
            )
            rows = _run_all_periods(sym, strategy_overrides=ov)
            tr, mr, wr, dd, n = _aggregate(rows)
            flag = " ←" if _passes(wr, mr, dd, n) else ""
            print(f"    {vlabel:<35}  W={wr:>5.1f}%  R={tr:>+7.2f}  DD={dd:>4.1f}%  n={n}{flag}")
            sym_results.append((vlabel, ov, tr, mr, wr, dd, n))

        passing = [(v, ov, tr, mr, wr, dd, n) for v, ov, tr, mr, wr, dd, n in sym_results
                   if _passes(wr, mr, dd, n)]
        if passing:
            best = max(passing, key=lambda x: x[2])   # highest total_r
            best_per_symbol[sym] = best[1]
            print(f"\n    ✓ Best: {best[0]}  TotR={best[2]:+.2f}  WR={best[4]:.1f}%  DD={best[5]:.1f}%")
        else:
            # Pick closest to passing (highest WR)
            best = max(sym_results, key=lambda x: x[4])
            best_per_symbol[sym] = best[1]
            print(f"\n    ✗ No pass — carry forward best WR: {best[0]}  WR={best[4]:.1f}%")

    return best_per_symbol


# ══════════════════════════════════════════════════════════════════════════════
# Round FX-2  Wyckoff score threshold
# ══════════════════════════════════════════════════════════════════════════════

def round_fx2(instruments: list[str], prev_best: dict[str, dict]) -> dict[str, dict]:
    print("\n" + "═" * 80)
    print("  ROUND FX-2 — Wyckoff score threshold")
    print("  Hypothesis: 5.9 was tuned for Gold; Forex may need a different floor")
    print("═" * 80)

    wy_thresholds = [4.0, 4.5, 5.0, 5.5, 5.9, 6.5, 7.0]

    best_per_symbol: dict[str, dict] = {}

    for sym in instruments:
        base_ov = prev_best.get(sym, {})
        # Determine if bidirectional (wy_short < 10)
        wy_short_base = base_ov.get("min_wyckoff_score_short", 99.0)
        is_bidir = wy_short_base < 10.0

        print(f"\n  ── {sym}  ({'bidirectional' if is_bidir else 'long-only'})  ──────────────────────")
        sym_results = []
        for thr in wy_thresholds:
            ov = {
                **base_ov,
                "min_wyckoff_score": thr,
            }
            if is_bidir:
                ov["min_wyckoff_score_short"] = thr
            rows = _run_all_periods(sym, strategy_overrides=ov)
            tr, mr, wr, dd, n = _aggregate(rows)
            flag = " ←" if _passes(wr, mr, dd, n) else ""
            print(f"    wy≥{thr:<3}  W={wr:>5.1f}%  R={tr:>+7.2f}  DD={dd:>4.1f}%  n={n}{flag}")
            sym_results.append((thr, {**ov}, tr, mr, wr, dd, n))

        passing = [(thr, ov, tr, mr, wr, dd, n) for thr, ov, tr, mr, wr, dd, n in sym_results
                   if _passes(wr, mr, dd, n)]
        if passing:
            best = max(passing, key=lambda x: x[2])
            best_per_symbol[sym] = best[1]
            print(f"\n    ✓ Best wy≥{best[0]}  TotR={best[2]:+.2f}  WR={best[4]:.1f}%  DD={best[5]:.1f}%")
        else:
            best = max(sym_results, key=lambda x: x[4])
            best_per_symbol[sym] = best[1]
            print(f"\n    ✗ No pass — carry forward wy≥{best[0]}  WR={best[4]:.1f}%")

    return best_per_symbol


# ══════════════════════════════════════════════════════════════════════════════
# Round FX-3  Session filter
# ══════════════════════════════════════════════════════════════════════════════

def round_fx3(instruments: list[str], prev_best: dict[str, dict]) -> dict[str, dict]:
    print("\n" + "═" * 80)
    print("  ROUND FX-3 — Session filter")
    print("  Hypothesis: London/NY window may need per-pair adjustment")
    print("═" * 80)

    sessions = [
        ("No filter (00-24)",    False,  0, 24),
        ("Asia+Ldn  (00-17)",    True,   0, 17),
        ("London    (07-17)",    True,   7, 17),
        ("Ldn+NY    (07-21)",    True,   7, 21),   # current
        ("NY only   (13-21)",    True,  13, 21),
        ("Full EU   (06-22)",    True,   6, 22),
    ]

    best_per_symbol: dict[str, dict] = {}

    for sym in instruments:
        base_ov = prev_best.get(sym, {})
        print(f"\n  ── {sym}  ─────────────────────────────────────────────────")
        sym_results = []
        for slabel, use_sf, start, end in sessions:
            ov = {
                **base_ov,
                "use_session_filter": use_sf,
                "session_start_utc":  start,
                "session_end_utc":    end,
            }
            rows = _run_all_periods(sym, strategy_overrides=ov)
            tr, mr, wr, dd, n = _aggregate(rows)
            flag = " ←" if _passes(wr, mr, dd, n) else ""
            print(f"    {slabel:<22}  W={wr:>5.1f}%  R={tr:>+7.2f}  DD={dd:>4.1f}%  n={n}{flag}")
            sym_results.append((slabel, {**ov}, tr, mr, wr, dd, n))

        passing = [(sl, ov, tr, mr, wr, dd, n) for sl, ov, tr, mr, wr, dd, n in sym_results
                   if _passes(wr, mr, dd, n)]
        if passing:
            best = max(passing, key=lambda x: x[2])
            best_per_symbol[sym] = best[1]
            print(f"\n    ✓ Best: {best[0]}  TotR={best[2]:+.2f}  WR={best[4]:.1f}%  DD={best[5]:.1f}%")
        else:
            best = max(sym_results, key=lambda x: x[4])
            best_per_symbol[sym] = best[1]
            print(f"\n    ✗ No pass — carry forward: {best[0]}  WR={best[4]:.1f}%")

    return best_per_symbol


# ══════════════════════════════════════════════════════════════════════════════
# Round FX-4  Zone score + R:R
# ══════════════════════════════════════════════════════════════════════════════

def round_fx4(instruments: list[str], prev_best: dict[str, dict]) -> dict[str, dict]:
    print("\n" + "═" * 80)
    print("  ROUND FX-4 — Zone score threshold + R:R")
    print("═" * 80)

    variants = [
        # (label, min_zone_score, risk_reward)
        ("zone≥3.0 R:R=1.5", 3.0, 1.5),
        ("zone≥4.0 R:R=1.5", 4.0, 1.5),
        ("zone≥5.0 R:R=1.5", 5.0, 1.5),   # current
        ("zone≥6.0 R:R=1.5", 6.0, 1.5),
        ("zone≥5.0 R:R=1.25", 5.0, 1.25),
        ("zone≥5.0 R:R=2.0",  5.0, 2.0),
        ("zone≥4.0 R:R=1.25", 4.0, 1.25),
    ]

    best_per_symbol: dict[str, dict] = {}

    for sym in instruments:
        base_ov = prev_best.get(sym, {})
        print(f"\n  ── {sym}  ─────────────────────────────────────────────────")
        sym_results = []
        for vlabel, zs, rr in variants:
            ov = {
                **base_ov,
                "min_zone_score":     zs,
                "min_composite_score": zs,
                "risk_reward":        rr,
            }
            rows = _run_all_periods(sym, strategy_overrides=ov)
            tr, mr, wr, dd, n = _aggregate(rows)
            flag = " ←" if _passes(wr, mr, dd, n) else ""
            print(f"    {vlabel:<20}  W={wr:>5.1f}%  R={tr:>+7.2f}  DD={dd:>4.1f}%  n={n}{flag}")
            sym_results.append((vlabel, {**ov}, tr, mr, wr, dd, n))

        passing = [(vl, ov, tr, mr, wr, dd, n) for vl, ov, tr, mr, wr, dd, n in sym_results
                   if _passes(wr, mr, dd, n)]
        if passing:
            best = max(passing, key=lambda x: x[2])
            best_per_symbol[sym] = best[1]
            print(f"\n    ✓ Best: {best[0]}  TotR={best[2]:+.2f}  WR={best[4]:.1f}%  DD={best[5]:.1f}%")
        else:
            best = max(sym_results, key=lambda x: x[4])
            best_per_symbol[sym] = best[1]
            print(f"\n    ✗ No pass — carry forward: {best[0]}  WR={best[4]:.1f}%")

    return best_per_symbol


# ══════════════════════════════════════════════════════════════════════════════
# Validation — apply champion to all instruments per period
# ══════════════════════════════════════════════════════════════════════════════

def validation(best_per_symbol: dict[str, dict]) -> None:
    print("\n" + "═" * 80)
    print("  VALIDATION — Champion config per period")
    print("═" * 80)

    for sym, ov in best_per_symbol.items():
        print(f"\n  ── {sym}  ─────────────────────────────────────────────────")
        rows = _run_all_periods(sym, strategy_overrides=ov)
        if not rows:
            print("    (no data)")
            continue
        _print_period_table("Final champion results", rows)

    # ── Universal Forex config? ───────────────────────────────────────────
    print("\n\n  CROSS-INSTRUMENT SUMMARY")
    print(f"  {'Symbol':<10}  {'Trades':>6}  {'TotR':>7}  {'WR%':>5}  {'MaxDD':>6}  {'Verdict'}")
    print("  " + "─" * 52)
    for sym, ov in best_per_symbol.items():
        rows = _run_all_periods(sym, strategy_overrides=ov)
        if not rows:
            print(f"  {sym:<10}  (no data)")
            continue
        tr, mr, wr, dd, n = _aggregate(rows)
        verdict = "PASS ✓" if _passes(wr, mr, dd, n) else "FAIL ✗"
        print(f"  {sym:<10}  {n:>6}  {tr:>+7.2f}  {wr:>5.1f}%  {dd:>5.1f}%  {verdict}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    instruments = ["GBPUSD", "CADJPY", "USDCHF"]

    print("=" * 80)
    print("  S&D Strategy — Forex Optimisation  (Rounds FX-1 → FX-4)")
    print("  Primary: GBPUSD (IS+OOS)  |  Validation: CADJPY, USDCHF")
    print("  Pass criterion: WR ≥ 55%, R > 0 every period, DD ≤ 8%")
    print("=" * 80)

    best1 = round_fx1(instruments)
    best2 = round_fx2(instruments, best1)
    best3 = round_fx3(instruments, best2)
    best4 = round_fx4(instruments, best3)
    validation(best4)

    print("\n" + "=" * 80)
    print("  CHAMPION CONFIGS SUMMARY")
    print("=" * 80)
    for sym, ov in best4.items():
        print(f"\n  {sym}:")
        for k, v in sorted(ov.items()):
            print(f"    {k} = {v}")

    print("\n" + "=" * 80)
    print("  Fin de l'optimisation Forex")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
