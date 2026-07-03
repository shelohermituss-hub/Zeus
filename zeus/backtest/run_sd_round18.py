"""
S&D Strategy — Round 18: OOS Validation + Signal Frequency Improvements
=========================================================================

Two goals:
  1. Validate V96 champion across all available data periods (OOS check)
  2. Find the best way to increase signal frequency from 7 → ≥10 signals/3 months
     while maintaining WR ≥ 65%

Section 1 — Multi-year OOS validation (V96 champion config)
    Periods tested:
    - 2024 full year  (IS-A : training reference)
    - 2025 full year  (IS-B : training reference)
    - 2026 Jan–Mar    (OOS  : never used in rounds 1–17)
    - 2026 Apr–Jun    (IS   : our 3-month benchmark)
    Pass criteria: WR ≥ 60% and positive R in every period.

Section 2 — Frequency improvements (2026 Apr–Jun IS for comparability)
    V96  : champion baseline (7 sig / 71.4% WR / +5.50R)
    V87  : R:R=2.5 reference
    V112 : wy≥5.5  — relax Wyckoff from 5.9
    V113 : wy≥5.0  — aggressive Wyckoff relaxation
    V114 : zone≥4.5 — relax zone score from 5.0
    V115 : no price_above_ema (slope direction only)
    V116 : zone≥4.5 + wy≥5.5 (combined relaxation)
    V117 : zone≥4.5 + wy≥5.0 (aggressive combined)
    V118 : TP1@1.0R 50% exit → SL→BE → full TP at 1.5R
    V119 : no trend filter at all
    V120 : session 02–22h UTC (add Asian session)
    V121 : cooldown=5 M1 bars (from 10) — faster zone re-entry

Usage
-----
    python -m zeus.backtest.run_sd_round18
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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

_ROOT   = Path(__file__).parent.parent.parent
_XAU    = _ROOT / "data" / "historical" / "xauusd"
_XAU_M1 = _XAU / "m1"

INITIAL_BALANCE = 10_000.0
RISK_PCT        = 0.01
SPREAD          = 0.30     # USD/oz — XAUUSD typical CFD spread

# ── V96 champion base params ───────────────────────────────────────────────────
_V96 = dict(
    risk_reward          = 1.5,
    min_zone_score       = 5.0,
    min_wyckoff_score    = 5.9,
    min_composite_score  = 5.0,
    signal_cooldown      = 10,
    use_trend_filter     = True,
    trend_slope_lookback = 6,
    use_price_above_ema  = True,
    use_session_filter   = True,
    session_start_utc    = 7,
    session_end_utc      = 21,
    max_signals_per_day  = 6,
    use_adx_filter       = False,
    use_h4_trend_filter  = False,
    use_rsi_filter       = False,
)

_WY = dict(
    lookback             = 200,
    max_accum_bars       = 20,
    accum_range_mult     = 6.0,
    mss_lookback         = 60,
    min_spring_sweep_pct = 0.05,
    min_mss_strength_pct = 0.03,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_strategy(**overrides) -> SDStrategy:
    p = {**_V96, **overrides}
    return SDStrategy(
        zone_detector    = ZoneDetector(),
        wyckoff_detector = WyckoffDetector(**_WY),
        **p
    )


def _run(
    m1_df:    pd.DataFrame,
    zone_tf:  str   = "15min",
    tp1_r:    float = 0.0,
    tp1_size: float = 0.5,
    **strat_kw,
) -> tuple[list[TradeResult], dict[str, Any], int]:
    zone_df = resample_ohlcv(m1_df, zone_tf)
    strat   = _build_strategy(**strat_kw)
    signals = strat.run(zone_df, m1_df)
    res, n_exp = simulate_all(
        signals, m1_df,
        risk_pct           = RISK_PCT,
        spread             = SPREAD,
        max_monthly_losses = 4,
        tp1_r              = tp1_r,
        tp1_size           = tp1_size,
    )
    m = compute_metrics(res, INITIAL_BALANCE, len(signals), n_exp)
    return res, m, len(signals)


def _load(files: list[Path]) -> pd.DataFrame:
    frames = [parse_histdata_csv(f) for f in files]
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="first")]


def _line(label: str, m: dict, n_sig: int) -> None:
    print(
        f"  {label:<55}  "
        f"sig={n_sig:>4}  "
        f"W={m['n_wins']:>3} L={m['n_losses']:>3}  "
        f"WR={m['win_rate']:>5.1f}%  "
        f"R={m['total_r']:>+7.2f}  "
        f"P&L={m['total_usd']:>+8.0f}$  "
        f"DD={m['max_dd']:>4.1f}%"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 100)
    print("  S&D Strategy — Round 18 · XAUUSD")
    print("=" * 100)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1 — Multi-year OOS validation of V96 champion
    # ─────────────────────────────────────────────────────────────────────────
    print("\n  SECTION 1 — V96 champion across all data periods (OOS check)")
    print("  Config: zone≥5.0, wy≥5.9, R:R=1.5, M15 zones, slope+price_ema, mloss4")
    print(f"\n{'─' * 100}")

    periods = [
        ("2024 Full Year (IS-A)",
            [_XAU_M1 / "DAT_MT_XAUUSD_M1_2024.csv"]),
        ("2025 Full Year (IS-B)",
            [_XAU_M1 / "DAT_MT_XAUUSD_M1_2025.csv"]),
        ("2026 Jan–Mar  (OOS)  ",
            [_XAU_M1 / "DAT_MT_XAUUSD_M1_202601.csv",
             _XAU_M1 / "DAT_MT_XAUUSD_M1_202602.csv",
             _XAU_M1 / "DAT_MT_XAUUSD_M1_202603.csv"]),
        ("2026 Apr–Jun  (IS)   ",
            [_XAU_M1 / "DAT_MT_XAUUSD_M1_202604.csv",
             _XAU_M1 / "DAT_MT_XAUUSD_M1_202605.csv",
             _XAU_M1 / "DAT_MT_XAUUSD_M1_202606.csv"]),
    ]

    oos_rows: list[tuple] = []
    m1_is: Optional[pd.DataFrame] = None

    for period_label, files in periods:
        print(f"\n  ── {period_label} ──")
        m1 = _load(files)
        d0 = m1.index[0].date()
        d1 = m1.index[-1].date()
        print(f"     {len(m1):,} M1 bars  ({d0} → {d1})")
        print("     Running V96 …", end="", flush=True)
        res, m, n_sig = _run(m1)
        print(f" {n_sig} signaux")
        _line(period_label, m, n_sig)
        if res:
            print_monthly_breakdown(res, INITIAL_BALANCE)
        oos_rows.append((period_label, m, n_sig))
        if "2026 Apr" in period_label:
            m1_is = m1   # keep IS data for Section 2

    print(f"\n{'─' * 100}")
    print("  SECTION 1 SUMMARY — V96 across all periods:")
    print(f"  {'Period':<30}  {'Sig':>4}  {'W':>3} {'L':>3}  {'WR%':>5}  {'R':>7}  {'P&L$':>8}  {'DD%':>5}")
    print("  " + "─" * 74)
    total_w = total_l = total_sig = 0
    for lbl, m, n in oos_rows:
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
    decided = total_w + total_l
    overall_wr = total_w / decided * 100 if decided else 0.0
    print("  " + "─" * 74)
    print(f"  {'TOTAL / COMBINED':<30}  {total_sig:>4}  "
          f"{total_w:>3} {total_l:>3}  {overall_wr:>5.1f}%")

    pass_fail = "PASS ✓" if overall_wr >= 60 and all(
        m['total_r'] > 0 for _, m, n in oos_rows if n > 0
    ) else "FAIL ✗ — review needed"
    print(f"\n  VERDICT: {pass_fail}  (criteria: combined WR ≥ 60% + R > 0 in every period)")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2 — Signal frequency experiments on 2026 Q2 IS
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 100}")
    print("  SECTION 2 — Signal frequency experiments  (2026 Apr–Jun IS)")
    print("  Baseline: V96 = 7 signals / 71.4% WR / +5.50R in 3 months")
    print("  Goal: ≥ 10 signals / 3 months  +  WR ≥ 65%")
    print(f"{'─' * 100}\n")

    freq_variants: list[tuple[str, dict, float, float]] = [
        # (label, strategy overrides, tp1_r, tp1_size)
        ("V96  · BASELINE zone≥5.0 wy≥5.9 R:R=1.5",
            {},                                                                     0.0, 0.5),
        ("V87  · REFERENCE zone≥5.0 wy≥5.9 R:R=2.5",
            dict(risk_reward=2.5),                                                  0.0, 0.5),
        # — Wyckoff relaxation —
        ("V112 · wy≥5.5 (from 5.9)",
            dict(min_wyckoff_score=5.5),                                            0.0, 0.5),
        ("V113 · wy≥5.0 (from 5.9)",
            dict(min_wyckoff_score=5.0),                                            0.0, 0.5),
        # — Zone relaxation —
        ("V114 · zone≥4.5 (from 5.0)",
            dict(min_zone_score=4.5, min_composite_score=4.5),                      0.0, 0.5),
        # — Trend filter relaxation —
        ("V115 · no price_above_ema (slope only)",
            dict(use_price_above_ema=False),                                        0.0, 0.5),
        ("V119 · no trend filter",
            dict(use_trend_filter=False),                                           0.0, 0.5),
        # — Combined relaxation —
        ("V116 · zone≥4.5 + wy≥5.5",
            dict(min_zone_score=4.5, min_composite_score=4.5,
                 min_wyckoff_score=5.5),                                             0.0, 0.5),
        ("V117 · zone≥4.5 + wy≥5.0 (aggressive)",
            dict(min_zone_score=4.5, min_composite_score=4.5,
                 min_wyckoff_score=5.0),                                             0.0, 0.5),
        # — TP1 partial exit —
        ("V118 · TP1@1.0R 50% → SL→BE → full 1.5R",
            {},                                                                     1.0, 0.5),
        # — Session & cooldown —
        ("V120 · session 02–22h UTC (add Asian open)",
            dict(session_start_utc=2, session_end_utc=22),                          0.0, 0.5),
        ("V121 · cooldown=5 M1 bars (from 10)",
            dict(signal_cooldown=5),                                                0.0, 0.5),
    ]

    freq_results: list[tuple[str, dict, int]] = []
    for label, kw, tp1_r, tp1_size in freq_variants:
        print(f"  [{label}] …", end="", flush=True)
        res, m, n_sig = _run(m1_is, tp1_r=tp1_r, tp1_size=tp1_size, **kw)
        freq_results.append((label, m, n_sig))
        print(f" {n_sig} signaux")

    print(f"\n{'─' * 100}")
    print("  TABLEAU — 2026 Apr–Jun IS  (comparaison fréquence + qualité)")
    print(f"{'─' * 100}")
    for label, m, n_sig in freq_results:
        _line(label, m, n_sig)

    # — Verdict on frequency improvement —
    baseline_sig = freq_results[0][2]
    baseline_wr  = freq_results[0][1]["win_rate"]

    viable = [
        (l, m, n) for l, m, n in freq_results[2:]
        if n >= 10 and m["win_rate"] >= 65.0 and m["total_r"] > freq_results[0][1]["total_r"]
    ]

    print(f"\n  Baseline V96: {baseline_sig} signaux / WR={baseline_wr:.1f}%")
    if viable:
        best = max(viable, key=lambda x: x[1]["total_r"])
        print(f"  MEILLEURE AMÉLIORATION (≥10 sig, WR≥65%, R > baseline):")
        print(f"    {best[0]}  →  {best[2]} sig / WR={best[1]['win_rate']:.1f}% / R={best[1]['total_r']:+.2f}")
    else:
        # Fallback: best R without strict criteria
        top_r = max(freq_results[2:], key=lambda x: x[1]["total_r"])
        top_wr = max(freq_results[2:], key=lambda x: x[1]["win_rate"])
        top_sig = max(freq_results[2:], key=lambda x: x[2])
        print("  Aucune variante ne remplit tous les critères simultanément.")
        print(f"  Meilleur R :      {top_r[0]}  →  R={top_r[1]['total_r']:+.2f}")
        print(f"  Meilleur WR:      {top_wr[0]}  →  WR={top_wr[1]['win_rate']:.1f}%")
        print(f"  + de signaux:     {top_sig[0]}  →  {top_sig[2]} signaux")

    print(f"\n{'=' * 100}")
    print("  Fin du Round 18")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    main()
