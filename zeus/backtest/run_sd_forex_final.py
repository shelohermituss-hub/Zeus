"""
S&D Strategy — Forex Final Validation  (Round FX-FINAL)
=========================================================

Summary of FX-1→FX-4 findings:
  • Long-only fails on Forex (WR 28–42% vs 71% on XAUUSD)
  • Shorts-only wins FX-1 on all 3 pairs → Supply zones more reliable than Demand
  • Per-instrument champions all PASS but configs differ widely (overfitting risk)

This round tests a single "Forex Universal" config across all pairs simultaneously.
A universal config that passes all instruments is far more likely to represent
genuine edge than per-instrument tuned params.

Forex-Universal candidate logic:
  - Bidirectional (supply+demand): supply zones are strong, demand zones viable
  - wy≥5.9 for both sides: Gold standard, conservative
  - session 07-21h: covers London+NY for all EU/USD pairs
  - zone≥4.0: slightly relaxed vs Gold (5.0) to compensate smaller signals
  - R:R=1.25: slightly tighter than Gold (1.5) — Forex moves are mean-reverting
  - TP1@0.75R (50%): early partial to bank gains on choppier Forex structure

Also tests a "Shorts-Dominant" variant (wy_long≥7.0, wy_short≥5.9) based on
the FX-1 finding that shorts systematically outperform longs on Forex.

Pass criterion: WR ≥ 55% combined AND R > 0 AND DD ≤ 8%
(relaxed from XAUUSD's 60% given limited Forex data)

Usage
-----
    python -m zeus.backtest.run_sd_forex_final
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

SPREAD = dict(GBPUSD=0.0001, CADJPY=0.020, USDCHF=0.0001)

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

_BASE_WY = dict(
    lookback=200, max_accum_bars=20, accum_range_mult=6.0,
    mss_lookback=60, min_spring_sweep_pct=0.10, min_mss_strength_pct=0.03,
)

# ── Per-instrument champions from FX-1→FX-4 ──────────────────────────────────
_CHAMPIONS = {
    "GBPUSD": dict(
        risk_reward=1.25, min_zone_score=4.0, min_composite_score=4.0,
        min_wyckoff_score=5.5, min_wyckoff_score_short=5.5,
        signal_cooldown=10, use_trend_filter=True, trend_slope_lookback=3,
        use_price_above_ema=True, use_session_filter=True,
        session_start_utc=0, session_end_utc=17,
        max_signals_per_day=6, use_adx_filter=False,
        use_h4_trend_filter=False, use_rsi_filter=False,
    ),
    "CADJPY": dict(
        risk_reward=1.5, min_zone_score=3.0, min_composite_score=3.0,
        min_wyckoff_score=5.9, min_wyckoff_score_short=5.9,
        signal_cooldown=10, use_trend_filter=True, trend_slope_lookback=3,
        use_price_above_ema=True, use_session_filter=True,
        session_start_utc=13, session_end_utc=21,
        max_signals_per_day=6, use_adx_filter=False,
        use_h4_trend_filter=False, use_rsi_filter=False,
    ),
    "USDCHF": dict(
        risk_reward=1.25, min_zone_score=4.0, min_composite_score=4.0,
        min_wyckoff_score=7.0, min_wyckoff_score_short=7.0,
        signal_cooldown=10, use_trend_filter=True, trend_slope_lookback=3,
        use_price_above_ema=True, use_session_filter=True,
        session_start_utc=7, session_end_utc=21,
        max_signals_per_day=6, use_adx_filter=False,
        use_h4_trend_filter=False, use_rsi_filter=False,
    ),
}

# ── Universal configs to test ─────────────────────────────────────────────────
_UNIVERSALS = [
    (
        "Universal-A (bidir wy≥5.9 07-21 zone≥4.0 R:R=1.25)",
        dict(
            risk_reward=1.25, min_zone_score=4.0, min_composite_score=4.0,
            min_wyckoff_score=5.9, min_wyckoff_score_short=5.9,
            signal_cooldown=10, use_trend_filter=True, trend_slope_lookback=3,
            use_price_above_ema=True, use_session_filter=True,
            session_start_utc=7, session_end_utc=21, max_signals_per_day=6,
            use_adx_filter=False, use_h4_trend_filter=False, use_rsi_filter=False,
        ),
        dict(tp1_r=0.0,  tp1_size=0.5),
    ),
    (
        "Universal-B (+TP1@0.75R 50%→1.25R)",
        dict(
            risk_reward=1.25, min_zone_score=4.0, min_composite_score=4.0,
            min_wyckoff_score=5.9, min_wyckoff_score_short=5.9,
            signal_cooldown=10, use_trend_filter=True, trend_slope_lookback=3,
            use_price_above_ema=True, use_session_filter=True,
            session_start_utc=7, session_end_utc=21, max_signals_per_day=6,
            use_adx_filter=False, use_h4_trend_filter=False, use_rsi_filter=False,
        ),
        dict(tp1_r=0.75, tp1_size=0.5),
    ),
    (
        "Universal-C (shorts-dom wy_L≥7.0 wy_S≥5.9)",
        dict(
            risk_reward=1.25, min_zone_score=4.0, min_composite_score=4.0,
            min_wyckoff_score=7.0, min_wyckoff_score_short=5.9,
            signal_cooldown=10, use_trend_filter=True, trend_slope_lookback=3,
            use_price_above_ema=True, use_session_filter=True,
            session_start_utc=7, session_end_utc=21, max_signals_per_day=6,
            use_adx_filter=False, use_h4_trend_filter=False, use_rsi_filter=False,
        ),
        dict(tp1_r=0.0,  tp1_size=0.5),
    ),
    (
        "Universal-D (shorts-dom + TP1@0.75R)",
        dict(
            risk_reward=1.25, min_zone_score=4.0, min_composite_score=4.0,
            min_wyckoff_score=7.0, min_wyckoff_score_short=5.9,
            signal_cooldown=10, use_trend_filter=True, trend_slope_lookback=3,
            use_price_above_ema=True, use_session_filter=True,
            session_start_utc=7, session_end_utc=21, max_signals_per_day=6,
            use_adx_filter=False, use_h4_trend_filter=False, use_rsi_filter=False,
        ),
        dict(tp1_r=0.75, tp1_size=0.5),
    ),
    (
        "Universal-E (shorts-dom zone≥5.0 R:R=1.5)",
        dict(
            risk_reward=1.5, min_zone_score=5.0, min_composite_score=5.0,
            min_wyckoff_score=7.0, min_wyckoff_score_short=5.9,
            signal_cooldown=10, use_trend_filter=True, trend_slope_lookback=3,
            use_price_above_ema=True, use_session_filter=True,
            session_start_utc=7, session_end_utc=21, max_signals_per_day=6,
            use_adx_filter=False, use_h4_trend_filter=False, use_rsi_filter=False,
        ),
        dict(tp1_r=0.0,  tp1_size=0.5),
    ),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(files: list[Path]) -> pd.DataFrame:
    frames = [parse_histdata_csv(f) for f in files]
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="first")]


def _run(m1_df: pd.DataFrame, symbol: str, strat_params: dict, tp1_r=0.0, tp1_size=0.5) -> tuple[dict, int]:
    m15   = resample_ohlcv(m1_df, "15min")
    strat = SDStrategy(
        zone_detector    = ZoneDetector(),
        wyckoff_detector = WyckoffDetector(**_BASE_WY),
        **strat_params,
    )
    sigs = strat.run(m15, m1_df)
    res, n_exp = simulate_all(
        sigs, m1_df,
        risk_pct=RISK_PCT, spread=SPREAD[symbol],
        max_monthly_losses=4, max_daily_losses=1,
        tp1_r=tp1_r, tp1_size=tp1_size,
    )
    m = compute_metrics(res, INITIAL_BALANCE, len(sigs), n_exp)
    return m, len(sigs)


def _passes(wr: float, min_r: float, dd: float, n: int) -> bool:
    return n >= 5 and wr >= 55.0 and min_r > 0 and dd <= 8.0


def _print_table(label: str, rows: list[tuple[str, dict, int]]) -> tuple[float, float, float, float, int]:
    print(f"\n  {label}")
    print(f"  {'Period':<24}  {'Sig':>4}  {'W':>3} {'L':>3}  {'WR%':>5}  {'R':>7}  {'P&L$':>8}  {'DD%':>5}")
    print("  " + "─" * 68)
    tw = tl = tr = max_dd = 0.0
    min_r = float("inf")
    for lbl, m, n in rows:
        print(
            f"  {lbl:<24}  {n:>4}  "
            f"{m['n_wins']:>3} {m['n_losses']:>3}  "
            f"{m['win_rate']:>5.1f}%  "
            f"{m['total_r']:>+7.2f}  "
            f"{m['total_usd']:>+8.0f}$  "
            f"{m['max_dd']:>4.1f}%"
        )
        tw += m["n_wins"]; tl += m["n_losses"]; tr += m["total_r"]
        max_dd = max(max_dd, m["max_dd"])
        min_r  = min(min_r, m["total_r"])
    decided  = int(tw + tl)
    wr = tw / decided * 100 if decided else 0.0
    if min_r == float("inf"): min_r = 0.0
    print("  " + "─" * 68)
    print(f"  {'COMBINED':<24}  {'':>4}  {int(tw):>3} {int(tl):>3}  {wr:>5.1f}%  {tr:>+7.2f}  {'':>8}  {max_dd:>4.1f}%")
    flag = " PASS ✓" if _passes(wr, min_r, max_dd, decided) else " FAIL ✗"
    print(f"  → Trades={decided}  MinR={min_r:+.2f}  Verdict:{flag}")
    return tr, min_r, wr, max_dd, decided


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    instruments = ["GBPUSD", "CADJPY", "USDCHF"]

    print("=" * 90)
    print("  S&D Strategy — Forex Final Validation  (Round FX-FINAL)")
    print("  Per-instrument champions vs Universal Forex configs")
    print("=" * 90)

    # ── Section 1: Per-instrument champions (from FX-1→FX-4) ─────────────
    print("\n" + "─" * 90)
    print("  SECTION 1 — Per-instrument champions  (WARNING: single-period IS, high overfitting risk)")
    print("─" * 90)

    champ_summary = []
    for sym in instruments:
        p = _CHAMPIONS[sym]
        rows = []
        for lbl, files in PERIODS[sym]:
            if not all(f.exists() for f in files):
                continue
            m1 = _load(files)
            m, n_sig = _run(m1, sym, p, tp1_r=1.25, tp1_size=0.5)
            rows.append((lbl, m, n_sig))
        tr, mr, wr, dd, n = _print_table(f"{sym} champion", rows)
        champ_summary.append((sym, tr, mr, wr, dd, n))

    # ── Section 2: Universal Forex configs ───────────────────────────────
    print("\n" + "─" * 90)
    print("  SECTION 2 — Universal Forex config (single params across ALL instruments)")
    print("  A passing universal config is much more robust than per-instrument tuning")
    print("─" * 90)

    univ_results: list[tuple[str, list]] = []
    for config_label, strat_p, exec_p in _UNIVERSALS:
        print(f"\n  ── {config_label}")
        per_sym = []
        all_pass = True
        for sym in instruments:
            rows = []
            for lbl, files in PERIODS[sym]:
                if not all(f.exists() for f in files):
                    continue
                m1 = _load(files)
                m, n_sig = _run(m1, sym, strat_p, **exec_p)
                rows.append((lbl, m, n_sig))
            tw = tl = tr = max_dd = 0.0
            min_r = float("inf")
            for _, m, _ in rows:
                tw += m["n_wins"]; tl += m["n_losses"]; tr += m["total_r"]
                max_dd = max(max_dd, m["max_dd"])
                min_r  = min(min_r, m["total_r"])
            decided = int(tw + tl)
            wr = tw / decided * 100 if decided else 0.0
            if min_r == float("inf"): min_r = 0.0
            ok = _passes(wr, min_r, max_dd, decided)
            if not ok: all_pass = False
            flag = " ✓" if ok else " ✗"
            print(f"    {sym:<10}  W={wr:>5.1f}%  R={tr:>+7.2f}  DD={max_dd:>4.1f}%  n={decided}{flag}")
            per_sym.append((sym, tr, min_r, wr, max_dd, decided, ok))
        flag = " ← ALL PASS" if all_pass else ""
        print(f"    → {'ALL PASS' if all_pass else 'NOT universal'}{flag}")
        univ_results.append((config_label, per_sym, all_pass))

    # ── Section 3: Universal champion detail ─────────────────────────────
    passing_universal = [(lbl, ps, ap) for lbl, ps, ap in univ_results if ap]
    if passing_universal:
        # Best universal = highest combined R across instruments
        def combined_r(x):
            return sum(ps[1] for ps in x[1])   # sum of total_r per instrument
        best_lbl, best_per_sym, _ = max(passing_universal, key=combined_r)
        best_idx = next(i for i, (lbl, _, _) in enumerate(_UNIVERSALS) if lbl == best_lbl)
        best_strat, best_exec = _UNIVERSALS[best_idx][1], _UNIVERSALS[best_idx][2]

        print("\n" + "─" * 90)
        print(f"  SECTION 3 — Best universal champion: {best_lbl}")
        print("─" * 90)
        for sym in instruments:
            rows = []
            for lbl, files in PERIODS[sym]:
                if not all(f.exists() for f in files):
                    continue
                m1 = _load(files)
                m, n_sig = _run(m1, sym, best_strat, **best_exec)
                rows.append((lbl, m, n_sig))
            _print_table(sym, rows)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  FINAL SUMMARY")
    print("=" * 90)

    print("\n  Per-instrument champions (IS only — overfitting likely):")
    print(f"  {'Symbol':<10}  {'Trades':>6}  {'TotR':>7}  {'WR%':>5}  {'MaxDD':>5}")
    for sym, tr, mr, wr, dd, n in champ_summary:
        print(f"  {sym:<10}  {n:>6}  {tr:>+7.2f}  {wr:>5.1f}%  {dd:>4.1f}%")

    print(f"\n  Universal configs passing ALL instruments: {len(passing_universal)} / {len(_UNIVERSALS)}")
    for lbl, ps, ap in univ_results:
        n_pass = sum(1 for p in ps if p[6])
        print(f"  {'PASS' if ap else 'FAIL':4}  [{n_pass}/{len(instruments)}]  {lbl}")

    if not passing_universal:
        print("\n  ⚠ No universal config passes all instruments.")
        print("  Conclusion: per-instrument configs needed → HIGH overfitting risk.")
        print("  Recommendation: collect more data before deploying on Forex.")
    else:
        print(f"\n  ✓ Universal champion: {passing_universal[0][0]}")
        print("  Conclusion: robust Forex edge found — validate on fresh data before live.")

    # Overfitting warning
    print("\n  ⚠ OVERFITTING WARNING:")
    print("    - GBPUSD OOS = 1 month, 1 trade (insufficient)")
    print("    - CADJPY / USDCHF = IS only (no OOS available yet)")
    print("    - All Forex configs require ≥6 months of OOS validation before live")
    print("    - Do NOT deploy Forex without additional out-of-sample data")

    print("\n" + "=" * 90)
    print("  Fin du Round FX-FINAL")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
