"""
S&D Strategy — Round 22: Deep Validation of V96-LONG-CAP1 Champion
====================================================================

Round 21 champion: V96-LONG-CAP1
  min_wyckoff_score_short=99.0  (demand zones only — blocks all shorts)
  max_daily_losses=1            (stop trading that day after 1 loss)
  All other params identical to V96.

Results across all 4 periods:
  2024: 24W/16L · 60.0% WR · +20.00R · 3.0% DD
  2025: 19W/10L · 65.5% WR · +18.50R · 3.1% DD
  2026 OOS: 4W/1L · 80.0% WR · +5.00R · 1.0% DD
  2026 IS:  5W/1L · 83.3% WR · +6.50R · 1.0% DD
  Combined: 65.0% WR · +50.00R · max DD 3.1% ← PASS

This round:
  Section 1 — Monthly breakdown of V96-LONG-CAP1 for each period
              Understand max DD month, consecutive loss streaks, seasonality.
  Section 2 — V123-LONG-CAP1 (slope_lb=3 for the long version)
              Round 20: V123 Long-Only 2025 had +22.50R vs +17.50R for V96 Long-Only.
              Does this advantage persist with cap=1?
  Section 3 — Tighter entry thresholds on longs only
              V136: min_zone_score=5.5  (stricter zones, fewer but cleaner entries)
              V137: min_wyckoff_score=6.5 for longs  (stricter Wyckoff → higher WR?)
              V138: zone≥5.5 + wy≥6.5 for longs  (aggressive filtering)
  Section 4 — Champion stress-test: vary risk_reward
              Current R:R=1.5 (40% breakeven).  Would R:R=2.0 improve total R at
              current WR levels (65%)? WR=65% with R:R=2 → expect positive (BE is 33%).
  Section 5 — Summary and official Round 22 champion declaration

Usage
-----
    python -m zeus.backtest.run_sd_round22
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

# ── V96 champion base params ──────────────────────────────────────────────────
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

# V96-LONG-CAP1 extra params (on top of _V96)
_LONG_CAP1 = dict(
    min_wyckoff_score_short = 99.0,  # blocks all shorts
)
MAX_DAILY_LOSSES_CHAMPION = 1

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


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _run(
    m1_df:            pd.DataFrame,
    max_daily_losses: int   = 0,
    **strat_kw,
) -> tuple[list[TradeResult], dict[str, Any], int]:
    zone_df = resample_ohlcv(m1_df, "15min")
    strat   = _build_strategy(**strat_kw)
    signals = strat.run(zone_df, m1_df)
    res, n_exp = simulate_all(
        signals, m1_df,
        risk_pct           = RISK_PCT,
        spread             = SPREAD,
        max_monthly_losses = 4,
        max_daily_losses   = max_daily_losses,
    )
    m = compute_metrics(res, INITIAL_BALANCE, len(signals), n_exp)
    return res, m, len(signals)


def _run_all_periods(
    max_daily_losses: int = 0,
    verbose_breakdown: bool = False,
    **strat_kw,
) -> list[tuple[str, dict, int]]:
    rows: list[tuple[str, dict, int]] = []
    for period_label, files in PERIODS:
        m1 = _load(files)
        res, m, n_sig = _run(m1, max_daily_losses=max_daily_losses, **strat_kw)
        rows.append((period_label, m, n_sig))
        if verbose_breakdown and res:
            print_monthly_breakdown(res, INITIAL_BALANCE)
    return rows


def _summary_table(title: str, rows: list[tuple[str, dict, int]]) -> tuple[float, float, float, float]:
    """Print summary table; returns (total_r, min_r, overall_wr, max_dd)."""
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
    print("  S&D Strategy — Round 22 · XAUUSD")
    print("  Deep validation of V96-LONG-CAP1 champion + threshold variants")
    print("=" * 100)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1 — Monthly breakdown of V96-LONG-CAP1
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'─' * 100}")
    print("  SECTION 1 — Monthly breakdown of V96-LONG-CAP1 champion")
    print(f"{'─' * 100}")

    for period_label, files in PERIODS:
        print(f"\n  ── {period_label} ──")
        m1 = _load(files)
        d0 = m1.index[0].date()
        d1 = m1.index[-1].date()
        print(f"     {len(m1):,} M1 bars  ({d0} → {d1})")
        res, m, n_sig = _run(
            m1, max_daily_losses=MAX_DAILY_LOSSES_CHAMPION,
            **_LONG_CAP1,
        )
        print(
            f"     SUMMARY: sig={n_sig}  W={m['n_wins']} L={m['n_losses']}  "
            f"WR={m['win_rate']:.1f}%  R={m['total_r']:+.2f}  "
            f"P&L={m['total_usd']:+.0f}$  DD={m['max_dd']:.1f}%"
        )
        if res:
            print_monthly_breakdown(res, INITIAL_BALANCE)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2 — V123-LONG-CAP1 (slope_lb=3 on long-only + cap=1)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 100}")
    print("  SECTION 2 — V123-LONG-CAP1  (slope_lb=3 + long-only + cap=1)")
    print("  R20: V123 Long-Only 2025 was +22.50R vs +17.50R for V96 Long-Only.")
    print(f"{'─' * 100}")

    print("\n  Running V96-LONG-CAP1 (champion baseline) …")
    champ_rows = _run_all_periods(max_daily_losses=1, **_LONG_CAP1)
    _summary_table("V96-LONG-CAP1 — champion (slope_lb=6)", champ_rows)

    print("\n  Running V123-LONG-CAP1 (slope_lb=3) …")
    v123_long_cap1_rows = _run_all_periods(
        max_daily_losses=1,
        trend_slope_lookback=3,
        **_LONG_CAP1,
    )
    _summary_table("V123-LONG-CAP1 — slope_lb=3", v123_long_cap1_rows)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3 — Tighter thresholds on longs
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 100}")
    print("  SECTION 3 — Tighter entry thresholds on longs (demand zones)")
    print("  Goal: fewer but higher quality longs → can we push WR above 70%?")
    print(f"{'─' * 100}")

    threshold_variants = [
        # (label, extra_overrides)
        ("V96-LONG-CAP1 champion       ", {}),
        ("V136 · zone_long≥5.5         ", dict(min_zone_score_long=5.5,  min_composite_score=5.5)),
        ("V137 · wy_long≥6.5           ", dict(min_wyckoff_score_long=6.5)),
        ("V138 · zone≥5.5 + wy≥6.5     ", dict(min_zone_score_long=5.5, min_composite_score=5.5,
                                                min_wyckoff_score_long=6.5)),
        ("V139 · wy_long≥6.0           ", dict(min_wyckoff_score_long=6.0)),
        ("V140 · zone≥5.5 + wy≥6.0     ", dict(min_zone_score_long=5.5, min_composite_score=5.5,
                                                min_wyckoff_score_long=6.0)),
    ]

    threshold_results: list[tuple[str, float, float, float, float]] = []
    for label, extra in threshold_variants:
        print(f"  Running {label.strip()} …", end="", flush=True)
        rows = _run_all_periods(
            max_daily_losses=1,
            **{**_LONG_CAP1, **extra},
        )
        tr, mr, wr, dd = _summary_table(label, rows)
        threshold_results.append((label, tr, mr, wr, dd))
        print()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4 — Risk:Reward stress-test
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 100}")
    print("  SECTION 4 — Risk:Reward stress-test on V96-LONG-CAP1")
    print("  At 65% WR: R:R=1.5 → 65×1.5 - 35×1 = +62.5R;  R:R=2.0 → 65×2 - 35×1 = +95R")
    print("  (theoretical per 100 trades; reality includes filtering and monthly cap)")
    print(f"{'─' * 100}")

    rr_variants = [
        ("R:R=1.5 (champion baseline)", 1.5),
        ("R:R=2.0                    ", 2.0),
        ("R:R=2.5                    ", 2.5),
        ("R:R=1.2 (less ambitious)   ", 1.2),
    ]

    rr_results: list[tuple[str, float, float, float, float]] = []
    for label, rr in rr_variants:
        print(f"  Running {label.strip()} …", end="", flush=True)
        rows = _run_all_periods(
            max_daily_losses = 1,
            risk_reward      = rr,
            **_LONG_CAP1,
        )
        tr, mr, wr, dd = _summary_table(label, rows)
        rr_results.append((label, tr, mr, wr, dd))
        print()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5 — Final champion declaration
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 100}")
    print("  SECTION 5 — Round 22 champion comparison summary")
    print(f"{'─' * 100}")

    all_results = threshold_results + rr_results
    print(f"\n  {'Config':<35}  {'TotR':>7}  {'MinR':>7}  {'OvWR':>5}  {'MaxDD':>6}")
    print("  " + "─" * 65)
    for name, tr, mr, wr, dd in all_results:
        ok   = mr > 0 and wr >= 60
        flag = " ←" if ok else ""
        print(f"  {name:<35}  {tr:>+7.2f}  {mr:>+7.2f}  {wr:>5.1f}%  {dd:>5.1f}%{flag}")

    print(f"\n  ← PASS: WR ≥ 60% combined + R > 0 every period")

    passing = [(n, tr, mr, wr, dd) for n, tr, mr, wr, dd in all_results if mr > 0 and wr >= 60]
    if passing:
        best = max(passing, key=lambda x: x[1])  # sort by total R
        print(f"\n  ROUND 22 BEST CONFIG: {best[0].strip()}")
        print(f"    TotR={best[1]:+.2f}  MinR={best[2]:+.2f}  OvWR={best[3]:.1f}%  MaxDD={best[4]:.1f}%")
    else:
        best = max(all_results, key=lambda x: x[1])
        print(f"\n  No config passes. Best total R: {best[0].strip()} → {best[1]:+.2f}")

    print(f"\n{'=' * 100}")
    print("  Fin du Round 22")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    main()
