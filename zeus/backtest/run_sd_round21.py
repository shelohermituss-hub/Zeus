"""
S&D Strategy — Round 21: H4 Trend Filter + Asymmetric Wyckoff Thresholds
==========================================================================

Following Round 20:
  - V96 champion: +49.00R total, max DD 4.1%, R > 0 in ALL periods (confirmed).
  - Long-only variants pass 60% WR but 2024 max DD rises to 6.3% (from 4.0%).
    Hypothesis: longs during H4 downtrends cause consecutive losses in 2024.
  - Break-even and V123 both rejected.

Round 21 investigates two targeted improvements:

Section 1 — H4 trend filter (use_h4_trend_filter=True)
    Add a higher-timeframe gate: only enter longs when H4 EMA50 slope is bullish,
    only enter shorts when H4 EMA50 slope is bearish.  Goal: filter out trades
    that go against the dominant 4-hour trend.
    Variants:
      V96+H4         : full V96 with H4 filter added
      V96-LONG+H4    : long-only + H4 bullish requirement
      V96+H4 lb=2    : H4 slope lookback=2 (more reactive)

Section 2 — Asymmetric Wyckoff thresholds (longs at 5.9, shorts at 6.5/7.0)
    Rather than blocking all shorts, raise the quality bar for supply signals only.
    This keeps profitable short setups while filtering the marginal ones that fail
    in trending markets.
    Variants:
      V133 : min_wyckoff_score_short=6.5  (was 5.9)
      V134 : min_wyckoff_score_short=7.0  (strict)
      V135 : min_wyckoff_score_short=6.5 + H4 filter

Section 3 — Daily loss cap combined with best Section 1/2 candidates
    Round 20 showed daily_loss_cap=1 improves total R from +49 to +53.50.
    Combine with H4 filter or asymmetric Wyckoff to see if both effects stack.

Usage
-----
    python -m zeus.backtest.run_sd_round21
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
    m1_df:        pd.DataFrame,
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
    **strat_kw,
) -> list[tuple[str, dict, int]]:
    rows: list[tuple[str, dict, int]] = []
    for period_label, files in PERIODS:
        m1 = _load(files)
        res, m, n_sig = _run(m1, max_daily_losses=max_daily_losses, **strat_kw)
        rows.append((period_label, m, n_sig))
    return rows


def _summary_table(title: str, rows: list[tuple[str, dict, int]]) -> None:
    print(f"\n  {title}")
    print(f"  {'Period':<30}  {'Sig':>4}  {'W':>3} {'L':>3}  {'WR%':>5}  {'R':>7}  {'P&L$':>8}  {'DD%':>5}")
    print("  " + "─" * 74)
    total_w = total_l = total_sig = total_r = 0.0
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
        if m['total_r'] <= 0 and n > 0:
            all_pos_r = False
    decided    = total_w + total_l
    overall_wr = total_w / decided * 100 if decided else 0.0
    print("  " + "─" * 74)
    print(f"  {'TOTAL / COMBINED':<30}  {int(total_sig):>4}  "
          f"{int(total_w):>3} {int(total_l):>3}  {overall_wr:>5.1f}%  {total_r:>+7.2f}")
    verdict = "PASS ✓" if overall_wr >= 60 and all_pos_r else "FAIL ✗"
    print(f"  VERDICT: {verdict}  (WR ≥ 60% combined + R > 0 every period)")


def main() -> None:
    print("=" * 100)
    print("  S&D Strategy — Round 21 · XAUUSD")
    print("  H4 Trend Filter + Asymmetric Wyckoff Thresholds + Daily Loss Cap")
    print("=" * 100)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1 — H4 trend filter
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'─' * 100}")
    print("  SECTION 1 — H4 trend filter (use_h4_trend_filter=True)")
    print("  H4 filter: long only when H4 EMA50 slope bullish; short only when bearish.")
    print("  V96 baseline for reference (reproduced): zone≥5.0, wy≥5.9, slope_lb=6")
    print(f"{'─' * 100}")

    print("\n  Running V96 baseline …")
    v96_rows = _run_all_periods()
    _summary_table("V96 — champion baseline", v96_rows)

    print("\n  Running V96 + H4 filter (slope_lb=3) …")
    v96_h4_rows = _run_all_periods(use_h4_trend_filter=True, h4_trend_slope_lb=3)
    _summary_table("V96 + H4 filter (h4_lb=3)", v96_h4_rows)

    print("\n  Running V96 + H4 filter (slope_lb=2) …")
    v96_h4_lb2_rows = _run_all_periods(use_h4_trend_filter=True, h4_trend_slope_lb=2)
    _summary_table("V96 + H4 filter (h4_lb=2)", v96_h4_lb2_rows)

    print("\n  Running V96 Long-Only + H4 filter (slope_lb=3) …")
    v96_long_h4_rows = _run_all_periods(
        use_h4_trend_filter   = True,
        h4_trend_slope_lb     = 3,
        min_wyckoff_score_short = 99.0,
    )
    _summary_table("V96 Long-Only + H4 filter (h4_lb=3)", v96_long_h4_rows)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2 — Asymmetric Wyckoff thresholds (shorts only)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 100}")
    print("  SECTION 2 — Asymmetric Wyckoff thresholds (longs=5.9, shorts raised)")
    print("  Goal: keep V96 long quality, tighten supply entries to reduce short losses.")
    print(f"{'─' * 100}")

    print("\n  Running V133 (wy_short=6.5) …")
    v133_rows = _run_all_periods(min_wyckoff_score_short=6.5)
    _summary_table("V133 — wy_short≥6.5 (long=5.9)", v133_rows)

    print("\n  Running V134 (wy_short=7.0) …")
    v134_rows = _run_all_periods(min_wyckoff_score_short=7.0)
    _summary_table("V134 — wy_short≥7.0 (long=5.9)", v134_rows)

    print("\n  Running V135 (wy_short=6.5 + H4 filter) …")
    v135_rows = _run_all_periods(
        min_wyckoff_score_short = 6.5,
        use_h4_trend_filter     = True,
        h4_trend_slope_lb       = 3,
    )
    _summary_table("V135 — wy_short≥6.5 + H4 filter", v135_rows)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3 — Daily loss cap combined with best Section 1/2 candidates
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 100}")
    print("  SECTION 3 — daily_loss_cap=1 stacked with best candidates")
    print("  Round 20: V96 + cap=1 gave +53.50R total vs +49.00R baseline.")
    print(f"{'─' * 100}")

    print("\n  Running V96 + daily_loss_cap=1 (recap from R20) …")
    v96_cap1_rows = _run_all_periods(max_daily_losses=1)
    _summary_table("V96 + daily_loss_cap=1", v96_cap1_rows)

    print("\n  Running V133 (wy_short=6.5) + daily_loss_cap=1 …")
    v133_cap1_rows = _run_all_periods(
        min_wyckoff_score_short = 6.5,
        max_daily_losses        = 1,
    )
    _summary_table("V133 (wy_short≥6.5) + daily_loss_cap=1", v133_cap1_rows)

    print("\n  Running V96+H4 (lb=3) + daily_loss_cap=1 …")
    v96_h4_cap1_rows = _run_all_periods(
        use_h4_trend_filter = True,
        h4_trend_slope_lb   = 3,
        max_daily_losses    = 1,
    )
    _summary_table("V96+H4 (lb=3) + daily_loss_cap=1", v96_h4_cap1_rows)

    print("\n  Running V96 Long-Only + daily_loss_cap=1 …")
    v96_long_cap1_rows = _run_all_periods(
        min_wyckoff_score_short = 99.0,
        max_daily_losses        = 1,
    )
    _summary_table("V96 Long-Only + daily_loss_cap=1", v96_long_cap1_rows)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4 — Head-to-head comparison
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 100}")
    print("  SECTION 4 — Head-to-head comparison (all variants)")
    print(f"{'─' * 100}")

    def _total_r(rows):   return sum(m["total_r"] for _, m, n in rows if n > 0)
    def _min_r(rows):     return min((m["total_r"] for _, m, n in rows if n > 0), default=float("-inf"))
    def _overall_wr(rows):
        w = sum(m["n_wins"] for _, m, _ in rows)
        l = sum(m["n_losses"] for _, m, _ in rows)
        return w / (w + l) * 100 if (w + l) else 0.0
    def _max_dd(rows):    return max(m["max_dd"] for _, m, _ in rows)
    def _total_sig(rows): return sum(n for _, _, n in rows)

    all_candidates = [
        # (name, rows)
        ("V96 baseline               ", v96_rows),
        ("V96 + H4 (lb=3)            ", v96_h4_rows),
        ("V96 + H4 (lb=2)            ", v96_h4_lb2_rows),
        ("V96 Long-Only + H4 (lb=3)  ", v96_long_h4_rows),
        ("V133 wy_short≥6.5          ", v133_rows),
        ("V134 wy_short≥7.0          ", v134_rows),
        ("V135 wy≥6.5+H4             ", v135_rows),
        ("V96 + cap=1                ", v96_cap1_rows),
        ("V133 wy≥6.5 + cap=1        ", v133_cap1_rows),
        ("V96+H4(lb=3) + cap=1       ", v96_h4_cap1_rows),
        ("V96 Long-Only + cap=1      ", v96_long_cap1_rows),
    ]

    print(f"\n  {'Config':<30}  {'Sig':>4}  {'TotR':>7}  {'MinR':>7}  {'OvWR':>5}  {'MaxDD':>6}")
    print("  " + "─" * 70)
    for name, rows in all_candidates:
        tr  = _total_r(rows)
        mr  = _min_r(rows)
        wr  = _overall_wr(rows)
        dd  = _max_dd(rows)
        sig = _total_sig(rows)
        ok  = mr > 0 and wr >= 60
        flag = " ←" if ok else ("  " if mr > 0 else " !")
        print(f"  {name:<30}  {sig:>4}  {tr:>+7.2f}  {mr:>+7.2f}  {wr:>5.1f}%  {dd:>5.1f}%{flag}")

    print(f"\n  ← PASS: WR ≥ 60% combined + R > 0 every period")
    print(f"  !  FAIL: negative R in at least one period\n")

    # Identify champion
    passing = [(n, rows) for n, rows in all_candidates if _min_r(rows) > 0 and _overall_wr(rows) >= 60]
    if passing:
        champion_name, champion_rows = max(passing, key=lambda x: _total_r(x[1]))
        print(f"  BEST PASSING CONFIG: {champion_name.strip()}")
        print(f"    TotR={_total_r(champion_rows):+.2f}  OvWR={_overall_wr(champion_rows):.1f}%  MaxDD={_max_dd(champion_rows):.1f}%")
    else:
        best_r = max(all_candidates, key=lambda x: _total_r(x[1]))
        print(f"  No config passes strict criteria. Best total R: {best_r[0].strip()}")
        print(f"    TotR={_total_r(best_r[1]):+.2f}  MinR={_min_r(best_r[1]):+.2f}  MaxDD={_max_dd(best_r[1]):.1f}%")

    print(f"\n{'=' * 100}")
    print("  Fin du Round 21")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    main()
