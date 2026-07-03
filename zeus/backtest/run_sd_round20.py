"""
S&D Strategy — Round 20: V123 Validation + BE Management + Long-Only Variant
==============================================================================

Following Round 19, V96 is confirmed as the champion (WR 52–71%, always positive R,
max DD ≤ 4.1% on the IS period).  Round 20 pursues three targeted improvements:

Section 1 — V123 multi-year validation
    V123 (trend_slope_lookback=3 vs V96's 6) showed 8 sig / 75.0% / +7.00R on IS —
    marginally better than V96's 7 / 71.4% / +5.50R.
    Question: does it hold on 2024 / 2025 / OOS?  If WR and R stay ≥ V96 in every
    period, V123 becomes the new champion.

Section 2 — Break-even management
    The simulation already supports use_be=True (SL moves to entry price once price
    hits +1R).  Applying this to V96 across all periods should reduce max DD without
    meaningfully hurting R:R (BE trades become "scratch" instead of losses).
    Goal: lower peak drawdown while keeping total R within 0.5R of baseline.

Section 3 — Long-only variant
    Gold is in a structural uptrend; supply-zone short trades may underperform demand-
    zone longs.  Blocking shorts (min_wyckoff_score_short=99.0) isolates only demand
    signals.  We test V96-LONG and V123-LONG across all four periods.
    Goal: WR improvement + DD reduction, even if total R is slightly lower (fewer trades).

Section 4 — Combined candidate
    If any long-only+BE or V123+BE combination consistently outperforms V96 on R and
    DD across all periods, it becomes the Round 20 champion.

Usage
-----
    python -m zeus.backtest.run_sd_round20
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
    m1_df:    pd.DataFrame,
    use_be:   bool  = False,
    tp1_r:    float = 0.0,
    tp1_size: float = 0.5,
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
        use_be             = use_be,
        tp1_r              = tp1_r,
        tp1_size           = tp1_size,
    )
    m = compute_metrics(res, INITIAL_BALANCE, len(signals), n_exp)
    return res, m, len(signals)


def _header_line(label: str, m: dict, n_sig: int, use_be: bool = False) -> None:
    be_tag = " [BE]" if use_be else "     "
    print(
        f"  {label:<50}{be_tag}  "
        f"sig={n_sig:>4}  "
        f"W={m['n_wins']:>3} L={m['n_losses']:>3}  "
        f"WR={m['win_rate']:>5.1f}%  "
        f"R={m['total_r']:>+7.2f}  "
        f"P&L={m['total_usd']:>+8.0f}$  "
        f"DD={m['max_dd']:>4.1f}%"
    )


def _run_all_periods(
    label:    str,
    use_be:   bool  = False,
    verbose:  bool  = False,
    **strat_kw,
) -> list[tuple[str, dict, int]]:
    rows: list[tuple[str, dict, int]] = []
    for period_label, files in PERIODS:
        m1 = _load(files)
        res, m, n_sig = _run(m1, use_be=use_be, **strat_kw)
        rows.append((period_label, m, n_sig))
        if verbose:
            _header_line(f"  {period_label}", m, n_sig, use_be)
            if res:
                print_monthly_breakdown(res, INITIAL_BALANCE)
    return rows


def _summary_table(title: str, rows: list[tuple[str, dict, int]], use_be: bool = False) -> None:
    be_tag = " +BE" if use_be else "    "
    print(f"\n  {title}{be_tag}")
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 100)
    print("  S&D Strategy — Round 20 · XAUUSD")
    print("  V123 validation + Break-Even management + Long-Only variant")
    print("=" * 100)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1 — V96 baseline recap + V123 (slope_lb=3) multi-year validation
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'─' * 100}")
    print("  SECTION 1 — V96 vs V123 (trend_slope_lookback=3) across all periods")
    print("  V123 from Round 19: 8 sig / 75.0% WR / +7.00R on IS vs V96 7 / 71.4% / +5.50R")
    print(f"{'─' * 100}")

    print("\n  Running V96 baseline …")
    v96_rows = _run_all_periods("V96 baseline")
    _summary_table("V96 — champion baseline (zone≥5.0, wy≥5.9, slope_lb=6)", v96_rows)

    print("\n  Running V123 (slope_lb=3) …")
    v123_rows = _run_all_periods("V123", trend_slope_lookback=3)
    _summary_table("V123 — slope_lb=3 (all else = V96)", v123_rows)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2 — Break-even management on V96 and V123
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 100}")
    print("  SECTION 2 — Break-Even (BE) at +1R on V96 and V123")
    print("  BE moves SL to entry once price reaches +1R; scratch instead of loss if reversed.")
    print("  Hypothesis: lowers max DD, comparable R to V96 baseline.")
    print(f"{'─' * 100}")

    print("\n  Running V96 + BE …")
    v96_be_rows = _run_all_periods("V96+BE", use_be=True)
    _summary_table("V96 + break-even", v96_be_rows, use_be=True)

    print("\n  Running V123 + BE …")
    v123_be_rows = _run_all_periods("V123+BE", use_be=True, trend_slope_lookback=3)
    _summary_table("V123 + break-even", v123_be_rows, use_be=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3 — Long-only variants (demand zones only, shorts blocked)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 100}")
    print("  SECTION 3 — Long-Only variants (min_wyckoff_score_short=99.0 blocks all shorts)")
    print("  Gold in structural uptrend → demand zones may be more reliable than supply zones.")
    print(f"{'─' * 100}")

    print("\n  Running V96-LONG …")
    v96_long_rows = _run_all_periods("V96-LONG", min_wyckoff_score_short=99.0)
    _summary_table("V96 Long-Only", v96_long_rows)

    print("\n  Running V123-LONG …")
    v123_long_rows = _run_all_periods("V123-LONG", min_wyckoff_score_short=99.0,
                                      trend_slope_lookback=3)
    _summary_table("V123 Long-Only", v123_long_rows)

    print("\n  Running V96-LONG + BE …")
    v96_long_be_rows = _run_all_periods("V96-LONG+BE", use_be=True,
                                        min_wyckoff_score_short=99.0)
    _summary_table("V96 Long-Only + BE", v96_long_be_rows, use_be=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4 — Daily loss limit (reduce consecutive loss streaks)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 100}")
    print("  SECTION 4 — Daily loss limit on V96 (stop trading after N losses/day)")
    print("  V96 occasionally has 2 losses in a single day; cap at 1 or 2 to limit streaks.")
    print(f"{'─' * 100}")

    def _run_with_daily_cap(m1_df, daily_cap, **strat_kw):
        zone_df = resample_ohlcv(m1_df, "15min")
        strat   = _build_strategy(**strat_kw)
        signals = strat.run(zone_df, m1_df)
        res, n_exp = simulate_all(
            signals, m1_df,
            risk_pct           = RISK_PCT,
            spread             = SPREAD,
            max_monthly_losses = 4,
            max_daily_losses   = daily_cap,
        )
        m = compute_metrics(res, INITIAL_BALANCE, len(signals), n_exp)
        return res, m, len(signals)

    for cap in (1, 2):
        cap_rows: list[tuple[str, dict, int]] = []
        print(f"\n  Running V96 daily_loss_cap={cap} …")
        for period_label, files in PERIODS:
            m1 = _load(files)
            res, m, n_sig = _run_with_daily_cap(m1, cap)
            cap_rows.append((period_label, m, n_sig))
        _summary_table(f"V96 daily_loss_cap={cap}", cap_rows)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5 — Champion comparison summary
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 100}")
    print("  SECTION 5 — Head-to-head comparison (best per period)")
    print(f"{'─' * 100}")

    def _total_r(rows):
        return sum(m["total_r"] for _, m, n in rows if n > 0)

    def _min_period_r(rows):
        return min((m["total_r"] for _, m, n in rows if n > 0), default=float("-inf"))

    def _overall_wr(rows):
        w = sum(m["n_wins"] for _, m, _ in rows)
        l = sum(m["n_losses"] for _, m, _ in rows)
        return w / (w + l) * 100 if (w + l) else 0.0

    def _max_dd(rows):
        return max(m["max_dd"] for _, m, _ in rows)

    candidates = [
        ("V96 baseline       ", v96_rows,      False),
        ("V123 slope_lb=3    ", v123_rows,      False),
        ("V96 + BE           ", v96_be_rows,    True),
        ("V123 + BE          ", v123_be_rows,   True),
        ("V96 Long-Only      ", v96_long_rows,  False),
        ("V123 Long-Only     ", v123_long_rows, False),
        ("V96 Long-Only + BE ", v96_long_be_rows, True),
    ]

    print(f"\n  {'Config':<25}  {'TotR':>7}  {'MinR':>7}  {'OvWR':>5}  {'MaxDD':>6}")
    print("  " + "─" * 60)
    for name, rows, use_be in candidates:
        tr   = _total_r(rows)
        mr   = _min_period_r(rows)
        wr   = _overall_wr(rows)
        dd   = _max_dd(rows)
        be   = " BE" if use_be else "   "
        flag = " ←" if mr > 0 and wr >= 60 else ""
        print(f"  {name:<25}{be}  {tr:>+7.2f}  {mr:>+7.2f}  {wr:>5.1f}%  {dd:>5.1f}%{flag}")

    print(f"\n  ← marks configs where WR ≥ 60% combined AND R > 0 in every period")
    print(f"\n{'=' * 100}")
    print("  Fin du Round 20")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    main()
