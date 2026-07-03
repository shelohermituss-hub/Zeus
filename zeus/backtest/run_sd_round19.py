"""
S&D Strategy — Round 19: Validating No-Trend-Filter + Trend Filter Ablation
=============================================================================

Key finding from Round 18:
  V119 (no trend filter): 32 sig / 63.2% WR / +19.00R on 2026 Q2 IS
  vs V96 (with trend filter): 7 sig / 71.4% WR / +5.50R

This round:
  1. Multi-year validation of V119 (no trend filter) across all 4 periods
     → confirm WR ≥ 55% and positive R across 2024 / 2025 / 2026
  2. Trend-filter ablation study: find the sweet spot between V96 and V119
     → slope only (no price_above_ema)
     → shorter lookback (lb=3 vs lb=6)
     → shorter lookback + no price_above_ema
     → asymmetric: trend filter for demand only (not supply)
     → separate demand vs supply signal counts (diagnostic)
  3. Combine V119 with TP1@1.0R to improve WR

Usage
-----
    python -m zeus.backtest.run_sd_round19
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

_ROOT   = Path(__file__).parent.parent.parent
_M1     = _ROOT / "data" / "historical" / "xauusd" / "m1"

INITIAL_BALANCE = 10_000.0
RISK_PCT        = 0.01
SPREAD          = 0.30

# ── V96 base params ───────────────────────────────────────────────────────────
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


def _build(**overrides) -> SDStrategy:
    p = {**_V96, **overrides}
    return SDStrategy(
        zone_detector    = ZoneDetector(),
        wyckoff_detector = WyckoffDetector(**_WY),
        **p
    )


def _run(
    m1_df: pd.DataFrame,
    tp1_r: float = 0.0,
    tp1_size: float = 0.5,
    **kw,
) -> tuple[list[TradeResult], dict[str, Any], int]:
    zone_df = resample_ohlcv(m1_df, "15min")
    strat   = _build(**kw)
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


def _row(label: str, m: dict, n: int) -> None:
    closed = m["n_wins"] + m["n_losses"]
    exp    = m.get("n_expired", 0)
    print(
        f"  {label:<55}  sig={n:>4}  "
        f"W={m['n_wins']:>3} L={m['n_losses']:>3}  "
        f"WR={m['win_rate']:>5.1f}%  "
        f"R={m['total_r']:>+8.2f}  "
        f"P&L={m['total_usd']:>+8.0f}$  "
        f"DD={m['max_dd']:>4.1f}%  "
        f"exp={exp:>3}"
    )


def main() -> None:
    print("=" * 110)
    print("  S&D Strategy — Round 19 · XAUUSD")
    print("=" * 110)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1 — Multi-year validation: V96 (trend) vs V119 (no trend)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n  SECTION 1 — V96 vs V119 across all data periods")
    print(f"{'─' * 110}")

    v96_rows: list[tuple]  = []
    v119_rows: list[tuple] = []
    m1_is = None

    for period_label, files in PERIODS:
        m1 = _load(files)
        d0, d1 = m1.index[0].date(), m1.index[-1].date()
        print(f"\n  ── {period_label}  ({len(m1):,} bars, {d0} → {d1}) ──")

        print("     V96  (trend+price_ema) …", end="", flush=True)
        r96, m96, n96 = _run(m1)
        print(f" {n96} sig")
        _row("V96  " + period_label, m96, n96)

        print("     V119 (no trend filter) …", end="", flush=True)
        r119, m119, n119 = _run(m1, use_trend_filter=False)
        print(f" {n119} sig")
        _row("V119 " + period_label, m119, n119)

        v96_rows.append((period_label, m96, n96))
        v119_rows.append((period_label, m119, n119))

        if "2026 Apr" in period_label:
            m1_is = m1
            print("\n     Monthly detail for V119 on IS period:")
            if r119:
                print_monthly_breakdown(r119, INITIAL_BALANCE)

    # — Section 1 summary —
    print(f"\n{'─' * 110}")
    print("  SECTION 1 SUMMARY")
    print(f"\n  {'Period':<30}  {'V96 WR%':>7}  {'V96 R':>8}  {'V96 sig':>7}  "
          f"{'V119 WR%':>8}  {'V119 R':>8}  {'V119 sig':>8}")
    print("  " + "─" * 90)
    for (lbl, m96, n96), (_, m119, n119) in zip(v96_rows, v119_rows):
        print(
            f"  {lbl:<30}  "
            f"{m96['win_rate']:>7.1f}%  {m96['total_r']:>+8.2f}  {n96:>7}  "
            f"{m119['win_rate']:>8.1f}%  {m119['total_r']:>+8.2f}  {n119:>8}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2 — Trend filter ablation on 2026 Q2 IS + all periods
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 110}")
    print("  SECTION 2 — Trend filter ablation study (2026 Apr–Jun IS)")
    print(f"{'─' * 110}\n")

    ablation_variants = [
        # (label, strategy overrides, tp1_r, tp1_size)
        ("V96  · full trend filter (slope_lb=6 + price_ema)",
            {},                                                         0.0, 0.5),
        ("V119 · no trend filter",
            dict(use_trend_filter=False),                               0.0, 0.5),
        # — Slope only, no price position —
        ("V122 · slope only lb=6 (no price_above_ema)",
            dict(use_price_above_ema=False),                            0.0, 0.5),
        # — Shorter slope lookback —
        ("V123 · slope lb=3 + price_ema",
            dict(trend_slope_lookback=3),                               0.0, 0.5),
        ("V124 · slope lb=3 no price_ema",
            dict(trend_slope_lookback=3, use_price_above_ema=False),    0.0, 0.5),
        # — Asymmetric: apply trend filter only for shorts (supply zones)
        #   demand zones always allowed, supply zones need bearish trend
        ("V125 · trend filter demand-only suppressed (supply free)",
            dict(min_zone_score_long=5.0, min_zone_score_short=5.0,
                 use_trend_filter=False),                                0.0, 0.5),
        # — Asymmetric: apply only slope (no price check) with shorter lb
        ("V126 · slope lb=4 no price_ema",
            dict(trend_slope_lookback=4, use_price_above_ema=False),    0.0, 0.5),
        # — V119 + TP1 —
        ("V127 · no trend + TP1@1.0R 50% → SL→BE",
            dict(use_trend_filter=False),                               1.0, 0.5),
        # — V119 + tighter Wyckoff (keep quality, get more signals) —
        ("V128 · no trend + wy≥6.0 (tighter quality)",
            dict(use_trend_filter=False, min_wyckoff_score=6.0),        0.0, 0.5),
        # — V119 + daily loss cap relaxed
        ("V129 · no trend + mloss6 (from 4)",
            dict(use_trend_filter=False),                               0.0, 0.5),
    ]

    ablation_results: list[tuple] = []
    for label, kw, tp1_r, tp1_size in ablation_variants:
        # V129: max_monthly_losses=6
        sim_kw = {}
        if "mloss6" in label:
            sim_kw["max_monthly_losses"] = 6

        print(f"  [{label}] …", end="", flush=True)
        # For V129 we need custom simulate_all — run manually
        if "mloss6" in label:
            zone_df = resample_ohlcv(m1_is, "15min")
            strat   = _build(**kw)
            signals = strat.run(zone_df, m1_is)
            res, n_exp = simulate_all(
                signals, m1_is,
                risk_pct=RISK_PCT, spread=SPREAD,
                max_monthly_losses=6,
                tp1_r=tp1_r, tp1_size=tp1_size,
            )
            m = compute_metrics(res, INITIAL_BALANCE, len(signals), n_exp)
            r, n = res, len(signals)
        else:
            r, m, n = _run(m1_is, tp1_r=tp1_r, tp1_size=tp1_size, **kw)
        ablation_results.append((label, m, n))
        print(f" {n} sig")

    print(f"\n{'─' * 110}")
    print("  TABLEAU ABLATION — 2026 Apr–Jun IS")
    print(f"{'─' * 110}")
    for label, m, n in ablation_results:
        _row(label, m, n)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3 — Best ablation variant validated across all periods
    # ─────────────────────────────────────────────────────────────────────────
    # Choose best ablation variant: highest R on IS with WR ≥ 55% and ≥ 10 signals
    candidates = [
        (l, m, n, kw, tp1_r, tp1_size)
        for (l, m, n), (_, kw, tp1_r, tp1_size)
        in zip(ablation_results, ablation_variants)
        if n >= 10 and m["win_rate"] >= 55.0 and m["total_r"] > 0
    ]

    if not candidates:
        print("\n  Aucune variante d'ablation ne satisfait tous les critères sur IS.")
        print("  Vérification de V119 sur tous les périodes en Section 3 quand même.")
        best_label = "V119 · no trend filter"
        best_kw    = dict(use_trend_filter=False)
        best_tp1_r = 0.0
        best_tp1_s = 0.5
    else:
        best = max(candidates, key=lambda x: x[1]["total_r"])
        best_label, _, _, best_kw, best_tp1_r, best_tp1_s = best
        print(f"\n  MEILLEURE VARIANTE D'ABLATION: {best_label}")

    print(f"\n\n{'=' * 110}")
    print(f"  SECTION 3 — {best_label} · validation multi-périodes")
    print(f"{'─' * 110}")

    best_rows: list[tuple] = []
    for period_label, files in PERIODS:
        m1 = _load(files)
        print(f"  {period_label} …", end="", flush=True)
        if "mloss6" in best_label:
            zone_df = resample_ohlcv(m1, "15min")
            strat   = _build(**best_kw)
            signals = strat.run(zone_df, m1)
            res, n_exp = simulate_all(
                signals, m1,
                risk_pct=RISK_PCT, spread=SPREAD,
                max_monthly_losses=6,
                tp1_r=best_tp1_r, tp1_size=best_tp1_s,
            )
            m_ = compute_metrics(res, INITIAL_BALANCE, len(signals), n_exp)
            n_ = len(signals)
        else:
            _, m_, n_ = _run(m1, tp1_r=best_tp1_r, tp1_size=best_tp1_s, **best_kw)
        print(f" {n_} sig")
        _row(period_label, m_, n_)
        best_rows.append((period_label, m_, n_))

    print(f"\n{'─' * 110}")
    print(f"  RÉSUMÉ {best_label}")
    print(f"  {'Period':<32}  {'Sig':>5}  {'W':>4} {'L':>4}  {'WR%':>6}  {'R':>8}  {'P&L$':>9}  {'DD%':>6}")
    print("  " + "─" * 85)
    total_w = total_l = total_n = 0
    for lbl, m, n in best_rows:
        print(
            f"  {lbl:<32}  {n:>5}  "
            f"{m['n_wins']:>4} {m['n_losses']:>4}  "
            f"{m['win_rate']:>6.1f}%  "
            f"{m['total_r']:>+8.2f}  "
            f"{m['total_usd']:>+9.0f}$  "
            f"{m['max_dd']:>5.1f}%"
        )
        total_w += m["n_wins"]
        total_l += m["n_losses"]
        total_n += n
    decided  = total_w + total_l
    ov_wr    = total_w / decided * 100 if decided else 0.0
    print("  " + "─" * 85)
    print(f"  {'TOTAL':<32}  {total_n:>5}  {total_w:>4} {total_l:>4}  {ov_wr:>6.1f}%")

    verdict = (
        "PASS ✓ — variant robuste sur toutes les périodes"
        if ov_wr >= 55 and all(m["total_r"] > 0 for _, m, n in best_rows if n > 0)
        else "REVIEW — certaines périodes sous le seuil"
    )
    print(f"\n  VERDICT: {verdict}")

    print(f"\n{'=' * 110}")
    print("  Fin du Round 19")
    print(f"{'=' * 110}\n")


if __name__ == "__main__":
    main()
