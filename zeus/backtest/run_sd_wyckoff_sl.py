"""
S&D Strategy — Zone SL vs Wyckoff M1 SL comparison
=====================================================

Compares two SL anchoring methods on the TIERED-5R exit structure:

  ZONE-SL   : SL = zone.wick_extreme  (M15 zone manipulation wick) — current default
  WYCKOFF-SL: SL = wyckoff.manip_extreme (M1 spring/upthrust wick) — tighter

Exit fixed at TIERED-5R: TP1@1.5R(33%) → TP2@5R(70%) → Runner@10R | 0.5% risk

Also reports average SL distance in pips per pair so we can see the difference.

Usage
-----
    python -m zeus.backtest.run_sd_wyckoff_sl
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv, resample_ohlcv
from zeus.backtest.sd_simulation import compute_metrics, simulate_all
from zeus.strategy.supply_demand.sd_strategy import SDStrategy
from zeus.strategy.supply_demand.wyckoff import WyckoffDetector
from zeus.strategy.supply_demand.zone_detector import ZoneDetector

_ROOT = Path(__file__).parent.parent.parent
_DATA = _ROOT / "data" / "historical"

INITIAL_BALANCE = 10_000.0

_BASE_STRATEGY = dict(
    risk_reward             = 1.25,
    min_zone_score          = 4.0,
    min_wyckoff_score       = 7.0,
    min_wyckoff_score_short = 5.9,
    min_composite_score     = 4.0,
    signal_cooldown         = 10,
    use_trend_filter        = True,
    trend_slope_lookback    = 3,
    use_price_above_ema     = True,
    ema_atr_tolerance       = 0.5,
    use_session_filter      = True,
    max_signals_per_day     = 6,
    use_adx_filter          = False,
    use_rsi_filter          = False,
    min_sl_pips             = 5,
    pip_size                = 0.0001,
    min_score_product       = 0.0,
    use_h4_trend_filter     = False,
    use_first_touch_only    = False,
)

_BASE_WYCKOFF = dict(
    lookback             = 200,
    max_accum_bars       = 20,
    accum_range_mult     = 6.0,
    mss_lookback         = 60,
    min_spring_sweep_pct = 0.10,
    min_mss_strength_pct = 0.03,
)

TIERED5R = dict(
    risk_pct       = 0.005,
    tp1_r          = 1.5,
    tp1_size       = 0.33,
    tp2_r          = 5.0,
    tp2_cumulative = 0.70,
    runner_rr      = 10.0,
)

PAPER_CLUSTER_2024 = [
    ("GBPUSD", 0.0001, 1.00, (7, 17), [_DATA / "gbpusd" / "m1" / "DAT_MT_GBPUSD_M1_2024.csv"]),
    ("EURUSD", 0.0001, 1.00, (7, 17), [_DATA / "eurusd" / "m1" / "DAT_MT_EURUSD_M1_2024.csv"]),
    ("GBPAUD", 0.0002, 1.52, (7, 17), [_DATA / "gbpaud" / "m1" / "DAT_MT_GBPAUD_M1_2024.csv"]),
    ("EURNZD", 0.0002, 1.62, (7, 17), [_DATA / "eurnzd" / "m1" / "DAT_MT_EURNZD_M1_2024.csv"]),
]

PAPER_CLUSTER_2023 = [
    ("GBPUSD", 0.0001, 1.00, (7, 17), [_DATA / "gbpusd" / "m1" / "DAT_MT_GBPUSD_M1_2023.csv"]),
    ("EURUSD", 0.0001, 1.00, (7, 17), [_DATA / "eurusd" / "m1" / "DAT_MT_EURUSD_M1_2023.csv"]),
    ("GBPAUD", 0.0002, 1.52, (7, 17), [_DATA / "gbpaud" / "m1" / "DAT_MT_GBPAUD_M1_2023.csv"]),
    ("EURNZD", 0.0002, 1.62, (7, 17), [_DATA / "eurnzd" / "m1" / "DAT_MT_EURNZD_M1_2023.csv"]),
]


def _run_combined(pairs: list, use_wyckoff_sl: bool) -> dict:
    risk_pct       = TIERED5R["risk_pct"]
    tp1_r          = TIERED5R["tp1_r"]
    tp1_size       = TIERED5R["tp1_size"]
    tp2_r          = TIERED5R["tp2_r"]
    tp2_cumulative = TIERED5R["tp2_cumulative"]
    runner_rr      = TIERED5R["runner_rr"]

    all_results: list = []
    all_signals  = 0
    n_expired    = 0
    equity       = INITIAL_BALANCE
    sl_pips_all: list[float] = []

    for sym, spread, q2u, session, files in pairs:
        frames = [parse_histdata_csv(f) for f in files if f.exists()]
        if not frames:
            continue
        m1_df  = pd.concat(frames).sort_index()
        m1_df  = m1_df[~m1_df.index.duplicated(keep="first")]
        m15_df = resample_ohlcv(m1_df, "15min")
        if len(m15_df) < 50:
            continue

        strategy = SDStrategy(
            zone_detector     = ZoneDetector(),
            wyckoff_detector  = WyckoffDetector(**_BASE_WYCKOFF),
            session_start_utc = session[0],
            session_end_utc   = session[1],
            use_wyckoff_sl    = use_wyckoff_sl,
            **_BASE_STRATEGY,
        )
        signals = strategy.run(m15_df, m1_df)
        all_signals += len(signals)
        if not signals:
            continue

        # Collect SL distances in pips
        pip = _BASE_STRATEGY["pip_size"]
        for s in signals:
            sl_pips_all.append(abs(s.entry_price - s.stop_loss) / pip)

        signals = [dataclasses.replace(s, risk_reward=runner_rr) for s in signals]

        results, exp = simulate_all(
            signals,
            m1_df,
            risk_pct           = risk_pct,
            spread             = spread,
            max_daily_losses   = 1,
            max_monthly_losses = 4,
            tp1_r              = tp1_r,
            tp1_size           = tp1_size,
            tp2_r              = tp2_r,
            tp2_cumulative_pct = tp2_cumulative,
            initial_equity     = equity,
            quote_to_usd_rate  = q2u,
        )
        all_results.extend(results)
        n_expired += exp
        for r in results:
            equity += r.pnl_usd

    if not all_results:
        return {"n_trades": 0, "win_rate": 0.0, "total_r": 0.0,
                "max_dd": 0.0, "avg_win_rr": 0.0, "avg_sl_pips": 0.0}

    m = compute_metrics(all_results, INITIAL_BALANCE, all_signals, n_expired)
    wins = [r for r in all_results if r.outcome == "win"]
    m["avg_win_rr"] = sum(r.pnl_r for r in wins) / len(wins) if wins else 0.0
    m["avg_sl_pips"] = sum(sl_pips_all) / len(sl_pips_all) if sl_pips_all else 0.0
    m["med_sl_pips"] = float(sorted(sl_pips_all)[len(sl_pips_all) // 2]) if sl_pips_all else 0.0
    return m


def _pass(m: dict) -> bool:
    return (m.get("win_rate", 0.0) >= 55.0
            and m.get("total_r",  0.0) > 0
            and m.get("max_dd",   0.0) <= 8.0)


def _print_table(label: str, year: str, m: dict) -> None:
    ok = "✅" if _pass(m) else "❌"
    print(f"  {label:<14} {m['n_trades']:>4}  {m['win_rate']:>6.1f}%  "
          f"{m['total_r']:>8.2f}R  {m['max_dd']:>5.1f}%  "
          f"{m['avg_win_rr']:>7.2f}R  SL≈{m['avg_sl_pips']:>5.1f}p "
          f"(med {m['med_sl_pips']:.1f}p)  {ok}")


def main() -> None:
    configs = [
        ("ZONE-SL  (M15)", False),
        ("WYCKOFF-SL (M1)", True),
    ]

    hdr = (f"  {'Config':<14} {'N':>4}  {'WR%':>7}  {'TotalR':>9}  "
           f"{'DD%':>6}  {'AvgWinR':>8}  {'Avg SL':>15}")
    sep = "─" * len(hdr)

    for year_label, pairs in [("2024 (optimisation)", PAPER_CLUSTER_2024),
                               ("2023 (cross-validation)", PAPER_CLUSTER_2023)]:
        print(f"\n{'=' * 72}")
        print(f"  {year_label} — 4 pairs — TIERED-5R exit")
        print(f"{'=' * 72}")
        print(hdr)
        print(sep)
        for label, use_wsl in configs:
            m = _run_combined(pairs, use_wsl)
            _print_table(label, year_label, m)

    print()
    print("  ZONE-SL    : SL = zone.wick_extreme   (wick M15 de la zone S&D)")
    print("  WYCKOFF-SL : SL = wyckoff.manip_extreme (wick M1 du spring/upthrust)")
    print("  Exit fixe  : TP1@1.5R(33%) → TP2@5R(70%) → Runner@10R | 0.5% risk")
    print()


if __name__ == "__main__":
    main()
