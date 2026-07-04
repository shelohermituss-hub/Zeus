"""
S&D Strategy — max_sl_pips sweep
==================================

Fixes the TIERED-5R exit and sweeps a max_sl_pips ceiling to find the
optimal SL tightness. Signals whose zone wick exceeds max_sl_pips are
discarded — keeping only tight, high-quality zone setups.

Configs tested (2024 optimise → 2023 cross-validate):
  max_sl_pips = 0 (no cap), 15, 18, 20, 22, 25, 30 pips

Reports N, WR%, TotalR, DD%, AvgWinR, and actual SL distribution.

Usage
-----
    python -m zeus.backtest.run_sd_max_sl_sweep
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

MAX_SL_CONFIGS = [0, 15, 18, 20, 22, 25, 30]  # pips; 0 = no cap

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


def _run_combined(pairs: list, max_sl_pips: float) -> dict:
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
    sl_pips_list: list[float] = []

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
            max_sl_pips       = max_sl_pips,
            **_BASE_STRATEGY,
        )
        signals = strategy.run(m15_df, m1_df)
        all_signals += len(signals)
        if not signals:
            continue

        pip = _BASE_STRATEGY["pip_size"]
        for s in signals:
            sl_pips_list.append(abs(s.entry_price - s.stop_loss) / pip)

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
                "max_dd": 0.0, "avg_win_rr": 0.0,
                "avg_sl_pips": 0.0, "med_sl_pips": 0.0, "max_sl_pips_obs": 0.0}

    m = compute_metrics(all_results, INITIAL_BALANCE, all_signals, n_expired)
    wins = [r for r in all_results if r.outcome == "win"]
    m["avg_win_rr"]    = sum(r.pnl_r for r in wins) / len(wins) if wins else 0.0
    sl_sorted          = sorted(sl_pips_list)
    m["avg_sl_pips"]   = sum(sl_sorted) / len(sl_sorted) if sl_sorted else 0.0
    m["med_sl_pips"]   = sl_sorted[len(sl_sorted) // 2] if sl_sorted else 0.0
    m["max_sl_pips_obs"] = sl_sorted[-1] if sl_sorted else 0.0
    return m


def _pass(m: dict) -> bool:
    return (m.get("win_rate", 0.0) >= 55.0
            and m.get("total_r",  0.0) > 0
            and m.get("max_dd",   0.0) <= 8.0
            and m.get("n_trades", 0)   >= 30)


def main() -> None:
    hdr = (f"  {'max_SL':>6}  {'N':>4}  {'WR%':>6}  {'TotalR':>8}  "
           f"{'DD%':>5}  {'AvgWR':>7}  {'AvgSL':>6}  {'MedSL':>6}")
    sep = "─" * len(hdr)

    results_2024: list[tuple[float, dict]] = []

    print("\nmax_sl_pips sweep — TIERED-5R exit — 2024")
    print(sep)
    print(hdr)
    print(sep)
    for cap in MAX_SL_CONFIGS:
        label = f"{cap}p" if cap > 0 else "no cap"
        m = _run_combined(PAPER_CLUSTER_2024, cap)
        results_2024.append((cap, m))
        ok = "✅" if _pass(m) else "❌"
        print(f"  {label:>6}  {m['n_trades']:>4}  {m['win_rate']:>6.1f}  "
              f"{m['total_r']:>8.2f}  {m['max_dd']:>5.1f}  {m['avg_win_rr']:>7.2f}  "
              f"{m['avg_sl_pips']:>5.1f}p  {m['med_sl_pips']:>5.1f}p  {ok}")

    print(f"\n{'=' * 72}")
    print("  Cross-validation 2023 — mêmes configs")
    print(f"{'=' * 72}")
    print(hdr)
    print(sep)
    for cap, m24 in results_2024:
        label = f"{cap}p" if cap > 0 else "no cap"
        m = _run_combined(PAPER_CLUSTER_2023, cap)
        ok  = "✅" if _pass(m) else "❌"
        ret = m["total_r"] / m24["total_r"] if m24["total_r"] else 0.0
        print(f"  {label:>6}  {m['n_trades']:>4}  {m['win_rate']:>6.1f}  "
              f"{m['total_r']:>8.2f}  {m['max_dd']:>5.1f}  {m['avg_win_rr']:>7.2f}  "
              f"{m['avg_sl_pips']:>5.1f}p  {m['med_sl_pips']:>5.1f}p  "
              f"{ok}  ret={ret:.0%}")

    print()
    print("  max_SL = plafond de distance SL en pips (0 = pas de plafond)")
    print("  Les signaux dont le wick de zone > max_SL sont rejetés")
    print("  Exit: TP1@1.5R(33%) → TP2@5R(70%) → Runner@10R | 0.5% risk")
    print()


if __name__ == "__main__":
    main()
