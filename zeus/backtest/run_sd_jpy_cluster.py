"""
S&D Strategy — JPY Cluster Backtest
======================================

Tests GBPJPY, USDJPY, EURJPY on the TIERED-5R exit structure.

Protocol : optimise on 2024 → cross-validate on 2025 (partial year)

JPY specifics
-------------
  pip_size = 0.01  (1 pip = 0.01 JPY, vs 0.0001 for majors)
  spread   : GBPJPY 3p / USDJPY 1p / EURJPY 2p  (in price units)
  q2u      = 1.0   (pip_size 100× larger compensates JPY price scale;
                    R metrics are directly comparable to the forex cluster)

Note: USD P&L amounts are approximations; focus on WR%, TotalR, DD% in R.

Usage
-----
    python -m zeus.backtest.run_sd_jpy_cluster
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
    pip_size                = 0.01,    # JPY pip size
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

# spread in price units (pips × pip_size) | q2u=1.0 for all JPY pairs
JPY_CLUSTER_2024 = [
    ("GBPJPY", 0.03, 1.0, (7, 17), [_DATA / "gbpjpy" / "m1" / "DAT_MT_GBPJPY_M1_2024.csv"]),
    ("USDJPY", 0.01, 1.0, (7, 17), [_DATA / "usdjpy" / "m1" / "DAT_MT_USDJPY_M1_2024.csv"]),
    ("EURJPY", 0.02, 1.0, (7, 17), [_DATA / "eurjpy" / "m1" / "DAT_MT_EURJPY_M1_2024.csv"]),
]

JPY_CLUSTER_2025 = [
    ("GBPJPY", 0.03, 1.0, (7, 17), [_DATA / "gbpjpy" / "m1" / "DAT_MT_GBPJPY_M1_2025.csv"]),
    ("USDJPY", 0.01, 1.0, (7, 17), [_DATA / "usdjpy" / "m1" / "DAT_MT_USDJPY_M1_2025.csv"]),
    ("EURJPY", 0.02, 1.0, (7, 17), [_DATA / "eurjpy" / "m1" / "DAT_MT_EURJPY_M1_2025.csv"]),
]


def _build_strategy(session: tuple[int, int]) -> SDStrategy:
    return SDStrategy(
        zone_detector     = ZoneDetector(),
        wyckoff_detector  = WyckoffDetector(**_BASE_WYCKOFF),
        session_start_utc = session[0],
        session_end_utc   = session[1],
        **_BASE_STRATEGY,
    )


def _run_combined(pairs: list, label: str) -> dict:
    runner_rr      = TIERED5R["runner_rr"]
    risk_pct       = TIERED5R["risk_pct"]
    tp1_r          = TIERED5R["tp1_r"]
    tp1_size       = TIERED5R["tp1_size"]
    tp2_r          = TIERED5R["tp2_r"]
    tp2_cumulative = TIERED5R["tp2_cumulative"]

    all_results: list = []
    all_signals  = 0
    n_expired    = 0
    equity       = INITIAL_BALANCE
    sl_pips_all: list[float] = []
    per_pair: list[dict] = []

    for sym, spread, q2u, session, files in pairs:
        frames = [parse_histdata_csv(f) for f in files if f.exists()]
        if not frames:
            continue
        m1_df  = pd.concat(frames).sort_index()
        m1_df  = m1_df[~m1_df.index.duplicated(keep="first")]
        m15_df = resample_ohlcv(m1_df, "15min")
        if len(m15_df) < 50:
            continue

        strategy = _build_strategy(session)
        signals  = strategy.run(m15_df, m1_df)
        all_signals += len(signals)
        if not signals:
            per_pair.append({"sym": sym, "n": 0, "wr": 0, "tr": 0})
            continue

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

        wins_pair = [r for r in results if r.outcome == "win"]
        n_pair    = len([r for r in results if r.outcome in ("win", "loss")])
        wr_pair   = len(wins_pair) / n_pair * 100 if n_pair else 0
        tr_pair   = sum(r.pnl_r for r in results)
        per_pair.append({"sym": sym, "n": len(results), "wr": wr_pair, "tr": tr_pair})

        for r in results:
            equity += r.pnl_usd

    if not all_results:
        return {"n_trades": 0, "win_rate": 0.0, "total_r": 0.0,
                "max_dd": 0.0, "avg_win_rr": 0.0, "avg_sl_pips": 0.0,
                "per_pair": per_pair}

    m = compute_metrics(all_results, INITIAL_BALANCE, all_signals, n_expired)
    wins = [r for r in all_results if r.outcome == "win"]
    m["avg_win_rr"] = sum(r.pnl_r for r in wins) / len(wins) if wins else 0.0
    m["avg_sl_pips"] = sum(sl_pips_all) / len(sl_pips_all) if sl_pips_all else 0.0
    m["per_pair"]   = per_pair
    return m


def _pass(m: dict) -> bool:
    return (m.get("win_rate", 0.0) >= 55.0
            and m.get("total_r",  0.0) > 0
            and m.get("max_dd",   0.0) <= 8.0)


def main() -> None:
    hdr = (f"  {'Année':<8} {'N':>4}  {'WR%':>6}  {'TotalR':>8}  "
           f"{'DD%':>5}  {'AvgWR':>7}  {'AvgSL':>6}")
    sep = "─" * len(hdr)

    print("\n" + "=" * 60)
    print("  JPY Cluster — TIERED-5R exit | 0.5% risk | pip=0.01")
    print("=" * 60)

    results: list[tuple[str, dict]] = []
    for year_label, cluster in [("2024 (opt)", JPY_CLUSTER_2024),
                                 ("2025 (OOS)", JPY_CLUSTER_2025)]:
        m = _run_combined(cluster, year_label)
        results.append((year_label, m))

    print(hdr)
    print(sep)
    for year_label, m in results:
        ok = "✅" if _pass(m) else "❌"
        print(f"  {year_label:<8} {m['n_trades']:>4}  {m['win_rate']:>6.1f}  "
              f"{m['total_r']:>8.2f}  {m['max_dd']:>5.1f}  "
              f"{m['avg_win_rr']:>7.2f}  {m['avg_sl_pips']:>5.1f}p  {ok}")

    # Per-pair breakdown for 2024
    print()
    print("  Détail par paire — 2024 :")
    print(f"  {'Paire':<8} {'N':>4}  {'WR%':>6}  {'TotalR':>8}")
    print("  " + "─" * 30)
    for pp in results[0][1].get("per_pair", []):
        n_wl = pp["n"]
        print(f"  {pp['sym']:<8} {n_wl:>4}  {pp['wr']:>6.1f}  {pp['tr']:>8.2f}")

    print()
    print("  Détail par paire — 2025 :")
    print(f"  {'Paire':<8} {'N':>4}  {'WR%':>6}  {'TotalR':>8}")
    print("  " + "─" * 30)
    for pp in results[1][1].get("per_pair", []):
        print(f"  {pp['sym']:<8} {pp['n']:>4}  {pp['wr']:>6.1f}  {pp['tr']:>8.2f}")

    print()

    # Comparison reference
    print("  === Référence cluster forex (même exit) ===")
    print("  2024 : N=159  WR=60.4%  TotalR=88.92R  DD=4.3%")
    print("  2023 : N=155  WR=55.5%  TotalR=83.12R  DD=5.4%")
    print()
    print("  Exit : TP1@1.5R(33%) → TP2@5R(70%) → Runner@10R | 0.5% risk")
    print("  Note : montants USD approximatifs — se fier aux métriques R")
    print()


if __name__ == "__main__":
    main()
