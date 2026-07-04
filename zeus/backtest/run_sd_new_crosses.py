"""
S&D Strategy — New Crosses Backtest
=====================================

Tests GBPNZD, GBPCAD, EURAUD on the TIERED-5R exit structure.

Two configs compared in parallel:
  Baseline : min_wyckoff_score = 7.0 / 5.9
  Candidate: min_wyckoff_score = 7.5 / 6.5  (pending OOS validation)

Protocol:
  2024 optimisation → 2025 OOS cross-validation
  EURAUD: also tested on 2023 (extra year available)

Spreads (price units = pips × 0.0001):
  GBPNZD  4p = 0.0004  | q2u = 1.62  (1/NZDUSD)
  GBPCAD  3p = 0.0003  | q2u = 1.36  (1/CADUSD = USDCAD)
  EURAUD  3p = 0.0003  | q2u = 1.53  (1/AUDUSD)

Pass criteria: WR ≥ 55%, TotalR > 0, DD ≤ 8%, N ≥ 30

Usage
-----
    python -m zeus.backtest.run_sd_new_crosses
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

_BASE = dict(
    risk_reward             = 1.25,
    min_zone_score          = 4.0,
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

# Two configs to compare
CONFIGS = {
    "WS7.0 (baseline)":   dict(min_wyckoff_score=7.0, min_wyckoff_score_short=5.9),
    "WS7.5 (candidat)":   dict(min_wyckoff_score=7.5, min_wyckoff_score_short=6.5),
}

# Cluster definition — (symbol, spread, q2u, session, files)
def _cluster(year: int) -> list:
    y = str(year)
    return [
        ("GBPNZD", 0.0004, 1.62, (7, 17),
         [_DATA / "gbpnzd" / "m1" / f"DAT_MT_GBPNZD_M1_{y}.csv"]),
        ("GBPCAD", 0.0003, 1.36, (7, 17),
         [_DATA / "gbpcad" / "m1" / f"DAT_MT_GBPCAD_M1_{y}.csv"]),
        ("EURAUD", 0.0003, 1.53, (7, 17),
         [_DATA / "euraud" / "m1" / f"DAT_MT_EURAUD_M1_{y}.csv"]),
    ]


def _run_combined(pairs: list, ws_long: float, ws_short: float) -> tuple[dict, list[dict]]:
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
            per_pair.append({"sym": sym, "n": 0, "wr": 0.0, "tr": 0.0})
            continue
        m1_df  = pd.concat(frames).sort_index()
        m1_df  = m1_df[~m1_df.index.duplicated(keep="first")]
        m15_df = resample_ohlcv(m1_df, "15min")
        if len(m15_df) < 50:
            per_pair.append({"sym": sym, "n": 0, "wr": 0.0, "tr": 0.0})
            continue

        strategy = SDStrategy(
            zone_detector           = ZoneDetector(),
            wyckoff_detector        = WyckoffDetector(**_BASE_WYCKOFF),
            session_start_utc       = session[0],
            session_end_utc         = session[1],
            min_wyckoff_score       = ws_long,
            min_wyckoff_score_short = ws_short,
            **_BASE,
        )
        signals = strategy.run(m15_df, m1_df)
        all_signals += len(signals)
        if not signals:
            per_pair.append({"sym": sym, "n": 0, "wr": 0.0, "tr": 0.0})
            continue

        pip = _BASE["pip_size"]
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

        wins_p = [r for r in results if r.outcome == "win"]
        n_p    = len([r for r in results if r.outcome in ("win", "loss")])
        wr_p   = len(wins_p) / n_p * 100 if n_p else 0.0
        tr_p   = sum(r.pnl_r for r in results)
        per_pair.append({"sym": sym, "n": len(results), "wr": wr_p, "tr": tr_p})

        for r in results:
            equity += r.pnl_usd

    if not all_results:
        return ({"n_trades": 0, "win_rate": 0.0, "total_r": 0.0,
                 "max_dd": 0.0, "avg_win_rr": 0.0, "avg_sl_pips": 0.0},
                per_pair)

    m = compute_metrics(all_results, INITIAL_BALANCE, all_signals, n_expired)
    wins = [r for r in all_results if r.outcome == "win"]
    m["avg_win_rr"]  = sum(r.pnl_r for r in wins) / len(wins) if wins else 0.0
    m["avg_sl_pips"] = sum(sl_pips_all) / len(sl_pips_all) if sl_pips_all else 0.0
    return m, per_pair


def _pass(m: dict) -> bool:
    return (m.get("win_rate", 0.0) >= 55.0
            and m.get("total_r",  0.0) > 0
            and m.get("max_dd",   0.0) <= 8.0
            and m.get("n_trades", 0)   >= 30)


def _print_section(label: str, m: dict, pp: list[dict],
                   ref_r: float | None = None) -> None:
    ok  = "✅" if _pass(m) else "❌"
    ret = f"  ret={m['total_r']/ref_r:.0%}" if ref_r and ref_r > 0 else ""
    print(f"  {label:<22} N={m['n_trades']:>3}  WR={m['win_rate']:>5.1f}%  "
          f"R={m['total_r']:>+8.2f}  DD={m['max_dd']:>4.1f}%  "
          f"AvgSL={m['avg_sl_pips']:>4.1f}p  {ok}{ret}")
    for p in pp:
        print(f"    {p['sym']:<8} N={p['n']:>3}  WR={p['wr']:>5.1f}%  R={p['tr']:>+7.2f}")


def main() -> None:
    print(f"\n{'=' * 68}")
    print("  Nouvelles paires — GBPNZD · GBPCAD · EURAUD")
    print("  Exit: TP1@1.5R(33%) → TP2@5R(70%) → Runner@10R | 0.5% risk")
    print(f"{'=' * 68}")

    years = [2024, 2025]

    for year in years:
        cluster = _cluster(year)
        year_label = "opt" if year == 2024 else "OOS"
        print(f"\n  ── {year} ({year_label}) ──────────────────────────────────────────")

        ref_r: dict[str, float] = {}
        for cfg_name, cfg_params in CONFIGS.items():
            m, pp = _run_combined(cluster, cfg_params["min_wyckoff_score"],
                                  cfg_params["min_wyckoff_score_short"])
            rr = ref_r.get(cfg_name)
            _print_section(cfg_name, m, pp, rr)
            if year == 2024:
                ref_r[cfg_name] = m["total_r"]

    # ── EURAUD seul sur 3 ans ──────────────────────────────────────────────
    print(f"\n  ── EURAUD seul — 3 ans (2023 bonus) ─────────────────────────")
    euraud_ref_r: dict[str, float] = {}
    for year in [2023, 2024, 2025]:
        cluster_ea = [
            ("EURAUD", 0.0003, 1.53, (7, 17),
             [_DATA / "euraud" / "m1" / f"DAT_MT_EURAUD_M1_{year}.csv"]),
        ]
        year_label = "ref" if year == 2023 else ("opt" if year == 2024 else "OOS")
        print(f"\n  {year} ({year_label})")
        for cfg_name, cfg_params in CONFIGS.items():
            m, pp = _run_combined(cluster_ea, cfg_params["min_wyckoff_score"],
                                  cfg_params["min_wyckoff_score_short"])
            rr = euraud_ref_r.get(cfg_name)
            _print_section(cfg_name, m, pp, rr)
            if year == 2024:
                euraud_ref_r[cfg_name] = m["total_r"]

    print()
    print("  Spreads : GBPNZD=4p · GBPCAD=3p · EURAUD=3p")
    print("  q2u     : GBPNZD=1.62 · GBPCAD=1.36 · EURAUD=1.53")
    print("  Référence forex cluster : 2024=88.92R WR=60.4% | 2023=83.12R WR=55.5%")
    print()


if __name__ == "__main__":
    main()
