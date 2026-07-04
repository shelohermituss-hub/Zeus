"""
S&D Strategy — USDJPY + EURJPY Parameter Sweep
================================================

Sweeps Wyckoff score thresholds and zone score to push WR above 55% for
USDJPY and EURJPY (GBPJPY excluded — incompatible with the strategy).

Baseline per-pair results (current params):
  USDJPY 2024: WR=46.3%  R=+8.86R
  EURJPY 2024: WR=47.1%  R=+6.52R
  USDJPY 2025: WR=50.0%  R=+12.39R (OOS)
  EURJPY 2025: WR=46.9%  R=-0.12R  (OOS)

Sweep grid:
  wyckoff_pair = [(long, short)] — 4 combos tightening both thresholds
  min_zone_score                 — 2 levels
  Total: 8 combos × 2 years = 16 runs

Exit: TIERED-5R (TP1@1.5R 33% → TP2@5R 70% → Runner@10R) | 0.5% risk

Usage
-----
    python -m zeus.backtest.run_sd_jpy_wr_sweep
"""
from __future__ import annotations

import dataclasses
import itertools
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

# Fixed parameters — JPY-specific
_BASE = dict(
    risk_reward             = 1.25,
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
    pip_size                = 0.01,
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

# Pairs: USDJPY + EURJPY only
# spread in price units (pips × 0.01), q2u=1.0 for JPY
PAIRS_2024 = [
    ("USDJPY", 0.01, 1.0, (7, 17), [_DATA / "usdjpy" / "m1" / "DAT_MT_USDJPY_M1_2024.csv"]),
    ("EURJPY", 0.02, 1.0, (7, 17), [_DATA / "eurjpy" / "m1" / "DAT_MT_EURJPY_M1_2024.csv"]),
]

PAIRS_2025 = [
    ("USDJPY", 0.01, 1.0, (7, 17), [_DATA / "usdjpy" / "m1" / "DAT_MT_USDJPY_M1_2025.csv"]),
    ("EURJPY", 0.02, 1.0, (7, 17), [_DATA / "eurjpy" / "m1" / "DAT_MT_EURJPY_M1_2025.csv"]),
]

# Sweep axes
# Each tuple = (min_wyckoff_score_long, min_wyckoff_score_short)
WYCKOFF_PAIRS = [
    (7.0, 5.9),   # baseline
    (7.5, 6.5),   # moderate tightening
    (8.0, 7.0),   # strict
    (8.5, 7.5),   # very strict
]

ZONE_SCORES = [4.0, 5.0]


def _run_combined(pairs: list, ws_long: float, ws_short: float,
                  zone_score: float) -> tuple[dict, list[dict]]:
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

        strategy = SDStrategy(
            zone_detector            = ZoneDetector(),
            wyckoff_detector         = WyckoffDetector(**_BASE_WYCKOFF),
            session_start_utc        = session[0],
            session_end_utc          = session[1],
            min_zone_score           = zone_score,
            min_wyckoff_score        = ws_long,
            min_wyckoff_score_short  = ws_short,
            **_BASE,
        )
        signals = strategy.run(m15_df, m1_df)
        all_signals += len(signals)
        if not signals:
            per_pair.append({"sym": sym, "n": 0, "wr": 0.0, "tr": 0.0})
            continue

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
                 "max_dd": 0.0, "avg_win_rr": 0.0},
                per_pair)

    m = compute_metrics(all_results, INITIAL_BALANCE, all_signals, n_expired)
    wins = [r for r in all_results if r.outcome == "win"]
    m["avg_win_rr"] = sum(r.pnl_r for r in wins) / len(wins) if wins else 0.0
    return m, per_pair


def _pass(m: dict) -> bool:
    return (m.get("win_rate", 0.0) >= 55.0
            and m.get("total_r",  0.0) > 0
            and m.get("max_dd",   0.0) <= 8.0
            and m.get("n_trades", 0)   >= 20)


def _pair_row(p: dict) -> str:
    return (f"    {p['sym']:<8} {p['n']:>4}  {p['wr']:>6.1f}%  {p['tr']:>+8.2f}R")


def main() -> None:
    combos = list(itertools.product(WYCKOFF_PAIRS, ZONE_SCORES))
    total  = len(combos)

    col_hdr = f"  {'Config':<22}  {'N':>4}  {'WR%':>6}  {'TotalR':>8}  {'DD%':>5}"
    sep     = "─" * len(col_hdr)

    print(f"\n{'=' * 65}")
    print("  USDJPY + EURJPY — Wyckoff & Zone score sweep")
    print(f"  Exit: TP1@1.5R(33%) → TP2@5R(70%) → Runner@10R | 0.5% risk")
    print(f"{'=' * 65}")

    # ── 2024 optimisation ───────────────────────────────────────────────────
    results_2024: list[tuple] = []

    print(f"\n  ── 2024 — optimisation ({'─' * 3} {total} combos {'─' * 3})")
    print(col_hdr)
    print(sep)
    for i, ((ws_l, ws_s), zs) in enumerate(combos, 1):
        label = f"WL{ws_l:.1f}|WS{ws_s:.1f}|ZS{zs:.1f}"
        m, pp = _run_combined(PAIRS_2024, ws_l, ws_s, zs)
        ok = "✅" if _pass(m) else "❌"
        print(f"  {label:<22}  {m['n_trades']:>4}  {m['win_rate']:>6.1f}%  "
              f"{m['total_r']:>+8.2f}R  {m['max_dd']:>5.1f}%  {ok}")
        for p in pp:
            print(_pair_row(p))
        results_2024.append(((ws_l, ws_s, zs), m, pp))
        if i < total:
            print()

    # ── 2025 OOS ────────────────────────────────────────────────────────────
    results_2025: list[tuple] = []

    print(f"\n  ── 2025 — OOS ({'─' * 3} {total} combos {'─' * 3})")
    print(col_hdr + "   ret%")
    print(sep + "───────")
    for i, ((ws_l, ws_s, zs), m24, _) in enumerate(results_2024, 1):
        label = f"WL{ws_l:.1f}|WS{ws_s:.1f}|ZS{zs:.1f}"
        m, pp = _run_combined(PAIRS_2025, ws_l, ws_s, zs)
        ok  = "✅" if _pass(m) else "❌"
        ret = f"{m['total_r'] / m24['total_r']:.0%}" if m24["total_r"] > 0 else "n/a"
        print(f"  {label:<22}  {m['n_trades']:>4}  {m['win_rate']:>6.1f}%  "
              f"{m['total_r']:>+8.2f}R  {m['max_dd']:>5.1f}%  {ok}  {ret}")
        for p in pp:
            print(_pair_row(p))
        results_2025.append(((ws_l, ws_s, zs), m, pp))
        if i < total:
            print()

    # ── Résumé côte-à-côte ──────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("  Résumé — combiné & par paire")
    print(f"{'=' * 65}")
    pair_hdr = f"  {'Config':<22}  {'Paire':<8}  {'N(24)':>5}  {'WR(24)':>7}  {'R(24)':>7}  {'N(25)':>5}  {'WR(25)':>7}  {'R(25)':>7}"
    print(pair_hdr)
    print("─" * len(pair_hdr))

    def _sym_data(pp: list[dict], sym: str) -> dict:
        return next((p for p in pp if p["sym"] == sym), {"n": 0, "wr": 0.0, "tr": 0.0})

    for (ws_l, ws_s, zs), m24, pp24 in results_2024:
        label = f"WL{ws_l:.1f}|WS{ws_s:.1f}|ZS{zs:.1f}"
        # Find matching 2025
        (_, m25, pp25) = next(
            (x for x in results_2025 if x[0] == (ws_l, ws_s, zs)),
            ((ws_l, ws_s, zs), {"n_trades": 0, "win_rate": 0.0, "total_r": 0.0}, [])
        )
        ok24 = "✅" if _pass(m24) else "❌"
        ok25 = "✅" if _pass(m25) else "❌"
        # Combined row
        print(f"  {label:<22}  {'TOTAL':<8}  {m24['n_trades']:>5}  "
              f"{m24['win_rate']:>6.1f}%  {m24['total_r']:>+7.2f}  "
              f"{m25['n_trades']:>5}  {m25['win_rate']:>6.1f}%  "
              f"{m25['total_r']:>+7.2f}  {ok24}/{ok25}")
        # Per-pair rows
        for sym in ("USDJPY", "EURJPY"):
            p24 = _sym_data(pp24, sym)
            p25 = _sym_data(pp25, sym)
            print(f"  {'':>22}  {sym:<8}  {p24['n']:>5}  "
                  f"{p24['wr']:>6.1f}%  {p24['tr']:>+7.2f}  "
                  f"{p25['n']:>5}  {p25['wr']:>6.1f}%  {p25['tr']:>+7.2f}")
        print()

    print("  WL = min_wyckoff_score (longs) | WS = min_wyckoff_score_short")
    print("  ZS = min_zone_score | pip_size=0.01 (JPY) | q2u=1.0")
    print()


if __name__ == "__main__":
    main()
