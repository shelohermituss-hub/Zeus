"""
S&D — 4 paires validées · WS7.0 vs WS7.5 · 2024 + 2025
=========================================================
Affiche les résultats par paire et par année pour comparer
les deux configs sur l'horizon optimisation (2024) + OOS (2025).

Usage
-----
    python -m zeus.backtest.run_sd_4p_compare
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

PAIRS = [
    ("GBPUSD", 0.0001, 1.00, (7, 17)),
    ("EURUSD", 0.0001, 1.00, (7, 17)),
    ("GBPAUD", 0.0002, 1.52, (7, 17)),
    ("EURNZD", 0.0002, 1.62, (7, 17)),
]

CONFIGS = {
    "WS7.0": dict(min_wyckoff_score=7.0, min_wyckoff_score_short=5.9),
    "WS7.5": dict(min_wyckoff_score=7.5, min_wyckoff_score_short=6.5),
}

YEARS = [2024, 2025]


def _run_pair(sym: str, spread: float, q2u: float, session: tuple,
              year: int, ws_long: float, ws_short: float) -> dict | None:
    files = [_DATA / sym.lower() / "m1" / f"DAT_MT_{sym}_M1_{year}.csv"]
    frames = [parse_histdata_csv(f) for f in files if f.exists()]
    if not frames:
        return None

    m1_df  = pd.concat(frames).sort_index()
    m1_df  = m1_df[~m1_df.index.duplicated(keep="first")]
    m15_df = resample_ohlcv(m1_df, "15min")
    if len(m15_df) < 50:
        return None

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
    if not signals:
        return {"n": 0, "wr": 0.0, "tr": 0.0, "dd": 0.0, "sl": 0.0}

    pip = _BASE["pip_size"]
    sl_pips = [abs(s.entry_price - s.stop_loss) / pip for s in signals]

    runner_rr = TIERED5R["runner_rr"]
    signals = [dataclasses.replace(s, risk_reward=runner_rr) for s in signals]

    results, exp = simulate_all(
        signals, m1_df,
        risk_pct           = TIERED5R["risk_pct"],
        spread             = spread,
        max_daily_losses   = 1,
        max_monthly_losses = 4,
        tp1_r              = TIERED5R["tp1_r"],
        tp1_size           = TIERED5R["tp1_size"],
        tp2_r              = TIERED5R["tp2_r"],
        tp2_cumulative_pct = TIERED5R["tp2_cumulative"],
        initial_equity     = INITIAL_BALANCE,
        quote_to_usd_rate  = q2u,
    )

    wins = [r for r in results if r.outcome == "win"]
    n_dec = len([r for r in results if r.outcome in ("win", "loss")])
    wr    = len(wins) / n_dec * 100 if n_dec else 0.0
    tr    = sum(r.pnl_r for r in results)

    m = compute_metrics(results, INITIAL_BALANCE, len(signals), exp)

    return {
        "n":  len(results),
        "wr": wr,
        "tr": tr,
        "dd": m.get("max_dd", 0.0),
        "sl": sum(sl_pips) / len(sl_pips) if sl_pips else 0.0,
    }


def _pass(d: dict) -> bool:
    return (d["wr"] >= 55.0 and d["tr"] > 0 and d["dd"] <= 8.0 and d["n"] >= 30)


def main() -> None:
    # results[year][cfg][sym] = dict | None
    results: dict = {}
    for year in YEARS:
        results[year] = {}
        for cfg, params in CONFIGS.items():
            results[year][cfg] = {}
            for sym, spread, q2u, session in PAIRS:
                r = _run_pair(sym, spread, q2u, session, year,
                              params["min_wyckoff_score"],
                              params["min_wyckoff_score_short"])
                results[year][cfg][sym] = r

    # ── Affichage ────────────────────────────────────────────────────────────
    hdr = f"  {'Paire':<8} {'Année':>4}  {'Config':<7}  {'N':>4}  {'WR %':>6}  {'Total R':>9}  {'DD %':>5}  {'SL moy':>6}"
    sep = "  " + "─" * (len(hdr) - 2)

    print(f"\n{'=' * 72}")
    print("  4 paires validées — WS7.0 vs WS7.5 — TIERED-5R 0.5 % risk")
    print(f"{'=' * 72}")

    for sym, _, _, _ in PAIRS:
        print(f"\n  ── {sym} {'─' * (60 - len(sym))}")
        print(hdr)
        print(sep)
        for year in YEARS:
            for cfg in CONFIGS:
                d = results[year][cfg].get(sym)
                if d is None:
                    print(f"  {sym:<8} {year:>4}  {cfg:<7}  {'N/A — données manquantes':>45}")
                    continue
                if d["n"] == 0:
                    print(f"  {sym:<8} {year:>4}  {cfg:<7}  {'aucun signal':>45}")
                    continue
                ok = "✅" if _pass(d) else "❌"
                print(f"  {sym:<8} {year:>4}  {cfg:<7}  {d['n']:>4}  "
                      f"{d['wr']:>6.1f}%  {d['tr']:>+9.2f}R  "
                      f"{d['dd']:>5.1f}%  {d['sl']:>5.1f}p  {ok}")

    # ── Totaux combinés ───────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  Cluster combiné (4 paires)")
    print(f"{'=' * 72}")
    print(f"  {'Année':>4}  {'Config':<7}  {'N total':>7}  {'WR %':>6}  {'Total R':>9}  {'DD %':>5}")
    print("  " + "─" * 50)

    for year in YEARS:
        for cfg in CONFIGS:
            all_r: list = []
            n_total = 0
            n_wins  = 0
            n_dec   = 0
            has_data = False
            for sym, spread, q2u, session in PAIRS:
                d = results[year][cfg].get(sym)
                if d is None or d["n"] == 0:
                    continue
                has_data = True
                n_total += d["n"]
                # Re-approximate wins from WR × n (approximation)
                n_dec_p = round(d["n"] * (d["wr"] / 100) / (d["wr"] / 100)) if d["wr"] > 0 else d["n"]
                n_wins  += round(d["n"] * d["wr"] / 100)
                n_dec   += d["n"]
                all_r.append(d["tr"])

            if not has_data:
                print(f"  {year:>4}  {cfg:<7}  {'données insuffisantes':>35}")
                continue

            wr_comb = n_wins / n_dec * 100 if n_dec else 0.0
            tr_comb = sum(all_r)
            ok = "✅" if (wr_comb >= 55 and tr_comb > 0) else "❌"
            miss = " ⚠ données partielles" if any(
                results[year][cfg].get(sym) is None for sym, *_ in PAIRS
            ) else ""
            print(f"  {year:>4}  {cfg:<7}  {n_total:>7}  "
                  f"{wr_comb:>6.1f}%  {tr_comb:>+9.2f}R  {'—':>5}  {ok}{miss}")

    print()
    print("  Spreads : GBPUSD=1p · EURUSD=1p · GBPAUD=2p · EURNZD=2p")
    print("  q2u     : GBPUSD=1.00 · EURUSD=1.00 · GBPAUD=1.52 · EURNZD=1.62")
    print()


if __name__ == "__main__":
    main()
