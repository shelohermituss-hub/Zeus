"""
S&D + Liquidity Pool Backtest
==============================

Compares five approaches on the paper cluster (2024 optimisation,
2023 cross-validation):

  BASELINE     : Universal-D fixed TP1@0.75R(50%) → RR=1.25
  POOL_1.0     : TP = nearest opposing swing pool, pool must be ≥ 1.0R away
  POOL_1.5     : TP = nearest opposing swing pool, pool must be ≥ 1.5R away
  POOL_2.0     : TP = nearest opposing swing pool, pool must be ≥ 2.0R away
  HYBRID       : Pool TP when pool ≥ 1.5R (else keep baseline RR); keeps ALL signals

When no valid pool is found and DROP_IF_NO_POOL=True, the signal is dropped.
In HYBRID mode the signal always executes — using the pool RR when available,
and the baseline RR otherwise.

Pool detection uses M15 swing highs/lows (5-bar left / 2-bar right).
The implied RR = distance_to_pool / sl_dist, patched into each signal.

Usage
-----
    python -m zeus.backtest.run_sd_liquidity
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv, resample_ohlcv
from zeus.backtest.sd_simulation import compute_metrics, simulate_all
from zeus.strategy.supply_demand.liquidity_detector import (
    find_pool_target,
    find_swing_levels,
    has_liquidity_sweep,
)
from zeus.strategy.supply_demand.sd_strategy import SDStrategy
from zeus.strategy.supply_demand.wyckoff import WyckoffDetector
from zeus.strategy.supply_demand.zone_detector import ZoneDetector

_ROOT = Path(__file__).parent.parent.parent
_DATA = _ROOT / "data" / "historical"

INITIAL_BALANCE  = 10_000.0
RISK_PCT         = 0.005
BASELINE_TP1_R   = 0.75
BASELINE_TP1_SZ  = 0.50
BASELINE_RR      = 1.25

SWING_LEFT       = 5
SWING_RIGHT      = 2     # 2 bars right = faster confirmation (30 min)
POOL_LOOKBACK    = 300
SWEEP_LOOKBACK   = 20
MIN_WICK_PIPS    = 3.0

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

_BASE_UNIVERSAL_D = dict(
    risk_reward             = BASELINE_RR,
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


def _build_strategy(session: tuple[int, int]) -> SDStrategy:
    return SDStrategy(
        zone_detector    = ZoneDetector(),
        wyckoff_detector = WyckoffDetector(**_BASE_WYCKOFF),
        session_start_utc = session[0],
        session_end_utc   = session[1],
        **_BASE_UNIVERSAL_D,
    )


def _patch_signals(
    signals:      list,
    m15_df:       pd.DataFrame,
    m1_df:        pd.DataFrame,
    pip_size:     float,
    min_pool_rr:  float,
    hybrid:       bool  = False,  # if True: keep all signals, use pool only when found
    req_sweep:    bool  = False,
) -> tuple[list, dict]:
    """
    Return patched signal list + stats dict.

    hybrid=True  → keep signal with baseline RR when no pool found or pool too close.
    hybrid=False → drop signal when no pool found or pool too close.
    """
    sh, sl_arr = find_swing_levels(m15_df, SWING_LEFT, SWING_RIGHT)
    pip_s      = pip_size

    patched = []
    stats   = {"n_pool": 0, "n_baseline_kept": 0, "n_dropped": 0, "implied_rrs": []}

    for sig in signals:
        entry_ts    = m1_df.index[sig.bar_index]
        m15_bar_idx = max(0, int(m15_df.index.searchsorted(entry_ts, side="right")) - 1)
        m15_bar_idx = min(m15_bar_idx, len(m15_df) - 1)

        entry   = sig.entry_price
        sl_dist = abs(entry - sig.stop_loss)

        pool_price = find_pool_target(
            swing_highs   = sh,
            swing_lows    = sl_arr,
            bar_idx       = m15_bar_idx,
            direction     = sig.direction,
            entry         = entry,
            pool_lookback = POOL_LOOKBACK,
            min_rr        = min_pool_rr,
            sl_dist       = sl_dist,
        )

        if pool_price is None:
            if hybrid:
                patched.append(sig)          # keep with original RR
                stats["n_baseline_kept"] += 1
            else:
                stats["n_dropped"] += 1
            continue

        if req_sweep:
            swept = has_liquidity_sweep(
                df             = m15_df,
                bar_idx        = m15_bar_idx,
                swing_highs    = sh,
                swing_lows     = sl_arr,
                direction      = sig.direction,
                sweep_lookback = SWEEP_LOOKBACK,
                min_wick_pips  = MIN_WICK_PIPS,
                pip_size       = pip_s,
            )
            if not swept:
                if hybrid:
                    patched.append(sig)
                    stats["n_baseline_kept"] += 1
                else:
                    stats["n_dropped"] += 1
                continue

        implied_rr = ((pool_price - entry) / sl_dist if sig.direction == "long"
                      else (entry - pool_price) / sl_dist)
        stats["implied_rrs"].append(implied_rr)
        stats["n_pool"] += 1
        patched.append(dataclasses.replace(sig, risk_reward=implied_rr))

    return patched, stats


def _run_combined(
    pairs:        list,
    mode:         str,
    min_pool_rr:  float = 1.5,
) -> dict:
    is_baseline = (mode == "BASELINE")
    is_hybrid   = ("HYBRID" in mode)
    req_sweep   = ("+SWEEP" in mode)
    use_pool    = not is_baseline

    all_results: list = []
    all_signals  = 0
    n_expired    = 0
    equity       = INITIAL_BALANCE
    total_n_pool        = 0
    total_n_bl_kept     = 0
    all_implied_rrs     = []

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
            continue

        if use_pool:
            signals, stats = _patch_signals(
                signals, m15_df, m1_df,
                pip_size    = _BASE_UNIVERSAL_D["pip_size"],
                min_pool_rr = min_pool_rr,
                hybrid      = is_hybrid,
                req_sweep   = req_sweep,
            )
            total_n_pool    += stats["n_pool"]
            total_n_bl_kept += stats["n_baseline_kept"]
            all_implied_rrs.extend(stats["implied_rrs"])

        if not signals:
            continue

        # Pool signals: no partial close (let price reach the pool)
        # Baseline signals in hybrid mode: use standard TP1 partial
        if is_baseline:
            tp1_r, tp1_sz = BASELINE_TP1_R, BASELINE_TP1_SZ
        elif is_hybrid:
            tp1_r, tp1_sz = 0.0, 0.0   # simplified: no partial for pool trades in hybrid
        else:
            tp1_r, tp1_sz = 0.0, 0.0

        results, exp = simulate_all(
            signals,
            m1_df,
            risk_pct           = RISK_PCT,
            spread             = spread,
            max_daily_losses   = 1,
            max_monthly_losses = 4,
            tp1_r              = tp1_r,
            tp1_size           = tp1_sz,
            initial_equity     = equity,
            quote_to_usd_rate  = q2u,
        )
        all_results.extend(results)
        n_expired += exp
        for r in results:
            equity += r.pnl_usd

    if not all_results:
        return {"n_trades": 0, "win_rate": 0.0, "total_r": 0.0, "max_dd": 0.0,
                "avg_win_rr": 0.0, "n_pool": 0, "avg_impl_rr": 0.0}

    m = compute_metrics(all_results, INITIAL_BALANCE, all_signals, n_expired)
    wins = [r for r in all_results if r.outcome == "win"]
    m["avg_win_rr"]  = sum(r.pnl_r for r in wins) / len(wins) if wins else 0.0
    m["n_pool"]      = total_n_pool
    m["n_bl_kept"]   = total_n_bl_kept
    m["avg_impl_rr"] = sum(all_implied_rrs) / len(all_implied_rrs) if all_implied_rrs else 0.0
    return m


def _pass(m: dict) -> bool:
    return (m.get("win_rate", 0.0) >= 55.0
            and m.get("total_r",  0.0) > 0
            and m.get("max_dd",   0.0) <= 8.0)


def main() -> None:
    configs = [
        ("BASELINE",        "BASELINE", 0.0),
        ("POOL ≥1.0R",      "POOL",     1.0),
        ("POOL ≥1.5R",      "POOL",     1.5),
        ("POOL ≥2.0R",      "POOL",     2.0),
        ("HYBRID ≥1.5R",    "HYBRID",   1.5),
        ("HYBRID+SWEEP",    "HYBRID+SWEEP", 1.5),
    ]

    hdr = (f"{'Config':<18} {'N':>4} {'WR%':>6} {'TotalR':>9} {'DD%':>5} "
           f"{'AvgWinR':>8} {'Pools':>6} {'AvgIRR':>7}")
    sep = "─" * len(hdr)

    for year_label, pairs in [("2024 (optimisation)", PAPER_CLUSTER_2024),
                               ("2023 (cross-validation)", PAPER_CLUSTER_2023)]:
        print(f"\n{'=' * 72}")
        print(f"  {year_label} — 4 pairs combined")
        print(f"{'=' * 72}")
        print(hdr)
        print(sep)

        for label, mode, min_rr in configs:
            m   = _run_combined(pairs, mode, min_rr)
            n   = m.get("n_trades",   0)
            wr  = m.get("win_rate",   0.0)
            tr  = m.get("total_r",   0.0)
            dd  = m.get("max_dd",    0.0)
            arw = m.get("avg_win_rr", 0.0)
            np_ = m.get("n_pool",    0)
            irr = m.get("avg_impl_rr", 0.0)
            ok  = "✅" if _pass(m) else "❌"
            print(f"{label:<18} {n:>4} {wr:>6.1f} {tr:>9.2f} {dd:>5.1f} "
                  f"{arw:>8.2f} {np_:>6} {irr:>7.2f}  {ok}")

    print()
    print(f"  Swing detection : left={SWING_LEFT} bars / right={SWING_RIGHT} bars (M15)")
    print(f"  Pool lookback   : {POOL_LOOKBACK} M15 bars (~{POOL_LOOKBACK*15//60}h)")
    print(f"  Sweep lookback  : {SWEEP_LOOKBACK} M15 bars | min wick {MIN_WICK_PIPS} pips")
    print(f"  HYBRID = pool TP when available, else keep baseline RR (all signals kept)")
    print()


if __name__ == "__main__":
    main()
