"""
S&D Strategy — Tiered Exit Backtest
=====================================

Tests a 3-tier exit structure vs. the current baseline and the HYBRID pool approach.

Configs tested (4 paper-cluster pairs combined, 2024 optimise → 2023 cross-validate):

  BASELINE        : TP1@0.75R(50%) → 1.25R  | 0.5% risk
  HYBRID 1%       : Pool TP ≥1.5R or 1.25R baseline | 1.0% risk (HYBRID ≥1.5R variant)
  TIERED-3R       : TP1@1.5R(33%) → TP2@3R(70%) → Runner@8R  | 0.5% risk
  TIERED-5R       : TP1@1.5R(33%) → TP2@5R(70%) → Runner@10R | 0.5% risk
  TIERED-3R+POOL  : TIERED-3R but runner target = pool (signal.rr patched from pool)
  TIERED-3R 1%    : TIERED-3R at 1.0% risk
  TIERED-5R 1%    : TIERED-5R at 1.0% risk

Pass criteria : WR ≥ 55%,  Total R > 0,  Max DD ≤ 8%

Usage
-----
    python -m zeus.backtest.run_sd_tiered_exit
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
)
from zeus.strategy.supply_demand.sd_strategy import SDStrategy
from zeus.strategy.supply_demand.wyckoff import WyckoffDetector
from zeus.strategy.supply_demand.zone_detector import ZoneDetector

_ROOT = Path(__file__).parent.parent.parent
_DATA = _ROOT / "data" / "historical"

INITIAL_BALANCE = 10_000.0

# ── Frozen strategy params ────────────────────────────────────────────────────
_BASE_UNIVERSAL_D = dict(
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

SWING_LEFT    = 5
SWING_RIGHT   = 2
POOL_LOOKBACK = 300
MIN_POOL_RR   = 1.5

# ── Paper cluster ─────────────────────────────────────────────────────────────
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


def _build_strategy(session: tuple[int, int]) -> SDStrategy:
    return SDStrategy(
        zone_detector     = ZoneDetector(),
        wyckoff_detector  = WyckoffDetector(**_BASE_WYCKOFF),
        session_start_utc = session[0],
        session_end_utc   = session[1],
        **_BASE_UNIVERSAL_D,
    )


def _patch_pool_rr(
    signals:     list,
    m15_df:      pd.DataFrame,
    m1_df:       pd.DataFrame,
    pip_size:    float,
    runner_rr:   float,
    min_pool_rr: float = MIN_POOL_RR,
) -> list:
    """
    For each signal, find the nearest pool target.
    If a valid pool (implied RR ≥ min_pool_rr) is found, patch risk_reward = pool implied RR.
    Otherwise keep signal.risk_reward = runner_rr (already patched by caller).
    Hybrid: all signals kept.
    """
    sh, sl_arr = find_swing_levels(m15_df, SWING_LEFT, SWING_RIGHT)
    patched = []
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

        if pool_price is not None:
            implied_rr = ((pool_price - entry) / sl_dist if sig.direction == "long"
                          else (entry - pool_price) / sl_dist)
            patched.append(dataclasses.replace(sig, risk_reward=implied_rr))
        else:
            patched.append(sig)   # keep runner_rr

    return patched


def _run_combined(pairs: list, cfg: dict) -> dict:
    """
    Run one config across all pairs with compounding equity.

    cfg keys:
        risk_pct         : position sizing
        tp1_r            : first partial TP in R (0 = disabled)
        tp1_size         : fraction closed at TP1
        tp2_r            : second partial TP in R (0 = disabled)
        tp2_cumulative   : cumulative fraction closed by TP2
        runner_rr        : final TP (signal.risk_reward)
        use_pool_runner  : if True, patch runner_rr from pool; else use fixed runner_rr
        hybrid           : if True, keep all signals (pool-patch for runner when available)
    """
    risk_pct        = cfg["risk_pct"]
    tp1_r           = cfg.get("tp1_r",          0.0)
    tp1_size        = cfg.get("tp1_size",        0.0)
    tp2_r           = cfg.get("tp2_r",           0.0)
    tp2_cumulative  = cfg.get("tp2_cumulative",  0.70)
    runner_rr       = cfg.get("runner_rr",       1.25)
    use_pool_runner = cfg.get("use_pool_runner", False)

    all_results: list = []
    all_signals  = 0
    n_expired    = 0
    equity       = INITIAL_BALANCE

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

        # Patch runner RR onto every signal
        signals = [dataclasses.replace(s, risk_reward=runner_rr) for s in signals]

        # Optionally overwrite runner RR with pool target
        if use_pool_runner:
            pip_size = _BASE_UNIVERSAL_D["pip_size"]
            signals = _patch_pool_rr(signals, m15_df, m1_df, pip_size, runner_rr)

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
        return {"n_trades": 0, "win_rate": 0.0, "total_r": 0.0, "max_dd": 0.0,
                "avg_win_rr": 0.0}

    m = compute_metrics(all_results, INITIAL_BALANCE, all_signals, n_expired)
    wins = [r for r in all_results if r.outcome == "win"]
    m["avg_win_rr"] = sum(r.pnl_r for r in wins) / len(wins) if wins else 0.0
    return m


def _pass(m: dict) -> bool:
    return (m.get("win_rate", 0.0) >= 55.0
            and m.get("total_r",  0.0) > 0
            and m.get("max_dd",   0.0) <= 8.0)


def main() -> None:
    configs = [
        # label,            cfg dict
        ("BASELINE 0.5%",  {"risk_pct": 0.005, "tp1_r": 0.75, "tp1_size": 0.50,
                             "runner_rr": 1.25}),
        ("HYBRID 1%",      {"risk_pct": 0.010, "tp1_r": 0.0,  "tp1_size": 0.0,
                             "runner_rr": 1.25, "use_pool_runner": True}),
        ("TIERED-3R 0.5%", {"risk_pct": 0.005, "tp1_r": 1.5,  "tp1_size": 0.33,
                             "tp2_r": 3.0, "tp2_cumulative": 0.70,
                             "runner_rr": 8.0}),
        ("TIERED-5R 0.5%", {"risk_pct": 0.005, "tp1_r": 1.5,  "tp1_size": 0.33,
                             "tp2_r": 5.0, "tp2_cumulative": 0.70,
                             "runner_rr": 10.0}),
        ("TIERED-3R+POOL", {"risk_pct": 0.005, "tp1_r": 1.5,  "tp1_size": 0.33,
                             "tp2_r": 3.0, "tp2_cumulative": 0.70,
                             "runner_rr": 8.0, "use_pool_runner": True}),
        ("TIERED-3R 1%",   {"risk_pct": 0.010, "tp1_r": 1.5,  "tp1_size": 0.33,
                             "tp2_r": 3.0, "tp2_cumulative": 0.70,
                             "runner_rr": 8.0}),
        ("TIERED-5R 1%",   {"risk_pct": 0.010, "tp1_r": 1.5,  "tp1_size": 0.33,
                             "tp2_r": 5.0, "tp2_cumulative": 0.70,
                             "runner_rr": 10.0}),
    ]

    hdr = (f"{'Config':<20} {'N':>4} {'WR%':>6} {'TotalR':>9} {'DD%':>5} "
           f"{'AvgWinR':>8} {'Risk%':>6}")
    sep = "─" * len(hdr)

    for year_label, pairs in [("2024 (optimisation)", PAPER_CLUSTER_2024),
                               ("2023 (cross-validation)", PAPER_CLUSTER_2023)]:
        print(f"\n{'=' * 72}")
        print(f"  {year_label} — 4 pairs combined")
        print(f"{'=' * 72}")
        print(hdr)
        print(sep)

        for label, cfg in configs:
            m   = _run_combined(pairs, cfg)
            n   = m.get("n_trades",   0)
            wr  = m.get("win_rate",   0.0)
            tr  = m.get("total_r",   0.0)
            dd  = m.get("max_dd",    0.0)
            arw = m.get("avg_win_rr", 0.0)
            rsk = cfg["risk_pct"] * 100
            ok  = "✅" if _pass(m) else "❌"
            print(f"{label:<20} {n:>4} {wr:>6.1f} {tr:>9.2f} {dd:>5.1f} "
                  f"{arw:>8.2f} {rsk:>5.1f}%  {ok}")

    print()
    print("  Exit structure:")
    print("  TIERED-3R : TP1@1.5R(33%) → TP2@3R(70%) → Runner@8R")
    print("  TIERED-5R : TP1@1.5R(33%) → TP2@5R(70%) → Runner@10R")
    print("  +POOL     : runner target replaced by nearest opposing swing pool")
    print("  After TP1 hit: SL moves to break-even")
    print("  After TP2 hit: SL moves to TP1 level")
    print()


if __name__ == "__main__":
    main()
