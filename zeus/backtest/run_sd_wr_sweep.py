"""
S&D Strategy — Win-Rate Optimisation Sweep
============================================

Fixes the TIERED-5R exit structure (TP1@1.5R/33% → TP2@5R/70% → Runner@10R,
0.5% risk) and sweeps signal-quality parameters to find the combination that
maximises win rate while keeping Total-R and DD in spec.

Parameters swept
----------------
  use_first_touch_only : False / True          (expected +8-12% WR, −30% N)
  use_h4_trend_filter  : False / True          (expected +5-8%  WR, −20% N)
  min_zone_score       : 4.0 / 4.5 / 5.0      (expected +3-5%  WR, −15% N)
  min_wyckoff_score    : 7.0 / 7.5            (expected +2-4%  WR, −10% N)
  max_signals_per_day  : 6 / 3               (expected +2-3%  WR, −10% N)
  min_sl_pips          : 5 / 10              (expected +2-3%  WR,  −5% N)

Total combinations : 2×2×3×2×2×2 = 96

Protocol
--------
  1. Run all 96 configs on 2024 data (optimisation).
  2. Sort by WR desc then TotalR desc; flag ✅/❌ against pass criteria.
  3. Cross-validate top-15 configs on 2023 (blind OOS).

Pass criteria : WR ≥ 60%, Total R > 0, Max DD ≤ 8%, N ≥ 30

Usage
-----
    python -m zeus.backtest.run_sd_wr_sweep
"""
from __future__ import annotations

import dataclasses
import itertools
from pathlib import Path
from typing import Any

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv, resample_ohlcv
from zeus.backtest.sd_simulation import compute_metrics, simulate_all
from zeus.strategy.supply_demand.sd_strategy import SDStrategy
from zeus.strategy.supply_demand.wyckoff import WyckoffDetector
from zeus.strategy.supply_demand.zone_detector import ZoneDetector

_ROOT = Path(__file__).parent.parent.parent
_DATA = _ROOT / "data" / "historical"

INITIAL_BALANCE = 10_000.0

# ── Fixed TIERED-5R exit ──────────────────────────────────────────────────────
EXIT_CFG = dict(
    risk_pct       = 0.005,
    tp1_r          = 1.5,
    tp1_size       = 0.33,
    tp2_r          = 5.0,
    tp2_cumulative = 0.70,
    runner_rr      = 10.0,
)

# ── Fixed (non-swept) strategy params ─────────────────────────────────────────
_BASE_FIXED = dict(
    risk_reward             = 1.25,
    min_composite_score     = 4.0,
    signal_cooldown         = 10,
    use_trend_filter        = True,
    trend_slope_lookback    = 3,
    use_price_above_ema     = True,
    ema_atr_tolerance       = 0.5,
    use_session_filter      = True,
    use_adx_filter          = False,
    use_rsi_filter          = False,
    pip_size                = 0.0001,
    min_score_product       = 0.0,
    min_wyckoff_score_short = 5.9,
)

_BASE_WYCKOFF = dict(
    lookback             = 200,
    max_accum_bars       = 20,
    accum_range_mult     = 6.0,
    mss_lookback         = 60,
    min_spring_sweep_pct = 0.10,
    min_mss_strength_pct = 0.03,
)

# ── Sweep axes ────────────────────────────────────────────────────────────────
# 6 leviers = 2×2×3×2×2×2 = 96 combos — 2024 uniquement
SWEEP_AXES: dict[str, list[Any]] = {
    "use_first_touch_only": [False, True],
    "use_h4_trend_filter":  [False, True],
    "min_zone_score":       [4.0, 4.5, 5.0],
    "min_wyckoff_score":    [7.0, 7.5],
    "max_signals_per_day":  [6, 3],
    "min_sl_pips":          [5, 10],
}

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

# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_strategy(session: tuple[int, int], sweep_params: dict) -> SDStrategy:
    params = {**_BASE_FIXED, **sweep_params}
    return SDStrategy(
        zone_detector     = ZoneDetector(),
        wyckoff_detector  = WyckoffDetector(**_BASE_WYCKOFF),
        session_start_utc = session[0],
        session_end_utc   = session[1],
        **params,
    )


def _run_combined(pairs: list, sweep_params: dict) -> dict:
    """Run TIERED-5R exit across all pairs; return metrics dict."""
    risk_pct       = EXIT_CFG["risk_pct"]
    tp1_r          = EXIT_CFG["tp1_r"]
    tp1_size       = EXIT_CFG["tp1_size"]
    tp2_r          = EXIT_CFG["tp2_r"]
    tp2_cumulative = EXIT_CFG["tp2_cumulative"]
    runner_rr      = EXIT_CFG["runner_rr"]

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

        strategy = _build_strategy(session, sweep_params)
        signals  = strategy.run(m15_df, m1_df)
        all_signals += len(signals)
        if not signals:
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
        for r in results:
            equity += r.pnl_usd

    if not all_results:
        return {"n_trades": 0, "win_rate": 0.0, "total_r": 0.0, "max_dd": 0.0,
                "avg_win_rr": 0.0}

    m = compute_metrics(all_results, INITIAL_BALANCE, all_signals, n_expired)
    wins = [r for r in all_results if r.outcome == "win"]
    m["avg_win_rr"] = sum(r.pnl_r for r in wins) / len(wins) if wins else 0.0
    return m


def _pass(m: dict, min_wr: float = 60.0, min_n: int = 30) -> bool:
    return (m.get("win_rate", 0.0) >= min_wr
            and m.get("total_r",  0.0) > 0
            and m.get("max_dd",   0.0) <= 8.0
            and m.get("n_trades", 0)   >= min_n)


def _label(p: dict) -> str:
    """Human-readable label for a sweep parameter set."""
    fto = "FTO" if p["use_first_touch_only"] else "---"
    h4  = "H4"  if p["use_h4_trend_filter"]  else "--"
    zs  = f"ZS{p['min_zone_score']:.1f}"
    ws  = f"WS{p['min_wyckoff_score']:.1f}"
    sd  = f"SPD{p['max_signals_per_day']}"
    sl  = f"SL{int(p['min_sl_pips'])}p"
    return f"{fto}|{h4}|{zs}|{ws}|{sd}|{sl}"


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    keys   = list(SWEEP_AXES.keys())
    combos = list(itertools.product(*SWEEP_AXES.values()))
    total  = len(combos)

    print(f"\nSweeping {total} signal-quality configs on TIERED-5R exit — 2024 uniquement")
    print(f"Exit: TP1@{EXIT_CFG['tp1_r']}R({EXIT_CFG['tp1_size']*100:.0f}%) "
          f"→ TP2@{EXIT_CFG['tp2_r']}R({EXIT_CFG['tp2_cumulative']*100:.0f}%) "
          f"→ Runner@{EXIT_CFG['runner_rr']}R  |  risk={EXIT_CFG['risk_pct']*100:.1f}%\n")

    rows: list[dict] = []

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        m = _run_combined(PAPER_CLUSTER_2024, params)
        rows.append({"label": _label(params), "params": params, **m})
        print(f"  [{i:>2}/{total}] {_label(params):20s}  "
              f"N={m.get('n_trades',0):>3}  WR={m.get('win_rate',0):>5.1f}%  "
              f"R={m.get('total_r',0):>7.2f}  DD={m.get('max_dd',0):>4.1f}%",
              flush=True)

    rows.sort(key=lambda r: (-r["win_rate"], -r["total_r"]))

    hdr = f"{'Config':<36} {'N':>4} {'WR%':>6} {'TotalR':>8} {'DD%':>5} {'AvgWR':>7}"
    sep = "─" * len(hdr)

    print(f"\n{'=' * 72}")
    print(f"  2024 — {total} configs — triés par WR% desc")
    print(f"{'=' * 72}")
    print(hdr)
    print(sep)

    for r in rows:
        ok = "✅" if _pass(r) else "❌"
        print(
            f"{r['label']:<36} {r['n_trades']:>4} {r['win_rate']:>6.1f} "
            f"{r['total_r']:>8.2f} {r['max_dd']:>5.1f} {r['avg_win_rr']:>7.2f}  {ok}"
        )

    print()
    print("  FTO=use_first_touch_only | H4=use_h4_trend_filter | ZS=min_zone_score")
    print("  WS=min_wyckoff_score | SPD=max_signals_per_day | SLp=min_sl_pips")
    print("  Pass: WR≥60%, TotalR>0, DD≤8%, N≥30")
    print()


if __name__ == "__main__":
    main()
