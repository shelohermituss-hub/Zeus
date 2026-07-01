"""
New filter variants — measure impact of Steps 1, 2, 3 on V2 winning config.

Base: V2 (OB+OTE, score≥3, no CHoCH, no FVG) — PF=1.61, 40 trades.

Variants
--------
  V2    Baseline (V2 winner, approach=OFF)
  V2A   + clean approach gate  (momentum≤0.6, spike≤1.5)
  V2P   + POC zone confluence  (poc in zone)
  V2S   + session sweep        (prev session H/L swept, lookback=30)
  V2AP  + approach + POC
  V2AS  + approach + session sweep
  V2PS  + POC + session sweep
  V2ALL + approach + POC + session sweep

Usage
-----
    python -m zeus.backtest.run_new_filters_variants
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from zeus.backtest.advanced_engine import AdvancedBacktestEngine
from zeus.backtest.data_loader import build_timeframes, load_m1_directory
from zeus.backtest.partial_close import PartialCloseConfig
from zeus.strategy.scalp_strategy import ScalpSMCStrategy
from zeus.utils.logger import setup_logger

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).parent.parent.parent
DATA_DIR    = _ROOT / "data" / "historical" / "xauusd" / "m1"
RESULTS_DIR = _ROOT / "data" / "results"

# ── Shared parameters (same as V2 winning config) ────────────────────────────
SYMBOL       = "XAUUSD"
INITIAL_BAL  = 10_000.0
RISK_PCT     = 0.01
SL_PIPS      = 6.0
MAX_SL_PIPS  = 10.0
MIN_RR       = 1.5
SLIPPAGE_PCT = 0.0002
START_DATE   = "2026-01-01"
END_DATE     = "2026-06-30"
ZONES        = ["OB", "OTE"]    # V2 config: no FVG

# ── Variant definitions ───────────────────────────────────────────────────────
VARIANTS: list[dict] = [
    {
        "id": "V2",    "name": "Baseline (V2)",
        "approach": False, "poc": False, "sweep": False,
    },
    {
        "id": "V2A",   "name": "Approach only",
        "approach": True,  "poc": False, "sweep": False,
    },
    {
        "id": "V2P",   "name": "POC zone only",
        "approach": False, "poc": True,  "sweep": False,
    },
    {
        "id": "V2S",   "name": "Session sweep only",
        "approach": False, "poc": False, "sweep": True,
    },
    {
        "id": "V2AP",  "name": "Approach + POC",
        "approach": True,  "poc": True,  "sweep": False,
    },
    {
        "id": "V2AS",  "name": "Approach + Sweep",
        "approach": True,  "poc": False, "sweep": True,
    },
    {
        "id": "V2PS",  "name": "POC + Sweep",
        "approach": False, "poc": True,  "sweep": True,
    },
    {
        "id": "V2ALL", "name": "All three filters",
        "approach": True,  "poc": True,  "sweep": True,
    },
]


def _make_engine(strategy: ScalpSMCStrategy) -> AdvancedBacktestEngine:
    return AdvancedBacktestEngine(
        strategy=strategy,
        initial_balance=INITIAL_BAL,
        stop_loss_pips=SL_PIPS,
        max_sl_pips=MAX_SL_PIPS,
        pip_value=1.0,
        min_rr=MIN_RR,
        max_position_pct=RISK_PCT,
        max_open_positions=1,
        fee_pct=0.0,
        slippage_pct=SLIPPAGE_PCT,
        partial_close=PartialCloseConfig.scalp(),
    )


def _make_strategy(
    v: dict,
    df_1h: pd.DataFrame,
    df_1d: pd.DataFrame,
) -> ScalpSMCStrategy:
    return ScalpSMCStrategy(
        df_htf_1h=df_1h,
        df_daily=df_1d,
        sl_pips=SL_PIPS,
        max_sl_pips=MAX_SL_PIPS,
        pip_value=1.0,
        min_htf_score=3.0,
        swing_length=20,
        internal_length=5,
        atr_period=100,
        ltf_lookback=30,
        killzone_only=True,
        require_entry_fvg=True,
        require_choch_candle=False,
        allowed_zones=ZONES,
        # New filters
        require_clean_approach=v["approach"],
        approach_lookback=5,
        approach_max_momentum=0.6,
        approach_max_body_atr=1.5,
        require_poc_zone_confluence=v["poc"],
        poc_zone_tolerance_pct=0.005,
        require_session_sweep=v["sweep"],
        session_sweep_lookback=30,
        # Frequency caps
        max_daily_signals=5,
        max_signals_per_session=2,
    )


def _sep(char: str = "─", w: int = 100) -> None:
    print(char * w)


def main() -> None:
    setup_logger("WARNING")

    print(f"\nLoading XAUUSD M1 data from {DATA_DIR} …")
    df_m1  = load_m1_directory(DATA_DIR)
    tfs    = build_timeframes(df_m1)
    df_1h  = tfs["1h"]
    df_1d  = tfs["1d"]
    df_ltf = tfs["m1"][START_DATE:END_DATE].copy()
    print(f"  1M bars (2026): {len(df_ltf):,}")

    # ── Phase 1: warm-up V2 baseline and cache HTF analysis ──────────────────
    print(f"\nPhase 1 — running {VARIANTS[0]['id']} ({VARIANTS[0]['name']}) "
          "and caching HTF analysis …")
    t0 = time.time()
    first_strat  = _make_strategy(VARIANTS[0], df_1h, df_1d)
    first_engine = _make_engine(first_strat)
    first_result = first_engine.run(df_ltf, symbol=SYMBOL)
    htf_cache    = first_strat._htf_cache
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s — {len(htf_cache)} HTF bars cached.")

    all_results = [(VARIANTS[0], first_result.summary(), first_result.closed_trades)]

    # ── Phase 2: inject cache into remaining variants ─────────────────────────
    print(f"\nPhase 2 — running {len(VARIANTS) - 1} more variants (cache shared) …")
    for v in VARIANTS[1:]:
        t1 = time.time()
        strat = _make_strategy(v, df_1h, df_1d)
        strat._htf_cache = htf_cache
        engine = _make_engine(strat)
        result = engine.run(df_ltf, symbol=SYMBOL)
        elapsed = time.time() - t1
        print(f"  {v['id']:5} {v['name']:22} — {result.summary()['n_trades']:3d} trades "
              f"  PF={result.summary()['profit_factor']:<6}  ({elapsed:.0f}s)")
        all_results.append((v, result.summary(), result.closed_trades))

    # ── Print comparison table ─────────────────────────────────────────────────
    print()
    _sep("═")
    print(f"  {'ID':5}  {'Variant':<22}  {'Trades':>6}  {'WR%':>5}  {'PF':>6}  "
          f"{'P&L':>8}  {'Ret%':>6}  {'MaxDD%':>7}  {'Sharpe':>8}")
    _sep("─")
    for v, s, _ in all_results:
        approach_tag = "A" if v["approach"] else "."
        poc_tag      = "P" if v["poc"]      else "."
        sweep_tag    = "S" if v["sweep"]    else "."
        flags        = f"[{approach_tag}{poc_tag}{sweep_tag}]"
        pf           = f"{s['profit_factor']:.2f}" if s["profit_factor"] else "—"
        print(
            f"  {v['id']:5}  {flags} {v['name']:<18}  {s['n_trades']:>6}  "
            f"{s['win_rate']:>5.1f}  {pf:>6}  "
            f"{s['net_pnl']:>+8.0f}  {s['total_return_pct']:>+5.1f}%  "
            f"{s['max_drawdown_pct']:>6.1f}%  {s['sharpe_ratio']:>8.4f}"
        )
    _sep("═")

    # ── Monthly P&L grid ──────────────────────────────────────────────────────
    from collections import defaultdict
    print("\nMonthly P&L (rows = months, cols = variants):")
    monthly: dict[str, dict[str, float]] = defaultdict(dict)
    for v, _, trades in all_results:
        for t in trades:
            if t.closed_at:
                key = t.closed_at.strftime("%Y-%m")
                monthly[key][v["id"]] = monthly[key].get(v["id"], 0) + t.realized_pnl

    ids = [v["id"] for v, _, _ in all_results]
    print(f"  {'Month':<8}  " + "  ".join(f"{vid:>6}" for vid in ids))
    _sep("─")
    for month in sorted(monthly):
        row = "  ".join(f"{monthly[month].get(vid, 0):>+6.0f}" for vid in ids)
        print(f"  {month:<8}  {row}")
    _sep("─")
    totals = "  ".join(
        f"{sum(monthly[m].get(vid, 0) for m in monthly):>+6.0f}" for vid in ids
    )
    print(f"  {'TOTAL':<8}  {totals}")
    _sep("═")

    # Filter effectiveness table
    print("\nFilter effectiveness vs V2 baseline:")
    base_s = all_results[0][1]
    print(f"  {'ID':5}  {'Trades Δ':>9}  {'PF Δ':>8}  {'P&L Δ':>8}  {'DD Δ':>8}")
    _sep("─", 60)
    for v, s, _ in all_results[1:]:
        dt  = s["n_trades"]       - base_s["n_trades"]
        dpf = (s["profit_factor"] or 0) - (base_s["profit_factor"] or 0)
        dp  = s["net_pnl"]        - base_s["net_pnl"]
        dd  = s["max_drawdown_pct"] - base_s["max_drawdown_pct"]
        print(f"  {v['id']:5}  {dt:>+9d}  {dpf:>+8.2f}  {dp:>+8.0f}  {dd:>+7.1f}%")
    _sep("═", 60)

    # ── Save ──────────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "meta": {
            "symbol": SYMBOL, "start": START_DATE, "end": END_DATE,
            "base_config": "V2 (OB+OTE, score≥3)",
            "filters_tested": ["require_clean_approach", "require_poc_zone_confluence", "require_session_sweep"],
        },
        "variants": [
            {"id": v["id"], "name": v["name"], "config": v, "summary": s}
            for v, s, _ in all_results
        ],
    }
    out_path = RESULTS_DIR / "new_filters_variants_xauusd_2026.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nResults saved → {out_path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
