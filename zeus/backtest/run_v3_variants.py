"""
V3 quality filters — measure impact on V2A_L3 reference config.

Reference: V2A_L3 (OB+OTE, score≥3, approach+sweep) — PF=1.94, 32 trades.

Variants
--------
  REF      V2A_L3 reference (score=3, SL=6, OTE=off, daily_bias=soft)
  V3_S     + score≥5  (force 3 real confirmation factors)
  V3_SL    + score≥5 + SL=20/30
  V3_OTE   + score≥5 + SL=20/30 + require_ote
  V3_DB    + score≥5 + SL=20/30 + require_daily_bias
  V3_FULL  + all 5 corrections (score=5, SL=20/30, OTE, daily_bias, sweep lb=3)

Usage
-----
    python -m zeus.backtest.run_v3_variants
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

SYMBOL       = "XAUUSD"
INITIAL_BAL  = 10_000.0
RISK_PCT     = 0.01
MIN_RR       = 1.5
SLIPPAGE_PCT = 0.0002
START_DATE   = "2026-01-01"
END_DATE     = "2026-06-30"
ZONES        = ["OB", "OTE"]

# ── Variant definitions ───────────────────────────────────────────────────────
VARIANTS: list[dict] = [
    {
        "id": "REF",    "name": "V2A_L3 reference",
        "score": 3.0,  "sl": 6.0,  "max_sl": 10.0,
        "ote": False,  "daily_bias": False,
    },
    {
        "id": "V3_S",   "name": "score≥5 only",
        "score": 5.0,  "sl": 6.0,  "max_sl": 10.0,
        "ote": False,  "daily_bias": False,
    },
    {
        "id": "V3_SL",  "name": "score≥5 + SL=20/30",
        "score": 5.0,  "sl": 20.0, "max_sl": 30.0,
        "ote": False,  "daily_bias": False,
    },
    {
        "id": "V3_OTE", "name": "score≥5 + SL + OTE gate",
        "score": 5.0,  "sl": 20.0, "max_sl": 30.0,
        "ote": True,   "daily_bias": False,
    },
    {
        "id": "V3_DB",  "name": "score≥5 + SL + daily bias",
        "score": 5.0,  "sl": 20.0, "max_sl": 30.0,
        "ote": False,  "daily_bias": True,
    },
    {
        "id": "V3_FULL","name": "All 5 corrections",
        "score": 5.0,  "sl": 20.0, "max_sl": 30.0,
        "ote": True,   "daily_bias": True,
    },
]


def _make_engine(v: dict, strategy: ScalpSMCStrategy) -> AdvancedBacktestEngine:
    return AdvancedBacktestEngine(
        strategy=strategy,
        initial_balance=INITIAL_BAL,
        stop_loss_pips=v["sl"],
        max_sl_pips=v["max_sl"],
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
        sl_pips=v["sl"],
        max_sl_pips=v["max_sl"],
        pip_value=1.0,
        min_htf_score=v["score"],
        swing_length=20,
        internal_length=5,
        atr_period=100,
        ltf_lookback=30,
        killzone_only=True,
        require_entry_fvg=True,
        require_choch_candle=False,
        allowed_zones=ZONES,
        require_clean_approach=True,
        approach_lookback=5,
        approach_max_momentum=0.6,
        approach_max_body_atr=1.5,
        require_poc_zone_confluence=False,
        require_session_sweep=False,
        require_ltf_sweep=True,
        ltf_sweep_lookback=3,
        require_ote=v["ote"],
        require_daily_bias=v["daily_bias"],
        max_daily_signals=5,
        max_signals_per_session=2,
    )


def _sep(char: str = "─", w: int = 108) -> None:
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

    # ── Phase 1: warm-up REF and cache HTF analysis ───────────────────────────
    v0 = VARIANTS[0]
    print(f"\nPhase 1 — running {v0['id']} ({v0['name']}) and caching HTF analysis …")
    t0 = time.time()
    first_strat  = _make_strategy(v0, df_1h, df_1d)
    first_engine = _make_engine(v0, first_strat)
    first_result = first_engine.run(df_ltf, symbol=SYMBOL)
    htf_cache    = first_strat._htf_cache
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s — {len(htf_cache)} HTF bars cached.")

    all_results = [(v0, first_result.summary(), first_result.closed_trades)]

    # ── Phase 2: inject cache into remaining variants ─────────────────────────
    print(f"\nPhase 2 — running {len(VARIANTS) - 1} more variants (cache shared) …")
    for v in VARIANTS[1:]:
        t1 = time.time()
        strat = _make_strategy(v, df_1h, df_1d)
        strat._htf_cache = htf_cache
        engine = _make_engine(v, strat)
        result = engine.run(df_ltf, symbol=SYMBOL)
        elapsed = time.time() - t1
        s = result.summary()
        pf = f"{s['profit_factor']:.2f}" if s["profit_factor"] else "—"
        print(f"  {v['id']:8} {v['name']:26} — {s['n_trades']:3d} trades "
              f"  WR={s['win_rate']:4.1f}%  PF={pf}  ({elapsed:.0f}s)")
        all_results.append((v, s, result.closed_trades))

    # ── Comparison table ──────────────────────────────────────────────────────
    print()
    _sep("═")
    print(f"  {'ID':8}  {'Filters':12}  {'Variant':<26}  {'Trades':>6}  {'WR%':>5}  {'PF':>6}  "
          f"{'P&L':>8}  {'Ret%':>6}  {'MaxDD%':>7}  {'Sharpe':>8}")
    _sep("─")
    for v, s, _ in all_results:
        flags = f"[s{'5' if v['score']>=5 else '3'} SL{'20' if v['sl']>=20 else ' 6'} {'OTE' if v['ote'] else '...'} {'DB' if v['daily_bias'] else '..'} LTF]"
        pf    = f"{s['profit_factor']:.2f}" if s["profit_factor"] else "—"
        print(
            f"  {v['id']:8}  {flags:12}  {v['name']:<26}  {s['n_trades']:>6}  "
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
    print(f"  {'Month':<8}  " + "  ".join(f"{vid:>8}" for vid in ids))
    _sep("─")
    for month in sorted(monthly):
        row = "  ".join(f"{monthly[month].get(vid, 0):>+8.0f}" for vid in ids)
        print(f"  {month:<8}  {row}")
    _sep("─")
    totals = "  ".join(
        f"{sum(monthly[m].get(vid, 0) for m in monthly):>+8.0f}" for vid in ids
    )
    print(f"  {'TOTAL':<8}  {totals}")
    _sep("═")

    # ── Delta vs REF ─────────────────────────────────────────────────────────
    print("\nFilter impact vs REF baseline:")
    base_s = all_results[0][1]
    print(f"  {'ID':8}  {'Trades Δ':>9}  {'WR Δ':>7}  {'PF Δ':>8}  {'P&L Δ':>8}  {'DD Δ':>8}")
    _sep("─", 65)
    for v, s, _ in all_results[1:]:
        dt  = s["n_trades"]          - base_s["n_trades"]
        dwr = s["win_rate"]          - base_s["win_rate"]
        dpf = (s["profit_factor"] or 0) - (base_s["profit_factor"] or 0)
        dp  = s["net_pnl"]           - base_s["net_pnl"]
        dd  = s["max_drawdown_pct"]  - base_s["max_drawdown_pct"]
        print(f"  {v['id']:8}  {dt:>+9d}  {dwr:>+6.1f}%  {dpf:>+8.2f}  {dp:>+8.0f}  {dd:>+7.1f}%")
    _sep("═", 65)

    # ── Save ──────────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "meta": {
            "symbol": SYMBOL, "start": START_DATE, "end": END_DATE,
            "reference": "V2A_L3 (OB+OTE, score≥3, approach+sweep lb=3)",
            "filters_tested": ["min_htf_score=5", "sl_pips=20", "require_ote", "require_daily_bias"],
        },
        "variants": [
            {"id": v["id"], "name": v["name"], "config": v, "summary": s}
            for v, s, _ in all_results
        ],
    }
    out_path = RESULTS_DIR / "v3_variants_xauusd_2026.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nResults saved → {out_path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
