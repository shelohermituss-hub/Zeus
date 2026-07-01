"""
ScalpSMCStrategy — variant comparison, XAUUSD Jan–Jun 2026.

Tests 8 filter configurations to find the best balance between
trade frequency and quality.

Performance trick: the 1H HTF SMC analysis cache is pre-populated
once by the first variant run and then injected into all subsequent
variants, so only the first run pays the full ~6-min analysis cost.

Variants
--------
  V1  Baseline        all zones  score≥3  no CHoCH  5/day
  V2  No FVG          OB+OTE     score≥3  no CHoCH  5/day
  V3  OTE only        OTE        score≥3  no CHoCH  5/day
  V4  Score 5+        all zones  score≥5  no CHoCH  5/day
  V5  CHoCH           all zones  score≥3  CHoCH     5/day
  V6  OB+OTE + Sc5    OB+OTE     score≥5  no CHoCH  5/day
  V7  OB+OTE + CHoCH  OB+OTE     score≥3  CHoCH     5/day
  V8  Full strict     OB+OTE     score≥5  CHoCH     3/day

Usage
-----
    python -m zeus.backtest.run_scalp_variants
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

# ── Paths ────────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).parent.parent.parent
DATA_DIR    = _ROOT / "data" / "historical" / "xauusd" / "m1"
RESULTS_DIR = _ROOT / "data" / "results"

# ── Shared parameters ─────────────────────────────────────────────────────────
SYMBOL        = "XAUUSD"
INITIAL_BAL   = 10_000.0
RISK_PCT      = 0.01
SL_PIPS       = 6.0
MAX_SL_PIPS   = 10.0
MIN_RR        = 1.5
SLIPPAGE_PCT  = 0.0002
START_DATE    = "2026-01-01"
END_DATE      = "2026-06-30"

# ── Variant definitions ───────────────────────────────────────────────────────
VARIANTS: list[dict] = [
    {
        "id":    "V1",
        "name":  "Baseline",
        "zones": None,          # all zones
        "score": 3.0,
        "choch": False,
        "daily": 5,
        "sess":  2,
    },
    {
        "id":    "V2",
        "name":  "No FVG (OB+OTE)",
        "zones": ["OB", "OTE"],
        "score": 3.0,
        "choch": False,
        "daily": 5,
        "sess":  2,
    },
    {
        "id":    "V3",
        "name":  "OTE only",
        "zones": ["OTE"],
        "score": 3.0,
        "choch": False,
        "daily": 5,
        "sess":  2,
    },
    {
        "id":    "V4",
        "name":  "Score ≥ 5",
        "zones": None,
        "score": 5.0,
        "choch": False,
        "daily": 5,
        "sess":  2,
    },
    {
        "id":    "V5",
        "name":  "CHoCH candle",
        "zones": None,
        "score": 3.0,
        "choch": True,
        "daily": 5,
        "sess":  2,
    },
    {
        "id":    "V6",
        "name":  "OB+OTE + Score 5",
        "zones": ["OB", "OTE"],
        "score": 5.0,
        "choch": False,
        "daily": 5,
        "sess":  2,
    },
    {
        "id":    "V7",
        "name":  "OB+OTE + CHoCH",
        "zones": ["OB", "OTE"],
        "score": 3.0,
        "choch": True,
        "daily": 5,
        "sess":  2,
    },
    {
        "id":    "V8",
        "name":  "Full strict",
        "zones": ["OB", "OTE"],
        "score": 5.0,
        "choch": True,
        "daily": 3,
        "sess":  1,
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
        df_mtf_15m=None,
        df_daily=df_1d,
        sl_pips=SL_PIPS,
        max_sl_pips=MAX_SL_PIPS,
        pip_value=1.0,
        min_htf_score=v["score"],
        swing_length=20,
        internal_length=5,
        atr_period=100,
        ltf_lookback=30,
        killzone_only=True,
        require_entry_fvg=True,
        require_choch_candle=v["choch"],
        max_daily_signals=v["daily"],
        max_signals_per_session=v["sess"],
        allowed_zones=v["zones"],
    )


def _sep(char: str = "─", w: int = 80) -> None:
    print(char * w)


def main() -> None:
    setup_logger("WARNING")   # suppress INFO noise during long runs

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading XAUUSD M1 data from {DATA_DIR} …")
    df_m1 = load_m1_directory(DATA_DIR)
    tfs   = build_timeframes(df_m1)
    df_1h = tfs["1h"]
    df_1d = tfs["1d"]
    df_ltf = tfs["m1"][START_DATE:END_DATE].copy()
    print(f"  1M bars (2026): {len(df_ltf):,}")

    # ── Phase 1: warm-up first variant and cache HTF analysis ─────────────────
    print(f"\nPhase 1 — running {VARIANTS[0]['id']} ({VARIANTS[0]['name']}) "
          "and caching HTF analysis …")
    t0 = time.time()

    first_strat = _make_strategy(VARIANTS[0], df_1h, df_1d)
    first_engine = _make_engine(first_strat)
    first_result = first_engine.run(df_ltf, symbol=SYMBOL)
    htf_cache = first_strat._htf_cache   # populated by the full run

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s — {len(htf_cache)} HTF bars cached.")

    all_results = [(VARIANTS[0], first_result.summary(), first_result.closed_trades)]

    # ── Phase 2: inject cache into remaining variants ─────────────────────────
    print(f"\nPhase 2 — running {len(VARIANTS) - 1} more variants (cache shared) …")
    for v in VARIANTS[1:]:
        t1 = time.time()
        strat = _make_strategy(v, df_1h, df_1d)
        strat._htf_cache = htf_cache   # skip expensive re-analysis
        engine = _make_engine(strat)
        result = engine.run(df_ltf, symbol=SYMBOL)
        elapsed = time.time() - t1
        print(f"  {v['id']} {v['name']:20s} — {result.summary()['n_trades']:3d} trades "
              f"  PF={result.summary()['profit_factor']:<6}  ({elapsed:.0f}s)")
        all_results.append((v, result.summary(), result.closed_trades))

    # ── Print comparison table ─────────────────────────────────────────────────
    print()
    _sep("═", 100)
    print(f"  {'ID':3}  {'Variant':<22}  {'Trades':>6}  {'WR%':>5}  {'PF':>6}  "
          f"{'P&L':>8}  {'Ret%':>6}  {'MaxDD%':>7}  {'Sharpe':>8}")
    _sep("─", 100)
    for v, s, _ in all_results:
        pf = f"{s['profit_factor']:.2f}" if s['profit_factor'] else "—"
        print(
            f"  {v['id']:3}  {v['name']:<22}  {s['n_trades']:>6}  "
            f"{s['win_rate']:>5.1f}  {pf:>6}  "
            f"{s['net_pnl']:>+8.0f}  {s['total_return_pct']:>+5.1f}%  "
            f"{s['max_drawdown_pct']:>6.1f}%  {s['sharpe_ratio']:>8.4f}"
        )
    _sep("═", 100)

    # ── Monthly breakdown for each variant ────────────────────────────────────
    from collections import defaultdict
    print("\nMonthly P&L grid (rows = months, cols = variants):")
    monthly: dict[str, dict[str, float]] = defaultdict(dict)
    for v, _, trades in all_results:
        for t in trades:
            if t.closed_at:
                key = t.closed_at.strftime("%Y-%m")
                monthly[key][v["id"]] = monthly[key].get(v["id"], 0) + t.realized_pnl

    ids = [v["id"] for v, _, _ in all_results]
    print(f"  {'Month':<8}  " + "  ".join(f"{vid:>7}" for vid in ids))
    _sep("─", 100)
    for month in sorted(monthly):
        row = "  ".join(f"{monthly[month].get(vid, 0):>+7.0f}" for vid in ids)
        print(f"  {month:<8}  {row}")
    _sep("─", 100)
    totals = "  ".join(
        f"{sum(monthly[m].get(vid, 0) for m in monthly):>+7.0f}" for vid in ids
    )
    print(f"  {'TOTAL':<8}  {totals}")
    _sep("═", 100)

    # ── Save results ──────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "meta": {
            "symbol": SYMBOL, "start": START_DATE, "end": END_DATE,
            "strategy": "ScalpSMCStrategy",
            "cascade": "1H → 1M",
            "shared_params": {
                "sl_pips": SL_PIPS, "max_sl_pips": MAX_SL_PIPS,
                "killzone_only": True, "require_entry_fvg": True,
                "partial_close": "scalp (1.5R/2.5R/4R)",
            },
        },
        "variants": [
            {
                "id": v["id"],
                "name": v["name"],
                "config": {k: v[k] for k in ("zones", "score", "choch", "daily", "sess")},
                "summary": s,
            }
            for v, s, _ in all_results
        ],
    }
    out_path = RESULTS_DIR / "scalp_variants_xauusd_2026_h1.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nResults saved → {out_path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
