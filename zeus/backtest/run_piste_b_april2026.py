"""
Piste B — Avril 2026 (focus mensuel).

Même matrice de variantes que run_piste_b, restreinte à avril 2026
pour isoler le comportement sur un seul mois volatile.

Usage
-----
    python -m zeus.backtest.run_piste_b_april2026
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

_ROOT       = Path(__file__).parent.parent.parent
DATA_DIR    = _ROOT / "data" / "historical" / "xauusd" / "m1"
RESULTS_DIR = _ROOT / "data" / "results"

SYMBOL       = "XAUUSD"
INITIAL_BAL  = 10_000.0
RISK_PCT     = 0.01
MIN_RR       = 1.5
SLIPPAGE_PCT = 0.0002
START_DATE   = "2026-04-01"
END_DATE     = "2026-04-30"
ZONES        = ["OB", "OTE"]

VARIANTS: list[dict] = [
    {
        "id": "REF",    "name": "V2A_L3 reference",
        "score": 3.0,  "sl": 6.0,  "max_sl": 10.0,
        "ote": False,  "daily_bias": False, "pattern": False,
        "tp_ladder": "scalp",
    },
    {
        "id": "PB_SL",  "name": "SL=20/30 + mtf_smc TP",
        "score": 3.0,  "sl": 20.0, "max_sl": 30.0,
        "ote": False,  "daily_bias": False, "pattern": False,
        "tp_ladder": "mtf_smc",
    },
    {
        "id": "PB_PAT", "name": "PB_SL + candle pattern",
        "score": 3.0,  "sl": 20.0, "max_sl": 30.0,
        "ote": False,  "daily_bias": False, "pattern": True,
        "tp_ladder": "mtf_smc",
    },
    {
        "id": "PB_DB",  "name": "PB_PAT + daily bias",
        "score": 3.0,  "sl": 20.0, "max_sl": 30.0,
        "ote": False,  "daily_bias": True,  "pattern": True,
        "tp_ladder": "mtf_smc",
    },
    {
        "id": "PB_OTE", "name": "All gates incl. OTE",
        "score": 3.0,  "sl": 20.0, "max_sl": 30.0,
        "ote": True,   "daily_bias": True,  "pattern": True,
        "tp_ladder": "mtf_smc",
    },
]


def _partial_close(v: dict) -> PartialCloseConfig:
    return PartialCloseConfig.mtf_smc() if v["tp_ladder"] == "mtf_smc" else PartialCloseConfig.scalp()


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
        fee_pct=0.0001,
        slippage_pct=SLIPPAGE_PCT,
        partial_close=_partial_close(v),
    )


def _make_strategy(v: dict, df_1h: pd.DataFrame, df_1d: pd.DataFrame) -> ScalpSMCStrategy:
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
        require_entry_pattern=v["pattern"],
        min_wick_ratio=0.60,
        max_daily_signals=5,
        max_signals_per_session=2,
    )


def _sep(char: str = "─", w: int = 112) -> None:
    print(char * w)


def main() -> None:
    setup_logger("WARNING")

    print(f"\nLoading XAUUSD M1 data from {DATA_DIR} …")
    df_m1  = load_m1_directory(DATA_DIR)
    tfs    = build_timeframes(df_m1)
    df_1h  = tfs["1h"]
    df_1d  = tfs["1d"]
    df_ltf = tfs["m1"][START_DATE:END_DATE].copy()
    print(f"  1M bars (avril 2026): {len(df_ltf):,}")

    v0 = VARIANTS[0]
    print(f"\nPhase 1 — {v0['id']} ({v0['name']}) + cache HTF …")
    t0 = time.time()
    first_strat  = _make_strategy(v0, df_1h, df_1d)
    first_engine = _make_engine(v0, first_strat)
    first_result = first_engine.run(df_ltf, symbol=SYMBOL)
    htf_cache    = first_strat._htf_cache
    elapsed = time.time() - t0
    s0 = first_result.summary()
    pf0 = f"{s0['profit_factor']:.2f}" if s0["profit_factor"] else "—"
    print(f"  Done in {elapsed:.0f}s — {len(htf_cache)} HTF bars cached.")
    print(f"  REF: {s0['n_trades']} trades  WR={s0['win_rate']:.1f}%  PF={pf0}")

    all_results = [(v0, s0, first_result.closed_trades)]

    print(f"\nPhase 2 — {len(VARIANTS) - 1} variantes (cache partagé) …")
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

    print()
    _sep("═")
    print(f"  {'ID':8}  {'SL  TP     PAT DB  OTE':22}  {'Variante':<26}  "
          f"{'Trades':>6}  {'WR%':>5}  {'PF':>6}  {'P&L':>8}  {'Ret%':>6}  {'MaxDD%':>7}  {'Sharpe':>8}")
    _sep("─")
    for v, s, _ in all_results:
        sl_tag  = f"SL{int(v['sl']):02d}"
        tp_tag  = "MTF" if v["tp_ladder"] == "mtf_smc" else "SCA"
        pat_tag = "PAT" if v["pattern"]    else "..."
        db_tag  = "DB"  if v["daily_bias"] else ".."
        ote_tag = "OTE" if v["ote"]        else "..."
        flags = f"{sl_tag}/{tp_tag}  {pat_tag} {db_tag}  {ote_tag}"
        pf    = f"{s['profit_factor']:.2f}" if s["profit_factor"] else "—"
        print(
            f"  {v['id']:8}  {flags:22}  {v['name']:<26}  {s['n_trades']:>6}  "
            f"{s['win_rate']:>5.1f}  {pf:>6}  "
            f"{s['net_pnl']:>+8.0f}  {s['total_return_pct']:>+5.1f}%  "
            f"{s['max_drawdown_pct']:>6.1f}%  {s['sharpe_ratio']:>8.4f}"
        )
    _sep("═")

    # Détail des trades par variante
    print("\nDétail des trades (PB_SL) :")
    _sep("─", 80)
    print(f"  {'#':>3}  {'Ouvert':>14}  {'Fermé':>14}  {'Dir':>5}  {'Entrée':>8}  {'P&L':>8}  {'Statut'}")
    _sep("─", 80)
    for v, s, trades in all_results:
        if v["id"] == "PB_SL":
            for i, t in enumerate(trades, 1):
                direction = "LONG" if t.is_long else "SHORT"
                opened  = t.opened_at.strftime("%m-%d %H:%M") if t.opened_at else "—"
                closed  = t.closed_at.strftime("%m-%d %H:%M") if t.closed_at else "ouvert"
                print(f"  {i:>3}  {opened:>14}  {closed:>14}  {direction:>5}  "
                      f"{t.entry_price:>8.1f}  {t.realized_pnl:>+8.0f}  {t.status.value}")
    _sep("═", 80)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "meta": {
            "symbol": SYMBOL, "start": START_DATE, "end": END_DATE,
            "note": "Focus mensuel — avril 2026",
        },
        "variants": [
            {"id": v["id"], "name": v["name"], "config": v, "summary": s}
            for v, s, _ in all_results
        ],
    }
    out_path = RESULTS_DIR / "piste_b_xauusd_april2026.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nRésultats sauvegardés → {out_path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
