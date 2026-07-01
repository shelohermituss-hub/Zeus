"""
Backtest multi-instruments — Juin 2026 (config NO_MSS).

Compare XAUUSD, GBPUSD, USDCAD, NZDUSD sur le mois de juin 2026 avec la
configuration NO_MSS (meilleure config identifiée sur XAUUSD).

Paramètres pip par instrument :
    XAUUSD  pip_value=1.0,     sl_pips=20, max_sl_pips=30  (1 pip = $1)
    GBPUSD  pip_value=0.0001,  sl_pips=20, max_sl_pips=30  (1 pip = 0.1 point)
    USDCAD  pip_value=0.0001,  sl_pips=20, max_sl_pips=30
    NZDUSD  pip_value=0.0001,  sl_pips=20, max_sl_pips=30

Usage
-----
    python -m zeus.backtest.run_forex_june2026
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from zeus.backtest.advanced_engine import AdvancedBacktestEngine
from zeus.backtest.data_loader import build_timeframes, load_m1_directory
from zeus.backtest.partial_close import PartialCloseConfig
from zeus.strategy.scalp_strategy import ScalpSMCStrategy
from zeus.utils.logger import setup_logger

_ROOT       = Path(__file__).parent.parent.parent
RESULTS_DIR = _ROOT / "data" / "results"

INITIAL_BAL  = 10_000.0
RISK_PCT     = 0.01
MIN_RR       = 1.5
SLIPPAGE_PCT = 0.0002
START        = "2026-06-01"
END          = "2026-06-30"
TRADING_DAYS = 21

INSTRUMENTS = [
    {
        "id":        "XAUUSD",
        "data_dir":  _ROOT / "data" / "historical" / "xauusd" / "m1",
        "pip_value": 1.0,
        "sl_pips":   20.0,
        "max_sl":    30.0,
    },
    {
        "id":        "GBPUSD",
        "data_dir":  _ROOT / "data" / "historical" / "gbpusd" / "m1",
        "pip_value": 0.0001,
        "sl_pips":   20.0,
        "max_sl":    30.0,
    },
    {
        "id":        "USDCAD",
        "data_dir":  _ROOT / "data" / "historical" / "usdcad" / "m1",
        "pip_value": 0.0001,
        "sl_pips":   20.0,
        "max_sl":    30.0,
    },
    {
        "id":        "NZDUSD",
        "data_dir":  _ROOT / "data" / "historical" / "nzdusd" / "m1",
        "pip_value": 0.0001,
        "sl_pips":   20.0,
        "max_sl":    30.0,
    },
]


def _make_strategy(inst: dict, tfs: dict) -> ScalpSMCStrategy:
    return ScalpSMCStrategy(
        df_htf_1h=tfs["1h"],
        df_mtf_15m=None,          # NO_MSS
        df_daily=tfs["1d"],
        sl_pips=inst["sl_pips"],
        max_sl_pips=inst["max_sl"],
        pip_value=inst["pip_value"],
        min_htf_score=3.0,
        swing_length=20,
        internal_length=5,
        atr_period=100,
        ltf_lookback=30,
        killzone_only=True,
        require_entry_fvg=True,
        require_choch_candle=False,
        allowed_zones=["OB", "OTE"],
        require_clean_approach=True,
        approach_lookback=5,
        approach_max_momentum=0.6,
        approach_max_body_atr=1.5,
        require_poc_zone_confluence=False,
        require_session_sweep=False,
        require_ltf_sweep=True,
        ltf_sweep_lookback=3,
        require_ote=False,
        require_daily_bias=False,
        require_entry_pattern=False,
        max_daily_signals=5,
        max_signals_per_session=2,
        mss_lookback=20,
    )


def _make_engine(inst: dict, strat: ScalpSMCStrategy) -> AdvancedBacktestEngine:
    return AdvancedBacktestEngine(
        strategy=strat,
        initial_balance=INITIAL_BAL,
        stop_loss_pips=inst["sl_pips"],
        max_sl_pips=inst["max_sl"],
        pip_value=inst["pip_value"],
        min_rr=MIN_RR,
        max_position_pct=RISK_PCT,
        max_open_positions=1,
        fee_pct=0.0001,
        slippage_pct=SLIPPAGE_PCT,
        partial_close=PartialCloseConfig.mtf_smc(),
    )


def _trade_stats(trades) -> dict:
    n = len(trades)
    if n == 0:
        return {"n": 0, "wins": 0, "wr": 0.0, "pnl": 0.0, "best": 0.0, "worst": 0.0}
    wins  = sum(1 for t in trades if t.realized_pnl > 0)
    pnl   = sum(t.realized_pnl for t in trades)
    best  = max(t.realized_pnl for t in trades)
    worst = min(t.realized_pnl for t in trades)
    return {
        "n": n, "wins": wins,
        "wr": 100 * wins / n,
        "pnl": pnl, "best": best, "worst": worst,
    }


def _sep(c: str = "─", w: int = 100) -> None:
    print(c * w)


def main() -> None:
    setup_logger("WARNING")

    _sep("═")
    print(f"  BACKTEST MULTI-INSTRUMENTS — Juin 2026  (NO_MSS, 1% risque)")
    _sep("═")
    print(f"  {'Instrument':10}  {'Trades':>6}  {'T/j':>4}  {'WR%':>5}  "
          f"{'PF':>6}  {'P&L':>8}  {'Ret%':>5}  {'MaxDD%':>6}  {'Sharpe':>7}  {'Temps':>6}")
    _sep()

    all_results = []

    for inst in INSTRUMENTS:
        t0 = time.time()
        print(f"  Chargement {inst['id']} …", end="", flush=True)

        try:
            df_m1 = load_m1_directory(inst["data_dir"])
        except FileNotFoundError as exc:
            print(f"\n  [SKIP] {exc}")
            continue

        tfs    = build_timeframes(df_m1)
        df_ltf = df_m1[START:END].copy()

        if len(df_ltf) == 0:
            print(f"\n  [SKIP] Aucune donnée pour {inst['id']} sur {START}→{END}")
            continue

        strat  = _make_strategy(inst, tfs)
        engine = _make_engine(inst, strat)
        result = engine.run(df_ltf, symbol=inst["id"])
        elapsed = time.time() - t0

        s      = result.summary()
        pf_raw = s["profit_factor"]
        pf     = "∞" if pf_raw == "inf" else (f"{pf_raw:.2f}" if pf_raw else "—")
        tpj    = s["n_trades"] / TRADING_DAYS

        print(f"\r  {inst['id']:10}  {s['n_trades']:>6}  {tpj:>4.2f}  "
              f"{s['win_rate']:>5.1f}  {pf:>6}  {s['net_pnl']:>+8.0f}  "
              f"{s['total_return_pct']:>+4.1f}%  {s['max_drawdown_pct']:>6.1f}%  "
              f"{s['sharpe_ratio']:>7.4f}  {elapsed:>5.0f}s")

        stats = _trade_stats(result.closed_trades)
        all_results.append({
            "instrument": inst["id"],
            "pip_value":  inst["pip_value"],
            "n_bars_m1":  len(df_ltf),
            "summary":    s,
            "trades":     result.closed_trades,
            "stats":      stats,
        })

    _sep("═")

    # ── Détail des trades par instrument ─────────────────────────────────
    for r in all_results:
        trades = r["trades"]
        if not trades:
            continue
        print(f"\n  TRADES — {r['instrument']}  (pip_value={r['pip_value']})")
        _sep("─", 80)
        print(f"  {'#':>3}  {'Dir':>5}  {'P&L':>9}  {'Statut':<20}  Raison signal")
        _sep("─", 80)
        for i, t in enumerate(trades, 1):
            direction = "LONG" if t.is_long else "SHORT"
            sgn       = "+" if t.realized_pnl >= 0 else ""
            reason    = (t.signal_reason or "")[:38]
            print(f"  {i:>3}  {direction:>5}  {sgn}{t.realized_pnl:>8.2f}$  "
                  f"{t.status.value:<20}  {reason}")
        _sep("─", 80)
        st = r["stats"]
        print(f"  Total: {st['n']} trades, {st['wins']} wins ({st['wr']:.1f}%)  "
              f"P&L net={st['pnl']:+.2f}$  "
              f"Meilleur={st['best']:+.2f}$  Pire={st['worst']:+.2f}$")
        _sep("═", 80)

    # ── Tableau comparatif ────────────────────────────────────────────────
    if all_results:
        print(f"\n  COMPARATIF — {START} → {END}  ({TRADING_DAYS} jours)")
        _sep("─", 70)
        print(f"  {'Instrument':10}  {'Trades':>6}  {'WR%':>5}  {'PF':>6}  "
              f"{'P&L':>8}  {'MaxDD%':>6}")
        _sep("─", 70)
        for r in all_results:
            s      = r["summary"]
            pf_raw = s["profit_factor"]
            pf     = "∞" if pf_raw == "inf" else (f"{pf_raw:.2f}" if pf_raw else "—")
            print(f"  {r['instrument']:10}  {s['n_trades']:>6}  "
                  f"{s['win_rate']:>5.1f}  {pf:>6}  "
                  f"{s['net_pnl']:>+8.0f}  {s['max_drawdown_pct']:>6.1f}%")
        _sep("═", 70)

    # ── Sauvegarde ────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "forex_june2026.json"
    with open(out_path, "w") as fh:
        json.dump(
            {
                "meta": {
                    "start": START, "end": END,
                    "trading_days": TRADING_DAYS,
                    "config": "NO_MSS",
                },
                "results": [
                    {
                        "instrument": r["instrument"],
                        "pip_value":  r["pip_value"],
                        "n_bars_m1":  r["n_bars_m1"],
                        "summary":    r["summary"],
                        "stats":      r["stats"],
                    }
                    for r in all_results
                ],
            },
            fh, indent=2, default=str,
        )
    print(f"\n  Résultats → {out_path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
