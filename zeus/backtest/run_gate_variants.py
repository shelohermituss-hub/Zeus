"""
Test des 4 leviers de fréquence sur TF1 (1H→15M→1M).

Variantes vs BASE (config de référence TF1) :
  V1  NO_MSS       — désactive le gate MSS 15M  (biggest lever: -71.9%)
  V2  NO_FVG       — désactive le FVG 1M requis  (-73.2% des barres en zone)
  V3  NO_BIAS      — désactive le filtre daily bias (-34.7%)
  V4  WIDE_SWEEP   — élargit le sweep lookback 3→10 (-47.3%)
  V5  ALL_OFF      — combine les 4 leviers

Période : avril-mai 2026 (~44 jours de trading)

Usage
-----
    python -m zeus.backtest.run_gate_variants
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
DATA_DIR    = _ROOT / "data" / "historical" / "xauusd" / "m1"
RESULTS_DIR = _ROOT / "data" / "results"

INITIAL_BAL  = 10_000.0
RISK_PCT     = 0.01
MIN_RR       = 1.5
SLIPPAGE_PCT = 0.0002

START        = "2026-04-01"
END          = "2026-05-31"
TRADING_DAYS = 44

VARIANTS = [
    {
        "id":   "BASE",
        "name": "TF1 baseline",
        "df_mtf":      True,   # utilise df_15m
        "entry_fvg":   True,
        "daily_bias":  True,   # df_daily fourni (soft mode)
        "sweep_lb":    3,
    },
    {
        "id":   "NO_MSS",
        "name": "sans MSS 15M",
        "df_mtf":      False,  # df_mtf=None → gate désactivé
        "entry_fvg":   True,
        "daily_bias":  True,
        "sweep_lb":    3,
    },
    {
        "id":   "NO_FVG",
        "name": "sans FVG 1M",
        "df_mtf":      True,
        "entry_fvg":   False,
        "daily_bias":  True,
        "sweep_lb":    3,
    },
    {
        "id":   "NO_BIAS",
        "name": "sans daily bias",
        "df_mtf":      True,
        "entry_fvg":   True,
        "daily_bias":  False,  # df_daily=None → gate désactivé
        "sweep_lb":    3,
    },
    {
        "id":   "WIDE_SWEEP",
        "name": "sweep lookback 10",
        "df_mtf":      True,
        "entry_fvg":   True,
        "daily_bias":  True,
        "sweep_lb":    10,
    },
    {
        "id":   "ALL_OFF",
        "name": "tous les leviers",
        "df_mtf":      False,
        "entry_fvg":   False,
        "daily_bias":  False,
        "sweep_lb":    10,
    },
]


def _make_strategy(v: dict, tfs: dict) -> ScalpSMCStrategy:
    return ScalpSMCStrategy(
        df_htf_1h=tfs["1h"],
        df_mtf_15m=tfs["15min"] if v["df_mtf"] else None,
        df_daily=tfs["1d"] if v["daily_bias"] else None,
        sl_pips=20.0,
        max_sl_pips=30.0,
        pip_value=1.0,
        min_htf_score=3.0,
        swing_length=20,
        internal_length=5,
        atr_period=100,
        ltf_lookback=30,
        killzone_only=True,
        require_entry_fvg=v["entry_fvg"],
        require_choch_candle=False,
        allowed_zones=["OB", "OTE"],
        require_clean_approach=True,
        approach_lookback=5,
        approach_max_momentum=0.6,
        approach_max_body_atr=1.5,
        require_poc_zone_confluence=False,
        require_session_sweep=False,
        require_ltf_sweep=True,
        ltf_sweep_lookback=v["sweep_lb"],
        require_ote=False,
        require_daily_bias=False,
        require_entry_pattern=False,
        max_daily_signals=5,
        max_signals_per_session=2,
        mss_lookback=20,
    )


def _make_engine(strat: ScalpSMCStrategy) -> AdvancedBacktestEngine:
    return AdvancedBacktestEngine(
        strategy=strat,
        initial_balance=INITIAL_BAL,
        stop_loss_pips=20.0,
        max_sl_pips=30.0,
        pip_value=1.0,
        min_rr=MIN_RR,
        max_position_pct=RISK_PCT,
        max_open_positions=1,
        fee_pct=0.0001,
        slippage_pct=SLIPPAGE_PCT,
        partial_close=PartialCloseConfig.mtf_smc(),
    )


def _monthly_pnl(trades) -> dict[str, float]:
    d: dict[str, float] = defaultdict(float)
    for t in trades:
        if t.closed_at:
            d[t.closed_at.strftime("%Y-%m")] += t.realized_pnl
    return dict(d)


def _monthly_n(trades) -> dict[str, int]:
    d: dict[str, int] = defaultdict(int)
    for t in trades:
        if t.closed_at:
            d[t.closed_at.strftime("%Y-%m")] += 1
    return dict(d)


def _sep(c: str = "─", w: int = 110) -> None:
    print(c * w)


def main() -> None:
    setup_logger("WARNING")

    print(f"\nLoading XAUUSD M1 data …")
    df_m1 = load_m1_directory(DATA_DIR)
    tfs   = build_timeframes(df_m1)
    df_ltf = df_m1[START:END].copy()
    print(f"  Période : {START} → {END}  ({len(df_ltf):,} barres M1, {TRADING_DAYS} jours)\n")

    _sep("═")
    print(f"  VARIANTES — leviers de fréquence sur TF1 (1H→15M→1M)")
    _sep("═")
    print(f"  {'ID':12}  {'Nom':22}  {'Trades':>7}  {'T/j':>4}  {'WR%':>5}  "
          f"{'PF':>6}  {'P&L':>8}  {'Ret%':>5}  {'MaxDD%':>6}  {'Sharpe':>7}")
    _sep()

    all_results = []
    htf_cache = None  # Toutes les variantes partagent le même cache 1H

    for v in VARIANTS:
        t0    = time.time()
        strat = _make_strategy(v, tfs)

        # Partager le cache HTF 1H (seule la MSS et la daily changent)
        if htf_cache is not None:
            strat._htf_cache = htf_cache

        engine = _make_engine(strat)
        result = engine.run(df_ltf, symbol="XAUUSD")

        # Sauvegarder le cache après le premier run (BASE)
        if htf_cache is None:
            htf_cache = strat._htf_cache

        elapsed = time.time() - t0
        s       = result.summary()
        pf      = f"{s['profit_factor']:.2f}" if s["profit_factor"] else "—"
        tpj     = s["n_trades"] / TRADING_DAYS
        flag    = " ✓" if tpj >= 1.0 else f" ({tpj:.2f}/j)"

        print(
            f"  {v['id']:12}  {v['name']:22}  {s['n_trades']:>7}  {tpj:>4.2f}  "
            f"{s['win_rate']:>5.1f}  {pf:>6}  "
            f"{s['net_pnl']:>+8.0f}  {s['total_return_pct']:>+4.1f}%  "
            f"{s['max_drawdown_pct']:>6.1f}%  {s['sharpe_ratio']:>7.4f}"
            f"{flag}  ({elapsed:.0f}s)"
        )

        all_results.append({
            "variant":     v,
            "summary":     s,
            "trades":      result.closed_trades,
            "monthly_pnl": _monthly_pnl(result.closed_trades),
            "monthly_n":   _monthly_n(result.closed_trades),
        })

    _sep("═")

    # ── Tableau mensuel ──────────────────────────────────────────────────
    all_months = sorted({m for r in all_results for m in r["monthly_pnl"]})
    if all_months:
        ids = [r["variant"]["id"] for r in all_results]
        col = 10
        print(f"\n  TABLEAU MENSUEL P&L ($) [trades]")
        _sep("─", 80)
        print("  " + f"{'Mois':>7}  " + "  ".join(f"{i:>{col+3}}" for i in ids))
        _sep("─", 80)
        for month in all_months:
            row = f"  {month:>7}  "
            for r in all_results:
                pnl = r["monthly_pnl"].get(month)
                n   = r["monthly_n"].get(month, 0)
                if pnl is None:
                    row += f"{'—':>{col}}      "
                else:
                    s = "+" if pnl >= 0 else ""
                    row += f"{s}{pnl:{col-1}.0f}$ [{n:>2}]  "
            print(row)
        _sep("─", 80)
        row = f"  {'TOTAL':>7}  "
        for r in all_results:
            total = sum(r["monthly_pnl"].values())
            n     = sum(r["monthly_n"].values())
            s = "+" if total >= 0 else ""
            row += f"{s}{total:{col-1}.0f}$ [{n:>2}]  "
        print(row)
        _sep("═", 80)

    # ── Recommandation ────────────────────────────────────────────────────
    print("\n  RECOMMANDATION (PF>1.5, MaxDD<10%, ≥1 trade/jour) :")
    _sep("─", 60)
    candidates = [
        r for r in all_results
        if (r["summary"]["profit_factor"] or 0) > 1.5
        and r["summary"]["max_drawdown_pct"] < 10.0
        and r["summary"]["n_trades"] / TRADING_DAYS >= 1.0
    ]
    if candidates:
        best = max(candidates, key=lambda x: x["summary"]["profit_factor"] or 0)
        v, s = best["variant"], best["summary"]
        print(f"  ★  {v['id']} — {v['name']}")
        print(f"     {s['n_trades']} trades  WR={s['win_rate']:.1f}%  "
              f"PF={s['profit_factor']:.2f}  P&L={s['net_pnl']:+.0f}$  "
              f"MaxDD={s['max_drawdown_pct']:.1f}%")
    else:
        print("  Aucun candidat ≥1 trade/jour avec PF>1.5. "
              "Assouplir les critères ou combiner les leviers.")
    _sep("═", 60)

    # ── Sauvegarde ────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "meta":  {"start": START, "end": END, "trading_days": TRADING_DAYS},
        "variants": [
            {
                "id":          r["variant"]["id"],
                "name":        r["variant"]["name"],
                "config":      r["variant"],
                "summary":     r["summary"],
                "monthly_pnl": r["monthly_pnl"],
                "monthly_n":   r["monthly_n"],
            }
            for r in all_results
        ],
    }
    out_path = RESULTS_DIR / "gate_variants_apr_may2026.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\n  Résultats → {out_path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
