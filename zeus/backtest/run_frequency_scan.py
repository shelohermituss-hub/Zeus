"""
Frequency scan — optimisation du nombre de trades (objectif ≥ 1/jour).

Base : PB_SL (SL=20/30, mtf_smc TP) — 5 trades en avril 2026.
Objectif : ~22 trades/mois (1 par jour de trading).

Variantes testées
-----------------
  PB_SL     Base de référence (config gagnante actuelle)
  FQ_SWEEP  + ltf_sweep_lookback 3→15  (garde le filtre, plus souple)
  FQ_FVG    + require_entry_fvg=False  (pas de FVG 1M exigé)
  FQ_APP    + require_clean_approach=False  (permet approches impulsives)
  FQ_KZ     + killzone_only=False  (toutes les heures)
  FQ_ALL    tous les leviers combinés

Usage
-----
    python -m zeus.backtest.run_frequency_scan
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
TRADING_DAYS = 22  # jours ouvrés en avril 2026

# ── Variantes ─────────────────────────────────────────────────────────────────
VARIANTS: list[dict] = [
    {
        "id": "PB_SL",    "name": "Base (référence)",
        "killzone": True,  "entry_fvg": True,  "approach": True,  "sweep_lb": 3,
    },
    {
        "id": "FQ_SWEEP", "name": "sweep_lb=15",
        "killzone": True,  "entry_fvg": True,  "approach": True,  "sweep_lb": 15,
    },
    {
        "id": "FQ_FVG",   "name": "sans FVG 1M",
        "killzone": True,  "entry_fvg": False, "approach": True,  "sweep_lb": 3,
    },
    {
        "id": "FQ_APP",   "name": "sans approche propre",
        "killzone": True,  "entry_fvg": True,  "approach": False, "sweep_lb": 3,
    },
    {
        "id": "FQ_KZ",    "name": "sans kill zone",
        "killzone": False, "entry_fvg": True,  "approach": True,  "sweep_lb": 3,
    },
    {
        "id": "FQ_ALL",   "name": "tous les leviers",
        "killzone": False, "entry_fvg": False, "approach": False, "sweep_lb": 15,
    },
]


def _make_engine(v: dict, strategy: ScalpSMCStrategy) -> AdvancedBacktestEngine:
    return AdvancedBacktestEngine(
        strategy=strategy,
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


def _make_strategy(v: dict, df_1h: pd.DataFrame, df_1d: pd.DataFrame) -> ScalpSMCStrategy:
    return ScalpSMCStrategy(
        df_htf_1h=df_1h,
        df_daily=df_1d,
        sl_pips=20.0,
        max_sl_pips=30.0,
        pip_value=1.0,
        min_htf_score=3.0,
        swing_length=20,
        internal_length=5,
        atr_period=100,
        ltf_lookback=30,
        killzone_only=v["killzone"],
        require_entry_fvg=v["entry_fvg"],
        require_choch_candle=False,
        allowed_zones=ZONES,
        require_clean_approach=v["approach"],
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
    )


def _sep(char: str = "─", w: int = 118) -> None:
    print(char * w)


def main() -> None:
    setup_logger("WARNING")

    print(f"\nLoading XAUUSD M1 data from {DATA_DIR} …")
    df_m1  = load_m1_directory(DATA_DIR)
    tfs    = build_timeframes(df_m1)
    df_1h  = tfs["1h"]
    df_1d  = tfs["1d"]
    df_ltf = tfs["m1"][START_DATE:END_DATE].copy()
    print(f"  1M bars (avril 2026): {len(df_ltf):,}  |  jours de trading cible : {TRADING_DAYS}")

    # Phase 1 — base + cache HTF
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
        trades_per_day = s["n_trades"] / TRADING_DAYS
        target = "✓" if trades_per_day >= 1.0 else f"→{trades_per_day:.2f}/j"
        print(f"  {v['id']:10} {v['name']:28} — {s['n_trades']:3d} trades  "
              f"PF={pf:>6}  {target}")
        all_results.append((v, s, result.closed_trades))

    # ── Tableau comparatif ────────────────────────────────────────────────────
    print()
    _sep("═")
    print(f"  {'ID':10}  {'Leviers':30}  {'Trades':>6}  {'T/jour':>6}  {'WR%':>5}  "
          f"{'PF':>6}  {'P&L':>8}  {'Ret%':>6}  {'MaxDD%':>7}  {'Sharpe':>8}  {'Objectif':>8}")
    _sep("─")
    for v, s, _ in all_results:
        kz  = "KZ"    if not v["killzone"] else ".."
        fvg = "noFVG" if not v["entry_fvg"] else "....."
        app = "noAPP" if not v["approach"]  else "....."
        sw  = f"sw{v['sweep_lb']:02d}" if v["sweep_lb"] != 3 else "sw03"
        leviers = f"[{kz} {fvg} {app} {sw}]"
        pf   = f"{s['profit_factor']:.2f}" if s["profit_factor"] else "—"
        tpj  = s["n_trades"] / TRADING_DAYS
        ok   = "✓ ATTEINT" if tpj >= 1.0 else f"× {tpj:.2f}/j"
        print(
            f"  {v['id']:10}  {leviers:30}  {s['n_trades']:>6}  {tpj:>6.2f}  "
            f"{s['win_rate']:>5.1f}  {pf:>6}  "
            f"{s['net_pnl']:>+8.0f}  {s['total_return_pct']:>+5.1f}%  "
            f"{s['max_drawdown_pct']:>6.1f}%  {s['sharpe_ratio']:>8.4f}  {ok:>8}"
        )
    _sep("═")

    # ── Impact de chaque levier vs PB_SL ─────────────────────────────────────
    print("\nImpact de chaque levier vs PB_SL (base) :")
    base_s = all_results[0][1]
    base_trades = base_s["n_trades"]
    print(f"  {'ID':10}  {'Trades Δ':>9}  {'T/j Δ':>7}  {'WR Δ':>7}  {'PF Δ':>8}  {'P&L Δ':>8}  {'DD Δ':>8}")
    _sep("─", 80)
    for v, s, _ in all_results[1:]:
        dt   = s["n_trades"] - base_trades
        dtpj = (s["n_trades"] - base_trades) / TRADING_DAYS
        dwr  = s["win_rate"] - base_s["win_rate"]
        dpf  = (s["profit_factor"] or 0) - (base_s["profit_factor"] or 0)
        dp   = s["net_pnl"] - base_s["net_pnl"]
        dd   = s["max_drawdown_pct"] - base_s["max_drawdown_pct"]
        print(f"  {v['id']:10}  {dt:>+9d}  {dtpj:>+6.2f}/j  {dwr:>+6.1f}%  "
              f"{dpf:>+8.2f}  {dp:>+8.0f}  {dd:>+7.1f}%")
    _sep("═", 80)

    # ── Recommandation automatique ────────────────────────────────────────────
    print("\nRecommandation (≥1 trade/jour ET PF>1.5 ET MaxDD<10%) :")
    candidates = [
        (v, s) for v, s, _ in all_results
        if s["n_trades"] / TRADING_DAYS >= 1.0
        and (s["profit_factor"] or 0) > 1.5
        and s["max_drawdown_pct"] < 10.0
    ]
    if candidates:
        best = max(candidates, key=lambda x: (x[1]["profit_factor"] or 0))
        print(f"  ✓ Meilleure config : {best[0]['id']} — {best[0]['name']}")
        print(f"    {best[1]['n_trades']} trades  PF={best[1]['profit_factor']:.2f}  "
              f"WR={best[1]['win_rate']:.1f}%  MaxDD={best[1]['max_drawdown_pct']:.1f}%")
    else:
        print("  ✗ Aucune config n'atteint simultanément tous les critères.")
        print("    Vérifier le résultat FQ_ALL ou assouplir le seuil PF.")

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "meta": {
            "symbol": SYMBOL, "start": START_DATE, "end": END_DATE,
            "trading_days": TRADING_DAYS,
            "goal": "≥1 trade/jour avec PF>1.5 et MaxDD<10%",
        },
        "variants": [
            {"id": v["id"], "name": v["name"], "config": v, "summary": s}
            for v, s, _ in all_results
        ],
    }
    out_path = RESULTS_DIR / "frequency_scan_april2026.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nRésultats sauvegardés → {out_path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
