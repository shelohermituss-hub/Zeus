"""
ConfluenceScalpStrategy sur ticks réels (1s) — XAUUSD février 2026.

Teste plusieurs sous-ensembles des 9-10 confirmations documentées, du
combo complet ("A++", déjà connu pour ne jamais se déclencher — testé
comme référence/sanity check) jusqu'à des combos plus proches d'un "A+"
(6-7 confirmations), en réutilisant intégralement l'infrastructure
existante (MTFSMCStrategy + zeus/strategy/smc/*).

HTF (1H/15M/1D) : dérivé du M1 XAUUSD déjà présent dans le dépôt.
LTF (entrée)     : ticks réels rééchantillonnés en barres 1s
                   (zeus.backtest.tick_loader) — résolution la plus fine
                   disponible, pour juger de la précision d'entrée décrite.

Usage
-----
    python -m zeus.backtest.run_confluence_tick_variants
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from zeus.backtest.advanced_engine import AdvancedBacktestEngine
from zeus.backtest.data_loader import build_timeframes, load_m1_directory
from zeus.backtest.partial_close import PartialCloseConfig
from zeus.backtest.tick_loader import load_tick_directory, resample_ticks
from zeus.strategy.confluence import PatternGrade
from zeus.strategy.confluence_scalp_strategy import ConfluenceScalpStrategy
from zeus.utils.logger import setup_logger

_ROOT     = Path(__file__).parent.parent.parent
M1_DIR    = _ROOT / "data" / "historical" / "xauusd" / "m1"
TICK_DIR  = _ROOT / "data" / "ticks" / "xauusd" / "2026"
RESULTS_DIR = _ROOT / "data" / "results"

SYMBOL        = "XAUUSD"
INITIAL_BAL   = 10_000.0
RISK_PCT      = 0.01
MAX_POSITIONS = 1
SL_PIPS       = 20.0
MAX_SL_PIPS   = 30.0
MIN_RR        = 1.5
SLIPPAGE_PCT  = 0.0002
START, END    = "2026-02-01", "2026-02-27"

# ── Variantes : sous-ensembles des gates de ConfluenceScalpStrategy ──────────
# Chaque variante override un sous-ensemble des kwargs par défaut (tous à
# True dans ConfluenceScalpStrategy). "extra" ajoute des overrides directs
# sur les attributs internes après construction (pas tous exposés au niveau
# __init__ selon MTFSMCStrategy).

VARIANTS: dict[str, dict] = {
    "FULL (A++, référence)": dict(min_grade=PatternGrade.B),  # tout activé — sanity check
    "CORE (structure+OTE+POC+FVG+killzone)": dict(
        min_grade=PatternGrade.C,
        _off=["require_weekly_bias", "require_asian_sweep", "require_session_sweep",
              "require_choch_candle", "require_pd_filter", "require_entry_pattern",
              "require_ltf_sweep"],
    ),
    "CORE + sweep entrée (#8)": dict(
        min_grade=PatternGrade.C,
        _off=["require_weekly_bias", "require_asian_sweep", "require_session_sweep",
              "require_choch_candle", "require_pd_filter", "require_entry_pattern"],
    ),
    "CORE + modèle d'entrée complet (sweep+pattern+choch)": dict(
        min_grade=PatternGrade.C,
        _off=["require_weekly_bias", "require_asian_sweep", "require_session_sweep",
              "require_pd_filter"],
    ),
    "CORE + 1 seule liquidité externe (session sweep)": dict(
        min_grade=PatternGrade.C,
        _off=["require_weekly_bias", "require_asian_sweep",
              "require_choch_candle", "require_pd_filter", "require_entry_pattern",
              "require_ltf_sweep"],
    ),
}


def _build_strategy(df_1h, df_15m, df_1d, config: dict) -> ConfluenceScalpStrategy:
    # df_mtf_15m intentionally NOT passed: the 1H/15M MSS gate is always in
    # pullback when we want to enter (same conjunction problem documented in
    # run_scalp_backtest.py), producing 0 setups regardless of every other
    # gate — matches the existing project convention of disabling it.
    off = config.get("_off", [])
    kwargs = dict(min_htf_score=3.0, min_grade=config.get("min_grade", PatternGrade.B))
    strat = ConfluenceScalpStrategy(
        df_htf_1h=df_1h, df_mtf_15m=None, df_daily=df_1d,
        sl_pips=SL_PIPS, max_sl_pips=MAX_SL_PIPS, pip_value=1.0,
        **kwargs,
    )
    for attr in off:
        setattr(strat, f"_{attr}", False)
    return strat


def main() -> None:
    setup_logger("WARNING")

    print(f"Chargement M1 XAUUSD (contexte HTF) …")
    df_m1 = load_m1_directory(M1_DIR, glob_pattern="DAT_MT_XAUUSD_M1_2026*.csv")
    tfs   = build_timeframes(df_m1)
    df_15m, df_1h, df_1d = tfs["15min"], tfs["1h"], tfs["1d"]

    print(f"Chargement ticks XAUUSD février 2026 …")
    ticks = load_tick_directory(TICK_DIR, synthetic_spread=0.30)
    df_ltf = resample_ticks(ticks, freq="1s")[START:END]
    print(f"  Ticks     : {len(ticks):,}")
    print(f"  Barres 1s : {len(df_ltf):,}  ({df_ltf.index[0]} → {df_ltf.index[-1]})\n")

    results_out = {}
    for name, cfg in VARIANTS.items():
        strategy = _build_strategy(df_1h, df_15m, df_1d, cfg)
        engine = AdvancedBacktestEngine(
            strategy=strategy, initial_balance=INITIAL_BAL,
            stop_loss_pips=SL_PIPS, max_sl_pips=MAX_SL_PIPS, pip_value=1.0,
            min_rr=MIN_RR, max_position_pct=RISK_PCT,
            max_open_positions=MAX_POSITIONS, fee_pct=0.0,
            slippage_pct=SLIPPAGE_PCT, partial_close=PartialCloseConfig.scalp(),
        )
        result  = engine.run(df_ltf, symbol=SYMBOL)
        summary = result.summary()
        results_out[name] = summary
        print(f"{name:<55} trades={summary['n_trades']:>4}  WR={summary['win_rate']:>5.1f}%  "
              f"PF={summary['profit_factor']:>7}  PnL={summary['net_pnl']:>+9,.2f}$  "
              f"DDmax={summary['max_drawdown_pct']:>5.2f}%")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "confluence_tick_variants_xauusd_202602.json"
    with open(out_path, "w") as fh:
        json.dump({"meta": {"symbol": SYMBOL, "period": f"{START}..{END}",
                             "ltf": "ticks 1s"}, "results": results_out},
                  fh, indent=2, default=str)
    print(f"\nRésultats sauvegardés → {out_path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
