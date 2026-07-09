"""
ConfluenceScalpStrategy — validation brute XAUUSD multi-années (2024, 2025,
2026 H1) — étape 2 du plan : voir si le combo complet des 9-10
confirmations documentées par l'utilisateur a un edge réel, avant tout
habillage propfirm.

Cascade : 1H (zones HTF) → 15M (MSS) → 1M (entrée), comme ScalpSMCStrategy.

Usage
-----
    python -m zeus.backtest.run_confluence_scalp_backtest
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from zeus.backtest.advanced_engine import AdvancedBacktestEngine
from zeus.backtest.data_loader import build_timeframes, load_m1_directory
from zeus.backtest.partial_close import PartialCloseConfig
from zeus.strategy.confluence_scalp_strategy import ConfluenceScalpStrategy
from zeus.utils.logger import setup_logger

_ROOT       = Path(__file__).parent.parent.parent
DATA_DIR    = _ROOT / "data" / "historical" / "xauusd" / "m1"
RESULTS_DIR = _ROOT / "data" / "results"

SYMBOL        = "XAUUSD"
INITIAL_BAL   = 10_000.0
RISK_PCT      = 0.01
MAX_POSITIONS = 1
SL_PIPS       = 20.0
MAX_SL_PIPS   = 30.0
MIN_RR        = 1.5
SLIPPAGE_PCT  = 0.0002

PERIODS = [
    ("2026 H1", "2026-01-01", "2026-06-30"),
]


def main() -> None:
    setup_logger("INFO")

    print(f"Chargement M1 XAUUSD depuis {DATA_DIR} …")
    df_m1 = load_m1_directory(DATA_DIR)
    tfs   = build_timeframes(df_m1)
    df_15m, df_1h, df_1d = tfs["15min"], tfs["1h"], tfs["1d"]
    print(f"  M1 total : {len(df_m1):,} barres  ({df_m1.index[0]} → {df_m1.index[-1]})\n")

    all_trades = []
    for label, start, end in PERIODS:
        df_ltf = df_m1[start:end].copy()
        if df_ltf.empty:
            print(f"  ✗ {label} : pas de données")
            continue

        strategy = ConfluenceScalpStrategy(
            df_htf_1h=df_1h, df_mtf_15m=df_15m, df_daily=df_1d,
            sl_pips=SL_PIPS, max_sl_pips=MAX_SL_PIPS, pip_value=1.0,
        )
        engine = AdvancedBacktestEngine(
            strategy=strategy, initial_balance=INITIAL_BAL,
            stop_loss_pips=SL_PIPS, max_sl_pips=MAX_SL_PIPS, pip_value=1.0,
            min_rr=MIN_RR, max_position_pct=RISK_PCT,
            max_open_positions=MAX_POSITIONS, fee_pct=0.0,
            slippage_pct=SLIPPAGE_PCT, partial_close=PartialCloseConfig.scalp(),
        )
        result  = engine.run(df_ltf, symbol=SYMBOL)
        summary = result.summary()
        all_trades.append((label, summary, result.closed_trades))

        print(f"  {label:<10}  trades={summary['n_trades']:>4}  "
              f"WR={summary['win_rate']:>5.1f}%  PF={summary['profit_factor']:>6}  "
              f"PnL={summary['net_pnl']:>+9,.2f}$  DDmax={summary['max_drawdown_pct']:>5.2f}%")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "meta": {"symbol": SYMBOL, "strategy": "ConfluenceScalpStrategy",
                 "sl_pips": SL_PIPS, "max_sl_pips": MAX_SL_PIPS,
                 "periods": [p[0] for p in PERIODS]},
        "results": {label: summary for label, summary, _ in all_trades},
    }
    out_path = RESULTS_DIR / "confluence_scalp_xauusd_multi_year.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nRésultats sauvegardés → {out_path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
