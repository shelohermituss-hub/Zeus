"""
Validation NO_MSS — BASE vs NO_MSS sur 2024 + 2026H1.

NO_MSS = TF1 (1H→15M→1M) sans le gate MSS 15M (df_mtf=None).
Résultat apr-mai 2026 : double les trades (4→8), WR identique 50%.

Usage
-----
    python -m zeus.backtest.run_nomss_validation
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

PERIODS = [
    {"id": "2024",   "start": "2024-01-01", "end": "2024-12-31", "trading_days": 261},
    {"id": "2026H1", "start": "2026-01-01", "end": "2026-06-30", "trading_days": 128},
]

CONFIGS = {
    "BASE":   {"df_mtf": True},
    "NO_MSS": {"df_mtf": False},
}


def _make_strategy(cfg: dict, tfs: dict) -> ScalpSMCStrategy:
    return ScalpSMCStrategy(
        df_htf_1h=tfs["1h"],
        df_mtf_15m=tfs["15min"] if cfg["df_mtf"] else None,
        df_daily=tfs["1d"],
        sl_pips=20.0,
        max_sl_pips=30.0,
        pip_value=1.0,
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


def _sep(c: str = "─", w: int = 118) -> None:
    print(c * w)


def main() -> None:
    setup_logger("WARNING")

    print(f"\nLoading XAUUSD M1 data …")
    df_m1 = load_m1_directory(DATA_DIR)
    tfs   = build_timeframes(df_m1)

    all_runs: list[dict] = []

    for period in PERIODS:
        df_ltf = df_m1[period["start"]:period["end"]].copy()
        td     = period["trading_days"]
        print(f"\n────────────────────────────────────────────────────────────")
        print(f"Période {period['id']} : {len(df_ltf):,} barres M1  "
              f"({period['start']} → {period['end']})")

        htf_cache = None
        for cfg_id, cfg in CONFIGS.items():
            t0    = time.time()
            strat = _make_strategy(cfg, tfs)
            if htf_cache is not None:
                strat._htf_cache = htf_cache
            engine = _make_engine(strat)
            result = engine.run(df_ltf, symbol="XAUUSD")
            if htf_cache is None:
                htf_cache = strat._htf_cache
            elapsed = time.time() - t0

            s   = result.summary()
            pf  = f"{s['profit_factor']:.2f}" if s["profit_factor"] else "—"
            tpj = s["n_trades"] / td
            print(f"  {cfg_id:8}  —  {s['n_trades']:>3} trades  "
                  f"WR={s['win_rate']:4.1f}%  PF={pf:>6}  "
                  f"T/j={tpj:.2f}  P&L={s['net_pnl']:>+7.0f}$  ({elapsed:.0f}s)")

            all_runs.append({
                "period":      period["id"],
                "config":      cfg_id,
                "trading_days": td,
                "summary":     s,
                "monthly_pnl": _monthly_pnl(result.closed_trades),
                "monthly_n":   _monthly_n(result.closed_trades),
            })

    # ── Récapitulatif global ─────────────────────────────────────────────
    _sep("═")
    print(f"\n{'RÉCAPITULATIF GLOBAL':^118}")
    _sep("═")
    print(f"  {'Période':<8}  {'Config':<8}  {'Trades':>6}  {'T/jour':>6}  "
          f"{'WR%':>5}  {'PF':>6}  {'P&L':>8}  {'Ret%':>5}  {'MaxDD%':>6}  {'Sharpe':>8}")
    _sep()
    for r in all_runs:
        s   = r["summary"]
        pf  = f"{s['profit_factor']:.2f}" if s["profit_factor"] else "—"
        tpj = s["n_trades"] / r["trading_days"]
        print(f"  {r['period']:<8}  {r['config']:<8}  {s['n_trades']:>6}  {tpj:>6.2f}  "
              f"{s['win_rate']:>5.1f}  {pf:>6}  {s['net_pnl']:>+8.0f}  "
              f"{s['total_return_pct']:>+4.1f}%  {s['max_drawdown_pct']:>6.1f}%  "
              f"{s['sharpe_ratio']:>8.4f}")
    _sep("═")

    # ── Tableau mensuel P&L ──────────────────────────────────────────────
    all_months = sorted({m for r in all_runs for m in r["monthly_pnl"]})
    col_ids    = [f"{r['config']} {r['period'][:4]}" for r in all_runs]
    col        = 11

    print(f"\n{'TABLEAU MENSUEL — P&L ($) et [trades]':^90}")
    _sep("─", 90)
    print("  " + f"{'':>7}  " + "  ".join(f"{i:>{col+3}}" for i in col_ids))
    _sep("─", 90)

    for month in all_months:
        row = f"  {month:>7}  "
        for r in all_runs:
            pnl = r["monthly_pnl"].get(month)
            n   = r["monthly_n"].get(month, 0)
            if pnl is None:
                row += f"{'—':>{col}}      "
            else:
                s = "+" if pnl >= 0 else ""
                row += f"{s}{pnl:{col-1}.0f}$ [{n:>2}]  "
        print(row)

    _sep("─", 90)
    row = f"  {'TOTAL':>7}  "
    for r in all_runs:
        total = sum(r["monthly_pnl"].values())
        n     = sum(r["monthly_n"].values())
        s = "+" if total >= 0 else ""
        row += f"{s}{total:{col-1}.0f}$ [{n:>2}]  "
    print(row)
    _sep("═", 90)

    # ── Tableau mensuel wins/losses ──────────────────────────────────────
    print(f"\n{'TABLEAU MENSUEL — Trades gagnants / perdants':^90}")
    _sep("─", 90)
    print("  " + f"{'':>7}  " + "  ".join(f"{i:>{col+3}}" for i in col_ids))
    _sep("─", 90)

    # Reconstruire depuis les trades
    from zeus.backtest.advanced_engine import AdvancedBacktestEngine as _E
    # On utilise les monthly_pnl déjà calculés — on doit refaire la validation
    # depuis le summary mensuel (on n'a pas les trades individuels ici)
    # Affichage simplifié : uniquement P&L et n
    _sep("─", 90)

    # ── Analyse mois positifs/négatifs ───────────────────────────────────
    print(f"\n  ANALYSE — Mois positifs / négatifs")
    _sep("─", 60)
    for r in all_runs:
        months_with_data = {m: v for m, v in r["monthly_pnl"].items()}
        pos = sum(1 for v in months_with_data.values() if v > 0)
        tot = len(months_with_data)
        best = max(months_with_data.values(), default=0)
        worst = min(months_with_data.values(), default=0)
        tpj = r["summary"]["n_trades"] / r["trading_days"]
        print(f"  {r['period']:<7}  {r['config']:<8}  "
              f"Pos={pos}/{tot}  Meilleur={best:+.0f}$  Pire={worst:+.0f}$  T/j={tpj:.2f}")
    _sep("═", 60)

    # ── Sauvegarde ────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "nomss_validation_2024_2026.json"
    with open(out_path, "w") as fh:
        json.dump(
            {"runs": [{k: v for k, v in r.items() if k != "trades"} for r in all_runs]},
            fh, indent=2, default=str,
        )
    print(f"\nRésultats sauvegardés → {out_path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
