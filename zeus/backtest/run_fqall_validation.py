"""
FQ_ALL validation — 2024 full-year + 2026 H1, tableau mensuel complet.

Config FQ_ALL (tous les leviers de fréquence) :
  - killzone_only=False       (toutes les heures)
  - require_entry_fvg=False   (pas de FVG 1M exigé)
  - require_clean_approach=False
  - ltf_sweep_lookback=15     (vs 3 en base)
  - SL=20/30, mtf_smc TP (1R BE, 3R 60%, 5R 85%, 10R 100%)

Comparé à PB_SL (config de référence gagnante) sur les mêmes périodes.

Usage
-----
    python -m zeus.backtest.run_fqall_validation
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

SYMBOL       = "XAUUSD"
INITIAL_BAL  = 10_000.0
RISK_PCT     = 0.01
MIN_RR       = 1.5
SLIPPAGE_PCT = 0.0002
ZONES        = ["OB", "OTE"]

PERIODS = [
    {"id": "2024", "start": "2024-01-01", "end": "2024-12-31", "trading_days": 261},
    {"id": "2026H1", "start": "2026-01-01", "end": "2026-06-30", "trading_days": 128},
]

CONFIGS = {
    "PB_SL": dict(
        killzone_only=True,
        require_entry_fvg=True,
        require_clean_approach=True,
        ltf_sweep_lookback=3,
    ),
    "FQ_ALL": dict(
        killzone_only=False,
        require_entry_fvg=False,
        require_clean_approach=False,
        ltf_sweep_lookback=15,
    ),
}


def _make_engine(strategy: ScalpSMCStrategy) -> AdvancedBacktestEngine:
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


def _make_strategy(cfg: dict, df_1h: pd.DataFrame, df_1d: pd.DataFrame) -> ScalpSMCStrategy:
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
        killzone_only=cfg["killzone_only"],
        require_entry_fvg=cfg["require_entry_fvg"],
        require_choch_candle=False,
        allowed_zones=ZONES,
        require_clean_approach=cfg["require_clean_approach"],
        approach_lookback=5,
        approach_max_momentum=0.6,
        approach_max_body_atr=1.5,
        require_poc_zone_confluence=False,
        require_session_sweep=False,
        require_ltf_sweep=True,
        ltf_sweep_lookback=cfg["ltf_sweep_lookback"],
        require_ote=False,
        require_daily_bias=False,
        require_entry_pattern=False,
        max_daily_signals=5,
        max_signals_per_session=2,
    )


def _sep(char: str = "─", w: int = 120) -> None:
    print(char * w)


def _monthly_pnl(trades) -> dict[str, float]:
    monthly: dict[str, float] = defaultdict(float)
    for t in trades:
        if t.closed_at:
            key = t.closed_at.strftime("%Y-%m")
            monthly[key] += t.realized_pnl
    return dict(monthly)


def _monthly_trades(trades) -> dict[str, int]:
    monthly: dict[str, int] = defaultdict(int)
    for t in trades:
        if t.closed_at:
            key = t.closed_at.strftime("%Y-%m")
            monthly[key] += 1
    return dict(monthly)


def main() -> None:
    setup_logger("WARNING")

    print(f"\nLoading XAUUSD M1 data from {DATA_DIR} …")
    df_m1 = load_m1_directory(DATA_DIR)
    tfs   = build_timeframes(df_m1)
    df_1h = tfs["1h"]
    df_1d = tfs["1d"]

    # ── Run all (period × config) combos ─────────────────────────────────────
    # Key: (period_id, config_id)
    results: dict[tuple[str, str], dict] = {}
    htf_cache: dict[str, dict] = {}  # keyed by period_id

    for period in PERIODS:
        pid   = period["id"]
        df_ltf = tfs["m1"][period["start"]:period["end"]].copy()
        print(f"\n{'─'*60}")
        print(f"Période {pid} : {len(df_ltf):,} barres M1  "
              f"({period['start']} → {period['end']})")

        for cfg_id, cfg in CONFIGS.items():
            t0 = time.time()
            strat = _make_strategy(cfg, df_1h, df_1d)

            # Partager le cache HTF entre les deux configs d'une même période
            if pid in htf_cache:
                strat._htf_cache = htf_cache[pid]

            engine = _make_engine(strat)
            result = engine.run(df_ltf, symbol=SYMBOL)

            if pid not in htf_cache:
                htf_cache[pid] = strat._htf_cache

            elapsed = time.time() - t0
            s = result.summary()
            pf = f"{s['profit_factor']:.2f}" if s["profit_factor"] else "—"
            tpj = s["n_trades"] / period["trading_days"]
            print(f"  {cfg_id:8} — {s['n_trades']:3d} trades  "
                  f"WR={s['win_rate']:4.1f}%  PF={pf:>6}  "
                  f"T/j={tpj:.2f}  P&L={s['net_pnl']:>+7.0f}$  ({elapsed:.0f}s)")

            results[(pid, cfg_id)] = {
                "summary": s,
                "trades":  result.closed_trades,
                "monthly_pnl":    _monthly_pnl(result.closed_trades),
                "monthly_trades": _monthly_trades(result.closed_trades),
                "trading_days":   period["trading_days"],
            }

    # ── Tableau récapitulatif global ──────────────────────────────────────────
    print(f"\n\n{'═'*120}")
    print("RÉCAPITULATIF GLOBAL")
    _sep("═")
    print(f"  {'Période':8}  {'Config':8}  {'Trades':>6}  {'T/jour':>6}  {'WR%':>5}  "
          f"{'PF':>6}  {'P&L':>8}  {'Ret%':>6}  {'MaxDD%':>7}  {'Sharpe':>8}")
    _sep("─")
    for period in PERIODS:
        pid = period["id"]
        for cfg_id in CONFIGS:
            r = results[(pid, cfg_id)]
            s = r["summary"]
            pf  = f"{s['profit_factor']:.2f}" if s["profit_factor"] else "—"
            tpj = s["n_trades"] / period["trading_days"]
            ok  = " ✓" if tpj >= 1.0 else f" ({tpj:.2f}/j)"
            print(
                f"  {pid:8}  {cfg_id:8}  {s['n_trades']:>6}  {tpj:>6.2f}  "
                f"{s['win_rate']:>5.1f}  {pf:>6}  "
                f"{s['net_pnl']:>+8.0f}  {s['total_return_pct']:>+5.1f}%  "
                f"{s['max_drawdown_pct']:>6.1f}%  {s['sharpe_ratio']:>8.4f}{ok}"
            )
        _sep("─")
    _sep("═")

    # ── Tableau mensuel P&L ───────────────────────────────────────────────────
    all_months = sorted({
        m
        for r in results.values()
        for m in r["monthly_pnl"]
    })

    col_w = 10
    headers = ["PB_SL 24", "FQ_ALL 24", "PB_SL 26", "FQ_ALL 26"]
    keys    = [("2024","PB_SL"), ("2024","FQ_ALL"), ("2026H1","PB_SL"), ("2026H1","FQ_ALL")]

    print("\n\nTABLEAU MENSUEL — P&L ($) et [trades]")
    _sep("─", 90)
    hdr = f"  {'Mois':>7}  " + "  ".join(f"{'P&L':>{col_w}} [N]" for _ in headers)
    lbl = f"  {'':>7}  " + "  ".join(f"{h:>{col_w+4}}" for h in headers)
    print(lbl)
    _sep("─", 90)

    yearly: dict[tuple[str,str], float] = defaultdict(float)
    yearly_n: dict[tuple[str,str], int] = defaultdict(int)

    for month in all_months:
        row = f"  {month:>7}  "
        for key in keys:
            r   = results.get(key, {})
            pnl = r.get("monthly_pnl", {}).get(month, None)
            n   = r.get("monthly_trades", {}).get(month, 0)
            if pnl is None:
                row += f"{'—':>{col_w}}  [-]  "
            else:
                sign = "+" if pnl >= 0 else ""
                row += f"{sign}{pnl:>{col_w-1}.0f}$  [{n:>2}]  "
                yearly[key]   += pnl
                yearly_n[key] += n
        print(row)

    _sep("─", 90)
    # Totaux annuels
    for year_label, year_keys in [("2024", [k for k in keys if k[0]=="2024"]),
                                   ("2026H1",[k for k in keys if k[0]=="2026H1"])]:
        row = f"  {f'TOT {year_label}':>7}  "
        for key in keys:
            if key in year_keys:
                pnl = yearly[key]
                n   = yearly_n[key]
                sign = "+" if pnl >= 0 else ""
                row += f"{sign}{pnl:>{col_w-1}.0f}$  [{n:>2}]  "
            else:
                row += f"{'':>{col_w}}  {'':>4}  "
        print(row)

    _sep("─", 90)
    # Totaux combinés (2024+2026H1)
    row = f"  {'TOTAL':>7}  "
    for key in keys:
        all_pnl = sum(results[key]["monthly_pnl"].values()) if key in results else 0
        all_n   = sum(results[key]["monthly_trades"].values()) if key in results else 0
        sign = "+" if all_pnl >= 0 else ""
        row += f"{sign}{all_pnl:>{col_w-1}.0f}$  [{all_n:>2}]  "
    print(row)
    _sep("═", 90)

    # ── Tableau mensuel WinRate ───────────────────────────────────────────────
    print("\n\nTABLEAU MENSUEL — Trades gagnants / perdants")
    _sep("─", 90)
    print(lbl)
    _sep("─", 90)

    monthly_wins:   dict[tuple[str,str], dict[str,int]] = {}
    monthly_losses: dict[tuple[str,str], dict[str,int]] = {}

    for key in keys:
        r = results.get(key, {})
        wins:   dict[str,int] = defaultdict(int)
        losses: dict[str,int] = defaultdict(int)
        for t in r.get("trades", []):
            if t.closed_at:
                m = t.closed_at.strftime("%Y-%m")
                if t.realized_pnl > 0:
                    wins[m]   += 1
                else:
                    losses[m] += 1
        monthly_wins[key]   = dict(wins)
        monthly_losses[key] = dict(losses)

    for month in all_months:
        row = f"  {month:>7}  "
        for key in keys:
            w = monthly_wins.get(key, {}).get(month, 0)
            l = monthly_losses.get(key, {}).get(month, 0)
            tot = w + l
            wr  = f"{100*w/tot:.0f}%" if tot > 0 else "—"
            row += f"  {w}W/{l}L {wr:>4}   "
        print(row)
    _sep("═", 90)

    # ── Analyse : mois positifs vs négatifs ──────────────────────────────────
    print("\n\nANALYSE — Mois positifs / négatifs")
    _sep("─", 60)
    for key in keys:
        pid, cfg_id = key
        monthly = results[key]["monthly_pnl"]
        pos = sum(1 for v in monthly.values() if v > 0)
        neg = sum(1 for v in monthly.values() if v < 0)
        total = pos + neg
        best  = max(monthly.values(), default=0)
        worst = min(monthly.values(), default=0)
        tpj   = results[key]["summary"]["n_trades"] / results[key]["trading_days"]
        print(f"  {pid:8} {cfg_id:8}  Pos={pos}/{total}  "
              f"Meilleur=+{best:.0f}$  Pire={worst:.0f}$  T/j={tpj:.2f}")
    _sep("═", 60)

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "meta": {
            "symbol": SYMBOL,
            "configs": list(CONFIGS.keys()),
            "periods": PERIODS,
            "goal": "≥1 trade/jour avec PF>1.5 et MaxDD<10%",
        },
        "results": {
            f"{pid}_{cfg_id}": {
                "summary": results[(pid, cfg_id)]["summary"],
                "monthly_pnl": results[(pid, cfg_id)]["monthly_pnl"],
                "monthly_trades": results[(pid, cfg_id)]["monthly_trades"],
            }
            for pid, cfg_id in results
        },
    }
    out_path = RESULTS_DIR / "fqall_validation_2024_2026.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nRésultats sauvegardés → {out_path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
