"""
Timeframe scan — trouver la cascade HTF→MTF→Entry optimale.

Contexte : stratégie intraday scalp SMC où le 3R arrive en ~5 min.
La cascade actuelle (1H→15M→1M) est peut-être trop large — les zones 1H
sont rares. On teste des cascades plus courtes pour augmenter les setups
sans sacrifier la qualité.

Variantes testées (HTF zones | MTF MSS | Entry bars)
-----------------------------------------------------
  TF1   1H  | 15M | 1M   ← baseline actuel
  TF2   1H  | 5M  | 1M   ← MSS plus fin
  TF3   30M | 15M | 1M   ← HTF plus court = plus de zones
  TF4   30M | 5M  | 1M   ← 30M zones, MSS 5M
  TF5   15M | 5M  | 1M   ← encore plus rapide (sw=30 ≈ 7.5h)
  TF6   15M | 3M  | 1M   ← ultra-fin
  TF7   4H  | 15M | 1M   ← contexte macro (sw=10 ≈ 40h)

swing_length adapté au HTF :
  4H  → sw=10  (40h ≈ 5 jours)
  1H  → sw=20  (20h ≈ 2-3 jours)
  30M → sw=20  (10h ≈ 1.5 jour)
  15M → sw=30  (7.5h ≈ 1 journée)

Usage
-----
    python -m zeus.backtest.run_timeframe_scan [april2026|2026q1|2026q2|2024|2026h1]
"""
from __future__ import annotations

import json
import sys
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

PERIODS = {
    "april2026": {"start": "2026-04-01", "end": "2026-04-30", "trading_days": 22,  "entry_tf": "m1"},
    "2026q2":    {"start": "2026-04-01", "end": "2026-06-30", "trading_days": 65,  "entry_tf": "m1"},
    "2026q1":    {"start": "2026-01-01", "end": "2026-03-31", "trading_days": 63,  "entry_tf": "m1"},
    "2024":      {"start": "2024-01-01", "end": "2024-12-31", "trading_days": 261, "entry_tf": "m1"},
    "2026h1":    {"start": "2026-01-01", "end": "2026-06-30", "trading_days": 128, "entry_tf": "m1"},
}

# ── Variantes : (htf_key, mtf_key, entry_key, swing_length, atr_period, mss_lookback) ──
VARIANTS: list[dict] = [
    {
        "id": "TF1",  "name": "1H | 15M | 1M  (baseline)",
        "htf": "1h",   "mtf": "15min", "entry": "m1",
        "swing": 20,   "atr": 100,     "mss_lb": 20,
    },
    {
        "id": "TF2",  "name": "1H | 5M  | 1M",
        "htf": "1h",   "mtf": "5min",  "entry": "m1",
        "swing": 20,   "atr": 100,     "mss_lb": 20,
    },
    {
        "id": "TF3",  "name": "30M | 15M | 1M",
        "htf": "30min","mtf": "15min", "entry": "m1",
        "swing": 20,   "atr": 100,     "mss_lb": 20,
    },
    {
        "id": "TF4",  "name": "30M | 5M  | 1M",
        "htf": "30min","mtf": "5min",  "entry": "m1",
        "swing": 20,   "atr": 100,     "mss_lb": 20,
    },
    {
        "id": "TF5",  "name": "15M | 5M  | 1M",
        "htf": "15min","mtf": "5min",  "entry": "m1",
        "swing": 30,   "atr": 200,     "mss_lb": 20,
    },
    {
        "id": "TF6",  "name": "15M | 3M  | 1M",
        "htf": "15min","mtf": "3min",  "entry": "m1",
        "swing": 30,   "atr": 200,     "mss_lb": 30,
    },
    {
        "id": "TF7",  "name": "4H  | 15M | 1M",
        "htf": "4h",   "mtf": "15min", "entry": "m1",
        "swing": 10,   "atr": 50,      "mss_lb": 20,
    },
]


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


def _make_strategy(v: dict, tfs: dict, df_1d: pd.DataFrame) -> ScalpSMCStrategy:
    return ScalpSMCStrategy(
        df_htf_1h=tfs[v["htf"]],       # HTF zones (nom trompeur — c'est n'importe quel TF)
        df_mtf_15m=tfs[v["mtf"]],      # MTF MSS gate
        df_daily=df_1d,
        sl_pips=20.0,
        max_sl_pips=30.0,
        pip_value=1.0,
        min_htf_score=3.0,
        swing_length=v["swing"],
        internal_length=5,
        atr_period=v["atr"],
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
        require_ote=False,
        require_daily_bias=False,
        require_entry_pattern=False,
        max_daily_signals=5,
        max_signals_per_session=2,
        mss_lookback=v["mss_lb"],
    )


def _sep(char: str = "─", w: int = 115) -> None:
    print(char * w)


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


def run_period(period_id: str, tfs: dict, variants: list[dict] | None = None) -> list[dict]:
    period   = PERIODS[period_id]
    df_ltf   = tfs[period["entry_tf"]][period["start"]:period["end"]].copy()
    df_1d    = tfs["1d"]
    td       = period["trading_days"]
    variants = variants if variants is not None else VARIANTS

    print(f"\n{'═'*115}")
    print(f"PÉRIODE : {period_id.upper()}  "
          f"({period['start']} → {period['end']})  "
          f"|  {len(df_ltf):,} barres {period['entry_tf']}  "
          f"|  {td} jours de trading")
    _sep("═")

    all_results = []
    htf_cache_by_tf: dict[str, dict] = {}

    for v in variants:
        t0    = time.time()
        strat = _make_strategy(v, tfs, df_1d)

        # Partager le cache HTF si même HTF TF déjà calculé
        if v["htf"] in htf_cache_by_tf:
            strat._htf_cache = htf_cache_by_tf[v["htf"]]

        engine = _make_engine(strat)
        result = engine.run(df_ltf, symbol="XAUUSD")

        # Sauvegarder le cache pour réutilisation
        htf_cache_by_tf[v["htf"]] = strat._htf_cache

        elapsed = time.time() - t0
        s       = result.summary()
        pf      = f"{s['profit_factor']:.2f}" if s["profit_factor"] else "—"
        tpj     = s["n_trades"] / td
        ok      = "✓" if tpj >= 1.0 else f"({tpj:.2f}/j)"

        print(
            f"  {v['id']:5}  {v['name']:28}  "
            f"{s['n_trades']:>4} trades  "
            f"WR={s['win_rate']:4.1f}%  PF={pf:>6}  "
            f"P&L={s['net_pnl']:>+7.0f}$  "
            f"MaxDD={s['max_drawdown_pct']:4.1f}%  "
            f"{ok}  ({elapsed:.0f}s)"
        )

        all_results.append({
            "variant":  v,
            "summary":  s,
            "trades":   result.closed_trades,
            "monthly_pnl": _monthly_pnl(result.closed_trades),
            "monthly_n":   _monthly_n(result.closed_trades),
        })

    return all_results


def print_comparison(all_results: list[dict], trading_days: int) -> None:
    _sep("═")
    print(f"\n{'COMPARAISON GLOBALE':^115}")
    _sep("═")
    print(f"  {'ID':5}  {'Cascade':28}  {'Trades':>6}  {'T/j':>4}  {'WR%':>5}  "
          f"{'PF':>6}  {'P&L':>8}  {'Ret%':>5}  {'MaxDD%':>6}  {'Sharpe':>7}")
    _sep("─")

    # Trier par PF décroissant
    sorted_r = sorted(all_results, key=lambda x: x["summary"]["profit_factor"] or 0, reverse=True)
    for r in sorted_r:
        v  = r["variant"]
        s  = r["summary"]
        pf = f"{s['profit_factor']:.2f}" if s["profit_factor"] else "—"
        tpj = s["n_trades"] / trading_days
        star = " ★" if (s["profit_factor"] or 0) == max(
            (x["summary"]["profit_factor"] or 0) for x in all_results
        ) else ""
        print(
            f"  {v['id']:5}  {v['name']:28}  {s['n_trades']:>6}  {tpj:>4.2f}  "
            f"{s['win_rate']:>5.1f}  {pf:>6}  "
            f"{s['net_pnl']:>+8.0f}  {s['total_return_pct']:>+4.1f}%  "
            f"{s['max_drawdown_pct']:>6.1f}%  {s['sharpe_ratio']:>7.4f}{star}"
        )
    _sep("═")


def print_monthly_table(all_results: list[dict]) -> None:
    all_months = sorted({
        m for r in all_results for m in r["monthly_pnl"]
    })
    if not all_months:
        return

    ids = [r["variant"]["id"] for r in all_results]
    col = 10

    print(f"\n{'TABLEAU MENSUEL P&L ($) [trades]':^115}")
    _sep("─", 90)
    hdr = "  " + f"{'Mois':>7}  " + "  ".join(f"{i:>{col+3}}" for i in ids)
    print(hdr)
    _sep("─", 90)

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

    _sep("─", 90)
    row = f"  {'TOTAL':>7}  "
    for r in all_results:
        total = sum(r["monthly_pnl"].values())
        n     = sum(r["monthly_n"].values())
        s = "+" if total >= 0 else ""
        row += f"{s}{total:{col-1}.0f}$ [{n:>2}]  "
    print(row)
    _sep("═", 90)


def main() -> None:
    setup_logger("WARNING")

    args = sys.argv[1:]
    period_id = args[0] if args else "april2026"
    if period_id not in PERIODS:
        print(f"Période inconnue. Choix : {list(PERIODS)}")
        sys.exit(1)

    # Optional second arg: comma-separated variant IDs to run (e.g. "TF1,TF2,TF3")
    active_variants = VARIANTS
    if len(args) > 1:
        variant_filter = {v.strip().upper() for v in args[1].split(",")}
        unknown = variant_filter - {v["id"] for v in VARIANTS}
        if unknown:
            print(f"Variante(s) inconnue(s) : {unknown}")
            sys.exit(1)
        active_variants = [v for v in VARIANTS if v["id"] in variant_filter]

    print(f"\nLoading XAUUSD M1 data from {DATA_DIR} …")
    df_m1 = load_m1_directory(DATA_DIR)
    tfs   = build_timeframes(df_m1)
    print(f"  Timeframes disponibles : {list(tfs)}")

    all_results = run_period(period_id, tfs, active_variants)
    print_comparison(all_results, PERIODS[period_id]["trading_days"])
    print_monthly_table(all_results)

    # Recommandation automatique
    print("\nRECOMMANDATION (PF>1.5, MaxDD<8%, ≥2 trades/mois) :")
    _sep("─", 60)
    candidates = [
        r for r in all_results
        if (r["summary"]["profit_factor"] or 0) > 1.5
        and r["summary"]["max_drawdown_pct"] < 8.0
        and r["summary"]["n_trades"] >= 2
    ]
    if candidates:
        best = max(candidates, key=lambda x: x["summary"]["profit_factor"] or 0)
        v, s = best["variant"], best["summary"]
        print(f"  ★  {v['id']} — {v['name']}")
        print(f"     {s['n_trades']} trades  WR={s['win_rate']:.1f}%  PF={s['profit_factor']:.2f}"
              f"  P&L={s['net_pnl']:+.0f}$  MaxDD={s['max_drawdown_pct']:.1f}%")
        print(f"     Cascade : HTF={v['htf']} (swing={v['swing']})  "
              f"MTF={v['mtf']} (mss_lb={v['mss_lb']})  Entry={v['entry']}")
    else:
        print("  Aucun candidat. Assouplir les critères ou tester d'autres periodes.")
    _sep("═", 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "meta": {
            "period": period_id,
            "start": PERIODS[period_id]["start"],
            "end":   PERIODS[period_id]["end"],
            "goal":  "cascade HTF/MTF/Entry optimale pour scalp intraday",
        },
        "variants": [
            {
                "id": r["variant"]["id"],
                "name": r["variant"]["name"],
                "config": r["variant"],
                "summary": r["summary"],
                "monthly_pnl": r["monthly_pnl"],
                "monthly_n": r["monthly_n"],
            }
            for r in all_results
        ],
    }
    out_path = RESULTS_DIR / f"timeframe_scan_{period_id}.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nRésultats → {out_path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
