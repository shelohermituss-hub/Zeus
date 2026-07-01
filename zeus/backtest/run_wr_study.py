"""
Étude WR + fréquence sur config NO_MSS — Apr-Mai 2026 (44 jours).

Problèmes à investiguer :
  1. Zone score trop bas  → accepte des zones faibles → pertes
  2. Zone tolerance trop serrée → rate des entrées au bord de zone
  3. Session London vs NY → l'une est-elle systématiquement perdante ?
  4. Clean approach trop ou pas assez strict → qualité d'entrée
  5. SL structurel vs fixe → SL trop serré = stop-out avant le move

Variantes testées (référence = NO_MSS baseline) :
  BASE       — NO_MSS tel quel (référence)
  SCORE4     — min_htf_score 3 → 4  (WR via zone quality)
  SCORE5     — min_htf_score 3 → 5  (WR encore plus strict)
  ZONE_WIDE  — zone_tolerance 0.003 → 0.006  (fréquence via zones plus larges)
  NO_APPROACH — sans clean approach gate  (fréquence)
  SWEEP10    — ltf_sweep_lookback 3 → 10  (fréquence)

Usage
-----
    python -m zeus.backtest.run_wr_study
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
        "id": "BASE",       "name": "NO_MSS référence",
        "score": 3.0, "zone_tol": 0.003, "approach": True,  "sweep_lb": 3,
    },
    {
        "id": "SCORE4",     "name": "score ≥ 4",
        "score": 4.0, "zone_tol": 0.003, "approach": True,  "sweep_lb": 3,
    },
    {
        "id": "SCORE5",     "name": "score ≥ 5",
        "score": 5.0, "zone_tol": 0.003, "approach": True,  "sweep_lb": 3,
    },
    {
        "id": "ZONE_WIDE",  "name": "zone tol 0.6%",
        "score": 3.0, "zone_tol": 0.006, "approach": True,  "sweep_lb": 3,
    },
    {
        "id": "NO_APPROACH","name": "sans approach gate",
        "score": 3.0, "zone_tol": 0.003, "approach": False, "sweep_lb": 3,
    },
    {
        "id": "SWEEP10",    "name": "sweep lookback 10",
        "score": 3.0, "zone_tol": 0.003, "approach": True,  "sweep_lb": 10,
    },
    {
        "id": "ALL_ZONES",  "name": "toutes zones (OB+OTE+FVG…)",
        "score": 3.0, "zone_tol": 0.003, "approach": True,  "sweep_lb": 3,
        "allowed_zones": None,   # toutes les zones
    },
]


def _make_strategy(v: dict, tfs: dict) -> ScalpSMCStrategy:
    return ScalpSMCStrategy(
        df_htf_1h=tfs["1h"],
        df_mtf_15m=None,                  # NO_MSS — gate désactivé
        df_daily=tfs["1d"],
        sl_pips=20.0,
        max_sl_pips=30.0,
        pip_value=1.0,
        min_htf_score=v["score"],
        swing_length=20,
        internal_length=5,
        atr_period=100,
        ltf_lookback=30,
        killzone_only=True,
        require_entry_fvg=True,
        require_choch_candle=False,
        allowed_zones=v.get("allowed_zones", ["OB", "OTE"]),
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


def _session_breakdown(trades) -> dict[str, dict]:
    """Analyse les trades par session d'entrée (London / NY / autre)."""
    from zeus.strategy.smc.session import killzone_name
    sessions: dict[str, dict] = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
    for t in trades:
        if t.opened_at:
            kz = killzone_name(pd.Timestamp(t.opened_at)) or "hors KZ"
        else:
            kz = "?"
        sessions[kz]["n"]    += 1
        sessions[kz]["pnl"]  += t.realized_pnl
        if t.realized_pnl > 0:
            sessions[kz]["wins"] += 1
    return dict(sessions)


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


def _sep(c: str = "─", w: int = 112) -> None:
    print(c * w)


def main() -> None:
    setup_logger("WARNING")

    print(f"\nLoading XAUUSD M1 data …")
    df_m1 = load_m1_directory(DATA_DIR)
    tfs   = build_timeframes(df_m1)
    df_ltf = df_m1[START:END].copy()
    print(f"  Période : {START} → {END}  ({len(df_ltf):,} barres M1, {TRADING_DAYS} jours)\n")

    _sep("═")
    print(f"  ÉTUDE WR + FRÉQUENCE — NO_MSS (1H→1M sans gate MSS)")
    _sep("═")
    print(f"  {'ID':12}  {'Nom':24}  {'Tr':>4}  {'T/j':>4}  "
          f"{'WR%':>5}  {'PF':>6}  {'P&L':>8}  {'MaxDD%':>6}  {'Sharpe':>7}")
    _sep()

    all_results = []
    htf_cache = None

    for v in VARIANTS:
        t0    = time.time()
        strat = _make_strategy(v, tfs)
        if htf_cache is not None:
            strat._htf_cache = htf_cache
        engine = _make_engine(strat)
        result = engine.run(df_ltf, symbol="XAUUSD")
        if htf_cache is None:
            htf_cache = strat._htf_cache
        elapsed = time.time() - t0

        s   = result.summary()
        pf  = f"{s['profit_factor']:.2f}" if s["profit_factor"] else "—"
        tpj = s["n_trades"] / TRADING_DAYS

        print(
            f"  {v['id']:12}  {v['name']:24}  {s['n_trades']:>4}  {tpj:>4.2f}  "
            f"{s['win_rate']:>5.1f}  {pf:>6}  {s['net_pnl']:>+8.0f}  "
            f"{s['max_drawdown_pct']:>6.1f}%  {s['sharpe_ratio']:>7.4f}"
            f"  ({elapsed:.0f}s)"
        )

        all_results.append({
            "variant":     v,
            "summary":     s,
            "trades":      result.closed_trades,
            "monthly_pnl": _monthly_pnl(result.closed_trades),
            "monthly_n":   _monthly_n(result.closed_trades),
            "sessions":    _session_breakdown(result.closed_trades),
        })

    _sep("═")

    # ── Tableau mensuel ──────────────────────────────────────────────────
    all_months = sorted({m for r in all_results for m in r["monthly_pnl"]})
    col = 10
    ids = [r["variant"]["id"] for r in all_results]
    print(f"\n  MENSUEL P&L ($) [n trades]")
    _sep("─", 85)
    print("  " + f"{'Mois':>7}  " + "  ".join(f"{i:>{col+3}}" for i in ids))
    _sep("─", 85)
    for month in all_months:
        row = f"  {month:>7}  "
        for r in all_results:
            pnl = r["monthly_pnl"].get(month)
            n   = r["monthly_n"].get(month, 0)
            if pnl is None:
                row += f"{'—':>{col}}      "
            else:
                sgn = "+" if pnl >= 0 else ""
                row += f"{sgn}{pnl:{col-1}.0f}$ [{n:>2}]  "
        print(row)
    _sep("─", 85)
    row = f"  {'TOTAL':>7}  "
    for r in all_results:
        total = sum(r["monthly_pnl"].values())
        n     = sum(r["monthly_n"].values())
        sgn = "+" if total >= 0 else ""
        row += f"{sgn}{total:{col-1}.0f}$ [{n:>2}]  "
    print(row)
    _sep("═", 85)

    # ── Détail des trades BASE ───────────────────────────────────────────
    base = next(r for r in all_results if r["variant"]["id"] == "BASE")
    print(f"\n  DÉTAIL TRADES — BASE (NO_MSS référence)")
    _sep("─", 80)
    print(f"  {'#':>3}  {'Dir':>5}  {'P&L':>8}  {'Statut':<20}  Raison signal")
    _sep("─", 80)
    for i, t in enumerate(base["trades"], 1):
        direction = "LONG" if t.is_long else "SHORT"
        sgn = "+" if t.realized_pnl >= 0 else ""
        reason = (t.signal_reason or "")[:40]
        print(
            f"  {i:>3}  {direction:>5}  "
            f"{sgn}{t.realized_pnl:>7.0f}$  {t.status.value:<20}  {reason}"
        )
    _sep("═", 80)

    # ── Sauvegarde ────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "wr_study_apr_may2026.json"
    with open(out_path, "w") as fh:
        json.dump(
            {
                "meta": {"start": START, "end": END, "trading_days": TRADING_DAYS},
                "variants": [
                    {
                        "id":          r["variant"]["id"],
                        "name":        r["variant"]["name"],
                        "config":      r["variant"],
                        "summary":     r["summary"],
                        "monthly_pnl": r["monthly_pnl"],
                        "sessions":    r["sessions"],
                    }
                    for r in all_results
                ],
            },
            fh, indent=2, default=str,
        )
    print(f"\n  Résultats → {out_path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
