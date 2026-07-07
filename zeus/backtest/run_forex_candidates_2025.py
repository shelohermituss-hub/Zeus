"""
Validation des paires candidates au portefeuille propfirm.
==========================================================

Protocole : config forex WS7.5 validée (identique au cluster GBPUSD/EURUSD/
GBPAUD/EURNZD) appliquée telle quelle aux nouvelles paires — aucune
optimisation par paire, pour éviter l'overfitting.  Une paire est admise si
elle est rentable avec la config gelée sur toutes ses années disponibles.

Critères d'admission (par paire, toutes années confondues) :
  WR ≥ 55% · R total > 0 · DD ≤ 8% · N ≥ 10 trades/an

Usage
-----
    python -m zeus.backtest.run_forex_candidates_2025
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd

from zeus.backtest.data_loader import parse_m1_csv, resample_ohlcv
from zeus.backtest.sd_simulation import simulate_all
from zeus.strategy.supply_demand.sd_strategy import SDStrategy
from zeus.strategy.supply_demand.wyckoff import WyckoffDetector
from zeus.strategy.supply_demand.zone_detector import ZoneDetector

_DATA = Path(__file__).parent.parent.parent / "data" / "historical"

BALANCE  = 100_000.0
RISK_PCT = 0.005
YEARS    = [2023, 2024, 2025]

_WY_FX = dict(lookback=200, max_accum_bars=20, accum_range_mult=6.0,
              mss_lookback=60, min_spring_sweep_pct=0.10, min_mss_strength_pct=0.03)

_EXITS_FX = dict(tp1_r=1.5, tp1_size=0.33,
                 tp2_r=5.0, tp2_cumulative_pct=0.70,
                 max_daily_losses=1, max_monthly_losses=4)
_RUNNER_RR = 10.0

# (symbol, pip_size, spread quote, q2u quote/USD)
CANDIDATES = [
    ("EURGBP", 0.0001, 0.00012, 0.79),
    ("USDCHF", 0.0001, 0.00012, 0.88),
    ("NZDUSD", 0.0001, 0.00012, 1.00),
    ("AUDCAD", 0.0001, 0.00015, 1.38),
    ("EURCHF", 0.0001, 0.00015, 0.88),
    ("AUDNZD", 0.0001, 0.00020, 1.68),
    ("EURCAD", 0.0001, 0.00020, 1.38),
    ("GBPCHF", 0.0001, 0.00025, 0.88),
    ("AUDJPY", 0.01,   0.015,   150.0),
    ("EURJPY", 0.01,   0.015,   150.0),
    ("CADJPY", 0.01,   0.018,   150.0),
    ("NZDJPY", 0.01,   0.018,   150.0),
    ("CHFJPY", 0.01,   0.025,   150.0),
    ("XAGUSD", 0.01,   0.025,   1.00),
    ("GRXEUR", 1.0,    1.5,     0.92),
]


def _strat(pip_size: float) -> SDStrategy:
    return SDStrategy(
        zone_detector           = ZoneDetector(),
        wyckoff_detector        = WyckoffDetector(**_WY_FX),
        risk_reward             = 1.25,
        min_zone_score          = 4.0,
        min_composite_score     = 4.0,
        signal_cooldown         = 10,
        use_trend_filter        = True,
        trend_slope_lookback    = 3,
        use_price_above_ema     = True,
        ema_atr_tolerance       = 0.5,
        use_session_filter      = True,
        session_start_utc       = 7,
        session_end_utc         = 17,
        max_signals_per_day     = 6,
        min_sl_pips             = 5,
        pip_size                = pip_size,
        min_wyckoff_score       = 7.5,
        min_wyckoff_score_short = 6.5,
    )


def main() -> None:
    print("═" * 100)
    print("  Candidates portefeuille propfirm — config forex WS7.5 gelée · TIERED-5R · risque 0.5%")
    print("═" * 100)
    print(f"\n  {'Paire':<8} {'Année':<6} {'N':>4} {'L/S':>7}  {'W':>3} {'L':>3} "
          f"{'WR%':>6}  {'R':>8}  {'DD%':>5}  Statut")
    print("  " + "─" * 76)

    admitted: list[str] = []
    for sym, pip, spread, q2u in CANDIDATES:
        folder = sym.lower()
        pair_ok = True
        seen_any = False
        for year in YEARS:
            f = _DATA / folder / "m1" / f"DAT_MT_{sym}_M1_{year}.csv"
            if not f.exists():
                continue
            seen_any = True
            m1 = parse_m1_csv(f)
            m1 = m1[~m1.index.duplicated(keep="first")]
            m15 = resample_ohlcv(m1, "15min")

            strategy = _strat(pip)
            signals = strategy.run(m15, m1)
            n_long = sum(1 for s in signals if s.direction == "long")
            n_short = len(signals) - n_long
            sigs = [dataclasses.replace(s, risk_reward=_RUNNER_RR) for s in signals]

            results, _ = simulate_all(
                sigs, m1, risk_pct=RISK_PCT, spread=spread,
                initial_equity=BALANCE, quote_to_usd_rate=q2u, **_EXITS_FX,
            )
            w = sum(1 for r in results if r.outcome == "win")
            l = sum(1 for r in results if r.outcome == "loss")
            tot_r = sum(r.pnl_r for r in results)
            wr = w / (w + l) * 100 if (w + l) else 0.0

            # DD en % du solde initial sur la séquence R (risque fixe)
            eq = peak = 0.0
            dd = 0.0
            for r in results:
                eq += r.pnl_r * RISK_PCT * 100
                peak = max(peak, eq)
                dd = max(dd, peak - eq)

            year_ok = wr >= 55.0 and tot_r > 0 and dd <= 8.0 and len(results) >= 10
            pair_ok &= year_ok
            print(f"  {sym:<8} {year:<6} {len(results):>4} {n_long}L/{n_short}S  "
                  f"{w:>3} {l:>3} {wr:>5.1f}%  {tot_r:>+8.2f}  {dd:>4.1f}%  "
                  f"{'✓' if year_ok else '✗'}")

        if seen_any and pair_ok:
            admitted.append(sym)
        print("  " + "─" * 76)

    print(f"\n  ADMISES ({len(admitted)}) : {', '.join(admitted) if admitted else 'aucune'}")
    print("═" * 100)


if __name__ == "__main__":
    main()
