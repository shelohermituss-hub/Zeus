"""
Validation des indices/US Dollar Index — historique complet 2024+2025.
========================================================================

Protocole identique aux candidates forex : config WS7.5 gelée (TIERED-5R),
AUCUNE optimisation par instrument.  Contrairement au lot précédent
(~7 mois seulement), on dispose ici de deux années complètes — le
jugement est donc fiable (même barre que le cluster forex validé).

Instruments : AUXAUD (ASX200), JPXJPY (Nikkei225), GRXEUR (DAX),
FRXEUR (CAC40), UKXGBP (FTSE100), UDXUSD (US Dollar Index),
SPXUSD (S&P500), NSXUSD (Nasdaq 100).

Critères d'admission (par année ET au total) :
  WR ≥ 55% · R total > 0 · DD ≤ 8% · N ≥ 10 trades

Usage
-----
    python -m zeus.backtest.run_indices_2024_2025
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
YEARS    = [2024, 2025]

_WY_FX = dict(lookback=200, max_accum_bars=20, accum_range_mult=6.0,
              mss_lookback=60, min_spring_sweep_pct=0.10, min_mss_strength_pct=0.03)

_EXITS_FX = dict(tp1_r=1.5, tp1_size=0.33,
                 tp2_r=5.0, tp2_cumulative_pct=0.70,
                 max_daily_losses=1, max_monthly_losses=4)
_RUNNER_RR = 10.0

# (symbol, folder, pip_size, spread, q2u, note)
INSTRUMENTS = [
    ("AUXAUD", "auxaud", 1.0,  2.0,  1.538, "ASX200 (AUD) · q2u=1/AUDUSD"),
    ("JPXJPY", "jpxjpy", 1.0,  10.0, 150.0, "Nikkei225 (JPY) · q2u=USDJPY≈150"),
    ("GRXEUR", "grxeur", 1.0,  1.5,  0.926, "DAX (EUR) · q2u=1/EURUSD"),
    ("FRXEUR", "frxeur", 1.0,  1.5,  0.926, "CAC40 (EUR) · q2u=1/EURUSD"),
    ("UKXGBP", "ukxgbp", 1.0,  1.5,  0.787, "FTSE100 (GBP) · q2u=1/GBPUSD"),
    ("UDXUSD", "udxusd", 0.01, 0.02, 1.0,   "US Dollar Index"),
    ("SPXUSD", "spxusd", 0.1,  0.5,  1.0,   "S&P500 CFD"),
    ("NSXUSD", "nsxusd", 1.0,  1.0,  1.0,   "Nasdaq 100 CFD"),
]


def _strat(pip: float) -> SDStrategy:
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
        pip_size                = pip,
        min_wyckoff_score       = 7.5,
        min_wyckoff_score_short = 6.5,
    )


def main() -> None:
    print("═" * 108)
    print("  Indices & US Dollar Index — config forex WS7.5 gelée · TIERED-5R · risque 0.5%")
    print("═" * 108)
    print(f"\n  {'Instrument':<10} {'Année':<6} {'N':>4} {'L/S':>7}  {'W':>3} {'L':>3} "
          f"{'WR%':>6}  {'R':>8}  {'DD%':>5}  Statut")
    print("  " + "─" * 88)

    admitted: list[str] = []
    for sym, folder, pip, spread, q2u, note in INSTRUMENTS:
        instr_ok = True
        total_r, total_w, total_l = 0.0, 0, 0
        seen_any = False

        for year in YEARS:
            p = _DATA / folder / "m1" / f"DAT_MT_{sym}_M1_{year}.csv"
            if not p.exists():
                print(f"  {sym:<10} {year:<6} MANQUANT")
                instr_ok = False
                continue
            seen_any = True

            m1 = parse_m1_csv(p)
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

            eq = peak = 0.0
            dd = 0.0
            for r in results:
                eq += r.pnl_r * RISK_PCT * 100
                peak = max(peak, eq)
                dd = max(dd, peak - eq)

            year_ok = wr >= 55.0 and tot_r > 0 and dd <= 8.0 and len(results) >= 10
            instr_ok &= year_ok
            total_r += tot_r; total_w += w; total_l += l
            print(f"  {sym:<10} {year:<6} {len(results):>4} {n_long}L/{n_short}S  "
                  f"{w:>3} {l:>3} {wr:>5.1f}%  {tot_r:>+8.2f}  {dd:>4.1f}%  "
                  f"{'✓' if year_ok else '✗'}")

        dec = total_w + total_l
        wr_tot = total_w / dec * 100 if dec else 0.0
        overall_ok = seen_any and instr_ok and total_r > 0 and wr_tot >= 55.0
        if overall_ok:
            admitted.append(sym)
        print(f"  {sym:<10} {'TOTAL':<6} {'':>4} {'':>7}  {total_w:>3} {total_l:>3} "
              f"{wr_tot:>5.1f}%  {total_r:>+8.2f}  {'':>5}  "
              f"{'✓ ADMIS' if overall_ok else '✗'}   ({note})")
        print("  " + "─" * 88)

    print(f"\n  ADMIS ({len(admitted)}) : {', '.join(admitted) if admitted else 'aucun'}")
    print("═" * 108)


if __name__ == "__main__":
    main()
