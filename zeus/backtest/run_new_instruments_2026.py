"""
Validation des nouveaux instruments — indices, commodity, crypto.
==================================================================

Protocole identique aux candidates forex : config WS7.5 gelée (TIERED-5R),
AUCUNE optimisation par instrument.  Fenêtre disponible : ~déc. 2025 → juil.
2026 (~7 mois continus, un seul fichier par instrument — pas de multi-année).

Découpage en 3 sous-périodes (~2 mois chacune) pour juger la régularité,
en plus du total — le même principe qu'un test année-par-année mais à
l'échelle du seul historique disponible.

Critères d'admission (par sous-période ET au total) :
  WR ≥ 55% · R total > 0 · DD ≤ 8% · N ≥ 10 trades

Spreads/conversions : approximations retail CFD documentées inline —
à ajuster selon le broker réel avant tout trading.

Usage
-----
    python -m zeus.backtest.run_new_instruments_2026
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

_WY_FX = dict(lookback=200, max_accum_bars=20, accum_range_mult=6.0,
              mss_lookback=60, min_spring_sweep_pct=0.10, min_mss_strength_pct=0.03)

_EXITS_FX = dict(tp1_r=1.5, tp1_size=0.33,
                 tp2_r=5.0, tp2_cumulative_pct=0.70,
                 max_daily_losses=1, max_monthly_losses=4)
_RUNNER_RR = 10.0

# (symbol, folder, file, pip_size, spread, q2u, note)
INSTRUMENTS = [
    ("USATECHIDXUSD", "usatechidxusd", "USATECHIDXUSD_M1_202512_202607.csv",
     1.0, 1.0, 1.0, "Nasdaq 100 CFD · spread≈1pt"),
    ("USA500IDXUSD", "usa500idxusd", "USA500IDXUSD_M1_202512_202607.csv",
     0.1, 0.5, 1.0, "S&P500 CFD · spread≈0.5pt"),
    ("USA30IDXUSD", "usa30idxusd", "USA30IDXUSD_M1_202512_202607.csv",
     1.0, 2.0, 1.0, "Dow Jones CFD · spread≈2pt"),
    ("DEUIDXEUR", "deuidxeur", "DEUIDXEUR_M1_202512_202607.csv",
     1.0, 1.0, 0.926, "DAX (EUR) · spread≈1pt · q2u=1/EURUSD≈0.926"),
    ("GBRIDXGBP", "gbridxgbp", "GBRIDXGBP_M1_202512_202607.csv",
     1.0, 0.8, 0.787, "FTSE100 (GBP) · spread≈0.8pt · q2u=1/GBPUSD≈0.787"),
    ("BRENTCMDUSD", "brentcmdusd", "BRENTCMDUSD_M1_202512_202607.csv",
     0.01, 0.03, 1.0, "Brent crude · spread≈0.03$/baril"),
    ("BTCUSD", "btcusd", "BTCUSD_M1_202512_202607.csv",
     1.0, 20.0, 1.0, "Bitcoin CFD · spread≈20$"),
    ("ETHUSD", "ethusd", "ETHUSD_M1_202512_202607.csv",
     0.1, 1.0, 1.0, "Ethereum CFD · spread≈1$"),
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


def _split_periods(m1: pd.DataFrame, n: int = 3) -> list[tuple[str, pd.DataFrame]]:
    start, end = m1.index[0], m1.index[-1]
    total_days = (end - start).days
    step = total_days / n
    out = []
    for i in range(n):
        lo = start + pd.Timedelta(days=step * i)
        hi = start + pd.Timedelta(days=step * (i + 1)) if i < n - 1 else end + pd.Timedelta(seconds=1)
        chunk = m1[(m1.index >= lo) & (m1.index < hi)]
        out.append((f"P{i+1} ({lo.date()}→{(hi - pd.Timedelta(seconds=1)).date()})", chunk))
    return out


def main() -> None:
    print("═" * 108)
    print("  Nouveaux instruments — config forex WS7.5 gelée · TIERED-5R · risque 0.5%")
    print("═" * 108)
    print(f"\n  {'Instrument':<15} {'Sous-période':<28} {'N':>4} {'L/S':>7}  "
          f"{'W':>3} {'L':>3} {'WR%':>6}  {'R':>8}  {'DD%':>5}  Statut")
    print("  " + "─" * 96)

    admitted: list[str] = []
    for sym, folder, fname, pip, spread, q2u, note in INSTRUMENTS:
        p = _DATA / folder / "m1" / fname
        if not p.exists():
            print(f"  ✗ {sym}: fichier manquant ({fname})")
            continue

        m1 = parse_m1_csv(p)
        m1 = m1[~m1.index.duplicated(keep="first")]
        periods = _split_periods(m1, 3)

        instr_ok = True
        total_r, total_w, total_l = 0.0, 0, 0
        for label, chunk in periods:
            if len(chunk) < 1000:
                print(f"  {sym:<15} {label:<28} {'—':>4}  (trop peu de barres)")
                instr_ok = False
                continue
            m15 = resample_ohlcv(chunk, "15min")
            strategy = _strat(pip)
            signals = strategy.run(m15, chunk)
            n_long = sum(1 for s in signals if s.direction == "long")
            n_short = len(signals) - n_long
            sigs = [dataclasses.replace(s, risk_reward=_RUNNER_RR) for s in signals]

            results, _ = simulate_all(
                sigs, chunk, risk_pct=RISK_PCT, spread=spread,
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

            period_ok = wr >= 55.0 and tot_r > 0 and dd <= 8.0 and len(results) >= 10
            instr_ok &= period_ok
            total_r += tot_r; total_w += w; total_l += l
            print(f"  {sym:<15} {label:<28} {len(results):>4} {n_long}L/{n_short}S  "
                  f"{w:>3} {l:>3} {wr:>5.1f}%  {tot_r:>+8.2f}  {dd:>4.1f}%  "
                  f"{'✓' if period_ok else '✗'}")

        dec = total_w + total_l
        wr_tot = total_w / dec * 100 if dec else 0.0
        overall_ok = instr_ok and total_r > 0 and wr_tot >= 55.0
        if overall_ok:
            admitted.append(sym)
        print(f"  {sym:<15} {'TOTAL (' + note + ')':<28} {'':>4} {'':>7}  "
              f"{total_w:>3} {total_l:>3} {wr_tot:>5.1f}%  {total_r:>+8.2f}  {'':>5}  "
              f"{'✓ ADMIS' if overall_ok else '✗'}")
        print("  " + "─" * 96)

    print(f"\n  ADMIS ({len(admitted)}) : {', '.join(admitted) if admitted else 'aucun'}")
    print("═" * 108)


if __name__ == "__main__":
    main()
