"""Portefeuille multi-symboles sous guard propfirm.

Portefeuille : XAUUSD (S6 : V4 + sorties scalp 1R/2R/5R)
             + cluster forex validé (GBPUSD, EURUSD, GBPAUD, EURNZD — WS7.5, TIERED-5R)

Les trades des 5 symboles sont fusionnés chronologiquement puis rejoués dans
le PropFirmGuard (risque fixe, limites internes strictes) à trois niveaux de
risque : 0.5%, 0.75%, 1%.

Résultats de référence (2024 + 2025, ~1.8 trades/jour tradé, WR ~64%) :
  0.50% : +23.9% / +42.8% par an · DD max 3.25% · cible +10% en 33-147 j
  0.75% : +35.8% / +64.1% par an · DD max 4.88% · cible +10% en 33-102 j  ← RETENU
  1.00% : FAIL 2025 — le drawdown interne 6% est touché en janvier et le
          compte est arrêté définitivement avant les mois gagnants.

Usage
-----
    python -m zeus.backtest.run_multi_symbol_propfirm
"""
import dataclasses
from collections import defaultdict
from pathlib import Path

import pandas as pd

from zeus.backtest.data_loader import parse_m1_csv, resample_ohlcv
from zeus.backtest.sd_simulation import simulate_all
from zeus.risk.propfirm import PropFirmConfig, PropFirmGuard
from zeus.strategy.supply_demand.sd_strategy import SDStrategy
from zeus.strategy.supply_demand.wyckoff import WyckoffDetector
from zeus.strategy.supply_demand.zone_detector import ZoneDetector

_DATA = Path(__file__).parent.parent.parent / "data" / "historical"
BALANCE = 100_000.0
RISK_PCT = 0.005          # risque de la passe de simulation (le replay guard varie)
RISK_LEVELS = (0.005, 0.0075, 0.01)

# ── Configs validées par symbole ──────────────────────────────────────────────

_WY_XAU = dict(lookback=200, max_accum_bars=20, accum_range_mult=6.0,
               mss_lookback=60, min_spring_sweep_pct=0.05, min_mss_strength_pct=0.03)
_WY_FX  = dict(lookback=200, max_accum_bars=20, accum_range_mult=6.0,
               mss_lookback=60, min_spring_sweep_pct=0.10, min_mss_strength_pct=0.03)

_STRAT_XAU = dict(
    min_zone_score=5.0, min_composite_score=5.0, signal_cooldown=10,
    use_trend_filter=True, trend_slope_lookback=3, use_price_above_ema=True,
    use_session_filter=True, session_start_utc=7, session_end_utc=21,
    max_signals_per_day=10, min_wyckoff_score=5.9, min_wyckoff_score_short=8.5,
)
_STRAT_FX = dict(
    risk_reward=1.25, min_zone_score=4.0, min_composite_score=4.0,
    signal_cooldown=10, use_trend_filter=True, trend_slope_lookback=3,
    use_price_above_ema=True, ema_atr_tolerance=0.5, use_session_filter=True,
    session_start_utc=7, session_end_utc=17, max_signals_per_day=6,
    min_sl_pips=5, pip_size=0.0001,
    min_wyckoff_score=7.5, min_wyckoff_score_short=6.5,
)

_EXITS_XAU = dict(use_be=True, tp1_r=1.0, tp1_size=0.5,
                  tp2_r=2.0, tp2_cumulative_pct=0.75,
                  tp3_r=5.0, tp3_cumulative_pct=1.0,
                  max_daily_losses=1, max_monthly_losses=4)
_EXITS_FX  = dict(tp1_r=1.5, tp1_size=0.33,
                  tp2_r=5.0, tp2_cumulative_pct=0.70,
                  max_daily_losses=1, max_monthly_losses=4)

SYMBOLS = [
    # (sym, folder, spread, q2u, wyckoff, strat, exits, rr, name_pattern)
    ("XAUUSD", "xauusd", 0.30, 1.00, _WY_XAU, _STRAT_XAU, _EXITS_XAU, 5.0),
    ("GBPUSD", "gbpusd", 0.0001, 1.00, _WY_FX, _STRAT_FX, _EXITS_FX, 10.0),
    ("EURUSD", "eurusd", 0.0001, 1.00, _WY_FX, _STRAT_FX, _EXITS_FX, 10.0),
    ("GBPAUD", "gbpaud", 0.0002, 1.52, _WY_FX, _STRAT_FX, _EXITS_FX, 10.0),
    ("EURNZD", "eurnzd", 0.0002, 1.62, _WY_FX, _STRAT_FX, _EXITS_FX, 10.0),
]

YEARS = [2024, 2025]


def _load(folder: str, sym: str, year: int) -> pd.DataFrame | None:
    p = _DATA / folder / "m1" / f"DAT_MT_{sym}_M1_{year}.csv"
    if not p.exists():
        return None
    df = parse_m1_csv(p)
    return df[~df.index.duplicated(keep="first")]


def main() -> None:
    for year in YEARS:
        print(f"\n{'═'*100}\n  ANNÉE {year} — portefeuille 5 symboles · risque fixe 0.5% · guard strict\n{'═'*100}")
        merged: list[tuple[pd.Timestamp, pd.Timestamp, float, str, str]] = []
        # (entry_ts, exit_ts, pnl_r, outcome, sym)

        for sym, folder, spread, q2u, wy, strat, exits, rr in SYMBOLS:
            m1 = _load(folder, sym, year)
            if m1 is None:
                print(f"  ✗ {sym} : pas de données {year}")
                continue
            m15 = resample_ohlcv(m1, "15min")
            strategy = SDStrategy(
                zone_detector=ZoneDetector(),
                wyckoff_detector=WyckoffDetector(**wy),
                **strat,
            )
            signals = strategy.run(m15, m1)
            signals = [dataclasses.replace(s, risk_reward=rr) for s in signals]
            results, _ = simulate_all(
                signals, m1, risk_pct=RISK_PCT, spread=spread,
                initial_equity=BALANCE, quote_to_usd_rate=q2u, **exits,
            )
            for tr in results:
                exit_ts = m1.index[tr.exit_bar] if tr.exit_bar < len(m1) else tr.signal.formed_at
                merged.append((tr.signal.formed_at, exit_ts, tr.pnl_r, tr.outcome, sym))
            n = len(results)
            r = sum(t.pnl_r for t in results)
            print(f"  ✓ {sym:<8} {n:>3} trades  R={r:+8.2f}")

        merged.sort(key=lambda x: x[0])

        for risk_pct in RISK_LEVELS:
            _replay(year, merged, risk_pct)


def _replay(year: int, merged: list, risk_pct: float) -> None:
    # Replay guard
    guard = PropFirmGuard(PropFirmConfig(risk_per_trade_pct=risk_pct), initial_balance=BALANCE)
    risk_usd = guard.risk_amount_usd()
    equity = peak = BALANCE
    max_dd = 0.0
    day_pnl = defaultdict(float)
    month_pnl = defaultdict(float)
    n_taken = n_blocked = w = l = 0
    tot_r = 0.0
    first_date = target_date = None

    for entry_ts, exit_ts, pnl_r, outcome, sym in merged:
        if first_date is None:
            first_date = entry_ts.date()
        if not guard.can_trade(entry_ts):
            n_blocked += 1
            continue
        pnl = pnl_r * risk_usd
        guard.on_trade_closed(exit_ts, pnl)
        n_taken += 1
        tot_r += pnl_r
        equity += pnl
        day_pnl[entry_ts.date()] += pnl
        month_pnl[entry_ts.strftime("%Y-%m")] += pnl
        w += outcome == "win"
        l += outcome == "loss"
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / BALANCE)
        if target_date is None and equity >= BALANCE * 1.10:
            target_date = exit_ts.date()

    dec = w + l
    wr = w / dec * 100 if dec else 0.0
    worst = min(day_pnl.values()) if day_pnl else 0.0
    n_days = len(day_pnl)
    tgt = f"{(target_date - first_date).days}j" if target_date else "non atteinte"

    print(f"\n  PORTEFEUILLE {year} — risque {risk_pct*100:.2f}% :")
    print(f"    Trades pris     : {n_taken}  (bloqués guard : {n_blocked})")
    print(f"    Jours tradés    : {n_days}  →  {n_taken/max(n_days,1):.2f} trades/jour tradé")
    print(f"    WR              : {wr:.1f}%   R total : {tot_r:+.2f}")
    print(f"    PnL             : {equity-BALANCE:+,.0f}$  ({(equity-BALANCE)/BALANCE*100:+.1f}%)")
    print(f"    DD max          : {max_dd*100:.2f}%   Pire journée : {-worst/BALANCE*100 if worst<0 else 0:.2f}%")
    print(f"    Cible +10%      : {tgt}")
    print(f"    Challenge failed: {guard.challenge_failed()}")
    print(f"    Par mois        :")
    for mo in sorted(month_pnl):
        print(f"      {mo} : {month_pnl[mo]:+10,.0f}$")


if __name__ == "__main__":
    main()
