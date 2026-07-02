"""
S&D Strategy — XAUUSD Full-Year 2025 Backtest

Loads the single annual M1 CSV for 2025 (354,011 bars, Jan–Dec),
resamples to M15 for zone detection, runs SDStrategy once on the full
dataset, then prints:

  • Global summary (full year)
  • Monthly breakdown (one line per calendar month)
  • Full trade log

Strict variant only — this is the production-candidate parameter set:
  accum_mult=3.0, mss_lb=10, cooldown=30, daily_cap=2

Simulation engine: zeus/backtest/sd_simulation.py
  - Entry at open of bar after MSS bar
  - Spread: 0.30 USD/oz applied at entry
  - TP recalculated from effective entry (not from MSS close)
  - Pessimistic SL/TP conflict: SL wins when both touched in same bar

Usage
-----
    python -m zeus.backtest.run_sd_2025_backtest
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv, resample_ohlcv
from zeus.backtest.sd_simulation import (
    SPREAD_PER_OZ,
    print_monthly_breakdown,
    print_report,
    simulate_all,
)
from zeus.strategy.supply_demand.sd_strategy import SDStrategy
from zeus.strategy.supply_demand.wyckoff import WyckoffDetector
from zeus.strategy.supply_demand.zone_detector import ZoneDetector

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT    = Path(__file__).parent.parent.parent
_M1_FILE = _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_2025.csv"

# ── Parameters ────────────────────────────────────────────────────────────────

INITIAL_BALANCE   = 10_000.0
RISK_PCT          = 0.01
RISK_REWARD       = 3.0
MIN_ZONE_SCORE    = 5.0
MIN_WYCKOFF_SCORE = 4.0
MIN_COMPOSITE     = 5.0


def _build_strategy(
    accum_mult: float = 3.0,
    mss_lb:     int   = 10,
    cooldown:   int   = 30,
    daily_cap:  int   = 2,
) -> SDStrategy:
    return SDStrategy(
        zone_detector        = ZoneDetector(),
        wyckoff_detector     = WyckoffDetector(
            accum_range_mult     = accum_mult,
            mss_lookback         = mss_lb,
            min_spring_sweep_pct = 0.20,
            min_mss_strength_pct = 0.10,
        ),
        risk_reward          = RISK_REWARD,
        min_zone_score       = MIN_ZONE_SCORE,
        min_wyckoff_score    = MIN_WYCKOFF_SCORE,
        min_composite_score  = MIN_COMPOSITE,
        signal_cooldown      = cooldown,
        max_signals_per_day  = daily_cap,
        use_trend_filter     = True,
        use_session_filter   = True,
    )


def main() -> None:
    print("=" * 62)
    print("  S&D Strategy — XAUUSD Full-Year 2025 Backtest")
    print("=" * 62)

    print(f"\nLoading M1 data from {_M1_FILE.name} …")
    m1_df = parse_histdata_csv(_M1_FILE)
    m1_df = m1_df[~m1_df.index.duplicated(keep="first")]
    print(f"  {len(m1_df):,} M1 bars  "
          f"({m1_df.index[0].date()}  →  {m1_df.index[-1].date()})")

    print("\nResampling M1 → M15 …")
    m15_df = resample_ohlcv(m1_df, "15min")
    print(f"  {len(m15_df):,} M15 bars\n")

    # ── Strict variant ────────────────────────────────────────────────────────
    label    = "strict: accum_mult=3.0, mss_lb=10, daily_cap=2"
    strategy = _build_strategy(accum_mult=3.0, mss_lb=10, cooldown=30, daily_cap=2)

    print("Running strategy …")
    signals = strategy.run(m15_df, m1_df)
    print(f"  {len(signals)} signals generated\n")

    results, n_expired = simulate_all(
        signals, m1_df, risk_pct=RISK_PCT, spread=SPREAD_PER_OZ
    )

    print(f"\n{'─' * 62}")
    print(f"  VARIANT: {label}")
    print_report(results, INITIAL_BALANCE, len(signals), n_expired, m1_df, label=label)
    print_monthly_breakdown(results, INITIAL_BALANCE)


if __name__ == "__main__":
    main()
