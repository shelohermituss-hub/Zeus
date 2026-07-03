"""
S&D Strategy — Forex Generalization Test
=========================================

Applies the V_FINAL_B champion configuration (tuned on XAUUSD) unchanged to
five Forex instruments to measure out-of-instrument robustness.

V_FINAL_B params (frozen from Round 23):
  trend_slope_lookback=3, min_wyckoff_score_short=99.0, max_daily_losses=1
  zone≥5.0, wy≥5.9 long, R:R=1.5, M15/M1, slope+EMA, session 07-21h
  TP1@1.25R (50% partial exit) → TP2@1.5R

Instruments and data available:
  GBPUSD  : 2025 (IS) · 2026-Jun (OOS)
  CADJPY  : 2025 (IS)
  USDCHF  : 2025 (IS)
  NZDUSD  : 2026-Jun (OOS only)
  USDCAD  : 2026-Jun (OOS only)

Spread values (in instrument price units):
  GBPUSD / NZDUSD / USDCAD / USDCHF : 0.0001 (1 pip)
  CADJPY                              : 0.02   (2 pips × 0.01)

Note: USD P&L figures for CADJPY are approximate (P&L computed in JPY,
applied against USD equity). R-based metrics (WR, total_r, DD) are
correct for all instruments.

Usage
-----
    python -m zeus.backtest.run_sd_forex
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv, resample_ohlcv
from zeus.backtest.sd_simulation import compute_metrics, simulate_all
from zeus.strategy.supply_demand.sd_strategy import SDStrategy
from zeus.strategy.supply_demand.wyckoff import WyckoffDetector
from zeus.strategy.supply_demand.zone_detector import ZoneDetector

_ROOT = Path(__file__).parent.parent.parent
_DATA = _ROOT / "data" / "historical"

INITIAL_BALANCE = 10_000.0
RISK_PCT        = 0.01

# ── V_FINAL_B champion params ─────────────────────────────────────────────────
_STRATEGY_PARAMS = dict(
    risk_reward             = 1.5,
    min_zone_score          = 5.0,
    min_wyckoff_score       = 5.9,
    min_composite_score     = 5.0,
    min_wyckoff_score_short = 99.0,  # long-only
    signal_cooldown         = 10,
    use_trend_filter        = True,
    trend_slope_lookback    = 3,
    use_price_above_ema     = True,
    use_session_filter      = True,
    session_start_utc       = 7,
    session_end_utc         = 21,
    max_signals_per_day     = 6,
    use_adx_filter          = False,
    use_h4_trend_filter     = False,
    use_rsi_filter          = False,
)

_WYCKOFF_PARAMS = dict(
    lookback             = 200,
    max_accum_bars       = 20,
    accum_range_mult     = 6.0,
    mss_lookback         = 60,
    min_spring_sweep_pct = 0.05,
    min_mss_strength_pct = 0.03,
)

TP1_R    = 1.25
TP1_SIZE = 0.50

# ── Instrument registry ───────────────────────────────────────────────────────
# Each entry: (symbol, pip_spread, [(label, [files])])
INSTRUMENTS = [
    (
        "GBPUSD",
        0.0001,
        [
            ("2025 Full Year (IS)",  [_DATA / "gbpusd" / "m1" / "DAT_MT_GBPUSD_M1_2025.csv"]),
            ("2026 Jun     (OOS)",   [_DATA / "gbpusd" / "m1" / "DAT_MT_GBPUSD_M1_202606.csv"]),
        ],
    ),
    (
        "CADJPY",
        0.02,
        [
            ("2025 Full Year (IS)",  [_DATA / "cadjpy" / "m1" / "DAT_MT_CADJPY_M1_2025.csv"]),
        ],
    ),
    (
        "USDCHF",
        0.0001,
        [
            ("2025 Full Year (IS)",  [_DATA / "usdchf" / "m1" / "DAT_MT_USDCHF_M1_2025.csv"]),
        ],
    ),
    (
        "NZDUSD",
        0.0001,
        [
            ("2026 Jun     (OOS)",   [_DATA / "nzdusd" / "m1" / "DAT_MT_NZDUSD_M1_202606.csv"]),
        ],
    ),
    (
        "USDCAD",
        0.0001,
        [
            ("2026 Jun     (OOS)",   [_DATA / "usdcad" / "m1" / "DAT_MT_USDCAD_M1_202606.csv"]),
        ],
    ),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_strategy() -> SDStrategy:
    return SDStrategy(
        zone_detector    = ZoneDetector(),
        wyckoff_detector = WyckoffDetector(**_WYCKOFF_PARAMS),
        **_STRATEGY_PARAMS,
    )


def _load(files: list[Path]) -> pd.DataFrame:
    frames = [parse_histdata_csv(f) for f in files]
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="first")]


def _run_period(
    m1_df:  pd.DataFrame,
    spread: float,
) -> tuple[list, dict[str, Any], int]:
    zone_df  = resample_ohlcv(m1_df, "15min")
    strategy = _build_strategy()
    signals  = strategy.run(zone_df, m1_df)
    res, n_exp = simulate_all(
        signals, m1_df,
        risk_pct           = RISK_PCT,
        spread             = spread,
        max_monthly_losses = 4,
        max_daily_losses   = 1,
        tp1_r              = TP1_R,
        tp1_size           = TP1_SIZE,
    )
    m = compute_metrics(res, INITIAL_BALANCE, len(signals), n_exp)
    return res, m, len(signals)


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_instrument(
    symbol:  str,
    spread:  float,
    periods: list[tuple[str, list[Path]]],
) -> tuple[float, float, float, float, int]:
    """Print per-period table for one instrument. Returns (total_r, min_r, wr, max_dd, n_trades)."""
    print(f"\n  ── {symbol}  (spread={spread})  ──────────────────────────────────")
    print(f"  {'Period':<24}  {'Sig':>4}  {'W':>3} {'L':>3}  {'WR%':>5}  {'R':>7}  {'P&L$':>8}  {'DD%':>5}")
    print("  " + "─" * 68)

    total_w = total_l = total_sig = total_r = 0.0
    max_dd  = 0.0
    min_r   = float("inf")
    all_pos_r = True
    total_trades = 0

    for label, files in periods:
        missing = [f for f in files if not f.exists()]
        if missing:
            print(f"  {label:<24}  [file not found: {missing[0].name}]")
            continue
        m1   = _load(files)
        _, m, n_sig = _run_period(m1, spread)
        decided = m["n_wins"] + m["n_losses"]
        print(
            f"  {label:<24}  {n_sig:>4}  "
            f"{m['n_wins']:>3} {m['n_losses']:>3}  "
            f"{m['win_rate']:>5.1f}%  "
            f"{m['total_r']:>+7.2f}  "
            f"{m['total_usd']:>+8.0f}$  "
            f"{m['max_dd']:>4.1f}%"
        )
        total_w   += m["n_wins"]
        total_l   += m["n_losses"]
        total_sig += n_sig
        total_r   += m["total_r"]
        total_trades += decided
        if n_sig > 0:
            min_r  = min(min_r, m["total_r"])
            max_dd = max(max_dd, m["max_dd"])
            if m["total_r"] <= 0:
                all_pos_r = False

    if min_r == float("inf"):
        min_r = 0.0

    decided   = total_w + total_l
    overall_wr = total_w / decided * 100 if decided else 0.0

    print("  " + "─" * 68)
    print(
        f"  {'TOTAL / COMBINED':<24}  {int(total_sig):>4}  "
        f"{int(total_w):>3} {int(total_l):>3}  "
        f"{overall_wr:>5.1f}%  {total_r:>+7.2f}"
    )
    verdict = "PASS ✓" if overall_wr >= 60 and all_pos_r and total_trades > 0 else "FAIL ✗"
    print(f"  VERDICT: {verdict}  (WR ≥ 60% + R > 0 every period)")

    return total_r, min_r, overall_wr, max_dd, int(decided)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 90)
    print("  S&D Strategy — Forex Generalization Test")
    print("  V_FINAL_B champion params · long-only · TP1@1.25R(50%)→TP2@1.5R")
    print("=" * 90)

    summary: list[tuple[str, float, float, float, float, int]] = []

    for symbol, spread, periods in INSTRUMENTS:
        tr, mr, wr, dd, n = _print_instrument(symbol, spread, periods)
        summary.append((symbol, tr, mr, wr, dd, n))

    # ── Cross-instrument summary ──────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("  FOREX GENERALIZATION — Cross-instrument summary")
    print(f"{'─' * 90}")
    print(f"\n  {'Symbol':<10}  {'Trades':>6}  {'TotR':>7}  {'MinR':>7}  {'OvWR':>5}  {'MaxDD':>6}  {'Verdict'}")
    print("  " + "─" * 65)

    n_pass = 0
    for symbol, tr, mr, wr, dd, n in summary:
        if n == 0:
            verdict = "NO DATA"
        elif wr >= 60 and mr > 0:
            verdict = "PASS ✓"
            n_pass += 1
        else:
            verdict = "FAIL ✗"
        print(f"  {symbol:<10}  {n:>6}  {tr:>+7.2f}  {mr:>+7.2f}  {wr:>5.1f}%  {dd:>5.1f}%  {verdict}")

    # ── XAUUSD reference (from Round 23 V_FINAL_B) ───────────────────────
    print("  " + "─" * 65)
    print(f"  {'XAUUSD':<10}  {'':>6}  {'+52.62':>7}  {'+4.50':>7}  {'71.1':>5}%  {'2.5':>5}%  REFERENCE")

    n_instruments = len([s for s in summary if s[5] > 0])
    print(f"\n  Instruments passing: {n_pass} / {n_instruments}  (≥60% WR + R>0 per period)")

    if n_pass > 0:
        passing = [(sym, tr, mr, wr, dd) for sym, tr, mr, wr, dd, n in summary if n > 0 and wr >= 60 and mr > 0]
        best    = max(passing, key=lambda x: x[1])
        print(f"  Best passing Forex instrument: {best[0]}  TotR={best[1]:+.2f}  WR={best[3]:.1f}%  DD={best[4]:.1f}%")
    else:
        print("  No Forex instrument passes the validation criteria.")
        print("  The V_FINAL_B edge appears XAUUSD-specific — do not deploy on Forex without re-optimisation.")

    print(f"\n{'=' * 90}")
    print("  Fin du test de généralisation Forex")
    print(f"{'=' * 90}\n")


if __name__ == "__main__":
    main()
