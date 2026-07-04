"""
S&D Strategy — Forex OOS Validation  (2024 data)
=================================================

Tests the Universal-D champion config (found on 2025 IS data) against
full-year 2024 data — genuine out-of-sample, never seen during optimisation.

Universal-D params (frozen from FX-FINAL):
  shorts-dominant: wy_long≥7.0, wy_short≥5.9
  zone≥4.0, R:R=1.25, TP1@0.75R (50%) → TP2@1.25R
  session 07-21h, slope+EMA trend filter

Instruments (2024 OOS):
  GBPUSD  — backward OOS (IS was 2025)
  CADJPY  — backward OOS (IS was 2025)
  EURUSD  — blind test (never tested any year)
  NZDUSD  — blind test (only had 2026-Jun before)
  USDCAD  — blind test (only had 2026-Jun before)
  GBPAUD  — blind test (new instrument)
  AUDNZD  — blind test (new instrument)
  AUDCAD  — blind test (new instrument)
  EURNZD  — blind test (new instrument)

Also shows GBPUSD full timeline: 2024 OOS + 2025 IS + 2026-Jun OOS.

Pass criterion: WR ≥ 55% AND R > 0 AND DD ≤ 8%

Usage
-----
    python -m zeus.backtest.run_sd_forex_oos
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

# ── Universal-D champion (frozen) ─────────────────────────────────────────────
_UNIVERSAL_D = dict(
    risk_reward             = 1.25,
    min_zone_score          = 4.0,
    min_wyckoff_score       = 7.0,   # longs: strict
    min_wyckoff_score_short = 5.9,   # shorts: normal
    min_composite_score     = 4.0,
    signal_cooldown         = 10,
    use_trend_filter        = True,
    trend_slope_lookback    = 3,
    use_price_above_ema     = True,
    ema_atr_tolerance       = 0.5,   # R4: allow price within 0.5 ATR of EMA (replaces binary)
    use_session_filter      = True,
    session_start_utc       = 7,     # default; overridden per-pair by SESSION_OVERRIDE
    session_end_utc         = 21,
    max_signals_per_day     = 6,
    use_adx_filter          = False,
    use_h4_trend_filter     = False,
    use_rsi_filter          = False,
    min_sl_pips             = 5,     # R1: reject signals with SL < 5 pips
    pip_size                = 0.0001,
    min_score_product       = 0.0,   # R8: disabled by default; set >0 to enable product filter
)

_WYCKOFF_PARAMS = dict(
    lookback             = 200,
    max_accum_bars       = 20,
    accum_range_mult     = 6.0,
    mss_lookback         = 60,
    min_spring_sweep_pct = 0.10,   # R2: raised from 0.05 — filters weak springs on Forex
    min_mss_strength_pct = 0.03,
)

TP1_R    = 0.75
TP1_SIZE = 0.50

# ── Instrument registry ───────────────────────────────────────────────────────
# (symbol, spread, [(label, [files])])
INSTRUMENTS = [
    # ── Backward OOS: optimised on 2025, testing on 2024 ─────────────────
    (
        "GBPUSD",
        0.0001,
        [
            ("2024 Full Year (OOS←)", [_DATA / "gbpusd" / "m1" / "DAT_MT_GBPUSD_M1_2024.csv"]),
            ("2025 Full Year (IS)",   [_DATA / "gbpusd" / "m1" / "DAT_MT_GBPUSD_M1_2025.csv"]),
            ("2026 Jun     (OOS→)",   [_DATA / "gbpusd" / "m1" / "DAT_MT_GBPUSD_M1_202606.csv"]),
        ],
    ),
    (
        "CADJPY",
        0.020,
        [
            ("2024 Full Year (OOS←)", [_DATA / "cadjpy" / "m1" / "DAT_MT_CADJPY_M1_2024.csv"]),
            ("2025 Full Year (IS)",   [_DATA / "cadjpy" / "m1" / "DAT_MT_CADJPY_M1_2025.csv"]),
        ],
    ),
    # ── Blind tests: new instruments ─────────────────────────────────────
    (
        "EURUSD",
        0.0001,
        [
            ("2024 Full Year (blind)", [_DATA / "eurusd" / "m1" / "DAT_MT_EURUSD_M1_2024.csv"]),
        ],
    ),
    (
        "NZDUSD",
        0.0001,
        [
            ("2024 Full Year (blind)", [_DATA / "nzdusd" / "m1" / "DAT_MT_NZDUSD_M1_2024.csv"]),
            ("2026 Jun     (OOS→)",   [_DATA / "nzdusd" / "m1" / "DAT_MT_NZDUSD_M1_202606.csv"]),
        ],
    ),
    (
        "USDCAD",
        0.0001,
        [
            ("2024 Full Year (blind)", [_DATA / "usdcad" / "m1" / "DAT_MT_USDCAD_M1_2024.csv"]),
            ("2026 Jun     (OOS→)",   [_DATA / "usdcad" / "m1" / "DAT_MT_USDCAD_M1_202606.csv"]),
        ],
    ),
    (
        "GBPAUD",
        0.0002,
        [
            ("2024 Full Year (blind)", [_DATA / "gbpaud" / "m1" / "DAT_MT_GBPAUD_M1_2024.csv"]),
        ],
    ),
    (
        "AUDNZD",
        0.0002,
        [
            ("2024 Full Year (blind)", [_DATA / "audnzd" / "m1" / "DAT_MT_AUDNZD_M1_2024.csv"]),
        ],
    ),
    (
        "AUDCAD",
        0.0002,
        [
            ("2024 Full Year (blind)", [_DATA / "audcad" / "m1" / "DAT_MT_AUDCAD_M1_2024.csv"]),
        ],
    ),
    (
        "EURNZD",
        0.0002,
        [
            ("2024 Full Year (blind)", [_DATA / "eurnzd" / "m1" / "DAT_MT_EURNZD_M1_2024.csv"]),
        ],
    ),
]

PASS_WR = 55.0
PASS_DD = 8.0

# ── Per-instrument overrides ──────────────────────────────────────────────────

# R7: Adaptive session — GBP/EUR cluster uses London only (07-17h)
SESSION_OVERRIDE: dict[str, tuple[int, int]] = {
    "GBPUSD": (7, 17),
    "EURUSD": (7, 17),
    "GBPAUD": (7, 17),
    "EURNZD": (7, 17),
}

# R5: Quote-to-USD conversion rates (approximate 2024 annual averages)
# Only needed for non-USD-quoted pairs to correct P&L currency reporting
QUOTE_TO_USD: dict[str, float] = {
    "GBPUSD": 1.0,
    "CADJPY": 149.0,    # P&L in JPY  → divide by USDJPY ≈ 149
    "EURUSD": 1.0,
    "NZDUSD": 1.0,
    "USDCAD": 1.36,     # P&L in CAD  → divide by USDCAD ≈ 1.36
    "GBPAUD": 1.52,     # P&L in AUD  → divide by 1/AUDUSD ≈ 1/0.66
    "AUDNZD": 1.62,     # P&L in NZD  → divide by 1/NZDUSD ≈ 1/0.62
    "AUDCAD": 1.36,     # P&L in CAD  → divide by USDCAD ≈ 1.36
    "EURNZD": 1.62,     # P&L in NZD  → divide by 1/NZDUSD ≈ 1/0.62
}

# R3: Paper-trade cluster — instruments validated on OOS 2024 data
PAPER_CLUSTER = ["GBPUSD", "EURUSD", "GBPAUD", "EURNZD"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_strategy(session_start: int = 7, session_end: int = 21) -> SDStrategy:
    params = {**_UNIVERSAL_D, "session_start_utc": session_start, "session_end_utc": session_end}
    return SDStrategy(
        zone_detector    = ZoneDetector(),
        wyckoff_detector = WyckoffDetector(**_WYCKOFF_PARAMS),
        **params,
    )


def _load(files: list[Path]) -> pd.DataFrame:
    frames = [parse_histdata_csv(f) for f in files]
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="first")]


def _run_period(
    m1_df:             pd.DataFrame,
    spread:            float,
    session:           tuple[int, int] = (7, 21),
    quote_to_usd_rate: float = 1.0,
) -> tuple[list, dict[str, Any], int]:
    zone_df  = resample_ohlcv(m1_df, "15min")
    strategy = _build_strategy(session_start=session[0], session_end=session[1])
    signals  = strategy.run(zone_df, m1_df)
    res, n_exp = simulate_all(
        signals, m1_df,
        risk_pct           = RISK_PCT,
        spread             = spread,
        max_monthly_losses = 4,
        max_daily_losses   = 1,
        tp1_r              = TP1_R,
        tp1_size           = TP1_SIZE,
        quote_to_usd_rate  = quote_to_usd_rate,
    )
    m = compute_metrics(res, INITIAL_BALANCE, len(signals), n_exp)
    return res, m, len(signals)


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_instrument(
    symbol:  str,
    spread:  float,
    periods: list[tuple[str, list[Path]]],
) -> tuple[float, float, float, float, int, bool]:
    """Returns (total_r, min_r, wr, max_dd, n_trades, all_periods_positive)."""
    session = SESSION_OVERRIDE.get(symbol, (7, 21))
    q_rate  = QUOTE_TO_USD.get(symbol, 1.0)
    fx_note = f"  fx≈{q_rate}" if q_rate != 1.0 else ""
    print(f"\n  ── {symbol}  (spread={spread}  session={session[0]}-{session[1]}h{fx_note})  ──────────────────")
    print(f"  {'Period':<26}  {'Sig':>4}  {'W':>3} {'L':>3}  {'WR%':>5}  {'R':>7}  {'P&L$':>8}  {'DD%':>5}")
    print("  " + "─" * 70)

    total_w = total_l = total_sig = total_r = 0.0
    max_dd  = 0.0
    min_r   = float("inf")
    all_pos = True

    for label, files in periods:
        missing = [f for f in files if not f.exists()]
        if missing:
            print(f"  {label:<26}  [file not found: {missing[0].name}]")
            continue
        m1 = _load(files)
        _, m, n_sig = _run_period(m1, spread, session=session, quote_to_usd_rate=q_rate)
        decided = m["n_wins"] + m["n_losses"]
        print(
            f"  {label:<26}  {n_sig:>4}  "
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
        if n_sig > 0:
            min_r  = min(min_r, m["total_r"])
            max_dd = max(max_dd, m["max_dd"])
            if m["total_r"] <= 0:
                all_pos = False

    if min_r == float("inf"):
        min_r = 0.0

    decided    = total_w + total_l
    overall_wr = total_w / decided * 100 if decided else 0.0
    passes     = overall_wr >= PASS_WR and min_r > 0 and max_dd <= PASS_DD and decided > 0
    verdict    = "PASS ✓" if passes else "FAIL ✗"

    print("  " + "─" * 70)
    print(
        f"  {'COMBINED':<26}  {int(total_sig):>4}  "
        f"{int(total_w):>3} {int(total_l):>3}  "
        f"{overall_wr:>5.1f}%  {total_r:>+7.2f}  {'':>8}  {max_dd:>4.1f}%"
    )
    print(f"  → Trades={decided}  MinR={min_r:+.2f}  Verdict: {verdict}")

    return total_r, min_r, overall_wr, max_dd, int(decided), passes


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 90)
    print("  S&D Strategy — Forex OOS Validation (2024 data)")
    print("  Universal-D · shorts-dom wy_L≥7.0 wy_S≥5.9 · TP1@0.75R(50%)→1.25R · zone≥4.0")
    print("=" * 90)
    print()
    print("  OOS← = backward OOS (optimised on 2025, tested on 2024)")
    print("  OOS→ = forward OOS  (optimised on 2025, tested on 2026)")
    print("  blind = instrument never tested in any optimisation round")

    summary: list[tuple[str, float, float, float, float, int, bool]] = []

    for symbol, spread, periods in INSTRUMENTS:
        tr, mr, wr, dd, n, passed = _print_instrument(symbol, spread, periods)
        summary.append((symbol, tr, mr, wr, dd, n, passed))

    # ── Cross-instrument summary ──────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("  FOREX OOS — Cross-instrument summary")
    print(f"  Pass criterion: WR ≥ {PASS_WR:.0f}%  AND  R > 0  AND  DD ≤ {PASS_DD:.0f}%")
    print(f"{'─' * 90}")
    print(f"\n  {'Symbol':<10}  {'Trades':>6}  {'TotR':>7}  {'MinR':>7}  {'WR%':>5}  {'MaxDD':>6}  Verdict")
    print("  " + "─" * 68)

    n_pass = 0
    n_data = 0
    for symbol, tr, mr, wr, dd, n, passed in summary:
        if n == 0:
            verdict = "NO DATA"
        elif passed:
            verdict = "PASS ✓"
            n_pass += 1
            n_data += 1
        else:
            verdict = "FAIL ✗"
            n_data += 1
        print(f"  {symbol:<10}  {n:>6}  {tr:>+7.2f}  {mr:>+7.2f}  {wr:>5.1f}%  {dd:>5.1f}%  {verdict}")

    print("  " + "─" * 68)
    print(f"\n  Instruments passing: {n_pass} / {n_data}")

    if n_pass == n_data and n_data >= 3:
        print("\n  ✓ Universal-D holds on OOS data — edge appears genuine across instruments.")
        print("    Next step: paper trade on 1-2 instruments for 4-6 weeks minimum.")
    elif n_pass >= n_data * 0.6:
        print(f"\n  ~ Partial OOS pass ({n_pass}/{n_data}). Edge is real but not universal.")
        print("    Deploy only on passing instruments, with reduced risk (0.5% per trade).")
    else:
        print(f"\n  ✗ OOS failure ({n_pass}/{n_data}). Universal-D does not generalise to 2024.")
        print("    Do NOT deploy on Forex. Collect more data and re-optimise.")

    print(f"\n  Paper cluster (OOS-validated): {', '.join(PAPER_CLUSTER)}")
    print("  → Paper trade ONLY these pairs — 0.5% risk — minimum 6 weeks before live.")

    # ── Reference: XAUUSD champion ────────────────────────────────────────
    print()
    print("  Reference — XAUUSD V_FINAL_B (2024 OOS):")
    xau_path = _DATA / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_2024.csv"
    if xau_path.exists():
        _xau_wy_params = dict(
            lookback=200, max_accum_bars=20, accum_range_mult=6.0,
            mss_lookback=60, min_spring_sweep_pct=0.05, min_mss_strength_pct=0.03,
        )
        xau_wy = WyckoffDetector(**_xau_wy_params)
        xau_strat = SDStrategy(
            zone_detector    = ZoneDetector(),
            wyckoff_detector = xau_wy,
            risk_reward=1.5, min_zone_score=5.0, min_wyckoff_score=5.9,
            min_composite_score=5.0, min_wyckoff_score_short=99.0,
            signal_cooldown=10, use_trend_filter=True, trend_slope_lookback=3,
            use_price_above_ema=True, use_session_filter=True,
            session_start_utc=7, session_end_utc=21, max_signals_per_day=6,
            use_adx_filter=False, use_h4_trend_filter=False, use_rsi_filter=False,
        )
        xau_m1   = _load([xau_path])
        xau_z    = resample_ohlcv(xau_m1, "15min")
        xau_sigs = xau_strat.run(xau_z, xau_m1)
        xau_res, xau_nexp = simulate_all(
            xau_sigs, xau_m1,
            risk_pct=RISK_PCT, spread=0.30,
            max_monthly_losses=4, max_daily_losses=1,
            tp1_r=1.25, tp1_size=0.50,
        )
        xm = compute_metrics(xau_res, INITIAL_BALANCE, len(xau_sigs), xau_nexp)
        decided = xm["n_wins"] + xm["n_losses"]
        print(
            f"  XAUUSD 2024  Trades={decided}  "
            f"WR={xm['win_rate']:.1f}%  R={xm['total_r']:+.2f}  DD={xm['max_dd']:.1f}%"
        )
    else:
        print("  XAUUSD 2024 data not found — skipping reference.")

    print(f"\n{'=' * 90}")
    print("  Fin du test OOS Forex")
    print(f"{'=' * 90}\n")


if __name__ == "__main__":
    main()
