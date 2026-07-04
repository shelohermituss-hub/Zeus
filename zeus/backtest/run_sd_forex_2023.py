"""
S&D Strategy — 2023 Blind OOS Validation
=========================================

Runs the frozen Universal-D config on 2023 M1 data — data never seen
during any prior optimisation or validation (2024 was used for OOS,
earlier periods for IS tuning). This is a genuine blind OOS test.

Pairs tested:
  Paper cluster (re-validation):  GBPUSD, EURUSD, GBPAUD, EURNZD
  New candidates:                 AUDUSD, EURAUD, EURGBP

Pass criteria (same as 2024 OOS):
  win_rate >= 55%,  total_r > 0,  max_dd_pct <= 8%

Sessions:
  EUR/GBP pairs (London-driven) : 07-17h UTC
  AUDUSD                        : 07-21h UTC (London + NY overlap)

Quote-to-USD rates (approximate 2023 averages):
  GBPUSD = 1.00,  EURUSD = 1.00,  GBPAUD = 1.52,  EURNZD = 1.62
  AUDUSD = 1.00,  EURAUD = 1.52,  EURGBP = 1.27

Usage
-----
    python -m zeus.backtest.run_sd_forex_2023
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv, resample_ohlcv
from zeus.backtest.sd_simulation import compute_metrics, simulate_all
from zeus.strategy.supply_demand.sd_strategy import SDStrategy
from zeus.strategy.supply_demand.wyckoff import WyckoffDetector
from zeus.strategy.supply_demand.zone_detector import ZoneDetector

_ROOT = Path(__file__).parent.parent.parent
_DATA = _ROOT / "data" / "historical"

INITIAL_BALANCE = 10_000.0
RISK_PCT        = 0.005
TP1_R           = 0.75
TP1_SIZE        = 0.50

# ── All test pairs ────────────────────────────────────────────────────────────
# (symbol, spread, quote_to_usd, session_utc, data_files, is_paper_cluster)
ALL_PAIRS = [
    # Paper cluster — re-validation
    ("GBPUSD", 0.0001, 1.00, (7, 17),  [_DATA / "gbpusd" / "m1" / "DAT_MT_GBPUSD_M1_2023.csv"], True),
    ("EURUSD", 0.0001, 1.00, (7, 17),  [_DATA / "eurusd" / "m1" / "DAT_MT_EURUSD_M1_2023.csv"], True),
    ("GBPAUD", 0.0002, 1.52, (7, 17),  [_DATA / "gbpaud" / "m1" / "DAT_MT_GBPAUD_M1_2023.csv"], True),
    ("EURNZD", 0.0002, 1.62, (7, 17),  [_DATA / "eurnzd" / "m1" / "DAT_MT_EURNZD_M1_2023.csv"], True),
    # New candidates
    ("AUDUSD", 0.0002, 1.00, (7, 21),  [_DATA / "audusd" / "m1" / "DAT_MT_AUDUSD_M1_2023.csv"], False),
    ("EURAUD", 0.0002, 1.52, (7, 17),  [_DATA / "euraud" / "m1" / "DAT_MT_EURAUD_M1_2023.csv"], False),
    ("EURGBP", 0.0001, 1.27, (7, 17),  [_DATA / "eurgbp" / "m1" / "DAT_MT_EURGBP_M1_2023.csv"], False),
]

# ── Universal-D frozen params (identical to 2024 OOS run) ─────────────────────
_BASE_UNIVERSAL_D = dict(
    risk_reward             = 1.25,
    min_zone_score          = 4.0,
    min_wyckoff_score       = 7.0,
    min_wyckoff_score_short = 5.9,
    min_composite_score     = 4.0,
    signal_cooldown         = 10,
    use_trend_filter        = True,
    trend_slope_lookback    = 3,
    use_price_above_ema     = True,
    ema_atr_tolerance       = 0.5,
    use_session_filter      = True,
    max_signals_per_day     = 6,
    use_adx_filter          = False,
    use_rsi_filter          = False,
    min_sl_pips             = 5,
    pip_size                = 0.0001,
    min_score_product       = 0.0,
    use_h4_trend_filter     = False,
    use_first_touch_only    = False,
)

_BASE_WYCKOFF = dict(
    lookback             = 200,
    max_accum_bars       = 20,
    accum_range_mult     = 6.0,
    mss_lookback         = 60,
    min_spring_sweep_pct = 0.10,
    min_mss_strength_pct = 0.03,
)

PASS_WR  = 55.0
PASS_DD  = 8.0


def _build_strategy(session: tuple[int, int]) -> SDStrategy:
    return SDStrategy(
        zone_detector    = ZoneDetector(),
        wyckoff_detector = WyckoffDetector(**_BASE_WYCKOFF),
        session_start_utc = session[0],
        session_end_utc   = session[1],
        **_BASE_UNIVERSAL_D,
    )


def _run_pair(
    sym:          str,
    spread:       float,
    q2u:          float,
    session:      tuple[int, int],
    files:        list[Path],
) -> dict:
    frames = [parse_histdata_csv(f) for f in files if f.exists()]
    if not frames:
        return {"error": "no data"}
    m1_df  = pd.concat(frames).sort_index()
    m1_df  = m1_df[~m1_df.index.duplicated(keep="first")]
    m15_df = resample_ohlcv(m1_df, "15min")
    if len(m15_df) < 50:
        return {"error": "insufficient bars"}

    strategy = _build_strategy(session)
    signals  = strategy.run(m15_df, m1_df)

    if not signals:
        return {"n_trades": 0, "win_rate": 0.0, "total_r": 0.0, "max_dd": 0.0}

    results, n_exp = simulate_all(
        signals,
        m1_df,
        risk_pct           = RISK_PCT,
        spread             = spread,
        max_daily_losses   = 1,
        max_monthly_losses = 4,
        tp1_r              = TP1_R,
        tp1_size           = TP1_SIZE,
        initial_equity     = INITIAL_BALANCE,
        quote_to_usd_rate  = q2u,
    )
    if not results:
        return {"n_trades": 0, "win_rate": 0.0, "total_r": 0.0, "max_dd": 0.0}

    return compute_metrics(results, INITIAL_BALANCE, len(signals), n_exp)


def _flag(m: dict) -> str:
    if "error" in m:
        return " ⚠️"
    wr = m.get("win_rate", 0.0)
    tr = m.get("total_r",  0.0)
    dd = m.get("max_dd",   0.0)
    return " ✅" if (wr >= PASS_WR and tr > 0 and dd <= PASS_DD) else " ❌"


def main() -> None:
    hdr = f"{'Symbol':<10} {'Tag':<12} {'N':>4} {'WR%':>6} {'TotalR':>8} {'DD%':>6} {'Session':>12}"
    sep = "─" * len(hdr)

    print("\n" + "=" * 70)
    print("S&D FOREX — 2023 Blind OOS Validation (Universal-D)")
    print("Data: 2023 M1  |  Risk: 0.5%  |  FIRST USE of this dataset")
    print("=" * 70)

    paper_results  = []
    new_results    = []
    all_results    = []
    all_signals    = 0
    n_expired      = 0
    equity         = INITIAL_BALANCE

    for sym, spread, q2u, session, files, is_paper in ALL_PAIRS:
        m = _run_pair(sym, spread, q2u, session, files)

        tag = "re-val" if is_paper else "new"
        n   = m.get("n_trades", 0)
        wr  = m.get("win_rate",  0.0)
        tr  = m.get("total_r",  0.0)
        dd  = m.get("max_dd",   0.0)
        sess_str = f"{session[0]}-{session[1]}h UTC"
        flag = _flag(m)
        m["symbol"] = sym
        m["tag"]    = tag

        row = {"label": sym, **m}
        if is_paper:
            paper_results.append(row)
        else:
            new_results.append(row)

        print(f"  {sym:<10} {tag:<12} {n:>4} {wr:>6.1f} {tr:>8.2f} {dd:>6.1f} {sess_str:>12}{flag}")

        # Accumulate for combined metrics
        frames = [parse_histdata_csv(f) for f in files if f.exists()]
        if not frames:
            continue
        m1_df  = pd.concat(frames).sort_index()
        m1_df  = m1_df[~m1_df.index.duplicated(keep="first")]
        m15_df = resample_ohlcv(m1_df, "15min")
        if len(m15_df) < 50:
            continue
        strategy = _build_strategy(session)
        signals  = strategy.run(m15_df, m1_df)
        all_signals += len(signals)
        if not signals:
            continue
        results, exp = simulate_all(
            signals,
            m1_df,
            risk_pct           = RISK_PCT,
            spread             = spread,
            max_daily_losses   = 1,
            max_monthly_losses = 4,
            tp1_r              = TP1_R,
            tp1_size           = TP1_SIZE,
            initial_equity     = equity,
            quote_to_usd_rate  = q2u,
        )
        all_results.extend(results)
        n_expired += exp
        for r in results:
            equity += r.pnl_usd

    # ── Combined summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  COMBINED — all 7 pairs")
    print("=" * 70)
    if all_results:
        cm = compute_metrics(all_results, INITIAL_BALANCE, all_signals, n_expired)
        flag = _flag(cm)
        print(f"  {'All 7':<10} {'combined':<12} {cm['n_trades']:>4} {cm['win_rate']:>6.1f} "
              f"{cm['total_r']:>8.2f} {cm['max_dd']:>6.1f}{flag}")

    # ── Paper cluster only combined ───────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  PAPER CLUSTER ONLY — GBPUSD, EURUSD, GBPAUD, EURNZD")
    print("─" * 70)
    pc_trades = 0
    pc_wr     = []
    pc_tr     = 0.0
    for r in paper_results:
        pc_trades += r.get("n_trades", 0)
        if r.get("n_trades", 0) > 0:
            pc_wr.append(r.get("win_rate", 0.0))
        pc_tr += r.get("total_r", 0.0)

    print(f"\n  {'Symbol':<10} {'N':>4} {'WR%':>6} {'TotalR':>8} {'DD%':>6}")
    print("  " + "─" * 44)
    for r in paper_results:
        flag = _flag(r)
        print(f"  {r['label']:<10} {r.get('n_trades',0):>4} {r.get('win_rate',0.0):>6.1f} "
              f"{r.get('total_r',0.0):>8.2f} {r.get('max_dd',0.0):>6.1f}{flag}")

    print("\n  New candidates:")
    print("  " + "─" * 44)
    for r in new_results:
        flag = _flag(r)
        print(f"  {r['label']:<10} {r.get('n_trades',0):>4} {r.get('win_rate',0.0):>6.1f} "
              f"{r.get('total_r',0.0):>8.2f} {r.get('max_dd',0.0):>6.1f}{flag}")

    print()


if __name__ == "__main__":
    main()
