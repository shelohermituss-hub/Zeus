"""
S&D Strategy — Forex Exploration  (Axes 1, 2, 3)
=================================================

Compares the Universal-D baseline against 3 enhancement axes on the
OOS-validated paper cluster (GBPUSD, EURUSD, GBPAUD, EURNZD) using 2024 data.

NOTE: 2024 data was used once for OOS validation. Using it again for further
exploration makes it semi-IS. Results show directional impact only — not a new
OOS validation. A genuine re-validation would require 2023 or later 2026 data.

Axes tested:
  Axe 1 — First-touch only   : use_first_touch_only=True (60-bar window)
  Axe 2 — H4 trend filter    : use_h4_trend_filter=True
  Axe 3 — Trailing stop      : use_trailing_stop=True, trailing_factor=0.5 × sl_dist

Configurations:
  BASELINE  : Universal-D as validated (no axes)
  +A1       : Baseline + first-touch only
  +A2       : Baseline + H4 filter
  +A3       : Baseline + trailing stop (0.5R trail after TP1)
  +A1+A2    : Baseline + first-touch + H4
  +A1+A2+A3 : All three axes combined

Usage
-----
    python -m zeus.backtest.run_sd_forex_explore
"""
from __future__ import annotations

from dataclasses import dataclass
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
RISK_PCT        = 0.005   # 0.5% — same as paper engine
TP1_R           = 0.75
TP1_SIZE        = 0.50
TRAILING_FACTOR = 0.5     # trail at 0.5R behind peak after TP1

# ── Paper cluster ─────────────────────────────────────────────────────────────
PAPER_CLUSTER = [
    ("GBPUSD", 0.0001, 1.0,  (7, 17), [_DATA / "gbpusd" / "m1" / "DAT_MT_GBPUSD_M1_2024.csv"]),
    ("EURUSD", 0.0001, 1.0,  (7, 17), [_DATA / "eurusd" / "m1" / "DAT_MT_EURUSD_M1_2024.csv"]),
    ("GBPAUD", 0.0002, 1.52, (7, 17), [_DATA / "gbpaud" / "m1" / "DAT_MT_GBPAUD_M1_2024.csv"]),
    ("EURNZD", 0.0002, 1.62, (7, 17), [_DATA / "eurnzd" / "m1" / "DAT_MT_EURNZD_M1_2024.csv"]),
]

# ── Universal-D frozen params ─────────────────────────────────────────────────
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
    # Axes off by default in baseline
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


# ── Configurations ────────────────────────────────────────────────────────────

@dataclass
class Config:
    label:              str
    use_h4:             bool  = False
    use_first_touch:    bool  = False
    use_trailing:       bool  = False
    trailing_factor:    float = 0.5


CONFIGS = [
    Config("BASELINE"),
    Config("+A1 first-touch",                   use_first_touch=True),
    Config("+A2 H4-filter",                     use_h4=True),
    Config("+A3 trailing",                       use_trailing=True),
    Config("+A1+A2",                             use_first_touch=True, use_h4=True),
    Config("+A1+A2+A3 all",                      use_first_touch=True, use_h4=True, use_trailing=True),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_strategy(session: tuple[int, int], use_h4: bool, use_first_touch: bool) -> SDStrategy:
    params = {
        **_BASE_UNIVERSAL_D,
        "session_start_utc":  session[0],
        "session_end_utc":    session[1],
        "use_h4_trend_filter":  use_h4,
        "use_first_touch_only": use_first_touch,
    }
    return SDStrategy(
        zone_detector    = ZoneDetector(),
        wyckoff_detector = WyckoffDetector(**_BASE_WYCKOFF),
        **params,
    )


def _run_symbol(
    sym:          str,
    spread:       float,
    quote_to_usd: float,
    session:      tuple[int, int],
    files:        list[Path],
    cfg:          Config,
) -> dict:
    """Run one config on one symbol. Returns metrics dict."""
    frames = [parse_histdata_csv(f) for f in files if f.exists()]
    if not frames:
        return {}
    m1_df = pd.concat(frames).sort_index()
    m1_df = m1_df[~m1_df.index.duplicated(keep="first")]
    m15_df = resample_ohlcv(m1_df, "15min")
    if len(m15_df) < 50:
        return {}

    strategy = _build_strategy(session, cfg.use_h4, cfg.use_first_touch)
    signals  = strategy.run(m15_df, m1_df)
    if not signals:
        return {"n_trades": 0, "win_rate": 0.0, "total_r": 0.0, "max_dd_pct": 0.0}

    # bar_index is M1-aligned → pass m1_df as df (no separate tick_df)
    results, n_exp = simulate_all(
        signals,
        m1_df,
        risk_pct          = RISK_PCT,
        spread            = spread,
        max_daily_losses  = 1,
        max_monthly_losses= 4,
        tp1_r             = TP1_R,
        tp1_size          = TP1_SIZE,
        initial_equity    = INITIAL_BALANCE,
        quote_to_usd_rate = quote_to_usd,
        use_trailing_stop = cfg.use_trailing,
        trailing_factor   = cfg.trailing_factor,
    )
    if not results:
        return {"n_trades": 0, "win_rate": 0.0, "total_r": 0.0, "max_dd": 0.0}

    return compute_metrics(results, INITIAL_BALANCE, len(signals), n_exp)


def _print_table(rows: list[dict]) -> None:
    """Print a compact comparison table."""
    hdr = f"{'Config':<24} {'N':>4} {'WR%':>6} {'TotalR':>8} {'DD%':>6}"
    print(hdr)
    print("─" * len(hdr))
    for r in rows:
        n   = r.get("n_trades", 0)
        wr  = r.get("win_rate", 0.0)
        tr  = r.get("total_r",  0.0)
        dd  = r.get("max_dd",   0.0)
        lbl = r["label"]
        flag = " ✅" if (wr >= 55 and tr > 0 and dd <= 8) else " ❌"
        print(f"{lbl:<24} {n:>4} {wr:>6.1f} {tr:>8.2f} {dd:>6.1f}{flag}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 70)
    print("S&D FOREX — Exploration: Axes 1 (First-touch) · 2 (H4) · 3 (Trailing)")
    print("Paper cluster: GBPUSD, EURUSD, GBPAUD, EURNZD  |  2024 data  |  0.5% risk")
    print("NOTE: 2024 was OOS validation data — directional comparison only")
    print("=" * 70)

    for sym, spread, q2u, session, files in PAPER_CLUSTER:
        print(f"\n{'─' * 70}")
        print(f"  {sym}  |  spread={spread}  |  session={session[0]}-{session[1]}h UTC")
        print(f"{'─' * 70}")

        rows = []
        for cfg in CONFIGS:
            m = _run_symbol(sym, spread, q2u, session, files, cfg)
            m["label"] = cfg.label
            rows.append(m)

        _print_table(rows)

    # ── Combined (all 4 pairs aggregated) ─────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  COMBINED — all 4 pairs (aggregated trades, shared 10k equity)")
    print(f"{'=' * 70}")

    combined_rows = []
    for cfg in CONFIGS:
        all_results  = []
        all_signals  = 0
        equity       = INITIAL_BALANCE
        n_expired    = 0

        for sym, spread, q2u, session, files in PAPER_CLUSTER:
            frames = [parse_histdata_csv(f) for f in files if f.exists()]
            if not frames:
                continue
            m1_df  = pd.concat(frames).sort_index()
            m1_df  = m1_df[~m1_df.index.duplicated(keep="first")]
            m15_df = resample_ohlcv(m1_df, "15min")
            if len(m15_df) < 50:
                continue

            strategy = _build_strategy(session, cfg.use_h4, cfg.use_first_touch)
            signals  = strategy.run(m15_df, m1_df)
            all_signals += len(signals)
            if not signals:
                continue

            results, exp = simulate_all(
                signals,
                m1_df,
                risk_pct          = RISK_PCT,
                spread            = spread,
                max_daily_losses  = 1,
                max_monthly_losses= 4,
                tp1_r             = TP1_R,
                tp1_size          = TP1_SIZE,
                initial_equity    = equity,
                quote_to_usd_rate = q2u,
                use_trailing_stop = cfg.use_trailing,
                trailing_factor   = cfg.trailing_factor,
            )
            all_results.extend(results)
            n_expired += exp
            for r in results:
                equity += r.pnl_usd

        if not all_results:
            combined_rows.append({"label": cfg.label, "n_trades": 0,
                                   "win_rate": 0.0, "total_r": 0.0,
                                   "max_dd_pct": 0.0, "min_r": 0.0})
            continue

        m = compute_metrics(all_results, INITIAL_BALANCE, all_signals, n_expired)
        m["label"] = cfg.label
        combined_rows.append(m)

    _print_table(combined_rows)
    print()


if __name__ == "__main__":
    main()
