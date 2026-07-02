"""
S&D Strategy — Iterative Optimisation on XAUUSD 2025
=====================================================

AUDIT FINDINGS (from sd_signal_funnel.py on 2025 full year):
─────────────────────────────────────────────────────────────
  M1 bars total              353,951
    → session 07–21 UTC      214,014   (60%)
    → price inside zone      104,087   (0.3% of zone-bar pairs)  ← normal
    → trend aligned (EMA50)   44,126   (57.6% cut)               ← aggressive
    → Wyckoff detected             21   (0.05% of eligible bars)  ← CRITICAL GAP
  SIGNALS EMITTED                  21

ROOT CAUSES:
  1. Wyckoff lookback = 30 M1 bars (30 min)
     The full accum→spring→MSS sequence must complete in 30 minutes.
     On M1 gold this is near-impossible: the pattern plays out over hours.
     _check_accum uses highs[0:accum_end] so large accum_end (≥9 bars required
     by the mss_lookback=10 constraint) forces a long tight-range requirement.
     Fix: lookback=200, max_accum_bars=20, mss_lookback=60.

  2. EMA50 slope over 20 M15 bars = 5 h is slow.
     May 2025: slope was still bullish from April's ATH despite May pullback →
     5 demand trades all stopped out (-5R, -$492).
     Fix: slope_lookback=8 M15 bars (2 h) + require price > EMA50.

  3. No market-regime filter: signals fire in choppy, low-ADX markets.
     Fix (V2+): ADX ≥ 18 on M15.

ITERATION PLAN:
  V0  Baseline (current production, but with old wyckoff defaults)
  V1  Wide Wyckoff + faster trend (main structural fix)
  V2  V1 + price-above-EMA50 + ADX ≥ 18 (quality gates)
  V3  V2 + raised Wyckoff/zone score floors (elite signal quality)

Usage
-----
    python -m zeus.backtest.run_sd_iterations
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv, resample_ohlcv
from zeus.backtest.sd_simulation import (
    SPREAD_PER_OZ,
    TradeResult,
    compute_metrics,
    print_monthly_breakdown,
    simulate_all,
)
from zeus.strategy.supply_demand.sd_strategy import SDStrategy
from zeus.strategy.supply_demand.wyckoff import WyckoffDetector
from zeus.strategy.supply_demand.zone_detector import ZoneDetector

_ROOT    = Path(__file__).parent.parent.parent
_M1_FILE = _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_2025.csv"

INITIAL_BALANCE = 10_000.0
RISK_PCT        = 0.01


# ── Variant builder ───────────────────────────────────────────────────────────

@dataclass
class VariantConfig:
    label:               str
    # Wyckoff
    wy_lookback:         int
    wy_max_accum:        int
    wy_accum_mult:       float
    wy_mss_lb:           int
    wy_sweep_pct:        float
    wy_mss_str_pct:      float
    # Strategy
    min_zone_score:      float
    min_wyckoff_score:   float
    trend_slope_lb:      int
    use_price_above_ema: bool
    signal_cooldown:     int
    daily_cap:           int
    use_adx_filter:      bool  = False
    adx_min:             float = 20.0
    # Simulation risk gates / exit management
    risk_reward:         float = 3.0  # R:R target ratio
    max_daily_losses:    int   = 0   # 0 = unlimited
    max_monthly_losses:  int   = 0   # 0 = unlimited
    use_be:              bool  = False  # break-even at +1R
    first_signal_per_zone: bool = False  # one signal per zone ever


# ── Wide Wyckoff base config (reused across rounds) ───────────────────────────
_WW = dict(
    wy_lookback        = 200,
    wy_max_accum       = 20,
    wy_accum_mult      = 6.0,
    wy_mss_lb          = 60,
    wy_sweep_pct       = 0.05,
    wy_mss_str_pct     = 0.03,
)

VARIANTS: list[VariantConfig] = [
    # ── Anchors ───────────────────────────────────────────────────────────────
    VariantConfig(
        label              = "V6  · zone≥6.0 wy≥5.5 R:R=3.0 [R2 best]",
        **_WW,
        min_zone_score     = 6.0, min_wyckoff_score=5.5,
        trend_slope_lb     = 6,   use_price_above_ema=False,
        signal_cooldown    = 25,  daily_cap=3,
    ),
    VariantConfig(
        label              = "V47 · zone≥5.5 wy≥5.9 cool15 R:R=2.5 [R10 best]",
        **_WW,
        min_zone_score     = 5.5, min_wyckoff_score=5.9,
        trend_slope_lb     = 6,   use_price_above_ema=False,
        signal_cooldown    = 15,  daily_cap=4,
        risk_reward        = 2.5,
    ),
    # ── Round 11 — robustness gates on V47 champion ───────────────────────────
    # V47 = 59 sigs, 47.5% WR, +39R, $4,285, DD=6.2%. Meets all targets.
    # Final pass: monthly loss cap + BE to confirm stability isn't luck.
    # Also test cool10 (max volume) and the 3.0 R:R on cool15 config.
    VariantConfig(
        label              = "V53 · V47 + BE",
        **_WW,
        min_zone_score     = 5.5, min_wyckoff_score=5.9,
        trend_slope_lb     = 6,   use_price_above_ema=False,
        signal_cooldown    = 15,  daily_cap=4,
        risk_reward        = 2.5,
        use_be             = True,
    ),
    VariantConfig(
        label              = "V54 · V47 + mloss4",
        **_WW,
        min_zone_score     = 5.5, min_wyckoff_score=5.9,
        trend_slope_lb     = 6,   use_price_above_ema=False,
        signal_cooldown    = 15,  daily_cap=4,
        risk_reward        = 2.5,
        max_monthly_losses = 4,
    ),
    VariantConfig(
        label              = "V55 · V47 + mloss4 + BE",
        **_WW,
        min_zone_score     = 5.5, min_wyckoff_score=5.9,
        trend_slope_lb     = 6,   use_price_above_ema=False,
        signal_cooldown    = 15,  daily_cap=4,
        risk_reward        = 2.5,
        max_monthly_losses = 4,
        use_be             = True,
    ),
    VariantConfig(
        label              = "V56 · zone≥5.5 wy≥5.9 cool10 R:R=2.5",
        **_WW,
        min_zone_score     = 5.5, min_wyckoff_score=5.9,
        trend_slope_lb     = 6,   use_price_above_ema=False,
        signal_cooldown    = 10,  daily_cap=5,
        risk_reward        = 2.5,
    ),
    VariantConfig(
        label              = "V57 · zone≥5.5 wy≥5.9 cool15 R:R=2.5 mloss5",
        **_WW,
        min_zone_score     = 5.5, min_wyckoff_score=5.9,
        trend_slope_lb     = 6,   use_price_above_ema=False,
        signal_cooldown    = 15,  daily_cap=4,
        risk_reward        = 2.5,
        max_monthly_losses = 5,
    ),
    VariantConfig(
        label              = "V58 · zone≥5.5 wy≥5.9 cool15 R:R=2.5 mloss3",
        **_WW,
        min_zone_score     = 5.5, min_wyckoff_score=5.9,
        trend_slope_lb     = 6,   use_price_above_ema=False,
        signal_cooldown    = 15,  daily_cap=4,
        risk_reward        = 2.5,
        max_monthly_losses = 3,
    ),
]


def _build_strategy(cfg: VariantConfig) -> SDStrategy:
    return SDStrategy(
        zone_detector        = ZoneDetector(),
        wyckoff_detector     = WyckoffDetector(
            lookback             = cfg.wy_lookback,
            max_accum_bars       = cfg.wy_max_accum,
            accum_range_mult     = cfg.wy_accum_mult,
            mss_lookback         = cfg.wy_mss_lb,
            min_spring_sweep_pct = cfg.wy_sweep_pct,
            min_mss_strength_pct = cfg.wy_mss_str_pct,
        ),
        risk_reward          = cfg.risk_reward,
        min_zone_score       = cfg.min_zone_score,
        min_wyckoff_score    = cfg.min_wyckoff_score,
        min_composite_score  = cfg.min_zone_score,
        signal_cooldown      = cfg.signal_cooldown,
        use_trend_filter     = True,
        trend_slope_lookback = cfg.trend_slope_lb,
        use_price_above_ema  = cfg.use_price_above_ema,
        use_session_filter   = True,
        max_signals_per_day  = cfg.daily_cap,
        use_adx_filter         = cfg.use_adx_filter,
        adx_min                = cfg.adx_min,
        first_signal_per_zone  = cfg.first_signal_per_zone,
    )


# ── Runner ────────────────────────────────────────────────────────────────────

def _run_variant(
    cfg:   VariantConfig,
    m1_df: pd.DataFrame,
    m15_df: pd.DataFrame,
) -> tuple[list[TradeResult], dict[str, Any], int]:
    strategy = _build_strategy(cfg)
    signals  = strategy.run(m15_df, m1_df)
    results, n_expired = simulate_all(
        signals, m1_df,
        risk_pct           = RISK_PCT,
        spread             = SPREAD_PER_OZ,
        max_daily_losses   = cfg.max_daily_losses,
        max_monthly_losses = cfg.max_monthly_losses,
        use_be             = cfg.use_be,
    )
    m = compute_metrics(results, INITIAL_BALANCE, len(signals), n_expired)
    return results, m, len(signals)


# ── Reporting ─────────────────────────────────────────────────────────────────

def _summary_line(cfg: VariantConfig, m: dict, n_sig: int) -> None:
    wr  = m["win_rate"]
    r   = m["total_r"]
    pnl = m["total_usd"]
    dd  = m["max_dd"]
    scr = m.get("n_scratches", 0)
    scr_str = f" S={scr:>2}" if scr > 0 else ""
    print(
        f"  {cfg.label:<44} "
        f"sig={n_sig:>4}  "
        f"W={m['n_wins']:>3} L={m['n_losses']:>3}{scr_str}  "
        f"WR={wr:>5.1f}%  "
        f"R={r:>+7.2f}  "
        f"P&L={pnl:>+8.0f}$  "
        f"DD={dd:>5.1f}%"
    )


def main() -> None:
    print("=" * 78)
    print("  S&D Strategy — Iterative Optimisation · XAUUSD 2025")
    print("=" * 78)

    print(f"\nLoading {_M1_FILE.name} …")
    m1_df = parse_histdata_csv(_M1_FILE)
    m1_df = m1_df[~m1_df.index.duplicated(keep="first")]
    m15_df = resample_ohlcv(m1_df, "15min")
    print(f"  {len(m1_df):,} M1 bars  |  {len(m15_df):,} M15 bars\n")

    print("Running variants (this takes a few minutes) …\n")
    all_results = []
    for cfg in VARIANTS:
        print(f"  [{cfg.label}] …", end="", flush=True)
        results, m, n_sig = _run_variant(cfg, m1_df, m15_df)
        all_results.append((cfg, results, m, n_sig))
        print(f" {n_sig} signals")

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n" + "─" * 78)
    print("  COMPARISON TABLE — full year 2025")
    print("─" * 78)
    print(
        f"  {'Variant':<40} "
        f"{'Signals':>7}  {'W':>4} {'L':>4}  "
        f"{'WR%':>6}  {'TotalR':>8}  {'P&L$':>9}  {'MaxDD':>6}"
    )
    print("  " + "─" * 74)
    for cfg, results, m, n_sig in all_results:
        _summary_line(cfg, m, n_sig)

    # ── Best variant detail ───────────────────────────────────────────────────
    best = max(all_results, key=lambda x: x[2]["total_r"])
    best_cfg, best_results, best_m, best_n_sig = best

    print(f"\n{'═' * 78}")
    print(f"  BEST VARIANT: {best_cfg.label}")
    print(f"{'═' * 78}")
    print(f"\n  Signals : {best_n_sig}")
    print(f"  Wins    : {best_m['n_wins']}   ({best_m['win_rate']:.1f} %)")
    print(f"  Losses  : {best_m['n_losses']}")
    print(f"  Total R : {best_m['total_r']:+.2f}R")
    print(f"  P&L     : ${best_m['total_usd']:+,.2f}")
    print(f"  Max DD  : {best_m['max_dd']:.1f} %")
    print(f"  Spread  : ${best_m['total_spread']:,.2f}")

    if best_results:
        print("\n  Trade log:")
        print(f"  {'#':>3}  {'Date':>11}  {'Dir':>5}  {'Entry':>9}  {'Exit':>9}  "
              f"{'R':>6}  {'P&L $':>8}  {'ZSc':>5}  {'WSc':>5}")
        print("  " + "─" * 75)
        for i, r in enumerate(best_results, 1):
            sig = r.signal
            ts  = sig.formed_at.strftime("%Y-%m-%d")
            print(
                f"  {i:>3}  {ts:>11}  {sig.direction:>5}  "
                f"{r.entry_price:>9.3f}  {r.exit_price:>9.3f}  "
                f"{r.pnl_r:>+5.1f}R  {r.pnl_usd:>+8.2f}  "
                f"{sig.zone_score:>5.2f}  {sig.wyckoff_score:>5.2f}"
            )

    print()
    print_monthly_breakdown(best_results, INITIAL_BALANCE)


if __name__ == "__main__":
    main()
