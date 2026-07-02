"""
S&D Strategy — Round 17: New angles on 3-month 2026 XAUUSD data
================================================================

Explores 4 groups vs V87 champion (price_ema, zone≥5.0, wy≥5.9, d6, no_tp, mloss4):

  Group A — R:R variations   : V96=1.5R  V97=2.0R  V98=3.0R  V99=3.5R
  Group B — H4 trend filter  : V100 (slope_lb=3)  V101 (slope_lb=5)
  Group C — Zone timeframe   : V102 M30  V103 H1  V104 M30+R:R=3.0  V105 H1+long-only
  Group D — RSI confirmation : V106 (<40/>60)  V107 (<35/>65)  V108 (<30/>70)

Data: XAUUSD M1 April+May+June 2026 (≈3 months)

Usage
-----
    python -m zeus.backtest.run_sd_round17
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv, resample_ohlcv
from zeus.backtest.sd_simulation import TradeResult, compute_metrics, simulate_all
from zeus.strategy.supply_demand.sd_strategy import SDStrategy
from zeus.strategy.supply_demand.wyckoff import WyckoffDetector
from zeus.strategy.supply_demand.zone_detector import ZoneDetector

_ROOT = Path(__file__).parent.parent.parent

INITIAL_BALANCE = 10_000.0
RISK_PCT        = 0.01

M1_FILES = [
    _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_202604.csv",
    _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_202605.csv",
    _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_202606.csv",
]

SPREAD_GOLD = 0.30   # USD per oz (≈ 0.1 pip on XAU/USD)


@dataclass
class VariantConfig:
    label:                   str
    risk_reward:             float         = 2.5
    min_zone_score:          float         = 5.0
    min_wyckoff_score:       float         = 5.9
    trend_slope_lb:          int           = 6
    use_price_above_ema:     bool          = True
    signal_cooldown:         int           = 10
    daily_cap:               int           = 6
    max_monthly_losses:      int           = 4
    tp1_r:                   float         = 0.0
    tp1_size:                float         = 0.5
    zone_timeframe:          str           = "15min"
    use_h4_trend_filter:     bool          = False
    h4_trend_slope_lb:       int           = 3
    use_rsi_filter:          bool          = False
    rsi_period:              int           = 14
    rsi_oversold:            float         = 35.0
    rsi_overbought:          float         = 65.0
    min_zone_score_long:     Optional[float] = None
    min_zone_score_short:    Optional[float] = None
    min_wyckoff_score_long:  Optional[float] = None
    min_wyckoff_score_short: Optional[float] = None


_WW = dict(
    wy_lookback        = 200,
    wy_max_accum       = 20,
    wy_accum_mult      = 6.0,
    wy_mss_lb          = 60,
    wy_sweep_pct       = 0.05,
    wy_mss_str_pct     = 0.03,
)

VARIANTS: list[VariantConfig] = [
    # ── Baseline: V87 champion ────────────────────────────────────────────────
    VariantConfig(label="V87  · BASELINE price_ema zone≥5.0 wy≥5.9 d6 no_tp mloss4 R:R=2.5"),

    # ── Group A: R:R variations (V96–V99) ─────────────────────────────────────
    VariantConfig(label="V96  · R:R=1.5  price_ema zone≥5.0 wy≥5.9 d6 no_tp mloss4", risk_reward=1.5),
    VariantConfig(label="V97  · R:R=2.0  price_ema zone≥5.0 wy≥5.9 d6 no_tp mloss4", risk_reward=2.0),
    VariantConfig(label="V98  · R:R=3.0  price_ema zone≥5.0 wy≥5.9 d6 no_tp mloss4", risk_reward=3.0),
    VariantConfig(label="V99  · R:R=3.5  price_ema zone≥5.0 wy≥5.9 d6 no_tp mloss4", risk_reward=3.5),

    # ── Group B: H4 trend filter (V100–V101) ─────────────────────────────────
    VariantConfig(
        label="V100 · H4_trend(lb=3) price_ema zone≥5.0 wy≥5.9 d6 no_tp mloss4",
        use_h4_trend_filter=True, h4_trend_slope_lb=3,
    ),
    VariantConfig(
        label="V101 · H4_trend(lb=5) price_ema zone≥5.0 wy≥5.9 d6 no_tp mloss4",
        use_h4_trend_filter=True, h4_trend_slope_lb=5,
    ),

    # ── Group C: Zone timeframe (V102–V105) ──────────────────────────────────
    VariantConfig(
        label="V102 · M30_zones price_ema zone≥5.0 wy≥5.9 d6 no_tp mloss4",
        zone_timeframe="30min",
    ),
    VariantConfig(
        label="V103 · H1_zones price_ema zone≥5.0 wy≥5.9 d6 no_tp mloss4",
        zone_timeframe="60min",
    ),
    VariantConfig(
        label="V104 · M30_zones R:R=3.0 price_ema zone≥5.0 wy≥5.9 d6 no_tp mloss4",
        zone_timeframe="30min", risk_reward=3.0,
    ),
    VariantConfig(
        label="V105 · H1_zones long-only price_ema zone≥5.0 wy≥5.9 d6 no_tp mloss4",
        zone_timeframe="60min",
        min_zone_score_long=5.0, min_zone_score_short=999.0,
    ),

    # ── Group D: RSI confirmation (V106–V108) ────────────────────────────────
    VariantConfig(
        label="V106 · RSI<40/>60 price_ema zone≥5.0 wy≥5.9 d6 no_tp mloss4",
        use_rsi_filter=True, rsi_oversold=40.0, rsi_overbought=60.0,
    ),
    VariantConfig(
        label="V107 · RSI<35/>65 price_ema zone≥5.0 wy≥5.9 d6 no_tp mloss4",
        use_rsi_filter=True, rsi_oversold=35.0, rsi_overbought=65.0,
    ),
    VariantConfig(
        label="V108 · RSI<30/>70 price_ema zone≥5.0 wy≥5.9 d6 no_tp mloss4",
        use_rsi_filter=True, rsi_oversold=30.0, rsi_overbought=70.0,
    ),
]


def _build_strategy(cfg: VariantConfig) -> SDStrategy:
    return SDStrategy(
        zone_detector        = ZoneDetector(),
        wyckoff_detector     = WyckoffDetector(
            lookback             = _WW["wy_lookback"],
            max_accum_bars       = _WW["wy_max_accum"],
            accum_range_mult     = _WW["wy_accum_mult"],
            mss_lookback         = _WW["wy_mss_lb"],
            min_spring_sweep_pct = _WW["wy_sweep_pct"],
            min_mss_strength_pct = _WW["wy_mss_str_pct"],
        ),
        risk_reward              = cfg.risk_reward,
        min_zone_score           = cfg.min_zone_score,
        min_wyckoff_score        = cfg.min_wyckoff_score,
        min_composite_score      = cfg.min_zone_score,
        signal_cooldown          = cfg.signal_cooldown,
        use_trend_filter         = True,
        trend_slope_lookback     = cfg.trend_slope_lb,
        use_price_above_ema      = cfg.use_price_above_ema,
        use_session_filter       = True,
        max_signals_per_day      = cfg.daily_cap,
        min_zone_score_long      = cfg.min_zone_score_long,
        min_zone_score_short     = cfg.min_zone_score_short,
        min_wyckoff_score_long   = cfg.min_wyckoff_score_long,
        min_wyckoff_score_short  = cfg.min_wyckoff_score_short,
        use_h4_trend_filter      = cfg.use_h4_trend_filter,
        h4_trend_slope_lb        = cfg.h4_trend_slope_lb,
        use_rsi_filter           = cfg.use_rsi_filter,
        rsi_period               = cfg.rsi_period,
        rsi_oversold             = cfg.rsi_oversold,
        rsi_overbought           = cfg.rsi_overbought,
    )


def _run_variant(
    cfg:    VariantConfig,
    m1_df:  pd.DataFrame,
    spread: float,
) -> tuple[list[TradeResult], dict[str, Any], int]:
    zone_df = resample_ohlcv(m1_df, cfg.zone_timeframe)
    strategy = _build_strategy(cfg)
    signals  = strategy.run(zone_df, m1_df)
    results, n_expired = simulate_all(
        signals, m1_df,
        risk_pct           = RISK_PCT,
        spread             = spread,
        max_monthly_losses = cfg.max_monthly_losses,
        tp1_r              = cfg.tp1_r,
        tp1_size           = cfg.tp1_size,
    )
    m = compute_metrics(results, INITIAL_BALANCE, len(signals), n_expired)
    return results, m, len(signals)


def _line(cfg: VariantConfig, m: dict, n_sig: int) -> None:
    print(
        f"  {cfg.label:<65} "
        f"sig={n_sig:>3}  "
        f"W={m['n_wins']:>2} L={m['n_losses']:>2}  "
        f"WR={m['win_rate']:>5.1f}%  "
        f"R={m['total_r']:>+7.2f}  "
        f"P&L={m['total_usd']:>+7.0f}$  "
        f"DD={m['max_dd']:>4.1f}%"
    )


def main() -> None:
    print("=" * 90)
    print("  S&D Strategy — Round 17 · XAUUSD  avril+mai+juin 2026  (≈3 mois)")
    print("=" * 90)

    # Load and concatenate 3 months of M1 data
    frames = []
    for path in M1_FILES:
        df = parse_histdata_csv(path)
        frames.append(df)
    m1_df = pd.concat(frames).sort_index()
    m1_df = m1_df[~m1_df.index.duplicated(keep="first")]
    print(f"\n  {len(m1_df):,} barres M1 chargées (3 mois : avr+mai+juin 2026)\n")

    rows: list[tuple] = []
    for cfg in VARIANTS:
        print(f"  [{cfg.label}] …", end="", flush=True)
        results, m, n_sig = _run_variant(cfg, m1_df, SPREAD_GOLD)
        rows.append((cfg, results, m, n_sig))
        print(f" {n_sig} signaux")

    print(f"\n{'─' * 90}")
    print("  TABLEAU DE COMPARAISON — Round 17  (avr+mai+juin 2026 XAUUSD)")
    print(f"{'─' * 90}")
    for cfg, _, m, n_sig in rows:
        _line(cfg, m, n_sig)

    best = max(rows, key=lambda x: x[2]["total_r"])
    b_cfg, b_res, b_m, b_n = best
    print(f"\n  MEILLEURE VARIANTE: {b_cfg.label}")
    print(
        f"  {b_n} signaux · {b_m['n_wins']}W {b_m['n_losses']}L · "
        f"WR={b_m['win_rate']:.1f}% · R={b_m['total_r']:+.2f} · "
        f"P&L={b_m['total_usd']:+,.0f}$ · DD={b_m['max_dd']:.1f}%"
    )

    if b_res:
        print(f"\n  {'#':>3}  {'Date':>11}  {'Dir':>5}  {'Entrée':>9}  {'Sortie':>9}  "
              f"{'R':>6}  {'P&L $':>8}")
        print("  " + "─" * 62)
        for k, r in enumerate(b_res, 1):
            sig = r.signal
            print(
                f"  {k:>3}  {str(sig.formed_at.date()):>11}  "
                f"{sig.direction:>5}  {r.entry_price:>9.2f}  {r.exit_price:>9.2f}  "
                f"{r.pnl_r:>+6.2f}  {r.pnl_usd:>+8.2f}"
            )

    print(f"\n{'=' * 90}")
    print("  Fin du Round 17")
    print(f"{'=' * 90}\n")


if __name__ == "__main__":
    main()
