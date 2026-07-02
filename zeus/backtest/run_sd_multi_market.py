"""
S&D Strategy — Backtest multi-marché  (GBPUSD · NZDUSD · USDCAD  juin 2026)
=============================================================================

Compare les meilleures variantes S&D identifiées sur XAUUSD 2025 contre les
3 marchés forex disponibles (données juin 2026).

Variantes testées (champions identifiés sur XAUUSD 2025) :
  V54  — champion bidirectionnel (cool15, zone≥5.5, wy≥5.9, R:R=2.5, mloss4)
  V73  — long-only, meilleur R/trade (no tp1)
  V74  — long-only + TP1=0.8R, WR 75.9 %
  V78  — long-only + TP1=0.6R, WR 79.3 %
  V82  — price_ema bidirectionnel, meilleur R/mois (no tp1)
  V88  — price_ema zone≥5.0 wy≥5.5 d6 no_tp (Round 16 élargi)
  V91  — Wyckoff asymétrique wy_L≥5.5 wy_S≥7.0 (Round 16 nouveau)

Spreads forex :
  GBPUSD  1.5 pip = 0.00015
  NZDUSD  2.0 pip = 0.00020
  USDCAD  2.0 pip = 0.00020

Usage
-----
    python -m zeus.backtest.run_sd_multi_market
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv, resample_ohlcv
from zeus.backtest.sd_simulation import (
    TradeResult,
    compute_metrics,
    simulate_all,
)
from zeus.strategy.supply_demand.sd_strategy import SDStrategy
from zeus.strategy.supply_demand.wyckoff import WyckoffDetector
from zeus.strategy.supply_demand.zone_detector import ZoneDetector

_ROOT = Path(__file__).parent.parent.parent

INITIAL_BALANCE = 10_000.0
RISK_PCT        = 0.01

# ── Marchés disponibles ───────────────────────────────────────────────────────

@dataclass
class MarketConfig:
    name:       str
    m1_file:    Path
    spread:     float   # spread en unités de prix (pas en pips)
    label:      str

MARKETS: list[MarketConfig] = [
    MarketConfig(
        name    = "GBPUSD",
        m1_file = _ROOT / "data" / "historical" / "gbpusd" / "m1" / "DAT_MT_GBPUSD_M1_202606.csv",
        spread  = 0.00015,   # 1.5 pip
        label   = "GBPUSD  juin 2026  (spread 1.5 pip)",
    ),
    MarketConfig(
        name    = "NZDUSD",
        m1_file = _ROOT / "data" / "historical" / "nzdusd" / "m1" / "DAT_MT_NZDUSD_M1_202606.csv",
        spread  = 0.00020,   # 2.0 pip
        label   = "NZDUSD  juin 2026  (spread 2.0 pip)",
    ),
    MarketConfig(
        name    = "USDCAD",
        m1_file = _ROOT / "data" / "historical" / "usdcad" / "m1" / "DAT_MT_USDCAD_M1_202606.csv",
        spread  = 0.00020,   # 2.0 pip
        label   = "USDCAD  juin 2026  (spread 2.0 pip)",
    ),
]

# ── Config variantes ──────────────────────────────────────────────────────────

@dataclass
class VariantConfig:
    label:                   str
    wy_lookback:             int
    wy_max_accum:            int
    wy_accum_mult:           float
    wy_mss_lb:               int
    wy_sweep_pct:            float
    wy_mss_str_pct:          float
    min_zone_score:          float
    min_wyckoff_score:       float
    trend_slope_lb:          int
    use_price_above_ema:     bool
    signal_cooldown:         int
    daily_cap:               int
    risk_reward:             float         = 2.5
    max_monthly_losses:      int           = 0
    min_zone_score_long:     Optional[float] = None
    min_zone_score_short:    Optional[float] = None
    min_wyckoff_score_long:  Optional[float] = None
    min_wyckoff_score_short: Optional[float] = None
    tp1_r:                   float         = 0.0
    tp1_size:                float         = 0.5


_WW = dict(
    wy_lookback        = 200,
    wy_max_accum       = 20,
    wy_accum_mult      = 6.0,
    wy_mss_lb          = 60,
    wy_sweep_pct       = 0.05,
    wy_mss_str_pct     = 0.03,
)

VARIANTS: list[VariantConfig] = [
    # V54 — champion XAUUSD 2025
    VariantConfig(
        label              = "V54  · zone≥5.5 wy≥5.9 cool15 R:R=2.5 mloss4 [champion]",
        **_WW,
        min_zone_score     = 5.5, min_wyckoff_score=5.9,
        trend_slope_lb     = 6,   use_price_above_ema=False,
        signal_cooldown    = 15,  daily_cap=4,
        max_monthly_losses = 4,
    ),
    # V73 — long-only, meilleur R par trade
    VariantConfig(
        label              = "V73  · long-only zone≥5.5 wy≥5.9 cool15 no_tp",
        **_WW,
        min_zone_score        = 5.5, min_wyckoff_score=5.9,
        trend_slope_lb        = 6,   use_price_above_ema=False,
        signal_cooldown       = 15,  daily_cap=4,
        max_monthly_losses    = 4,
        min_zone_score_long   = 5.5,
        min_zone_score_short  = 999.0,
    ),
    # V74 — long-only + TP1=0.8R (WR 75.9 % sur XAUUSD)
    VariantConfig(
        label              = "V74  · long-only zone≥5.5 wy≥5.9 cool15 TP1=0.8R",
        **_WW,
        min_zone_score        = 5.5, min_wyckoff_score=5.9,
        trend_slope_lb        = 6,   use_price_above_ema=False,
        signal_cooldown       = 15,  daily_cap=4,
        max_monthly_losses    = 4,
        min_zone_score_long   = 5.5,
        min_zone_score_short  = 999.0,
        tp1_r                 = 0.8,
    ),
    # V78 — long-only + TP1=0.6R (WR 79.3 % sur XAUUSD)
    VariantConfig(
        label              = "V78  · long-only zone≥5.5 wy≥5.9 cool10 TP1=0.6R",
        **_WW,
        min_zone_score        = 5.5, min_wyckoff_score=5.9,
        trend_slope_lb        = 6,   use_price_above_ema=False,
        signal_cooldown       = 10,  daily_cap=6,
        max_monthly_losses    = 4,
        min_zone_score_long   = 5.5,
        min_zone_score_short  = 999.0,
        tp1_r                 = 0.6,
    ),
    # V82 — price_ema bidirectionnel, meilleur R/mois sur XAUUSD
    VariantConfig(
        label              = "V82  · price_ema zone≥5.5 wy≥5.9 cool15 no_tp",
        **_WW,
        min_zone_score     = 5.5, min_wyckoff_score=5.9,
        trend_slope_lb     = 6,   use_price_above_ema=True,
        signal_cooldown    = 15,  daily_cap=4,
        max_monthly_losses = 4,
    ),
    # V88 — price_ema zone élargie (Round 16 Group A)
    VariantConfig(
        label              = "V88  · price_ema zone≥5.0 wy≥5.5 cool10 d6 no_tp",
        **_WW,
        min_zone_score     = 5.0, min_wyckoff_score=5.5,
        trend_slope_lb     = 6,   use_price_above_ema=True,
        signal_cooldown    = 10,  daily_cap=6,
        max_monthly_losses = 4,
    ),
    # V91 — Wyckoff asymétrique (Round 16 Group B)
    VariantConfig(
        label              = "V91  · asym wy_L≥5.5 wy_S≥7.0 zone≥5.0 d6 price_ema",
        **_WW,
        min_zone_score     = 5.0, min_wyckoff_score=5.5,
        trend_slope_lb     = 6,   use_price_above_ema=True,
        signal_cooldown    = 10,  daily_cap=6,
        max_monthly_losses = 4,
        min_wyckoff_score_long  = 5.5,
        min_wyckoff_score_short = 7.0,
    ),
]


# ── Builder ───────────────────────────────────────────────────────────────────

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
    )


def _run_variant(
    cfg:    VariantConfig,
    m1_df:  pd.DataFrame,
    m15_df: pd.DataFrame,
    spread: float,
) -> tuple[list[TradeResult], dict[str, Any], int]:
    strategy = _build_strategy(cfg)
    signals  = strategy.run(m15_df, m1_df)
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
        f"  {cfg.label:<52} "
        f"sig={n_sig:>3}  "
        f"W={m['n_wins']:>2} L={m['n_losses']:>2}  "
        f"WR={m['win_rate']:>5.1f}%  "
        f"R={m['total_r']:>+7.2f}  "
        f"P&L={m['total_usd']:>+7.0f}$  "
        f"DD={m['max_dd']:>4.1f}%"
    )


def main() -> None:
    print("=" * 82)
    print("  S&D Strategy — Backtest multi-marché  ·  GBPUSD / NZDUSD / USDCAD  juin 2026")
    print("=" * 82)

    for mkt in MARKETS:
        print(f"\n{'─' * 82}")
        print(f"  {mkt.label}")
        print(f"{'─' * 82}")

        m1_df = parse_histdata_csv(mkt.m1_file)
        m1_df = m1_df[~m1_df.index.duplicated(keep="first")]
        m15_df = resample_ohlcv(m1_df, "15min")
        print(f"  {len(m1_df):,} barres M1  |  {len(m15_df):,} barres M15\n")

        rows = []
        for cfg in VARIANTS:
            print(f"  [{cfg.label}] …", end="", flush=True)
            results, m, n_sig = _run_variant(cfg, m1_df, m15_df, mkt.spread)
            rows.append((cfg, results, m, n_sig))
            print(f" {n_sig} signaux")

        print()
        for cfg, results, m, n_sig in rows:
            _line(cfg, m, n_sig)

        # Meilleure variante par R
        best = max(rows, key=lambda x: x[2]["total_r"])
        b_cfg, b_res, b_m, b_n = best
        print(f"\n  ★ Meilleure variante (R total) : {b_cfg.label}")
        print(f"    {b_n} signaux · {b_m['n_wins']}W {b_m['n_losses']}L · "
              f"WR={b_m['win_rate']:.1f}% · R={b_m['total_r']:+.2f} · "
              f"P&L={b_m['total_usd']:+,.0f}$ · DD={b_m['max_dd']:.1f}%")

        if b_res:
            print(f"\n  {'#':>3}  {'Date':>11}  {'Dir':>5}  {'Entrée':>9}  {'Sortie':>9}  "
                  f"{'R':>6}  {'P&L $':>7}")
            print("  " + "─" * 60)
            for k, r in enumerate(b_res, 1):
                sig = r.signal
                print(
                    f"  {k:>3}  {str(sig.formed_at.date()):>11}  "
                    f"{sig.direction:>5}  {r.entry_price:>9.5f}  {r.exit_price:>9.5f}  "
                    f"{r.pnl_r:>+6.2f}  {r.pnl_usd:>+7.2f}"
                )

    print(f"\n{'=' * 82}")
    print("  Fin du backtest multi-marché")
    print(f"{'=' * 82}\n")


if __name__ == "__main__":
    main()
