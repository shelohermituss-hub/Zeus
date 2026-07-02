"""
FVG Retest Strategy — backtest on XAUUSD 2024 & 2025.
======================================================

Tests 12 parameter combinations across 5 groups.
2024 = development / exploration  |  2025 = validation (out-of-sample for 2025 params)

Usage
-----
    python -m zeus.backtest.run_fvg_backtest

Simulation notes
----------------
Signals are generated on M15 data; sd_simulation.simulate_all() runs on the
same M15 DataFrame (bar_index = M15 index, entry at next M15 open).  Spread
and risk_pct match the S&D benchmarks for fair comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv, resample_ohlcv
from zeus.backtest.sd_simulation import compute_metrics, simulate_all
from zeus.strategy.fvg_retest_strategy import FVGRetestStrategy

_ROOT = Path(__file__).parent.parent.parent

INITIAL_BALANCE = 10_000.0
RISK_PCT        = 0.01
SPREAD_GOLD     = 0.30    # USD/oz

M1_2024 = _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_2024.csv"
M1_2025 = _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_2025.csv"


@dataclass
class VariantConfig:
    label:                str
    risk_reward:          float = 2.0
    min_fvg_atr_mult:     float = 0.10
    max_fvg_age_bars:     int   = 96    # M15 bars = 24 h
    use_h4_trend:         bool  = True
    signal_cooldown:      int   = 8     # M15 bars = 2 h
    max_signals_per_day:  int   = 4
    sl_buffer_atr_mult:   float = 0.10
    long_only:            bool  = False


VARIANTS: list[VariantConfig] = [
    # ── Group A: R:R search ───────────────────────────────────────────────
    VariantConfig("FA1 · R:R=1.5  H4 atr≥0.10 age24h cool2h d4",
                  risk_reward=1.5),
    VariantConfig("FA2 · R:R=2.0  H4 atr≥0.10 age24h cool2h d4 [BASELINE]",
                  risk_reward=2.0),
    VariantConfig("FA3 · R:R=2.5  H4 atr≥0.10 age24h cool2h d4",
                  risk_reward=2.5),
    VariantConfig("FA4 · R:R=3.0  H4 atr≥0.10 age24h cool2h d4",
                  risk_reward=3.0),

    # ── Group B: FVG quality / age ────────────────────────────────────────
    VariantConfig("FB1 · R:R=2.0  H4 atr≥0.20 age24h cool2h d4 (large FVG)",
                  min_fvg_atr_mult=0.20),
    VariantConfig("FB2 · R:R=2.0  H4 atr≥0.05 age24h cool2h d4 (small FVG)",
                  min_fvg_atr_mult=0.05),
    VariantConfig("FB3 · R:R=2.0  H4 atr≥0.10 age12h cool2h d4 (fresh FVG)",
                  max_fvg_age_bars=48),

    # ── Group C: H4 trend filter ──────────────────────────────────────────
    VariantConfig("FC1 · R:R=2.0  noH4 atr≥0.10 age24h cool2h d4",
                  use_h4_trend=False),

    # ── Group D: Signal frequency ─────────────────────────────────────────
    VariantConfig("FD1 · R:R=2.0  H4 atr≥0.10 age24h cool1h d6 (more)",
                  signal_cooldown=4, max_signals_per_day=6),
    VariantConfig("FD2 · R:R=2.0  H4 atr≥0.10 age24h cool3h d3 (fewer)",
                  signal_cooldown=12, max_signals_per_day=3),

    # ── Group E: Long-only (gold 2025 = strong bull trend) ───────────────
    VariantConfig("FE1 · R:R=2.0  H4 atr≥0.10 age24h cool2h d4 long-only",
                  long_only=True),
    VariantConfig("FE2 · R:R=1.5  H4 atr≥0.10 age24h cool1.5h d4 long-only",
                  risk_reward=1.5, signal_cooldown=6, long_only=True),
]


def _build_strategy(cfg: VariantConfig) -> FVGRetestStrategy:
    return FVGRetestStrategy(
        risk_reward         = cfg.risk_reward,
        min_fvg_atr_mult    = cfg.min_fvg_atr_mult,
        max_fvg_age_bars    = cfg.max_fvg_age_bars,
        use_h4_trend        = cfg.use_h4_trend,
        signal_cooldown     = cfg.signal_cooldown,
        max_signals_per_day = cfg.max_signals_per_day,
        sl_buffer_atr_mult  = cfg.sl_buffer_atr_mult,
        long_only           = cfg.long_only,
    )


def _run_variant(
    cfg:    VariantConfig,
    m15_df: pd.DataFrame,
) -> tuple[list, dict[str, Any], int]:
    strategy = _build_strategy(cfg)
    signals  = strategy.run(m15_df)
    results, n_expired = simulate_all(
        signals, m15_df,
        risk_pct           = RISK_PCT,
        spread             = SPREAD_GOLD,
        max_monthly_losses = 0,
    )
    m = compute_metrics(results, INITIAL_BALANCE, len(signals), n_expired)
    return results, m, len(signals)


def _line(label: str, m: dict, n_sig: int, n_months: float) -> None:
    sig_pm = n_sig / n_months if n_months > 0 else 0
    r_pm   = m["total_r"] / n_months if n_months > 0 else 0
    print(
        f"  {label:<58} "
        f"sig={n_sig:>4} ({sig_pm:>4.1f}/m)  "
        f"W={m['n_wins']:>3} L={m['n_losses']:>3}  "
        f"WR={m['win_rate']:>5.1f}%  "
        f"R={m['total_r']:>+7.2f} ({r_pm:>+5.2f}/m)  "
        f"DD={m['max_dd']:>4.1f}%"
    )


def _run_year(label: str, m1_path: Path, n_months: float) -> dict[str, dict]:
    print(f"\n{'═' * 100}")
    print(f"  {label}")
    print(f"{'═' * 100}")

    m1_df  = parse_histdata_csv(m1_path)
    m1_df  = m1_df[~m1_df.index.duplicated(keep="first")]
    m15_df = resample_ohlcv(m1_df, "15min")
    print(f"  {len(m1_df):,} barres M1  →  {len(m15_df):,} barres M15\n")

    rows: list[tuple] = []
    for cfg in VARIANTS:
        print(f"  [{cfg.label}] …", end="", flush=True)
        results, m, n_sig = _run_variant(cfg, m15_df)
        rows.append((cfg, results, m, n_sig))
        print(f" {n_sig} signaux")

    print(f"\n{'─' * 100}")
    print("  RÉSULTATS DÉTAILLÉS")
    print(f"{'─' * 100}")
    for cfg, _, m, n_sig in rows:
        _line(cfg.label, m, n_sig, n_months)

    best_r  = max(rows, key=lambda x: x[2]["total_r"])
    best_wr = max(rows, key=lambda x: x[2]["win_rate"] if x[3] >= 10 else 0)

    b_cfg, b_res, b_m, b_n = best_r
    print(f"\n  ★ MEILLEUR R TOTAL : {b_cfg.label}")
    print(
        f"    {b_n} signaux ({b_n/n_months:.1f}/mois)  "
        f"WR={b_m['win_rate']:.1f}%  R={b_m['total_r']:+.2f}  "
        f"DD={b_m['max_dd']:.1f}%"
    )

    w_cfg, _, w_m, w_n = best_wr
    print(f"\n  ★ MEILLEUR WR (≥10 sig) : {w_cfg.label}")
    print(
        f"    {w_n} signaux ({w_n/n_months:.1f}/mois)  "
        f"WR={w_m['win_rate']:.1f}%  R={w_m['total_r']:+.2f}  "
        f"DD={w_m['max_dd']:.1f}%"
    )

    if b_res:
        print(f"\n  Trades du meilleur R — {b_cfg.label}")
        print(f"  {'#':>3}  {'Date':>11}  {'Dir':>5}  {'Entrée':>8}  {'Sortie':>8}  "
              f"{'FVG range':>12}  {'R':>6}  {'P&L $':>8}")
        print("  " + "─" * 72)
        for k, r in enumerate(b_res, 1):
            sig = r.signal
            fvg_range = f"{sig.fvg_bottom:.0f}–{sig.fvg_top:.0f}"
            print(
                f"  {k:>3}  {str(sig.formed_at.date()):>11}  "
                f"{sig.direction:>5}  {r.entry_price:>8.2f}  {r.exit_price:>8.2f}  "
                f"{fvg_range:>12}  {r.pnl_r:>+6.2f}  {r.pnl_usd:>+8.2f}"
            )

    # Return summary dict for cross-year comparison
    return {cfg.label: m for cfg, _, m, _ in rows}


def main() -> None:
    print("=" * 100)
    print("  FVG Retest Strategy — Backtest XAUUSD  |  2024 (explore) + 2025 (validate)")
    print("=" * 100)

    res_2024 = _run_year("XAUUSD 2024 — EXPLORATION  (gold +27.1%)", M1_2024, 12.0)
    res_2025 = _run_year("XAUUSD 2025 — VALIDATION   (gold +64.5%)", M1_2025, 12.0)

    # Cross-year summary
    print(f"\n{'═' * 100}")
    print("  COMPARAISON CROISÉE  2024 vs 2025")
    print(f"{'─' * 100}")
    print(f"  {'Variante':<58}  {'──── 2024 ────':>26}  {'──── 2025 ────':>26}")
    print(f"  {'':58}  {'WR%':>6}  {'R tot':>7}  {'DD%':>5}  {'WR%':>6}  {'R tot':>7}  {'DD%':>5}")
    print(f"  {'─' * 98}")
    for cfg in VARIANTS:
        m24 = res_2024.get(cfg.label, {})
        m25 = res_2025.get(cfg.label, {})
        if not m24 or not m25:
            continue
        print(
            f"  {cfg.label:<58}  "
            f"{m24['win_rate']:>6.1f}%  {m24['total_r']:>+7.2f}  {m24['max_dd']:>4.1f}%  "
            f"{m25['win_rate']:>6.1f}%  {m25['total_r']:>+7.2f}  {m25['max_dd']:>4.1f}%"
        )

    print(f"\n{'═' * 100}")
    print("  Fin du backtest FVG")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    main()
