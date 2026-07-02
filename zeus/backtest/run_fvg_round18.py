"""
FVG Retest — Round 18 : Long-Only × R:R élevé.
================================================

Objectif : valider si long-only + R:R=3.0 améliore le profil EV/DD.

Hypothèse :
    FE1 (long-only R:R=2.0) → WR≈37%, EV≈+0.11R/trade, DD≈26%
    FA4 (bidir  R:R=3.0)    → WR≈29%, EV≈+0.16R/trade, DD≈31%
    FE3 (long-only R:R=3.0) → ?  EV = 0.37×3 − 0.63×1 ≈ +0.48R/trade (théorique)

Grille testée :
    R:R sweep long-only          : R:R 2.0 / 2.5 / 3.0 / 3.5
    Fréquence (cool/daily cap)   : cool1h d6 / cool2h d4 [base] / cool3h d3
    FVG age                      : ≤12h / ≤24h [base]
    FVG size                     : atr≥0.05 / atr≥0.10 [base] / atr≥0.20

Usage
-----
    python -m zeus.backtest.run_fvg_round18
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
SPREAD_GOLD     = 0.30

M1_2024 = _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_2024.csv"
M1_2025 = _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_2025.csv"


@dataclass
class Cfg:
    label:               str
    risk_reward:         float = 3.0
    min_fvg_atr_mult:    float = 0.10
    max_fvg_age_bars:    int   = 96
    signal_cooldown:     int   = 8
    max_signals_per_day: int   = 4
    long_only:           bool  = True


VARIANTS: list[Cfg] = [
    # ── Group A : R:R sweep (long-only) ──────────────────────────────────
    Cfg("R18-A1 · long-only R:R=2.0 [ref FE1]",  risk_reward=2.0),
    Cfg("R18-A2 · long-only R:R=2.5",             risk_reward=2.5),
    Cfg("R18-A3 · long-only R:R=3.0 ★",           risk_reward=3.0),
    Cfg("R18-A4 · long-only R:R=3.5",             risk_reward=3.5),

    # ── Group B : fréquence avec R:R=3.0 ─────────────────────────────────
    Cfg("R18-B1 · R:R=3.0 cool1h  d6 (plus)",    signal_cooldown=4,  max_signals_per_day=6),
    Cfg("R18-B2 · R:R=3.0 cool3h  d3 (moins)",   signal_cooldown=12, max_signals_per_day=3),
    Cfg("R18-B3 · R:R=3.0 cool4h  d2 (strict)",  signal_cooldown=16, max_signals_per_day=2),

    # ── Group C : âge FVG avec R:R=3.0 ──────────────────────────────────
    Cfg("R18-C1 · R:R=3.0 age≤12h (frais)",      max_fvg_age_bars=48),
    Cfg("R18-C2 · R:R=3.0 age≤48h (étendu)",     max_fvg_age_bars=192),

    # ── Group D : taille FVG avec R:R=3.0 ───────────────────────────────
    Cfg("R18-D1 · R:R=3.0 atr≥0.05 (petits)",    min_fvg_atr_mult=0.05),
    Cfg("R18-D2 · R:R=3.0 atr≥0.20 (grands)",    min_fvg_atr_mult=0.20),
    Cfg("R18-D3 · R:R=3.0 atr≥0.30 (très grands)", min_fvg_atr_mult=0.30),
]


def _build(cfg: Cfg) -> FVGRetestStrategy:
    return FVGRetestStrategy(
        risk_reward         = cfg.risk_reward,
        min_fvg_atr_mult    = cfg.min_fvg_atr_mult,
        max_fvg_age_bars    = cfg.max_fvg_age_bars,
        use_h4_trend        = True,
        signal_cooldown     = cfg.signal_cooldown,
        max_signals_per_day = cfg.max_signals_per_day,
        long_only           = cfg.long_only,
    )


def _run(cfg: Cfg, m15_df: pd.DataFrame) -> tuple[list, dict[str, Any], int]:
    strat   = _build(cfg)
    signals = strat.run(m15_df)
    results, n_exp = simulate_all(
        signals, m15_df, risk_pct=RISK_PCT, spread=SPREAD_GOLD, max_monthly_losses=0,
    )
    m = compute_metrics(results, INITIAL_BALANCE, len(signals), n_exp)
    return results, m, len(signals)


def _line(label: str, m: dict, n_sig: int, n_months: float) -> None:
    spm = n_sig / n_months if n_months else 0
    rpm = m["total_r"] / n_months if n_months else 0
    ev  = m["total_r"] / n_sig if n_sig else 0
    print(
        f"  {label:<48}  "
        f"sig={n_sig:>3} ({spm:>4.1f}/m)  "
        f"W={m['n_wins']:>3} L={m['n_losses']:>3}  "
        f"WR={m['win_rate']:>5.1f}%  "
        f"R={m['total_r']:>+7.2f} ({rpm:>+5.2f}/m)  "
        f"EV={ev:>+5.3f}R  "
        f"DD={m['max_dd']:>5.1f}%"
    )


def _run_year(label: str, m1_path: Path, n_months: float) -> dict[str, dict]:
    print(f"\n{'═' * 110}")
    print(f"  {label}")
    print(f"{'═' * 110}")

    m1_df  = parse_histdata_csv(m1_path)
    m1_df  = m1_df[~m1_df.index.duplicated(keep="first")]
    m15_df = resample_ohlcv(m1_df, "15min")
    print(f"  {len(m1_df):,} M1  →  {len(m15_df):,} M15\n")

    rows: list[tuple] = []
    for cfg in VARIANTS:
        print(f"  [{cfg.label}] …", end="", flush=True)
        results, m, n_sig = _run(cfg, m15_df)
        rows.append((cfg, results, m, n_sig))
        print(f" {n_sig} signaux")

    print(f"\n{'─' * 110}")
    print("  RÉSULTATS")
    print(f"{'─' * 110}")
    for cfg, _, m, n_sig in rows:
        _line(cfg.label, m, n_sig, n_months)

    best_r  = max(rows, key=lambda x: x[2]["total_r"])
    best_ev = max(rows, key=lambda x: x[2]["total_r"] / x[3] if x[3] >= 5 else -99)
    best_dd = min(rows, key=lambda x: x[2]["max_dd"] if x[3] >= 5 else 99)

    b_cfg, b_res, b_m, b_n = best_r
    e_cfg, _, e_m, e_n     = best_ev
    d_cfg, _, d_m, d_n     = best_dd

    print(f"\n  ★ Meilleur R total : {b_cfg.label}")
    print(f"    {b_n} sig · WR={b_m['win_rate']:.1f}% · R={b_m['total_r']:+.2f} · DD={b_m['max_dd']:.1f}%")
    print(f"\n  ★ Meilleur EV/trade : {e_cfg.label}")
    print(f"    {e_n} sig · WR={e_m['win_rate']:.1f}% · R={e_m['total_r']:+.2f} · EV={e_m['total_r']/e_n:+.3f}R")
    print(f"\n  ★ Meilleur DD (≥5 sig) : {d_cfg.label}")
    print(f"    {d_n} sig · WR={d_m['win_rate']:.1f}% · R={d_m['total_r']:+.2f} · DD={d_m['max_dd']:.1f}%")

    if b_res:
        sig0 = b_res[0].signal
        print(f"\n  Trades du meilleur R — {b_cfg.label}")
        print(f"  {'#':>3}  {'Date':>11}  {'Entrée':>8}  {'Sortie':>8}  {'FVG':>12}  {'R':>6}  {'P&L $':>8}")
        print("  " + "─" * 68)
        for k, r in enumerate(b_res, 1):
            sig = r.signal
            print(
                f"  {k:>3}  {str(sig.formed_at.date()):>11}  "
                f"{r.entry_price:>8.2f}  {r.exit_price:>8.2f}  "
                f"{sig.fvg_bottom:.0f}–{sig.fvg_top:.0f}  "
                f"{r.pnl_r:>+6.2f}  {r.pnl_usd:>+8.2f}"
            )

    return {cfg.label: m for cfg, _, m, _ in rows}


def main() -> None:
    print("=" * 110)
    print("  FVG Round 18 — Long-Only × R:R élevé  |  XAUUSD 2024 + 2025")
    print("=" * 110)

    res_2024 = _run_year("XAUUSD 2024 — EXPLORATION  (gold +27.1%)", M1_2024, 12.0)
    res_2025 = _run_year("XAUUSD 2025 — VALIDATION   (gold +64.5%)", M1_2025, 12.0)

    print(f"\n{'═' * 110}")
    print("  COMPARAISON CROISÉE  2024 vs 2025")
    print(f"{'─' * 110}")
    print(f"  {'Variante':<48}  {'──── 2024 ────':>32}  {'──── 2025 ────':>32}")
    print(f"  {'':48}  {'WR%':>6}  {'R tot':>7}  {'EV':>7}  {'DD%':>5}  {'WR%':>6}  {'R tot':>7}  {'EV':>7}  {'DD%':>5}")
    print(f"  {'─' * 108}")
    for cfg in VARIANTS:
        m24 = res_2024.get(cfg.label, {})
        m25 = res_2025.get(cfg.label, {})
        if not m24 or not m25:
            continue
        n24 = sum(1 for v in res_2024.values() if v is m24) or 1
        ev24 = m24["total_r"] / (m24["n_wins"] + m24["n_losses"] + m24.get("n_expired", 0)) if (m24["n_wins"] + m24["n_losses"]) > 0 else 0
        ev25 = m25["total_r"] / (m25["n_wins"] + m25["n_losses"] + m25.get("n_expired", 0)) if (m25["n_wins"] + m25["n_losses"]) > 0 else 0
        print(
            f"  {cfg.label:<48}  "
            f"{m24['win_rate']:>6.1f}%  {m24['total_r']:>+7.2f}  {ev24:>+6.3f}R  {m24['max_dd']:>4.1f}%  "
            f"{m25['win_rate']:>6.1f}%  {m25['total_r']:>+7.2f}  {ev25:>+6.3f}R  {m25['max_dd']:>4.1f}%"
        )

    print(f"\n{'═' * 110}")
    print("  Fin Round 18")
    print(f"{'=' * 110}\n")


if __name__ == "__main__":
    main()
