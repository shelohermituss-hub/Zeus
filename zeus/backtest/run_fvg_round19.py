"""
FVG Round 19 — Audit des filtres qualité A–H.
=============================================

Objectif : atteindre WR 55-75% en empilant des filtres de qualité sur la
configuration R18-B2 (long-only, R:R=3.0, cooldown 3h, max 3/jour).

Données : 2025 Q1 (jan-mar, exploration) + Q4 (oct-déc, validation out-of-sample).
Test 3 mois × 2 fenêtres pour un résultat fiable sans temps de calcul excessif.

Filtres testés
--------------
A  require_bullish_bar       close > open sur la barre d'entrée
B  fvg_max_penetration_pct   price doit rester dans le tiers/moitié haute du FVG
C  use_h1_trend              close > H1 EMA21
D  use_m15_bos_filter        dernier BOS/CHoCH M15 = haussier (lookback 30 bars)
E  min_impulse_atr_mult=1.5  barre impulse du FVG ≥ 1.5× ATR14
F  session London seulement  7h–13h UTC
G  risk_reward=1.5           TP plus proche → WR plus haute
H  use_sd_confluence         FVG doit chevaucher une zone de demande M15 active

Usage
-----
    python -m zeus.backtest.run_fvg_round19
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

M1_2025 = _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_2025.csv"


@dataclass
class Cfg:
    """
    All parameters map to FVGRetestStrategy kwargs.
    Defaults = R18-B2 baseline.
    """
    label:                   str
    risk_reward:             float = 3.0
    signal_cooldown:         int   = 12     # 3h
    max_signals_per_day:     int   = 3
    session_start_utc:       int   = 7
    session_end_utc:         int   = 21
    # A
    require_bullish_bar:     bool  = False
    # B
    fvg_max_penetration_pct: float = 1.0
    # C
    use_h1_trend:            bool  = False
    h1_ema_span:             int   = 21
    # D
    use_m15_bos_filter:      bool  = False
    m15_bos_lookback:        int   = 30
    # E
    min_impulse_atr_mult:    float = 0.0
    # H
    use_sd_confluence:       bool  = False
    sd_zone_buffer_atr_mult: float = 0.30


# ── Variant grid ─────────────────────────────────────────────────────────────

VARIANTS: list[Cfg] = [
    # ── Baseline ─────────────────────────────────────────────────────────────
    Cfg("BASE · R18-B2 référence"),

    # ── Filtres individuels ───────────────────────────────────────────────────
    Cfg("A    · bougie bullish seul",
        require_bullish_bar=True),
    Cfg("B    · 50% FVG max seul",
        fvg_max_penetration_pct=0.50),
    Cfg("C    · H1 EMA21 seul",
        use_h1_trend=True),
    Cfg("D    · BOS M15 seul",
        use_m15_bos_filter=True),
    Cfg("E    · impulse 1.5x ATR seul",
        min_impulse_atr_mult=1.5),
    Cfg("F    · London 7h-13h seul",
        session_end_utc=13),
    Cfg("G    · R:R=1.5 seul",
        risk_reward=1.5),
    Cfg("H    · SD confluence seul",
        use_sd_confluence=True),

    # ── Cumul A→D ─────────────────────────────────────────────────────────────
    Cfg("AB   · bougie + 50%FVG",
        require_bullish_bar=True,
        fvg_max_penetration_pct=0.50),
    Cfg("ABC  · +H1 EMA21",
        require_bullish_bar=True,
        fvg_max_penetration_pct=0.50,
        use_h1_trend=True),
    Cfg("ABCD · +BOS M15",
        require_bullish_bar=True,
        fvg_max_penetration_pct=0.50,
        use_h1_trend=True,
        use_m15_bos_filter=True),

    # ── Ajout E & F sur ABCD ──────────────────────────────────────────────────
    Cfg("ABCDE  · +impulse 1.5x",
        require_bullish_bar=True,
        fvg_max_penetration_pct=0.50,
        use_h1_trend=True,
        use_m15_bos_filter=True,
        min_impulse_atr_mult=1.5),
    Cfg("ABCDF  · +London 7h-13h",
        require_bullish_bar=True,
        fvg_max_penetration_pct=0.50,
        use_h1_trend=True,
        use_m15_bos_filter=True,
        session_end_utc=13),
    Cfg("ABCDEF · +impulse +London",
        require_bullish_bar=True,
        fvg_max_penetration_pct=0.50,
        use_h1_trend=True,
        use_m15_bos_filter=True,
        min_impulse_atr_mult=1.5,
        session_end_utc=13),

    # ── Avec G (R:R=1.5) ──────────────────────────────────────────────────────
    Cfg("ABG   · AB + R:R=1.5",
        risk_reward=1.5,
        require_bullish_bar=True,
        fvg_max_penetration_pct=0.50),
    Cfg("ABCG  · ABC + R:R=1.5",
        risk_reward=1.5,
        require_bullish_bar=True,
        fvg_max_penetration_pct=0.50,
        use_h1_trend=True),
    Cfg("ABCDG · ABCD + R:R=1.5",
        risk_reward=1.5,
        require_bullish_bar=True,
        fvg_max_penetration_pct=0.50,
        use_h1_trend=True,
        use_m15_bos_filter=True),
    Cfg("ABCDEG· ABCDE + R:R=1.5",
        risk_reward=1.5,
        require_bullish_bar=True,
        fvg_max_penetration_pct=0.50,
        use_h1_trend=True,
        use_m15_bos_filter=True,
        min_impulse_atr_mult=1.5),

    # ── Filtre H (S&D confluence) ─────────────────────────────────────────────
    Cfg("ABH   · AB + SD",
        require_bullish_bar=True,
        fvg_max_penetration_pct=0.50,
        use_sd_confluence=True),
    Cfg("ABCH  · ABC + SD",
        require_bullish_bar=True,
        fvg_max_penetration_pct=0.50,
        use_h1_trend=True,
        use_sd_confluence=True),
    Cfg("ABCDH · ABCD + SD",
        require_bullish_bar=True,
        fvg_max_penetration_pct=0.50,
        use_h1_trend=True,
        use_m15_bos_filter=True,
        use_sd_confluence=True),

    # ── Combinaisons finales : tentative 70%+ WR ──────────────────────────────
    Cfg("FULL20 · tous filtres R:R=2.0",
        risk_reward=2.0,
        require_bullish_bar=True,
        fvg_max_penetration_pct=0.50,
        use_h1_trend=True,
        use_m15_bos_filter=True,
        min_impulse_atr_mult=1.5,
        session_end_utc=13,
        use_sd_confluence=True),
    Cfg("FULL15 · tous filtres R:R=1.5",
        risk_reward=1.5,
        require_bullish_bar=True,
        fvg_max_penetration_pct=0.50,
        use_h1_trend=True,
        use_m15_bos_filter=True,
        min_impulse_atr_mult=1.5,
        session_end_utc=13,
        use_sd_confluence=True),
]


def _build(cfg: Cfg) -> FVGRetestStrategy:
    return FVGRetestStrategy(
        risk_reward             = cfg.risk_reward,
        signal_cooldown         = cfg.signal_cooldown,
        max_signals_per_day     = cfg.max_signals_per_day,
        use_session_filter      = True,
        session_start_utc       = cfg.session_start_utc,
        session_end_utc         = cfg.session_end_utc,
        long_only               = True,
        use_h4_trend            = True,
        require_bullish_bar     = cfg.require_bullish_bar,
        fvg_max_penetration_pct = cfg.fvg_max_penetration_pct,
        use_h1_trend            = cfg.use_h1_trend,
        h1_ema_span             = cfg.h1_ema_span,
        use_m15_bos_filter      = cfg.use_m15_bos_filter,
        m15_bos_lookback        = cfg.m15_bos_lookback,
        min_impulse_atr_mult    = cfg.min_impulse_atr_mult,
        use_sd_confluence       = cfg.use_sd_confluence,
        sd_zone_buffer_atr_mult = cfg.sd_zone_buffer_atr_mult,
    )


def _run(cfg: Cfg, m15_df: pd.DataFrame) -> tuple[list, dict[str, Any], int]:
    strat   = _build(cfg)
    signals = strat.run(m15_df)
    if not signals:
        return [], {"win_rate": 0, "total_r": 0, "max_dd": 0,
                    "n_wins": 0, "n_losses": 0}, 0
    results, n_exp = simulate_all(
        signals, m15_df, risk_pct=RISK_PCT, spread=SPREAD_GOLD, max_monthly_losses=0,
    )
    m = compute_metrics(results, INITIAL_BALANCE, len(signals), n_exp)
    return results, m, len(signals)


def _line(label: str, m: dict, n_sig: int, n_months: float) -> None:
    spm = n_sig / n_months if n_months else 0
    ev  = m["total_r"] / n_sig if n_sig else 0
    print(
        f"  {label:<42}  "
        f"sig={n_sig:>3} ({spm:>4.1f}/m)  "
        f"W={m['n_wins']:>3} L={m['n_losses']:>3}  "
        f"WR={m['win_rate']:>5.1f}%  "
        f"R={m['total_r']:>+7.2f}  "
        f"EV={ev:>+6.3f}R  "
        f"DD={m['max_dd']:>5.1f}%"
    )


def _run_window(
    label:    str,
    m15_df:   pd.DataFrame,
    n_months: float,
) -> dict[str, dict]:
    print(f"\n{'═' * 105}")
    print(f"  {label}  ({len(m15_df):,} barres M15)")
    print(f"{'═' * 105}")

    rows: list[tuple] = []
    for cfg in VARIANTS:
        print(f"  [{cfg.label}] …", end="", flush=True)
        _, m, n_sig = _run(cfg, m15_df)
        rows.append((cfg, m, n_sig))
        print(f" {n_sig} sig  WR={m['win_rate']:.1f}%")

    print(f"\n{'─' * 105}")
    print("  RÉSULTATS DÉTAILLÉS")
    print(f"{'─' * 105}")
    for cfg, m, n_sig in rows:
        _line(cfg.label, m, n_sig, n_months)

    # Targets
    target_70 = [(cfg.label, m, n_sig) for cfg, m, n_sig in rows
                 if m["win_rate"] >= 70 and n_sig >= 3]
    target_55 = [(cfg.label, m, n_sig) for cfg, m, n_sig in rows
                 if m["win_rate"] >= 55 and n_sig >= 5]

    if target_70:
        print(f"\n  🎯 ≥70% WR (≥3 sig) :")
        for lb, m, n in target_70:
            print(f"     {lb} → WR={m['win_rate']:.1f}% R={m['total_r']:+.2f} "
                  f"sig={n} DD={m['max_dd']:.1f}%")
    elif target_55:
        print(f"\n  ✓ ≥55% WR (≥5 sig) [objectif intermédiaire] :")
        for lb, m, n in target_55:
            print(f"     {lb} → WR={m['win_rate']:.1f}% R={m['total_r']:+.2f} "
                  f"sig={n} DD={m['max_dd']:.1f}%")
    else:
        best = max(rows, key=lambda x: x[1]["win_rate"] if x[2] >= 3 else 0)
        print(f"\n  Meilleur WR (≥3 sig) : {best[0].label} "
              f"WR={best[1]['win_rate']:.1f}% sig={best[2]}")

    return {cfg.label: (m, n_sig) for cfg, m, n_sig in rows}


def main() -> None:
    print("=" * 105)
    print("  FVG Round 19 — Audit filtres A–H  |  XAUUSD 2025")
    print("  Objectif : WR 70–80%  |  Exploration Q1 · Validation Q4")
    print("=" * 105)

    m1_full = parse_histdata_csv(M1_2025)
    m1_full = m1_full[~m1_full.index.duplicated(keep="first")]
    print(f"\n  Données 2025 complètes : {len(m1_full):,} barres M1")

    # Q1 2025 (jan–mar) = exploration
    m1_q1  = m1_full.loc["2025-01-01":"2025-03-31"]
    m15_q1 = resample_ohlcv(m1_q1, "15min")
    print(f"  Q1 2025 : {len(m1_q1):,} M1 → {len(m15_q1):,} M15  (~3 mois)")

    # Q4 2025 (oct–déc) = validation out-of-sample
    m1_q4  = m1_full.loc["2025-10-01":"2025-12-31"]
    m15_q4 = resample_ohlcv(m1_q4, "15min")
    print(f"  Q4 2025 : {len(m1_q4):,} M1 → {len(m15_q4):,} M15  (~3 mois)")

    res_q1 = _run_window("XAUUSD Q1 2025 — EXPLORATION  (jan–mar)", m15_q1, 3.0)
    res_q4 = _run_window("XAUUSD Q4 2025 — VALIDATION   (oct–déc)", m15_q4, 3.0)

    # ── Tableau croisé ──────────────────────────────────────────────────────
    print(f"\n{'═' * 105}")
    print("  COMPARAISON Q1 vs Q4 2025")
    print(f"{'─' * 105}")
    print(f"  {'Variante':<42}  {'── Q1 2025 (explore) ──':>35}  {'── Q4 2025 (valid) ──':>33}")
    print(f"  {'':42}  {'sig':>3}  {'WR%':>6}  {'R':>7}  {'EV':>7}  {'DD%':>5}  "
          f"{'sig':>3}  {'WR%':>6}  {'R':>7}  {'EV':>7}  {'DD%':>5}")
    print(f"  {'─' * 103}")

    for cfg in VARIANTS:
        m1, n1 = res_q1.get(cfg.label, ({}, 0))
        m4, n4 = res_q4.get(cfg.label, ({}, 0))
        if not m1 or not m4:
            continue
        ev1 = m1["total_r"] / n1 if n1 else 0
        ev4 = m4["total_r"] / n4 if n4 else 0
        flag = " ★" if m1["win_rate"] >= 65 and m4["win_rate"] >= 65 else \
               " ✓" if m1["win_rate"] >= 55 and m4["win_rate"] >= 55 else ""
        print(
            f"  {cfg.label:<42}  "
            f"{n1:>3}  {m1['win_rate']:>6.1f}%  {m1['total_r']:>+7.2f}  "
            f"{ev1:>+6.3f}R  {m1['max_dd']:>4.1f}%  "
            f"{n4:>3}  {m4['win_rate']:>6.1f}%  {m4['total_r']:>+7.2f}  "
            f"{ev4:>+6.3f}R  {m4['max_dd']:>4.1f}%"
            f"{flag}"
        )

    print(f"\n{'═' * 105}")
    print("  Fin Round 19")
    print(f"{'=' * 105}\n")


if __name__ == "__main__":
    main()
