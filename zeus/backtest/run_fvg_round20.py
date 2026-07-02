"""
FVG Round 20 — Validation ABCDG sur 2024 complet + tuning paramètres.
=======================================================================

Contexte
--------
Round 19 a identifié ABCDG (A=bougie bullish, B=50% pénétration FVG,
C=H1 EMA21, D=BOS M15, G=R:R=1.5) comme la seule configuration atteignant
70% WR (Q4 2025 bull market).

Objectifs de Round 20
---------------------
1. Validation sur 2024 complet — 4 trimestres consécutifs — pour évaluer la
   robustesse hors période haussière 2025.
2. Tuning fine des paramètres ABCDG :
   - m15_bos_lookback   (15 / 20 / 30 / 45)
   - h1_ema_span        (9 / 21 / 50)
   - fvg_max_penetration (0.30 / 0.50 / 0.70)
   - risk_reward        (1.0 / 1.5 / 2.0)
3. Identifier la variante ABCDG la plus stable cross-régimes.

Usage
-----
    python -m zeus.backtest.run_fvg_round20
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
    label:                   str
    risk_reward:             float = 1.5
    signal_cooldown:         int   = 12      # 3 h
    max_signals_per_day:     int   = 3
    session_start_utc:       int   = 7
    session_end_utc:         int   = 21
    require_bullish_bar:     bool  = True    # A toujours ON
    fvg_max_penetration_pct: float = 0.50   # B toujours ON
    use_h1_trend:            bool  = True   # C toujours ON
    h1_ema_span:             int   = 21
    use_m15_bos_filter:      bool  = True   # D toujours ON
    m15_bos_lookback:        int   = 30


# ── Variantes ────────────────────────────────────────────────────────────────

VARIANTS: list[Cfg] = [
    # ── Référence R19 champion ────────────────────────────────────────────────
    Cfg("ABCDG-REF  · R19 champion"),

    # ── Tuning bos_lookback ──────────────────────────────────────────────────
    Cfg("ABCDG-BOS15  · lookback=15",   m15_bos_lookback=15),
    Cfg("ABCDG-BOS20  · lookback=20",   m15_bos_lookback=20),
    Cfg("ABCDG-BOS45  · lookback=45",   m15_bos_lookback=45),
    Cfg("ABCDG-BOS60  · lookback=60",   m15_bos_lookback=60),

    # ── Tuning h1_ema_span ───────────────────────────────────────────────────
    Cfg("ABCDG-EMA9   · H1 EMA9",       h1_ema_span=9),
    Cfg("ABCDG-EMA50  · H1 EMA50",      h1_ema_span=50),
    Cfg("ABCDG-EMA100 · H1 EMA100",     h1_ema_span=100),

    # ── Tuning pénétration FVG ───────────────────────────────────────────────
    Cfg("ABCDG-PEN30  · 30% max",       fvg_max_penetration_pct=0.30),
    Cfg("ABCDG-PEN70  · 70% max",       fvg_max_penetration_pct=0.70),

    # ── Tuning R:R ───────────────────────────────────────────────────────────
    Cfg("ABCDG-RR10   · R:R=1.0",       risk_reward=1.0),
    Cfg("ABCDG-RR20   · R:R=2.0",       risk_reward=2.0),
    Cfg("ABCDG-RR30   · R:R=3.0",       risk_reward=3.0),

    # ── Combos prometteurs ───────────────────────────────────────────────────
    Cfg("ABCDG-OPT1   · BOS20+EMA50",
        m15_bos_lookback=20, h1_ema_span=50),
    Cfg("ABCDG-OPT2   · BOS45+EMA9",
        m15_bos_lookback=45, h1_ema_span=9),
    Cfg("ABCDG-OPT3   · BOS20+PEN30+RR15",
        m15_bos_lookback=20, fvg_max_penetration_pct=0.30),
    Cfg("ABCDG-OPT4   · BOS45+EMA50+PEN30",
        m15_bos_lookback=45, h1_ema_span=50, fvg_max_penetration_pct=0.30),

    # ── Sans filtre C (test si C aide en consolidation) ──────────────────────
    Cfg("ABD G-NOEMA  · sans H1 EMA",
        use_h1_trend=False),

    # ── Sans filtre D (test si D seul cause Q1 perte) ────────────────────────
    Cfg("ABC G-NOBOS  · sans BOS M15",
        use_m15_bos_filter=False),
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


def _line(label: str, m: dict, n_sig: int, n_months: float) -> str:
    spm = n_sig / n_months if n_months else 0
    ev  = m["total_r"] / n_sig if n_sig else 0
    return (
        f"  {label:<44}  "
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
) -> dict[str, tuple[dict, int]]:
    print(f"\n{'═' * 110}")
    print(f"  {label}  ({len(m15_df):,} barres M15)")
    print(f"{'═' * 110}")

    rows: list[tuple[Cfg, dict, int]] = []
    for cfg in VARIANTS:
        print(f"  [{cfg.label}] …", end="", flush=True)
        _, m, n_sig = _run(cfg, m15_df)
        rows.append((cfg, m, n_sig))
        print(f" {n_sig} sig  WR={m['win_rate']:.1f}%")

    print(f"\n{'─' * 110}")
    print("  RÉSULTATS DÉTAILLÉS")
    print(f"{'─' * 110}")
    for cfg, m, n_sig in rows:
        print(_line(cfg.label, m, n_sig, n_months))

    target_70 = [(cfg, m, n) for cfg, m, n in rows if m["win_rate"] >= 70 and n >= 3]
    target_60 = [(cfg, m, n) for cfg, m, n in rows if m["win_rate"] >= 60 and n >= 3]

    if target_70:
        print(f"\n  🎯 ≥70% WR (≥3 sig) :")
        for c, m, n in target_70:
            ev = m["total_r"] / n if n else 0
            print(f"     {c.label} → WR={m['win_rate']:.1f}% "
                  f"R={m['total_r']:+.2f} EV={ev:+.3f}R sig={n} DD={m['max_dd']:.1f}%")
    elif target_60:
        print(f"\n  ✓ ≥60% WR (≥3 sig) :")
        for c, m, n in target_60:
            ev = m["total_r"] / n if n else 0
            print(f"     {c.label} → WR={m['win_rate']:.1f}% "
                  f"R={m['total_r']:+.2f} EV={ev:+.3f}R sig={n} DD={m['max_dd']:.1f}%")
    else:
        valid = [(c, m, n) for c, m, n in rows if n >= 3]
        if valid:
            best = max(valid, key=lambda x: x[1]["win_rate"])
            ev = best[1]["total_r"] / best[2] if best[2] else 0
            print(f"\n  Meilleur WR (≥3 sig) : {best[0].label} "
                  f"WR={best[1]['win_rate']:.1f}% sig={best[2]} EV={ev:+.3f}R")

    return {cfg.label: (m, n_sig) for cfg, m, n_sig in rows}


def _print_cross_table(windows: list[tuple[str, dict[str, tuple[dict, int]]]]) -> None:
    """Tableau croisé : variante × trimestre."""
    print(f"\n{'═' * 130}")
    print("  TABLEAU CROISÉ — ABCDG variants × 2024 trimestres + Q1/Q4 2025")
    print(f"{'─' * 130}")

    headers = [lbl for lbl, _ in windows]
    # En-tête colonnes
    col_w = 9
    header_row = f"  {'Variante':<44}  "
    for lbl in headers:
        # Abrégé sur 8 chars
        short = lbl[:8]
        header_row += f"{short:>{col_w}} "
    print(header_row)

    subrow = f"  {'':44}  "
    for _ in headers:
        subrow += f"{'WR%':>{col_w}} "
    print(subrow)
    print(f"  {'─' * 105}")

    for cfg in VARIANTS:
        row = f"  {cfg.label:<44}  "
        wrs = []
        for _, res in windows:
            m, n = res.get(cfg.label, ({}, 0))
            if not m or n < 3:
                row += f"{'—':>{col_w}} "
                wrs.append(None)
            else:
                wr = m.get("win_rate", 0)
                row += f"{wr:>{col_w}.1f}% "
                wrs.append(wr)

        valid = [w for w in wrs if w is not None]
        if valid:
            avg = sum(valid) / len(valid)
            mn  = min(valid)
            row += f"  avg={avg:.1f}% min={mn:.1f}%"
        print(row)


def main() -> None:
    print("=" * 110)
    print("  FVG Round 20 — Validation ABCDG sur 2024 complet + tuning paramètres")
    print("  Objectif : WR 70% stable cross-régimes  |  2024 Q1→Q4 + 2025 Q1/Q4")
    print("=" * 110)

    # ── Données 2024 ─────────────────────────────────────────────────────────
    m1_2024 = parse_histdata_csv(M1_2024)
    m1_2024 = m1_2024[~m1_2024.index.duplicated(keep="first")]
    print(f"\n  2024 complet : {len(m1_2024):,} barres M1")

    slices_2024 = {
        "Q1 2024 (jan-mar)": ("2024-01-01", "2024-03-31"),
        "Q2 2024 (avr-jun)": ("2024-04-01", "2024-06-30"),
        "Q3 2024 (jul-sep)": ("2024-07-01", "2024-09-30"),
        "Q4 2024 (oct-déc)": ("2024-10-01", "2024-12-31"),
    }

    # ── Données 2025 ─────────────────────────────────────────────────────────
    m1_2025 = parse_histdata_csv(M1_2025)
    m1_2025 = m1_2025[~m1_2025.index.duplicated(keep="first")]
    print(f"  2025 complet : {len(m1_2025):,} barres M1")

    slices_2025 = {
        "Q1 2025 (jan-mar)": ("2025-01-01", "2025-03-31"),
        "Q4 2025 (oct-déc)": ("2025-10-01", "2025-12-31"),
    }

    # ── Run toutes les fenêtres ───────────────────────────────────────────────
    all_windows: list[tuple[str, dict]] = []

    for name, (start, end) in slices_2024.items():
        m1_sl  = m1_2024.loc[start:end]
        m15_sl = resample_ohlcv(m1_sl, "15min")
        print(f"  {name} : {len(m1_sl):,} M1 → {len(m15_sl):,} M15")
        res = _run_window(f"XAUUSD {name}", m15_sl, 3.0)
        all_windows.append((name, res))

    for name, (start, end) in slices_2025.items():
        m1_sl  = m1_2025.loc[start:end]
        m15_sl = resample_ohlcv(m1_sl, "15min")
        print(f"  {name} : {len(m1_sl):,} M1 → {len(m15_sl):,} M15")
        res = _run_window(f"XAUUSD {name}", m15_sl, 3.0)
        all_windows.append((name, res))

    _print_cross_table(all_windows)

    print(f"\n{'═' * 110}")
    print("  Fin Round 20")
    print(f"{'=' * 110}\n")


if __name__ == "__main__":
    main()
