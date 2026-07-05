"""
S&D Strategy — Round 24 · XAUUSD · Tick-Optimized Entry & SL
==============================================================

Ce script reprend le champion du Round 24 (4T TP3@8R Run@20R) et le compare
en trois modes :

  Mode A — Référence M1  : entrée à l'open de la barre M1+1, SL M1-derivé
  Mode B — Tick Entry    : entrée au tick exact qui franchit mss_close, SL M1
  Mode C — Tick Entry+SL : entrée tick + SL filtré au percentile 5 des ticks bid

Utilisation
-----------
    # 1. Place tes fichiers ticks dans :
    #    data/ticks/xauusd/<année>/  (CSV exportés depuis MT5)
    #
    # 2. Lance :
    python -m zeus.backtest.run_sd_round24_ticks

Arguments attendus
------------------
Les fichiers ticks doivent se trouver dans :
  _TICK_DIR / "DAT_MT_XAUUSD_TICKS_<YYYY>.csv"   (pattern configurable)
ou dans des sous-répertoires par année.

Le script affiche un avertissement si les données tick sont manquantes pour
une période et utilise alors le Mode A (fallback M1) pour cette période.
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from zeus.backtest.data_loader    import parse_histdata_csv, resample_ohlcv
from zeus.backtest.tick_loader    import load_tick_directory, parse_mt5_ticks, resample_ticks
from zeus.backtest.tick_optimizer import optimize_signals_with_ticks
from zeus.backtest.sd_simulation  import TradeResult, compute_metrics, simulate_all
from zeus.strategy.supply_demand.sd_strategy  import SDStrategy
from zeus.strategy.supply_demand.wyckoff      import WyckoffDetector
from zeus.strategy.supply_demand.zone_detector import ZoneDetector

logging.basicConfig(level=logging.WARNING)

# ── Chemins ──────────────────────────────────────────────────────────────────

_ROOT     = Path(__file__).parent.parent.parent
_M1       = _ROOT / "data" / "historical" / "xauusd" / "m1"
_TICK_DIR = _ROOT / "data" / "ticks" / "xauusd"

# ── Paramètres stratégie (identiques Round 24) ───────────────────────────────

INITIAL_BALANCE = 10_000.0
RISK_PCT        = 0.01
SPREAD          = 0.30   # USD/oz — utilisé en mode M1
SPREAD_TICK     = 0.0    # spread déjà dans le prix tick

_V96 = dict(
    risk_reward             = 1.5,
    min_zone_score          = 5.0,
    min_wyckoff_score       = 5.9,
    min_composite_score     = 5.0,
    signal_cooldown         = 10,
    use_trend_filter        = True,
    trend_slope_lookback    = 3,
    use_price_above_ema     = True,
    use_session_filter      = True,
    session_start_utc       = 7,
    session_end_utc         = 21,
    max_signals_per_day     = 6,
    use_adx_filter          = False,
    use_h4_trend_filter     = False,
    use_rsi_filter          = False,
    min_wyckoff_score_short = 99.0,
)

_WY = dict(
    lookback             = 200,
    max_accum_bars       = 20,
    accum_range_mult     = 6.0,
    mss_lookback         = 60,
    min_spring_sweep_pct = 0.05,
    min_mss_strength_pct = 0.03,
)

# Champion Round 24 : 4T TP3@8R Runner@20R
_4T_BEST = dict(
    tp1_r              = 1.0,
    tp1_size           = 0.0,
    tp2_r              = 3.0,
    tp2_cumulative_pct = 0.60,
    tp3_r              = 8.0,
    tp3_cumulative_pct = 0.85,
    max_monthly_losses = 4,
    max_daily_losses   = 1,
)
RUNNER_RR = 20.0

# Optimiseur ticks
_OPT = dict(
    refine_entry     = True,
    refine_sl        = True,
    sl_percentile    = 5.0,
    sl_buffer_usd    = 0.10,
    sl_lookback_m1   = 3,
    max_entry_wait_s = 300,
)

# Résolution OHLCV pour le scan SL/TP quand les ticks sont disponibles
TICK_RESAMPLE_FREQ = "1s"

# ── Périodes de test ─────────────────────────────────────────────────────────

PERIODS = [
    ("2024 Full Year (IS-A)", [_M1 / "DAT_MT_XAUUSD_M1_2024.csv"], 2024),
    ("2025 Full Year (IS-B)", [_M1 / "DAT_MT_XAUUSD_M1_2025.csv"], 2025),
    ("2026 Jan–Mar  (OOS)  ", [
        _M1 / "DAT_MT_XAUUSD_M1_202601.csv",
        _M1 / "DAT_MT_XAUUSD_M1_202602.csv",
        _M1 / "DAT_MT_XAUUSD_M1_202603.csv",
    ], 2026),
    ("2026 Apr–Jun  (IS)   ", [
        _M1 / "DAT_MT_XAUUSD_M1_202604.csv",
        _M1 / "DAT_MT_XAUUSD_M1_202605.csv",
        _M1 / "DAT_MT_XAUUSD_M1_202606.csv",
    ], 2026),
]


# ── Chargement données ────────────────────────────────────────────────────────

def _load_m1(files: list[Path]) -> pd.DataFrame:
    frames = [parse_histdata_csv(f) for f in files if f.exists()]
    if not frames:
        raise FileNotFoundError(f"Données M1 manquantes : {files}")
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="first")]


def _load_ticks_for_year(year: int) -> pd.DataFrame | None:
    """
    Cherche les fichiers ticks pour l'année donnée dans _TICK_DIR.
    Supporte plusieurs structures de répertoire :
      - _TICK_DIR / XAUUSD_<YEAR>_ticks.csv
      - _TICK_DIR / <YEAR> / *.csv   (tous les fichiers du sous-répertoire)
      - _TICK_DIR / DAT_MT_XAUUSD_TICKS_<YEAR>.csv
    Retourne None si aucun fichier trouvé.
    """
    candidates = [
        _TICK_DIR / f"XAUUSD_{year}_ticks.csv",
        _TICK_DIR / f"XAUUSD_{year}.csv",
        _TICK_DIR / f"DAT_MT_XAUUSD_TICKS_{year}.csv",
        _TICK_DIR / f"{year}",             # sous-répertoire
    ]

    for c in candidates:
        if c.is_file():
            try:
                return parse_mt5_ticks(c, synthetic_spread=SPREAD)
            except Exception as e:
                print(f"  ⚠ Erreur lecture ticks {c}: {e}")
                return None
        if c.is_dir():
            try:
                return load_tick_directory(c, synthetic_spread=SPREAD)
            except Exception as e:
                print(f"  ⚠ Erreur lecture ticks {c}: {e}")
                return None

    return None


# ── Stratégie & simulation ────────────────────────────────────────────────────

def _build_strategy() -> SDStrategy:
    return SDStrategy(
        zone_detector    = ZoneDetector(),
        wyckoff_detector = WyckoffDetector(**_WY),
        risk_reward      = RUNNER_RR,
        **{k: v for k, v in _V96.items() if k != "risk_reward"},
    )


def _run_period(
    m1_df:    pd.DataFrame,
    tick_df:  pd.DataFrame | None,
    mode:     str,
) -> tuple[dict, int]:
    """
    Lance un backtest sur une période.

    mode : "m1"         — référence M1, pas de tick
           "tick_entry" — entrée tick seulement
           "tick_full"  — entrée tick + SL tick
    """
    zone_df  = resample_ohlcv(m1_df, "15min")
    strategy = _build_strategy()
    signals  = strategy.run(zone_df, m1_df)
    signals  = [dataclasses.replace(s, risk_reward=RUNNER_RR) for s in signals]

    sim_tick_df = None
    spread      = SPREAD

    if mode != "m1" and tick_df is not None:
        # Raffinement tick des signaux
        refine_sl  = (mode == "tick_full")
        signals    = optimize_signals_with_ticks(
            signals, tick_df, m1_df,
            refine_entry   = True,
            refine_sl      = refine_sl,
            **{k: v for k, v in _OPT.items() if k not in ("refine_entry", "refine_sl")},
        )
        # OHLCV 1s pour scan SL/TP (meilleure résolution)
        sim_tick_df = resample_ticks(tick_df, TICK_RESAMPLE_FREQ)
        spread      = SPREAD_TICK   # spread déjà dans le prix tick

    use_signal_entry = (mode != "m1" and tick_df is not None)

    results, n_exp = simulate_all(
        signals, m1_df,
        risk_pct         = RISK_PCT,
        spread           = spread,
        use_signal_entry = use_signal_entry,
        tick_df          = sim_tick_df,
        **_4T_BEST,
    )
    m = compute_metrics(results, INITIAL_BALANCE, len(signals), n_exp)
    return m, len(signals)


# ── Affichage ─────────────────────────────────────────────────────────────────

def _print_table(title: str, rows: list[tuple[str, dict, int]]) -> tuple[float, float, float, float]:
    print(f"\n  {title}")
    print(f"  {'Period':<30}  {'Sig':>4}  {'W':>3} {'L':>3}  "
          f"{'WR%':>5}  {'R':>7}  {'P&L$':>8}  {'DD%':>5}")
    print("  " + "─" * 76)
    tw = tl = tsig = tr = 0.0
    min_r  = float("inf")
    max_dd = 0.0
    all_pos = True
    for lbl, m, n in rows:
        print(
            f"  {lbl:<30}  {n:>4}  "
            f"{m['n_wins']:>3} {m['n_losses']:>3}  "
            f"{m['win_rate']:>5.1f}%  "
            f"{m['total_r']:>+7.2f}  "
            f"{m['total_usd']:>+8.0f}$  "
            f"{m['max_dd']:>4.1f}%"
        )
        tw   += m["n_wins"];  tl   += m["n_losses"]
        tsig += n;            tr   += m["total_r"]
        if n > 0:
            min_r  = min(min_r, m["total_r"])
            max_dd = max(max_dd, m["max_dd"])
            if m["total_r"] <= 0:
                all_pos = False
    decided = tw + tl
    wr = tw / decided * 100 if decided else 0.0
    if min_r == float("inf"):
        min_r = 0.0
    print("  " + "─" * 76)
    print(f"  {'TOTAL / COMBINED':<30}  {int(tsig):>4}  "
          f"{int(tw):>3} {int(tl):>3}  {wr:>5.1f}%  {tr:>+7.2f}")
    verdict = "PASS ✓" if wr >= 60 and all_pos else "FAIL ✗"
    print(f"  VERDICT: {verdict}")
    return tr, min_r, wr, max_dd


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 100)
    print("  S&D Round 24 — XAUUSD · Comparaison M1 vs Ticks")
    print("  Champion : 4T · TP1@1R(BE) · TP2@3R(60%) · TP3@8R(85%) · Runner@20R")
    print("=" * 100)

    MODES = [
        ("m1",         "Mode A — Référence M1         "),
        ("tick_entry", "Mode B — Tick Entry (SL M1)   "),
        ("tick_full",  "Mode C — Tick Entry + Tick SL "),
    ]

    summary: list[tuple[str, float, float, float, float]] = []

    for mode, mode_label in MODES:
        print(f"\n{'─' * 80}")
        print(f"  {mode_label.strip()}")
        print(f"{'─' * 80}")

        rows = []
        for period_label, m1_files, year in PERIODS:
            m1_df    = _load_m1(m1_files)
            tick_df  = _load_ticks_for_year(year) if mode != "m1" else None

            if mode != "m1" and tick_df is None:
                print(f"  ⚠ Ticks manquants pour {year} — fallback M1 pour cette période")

            m, n_sig = _run_period(m1_df, tick_df, mode if tick_df is not None else "m1")
            rows.append((period_label, m, n_sig))

        tr, mr, wr, dd = _print_table(mode_label, rows)
        summary.append((mode_label, tr, mr, wr, dd))

    # ── Comparaison finale ────────────────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print("  COMPARAISON FINALE — Impact des données tick")
    print(f"{'─' * 100}")
    print(f"\n  {'Mode':<38}  {'TotR':>7}  {'MinR':>7}  {'WR':>5}  {'MaxDD':>6}  {'vs M1':>8}")
    print("  " + "─" * 75)

    base_r = summary[0][1] if summary else 0.0
    for name, tr, mr, wr, dd in summary:
        delta = f"{tr - base_r:>+7.2f}" if tr != base_r else "   ref"
        ok    = "←" if mr > 0 and wr >= 60 else ""
        print(f"  {name:<38}  {tr:>+7.2f}  {mr:>+7.2f}  {wr:>5.1f}%  {dd:>5.1f}%  {delta}  {ok}")

    print(f"\n{'=' * 100}")
    print("  Fin")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    main()
