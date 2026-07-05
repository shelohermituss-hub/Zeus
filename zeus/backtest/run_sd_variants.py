"""
S&D Strategy — Comparaison de variantes · XAUUSD · Mode A (M1 référence)
=========================================================================

Variantes testées
-----------------
  V0 — Baseline    : WS_long≥5.9  · session 07–21h · long-only
  V1 — WS7.5       : WS_long≥7.5  · session 07–21h · long-only
  V2 — Session17   : WS_long≥5.9  · session 07–17h · long-only
  V3 — WS7.5+Ses17 : WS_long≥7.5  · session 07–17h · long-only

  Gold est toujours long-only (min_wyckoff_score_short=99.0).
  "Short WS≥8.0" = future variante pour paires non-gold.

Résultats affichés par : Variante × Année × Paire (XAUUSD)
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd

from zeus.backtest.data_loader    import parse_histdata_csv, resample_ohlcv
from zeus.backtest.sd_simulation  import compute_metrics, simulate_all
from zeus.strategy.supply_demand.sd_strategy   import SDStrategy
from zeus.strategy.supply_demand.wyckoff       import WyckoffDetector
from zeus.strategy.supply_demand.zone_detector import ZoneDetector

# ── Chemins ───────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parent.parent.parent
_M1   = _ROOT / "data" / "historical" / "xauusd" / "m1"

# ── Paramètres fixes ──────────────────────────────────────────────────────────

INITIAL_BALANCE = 10_000.0
RISK_PCT        = 0.01
SPREAD          = 0.30
RUNNER_RR       = 20.0

_WY = dict(
    lookback             = 200,
    max_accum_bars       = 20,
    accum_range_mult     = 6.0,
    mss_lookback         = 60,
    min_spring_sweep_pct = 0.05,
    min_mss_strength_pct = 0.03,
)

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

# Paramètres communs à toutes les variantes
_BASE = dict(
    min_zone_score          = 5.0,
    min_composite_score     = 5.0,
    signal_cooldown         = 10,
    use_trend_filter        = True,
    trend_slope_lookback    = 3,
    use_price_above_ema     = True,
    use_session_filter      = True,
    max_signals_per_day     = 6,
    use_adx_filter          = False,
    use_h4_trend_filter     = False,
    use_rsi_filter          = False,
    min_wyckoff_score_short = 99.0,   # long-only sur or
)

# ── Définition des variantes ──────────────────────────────────────────────────

VARIANTS: dict[str, dict] = {
    "V0 — Baseline    (WS≥5.9  · 07–21h)": dict(
        min_wyckoff_score = 5.9,
        session_start_utc = 7,
        session_end_utc   = 21,
    ),
    "V1 — WS7.5       (WS≥7.5  · 07–21h)": dict(
        min_wyckoff_score = 7.5,
        session_start_utc = 7,
        session_end_utc   = 21,
    ),
    "V2 — Session17   (WS≥5.9  · 07–17h)": dict(
        min_wyckoff_score = 5.9,
        session_start_utc = 7,
        session_end_utc   = 17,
    ),
    "V3 — WS7.5+Ses17 (WS≥7.5  · 07–17h)": dict(
        min_wyckoff_score = 7.5,
        session_start_utc = 7,
        session_end_utc   = 17,
    ),
}

# ── Données ───────────────────────────────────────────────────────────────────

PERIODS = [
    ("2024", [_M1 / "DAT_MT_XAUUSD_M1_2024.csv"]),
    ("2025", [_M1 / "DAT_MT_XAUUSD_M1_2025.csv"]),
    ("2026 Jan–Mar (OOS)", [
        _M1 / "DAT_MT_XAUUSD_M1_202601.csv",
        _M1 / "DAT_MT_XAUUSD_M1_202602.csv",
        _M1 / "DAT_MT_XAUUSD_M1_202603.csv",
    ]),
    ("2026 Apr–Jun", [
        _M1 / "DAT_MT_XAUUSD_M1_202604.csv",
        _M1 / "DAT_MT_XAUUSD_M1_202605.csv",
        _M1 / "DAT_MT_XAUUSD_M1_202606.csv",
    ]),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_m1(files: list[Path]) -> pd.DataFrame:
    frames = [parse_histdata_csv(f) for f in files if f.exists()]
    if not frames:
        raise FileNotFoundError(f"M1 data missing: {files}")
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="first")]


def _build_strategy(variant_params: dict) -> SDStrategy:
    params = {**_BASE, **variant_params}
    return SDStrategy(
        zone_detector    = ZoneDetector(),
        wyckoff_detector = WyckoffDetector(**_WY),
        risk_reward      = RUNNER_RR,
        **{k: v for k, v in params.items() if k != "risk_reward"},
    )


def _run(m1_df: pd.DataFrame, variant_params: dict) -> tuple[dict, int]:
    """Returns (metrics, n_signals)."""
    zone_df  = resample_ohlcv(m1_df, "15min")
    strategy = _build_strategy(variant_params)
    signals  = strategy.run(zone_df, m1_df)
    n_sig    = len(signals)
    signals  = [dataclasses.replace(s, risk_reward=RUNNER_RR) for s in signals]
    results, n_exp = simulate_all(
        signals, m1_df,
        risk_pct = RISK_PCT,
        spread   = SPREAD,
        **_4T_BEST,
    )
    m = compute_metrics(results, INITIAL_BALANCE, n_sig, n_exp)
    m["n_signals"] = n_sig
    return m, n_sig


# ── Affichage ─────────────────────────────────────────────────────────────────

_SEP  = "─" * 102
_SEP2 = "═" * 102


def _print_variant_block(
    variant_name: str,
    rows: list[tuple[str, dict]],
) -> dict:
    """Print one variant's results per period. Returns totals dict."""
    print(f"\n  ┌{'─' * 98}┐")
    print(f"  │  {variant_name:<95} │")
    print(f"  └{'─' * 98}┘")
    print(f"  {'Paire':<8}  {'Année / Période':<24}  "
          f"{'Sig':>4}  {'W':>3} {'L':>3}  {'WR%':>5}  {'R':>7}  {'P&L$':>9}  {'DD%':>5}  {'AvgR':>6}")
    print("  " + _SEP)

    tw = tl = tr = 0
    max_dd = 0.0
    for period_label, m in rows:
        n      = m["n_trades"] + m.get("n_expired", 0)   # approximation
        n_sig  = m.get("n_signals", 0)
        w, l   = m["n_wins"], m["n_losses"]
        wr     = m["win_rate"]
        tot_r  = m["total_r"]
        pnl    = m["total_usd"]
        dd     = m["max_dd"]
        avg_r  = tot_r / (w + l) if (w + l) else 0.0
        print(
            f"  {'XAUUSD':<8}  {period_label:<24}  "
            f"{n_sig:>4}  {w:>3} {l:>3}  {wr:>5.1f}%  "
            f"{tot_r:>+7.2f}  {pnl:>+9.0f}$  {dd:>5.1f}%  {avg_r:>+6.2f}"
        )
        tw += w; tl += l; tr += tot_r
        max_dd = max(max_dd, dd)

    decided = tw + tl
    wr_tot  = tw / decided * 100 if decided else 0.0
    print("  " + _SEP)
    print(f"  {'':8}  {'TOTAL':24}  {'':4}  {tw:>3} {tl:>3}  "
          f"{wr_tot:>5.1f}%  {tr:>+7.2f}  {'':9}  {max_dd:>5.1f}%")
    verdict = "PASS ✓" if wr_tot >= 60 and all(
        m["total_r"] >= 0 for _, m in rows
    ) else "FAIL ✗"
    print(f"  VERDICT : {verdict}")

    return dict(tw=tw, tl=tl, wr=wr_tot, tr=tr, max_dd=max_dd)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(_SEP2)
    print("  S&D Strategy — XAUUSD · Comparaison variantes (Mode A — M1 référence)")
    print("  Paire : XAUUSD  |  Long-only  |  4T : TP1@1R(BE) TP2@3R(60%) TP3@8R(85%) Runner@20R")
    print(_SEP2)

    # Pré-chargement M1 (une seule fois pour toutes les variantes)
    print("\n  Chargement des données M1…")
    period_data: list[tuple[str, pd.DataFrame]] = []
    for period_label, files in PERIODS:
        try:
            df = _load_m1(files)
            period_data.append((period_label, df))
            print(f"    ✓  {period_label}  ({len(df):,} barres M1)")
        except FileNotFoundError as e:
            print(f"    ✗  {period_label}  MANQUANT — {e}")

    if not period_data:
        print("  Aucune donnée disponible.")
        return

    # Stocke tous les résultats pour le tableau récapitulatif
    summary: list[tuple[str, dict]] = []

    for variant_name, variant_params in VARIANTS.items():
        rows: list[tuple[str, dict]] = []
        for period_label, m1_df in period_data:
            m, _ = _run(m1_df, variant_params)
            rows.append((period_label, m))

        totals = _print_variant_block(variant_name, rows)
        summary.append((variant_name, totals))

    # ── Tableau récapitulatif final ───────────────────────────────────────────
    print(f"\n{_SEP2}")
    print("  RÉCAPITULATIF GLOBAL — Classement par Total R")
    print(f"{'─' * 102}")
    print(f"\n  {'Variante':<42}  {'W':>3} {'L':>3}  {'WR%':>5}  {'TotR':>7}  {'MaxDD':>6}  {'vs V0':>8}")
    print("  " + "─" * 80)

    base_r = summary[0][1]["tr"] if summary else 0.0
    ranked = sorted(summary, key=lambda x: x[1]["tr"], reverse=True)
    for name, t in ranked:
        delta = f"{t['tr'] - base_r:>+7.2f}" if abs(t['tr'] - base_r) > 0.001 else "   ref"
        ok    = " ← MEILLEUR" if t["tr"] == max(x["tr"] for _, x in summary) else ""
        print(
            f"  {name:<42}  {t['tw']:>3} {t['tl']:>3}  "
            f"{t['wr']:>5.1f}%  {t['tr']:>+7.2f}  {t['max_dd']:>5.1f}%  {delta}{ok}"
        )

    print(f"\n{_SEP2}")
    print("  Fin")
    print(f"{_SEP2}\n")


if __name__ == "__main__":
    main()
