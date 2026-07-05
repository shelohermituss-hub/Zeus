"""
S&D V4 — Or vs autres devises · XAUEUR · XAUAUD · XAUCHF · XAUGBP · USATECHIDXUSD
====================================================================================

Teste la stratégie S&D (V4 bidirectionnel, même paramètres que XAUUSD champion)
sur les paires or/devises hors USD et sur le Nasdaq 100 (USATECHIDXUSD).

Protocole
---------
  V0 — Long-only (WS_long≥5.9)            baseline sur chaque paire
  V4 — Bidirectionnel (WS_short≥8.5)      variante production

Données attendues (format histdata MT4/MT5, même structure que XAUUSD) :
  data/historical/xaueur/m1/DAT_MT_XAUEUR_M1_<YYYY>.csv
  data/historical/xauaud/m1/DAT_MT_XAUAUD_M1_<YYYY>.csv
  data/historical/xauchf/m1/DAT_MT_XAUCHF_M1_<YYYY>.csv
  data/historical/xaugbp/m1/DAT_MT_XAUGBP_M1_<YYYY>.csv
  data/historical/usatechidxusd/m1/DAT_MT_USATECHIDXUSD_M1_<YYYY>.csv

  Année 2026 Jan-Jun : nommer les fichiers mensuels
  DAT_MT_XAUEUR_M1_202601.csv … DAT_MT_XAUEUR_M1_202606.csv
  ou fichier annuel DAT_MT_XAUEUR_M1_202601-202606.csv

Spreads (valeurs typiques retail, ajuster selon broker) :
  XAUEUR        0.50 EUR/oz    q2u≈0.926  (USDEUR = 1/EURUSD)
  XAUAUD        0.80 AUD/oz    q2u≈1.538  (USDAUD = 1/AUDUSD)
  XAUCHF        0.50 CHF/oz    q2u≈0.900  (USDCHF)
  XAUGBP        0.40 GBP/oz    q2u≈0.787  (USDGBP = 1/GBPUSD)
  USATECHIDXUSD 1.00 USD/point q2u=1.000  (déjà en USD, pas de conversion)

quote_to_usd_rate convention : 1 quote-currency = (1/q2u) USD
  → pnl_usd = pnl_quote / q2u   (ex. EUR: pnl_usd = pnl_eur / 0.926 = pnl_eur × 1.08)

Usage
-----
    python -m zeus.backtest.run_sd_xau_crosses
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd

from zeus.backtest.data_loader    import parse_m1_csv, resample_ohlcv
from zeus.backtest.sd_simulation  import compute_metrics, simulate_all
from zeus.strategy.supply_demand.sd_strategy   import SDStrategy
from zeus.strategy.supply_demand.wyckoff       import WyckoffDetector
from zeus.strategy.supply_demand.zone_detector import ZoneDetector

# ── Chemins ───────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parent.parent.parent
_DATA = _ROOT / "data" / "historical"

# ── Paramètres fixes ──────────────────────────────────────────────────────────

INITIAL_BALANCE = 10_000.0
RISK_PCT        = 0.01
RUNNER_RR       = 20.0
ATR_PERIOD      = 14

_WY = dict(
    lookback             = 200,
    max_accum_bars       = 20,
    accum_range_mult     = 6.0,
    mss_lookback         = 60,
    min_spring_sweep_pct = 0.05,
    min_mss_strength_pct = 0.03,
)

_4T = dict(
    tp1_r              = 1.0,
    tp1_size           = 0.0,
    tp2_r              = 3.0,
    tp2_cumulative_pct = 0.60,
    tp3_r              = 8.0,
    tp3_cumulative_pct = 0.85,
    max_monthly_losses = 4,
    max_daily_losses   = 1,
)

_BASE = dict(
    min_zone_score              = 5.0,
    min_composite_score         = 5.0,
    signal_cooldown             = 10,
    use_trend_filter            = True,
    trend_slope_lookback        = 3,
    trend_slope_lookback_short  = 20,
    use_price_above_ema         = True,
    use_price_above_ema_for_shorts = False,
    use_session_filter          = True,
    session_start_utc           = 7,
    session_end_utc             = 21,
    max_signals_per_day         = 6,
    use_adx_filter              = False,
    use_h4_trend_filter         = False,
    use_rsi_filter              = False,
)

# ── Définition des paires ─────────────────────────────────────────────────────
#
# spread   : en unités de la devise cotée par once d'or
# q2u      : USDXXX = 1 / XXXUSD  (convention simulate_all)
#            pnl_usd = pnl_quote / q2u
#
# Ces valeurs sont des approximations 2024-2025 moyennes.
# À ajuster selon le broker et la période testée.

PAIRS: list[dict] = [
    dict(
        symbol = "XAUEUR",
        folder = "xaueur",
        spread = 0.50,
        q2u    = 0.926,   # USDEUR ≈ 1/1.08
        note   = "spread≈0.50 EUR/oz · q2u=1/EURUSD≈0.926",
    ),
    dict(
        symbol = "XAUAUD",
        folder = "xauaud",
        spread = 0.80,
        q2u    = 1.538,   # USDAUD ≈ 1/0.65
        note   = "spread≈0.80 AUD/oz · q2u=1/AUDUSD≈1.538",
    ),
    dict(
        symbol = "XAUCHF",
        folder = "xauchf",
        spread = 0.50,
        q2u    = 0.900,   # USDCHF ≈ 0.90
        note   = "spread≈0.50 CHF/oz · q2u=USDCHF≈0.900",
    ),
    dict(
        symbol = "XAUGBP",
        folder = "xaugbp",
        spread = 0.40,
        q2u    = 0.787,   # USDGBP ≈ 1/1.27
        note   = "spread≈0.40 GBP/oz · q2u=1/GBPUSD≈0.787",
    ),
    dict(
        symbol = "USATECHIDXUSD",
        folder = "usatechidxusd",
        spread = 1.00,    # ~1 point USD (NAS100 CFD typique retail)
        q2u    = 1.000,   # déjà en USD, pas de conversion
        note   = "spread≈1.0 USD/pt · q2u=1.0 (Nasdaq 100 CFD)",
    ),
]

# ── Périodes (ajout automatique si fichiers présents) ─────────────────────────

def _period_files(folder: str) -> list[tuple[str, list[Path]]]:
    """Return (label, [files]) for each available period in this pair's directory."""
    sym = folder.upper()
    d   = _DATA / folder / "m1"
    periods = []

    for year in [2024, 2025]:
        candidates = [
            d / f"DAT_MT_{sym}_M1_{year}.csv",
            d / f"{sym}_M1_{year}.csv",
        ]
        found = next((f for f in candidates if f.exists()), None)
        if found:
            periods.append((str(year), [found]))

    # 2026 : fichier annuel, mensuel, ou nom libre (any *2026*.csv)
    annual_candidates = [
        d / f"DAT_MT_{sym}_M1_2026.csv",
        d / f"{sym}_M1_2026.csv",
    ]
    annual = next((f for f in annual_candidates if f.exists()), None)
    if annual:
        periods.append(("2026", [annual]))
    else:
        monthly = [d / f"DAT_MT_{sym}_M1_2026{m:02d}.csv" for m in range(1, 7)]
        found_m  = [f for f in monthly if f.exists()]
        if found_m:
            periods.append(("2026 Jan–Jun", found_m))
        else:
            # Fichier avec nom quelconque contenant "2026" (ex. export broker)
            any_2026 = sorted(d.glob("*2026*.csv")) if d.exists() else []
            if any_2026:
                periods.append(("2026", any_2026))

    return periods

# ── Variantes ─────────────────────────────────────────────────────────────────

VARIANTS: dict[str, dict] = {
    "V0 — Long-only (WS_long≥5.9)": dict(
        min_wyckoff_score       = 5.9,
        min_wyckoff_score_short = 99.0,
    ),
    "V4 — Bidirectionnel (WS_short≥8.5)": dict(
        min_wyckoff_score       = 5.9,
        min_wyckoff_score_short = 8.5,
    ),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_m1(files: list[Path]) -> pd.DataFrame:
    frames = [parse_m1_csv(f) for f in files if f.exists()]
    if not frames:
        raise FileNotFoundError(str(files))
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="first")]


def _compute_h4_atr(m1_df: pd.DataFrame) -> pd.Series:
    h4 = resample_ohlcv(m1_df, "4h")
    prev_close = h4["close"].shift(1)
    tr = pd.concat([
        h4["high"] - h4["low"],
        (h4["high"] - prev_close).abs(),
        (h4["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=ATR_PERIOD, adjust=False).mean().dropna()


def _build_strategy(variant_params: dict) -> SDStrategy:
    params = {**_BASE, **variant_params}
    return SDStrategy(
        zone_detector    = ZoneDetector(),
        wyckoff_detector = WyckoffDetector(**_WY),
        risk_reward      = RUNNER_RR,
        **params,
    )


def _run_period(
    m1_df:   pd.DataFrame,
    variant: dict,
    spread:  float,
    q2u:     float,
    h4_atr:  pd.Series,
) -> tuple[dict, int, int, int]:
    zone_df  = resample_ohlcv(m1_df, "15min")
    strategy = _build_strategy(variant)
    signals  = strategy.run(zone_df, m1_df)
    n_sig    = len(signals)
    n_long   = sum(1 for s in signals if s.direction == "long")
    n_short  = n_sig - n_long
    signals  = [dataclasses.replace(s, risk_reward=RUNNER_RR) for s in signals]

    results, n_exp = simulate_all(
        signals, m1_df,
        risk_pct          = RISK_PCT,
        spread            = spread,
        h4_atr            = h4_atr,
        use_tp2_trailing  = False,
        quote_to_usd_rate = q2u,
        **_4T,
    )
    m = compute_metrics(results, INITIAL_BALANCE, n_sig, n_exp)
    return m, n_sig, n_long, n_short


# ── Affichage ─────────────────────────────────────────────────────────────────

_W    = 110
_SEP  = "─" * _W
_SEP2 = "═" * _W


def _print_pair_block(symbol: str, variant_name: str, rows: list[tuple]) -> dict:
    tw = tl = max_dd = 0
    tr = 0.0
    all_pos = True

    for period_label, m, n_sig, n_long, n_short in rows:
        w, l  = m["n_wins"], m["n_losses"]
        tot_r = m["total_r"]
        pnl   = m["total_usd"]
        dd    = m["max_dd"]
        avg_r = tot_r / (w + l) if (w + l) else 0.0
        ls    = f"{n_long}L/{n_short}S"
        print(
            f"  {symbol:<8}  {period_label:<22}  "
            f"{n_sig:>4} {ls:>6}  {w:>3} {l:>3}  {(w/(w+l)*100 if (w+l) else 0):>5.1f}%  "
            f"{tot_r:>+7.2f}  {pnl:>+9.0f}$  {dd:>5.1f}%  {avg_r:>+6.2f}"
        )
        tw += w; tl += l; tr += tot_r
        max_dd = max(max_dd, dd)
        if tot_r < 0:
            all_pos = False

    return dict(tw=tw, tl=tl, tr=tr, max_dd=max_dd, all_pos=all_pos)


def _print_variant_block(variant_name: str, pair_results: dict) -> dict:
    print(f"\n  ┌{'─' * (_W - 4)}┐")
    print(f"  │  {variant_name:<{_W - 7}} │")
    print(f"  └{'─' * (_W - 4)}┘")
    print(f"  {'Paire':<8}  {'Période':<22}  "
          f"{'Sig':>4} {'L/S':>6}  {'W':>3} {'L':>3}  {'WR%':>5}  "
          f"{'R':>7}  {'P&L$':>9}  {'DD%':>5}  {'AvgR':>6}")
    print("  " + _SEP)

    totals_per_pair: dict[str, dict] = {}
    for symbol, rows in pair_results.items():
        if not rows:
            print(f"  {symbol:<8}  — aucune donnée disponible")
            continue
        t = _print_pair_block(symbol, variant_name, rows)
        totals_per_pair[symbol] = t

    if not totals_per_pair:
        return {}

    # ── Ligne TOTAL toutes paires ──────────────────────────────────────
    tw = sum(t["tw"] for t in totals_per_pair.values())
    tl = sum(t["tl"] for t in totals_per_pair.values())
    tr = sum(t["tr"] for t in totals_per_pair.values())
    mdd = max(t["max_dd"] for t in totals_per_pair.values())
    dec = tw + tl
    wr  = tw / dec * 100 if dec else 0.0
    all_pos = all(t["all_pos"] for t in totals_per_pair.values())

    print("  " + _SEP)
    print(f"  {'':8}  {'TOTAL toutes paires':<22}  {'':6}  {tw:>3} {tl:>3}  "
          f"{wr:>5.1f}%  {tr:>+7.2f}")
    verdict = "PASS ✓" if wr >= 60 and all_pos and dec >= 10 else "FAIL ✗"
    if dec < 10:
        verdict += f"  (⚠ seulement {dec} trades)"
    print(f"  VERDICT : {verdict}")

    return dict(tw=tw, tl=tl, tr=tr, max_dd=mdd, wr=wr, decided=dec)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(_SEP2)
    print("  S&D — Or vs devises · XAUEUR · XAUAUD · XAUCHF · XAUGBP · USATECHIDXUSD (Nasdaq 100)")
    print("  V4 bidirectionnel (WS_long≥5.9 · WS_short≥8.5) · 4T : TP1@1R · TP2@3R(60%) · TP3@8R(85%) · Runner@20R")
    print(_SEP2)

    # ── Pré-chargement par paire ──────────────────────────────────────────────
    print("\n  Données disponibles :")
    pair_data: dict[str, list[tuple[str, pd.DataFrame, pd.Series, float, float]]] = {}

    for p in PAIRS:
        sym    = p["symbol"]
        folder = p["folder"]
        spread = p["spread"]
        q2u    = p["q2u"]
        periods = _period_files(folder)

        if not periods:
            print(f"    ✗  {sym:<8}  aucune donnée — déposer les CSV dans data/historical/{folder}/m1/")
            pair_data[sym] = []
            continue

        pair_data[sym] = []
        for period_label, files in periods:
            try:
                m1  = _load_m1(files)
                atr = _compute_h4_atr(m1)
                open_p  = m1["open"].iloc[0]
                close_p = m1["close"].iloc[-1]
                chg     = (close_p - open_p) / open_p * 100
                print(f"    ✓  {sym:<8}  {period_label:<16}  {len(m1):>8,} barres  "
                      f"|  {open_p:.0f}→{close_p:.0f}  ({chg:>+.1f}%)  "
                      f"|  {p['note']}")
                pair_data[sym].append((period_label, m1, atr, spread, q2u))
            except Exception as e:
                print(f"    ✗  {sym:<8}  {period_label}  ERREUR — {e}")

    available = {sym: rows for sym, rows in pair_data.items() if rows}
    if not available:
        print("\n  Aucune donnée disponible. Télécharger les CSV et relancer.")
        return

    # ── Run variantes ─────────────────────────────────────────────────────────
    summary: list[tuple[str, dict]] = []

    for variant_name, variant_params in VARIANTS.items():
        pair_results: dict[str, list[tuple]] = {}
        for sym, periods in pair_data.items():
            rows = []
            for period_label, m1_df, h4_atr, spread, q2u in periods:
                m, n_sig, n_long, n_short = _run_period(
                    m1_df, variant_params, spread, q2u, h4_atr
                )
                rows.append((period_label, m, n_sig, n_long, n_short))
            pair_results[sym] = rows

        totals = _print_variant_block(variant_name, pair_results)
        summary.append((variant_name, totals))

    # ── Récapitulatif global ──────────────────────────────────────────────────
    print(f"\n{_SEP2}")
    print("  RÉCAPITULATIF GLOBAL — classé par Total R")
    print(f"{'─' * _W}")
    print(f"\n  {'Variante':<46}  {'W':>3} {'L':>3}  {'WR%':>5}  {'TotR':>7}  {'MaxDD':>6}  {'vs V0':>8}")
    print("  " + "─" * 85)

    base_r = next((t["tr"] for n, t in summary if n.startswith("V0")), 0.0)
    best_r = max((t["tr"] for _, t in summary if t), default=0.0)
    ranked = sorted(summary, key=lambda x: x[1].get("tr", 0.0), reverse=True)

    for name, t in ranked:
        if not t:
            print(f"  {name:<46}  — aucune donnée")
            continue
        delta = f"{t['tr'] - base_r:>+7.2f}" if abs(t["tr"] - base_r) > 0.001 else "    ref"
        best  = " ← MEILLEUR" if abs(t["tr"] - best_r) < 0.001 else ""
        print(
            f"  {name:<46}  {t['tw']:>3} {t['tl']:>3}  "
            f"{t['wr']:>5.1f}%  {t['tr']:>+7.2f}  {t['max_dd']:>5.1f}%  {delta}{best}"
        )

    print("\n  Paramètres par instrument :")
    for p in PAIRS:
        print(f"    {p['symbol']:<16}  {p['note']}")
    print(f"  Référence XAUUSD V4 : +113.90R · WR=73.7% (2024+2025+2026 Jan-Jun)")
    print(f"\n{_SEP2}")
    print("  Fin")
    print(f"{_SEP2}\n")


if __name__ == "__main__":
    main()
