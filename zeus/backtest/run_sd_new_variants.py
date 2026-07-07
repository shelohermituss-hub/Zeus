"""
S&D Strategy — Nouvelles variantes · XAUUSD · Mode A (M1 référence)
=====================================================================

Variantes testées
-----------------
  V0 — Baseline       : WS_long≥5.9 · long-only · TP3@8R + Runner@20R
  V4 — Short sélectif : WS_long≥5.9 · WS_short≥8.5 · trend-filter obligatoire
  V5 — Trailing ATR   : WS_long≥5.9 · long-only · trail 1.5×ATR(H4) après TP2
  V6 — Short+Trail    : WS_short≥8.5 + Trailing ATR combinés

Résultats : par variante × par année × XAUUSD
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
_M1   = _ROOT / "data" / "historical" / "xauusd" / "m1"

# ── Paramètres fixes ──────────────────────────────────────────────────────────

INITIAL_BALANCE  = 10_000.0
RISK_PCT         = 0.01
SPREAD           = 0.30
RUNNER_RR        = 20.0
ATR_PERIOD       = 14       # ATR lookback for both H4 and M30

_WY = dict(
    lookback             = 200,
    max_accum_bars       = 20,
    accum_range_mult     = 6.0,
    mss_lookback         = 60,
    min_spring_sweep_pct = 0.05,
    min_mss_strength_pct = 0.03,
)

# 4-tier exit structure (identique Round 24)
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

# Base partagée (non overridée par variante)
_BASE = dict(
    min_zone_score          = 5.0,
    min_composite_score     = 5.0,
    signal_cooldown         = 10,
    use_trend_filter        = True,   # obligatoire pour longs et shorts
    trend_slope_lookback    = 3,
    use_price_above_ema     = True,
    use_session_filter      = True,
    session_start_utc       = 7,
    session_end_utc         = 21,
    max_signals_per_day     = 6,
    use_adx_filter          = False,
    use_h4_trend_filter     = False,
    use_rsi_filter          = False,
)

# ── Définition des variantes ──────────────────────────────────────────────────

VARIANTS: dict[str, dict] = {
    "V0 — Baseline (WS≥5.9 · long-only · TP3+Runner)": dict(
        strat   = dict(min_wyckoff_score=5.9, min_wyckoff_score_short=99.0),
        sim     = dict(use_tp2_trailing=False, tp2_trailing_factor=1.5),
        atr_src = "h4",
    ),
    "V4 — Short sélectif (WS_short≥8.5 · trend filtre)": dict(
        strat   = dict(min_wyckoff_score=5.9, min_wyckoff_score_short=8.5),
        sim     = dict(use_tp2_trailing=False, tp2_trailing_factor=1.5),
        atr_src = "h4",
    ),
    "V5 — Trail H4×1.5 (long-only)": dict(
        strat   = dict(min_wyckoff_score=5.9, min_wyckoff_score_short=99.0),
        sim     = dict(use_tp2_trailing=True,  tp2_trailing_factor=1.5),
        atr_src = "h4",
    ),
    "V6 — Trail H4×1.5 + Short≥8.5": dict(
        strat   = dict(min_wyckoff_score=5.9, min_wyckoff_score_short=8.5),
        sim     = dict(use_tp2_trailing=True,  tp2_trailing_factor=1.5),
        atr_src = "h4",
    ),
    "V7 — Trail H4×0.5 serré + Short≥8.5": dict(
        strat   = dict(min_wyckoff_score=5.9, min_wyckoff_score_short=8.5),
        sim     = dict(use_tp2_trailing=True,  tp2_trailing_factor=0.5),
        atr_src = "h4",
    ),
    "V8 — Trail M30×1.5 serré + Short≥8.5": dict(
        strat   = dict(min_wyckoff_score=5.9, min_wyckoff_score_short=8.5),
        sim     = dict(use_tp2_trailing=True,  tp2_trailing_factor=1.5),
        atr_src = "m30",
    ),
}

# ── Données par période ───────────────────────────────────────────────────────

PERIODS = [
    ("2017", [_M1 / "DAT_MS_XAUUSD_M1_2017.csv"]),
    ("2018", [_M1 / "DAT_MS_XAUUSD_M1_2018.csv"]),
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
    frames = [parse_m1_csv(f) for f in files if f.exists()]
    if not frames:
        raise FileNotFoundError(f"M1 data missing: {files}")
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="first")]


def _atr_from_ohlcv(ohlcv: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    prev_close = ohlcv["close"].shift(1)
    tr = pd.concat([
        ohlcv["high"] - ohlcv["low"],
        (ohlcv["high"] - prev_close).abs(),
        (ohlcv["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean().dropna()


def _compute_h4_atr(m1_df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    return _atr_from_ohlcv(resample_ohlcv(m1_df, "4h"), period)


def _compute_m30_atr(m1_df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    return _atr_from_ohlcv(resample_ohlcv(m1_df, "30min"), period)


def _build_strategy(strat_params: dict) -> SDStrategy:
    params = {**_BASE, **strat_params}
    return SDStrategy(
        zone_detector    = ZoneDetector(),
        wyckoff_detector = WyckoffDetector(**_WY),
        risk_reward      = RUNNER_RR,
        **{k: v for k, v in params.items()},
    )


def _run(
    m1_df:       pd.DataFrame,
    strat_params: dict,
    sim_extra:   dict,
    h4_atr:      pd.Series,
    m30_atr:     pd.Series,
    atr_src:     str,
) -> tuple[dict, int, int, int]:
    """Returns (metrics, n_signals, n_long, n_short)."""
    atr_series = h4_atr if atr_src == "h4" else m30_atr
    # Extract tp2_trailing_factor from sim_extra so it doesn't conflict
    sim_kwargs = dict(sim_extra)
    trail_factor = sim_kwargs.pop("tp2_trailing_factor", 1.5)

    zone_df  = resample_ohlcv(m1_df, "15min")
    strategy = _build_strategy(strat_params)
    signals  = strategy.run(zone_df, m1_df)
    n_sig    = len(signals)
    n_long   = sum(1 for s in signals if s.direction == "long")
    n_short  = n_sig - n_long
    signals  = [dataclasses.replace(s, risk_reward=RUNNER_RR) for s in signals]

    results, n_exp = simulate_all(
        signals, m1_df,
        risk_pct            = RISK_PCT,
        spread              = SPREAD,
        h4_atr              = atr_series,
        tp2_trailing_factor = trail_factor,
        **sim_kwargs,
        **_4T_BEST,
    )
    m = compute_metrics(results, INITIAL_BALANCE, n_sig, n_exp)
    m["n_signals"] = n_sig
    return m, n_sig, n_long, n_short


# ── Affichage ─────────────────────────────────────────────────────────────────

_W = 108
_SEP  = "─" * _W
_SEP2 = "═" * _W


def _print_block(variant_name: str, rows: list[tuple[str, dict, int, int, int]]) -> dict:
    print(f"\n  ┌{'─' * (_W - 4)}┐")
    print(f"  │  {variant_name:<{_W - 7}} │")
    print(f"  └{'─' * (_W - 4)}┘")
    print(f"  {'Paire':<8}  {'Période':<24}  "
          f"{'Sig':>4} {'L/S':>5}  {'W':>3} {'L':>3}  {'WR%':>5}  "
          f"{'R':>7}  {'P&L$':>9}  {'DD%':>5}  {'AvgR':>6}")
    print("  " + _SEP)

    tw = tl = max_dd = 0
    tr = 0.0
    all_pos = True

    for period_label, m, n_sig, n_long, n_short in rows:
        w, l   = m["n_wins"], m["n_losses"]
        wr     = m["win_rate"]
        tot_r  = m["total_r"]
        pnl    = m["total_usd"]
        dd     = m["max_dd"]
        avg_r  = tot_r / (w + l) if (w + l) else 0.0
        ls_str = f"{n_long}L/{n_short}S"
        print(
            f"  {'XAUUSD':<8}  {period_label:<24}  "
            f"{n_sig:>4} {ls_str:>5}  {w:>3} {l:>3}  {wr:>5.1f}%  "
            f"{tot_r:>+7.2f}  {pnl:>+9.0f}$  {dd:>5.1f}%  {avg_r:>+6.2f}"
        )
        tw += w; tl += l; tr += tot_r
        max_dd = max(max_dd, dd)
        if tot_r < 0:
            all_pos = False

    decided = tw + tl
    wr_tot  = tw / decided * 100 if decided else 0.0
    print("  " + _SEP)
    print(f"  {'':8}  {'TOTAL':<24}  {'':5}  {tw:>3} {tl:>3}  "
          f"{wr_tot:>5.1f}%  {tr:>+7.2f}")
    verdict = "PASS ✓" if wr_tot >= 60 and all_pos else "FAIL ✗"
    print(f"  VERDICT : {verdict}")

    return dict(tw=tw, tl=tl, wr=wr_tot, tr=tr, max_dd=max_dd)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(_SEP2)
    print("  S&D — Nouvelles variantes · XAUUSD · 4T Runner (TP1@1R · TP2@3R(60%) · TP3@8R(85%) · Runner@20R)")
    print(_SEP2)

    # Pré-chargement
    print("\n  Chargement M1…")
    period_data: list[tuple[str, pd.DataFrame, pd.Series, pd.Series]] = []
    for period_label, files in PERIODS:
        try:
            m1      = _load_m1(files)
            h4_atr  = _compute_h4_atr(m1)
            m30_atr = _compute_m30_atr(m1)
            period_data.append((period_label, m1, h4_atr, m30_atr))
            print(f"    ✓  {period_label:<24}  {len(m1):>8,} barres M1  "
                  f"|  ATR(H4)={h4_atr.mean():.2f}  ATR(M30)={m30_atr.mean():.2f} USD/oz")
        except FileNotFoundError as e:
            print(f"    ✗  {period_label}  MANQUANT — {e}")

    if not period_data:
        return

    summary: list[tuple[str, dict]] = []

    for variant_name, vcfg in VARIANTS.items():
        rows = []
        for period_label, m1_df, h4_atr, m30_atr in period_data:
            m, n_sig, n_long, n_short = _run(
                m1_df,
                strat_params = vcfg["strat"],
                sim_extra    = vcfg["sim"],
                h4_atr       = h4_atr,
                m30_atr      = m30_atr,
                atr_src      = vcfg["atr_src"],
            )
            rows.append((period_label, m, n_sig, n_long, n_short))
        totals = _print_block(variant_name, rows)
        summary.append((variant_name, totals))

    # ── Récapitulatif global ──────────────────────────────────────────────────
    print(f"\n{_SEP2}")
    print("  RÉCAPITULATIF GLOBAL — classé par Total R")
    print(f"{'─' * _W}")
    print(f"\n  {'Variante':<52}  {'W':>3} {'L':>3}  {'WR%':>5}  {'TotR':>7}  {'MaxDD':>6}  {'vs V0':>8}")
    print("  " + "─" * 85)

    base_r  = next((t["tr"] for n, t in summary if n.startswith("V0")), 0.0)
    best_r  = max(t["tr"] for _, t in summary)
    ranked  = sorted(summary, key=lambda x: x[1]["tr"], reverse=True)

    for name, t in ranked:
        delta = f"{t['tr'] - base_r:>+7.2f}" if abs(t["tr"] - base_r) > 0.001 else "   ref"
        best  = " ← MEILLEUR" if t["tr"] == best_r else ""
        print(
            f"  {name:<52}  {t['tw']:>3} {t['tl']:>3}  "
            f"{t['wr']:>5.1f}%  {t['tr']:>+7.2f}  {t['max_dd']:>5.1f}%  {delta}{best}"
        )

    print(f"\n{_SEP2}")
    print("  Fin")
    print(f"{_SEP2}\n")


if __name__ == "__main__":
    main()
