"""
S&D Strategy — Bear Market Test · XAUUSD · 2011–2015
=====================================================

Gold bear market: peak Sep 2011 (~$1921) → low Dec 2015 (~$1045), -45%.
Goal: validate short-only variant on a sustained downtrend.

Variantes
---------
  V0 — Long-only baseline : comportement en bear market (attendu: mauvais)
  V7 — Short-only WS≥8.5  : signaux short sélectifs uniquement
  V8 — Short-only WS≥7.0  : seuil plus bas → plus de signaux
  V9 — Bidirectionnel     : WS_long≥5.9 + WS_short≥8.5 (comme V4 sur bull)
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

INITIAL_BALANCE     = 10_000.0
RISK_PCT            = 0.01
SPREAD              = 0.30
RUNNER_RR           = 20.0
ATR_PERIOD          = 14

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

_BASE = dict(
    min_zone_score              = 5.0,
    min_composite_score         = 5.0,
    signal_cooldown             = 10,
    use_trend_filter            = True,
    trend_slope_lookback        = 3,
    trend_slope_lookback_short  = 20,  # longer lookback for shorts: allows entry during bounces
    use_price_above_ema         = True,
    use_session_filter          = True,
    session_start_utc           = 7,
    session_end_utc             = 21,
    max_signals_per_day         = 6,
    use_adx_filter              = False,
    use_h4_trend_filter         = False,
    use_rsi_filter              = False,
)

# ── Variantes ─────────────────────────────────────────────────────────────────

VARIANTS: dict[str, dict] = {
    "V0 — Long-only baseline (WS_long≥5.9)": dict(
        min_wyckoff_score       = 5.9,
        min_wyckoff_score_short = 99.0,
    ),
    "V7 — Short-only (WS_short≥8.5)": dict(
        min_wyckoff_score       = 99.0,
        min_wyckoff_score_short = 8.5,
    ),
    "V8 — Short-only (WS_short≥7.0)": dict(
        min_wyckoff_score       = 99.0,
        min_wyckoff_score_short = 7.0,
    ),
    "V9 — Bidirectionnel (WS_long≥5.9 · WS_short≥8.5)": dict(
        min_wyckoff_score       = 5.9,
        min_wyckoff_score_short = 8.5,
    ),
}

# ── Périodes ──────────────────────────────────────────────────────────────────

PERIODS = [
    ("2011 (peak→correction)", [_M1 / "DAT_MT_XAUUSD_M1_2011.csv"]),
    ("2012 (range/dist)"     , [_M1 / "DAT_MT_XAUUSD_M1_2012.csv"]),
    ("2013 (crash -28%)"     , [_M1 / "DAT_MT_XAUUSD_M1_2013.csv"]),
    ("2014 (bear continue)"  , [_M1 / "DAT_MT_XAUUSD_M1_2014.csv"]),
    ("2015 (capitulation)"   , [_M1 / "DAT_MT_XAUUSD_M1_2015.csv"]),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_m1(files: list[Path]) -> pd.DataFrame:
    frames = [parse_histdata_csv(f) for f in files if f.exists()]
    if not frames:
        raise FileNotFoundError(f"M1 data missing: {files}")
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="first")]


def _compute_h4_atr(m1_df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    h4 = resample_ohlcv(m1_df, "4h")
    prev_close = h4["close"].shift(1)
    tr = pd.concat([
        h4["high"] - h4["low"],
        (h4["high"] - prev_close).abs(),
        (h4["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean().dropna()


def _build_strategy(variant_params: dict) -> SDStrategy:
    params = {**_BASE, **variant_params}
    return SDStrategy(
        zone_detector    = ZoneDetector(),
        wyckoff_detector = WyckoffDetector(**_WY),
        risk_reward      = RUNNER_RR,
        **params,
    )


def _run(
    m1_df:        pd.DataFrame,
    variant_params: dict,
    h4_atr:       pd.Series,
) -> tuple[dict, int, int, int]:
    zone_df  = resample_ohlcv(m1_df, "15min")
    strategy = _build_strategy(variant_params)
    signals  = strategy.run(zone_df, m1_df)
    n_sig    = len(signals)
    n_long   = sum(1 for s in signals if s.direction == "long")
    n_short  = n_sig - n_long
    signals  = [dataclasses.replace(s, risk_reward=RUNNER_RR) for s in signals]

    results, n_exp = simulate_all(
        signals, m1_df,
        risk_pct            = RISK_PCT,
        spread              = SPREAD,
        h4_atr              = h4_atr,
        use_tp2_trailing    = False,
        tp2_trailing_factor = 1.5,
        **_4T_BEST,
    )
    m = compute_metrics(results, INITIAL_BALANCE, n_sig, n_exp)
    return m, n_sig, n_long, n_short

# ── Affichage ─────────────────────────────────────────────────────────────────

_W    = 112
_SEP  = "─" * _W
_SEP2 = "═" * _W


def _print_block(variant_name: str, rows: list[tuple]) -> dict:
    print(f"\n  ┌{'─' * (_W - 4)}┐")
    print(f"  │  {variant_name:<{_W - 7}} │")
    print(f"  └{'─' * (_W - 4)}┘")
    print(f"  {'Paire':<8}  {'Période':<26}  "
          f"{'Sig':>4} {'L/S':>6}  {'W':>3} {'L':>3}  {'WR%':>5}  "
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
            f"  {'XAUUSD':<8}  {period_label:<26}  "
            f"{n_sig:>4} {ls_str:>6}  {w:>3} {l:>3}  {wr:>5.1f}%  "
            f"{tot_r:>+7.2f}  {pnl:>+9.0f}$  {dd:>5.1f}%  {avg_r:>+6.2f}"
        )
        tw += w; tl += l; tr += tot_r
        max_dd = max(max_dd, dd)
        if tot_r < 0:
            all_pos = False

    decided = tw + tl
    wr_tot  = tw / decided * 100 if decided else 0.0
    print("  " + _SEP)
    print(f"  {'':8}  {'TOTAL':<26}  {'':6}  {tw:>3} {tl:>3}  "
          f"{wr_tot:>5.1f}%  {tr:>+7.2f}")
    verdict = "PASS ✓" if wr_tot >= 60 and all_pos and decided >= 10 else "FAIL ✗"
    if decided < 10:
        verdict += f"  (⚠ seulement {decided} trades)"
    print(f"  VERDICT : {verdict}")

    return dict(tw=tw, tl=tl, wr=wr_tot, tr=tr, max_dd=max_dd, decided=decided)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(_SEP2)
    print("  S&D — Bear Market Test · XAUUSD · 2011–2015 (pic Sep'11 ~$1921 → creux Déc'15 ~$1045, -45%)")
    print("  4T : TP1@1R(BE) · TP2@3R(60%) · TP3@8R(85%) · Runner@20R")
    print(_SEP2)

    print("\n  Chargement M1…")
    period_data: list[tuple] = []
    for period_label, files in PERIODS:
        try:
            m1  = _load_m1(files)
            atr = _compute_h4_atr(m1)
            period_data.append((period_label, m1, atr))
            open_price = m1["open"].iloc[0]
            close_price = m1["close"].iloc[-1]
            chg = (close_price - open_price) / open_price * 100
            print(f"    ✓  {period_label:<26}  {len(m1):>8,} barres  "
                  f"|  {open_price:.0f}→{close_price:.0f}$  ({chg:>+.1f}%)  "
                  f"|  ATR(H4) moy = {atr.mean():.2f}")
        except FileNotFoundError as e:
            print(f"    ✗  {period_label}  MANQUANT — {e}")

    if not period_data:
        return

    summary: list[tuple[str, dict]] = []

    for variant_name, vcfg in VARIANTS.items():
        rows = []
        for period_label, m1_df, h4_atr in period_data:
            m, n_sig, n_long, n_short = _run(m1_df, vcfg, h4_atr)
            rows.append((period_label, m, n_sig, n_long, n_short))
        totals = _print_block(variant_name, rows)
        summary.append((variant_name, totals))

    # ── Récapitulatif ─────────────────────────────────────────────────────────
    print(f"\n{_SEP2}")
    print("  RÉCAPITULATIF GLOBAL — classé par Total R")
    print(f"{'─' * _W}")
    print(f"\n  {'Variante':<50}  {'Trades':>6}  {'W':>3} {'L':>3}  {'WR%':>5}  {'TotR':>7}  {'MaxDD':>6}")
    print("  " + "─" * 85)

    ranked = sorted(summary, key=lambda x: x[1]["tr"], reverse=True)
    best_r = max(t["tr"] for _, t in summary)

    for name, t in ranked:
        best = " ← MEILLEUR" if t["tr"] == best_r else ""
        print(
            f"  {name:<50}  {t['decided']:>6}  {t['tw']:>3} {t['tl']:>3}  "
            f"{t['wr']:>5.1f}%  {t['tr']:>+7.2f}  {t['max_dd']:>5.1f}%{best}"
        )

    print(f"\n{_SEP2}")
    print("  Fin")
    print(f"{_SEP2}\n")


if __name__ == "__main__":
    main()
