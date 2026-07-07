"""
Harnais d'équivalence EA MQL5 ↔ stratégie Python de référence.
================================================================

Protocole :
  1. Dans MT5 : script ExportBars.mq5 sur le graphique du symbole
     → MQL5/Files/ZeusBars_<SYM>.csv   (barres M1 du broker, UTC)
  2. Dans MT5 : EA ZeusP11 en mode InpSignalLogMode=true (testeur de
     stratégie ou live-observation) → ZeusP11_signals_*.csv
  3. Ici :
       python -m zeus.backtest.compare_ea_signals \\
           --bars ZeusBars_XAUUSD.csv --signals ZeusP11_signals.csv \\
           --symbol XAUUSD

Le script rejoue la stratégie de référence (configs P11-V4 gelées) sur
les MÊMES barres broker et compare signal par signal (timestamp +
direction ; prix à tolérance 1e-4).  Exigence avant tout trading réel :
100% de correspondance.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from zeus.backtest.data_loader import resample_ohlcv
from zeus.strategy.supply_demand.sd_strategy import SDStrategy
from zeus.strategy.supply_demand.wyckoff import WyckoffDetector
from zeus.strategy.supply_demand.zone_detector import ZoneDetector

_WY_XAU = dict(lookback=200, max_accum_bars=20, accum_range_mult=6.0,
               mss_lookback=60, min_spring_sweep_pct=0.05, min_mss_strength_pct=0.03)
_WY_FX  = dict(_WY_XAU, min_spring_sweep_pct=0.10)

_STRAT_XAU = dict(
    risk_reward=20.0, min_zone_score=5.0, min_composite_score=5.0,
    signal_cooldown=10, use_trend_filter=True, trend_slope_lookback=3,
    use_price_above_ema=True, use_session_filter=True,
    session_start_utc=7, session_end_utc=21, max_signals_per_day=10,
    min_wyckoff_score=5.9, min_wyckoff_score_short=8.5,
)

def _strat_fx(pip: float) -> dict:
    return dict(
        risk_reward=10.0, min_zone_score=4.0, min_composite_score=4.0,
        signal_cooldown=10, use_trend_filter=True, trend_slope_lookback=3,
        use_price_above_ema=True, ema_atr_tolerance=0.5, use_session_filter=True,
        session_start_utc=7, session_end_utc=17, max_signals_per_day=6,
        min_sl_pips=5, pip_size=pip,
        min_wyckoff_score=7.5, min_wyckoff_score_short=6.5,
    )


def _load_bars(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep=";", header=None,
        names=["datetime", "open", "high", "low", "close", "volume"],
    )
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y.%m.%d %H:%M")
    df = df.set_index("datetime").sort_index()
    return df[~df.index.duplicated(keep="first")]


def _load_ea_signals(path: Path, symbol: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df = df[df["symbol"] == symbol].copy()
    df["formed_at_utc"] = pd.to_datetime(df["formed_at_utc"], format="%Y.%m.%d %H:%M")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", required=True, help="ZeusBars_<SYM>.csv (ExportBars.mq5)")
    ap.add_argument("--signals", required=True, help="ZeusP11_signals_*.csv (EA)")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--pip", type=float, default=None,
                    help="pip size (défaut : 0.01 JPY/metal-idx, sinon 0.0001)")
    ap.add_argument("--xau", action="store_true",
                    help="utiliser la config V4 or (sinon forex WS7.5)")
    ap.add_argument("--start", default=None,
                    help="ignorer les signaux avant cette date (warm-up EA), "
                         "ex. 2025-01-15")
    ap.add_argument("--all-per-bar", action="store_true",
                    help="garder tous les signaux de référence par barre "
                         "(défaut : 1er seulement, comme l'EA/live)")
    args = ap.parse_args()

    pip = args.pip if args.pip is not None else (0.01 if "JPY" in args.symbol else 0.0001)

    m1 = _load_bars(Path(args.bars))
    m15 = resample_ohlcv(m1, "15min")
    print(f"Barres broker : {len(m1):,} M1 · {len(m15):,} M15  "
          f"({m1.index[0]} → {m1.index[-1]})")

    if args.xau or args.symbol.upper().startswith("XAU"):
        strat_params, wy = _STRAT_XAU, _WY_XAU
    else:
        strat_params, wy = _strat_fx(pip), _WY_FX

    strategy = SDStrategy(zone_detector=ZoneDetector(),
                          wyckoff_detector=WyckoffDetector(**wy), **strat_params)
    ref_signals = strategy.run(m15, m1)
    ref = pd.DataFrame([
        dict(formed_at=s.formed_at, direction=s.direction,
             entry=s.entry_price, sl=s.stop_loss,
             zone=s.zone_score, ws=s.wyckoff_score)
        for s in ref_signals
    ])
    if not args.all_per_bar and not ref.empty:
        # L'EA/live émet au plus 1 signal par barre (le 1er en ordre de zone) —
        # aligner la référence sur cette sémantique
        ref = ref.drop_duplicates(subset=["formed_at"], keep="first")
    if args.start and not ref.empty:
        cutoff = pd.Timestamp(args.start)
        ref = ref[ref["formed_at"] >= cutoff]
    print(f"Référence Python : {len(ref)} signaux")

    ea = _load_ea_signals(Path(args.signals), args.symbol)
    if args.start and not ea.empty:
        ea = ea[ea["formed_at_utc"] >= pd.Timestamp(args.start)]
    print(f"EA MQL5          : {len(ea)} signaux")

    # ── Comparaison par (timestamp, direction) ────────────────────────
    ref_keys = {(r.formed_at, r.direction): r for r in ref.itertuples()}
    ea_keys  = {(r.formed_at_utc, r.direction): r for r in ea.itertuples()}

    matched, price_diff = [], []
    missing_in_ea  = [k for k in ref_keys if k not in ea_keys]
    extra_in_ea    = [k for k in ea_keys  if k not in ref_keys]
    for k, r in ref_keys.items():
        if k not in ea_keys:
            continue
        e = ea_keys[k]
        matched.append(k)
        if abs(float(e.entry) - r.entry) > 1e-4 or abs(float(e.sl) - r.sl) > 1e-4:
            price_diff.append((k, r.entry, float(e.entry), r.sl, float(e.sl)))

    total_ref = max(len(ref_keys), 1)
    match_pct = len(matched) / total_ref * 100

    print("\n══ RÉSULTAT ══")
    print(f"  Correspondances : {len(matched)}/{len(ref_keys)}  ({match_pct:.1f}%)")
    print(f"  Manquants EA    : {len(missing_in_ea)}")
    print(f"  En trop EA      : {len(extra_in_ea)}")
    print(f"  Écarts de prix  : {len(price_diff)}")

    for k in missing_in_ea[:10]:
        print(f"    manquant : {k[0]} {k[1]}")
    for k in extra_in_ea[:10]:
        print(f"    en trop  : {k[0]} {k[1]}")
    for k, re_, ee, rs, es in price_diff[:10]:
        print(f"    prix     : {k[0]} {k[1]} entry {re_} vs {ee} · sl {rs} vs {es}")

    ok = (match_pct == 100.0 and not extra_in_ea and not price_diff)
    print(f"\n  VERDICT : {'ÉQUIVALENCE CONFIRMÉE ✓' if ok else 'DIVERGENCE — NE PAS TRADER ✗'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
