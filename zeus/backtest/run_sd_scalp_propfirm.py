"""
S&D Scalp propfirm — déclinaison scalp de la V4 sous contraintes propfirm strictes.
===================================================================================

Base : logique S&D/Wyckoff validée en production (V4), adaptée au scalping :
  - Zones M5 (au lieu de M15)  → plus de setups intraday
  - Confirmation Wyckoff M1     → identique à la V4
  - Sorties rapides             → TP1@1R(50%+BE) · TP2@2R(75%) · TP3@3R(100%)
                                  (pas de runner 20R : une propfirm paie la régularité)

Contraintes propfirm (PropFirmGuard, limites internes strictes) :
  - Risque fixe 0.3% du solde initial par trade (pas de compounding)
  - Arrêt journalier : perte 3% ou 3 pertes dans la journée
  - Arrêt définitif  : drawdown total 6% depuis le solde initial
  - Aucune entrée après 21h UTC

Variantes :
  S1 — Zones M5 · WS 5.9/8.5 · sorties scalp
  S2 — Zones M5 · WS 6.5/8.5 · sorties scalp (qualité stricte)
  S3 — Zones M15 (V4) · sorties scalp        (isole l'effet des sorties)

Le rapport mesure, par variante × période : trades, WR, R total, trades/jour,
pire journée, DD max sur equity propfirm, trades bloqués par le guard,
et le temps pour atteindre la cible challenge +10%.
"""
from __future__ import annotations

import dataclasses
from collections import defaultdict
from pathlib import Path

import pandas as pd

from zeus.backtest.data_loader import parse_m1_csv, resample_ohlcv
from zeus.backtest.sd_simulation import simulate_all
from zeus.risk.propfirm import PropFirmConfig, PropFirmGuard
from zeus.strategy.supply_demand.sd_strategy import SDStrategy
from zeus.strategy.supply_demand.wyckoff import WyckoffDetector
from zeus.strategy.supply_demand.zone_detector import ZoneDetector

# ── Chemins ───────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parent.parent.parent
_M1   = _ROOT / "data" / "historical" / "xauusd" / "m1"

# ── Paramètres ────────────────────────────────────────────────────────────────

ACCOUNT_BALANCE = 100_000.0   # compte propfirm type
SPREAD          = 0.30        # USD/oz
SCALP_RR        = 3.0         # TP final à 3R

_WY = dict(
    lookback             = 200,
    max_accum_bars       = 20,
    accum_range_mult     = 6.0,
    mss_lookback         = 60,
    min_spring_sweep_pct = 0.05,
    min_mss_strength_pct = 0.03,
)

# Sorties scalp : TP1@1R (50% + BE) · TP2@2R (75%) · TP3@3R (100%)
_SCALP_EXITS = dict(
    use_be             = True,
    tp1_r              = 1.0,
    tp1_size           = 0.5,
    tp2_r              = 2.0,
    tp2_cumulative_pct = 0.75,
    tp3_r              = 3.0,
    tp3_cumulative_pct = 1.0,
)

_BASE = dict(
    min_zone_score          = 5.0,
    min_composite_score     = 5.0,
    signal_cooldown         = 10,
    use_trend_filter        = True,
    trend_slope_lookback    = 3,
    use_price_above_ema     = True,
    use_session_filter      = True,
    session_start_utc       = 7,
    session_end_utc         = 21,
    max_signals_per_day     = 10,   # le guard propfirm limite en pratique
    use_adx_filter          = False,
    use_h4_trend_filter     = False,
    use_rsi_filter          = False,
)

VARIANTS: dict[str, dict] = {
    "S1 — Zones M5 · WS 5.9/8.5 · sorties scalp": dict(
        zone_tf = "5min",
        strat   = dict(min_wyckoff_score=5.9, min_wyckoff_score_short=8.5),
    ),
    "S2 — Zones M5 · WS 6.5/8.5 · qualité stricte": dict(
        zone_tf = "5min",
        strat   = dict(min_wyckoff_score=6.5, min_wyckoff_score_short=8.5),
    ),
    "S3 — Zones M15 (V4) · sorties scalp": dict(
        zone_tf = "15min",
        strat   = dict(min_wyckoff_score=5.9, min_wyckoff_score_short=8.5),
    ),
}

PERIODS = [
    ("2017", [_M1 / "DAT_MS_XAUUSD_M1_2017.csv"]),
    ("2018", [_M1 / "DAT_MS_XAUUSD_M1_2018.csv"]),
    ("2024", [_M1 / "DAT_MT_XAUUSD_M1_2024.csv"]),
    ("2025", [_M1 / "DAT_MT_XAUUSD_M1_2025.csv"]),
    ("2026 H1", [
        _M1 / f"DAT_MT_XAUUSD_M1_2026{m:02d}.csv" for m in range(1, 7)
    ]),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_m1(files: list[Path]) -> pd.DataFrame:
    frames = [parse_m1_csv(f) for f in files if f.exists()]
    if not frames:
        raise FileNotFoundError(f"M1 data missing: {files}")
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="first")]


def _build_strategy(strat_params: dict) -> SDStrategy:
    params = {**_BASE, **strat_params}
    return SDStrategy(
        zone_detector    = ZoneDetector(),
        wyckoff_detector = WyckoffDetector(**_WY),
        risk_reward      = SCALP_RR,
        **params,
    )


def _replay_through_guard(
    results: list,
    m1_index: pd.DatetimeIndex,
) -> dict:
    """Rejoue les trades chronologiquement dans le PropFirmGuard (risque fixe).

    Retourne les métriques propfirm : equity finale, DD max, pire journée,
    trades bloqués, jours d'arrêt, temps jusqu'à la cible +10%.
    """
    guard = PropFirmGuard(PropFirmConfig(), initial_balance=ACCOUNT_BALANCE)
    risk_usd = guard.risk_amount_usd()

    equity      = ACCOUNT_BALANCE
    peak        = ACCOUNT_BALANCE
    max_dd_pct  = 0.0
    n_taken     = 0
    n_blocked   = 0
    w = l = 0
    tot_r       = 0.0
    day_pnl: dict = defaultdict(float)
    target_date = None
    first_date  = None

    for tr in results:
        entry_ts = tr.signal.formed_at
        exit_ts  = m1_index[tr.exit_bar] if tr.exit_bar < len(m1_index) else entry_ts
        if first_date is None:
            first_date = entry_ts.date()

        if not guard.can_trade(entry_ts):
            n_blocked += 1
            continue

        pnl_usd = tr.pnl_r * risk_usd
        guard.on_trade_closed(exit_ts, pnl_usd)

        n_taken += 1
        tot_r   += tr.pnl_r
        equity  += pnl_usd
        day_pnl[entry_ts.date()] += pnl_usd
        if tr.outcome == "win":
            w += 1
        elif tr.outcome == "loss":
            l += 1

        peak = max(peak, equity)
        dd = (peak - equity) / ACCOUNT_BALANCE
        max_dd_pct = max(max_dd_pct, dd)

        if target_date is None and equity >= ACCOUNT_BALANCE * 1.10:
            target_date = exit_ts.date()

    worst_day_usd = min(day_pnl.values()) if day_pnl else 0.0
    days_to_target = (target_date - first_date).days if target_date and first_date else None

    return dict(
        n_taken        = n_taken,
        n_blocked      = n_blocked,
        w              = w,
        l              = l,
        tot_r          = tot_r,
        equity         = equity,
        max_dd_pct     = max_dd_pct * 100,
        worst_day_pct  = -worst_day_usd / ACCOUNT_BALANCE * 100 if worst_day_usd < 0 else 0.0,
        n_trade_days   = len(day_pnl),
        days_to_target = days_to_target,
        failed         = guard.challenge_failed(),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

_W = 116


def main() -> None:
    print("═" * _W)
    print("  S&D Scalp propfirm — risque fixe 0.3% · daily 3% · DD total 6% · "
          "sorties TP1@1R(50%)·TP2@2R(75%)·TP3@3R")
    print("═" * _W)

    print("\n  Chargement M1…")
    period_data: list[tuple[str, pd.DataFrame]] = []
    for label, files in PERIODS:
        try:
            m1 = _load_m1(files)
            period_data.append((label, m1))
            print(f"    ✓  {label:<10} {len(m1):>8,} barres M1")
        except FileNotFoundError as e:
            print(f"    ✗  {label}  MANQUANT — {e}")

    for vname, vcfg in VARIANTS.items():
        print(f"\n  ┌{'─' * (_W - 4)}┐")
        print(f"  │  {vname:<{_W - 7}} │")
        print(f"  └{'─' * (_W - 4)}┘")
        print(f"  {'Période':<10} {'Pris':>5} {'Bloq':>5}  {'W':>3} {'L':>3} "
              f"{'WR%':>6}  {'R':>8}  {'PnL$':>9}  {'DDmax':>6}  {'PireJ':>6}  "
              f"{'Jours':>5}  {'→+10%':>6}  Statut")
    # (en-tête répété par variante pour lisibilité des blocs)
        print("  " + "─" * (_W - 2))

        all_ok = True
        for label, m1 in period_data:
            zone_df  = resample_ohlcv(m1, vcfg["zone_tf"])
            strategy = _build_strategy(vcfg["strat"])
            signals  = strategy.run(zone_df, m1)
            signals  = [dataclasses.replace(s, risk_reward=SCALP_RR) for s in signals]

            results, _ = simulate_all(
                signals, m1,
                risk_pct = 0.003,
                spread   = SPREAD,
                initial_equity = ACCOUNT_BALANCE,
                **_SCALP_EXITS,
            )
            m = _replay_through_guard(results, m1.index)

            dec = m["w"] + m["l"]
            wr  = m["w"] / dec * 100 if dec else 0.0
            pnl = m["equity"] - ACCOUNT_BALANCE
            tgt = f"{m['days_to_target']}j" if m["days_to_target"] is not None else "—"
            status = "FAIL DD✗" if m["failed"] else ("OK" if m["tot_r"] > 0 else "négatif")
            if m["failed"] or m["tot_r"] <= 0:
                all_ok = False
            print(f"  {label:<10} {m['n_taken']:>5} {m['n_blocked']:>5}  "
                  f"{m['w']:>3} {m['l']:>3} {wr:>5.1f}%  {m['tot_r']:>+8.2f}  "
                  f"{pnl:>+9.0f}  {m['max_dd_pct']:>5.2f}%  {m['worst_day_pct']:>5.2f}%  "
                  f"{m['n_trade_days']:>5}  {tgt:>6}  {status}")

        print("  " + "─" * (_W - 2))
        print(f"  VERDICT : {'PASS ✓' if all_ok else 'FAIL ✗'} "
              f"(toutes périodes positives, zéro breach DD)")

    print(f"\n{'═' * _W}\n  Fin\n{'═' * _W}\n")


if __name__ == "__main__":
    main()
