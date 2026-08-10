"""
Test d'hypothèse — filtre de régime de volatilité sur les signaux Open US.

Constat des deux sweeps précédents (run_open_us_sweep.py / run_open_us_orb_sweep.py) :
2024 est systématiquement plus faible que 2025/2026H1, quelle que soit
l'approche (retournement Wyckoff ou breakout ORB) ou les paramètres. Hypothèse
testée ici : ce n'est pas un problème de paramétrage mais un effet de régime
(2024 moins directionnel après l'ouverture US) — un filtre qui ne trade que
lorsque la volatilité récente (ATR journalier) est au-dessus de sa médiane
glissante devrait réduire l'écart entre les périodes.

Le filtre utilise l'ATR de la veille (pas de la journée en cours) et un rang
percentile sur une fenêtre glissante de 60 jours — aucune anticipation
(look-ahead) : au moment où un signal se forme, seule l'information
disponible avant l'ouverture du jour est utilisée.

Usage
-----
    python -m zeus.backtest.run_open_us_regime_filter
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv
from zeus.backtest.sd_simulation import compute_metrics, simulate_all
from zeus.strategy.open_us_orb_strategy import OpenUSORBStrategy
from zeus.strategy.open_us_strategy import OpenUSStrategy
from zeus.strategy.supply_demand.wyckoff import WyckoffDetector

_ROOT   = Path(__file__).parent.parent.parent
_M1_DIR = _ROOT / "data" / "historical" / "xauusd" / "m1"

INITIAL_BALANCE = 10_000.0
RISK_PCT        = 0.01

PERIOD_FILES = {
    "2024":   ["DAT_MT_XAUUSD_M1_2024.csv"],
    "2025":   ["DAT_MT_XAUUSD_M1_2025.csv"],
    "2026H1": [f"DAT_MT_XAUUSD_M1_2026{m:02d}.csv" for m in range(1, 7)],
}


def _daily_atr_percentile(m1_df: pd.DataFrame, atr_period: int = 14, rank_window: int = 60) -> pd.Series:
    """Rang percentile (0-1) de l'ATR(14) journalier de la VEILLE, sur une
    fenêtre glissante de rank_window jours — indexé par date (pas datetime)."""
    daily = m1_df.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    prev_close = daily["close"].shift(1)
    tr = pd.concat([
        daily["high"] - daily["low"],
        (daily["high"] - prev_close).abs(),
        (daily["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    atr_prev = atr.shift(1)   # connu avant l'ouverture du jour
    pct_rank = atr_prev.rolling(rank_window).apply(lambda w: (w <= w[-1]).mean(), raw=True)
    pct_rank.index = pct_rank.index.date
    return pct_rank


def _filter_signals(signals: list, pct_rank: pd.Series, min_pct: float, max_pct: float) -> list:
    out = []
    for s in signals:
        d = s.formed_at.date()
        r = pct_rank.get(d)
        if r is None or pd.isna(r):
            continue
        if min_pct <= r <= max_pct:
            out.append(s)
    return out


def _report(label: str, signals: list, full_df: pd.DataFrame, dfs_by_period: dict[str, pd.DataFrame], risk_reward_override: float | None = None) -> dict[str, dict]:
    """Simule TOUJOURS contre full_df (les bar_index des signaux sont relatifs
    à full_df, pas aux DataFrames par-période) — seul le sous-ensemble de
    signaux (via formed_at) change par période. Utiliser un df par-période
    pour la simulation ferait pointer bar_index sur les mauvaises lignes."""
    out = {}
    for period, period_df in dfs_by_period.items():
        period_sigs = [s for s in signals if period_df.index[0] <= s.formed_at <= period_df.index[-1]]
        if risk_reward_override is not None:
            import dataclasses
            period_sigs = [dataclasses.replace(s, risk_reward=risk_reward_override) for s in period_sigs]
        results, n_expired = simulate_all(period_sigs, full_df, risk_pct=RISK_PCT, initial_equity=INITIAL_BALANCE)
        m = compute_metrics(results, INITIAL_BALANCE, len(period_sigs), n_expired)
        out[period] = m
    return out


def main() -> None:
    print("Chargement des données (2024-2026H1 continu, pour un ATR glissant réaliste)...")
    all_files = [f for files in PERIOD_FILES.values() for f in files]
    full_df = pd.concat([parse_histdata_csv(_M1_DIR / f) for f in all_files]).sort_index()
    full_df = full_df[~full_df.index.duplicated(keep="first")]
    print(f"  {len(full_df):,} barres M1, {full_df.index[0]} -> {full_df.index[-1]}")

    dfs_by_period = {}
    for label, files in PERIOD_FILES.items():
        dfs_by_period[label] = pd.concat([parse_histdata_csv(_M1_DIR / f) for f in files]).sort_index()

    pct_rank = _daily_atr_percentile(full_df)
    print(f"  ATR percentile calculé sur {pct_rank.notna().sum()} jours")

    configs = [
        ("Wyckoff window=120 default min_score=0 RR=3",
         OpenUSStrategy(detector=WyckoffDetector(), window_minutes=120, min_score=0.0, risk_reward=3.0).generate_signals(full_df),
         None),
        ("ORB or=30 search=60 min_rng=4 RR=3",
         OpenUSORBStrategy(or_minutes=30, search_minutes=60, min_range_price=4.0, risk_reward=3.0).generate_signals(full_df),
         None),
        ("ORB or=60 search=120 min_rng=4 RR=1.5",
         OpenUSORBStrategy(or_minutes=60, search_minutes=120, min_range_price=4.0, risk_reward=1.5).generate_signals(full_df),
         None),
    ]

    for name, all_signals, rr_override in configs:
        print("\n" + "=" * 100)
        print(f"  {name}  ({len(all_signals)} signaux bruts sur toute la période)")
        print("=" * 100)

        baseline = _report("baseline", all_signals, full_df, dfs_by_period, rr_override)
        print(f"  {'':>28} {'2024':>14} {'2025':>14} {'2026H1':>14}")
        def _line(tag, m):
            cells = []
            for p in ["2024", "2025", "2026H1"]:
                v = m[p]
                cells.append(f"R={v['total_r']:+.1f} n={v['n_trades']}")
            print(f"  {tag:>28} {cells[0]:>14} {cells[1]:>14} {cells[2]:>14}")
        _line("SANS FILTRE", baseline)

        for min_pct, max_pct, tag in [(0.5, 1.0, "haute vol (>=p50)"), (0.65, 1.0, "haute vol (>=p65)"),
                                        (0.0, 0.5, "basse vol (<=p50)")]:
            filtered = _filter_signals(all_signals, pct_rank, min_pct, max_pct)
            m = _report("filtered", filtered, full_df, dfs_by_period, rr_override)
            _line(tag, m)


if __name__ == "__main__":
    main()
