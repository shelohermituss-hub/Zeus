"""
Sweep de paramètres — stratégie Open US (sweep + retournement), XAUUSD M1.

Exploration systématique, pas une conclusion : cherche des combinaisons
(window_minutes, variante WyckoffDetector, min_score, risk_reward) qui
sont profitables sur LES TROIS périodes séparément (2024, 2025, 2026 H1),
pas seulement en agrégat — un edge qui n'existe que sur une période n'est
pas un edge robuste (règle du projet : "ne jamais conclure qu'une
stratégie est bonne sans backtest réaliste").

Optimisation : la détection Wyckoff (coûteuse, ~1 min/année) n'est relancée
que pour chaque (window_minutes, variante détecteur) — min_score et
risk_reward sont appliqués a posteriori sur les signaux déjà détectés
(filtrage de score + dataclasses.replace du risk_reward), donc gratuits.

Usage
-----
    python -m zeus.backtest.run_open_us_sweep
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv
from zeus.backtest.sd_simulation import compute_metrics, simulate_all
from zeus.strategy.open_us_strategy import OpenUSStrategy
from zeus.strategy.supply_demand.wyckoff import WyckoffDetector

_ROOT   = Path(__file__).parent.parent.parent
_M1_DIR = _ROOT / "data" / "historical" / "xauusd" / "m1"
_OUT    = _ROOT / "data" / "results" / "open_us_sweep.json"

INITIAL_BALANCE = 10_000.0
RISK_PCT        = 0.01

PERIODS = {
    "2024": ["DAT_MT_XAUUSD_M1_2024.csv"],
    "2025": ["DAT_MT_XAUUSD_M1_2025.csv"],
    "2026H1": [f"DAT_MT_XAUUSD_M1_2026{m:02d}.csv" for m in range(1, 7)],
}

WINDOWS = [60, 90, 120]

# "default"  : seuils actuels de wyckoff.py (assouplis pour la volatilité M1 gold)
# "strict"   : seuils originaux avant assouplissement (voir commentaires wyckoff.py)
DETECTOR_VARIANTS = {
    "default": dict(),
    "strict": dict(
        accum_range_mult=3.0, mss_lookback=30,
        min_spring_sweep_pct=0.20, min_mss_strength_pct=0.10,
    ),
}

MIN_SCORES   = [0.0, 4.0, 5.0, 6.0, 7.0]
RISK_REWARDS = [1.5, 2.0, 2.5, 3.0, 4.0]


def _load(period: str) -> pd.DataFrame:
    frames = [parse_histdata_csv(_M1_DIR / f) for f in PERIODS[period]]
    return pd.concat(frames).sort_index()


def main() -> None:
    dfs = {p: _load(p) for p in PERIODS}
    for p, df in dfs.items():
        print(f"  {p}: {len(df):,} barres M1 ({df.index[0].date()} -> {df.index[-1].date()})")

    raw_signals: dict[tuple[str, str, str], list] = {}
    t0 = time.time()
    for window in WINDOWS:
        for variant_name, variant_kwargs in DETECTOR_VARIANTS.items():
            for period, df in dfs.items():
                key = (window, variant_name, period)
                t1 = time.time()
                detector = WyckoffDetector(**variant_kwargs)
                strat = OpenUSStrategy(detector=detector, window_minutes=window, min_score=0.0)
                sigs = strat.generate_signals(df)
                raw_signals[key] = sigs
                print(f"  scan window={window} variant={variant_name} period={period}: "
                      f"{len(sigs)} signaux bruts ({time.time()-t1:.1f}s)")
    print(f"Total scan time: {time.time()-t0:.0f}s")

    rows = []
    for (window, variant_name, period), sigs in raw_signals.items():
        df = dfs[period]
        for min_score in MIN_SCORES:
            filtered = [s for s in sigs if s.zone_score >= min_score]
            if not filtered:
                continue
            for rr in RISK_REWARDS:
                variant_sigs = [dataclasses.replace(s, risk_reward=rr) for s in filtered]
                results, n_expired = simulate_all(
                    variant_sigs, df, risk_pct=RISK_PCT, initial_equity=INITIAL_BALANCE,
                )
                m = compute_metrics(results, INITIAL_BALANCE, len(variant_sigs), n_expired)
                rows.append(dict(
                    window=window, variant=variant_name, period=period,
                    min_score=min_score, risk_reward=rr,
                    n_trades=m["n_trades"], win_rate=m["win_rate"],
                    total_r=m["total_r"], total_usd=m["total_usd"], max_dd=m["max_dd"],
                ))

    with open(_OUT, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n{len(rows)} lignes écrites dans {_OUT}")

    # ── Agrégation par config (sur les 3 périodes) ──────────────────────────
    by_config: dict[tuple, dict[str, dict]] = {}
    for r in rows:
        cfg = (r["window"], r["variant"], r["min_score"], r["risk_reward"])
        by_config.setdefault(cfg, {})[r["period"]] = r

    robust = []
    for cfg, per_period in by_config.items():
        if set(per_period) != set(PERIODS):
            continue
        n_trades_min = min(v["n_trades"] for v in per_period.values())
        if n_trades_min < 10:
            continue   # échantillon trop petit pour être significatif
        all_positive = all(v["total_r"] > 0 for v in per_period.values())
        total_r_sum  = sum(v["total_r"] for v in per_period.values())
        robust.append((cfg, all_positive, total_r_sum, n_trades_min, per_period))

    robust.sort(key=lambda x: (-x[1], -x[2]))

    print("\n" + "=" * 100)
    print("  Top configs (triées : positif sur les 3 périodes d'abord, puis R total)")
    print("=" * 100)
    print(f"  {'window':>6} {'variant':>8} {'min_sc':>7} {'RR':>5}  "
          f"{'2024 R':>8} {'2025 R':>8} {'2026H1 R':>9}  {'min_trades':>10}  {'all+':>5}")
    for cfg, all_pos, total_r_sum, n_trades_min, per_period in robust[:25]:
        window, variant, min_score, rr = cfg
        r24 = per_period.get("2024", {}).get("total_r", float("nan"))
        r25 = per_period.get("2025", {}).get("total_r", float("nan"))
        r26 = per_period.get("2026H1", {}).get("total_r", float("nan"))
        print(f"  {window:>6} {variant:>8} {min_score:>7.1f} {rr:>5.1f}  "
              f"{r24:>+8.1f} {r25:>+8.1f} {r26:>+9.1f}  {n_trades_min:>10}  {'YES' if all_pos else '':>5}")


if __name__ == "__main__":
    main()
