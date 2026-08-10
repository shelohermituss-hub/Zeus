"""
Sweep de paramètres — ORB Open US (voir open_us_orb_strategy.py), XAUUSD M1.

Même méthodologie de robustesse que run_open_us_sweep.py : cherche des
configs profitables sur 2024 ET 2025 ET 2026 H1 séparément, pas seulement
en agrégat.

Usage
-----
    python -m zeus.backtest.run_open_us_orb_sweep
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv
from zeus.backtest.sd_simulation import compute_metrics, simulate_all
from zeus.strategy.open_us_orb_strategy import OpenUSORBStrategy

_ROOT   = Path(__file__).parent.parent.parent
_M1_DIR = _ROOT / "data" / "historical" / "xauusd" / "m1"
_OUT    = _ROOT / "data" / "results" / "open_us_orb_sweep.json"

INITIAL_BALANCE = 10_000.0
RISK_PCT        = 0.01

PERIODS = {
    "2024": ["DAT_MT_XAUUSD_M1_2024.csv"],
    "2025": ["DAT_MT_XAUUSD_M1_2025.csv"],
    "2026H1": [f"DAT_MT_XAUUSD_M1_2026{m:02d}.csv" for m in range(1, 7)],
}

OR_MINUTES      = [5, 15, 30, 60]
SEARCH_MINUTES  = [30, 60, 120]
MIN_RANGE_USD   = [0.0, 2.0, 4.0]
RISK_REWARDS    = [1.0, 1.5, 2.0, 3.0]


def _load(period: str) -> pd.DataFrame:
    frames = [parse_histdata_csv(_M1_DIR / f) for f in PERIODS[period]]
    return pd.concat(frames).sort_index()


def main() -> None:
    dfs = {p: _load(p) for p in PERIODS}
    for p, df in dfs.items():
        print(f"  {p}: {len(df):,} barres M1")

    raw_signals: dict[tuple, list] = {}
    t0 = time.time()
    for or_min in OR_MINUTES:
        for search_min in SEARCH_MINUTES:
            for min_rng in MIN_RANGE_USD:
                for period, df in dfs.items():
                    strat = OpenUSORBStrategy(
                        or_minutes=or_min, search_minutes=search_min,
                        min_range_price=min_rng, risk_reward=1.0,
                    )
                    sigs = strat.generate_signals(df)
                    raw_signals[(or_min, search_min, min_rng, period)] = sigs
    print(f"Total scan time: {time.time()-t0:.0f}s")

    rows = []
    for (or_min, search_min, min_rng, period), sigs in raw_signals.items():
        if not sigs:
            continue
        df = dfs[period]
        for rr in RISK_REWARDS:
            variant_sigs = [dataclasses.replace(s, risk_reward=rr) for s in sigs]
            results, n_expired = simulate_all(
                variant_sigs, df, risk_pct=RISK_PCT, initial_equity=INITIAL_BALANCE,
            )
            m = compute_metrics(results, INITIAL_BALANCE, len(variant_sigs), n_expired)
            rows.append(dict(
                or_minutes=or_min, search_minutes=search_min, min_range_usd=min_rng,
                period=period, risk_reward=rr,
                n_trades=m["n_trades"], win_rate=m["win_rate"],
                total_r=m["total_r"], total_usd=m["total_usd"], max_dd=m["max_dd"],
            ))

    with open(_OUT, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n{len(rows)} lignes écrites dans {_OUT}")

    by_config: dict[tuple, dict[str, dict]] = {}
    for r in rows:
        cfg = (r["or_minutes"], r["search_minutes"], r["min_range_usd"], r["risk_reward"])
        by_config.setdefault(cfg, {})[r["period"]] = r

    robust = []
    for cfg, per_period in by_config.items():
        if set(per_period) != set(PERIODS):
            continue
        n_trades_min = min(v["n_trades"] for v in per_period.values())
        if n_trades_min < 10:
            continue
        all_positive = all(v["total_r"] > 0 for v in per_period.values())
        total_r_sum  = sum(v["total_r"] for v in per_period.values())
        robust.append((cfg, all_positive, total_r_sum, n_trades_min, per_period))

    robust.sort(key=lambda x: (-x[1], -x[2]))

    print("\n" + "=" * 100)
    print("  Top configs ORB (triées : positif sur les 3 périodes d'abord, puis R total)")
    print("=" * 100)
    print(f"  {'or_min':>6} {'search':>7} {'min_rng':>7} {'RR':>5}  "
          f"{'2024 R':>8} {'2025 R':>8} {'2026H1 R':>9}  {'min_trades':>10}  {'all+':>5}")
    for cfg, all_pos, total_r_sum, n_trades_min, per_period in robust[:25]:
        or_min, search_min, min_rng, rr = cfg
        r24 = per_period.get("2024", {}).get("total_r", float("nan"))
        r25 = per_period.get("2025", {}).get("total_r", float("nan"))
        r26 = per_period.get("2026H1", {}).get("total_r", float("nan"))
        print(f"  {or_min:>6} {search_min:>7} {min_rng:>7.1f} {rr:>5.1f}  "
              f"{r24:>+8.1f} {r25:>+8.1f} {r26:>+9.1f}  {n_trades_min:>10}  {'YES' if all_pos else '':>5}")


if __name__ == "__main__":
    main()
