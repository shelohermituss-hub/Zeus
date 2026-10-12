"""
Étude options binaires (Pocket Option, payout 92 %) — retour à la moyenne vs momentum.

Protocole :
- Entraînement : fichiers annuels M1 2023-2025 des majeures disponibles.
- Hors échantillon : fichiers juin 2026 (``*_202606.csv``), jamais utilisés pour choisir.
- Une grille courte par stratégie (12 variantes) × expirations 1/3/5 min.
- Coûts : entrée à la bougie suivante (latence) + slippage adverse de 0,2 pip.
- Mesure d'edge : risk manager relâché (pas d'arrêt journalier) mais 1 trade
  ouvert à la fois, comme en réel.

Critère de validation (sur l'entraînement) :
- borne basse de Wilson 95 % du win rate agrégé > seuil 1/(1+payout) ;
- win rate > seuil sur au moins 80 % des jeux (paire × année).
Avec 72 configurations testées, un passage par chance reste possible :
seul un résultat confirmé hors échantillon compte.

Usage
-----
    python -m zeus.backtest.research.run_binary_study
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import pandas as pd

from zeus.backtest.binary_engine import BinaryBacktestConfig, BinaryBacktestResult, run_binary_backtest
from zeus.backtest.data_loader import parse_m1_csv
from zeus.risk.binary_risk import BinaryRiskConfig, BinaryRiskManager
from zeus.strategy.binary_options import mean_reversion, momentum, to_signals

_ROOT = Path(__file__).parents[3]
_DATA = _ROOT / "data" / "historical"
_OUT = _ROOT / "data" / "results" / "binary_study.json"

PAIRS = ["eurusd", "gbpusd", "usdjpy", "audusd", "usdcad", "nzdusd", "usdchf"]
HOLDOUT_TAG = "202606"
PAYOUT = 0.92
EXPIRIES = [1, 3, 5]
SLIPPAGE_PIPS = 0.2
MIN_SHARE_OF_DATASETS_ABOVE_BREAKEVEN = 0.8

# Mesure d'edge : seuls les limites de concurrence/exposition restent actives.
STUDY_RISK = BinaryRiskConfig(
    max_daily_loss_fraction=0.99, max_drawdown_fraction=0.99,
    max_consecutive_losses=10**6, max_open_trades=1,
)


def variants() -> list[tuple[str, object, object]]:
    out = []
    for std, (lo, hi), (s, e) in itertools.product([2.0, 2.5, 3.0], [(25, 75), (15, 85)], [(0, 6), (0, 24)]):
        cfg = mean_reversion.MeanReversionConfig(bb_std=std, rsi_low=lo, rsi_high=hi,
                                                 session_start_utc=s, session_end_utc=e)
        out.append((f"mr_std{std}_rsi{lo}-{hi}_h{s}-{e}", mean_reversion, cfg))
    for streak, body, (s, e) in itertools.product([2, 3, 4], [0.3, 0.6], [(7, 16), (0, 24)]):
        cfg = momentum.MomentumConfig(streak=streak, min_body_atr=body, session_start_utc=s, session_end_utc=e)
        out.append((f"mo_streak{streak}_body{body}_h{s}-{e}", momentum, cfg))
    return out


def load_datasets() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    train, holdout = {}, {}
    for pair in PAIRS:
        for path in sorted((_DATA / pair / "m1").glob("*.csv")):
            tag = path.stem.rsplit("_", 1)[-1]
            target = holdout if tag == HOLDOUT_TAG else train
            target[f"{pair}_{tag}"] = parse_m1_csv(path)
    return train, holdout


def pip_size(dataset: str) -> float:
    return 0.01 if "jpy" in dataset else 0.0001


def run_one(df: pd.DataFrame, signals, dataset: str, expiry: int) -> BinaryBacktestResult:
    cfg = BinaryBacktestConfig(payout=PAYOUT, expiry_minutes=expiry,
                               entry_slippage=SLIPPAGE_PIPS * pip_size(dataset))
    return run_binary_backtest(df, signals, cfg, BinaryRiskManager(STUDY_RISK, 1_000_000.0))


def aggregate(results: dict[str, BinaryBacktestResult]) -> dict:
    pooled = BinaryBacktestResult(
        trades=[t for r in results.values() for t in r.trades],
        initial_balance=0.0, final_balance=0.0, payout=PAYOUT,
    )
    per_dataset = {name: round(r.win_rate, 4) for name, r in results.items()}
    above = sum(r.win_rate > r.breakeven_win_rate for r in results.values() if r.trades)
    with_trades = sum(1 for r in results.values() if r.trades)
    return {
        "trades": len(pooled.trades),
        "win_rate": round(pooled.win_rate, 4),
        "win_rate_lower_95": round(pooled.win_rate_lower_bound, 4),
        "share_above_breakeven": round(above / with_trades, 3) if with_trades else 0.0,
        "per_dataset_win_rate": per_dataset,
    }


def passes(stats: dict) -> bool:
    return (stats["win_rate_lower_95"] > 1 / (1 + PAYOUT)
            and stats["share_above_breakeven"] >= MIN_SHARE_OF_DATASETS_ABOVE_BREAKEVEN)


def main() -> None:
    t0 = time.time()
    train, holdout = load_datasets()
    print(f"Jeux d'entraînement : {sorted(train)}\nHors échantillon : {sorted(holdout)}")
    print(f"Seuil de rentabilité : {1 / (1 + PAYOUT):.4f}\n")

    rows = []
    for name, module, cfg in variants():
        sig_train = {k: to_signals(module.directions(df, cfg), name) for k, df in train.items()}
        sig_hold = {k: to_signals(module.directions(df, cfg), name) for k, df in holdout.items()}
        for expiry in EXPIRIES:
            tr = aggregate({k: run_one(train[k], s, k, expiry) for k, s in sig_train.items()})
            ho = aggregate({k: run_one(holdout[k], s, k, expiry) for k, s in sig_hold.items()})
            row = {"variant": name, "expiry_min": expiry, "train": tr, "holdout": ho, "passes_train": passes(tr)}
            rows.append(row)
            print(f"{name:34s} exp={expiry}  train n={tr['trades']:6d} wr={tr['win_rate']:.4f} "
                  f"lo95={tr['win_rate_lower_95']:.4f} share={tr['share_above_breakeven']:.2f}  "
                  f"holdout n={ho['trades']:5d} wr={ho['win_rate']:.4f}  {'PASS' if row['passes_train'] else ''}")

    rows.sort(key=lambda r: r["train"]["win_rate_lower_95"], reverse=True)
    _OUT.write_text(json.dumps({
        "payout": PAYOUT, "breakeven_win_rate": round(1 / (1 + PAYOUT), 4),
        "slippage_pips": SLIPPAGE_PIPS, "train_datasets": sorted(train),
        "holdout_datasets": sorted(holdout), "results": rows,
    }, indent=2))
    print(f"\n{sum(r['passes_train'] for r in rows)} configuration(s) passent le critère d'entraînement.")
    print(f"Résultats : {_OUT.relative_to(_ROOT)}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
