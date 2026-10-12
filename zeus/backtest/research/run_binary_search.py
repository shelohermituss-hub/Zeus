"""
Recherche élargie options binaires (payout 92 %, écart d'exécution 0,2 pip).

Six familles de stratégies × timeframes M5 / M15 / H1 × expirations de 1 à 3 bougies,
sur tous les instruments disponibles (majeures, croisées, or).

Protocole en trois temps, pour qu'un win rate élevé ne soit pas un artefact de sélection :
1. Entraînement (2023-2024) : classement des configurations ; présélection si
   n >= MIN_TRAIN_TRADES, borne basse Wilson 95 % > seuil, et seuil battu sur
   >= 70 % des jeux.
2. Validation (2025, jamais vue) : la présélection doit y battre le seuil
   (borne basse > seuil, n >= MIN_VALIDATION_TRADES).
3. Hors échantillon final (2026) : rapporté, jamais utilisé pour choisir.

Usage
-----
    python -m zeus.backtest.research.run_binary_search
"""
from __future__ import annotations

import itertools
import json
import time
from multiprocessing import Pool
from pathlib import Path

from zeus.backtest.binary_engine import BinaryBacktestConfig, BinaryOutcome, run_binary_backtest
from zeus.backtest.binary_evaluation import Tally, load_timeframe, pip_size, pool
from zeus.risk.binary_risk import BinaryRiskConfig, BinaryRiskManager
from zeus.strategy.binary_options import (
    donchian, exhaustion, keltner_stretch, mean_reversion, momentum, rsi_pullback, to_signals,
)

_ROOT = Path(__file__).parents[3]
_DATA = _ROOT / "data" / "historical"
_OUT = _ROOT / "data" / "results" / "binary_search.json"

PAYOUT = 0.92
BREAKEVEN = 1 / (1 + PAYOUT)
SLIPPAGE_PIPS = 0.2
TIMEFRAMES = [5, 15, 60]
EXPIRY_BARS = [1, 2, 3]
MIN_TRAIN_TRADES = 300
MIN_VALIDATION_TRADES = 100
MIN_SHARE_ABOVE_BREAKEVEN = 0.7

STUDY_RISK = BinaryRiskConfig(
    max_daily_loss_fraction=0.99, max_drawdown_fraction=0.99,
    max_consecutive_losses=10**6, max_open_trades=1,
)


def variants() -> list[tuple[str, object, object]]:
    v = []
    for std, (lo, hi) in itertools.product([2.0, 2.5, 3.0], [(25, 75), (15, 85)]):
        v.append((f"bb_rsi_std{std}_rsi{lo}-{hi}", mean_reversion,
                  mean_reversion.MeanReversionConfig(bb_std=std, rsi_low=lo, rsi_high=hi,
                                                     session_start_utc=0, session_end_utc=24)))
    for (lo, hi), trend in itertools.product([(2, 98), (5, 95), (10, 90)], [0, 50, 200]):
        v.append((f"rsi2_{lo}-{hi}_trend{trend}", rsi_pullback,
                  rsi_pullback.RsiPullbackConfig(rsi_low=lo, rsi_high=hi, trend_period=trend)))
    for streak, move in itertools.product([4, 5, 6, 7], [0.0, 1.5]):
        v.append((f"exhaust_streak{streak}_move{move}", exhaustion,
                  exhaustion.ExhaustionConfig(streak=streak, min_move_atr=move)))
    for mult, rej in itertools.product([1.5, 2.0, 2.5, 3.0], [False, True]):
        v.append((f"keltner_{mult}atr_rej{int(rej)}", keltner_stretch,
                  keltner_stretch.KeltnerStretchConfig(atr_mult=mult, require_rejection=rej)))
    for period, mode in itertools.product([20, 55], ["breakout", "fade"]):
        v.append((f"donchian{period}_{mode}", donchian, donchian.DonchianConfig(period=period, mode=mode)))
    for streak in (2, 3):
        v.append((f"momentum_streak{streak}", momentum,
                  momentum.MomentumConfig(streak=streak, min_body_atr=0.6, session_start_utc=0, session_end_utc=24)))
    return v


def split_of(tag: str) -> str:
    if tag.startswith(("2023", "2024")):
        return "train"
    return "validation" if tag.startswith("2025") else "holdout"


def datasets() -> list[tuple[str, str, Path]]:
    out = []
    for path in sorted(_DATA.glob("*/m1/*.csv")):
        symbol, tag = path.parts[-3], path.stem.rsplit("_", 1)[-1]
        out.append((f"{symbol}_{tag}", split_of(tag), path))
    return out


def evaluate_dataset(item: tuple[str, str, Path]) -> tuple[str, str, dict]:
    """Toutes les configurations sur un jeu → {clé de config: (gains, pertes)}."""
    name, split, path = item
    symbol = name.split("_")[0]
    tallies = {}
    for tf in TIMEFRAMES:
        df = load_timeframe(path, tf)
        for vname, module, cfg in variants():
            signals = to_signals(module.directions(df, cfg), vname)
            for bars in EXPIRY_BARS:
                bt = BinaryBacktestConfig(payout=PAYOUT, bar_minutes=tf, expiry_minutes=tf * bars,
                                          entry_slippage=SLIPPAGE_PIPS * pip_size(symbol))
                res = run_binary_backtest(df, signals, bt, BinaryRiskManager(STUDY_RISK, 1e9))
                wins = sum(t.outcome is BinaryOutcome.WIN for t in res.trades)
                losses = sum(t.outcome is BinaryOutcome.LOSS for t in res.trades)
                tallies[f"{vname}|tf{tf}|exp{bars}"] = (wins, losses)
    return name, split, tallies


def main() -> None:
    t0 = time.time()
    items = datasets()
    with Pool(4) as workers:
        per_dataset = workers.map(evaluate_dataset, items)
    print(f"{len(items)} jeux évalués en {time.time() - t0:.0f} s ; seuil {BREAKEVEN:.4f}\n")

    keys = per_dataset[0][2].keys()
    rows = []
    for key in keys:
        row = {"config": key}
        for split in ("train", "validation", "holdout"):
            row[split] = pool({n: Tally(*t[key]) for n, s, t in per_dataset if s == split}, BREAKEVEN)
        tr, va = row["train"], row["validation"]
        row["shortlisted"] = (tr["trades"] >= MIN_TRAIN_TRADES and tr["win_rate_lower_95"] > BREAKEVEN
                              and tr["share_above_breakeven"] >= MIN_SHARE_ABOVE_BREAKEVEN)
        row["validated"] = (row["shortlisted"] and va["trades"] >= MIN_VALIDATION_TRADES
                            and va["win_rate_lower_95"] > BREAKEVEN)
        rows.append(row)

    rows.sort(key=lambda r: r["train"]["win_rate_lower_95"], reverse=True)
    _OUT.write_text(json.dumps({"payout": PAYOUT, "breakeven": round(BREAKEVEN, 4),
                                "slippage_pips": SLIPPAGE_PIPS, "results": rows}, indent=1))

    def line(r):
        f = lambda s: f"n={r[s]['trades']:6d} wr={r[s]['win_rate']:.3f} lo={r[s]['win_rate_lower_95']:.3f}"
        flag = "VALIDÉ" if r["validated"] else ("présél." if r["shortlisted"] else "")
        return f"{r['config']:42s} train {f('train')} sh={r['train']['share_above_breakeven']:.2f} | val {f('validation')} | hold {f('holdout')} {flag}"

    print("Top 25 (borne basse entraînement) :")
    for r in rows[:25]:
        print(line(r))
    print(f"\nPrésélectionnées : {sum(r['shortlisted'] for r in rows)} / {len(rows)} ; "
          f"validées 2025 : {sum(r['validated'] for r in rows)}")
    for r in rows:
        if r["validated"]:
            print(line(r))
    print(f"\nRésultats : {_OUT.relative_to(_ROOT)}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
