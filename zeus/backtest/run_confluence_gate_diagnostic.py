"""
Diagnostic de rejet — ConfluenceScalpStrategy (variante CORE) sur ticks 1s.

Compte, gate par gate, le nombre de barres rejetées pour chaque raison,
sur un échantillon de quelques jours (rapide), pour identifier le vrai
goulot d'étranglement plutôt que de deviner.

Usage
-----
    python -m zeus.backtest.run_confluence_gate_diagnostic
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from zeus.backtest.data_loader import build_timeframes, load_m1_directory
from zeus.backtest.tick_loader import load_tick_directory, resample_ticks
from zeus.strategy.base import SignalType
from zeus.strategy.confluence import PatternGrade
from zeus.strategy.confluence_scalp_strategy import ConfluenceScalpStrategy

_ROOT    = Path(__file__).parent.parent.parent
M1_DIR   = _ROOT / "data" / "historical" / "xauusd" / "m1"
TICK_DIR = _ROOT / "data" / "ticks" / "xauusd" / "2026"


def main() -> None:
    df_m1 = load_m1_directory(M1_DIR, glob_pattern="DAT_MT_XAUUSD_M1_2026*.csv")
    tfs   = build_timeframes(df_m1)
    df_15m, df_1h, df_1d = tfs["15min"], tfs["1h"], tfs["1d"]

    ticks  = load_tick_directory(TICK_DIR, synthetic_spread=0.30)
    df_ltf = resample_ticks(ticks, freq="1s")["2026-02-02":"2026-02-13"]
    print(f"Échantillon : {len(df_ltf):,} barres 1s\n")

    strat = ConfluenceScalpStrategy(
        df_htf_1h=df_1h, df_mtf_15m=df_15m, df_daily=df_1d,
        sl_pips=20.0, max_sl_pips=30.0, pip_value=1.0,
        min_htf_score=6.0, min_grade=PatternGrade.C,
    )
    for attr in ("require_weekly_bias", "require_asian_sweep", "require_session_sweep",
                 "require_choch_candle", "require_pd_filter", "require_entry_pattern",
                 "require_ltf_sweep"):
        setattr(strat, f"_{attr}", False)

    reasons: Counter = Counter()
    n_signals = 0
    for i in range(len(df_ltf)):
        sig = strat.generate_signal(df_ltf, i)
        if sig.type == SignalType.NONE:
            # Bucket the reason to its stable prefix (many include dynamic values)
            reason = sig.reason.split(":")[0].split("(")[0].strip()
            reasons[reason] += 1
        else:
            n_signals += 1

    print(f"Signaux émis : {n_signals}\n")
    print(f"{'Raison de rejet':<45}{'Occurrences':>12}{'%':>8}")
    print("-" * 65)
    total = len(df_ltf)
    for reason, count in reasons.most_common(20):
        print(f"{reason:<45}{count:>12,}{count/total*100:>7.2f}%")


if __name__ == "__main__":
    main()
