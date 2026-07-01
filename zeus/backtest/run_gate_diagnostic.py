"""
Diagnostic de fréquence — TF1 (1H→15M→1M).

Passe sur chaque barre M1 d'avril 2026, appelle generate_signal() et compte
les rejets gate par gate. Permet d'identifier précisément le bottleneck.

Usage
-----
    python -m zeus.backtest.run_gate_diagnostic [april2026|2026q2|2024]
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from zeus.backtest.data_loader import build_timeframes, load_m1_directory
from zeus.backtest.partial_close import PartialCloseConfig
from zeus.strategy.scalp_strategy import ScalpSMCStrategy
from zeus.utils.logger import setup_logger

_ROOT    = Path(__file__).parent.parent.parent
DATA_DIR = _ROOT / "data" / "historical" / "xauusd" / "m1"

PERIODS = {
    "april2026": ("2026-04-01", "2026-04-30"),
    "2026q2":    ("2026-04-01", "2026-06-30"),
    "2024":      ("2024-01-01", "2024-12-31"),
}

# Ordre des gates (pour affichage trié)
GATE_ORDER = [
    "outside killzone",
    "insufficient HTF data",
    "no HTF swing bias",
    "daily bias",
    "1H MSS not confirmed",
    "price outside HTF zone",
    "zone not in allowed_zones",
    "dirty approach",
    "HTF score below threshold",
    "LTF entry not confirmed",
    "LTF sweep not confirmed",
    "SL too wide",
    "daily signal limit",
    "session limit",
    "SIGNAL",
]


def _gate_key(reason: str) -> str:
    """Normalise la raison en clé de gate."""
    r = reason.lower()
    if "killzone"       in r: return "outside killzone"
    if "insufficient"   in r: return "insufficient HTF data"
    if "swing bias"     in r: return "no HTF swing bias"
    if "daily bias"     in r: return "daily bias"
    if "mss"            in r: return "1H MSS not confirmed"
    if "outside htf"    in r: return "price outside HTF zone"
    if "allowed_zones"  in r: return "zone not in allowed_zones"
    if "dirty approach" in r: return "dirty approach"
    if "htf score"      in r: return "HTF score below threshold"
    if "ltf entry"      in r: return "LTF entry not confirmed"
    if "ltf sweep"      in r: return "LTF sweep not confirmed"
    if "sl too wide"    in r: return "SL too wide"
    if "daily signal"   in r: return "daily signal limit"
    if "session limit"  in r: return "session limit"
    return reason  # inattendu


def _make_strategy(tfs: dict) -> ScalpSMCStrategy:
    return ScalpSMCStrategy(
        df_htf_1h=tfs["1h"],
        df_mtf_15m=tfs["15min"],
        df_daily=tfs["1d"],
        sl_pips=20.0,
        max_sl_pips=30.0,
        pip_value=1.0,
        min_htf_score=3.0,
        swing_length=20,
        internal_length=5,
        atr_period=100,
        ltf_lookback=30,
        killzone_only=True,
        require_entry_fvg=True,
        require_choch_candle=False,
        allowed_zones=["OB", "OTE"],
        require_clean_approach=True,
        approach_lookback=5,
        approach_max_momentum=0.6,
        approach_max_body_atr=1.5,
        require_poc_zone_confluence=False,
        require_session_sweep=False,
        require_ltf_sweep=True,
        ltf_sweep_lookback=3,
        require_ote=False,
        require_daily_bias=False,
        require_entry_pattern=False,
        max_daily_signals=5,
        max_signals_per_session=2,
        mss_lookback=20,
    )


def run_diagnostic(period_id: str) -> None:
    setup_logger("ERROR")
    start, end = PERIODS[period_id]

    print(f"\nLoading M1 data …")
    df_m1 = load_m1_directory(DATA_DIR)
    tfs   = build_timeframes(df_m1)

    df_slice = df_m1[start:end]
    n_bars   = len(df_slice)
    print(f"  Période : {start} → {end}  ({n_bars:,} barres M1)\n")

    strat = _make_strategy(tfs)

    gate_counts: Counter[str] = Counter()
    # Compte les barres qui arrivent à chaque gate (surviving)
    survivors: dict[str, int] = defaultdict(int)

    for i, ts in enumerate(df_slice.index):
        global_i = df_m1.index.get_loc(ts)
        from zeus.strategy.base import SignalType
        sig = strat.generate_signal(df_m1, global_i)
        if sig.type != SignalType.NONE:
            gate_counts["SIGNAL"] += 1
        else:
            gate_counts[_gate_key(sig.reason)] += 1

    total = sum(gate_counts.values())

    # ── Funnel progressif ────────────────────────────────────────────────
    print(f"{'═'*70}")
    print(f"  FUNNEL DE REJECTION — TF1 (1H→15M→1M) — {period_id.upper()}")
    print(f"{'═'*70}")
    print(f"  {'Gate / Raison':<35}  {'Rejets':>7}  {'% total':>7}  {'cum% rejected':>14}")
    print(f"{'─'*70}")

    cumulative_rejected = 0
    for gate in GATE_ORDER:
        n = gate_counts.get(gate, 0)
        if n == 0:
            continue
        pct  = 100 * n / total
        cumulative_rejected += n
        cum_pct = 100 * cumulative_rejected / total
        prefix = "  ★ " if gate == "SIGNAL" else "    "
        print(f"{prefix}{gate:<35}  {n:>7,}  {pct:>6.2f}%  {cum_pct:>13.2f}%")

    # Clés inattendues
    for gate, n in gate_counts.items():
        if gate not in GATE_ORDER:
            pct = 100 * n / total
            print(f"  ? {gate:<33}  {n:>7,}  {pct:>6.2f}%")

    print(f"{'─'*70}")
    print(f"  {'TOTAL barres M1':<35}  {total:>7,}")
    n_sig = gate_counts.get("SIGNAL", 0)
    print(f"  {'→ Signaux émis':<35}  {n_sig:>7,}  ({100*n_sig/total:.3f}%)")
    print(f"{'═'*70}")

    # ── Analyse par gate ──────────────────────────────────────────────────
    print(f"\n  ANALYSE — top bottlenecks (hors killzone)")
    print(f"{'─'*50}")
    inside_kz = total - gate_counts.get("outside killzone", 0)
    inside_kz -= gate_counts.get("insufficient HTF data", 0)
    remaining = inside_kz
    for gate in GATE_ORDER[2:]:   # skip killzone + insuff data
        n = gate_counts.get(gate, 0)
        if n == 0 or gate == "SIGNAL":
            continue
        pct = 100 * n / remaining if remaining else 0
        print(f"    {gate:<33}  rejet {pct:>5.1f}% des barres en killzone")
        remaining -= n
    print(f"{'─'*50}")
    print(f"    {'Signaux sur barres en KZ':<33}  {100*n_sig/inside_kz:.3f}%" if inside_kz else "")
    print(f"{'═'*70}\n")


def main() -> None:
    period_id = sys.argv[1] if len(sys.argv) > 1 else "april2026"
    if period_id not in PERIODS:
        print(f"Période inconnue. Choix : {list(PERIODS)}")
        sys.exit(1)
    run_diagnostic(period_id)


if __name__ == "__main__":
    main()
