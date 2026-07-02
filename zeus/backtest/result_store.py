"""
Backtest result persistence — JSON save / load.

Every run_*.py script can optionally write its metrics to a JSON file.
A compare utility then diffs two run files to detect regressions.

Usage (in a run_*.py script)::

    from zeus.backtest.result_store import save_results, load_results, compare_runs

    save_results("results/harmonic_round2.json", run_meta, variant_rows)

    # Later:
    diff = compare_runs("results/harmonic_round1.json", "results/harmonic_round2.json")
    print_diff(diff)

JSON schema::

    {
      "meta": {
        "script": "run_harmonic_round2",
        "date":   "2026-07-02",
        "symbol": "XAUUSD",
        ...
      },
      "variants": [
        {
          "label":    "PIVOT7 RR1.0",
          "window":   "Q4 2024",
          "n_sig":    9,
          "win_rate": 66.7,
          "total_r":  4.00,
          "ev_r":     0.444,
          "max_dd":   1.1
        },
        ...
      ]
    }
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any


@dataclass
class VariantRow:
    """One row in the results table — one variant × one window."""
    label:    str
    window:   str
    n_sig:    int
    win_rate: float
    total_r:  float
    ev_r:     float
    max_dd:   float
    n_wins:   int   = 0
    n_losses: int   = 0


def save_results(
    path: str | Path,
    meta: dict[str, Any],
    variants: list[VariantRow],
) -> None:
    """
    Persist backtest results to *path* as JSON.

    Args:
        path:     Destination file (parent directory must exist).
        meta:     Free-form dict describing the run (script name, date, etc.).
        variants: List of VariantRow results.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta":     {**meta, "saved_at": str(date.today())},
        "variants": [asdict(v) for v in variants],
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_results(path: str | Path) -> dict[str, Any]:
    """Load a previously saved results JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_runs(
    baseline_path: str | Path,
    current_path:  str | Path,
) -> list[dict[str, Any]]:
    """
    Compare two run files.  Returns a list of diffs — one per variant×window
    combination that exists in both files.

    Each diff dict has keys: label, window, delta_wr, delta_total_r, delta_ev.
    Positive delta = current is better; negative = regression.
    """
    baseline = {
        (v["label"], v["window"]): v
        for v in load_results(baseline_path)["variants"]
    }
    current = {
        (v["label"], v["window"]): v
        for v in load_results(current_path)["variants"]
    }

    diffs: list[dict[str, Any]] = []
    for key in sorted(set(baseline) & set(current)):
        b, c = baseline[key], current[key]
        diffs.append({
            "label":        key[0],
            "window":       key[1],
            "base_wr":      b["win_rate"],
            "curr_wr":      c["win_rate"],
            "delta_wr":     round(c["win_rate"]  - b["win_rate"],  2),
            "delta_total_r": round(c["total_r"]  - b["total_r"],   2),
            "delta_ev":     round(c["ev_r"]       - b["ev_r"],      4),
        })
    return diffs


def print_diff(diffs: list[dict[str, Any]]) -> None:
    """Print a human-readable comparison table."""
    if not diffs:
        print("  (no common variants found)")
        return

    print(f"\n  {'Label':<30}  {'Window':<12}  "
          f"{'BaseWR':>7}  {'CurrWR':>7}  {'ΔWR':>6}  {'ΔR':>6}  {'ΔEV':>7}")
    print("  " + "─" * 84)
    for d in diffs:
        sign = "▲" if d["delta_wr"] > 0 else ("▼" if d["delta_wr"] < 0 else " ")
        print(
            f"  {d['label']:<30}  {d['window']:<12}  "
            f"{d['base_wr']:>6.1f}%  {d['curr_wr']:>6.1f}%  "
            f"{sign}{abs(d['delta_wr']):>5.1f}  "
            f"{d['delta_total_r']:>+5.2f}  "
            f"{d['delta_ev']:>+6.3f}"
        )
    print()
