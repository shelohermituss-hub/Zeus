"""
S&D Strategy — RR / TP Grid Sweep
==================================

Sweeps combinations of (tp1_r, tp1_size, rr_final) on 2024 data (optimisation),
then cross-validates the top configs on 2023 data (validation).

Grid:
  tp1_r     : None (no partial), 0.50, 0.75, 1.00  — R-level for partial close
  tp1_size  : 0.33, 0.50                            — fraction closed at TP1
  rr_final  : 1.00, 1.25, 1.50, 2.00, 3.00         — final TP in R multiples

Constraint: tp1_r < rr_final (partial must come before final TP).

Signals are generated once per pair/year with the frozen strategy params, then
signal.risk_reward is replaced per config using dataclasses.replace() — no
re-running the expensive strategy.run() call per RR variant.

Usage
-----
    python -m zeus.backtest.run_sd_rr_sweep
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv, resample_ohlcv
from zeus.backtest.sd_simulation import compute_metrics, simulate_all
from zeus.strategy.supply_demand.sd_strategy import SDStrategy
from zeus.strategy.supply_demand.wyckoff import WyckoffDetector
from zeus.strategy.supply_demand.zone_detector import ZoneDetector

_ROOT = Path(__file__).parent.parent.parent
_DATA = _ROOT / "data" / "historical"

INITIAL_BALANCE = 10_000.0
RISK_PCT        = 0.005

PASS_WR  = 55.0
PASS_DD  = 8.0

# ── Paper cluster ─────────────────────────────────────────────────────────────
PAIRS_2024 = [
    ("GBPUSD", 0.0001, 1.00, (7, 17), [_DATA / "gbpusd" / "m1" / "DAT_MT_GBPUSD_M1_2024.csv"]),
    ("EURUSD", 0.0001, 1.00, (7, 17), [_DATA / "eurusd" / "m1" / "DAT_MT_EURUSD_M1_2024.csv"]),
    ("GBPAUD", 0.0002, 1.52, (7, 17), [_DATA / "gbpaud" / "m1" / "DAT_MT_GBPAUD_M1_2024.csv"]),
    ("EURNZD", 0.0002, 1.62, (7, 17), [_DATA / "eurnzd" / "m1" / "DAT_MT_EURNZD_M1_2024.csv"]),
]

PAIRS_2023 = [
    ("GBPUSD", 0.0001, 1.00, (7, 17), [_DATA / "gbpusd" / "m1" / "DAT_MT_GBPUSD_M1_2023.csv"]),
    ("EURUSD", 0.0001, 1.00, (7, 17), [_DATA / "eurusd" / "m1" / "DAT_MT_EURUSD_M1_2023.csv"]),
    ("GBPAUD", 0.0002, 1.52, (7, 17), [_DATA / "gbpaud" / "m1" / "DAT_MT_GBPAUD_M1_2023.csv"]),
    ("EURNZD", 0.0002, 1.62, (7, 17), [_DATA / "eurnzd" / "m1" / "DAT_MT_EURNZD_M1_2023.csv"]),
]

# ── Universal-D frozen strategy params ────────────────────────────────────────
_BASE_UNIVERSAL_D = dict(
    risk_reward             = 1.25,   # overridden per config via signal patching
    min_zone_score          = 4.0,
    min_wyckoff_score       = 7.0,
    min_wyckoff_score_short = 5.9,
    min_composite_score     = 4.0,
    signal_cooldown         = 10,
    use_trend_filter        = True,
    trend_slope_lookback    = 3,
    use_price_above_ema     = True,
    ema_atr_tolerance       = 0.5,
    use_session_filter      = True,
    max_signals_per_day     = 6,
    use_adx_filter          = False,
    use_rsi_filter          = False,
    min_sl_pips             = 5,
    pip_size                = 0.0001,
    min_score_product       = 0.0,
    use_h4_trend_filter     = False,
    use_first_touch_only    = False,
)

_BASE_WYCKOFF = dict(
    lookback             = 200,
    max_accum_bars       = 20,
    accum_range_mult     = 6.0,
    mss_lookback         = 60,
    min_spring_sweep_pct = 0.10,
    min_mss_strength_pct = 0.03,
)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RRConfig:
    tp1_r:    float | None  # None = no partial close
    tp1_size: float         # fraction closed at TP1 (0 if no partial)
    rr:       float         # final TP in R

    @property
    def label(self) -> str:
        if self.tp1_r is None:
            return f"noTP1  RR={self.rr:.2f}"
        pct = int(round(self.tp1_size * 100))
        return f"TP1@{self.tp1_r:.2f}R({pct}%)  RR={self.rr:.2f}"


def _build_grid() -> list[RRConfig]:
    rr_levels  = [1.00, 1.25, 1.50, 2.00, 3.00]
    tp1_levels = [0.50, 0.75, 1.00]
    tp1_sizes  = [0.33, 0.50]

    configs: list[RRConfig] = []

    for rr in rr_levels:
        configs.append(RRConfig(tp1_r=None, tp1_size=0.0, rr=rr))

    for tp1_r, tp1_size, rr in product(tp1_levels, tp1_sizes, rr_levels):
        if tp1_r < rr:
            configs.append(RRConfig(tp1_r=tp1_r, tp1_size=tp1_size, rr=rr))

    return configs


# ── Data / signal cache ───────────────────────────────────────────────────────
_M1_CACHE:     dict[str, pd.DataFrame] = {}
_SIGNAL_CACHE: dict[str, list]         = {}


def _load_m1(sym: str, files: list[Path]) -> pd.DataFrame | None:
    if sym in _M1_CACHE:
        return _M1_CACHE[sym]
    frames = [parse_histdata_csv(f) for f in files if f.exists()]
    if not frames:
        _M1_CACHE[sym] = None
        return None
    m1_df = pd.concat(frames).sort_index()
    m1_df = m1_df[~m1_df.index.duplicated(keep="first")]
    _M1_CACHE[sym] = m1_df
    return m1_df


def _get_signals(sym: str, session: tuple[int, int], files: list[Path]) -> list:
    cache_key = f"{sym}_{session[0]}-{session[1]}"
    if cache_key in _SIGNAL_CACHE:
        return _SIGNAL_CACHE[cache_key]

    m1_df = _load_m1(sym, files)
    if m1_df is None:
        _SIGNAL_CACHE[cache_key] = []
        return []

    m15_df = resample_ohlcv(m1_df, "15min")
    if len(m15_df) < 50:
        _SIGNAL_CACHE[cache_key] = []
        return []

    params = {**_BASE_UNIVERSAL_D, "session_start_utc": session[0], "session_end_utc": session[1]}
    strategy = SDStrategy(
        zone_detector    = ZoneDetector(),
        wyckoff_detector = WyckoffDetector(**_BASE_WYCKOFF),
        **params,
    )
    sigs = strategy.run(m15_df, m1_df)
    _SIGNAL_CACHE[cache_key] = sigs
    return sigs


def _patch_signals(signals: list, rr: float) -> list:
    """Return a copy of signals with risk_reward replaced by rr."""
    return [dataclasses.replace(s, risk_reward=rr) for s in signals]


def _run_config_combined(pairs: list, cfg: RRConfig) -> dict:
    all_results: list = []
    all_signals = 0
    n_expired   = 0
    equity      = INITIAL_BALANCE

    for sym, spread, q2u, session, files in pairs:
        m1_df = _load_m1(sym, files)
        if m1_df is None:
            continue

        raw_signals = _get_signals(sym, session, files)
        all_signals += len(raw_signals)
        if not raw_signals:
            continue

        signals = _patch_signals(raw_signals, cfg.rr)

        tp1_r    = cfg.tp1_r    if cfg.tp1_r is not None else 0.0
        tp1_size = cfg.tp1_size if cfg.tp1_r is not None else 0.0

        results, exp = simulate_all(
            signals,
            m1_df,
            risk_pct           = RISK_PCT,
            spread             = spread,
            max_daily_losses   = 1,
            max_monthly_losses = 4,
            tp1_r              = tp1_r,
            tp1_size           = tp1_size,
            initial_equity     = equity,
            quote_to_usd_rate  = q2u,
        )
        all_results.extend(results)
        n_expired += exp
        for r in results:
            equity += r.pnl_usd

    if not all_results:
        return {"n_trades": 0, "win_rate": 0.0, "total_r": 0.0, "max_dd": 0.0}

    return compute_metrics(all_results, INITIAL_BALANCE, all_signals, n_expired)


def _pass(m: dict) -> bool:
    return (m.get("win_rate", 0.0) >= PASS_WR
            and m.get("total_r",  0.0) > 0
            and m.get("max_dd",   0.0) <= PASS_DD)


def _score(m: dict) -> float:
    tr = m.get("total_r", 0.0)
    wr = m.get("win_rate", 0.0)
    dd = m.get("max_dd",   0.0)
    return tr * (wr / 100) / (1 + dd / 10) if m.get("n_trades", 0) > 0 else 0.0


def _print_phase(rows: list[tuple[RRConfig, dict]], show_score: bool = False) -> None:
    hdr = f"{'Config':<32} {'N':>4} {'WR%':>6} {'TotalR':>8} {'DD%':>5}"
    if show_score:
        hdr += f" {'Score':>7}"
    print(hdr)
    print("─" * len(hdr))
    for cfg, m in rows:
        n  = m.get("n_trades", 0)
        wr = m.get("win_rate",  0.0)
        tr = m.get("total_r",  0.0)
        dd = m.get("max_dd",   0.0)
        ok = "✅" if _pass(m) else "❌"
        line = f"{cfg.label:<32} {n:>4} {wr:>6.1f} {tr:>8.2f} {dd:>5.1f}"
        if show_score:
            line += f" {_score(m):>7.2f}"
        print(line + f" {ok}")


def main() -> None:
    configs = _build_grid()

    print("\n" + "=" * 76)
    print("S&D FOREX — RR / TP Grid Sweep")
    print(f"{len(configs)} configurations | Optimise 2024 → Valide 2023")
    print("=" * 76)

    # ── Phase 1: sweep on 2024 ────────────────────────────────────────────────
    print("\nPhase 1 — Optimisation sur 2024 (tri par Total R)")
    print("─" * 76)

    global _M1_CACHE, _SIGNAL_CACHE
    _M1_CACHE     = {}
    _SIGNAL_CACHE = {}

    results_2024: list[tuple[RRConfig, dict]] = []
    for cfg in configs:
        m = _run_config_combined(PAIRS_2024, cfg)
        results_2024.append((cfg, m))

    results_2024.sort(key=lambda x: x[1].get("total_r", -999), reverse=True)
    _print_phase(results_2024, show_score=True)

    # ── Phase 2: cross-validate on 2023 ──────────────────────────────────────
    passing_2024 = [(cfg, m) for cfg, m in results_2024 if _pass(m)]
    top_n        = min(15, len(passing_2024))
    top_configs  = passing_2024[:top_n]

    print(f"\n{'=' * 76}")
    print(f"Phase 2 — Cross-validation 2023 (top {top_n} qui passent 2024)")
    print("─" * 76)
    print(f"{'Config':<32} {'N':>4} {'WR%':>6} {'TotalR':>8} {'DD%':>5} "
          f"{'2024':>5} {'2023':>5} {'R moy':>7}")
    print("─" * 76)

    _M1_CACHE     = {}
    _SIGNAL_CACHE = {}

    cross: list[tuple[RRConfig, dict, dict]] = []
    for cfg, m24 in top_configs:
        m23 = _run_config_combined(PAIRS_2023, cfg)
        cross.append((cfg, m24, m23))

    cross.sort(
        key=lambda x: (int(_pass(x[1]) and _pass(x[2])),
                       (x[1].get("total_r", 0) + x[2].get("total_r", 0)) / 2),
        reverse=True,
    )

    for cfg, m24, m23 in cross:
        n  = m23.get("n_trades", 0)
        wr = m23.get("win_rate",  0.0)
        tr = m23.get("total_r",  0.0)
        dd = m23.get("max_dd",   0.0)
        ok24 = "✅" if _pass(m24) else "❌"
        ok23 = "✅" if _pass(m23) else "❌"
        avg  = (m24.get("total_r", 0) + m23.get("total_r", 0)) / 2
        print(f"{cfg.label:<32} {n:>4} {wr:>6.1f} {tr:>8.2f} {dd:>5.1f} "
              f"{ok24:>5} {ok23:>5} {avg:>7.2f}")

    # ── Final summary ─────────────────────────────────────────────────────────
    both = [(cfg, m24, m23) for cfg, m24, m23 in cross if _pass(m24) and _pass(m23)]
    print(f"\n{'=' * 76}")
    print(f"  {len(both)} config(s) validé(s) sur 2024 ET 2023")
    if both:
        print(f"\n  TOP 5 configs robustes (classés par R moyen 2023+2024) :")
        print(f"  {'Config':<32}  {'R moy':>7}  {'2024 R/WR%':>14}  {'2023 R/WR%':>14}")
        print("  " + "─" * 72)
        for i, (cfg, m24, m23) in enumerate(both[:5], 1):
            avg = (m24.get("total_r", 0) + m23.get("total_r", 0)) / 2
            r24 = f"{m24.get('total_r',0):.2f}/{m24.get('win_rate',0):.1f}%"
            r23 = f"{m23.get('total_r',0):.2f}/{m23.get('win_rate',0):.1f}%"
            print(f"  {i}. {cfg.label:<32}  {avg:>7.2f}  {r24:>14}  {r23:>14}")

    # ── Baseline reference ────────────────────────────────────────────────────
    baseline_cfg = RRConfig(tp1_r=0.75, tp1_size=0.50, rr=1.25)
    print(f"\n  Référence BASELINE actuelle : {baseline_cfg.label}")
    bl24 = next((m for c, m in results_2024 if c == baseline_cfg), None)
    if bl24:
        print(f"    2024 : {bl24.get('total_r',0):.2f}R  WR={bl24.get('win_rate',0):.1f}%  DD={bl24.get('max_dd',0):.1f}%")
    bl23_entry = next(((c, m24, m23) for c, m24, m23 in cross if c == baseline_cfg), None)
    if bl23_entry:
        _, _, m23 = bl23_entry
        print(f"    2023 : {m23.get('total_r',0):.2f}R  WR={m23.get('win_rate',0):.1f}%  DD={m23.get('max_dd',0):.1f}%")
    print()


if __name__ == "__main__":
    main()
