"""
Signal generation funnel diagnostic for the S&D strategy.

Traces exactly how many candidates are eliminated at each filter stage
so we can identify the primary bottleneck(s) limiting signal count.

Usage
-----
    python -m zeus.backtest.sd_signal_funnel
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv, resample_ohlcv
from zeus.strategy.supply_demand.pivot_candle import PivotSide
from zeus.strategy.supply_demand.sd_strategy import SDStrategy
from zeus.strategy.supply_demand.wyckoff import WyckoffDetector
from zeus.strategy.supply_demand.zone_detector import ZoneDetector

_ROOT    = Path(__file__).parent.parent.parent
_M1_FILE = _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_2025.csv"

INITIAL_BALANCE   = 10_000.0
RISK_PCT          = 0.01
RISK_REWARD       = 3.0
MIN_ZONE_SCORE    = 5.0
MIN_WYCKOFF_SCORE = 4.0
MIN_COMPOSITE     = 5.0
SIGNAL_COOLDOWN   = 30
SESSION_START     = 7
SESSION_END       = 21
TREND_SLOPE_LB    = 20
DAILY_CAP         = 2


def run_funnel(m1_df: pd.DataFrame, m15_df: pd.DataFrame) -> None:
    wd = WyckoffDetector(
        accum_range_mult     = 3.0,
        mss_lookback         = 10,
        min_spring_sweep_pct = 0.20,
        min_mss_strength_pct = 0.10,
    )
    zd = ZoneDetector()
    all_zones = zd.detect_zones(m15_df)

    m15_ema50 = m15_df["close"].ewm(span=50, adjust=False).mean()

    def _is_bullish(ts: pd.Timestamp) -> bool:
        pos = m15_ema50.index.searchsorted(ts, side="right") - 1
        if pos < TREND_SLOPE_LB:
            return False
        return float(m15_ema50.iloc[pos]) > float(m15_ema50.iloc[pos - TREND_SLOPE_LB])

    # ── Counters ──────────────────────────────────────────────────────────────
    n_m1_bars              = len(m1_df)
    n_bars_in_session      = 0
    n_zones_total          = len(all_zones)
    n_zone_formed          = 0
    n_zone_not_mitigated   = 0
    n_zone_score_ok        = 0
    n_price_in_zone        = 0
    n_trend_ok             = 0
    n_cooldown_ok          = 0
    n_wyckoff_detected     = 0
    n_wyckoff_score_ok     = 0
    n_mss_on_current_bar   = 0
    n_composite_ok         = 0

    _last_signal: dict[tuple, int] = {}
    _daily_count: dict[str, int]   = {}
    signals: list = []

    for i in range(n_m1_bars):
        ts  = m1_df.index[i]
        bar = m1_df.iloc[i]

        formed = [z for z in all_zones if z.formed_at <= ts]
        zd.update_zones(formed, bar, ts)

        if not (SESSION_START <= ts.hour < SESSION_END):
            continue
        n_bars_in_session += 1

        date_key = ts.strftime("%Y-%m-%d")

        for zone in formed:
            if DAILY_CAP > 0 and _daily_count.get(date_key, 0) >= DAILY_CAP:
                break

            n_zone_formed += 1

            if zone.is_mitigated:
                continue
            n_zone_not_mitigated += 1

            if zone.score.total < MIN_ZONE_SCORE:
                continue
            n_zone_score_ok += 1

            if not zone.price_in_zone(float(bar["low"]), float(bar["high"])):
                continue
            n_price_in_zone += 1

            bullish = _is_bullish(ts)
            if zone.side == PivotSide.DEMAND and not bullish:
                continue
            if zone.side == PivotSide.SUPPLY and bullish:
                continue
            n_trend_ok += 1

            z_key = (zone.formed_at, zone.side, zone.zone_bottom)
            if i - _last_signal.get(z_key, -(SIGNAL_COOLDOWN + 1)) < SIGNAL_COOLDOWN:
                continue
            n_cooldown_ok += 1

            wyckoff = wd.detect(m1_df, zone.side, end_idx=i + 1)
            if wyckoff is None:
                continue
            n_wyckoff_detected += 1

            if wyckoff.score < MIN_WYCKOFF_SCORE:
                continue
            n_wyckoff_score_ok += 1

            if wyckoff.mss_bar != i:
                continue
            n_mss_on_current_bar += 1

            n_composite_ok += 1
            _last_signal[z_key] = i
            _daily_count[date_key] = _daily_count.get(date_key, 0) + 1
            signals.append((i, ts, zone.side.name, round(zone.score.total, 2)))

    # ── Print funnel ──────────────────────────────────────────────────────────
    print("=" * 62)
    print("  S&D Signal Generation Funnel — XAUUSD 2025")
    print("=" * 62)
    print()

    def pct(num: int, den: int) -> str:
        return f"{num / den * 100:.1f}%" if den else "—"

    rows = [
        ("M1 bars total",             n_m1_bars,            n_m1_bars,          ""),
        ("  → in session (07–21 UTC)", n_bars_in_session,    n_m1_bars,          f"{pct(n_bars_in_session, n_m1_bars)} of all bars"),
        ("M15 zones detected",         n_zones_total,        n_zones_total,      ""),
        ("Zone-bar pairs (in session):", n_zone_formed,      n_zone_formed,      "one per zone per bar"),
        ("  → zone not mitigated",     n_zone_not_mitigated, n_zone_formed,      pct(n_zone_not_mitigated, n_zone_formed)),
        ("  → zone score ≥ 5.0",       n_zone_score_ok,      n_zone_not_mitigated, pct(n_zone_score_ok, n_zone_not_mitigated)),
        ("  → price inside zone",      n_price_in_zone,      n_zone_score_ok,    pct(n_price_in_zone, n_zone_score_ok)),
        ("  → trend aligned (EMA50)",  n_trend_ok,           n_price_in_zone,    pct(n_trend_ok, n_price_in_zone)),
        ("  → cooldown elapsed",       n_cooldown_ok,        n_trend_ok,         pct(n_cooldown_ok, n_trend_ok)),
        ("  → Wyckoff pattern found",  n_wyckoff_detected,   n_cooldown_ok,      pct(n_wyckoff_detected, n_cooldown_ok)),
        ("  → Wyckoff score ≥ 4.0",    n_wyckoff_score_ok,   n_wyckoff_detected, pct(n_wyckoff_score_ok, n_wyckoff_detected)),
        ("  → MSS fires THIS bar",     n_mss_on_current_bar, n_wyckoff_score_ok, pct(n_mss_on_current_bar, n_wyckoff_score_ok)),
        ("  → composite score ≥ 5.0",  n_composite_ok,       n_mss_on_current_bar, pct(n_composite_ok, n_mss_on_current_bar)),
        ("",                           None, None, ""),
        ("SIGNALS EMITTED",            len(signals),         n_mss_on_current_bar, ""),
    ]

    for label, val, ref, note in rows:
        if val is None:
            print()
            continue
        if ref == val and ref == n_m1_bars or ref == n_zones_total or ref == n_zone_formed or label.startswith("M"):
            print(f"  {label:<38} {val:>8,}   {note}")
        else:
            print(f"  {label:<38} {val:>8,}   {note}")

    print()
    print("  Biggest drops (by absolute count):")
    stages = [
        ("bars in session → zone-bar pairs",   n_bars_in_session,    n_zone_formed),
        ("not mitigated drop",                  n_zone_formed,        n_zone_not_mitigated),
        ("score filter drop",                   n_zone_not_mitigated, n_zone_score_ok),
        ("price-in-zone drop",                  n_zone_score_ok,      n_price_in_zone),
        ("trend filter drop",                   n_price_in_zone,      n_trend_ok),
        ("cooldown drop",                       n_trend_ok,           n_cooldown_ok),
        ("Wyckoff detection drop",              n_cooldown_ok,        n_wyckoff_detected),
        ("Wyckoff score drop",                  n_wyckoff_detected,   n_wyckoff_score_ok),
        ("MSS current-bar drop",                n_wyckoff_score_ok,   n_mss_on_current_bar),
    ]
    stages_sorted = sorted(stages, key=lambda x: x[1] - x[2], reverse=True)
    for label, before, after in stages_sorted[:5]:
        drop = before - after
        print(f"    {label:<36} -{drop:>8,}  ({pct(drop, before)} filtered)")

    print()
    print("  Signals detail:")
    for idx, ts, side, score in signals:
        print(f"    bar {idx:>6}  {ts.date()}  {side:<7}  zone_score={score}")


def main() -> None:
    m1_df = parse_histdata_csv(_M1_FILE)
    m1_df = m1_df[~m1_df.index.duplicated(keep="first")]
    m15_df = resample_ohlcv(m1_df, "15min")
    run_funnel(m1_df, m15_df)


if __name__ == "__main__":
    main()
