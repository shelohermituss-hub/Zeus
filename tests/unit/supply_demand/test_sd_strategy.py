"""Tests for SDStrategy — main signal generator (step 5)."""
from __future__ import annotations

import pandas as pd
import pytest
from unittest.mock import MagicMock

from zeus.strategy.supply_demand.fibonacci import compute_fib_levels
from zeus.strategy.supply_demand.pivot_candle import PivotCandle, PivotSide
from zeus.strategy.supply_demand.sd_strategy import SDSignal, SDStrategy
from zeus.strategy.supply_demand.wyckoff import WyckoffPattern
from zeus.strategy.supply_demand.zone_detector import SDZone, ZoneScore


# ── Shared timestamp ──────────────────────────────────────────────────────────

_T0 = pd.Timestamp("2026-06-01 01:00")


# ── DataFrame builders ────────────────────────────────────────────────────────

def _m1_demand(n: int = 10) -> pd.DataFrame:
    """M1 bars with price overlapping demand zone [1.108, 1.110]."""
    idx = pd.date_range(_T0 + pd.Timedelta("1min"), periods=n, freq="1min")
    return pd.DataFrame(
        {"open": [1.109]*n, "high": [1.111]*n, "low": [1.107]*n, "close": [1.109]*n},
        index=idx,
    )


def _m1_supply(n: int = 10) -> pd.DataFrame:
    """M1 bars with price overlapping supply zone [1.112, 1.115]."""
    idx = pd.date_range(_T0 + pd.Timedelta("1min"), periods=n, freq="1min")
    return pd.DataFrame(
        {"open": [1.113]*n, "high": [1.114]*n, "low": [1.112]*n, "close": [1.113]*n},
        index=idx,
    )


def _m15_df() -> pd.DataFrame:
    """Placeholder M15 df — zone detector is mocked."""
    idx = pd.date_range(_T0, periods=6, freq="15min")
    return pd.DataFrame(
        {"open": [1.100]*6, "high": [1.120]*6, "low": [1.095]*6, "close": [1.110]*6},
        index=idx,
    )


# ── Object builders ───────────────────────────────────────────────────────────

def _pivot(ts: pd.Timestamp = _T0, side: PivotSide = PivotSide.DEMAND) -> PivotCandle:
    return PivotCandle(
        index=ts, open=1.108, high=1.113, low=1.095, close=1.110,
        side=side, score=8.0,
        body_high=1.110, body_low=1.108, wick_high=1.113, wick_low=1.095,
        upper_wick_ratio=0.10, lower_wick_ratio=0.80, body_ratio=0.10,
    )


def _demand_zone(formed_at: pd.Timestamp = _T0) -> SDZone:
    return SDZone(
        side=PivotSide.DEMAND,
        zone_top=1.110, zone_bottom=1.108, wick_extreme=1.095,
        pivot=_pivot(formed_at, PivotSide.DEMAND),
        pivot_bar=2, formed_at=formed_at, bos_bar=5, bos_level=1.113,
        base_candles=2,
        score=ZoneScore(bos=2.0, impulse=1.5, time=2.0, fresh=2.0, sweep=0.0),
    )


def _supply_zone(formed_at: pd.Timestamp = _T0) -> SDZone:
    return SDZone(
        side=PivotSide.SUPPLY,
        zone_top=1.115, zone_bottom=1.112, wick_extreme=1.120,
        pivot=_pivot(formed_at, PivotSide.SUPPLY),
        pivot_bar=2, formed_at=formed_at, bos_bar=5, bos_level=1.108,
        base_candles=2,
        score=ZoneScore(bos=2.0, impulse=1.5, time=2.0, fresh=2.0, sweep=0.0),
    )


def _wy_demand(mss_bar: int, ts: pd.Timestamp) -> WyckoffPattern:
    return WyckoffPattern(
        side=PivotSide.DEMAND,
        accum_high=1.110, accum_low=1.108, accum_bars=3,
        manip_bar=mss_bar - 1, manip_extreme=1.103,
        mss_bar=mss_bar, mss_close=1.113,
        score=7.5, formed_at=ts,
    )


def _wy_supply(mss_bar: int, ts: pd.Timestamp) -> WyckoffPattern:
    return WyckoffPattern(
        side=PivotSide.SUPPLY,
        accum_high=1.115, accum_low=1.112, accum_bars=3,
        manip_bar=mss_bar - 1, manip_extreme=1.120,
        mss_bar=mss_bar, mss_close=1.109,
        score=7.5, formed_at=ts,
    )


def _make_strategy(
    zone: SDZone,
    wyckoff_map: dict[int, WyckoffPattern | None],
    **kwargs,
) -> SDStrategy:
    """
    SDStrategy with mocked zone + Wyckoff detectors.

    wyckoff_map: bar-index → WyckoffPattern (or None). The lambda maps
    end_idx back to bar index: wyckoff_map.get(end_idx - 1).

    Trend filter is disabled by default so tests focus on orchestration;
    use trend_filter tests below to validate that filter in isolation.
    """
    mock_zones = MagicMock()
    mock_zones.detect_zones.return_value = [zone]
    mock_zones.update_zones.return_value = None

    mock_wy = MagicMock()
    mock_wy.detect.side_effect = lambda df, side, end_idx: wyckoff_map.get(end_idx - 1)

    kwargs.setdefault("use_trend_filter", False)
    return SDStrategy(zone_detector=mock_zones, wyckoff_detector=mock_wy, **kwargs)


# ── Signal detection ──────────────────────────────────────────────────────────

def test_no_zones_returns_empty():
    mock_zones = MagicMock()
    mock_zones.detect_zones.return_value = []
    strat = SDStrategy(zone_detector=mock_zones, wyckoff_detector=MagicMock())
    assert strat.run(_m15_df(), _m1_demand()) == []


def test_detects_demand_signal():
    zone = _demand_zone(_T0)
    m1   = _m1_demand()
    strat = _make_strategy(zone, {5: _wy_demand(5, m1.index[5])})
    signals = strat.run(_m15_df(), m1)
    assert len(signals) == 1
    assert signals[0].direction == "long"


def test_detects_supply_signal():
    zone = _supply_zone(_T0)
    m1   = _m1_supply()
    strat = _make_strategy(zone, {5: _wy_supply(5, m1.index[5])})
    signals = strat.run(_m15_df(), m1)
    assert len(signals) == 1
    assert signals[0].direction == "short"


# ── Entry / SL / TP values ────────────────────────────────────────────────────

def test_demand_signal_entry_sl_tp():
    zone = _demand_zone(_T0)
    m1   = _m1_demand()
    sig  = _make_strategy(zone, {5: _wy_demand(5, m1.index[5])}).run(_m15_df(), m1)[0]
    assert sig.entry_price == round(1.113, 6)
    assert sig.stop_loss   == round(1.095, 6)
    risk = round(abs(1.113 - 1.095), 6)
    assert sig.take_profit == round(1.113 + 3 * risk, 6)


def test_supply_signal_entry_sl_tp():
    zone = _supply_zone(_T0)
    m1   = _m1_supply()
    sig  = _make_strategy(zone, {5: _wy_supply(5, m1.index[5])}).run(_m15_df(), m1)[0]
    assert sig.entry_price == round(1.109, 6)
    assert sig.stop_loss   == round(1.120, 6)
    risk = round(abs(1.109 - 1.120), 6)
    assert sig.take_profit == round(1.109 - 3 * risk, 6)


def test_signal_risk_and_reward_properties():
    zone = _demand_zone(_T0)
    m1   = _m1_demand()
    sig  = _make_strategy(zone, {5: _wy_demand(5, m1.index[5])}).run(_m15_df(), m1)[0]
    assert sig.risk   == pytest.approx(abs(sig.entry_price - sig.stop_loss),  abs=1e-6)
    assert sig.reward == pytest.approx(abs(sig.take_profit - sig.entry_price), abs=1e-6)
    assert sig.risk_reward == 3.0


def test_signal_score_fields():
    zone = _demand_zone(_T0)
    m1   = _m1_demand()
    sig  = _make_strategy(zone, {5: _wy_demand(5, m1.index[5])}).run(_m15_df(), m1)[0]
    assert sig.zone_score      == round(7.5, 2)
    assert sig.wyckoff_score   == round(7.5, 2)
    assert sig.fib_bonus       == 0.0
    assert sig.composite_score == round(7.5, 2)


def test_signal_references_zone_and_wyckoff():
    zone = _demand_zone(_T0)
    m1   = _m1_demand()
    w    = _wy_demand(5, m1.index[5])
    sig  = _make_strategy(zone, {5: w}).run(_m15_df(), m1)[0]
    assert sig.zone    is zone
    assert sig.wyckoff is w
    assert sig.bar_index == 5
    assert sig.formed_at == m1.index[5]


# ── Temporal causality ────────────────────────────────────────────────────────

def test_no_signal_before_zone_formed():
    # Zone formed 20 min after M1 bars start → none qualify
    zone = _demand_zone(_T0 + pd.Timedelta("20min"))
    m1   = _m1_demand()  # bars T0+1min → T0+10min, all before zone_formed
    strat = _make_strategy(zone, {})
    assert strat.run(_m15_df(), m1) == []


# ── Zone state filters ────────────────────────────────────────────────────────

def test_no_signal_when_zone_mitigated():
    zone = _demand_zone(_T0)
    zone.is_mitigated = True
    m1   = _m1_demand()
    strat = _make_strategy(zone, {5: _wy_demand(5, m1.index[5])})
    assert strat.run(_m15_df(), m1) == []


def test_no_signal_below_min_zone_score():
    zone = _demand_zone(_T0)
    zone.score = ZoneScore(bos=0.5, impulse=0.5, time=0.5, fresh=0.5, sweep=0.5)  # total=2.5
    m1   = _m1_demand()
    strat = _make_strategy(zone, {5: _wy_demand(5, m1.index[5])}, min_zone_score=5.0)
    assert strat.run(_m15_df(), m1) == []


# ── Wyckoff filters ───────────────────────────────────────────────────────────

def test_no_signal_when_wyckoff_returns_none():
    zone  = _demand_zone(_T0)
    strat = _make_strategy(zone, {})   # empty map → detect always returns None
    assert strat.run(_m15_df(), _m1_demand()) == []


def test_no_signal_below_min_wyckoff_score():
    zone = _demand_zone(_T0)
    m1   = _m1_demand()
    w = WyckoffPattern(
        side=PivotSide.DEMAND,
        accum_high=1.110, accum_low=1.108, accum_bars=3,
        manip_bar=4, manip_extreme=1.103,
        mss_bar=5, mss_close=1.113,
        score=2.0,   # below min_wyckoff_score=4.0
        formed_at=m1.index[5],
    )
    strat = _make_strategy(zone, {5: w}, min_wyckoff_score=4.0)
    assert strat.run(_m15_df(), m1) == []


def test_no_signal_when_mss_not_on_current_bar():
    zone = _demand_zone(_T0)
    m1   = _m1_demand()
    # mss_bar=4 but queried at i=5 → mss_bar != i → skip
    w = WyckoffPattern(
        side=PivotSide.DEMAND,
        accum_high=1.110, accum_low=1.108, accum_bars=3,
        manip_bar=3, manip_extreme=1.103,
        mss_bar=4, mss_close=1.113,   # MSS on previous bar
        score=7.5, formed_at=m1.index[4],
    )
    strat = _make_strategy(zone, {5: w})
    assert strat.run(_m15_df(), m1) == []


# ── Cooldown ──────────────────────────────────────────────────────────────────

def test_cooldown_prevents_duplicate_signal():
    zone = _demand_zone(_T0)
    m1   = _m1_demand(n=40)
    strat = _make_strategy(
        zone,
        {5: _wy_demand(5, m1.index[5]), 6: _wy_demand(6, m1.index[6])},
        signal_cooldown=30,
    )
    signals = strat.run(_m15_df(), m1)
    assert len(signals) == 1
    assert signals[0].bar_index == 5


def test_signal_allowed_after_cooldown_expires():
    zone = _demand_zone(_T0)
    m1   = _m1_demand(n=50)
    strat = _make_strategy(
        zone,
        {5: _wy_demand(5, m1.index[5]), 36: _wy_demand(36, m1.index[36])},
        signal_cooldown=30,
    )
    signals = strat.run(_m15_df(), m1)
    assert len(signals) == 2
    assert signals[0].bar_index == 5
    assert signals[1].bar_index == 36


# ── Fibonacci confluence ──────────────────────────────────────────────────────

def test_no_fib_bonus_when_htf_fibs_is_none():
    zone = _demand_zone(_T0)
    m1   = _m1_demand()
    sig  = _make_strategy(zone, {5: _wy_demand(5, m1.index[5])}).run(_m15_df(), m1, htf_fibs=None)[0]
    assert sig.fib_bonus == 0.0
    assert sig.fib is None


def test_fib_bonus_added_to_composite():
    # Bullish swing (1.090, 1.130): equilibrium=1.110, zone midpoint=1.109 → discount → +0.5
    zone = _demand_zone(_T0)
    m1   = _m1_demand()
    fibs = compute_fib_levels(swing_low=1.090, swing_high=1.130, is_bullish=True)
    sig  = _make_strategy(zone, {5: _wy_demand(5, m1.index[5])}).run(_m15_df(), m1, htf_fibs=fibs)[0]
    assert sig.fib_bonus > 0.0
    assert sig.fib is not None
    assert sig.composite_score == round(sig.zone_score + sig.fib_bonus, 2)


# ── Composite score filter ────────────────────────────────────────────────────

def test_no_signal_below_min_composite_score():
    zone = _demand_zone(_T0)   # zone_score=7.5, fib_bonus=0 → composite=7.5
    m1   = _m1_demand()
    strat = _make_strategy(zone, {5: _wy_demand(5, m1.index[5])}, min_composite_score=9.0)
    assert strat.run(_m15_df(), m1) == []


# ── Trend filter ─────────────────────────────────────────────────────────────

def _m15_df_trending(n: int = 60, bullish: bool = True) -> pd.DataFrame:
    """60 M15 bars with a clear slope so EMA50 has enough history."""
    idx = pd.date_range(_T0, periods=n, freq="15min")
    if bullish:
        closes = [1.090 + i * 0.001 for i in range(n)]   # rising
    else:
        closes = [1.150 - i * 0.001 for i in range(n)]   # falling
    return pd.DataFrame(
        {"open": closes, "high": [c + 0.002 for c in closes],
         "low":  [c - 0.002 for c in closes], "close": closes},
        index=idx,
    )


def _m1_at_end_of_m15(m15_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """M1 bars starting 1 min after the last M15 bar."""
    start = m15_df.index[-1] + pd.Timedelta("1min")
    idx   = pd.date_range(start, periods=n, freq="1min")
    return pd.DataFrame(
        {"open": [1.109]*n, "high": [1.111]*n, "low": [1.107]*n, "close": [1.109]*n},
        index=idx,
    )


def test_demand_signal_blocked_in_downtrend():
    """Demand zone must not signal when M15 EMA50 slope is down."""
    m15  = _m15_df_trending(bullish=False)
    zone = _demand_zone(m15.index[0])
    m1   = _m1_at_end_of_m15(m15)
    strat = _make_strategy(zone, {5: _wy_demand(5, m1.index[5])}, use_trend_filter=True)
    assert strat.run(m15, m1) == []


def test_supply_signal_blocked_in_uptrend():
    """Supply zone must not signal when M15 EMA50 slope is up."""
    m15  = _m15_df_trending(bullish=True)
    zone = _supply_zone(m15.index[0])
    m1   = _m1_at_end_of_m15(m15)
    m1["open"]  = 1.113; m1["high"] = 1.114; m1["low"] = 1.112; m1["close"] = 1.113
    strat = _make_strategy(zone, {5: _wy_supply(5, m1.index[5])}, use_trend_filter=True)
    assert strat.run(m15, m1) == []


def test_demand_signal_passes_in_uptrend():
    """Demand zone should signal when M15 EMA50 slope is up."""
    m15  = _m15_df_trending(bullish=True)
    zone = _demand_zone(m15.index[0])
    m1   = _m1_at_end_of_m15(m15)
    strat = _make_strategy(zone, {5: _wy_demand(5, m1.index[5])}, use_trend_filter=True)
    signals = strat.run(m15, m1)
    assert len(signals) == 1
    assert signals[0].direction == "long"


def test_supply_signal_passes_in_downtrend():
    """Supply zone should signal when M15 EMA50 slope is down."""
    m15  = _m15_df_trending(bullish=False)
    zone = _supply_zone(m15.index[0])
    m1   = _m1_at_end_of_m15(m15)
    m1["open"]  = 1.113; m1["high"] = 1.114; m1["low"] = 1.112; m1["close"] = 1.113
    strat = _make_strategy(zone, {5: _wy_supply(5, m1.index[5])}, use_trend_filter=True)
    signals = strat.run(m15, m1)
    assert len(signals) == 1
    assert signals[0].direction == "short"


# ── Risk validation ───────────────────────────────────────────────────────────

def test_no_signal_when_entry_equals_stop():
    zone = _demand_zone(_T0)
    m1   = _m1_demand()
    # mss_close == wick_extreme → risk = 0 → rejected in _build_signal
    w = WyckoffPattern(
        side=PivotSide.DEMAND,
        accum_high=1.110, accum_low=1.108, accum_bars=3,
        manip_bar=4, manip_extreme=1.095,
        mss_bar=5, mss_close=1.095,   # same as wick_extreme
        score=7.5, formed_at=m1.index[5],
    )
    strat = _make_strategy(zone, {5: w})
    assert strat.run(_m15_df(), m1) == []
