"""
Tests for zeus.strategy.supply_demand.zone_detector

Synthetic datasets
------------------
All prices near 2000 to mimic XAUUSD.  One "pip" ≈ 0.10.

demand_df  : 10 flat base bars at 2000, one demand pivot, then bullish impulse
             → expect one DEMAND zone around 2000
supply_df  : 10 flat base bars at 2020, one supply pivot, then bearish impulse
             → expect one SUPPLY zone around 2020
mitigation : extends demand_df with bars that re-enter the zone and close through
sweep_df   : demand zone formed after a wick spike below prior lows (spring)
"""
from __future__ import annotations

import pandas as pd
import pytest

from zeus.strategy.supply_demand.zone_detector import (
    SDZone,
    ZoneDetector,
    ZoneScore,
)
from zeus.strategy.supply_demand.pivot_candle import PivotSide

# ── Helpers ───────────────────────────────────────────────────────────────────

def _df(candles: list[tuple], freq: str = "15min") -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from (o, h, l, c) tuples."""
    idx = pd.date_range("2026-06-01 09:00", periods=len(candles), freq=freq)
    return pd.DataFrame(
        [{"open": o, "high": h, "low": l, "close": c, "volume": 0}
         for o, h, l, c in candles],
        index=idx,
    )


def _flat(price: float = 2000.0, n: int = 6) -> list[tuple]:
    """n small-range doji-like bars at a price (simulate tight base)."""
    return [(price, price + 0.5, price - 0.5, price)] * n


def _demand_pivot() -> tuple:
    """Classic demand pivot: long lower wick, small body near top."""
    # o=2000, h=2001, l=1990, c=2000.8 → lower wick=10, body=0.8, upper wick=0.2
    return (2000.0, 2001.0, 1990.0, 2000.8)


def _supply_pivot() -> tuple:
    """Classic supply pivot: long upper wick, small body near bottom."""
    # o=2020.8, h=2030, l=2020, c=2020.2 → upper wick=9.2, body=0.6, lower wick=0.2
    return (2020.8, 2030.0, 2020.0, 2020.2)


def _bullish_impulse(n: int = 5, start: float = 2000.0, step: float = 3.0) -> list[tuple]:
    """n strong bullish bars pushing price up."""
    out = []
    p = start
    for _ in range(n):
        out.append((p, p + step + 0.5, p - 0.2, p + step))
        p += step
    return out


def _bearish_impulse(n: int = 5, start: float = 2020.0, step: float = 3.0) -> list[tuple]:
    """n strong bearish bars pushing price down."""
    out = []
    p = start
    for _ in range(n):
        out.append((p, p + 0.2, p - step - 0.5, p - step))
        p -= step
    return out


def _build_demand_df(base_n: int = 6, impulse_n: int = 5) -> pd.DataFrame:
    """
    base_n flat bars → demand pivot → impulse_n bullish bars
    The impulse will break above the flat base highs → BOS.
    """
    candles = (
        _flat(2000.0, base_n)
        + [_demand_pivot()]
        + _bullish_impulse(impulse_n, start=2001.0, step=4.0)
    )
    return _df(candles)


def _build_supply_df(base_n: int = 6, impulse_n: int = 5) -> pd.DataFrame:
    candles = (
        _flat(2020.0, base_n)
        + [_supply_pivot()]
        + _bearish_impulse(impulse_n, start=2019.0, step=4.0)
    )
    return _df(candles)


# ── ZoneDetector.detect_zones ─────────────────────────────────────────────────

class TestDetectZones:
    def test_detects_demand_zone(self):
        df = _build_demand_df()
        det = ZoneDetector(swing_lookback=3, bos_lookback=10, bos_max_bars=20,
                           min_pivot_score=3.0, min_zone_score=3.0)
        zones = det.detect_zones(df)
        demand = [z for z in zones if z.side == PivotSide.DEMAND]
        assert len(demand) >= 1, f"Expected ≥1 DEMAND zone, got {zones}"

    def test_demand_zone_price_level(self):
        df = _build_demand_df()
        det = ZoneDetector(swing_lookback=3, bos_lookback=10, bos_max_bars=20,
                           min_pivot_score=3.0, min_zone_score=3.0)
        zones = det.detect_zones(df)
        demand = [z for z in zones if z.side == PivotSide.DEMAND]
        assert demand, "No demand zone detected"
        z = demand[0]
        # Zone should be near 2000 (pivot candle body)
        assert 1998.0 <= z.zone_bottom <= 2002.0, f"zone_bottom={z.zone_bottom}"
        assert z.zone_top >= z.zone_bottom

    def test_demand_zone_wick_extreme_below_body(self):
        df = _build_demand_df()
        det = ZoneDetector(swing_lookback=3, bos_lookback=10, bos_max_bars=20,
                           min_pivot_score=3.0, min_zone_score=3.0)
        zones = det.detect_zones(df)
        demand = [z for z in zones if z.side == PivotSide.DEMAND]
        assert demand
        z = demand[0]
        assert z.wick_extreme < z.zone_bottom, (
            f"wick_extreme={z.wick_extreme} should be below zone_bottom={z.zone_bottom}"
        )

    def test_detects_supply_zone(self):
        df = _build_supply_df()
        det = ZoneDetector(swing_lookback=3, bos_lookback=10, bos_max_bars=20,
                           min_pivot_score=3.0, min_zone_score=3.0)
        zones = det.detect_zones(df)
        supply = [z for z in zones if z.side == PivotSide.SUPPLY]
        assert len(supply) >= 1, f"Expected ≥1 SUPPLY zone, got {zones}"

    def test_supply_zone_wick_extreme_above_body(self):
        df = _build_supply_df()
        det = ZoneDetector(swing_lookback=3, bos_lookback=10, bos_max_bars=20,
                           min_pivot_score=3.0, min_zone_score=3.0)
        zones = det.detect_zones(df)
        supply = [z for z in zones if z.side == PivotSide.SUPPLY]
        assert supply
        z = supply[0]
        assert z.wick_extreme > z.zone_top, (
            f"wick_extreme={z.wick_extreme} should be above zone_top={z.zone_top}"
        )

    def test_zone_sorted_chronologically(self):
        # two demand zones at different times
        candles = (
            _flat(2000.0, 6) + [_demand_pivot()] + _bullish_impulse(5, 2001.0, 4.0)
            + _flat(2030.0, 6) + [_demand_pivot()] + _bullish_impulse(5, 2031.0, 4.0)
        )
        df = _df(candles)
        det = ZoneDetector(swing_lookback=3, bos_lookback=15, bos_max_bars=20,
                           min_pivot_score=3.0, min_zone_score=3.0)
        zones = det.detect_zones(df)
        if len(zones) >= 2:
            for a, b in zip(zones, zones[1:]):
                assert a.formed_at <= b.formed_at

    def test_empty_df_returns_empty(self):
        df = _df([])
        det = ZoneDetector()
        assert det.detect_zones(df) == []

    def test_too_short_df_returns_empty(self):
        df = _df([(2000, 2001, 1999, 2000)] * 3)
        det = ZoneDetector()
        assert det.detect_zones(df) == []

    def test_no_bos_returns_no_zone(self):
        # flat price forever → no BOS
        df = _df(_flat(2000.0, 60))
        det = ZoneDetector(min_pivot_score=3.0, min_zone_score=3.0)
        zones = det.detect_zones(df)
        assert len(zones) == 0, f"Expected 0 zones on flat data, got {len(zones)}"

    def test_fresh_score_2_on_detection(self):
        df = _build_demand_df()
        det = ZoneDetector(swing_lookback=3, bos_lookback=10, bos_max_bars=20,
                           min_pivot_score=3.0, min_zone_score=3.0)
        zones = det.detect_zones(df)
        demand = [z for z in zones if z.side == PivotSide.DEMAND]
        assert demand
        assert demand[0].score.fresh == pytest.approx(2.0)

    def test_zone_not_mitigated_on_detection(self):
        df = _build_demand_df()
        det = ZoneDetector(swing_lookback=3, bos_lookback=10, bos_max_bars=20,
                           min_pivot_score=3.0, min_zone_score=3.0)
        zones = det.detect_zones(df)
        assert all(not z.is_mitigated for z in zones)


# ── ZoneDetector.update_zones ─────────────────────────────────────────────────

def _demand_zone_fixture() -> SDZone:
    """Build a minimal DEMAND zone directly for update tests."""
    from zeus.strategy.supply_demand.pivot_candle import PivotCandle
    pc = PivotCandle(
        index=pd.Timestamp("2026-06-01 09:00"),
        open=2000.0, high=2001.0, low=1990.0, close=2000.8,
        side=PivotSide.DEMAND,
        score=7.5,
        body_high=2000.8, body_low=2000.0,
        wick_high=2001.0, wick_low=1990.0,
        upper_wick_ratio=0.03, lower_wick_ratio=0.72, body_ratio=0.07,
    )
    return SDZone(
        side=PivotSide.DEMAND,
        zone_top=2000.8, zone_bottom=2000.0,
        wick_extreme=1990.0,
        pivot=pc, pivot_bar=6,
        formed_at=pd.Timestamp("2026-06-01 11:30"),
        bos_bar=12, bos_level=2001.0,
        base_candles=4,
        score=ZoneScore(bos=2.0, impulse=1.5, time=1.5, fresh=2.0, sweep=2.0),
    )


class TestUpdateZones:
    def test_touch_count_increments_on_entry(self):
        z   = _demand_zone_fixture()
        det = ZoneDetector()
        # bar wicks into zone (low=1999, high=2000.5)
        bar = pd.Series({"open": 2001.0, "high": 2001.5, "low": 1999.0, "close": 2000.3})
        det.update_zones([z], bar, pd.Timestamp("2026-06-02 09:15"))
        assert z.touch_count == 1

    def test_no_touch_when_price_outside_zone(self):
        z   = _demand_zone_fixture()
        det = ZoneDetector()
        # bar stays well above zone
        bar = pd.Series({"open": 2010.0, "high": 2015.0, "low": 2008.0, "close": 2012.0})
        det.update_zones([z], bar, pd.Timestamp("2026-06-02 09:15"))
        assert z.touch_count == 0
        assert not z.is_mitigated

    def test_freshness_drops_to_1_on_first_touch(self):
        z   = _demand_zone_fixture()
        det = ZoneDetector()
        bar = pd.Series({"open": 2001.0, "high": 2001.5, "low": 1999.0, "close": 2000.3})
        det.update_zones([z], bar, pd.Timestamp("2026-06-02 09:15"))
        assert z.score.fresh == pytest.approx(1.0)

    def test_freshness_stays_1_on_second_touch(self):
        z   = _demand_zone_fixture()
        det = ZoneDetector()
        bar = pd.Series({"open": 2001.0, "high": 2001.5, "low": 1999.0, "close": 2000.3})
        ts  = pd.Timestamp("2026-06-02 09:15")
        det.update_zones([z], bar, ts)
        det.update_zones([z], bar, ts + pd.Timedelta(minutes=15))
        assert z.score.fresh == pytest.approx(1.0)   # stays 1.0, not 0.0
        assert z.touch_count == 2

    def test_mitigation_on_close_through_wick(self):
        z   = _demand_zone_fixture()
        det = ZoneDetector()
        # close below wick_extreme (1990.0) → mitigated
        bar = pd.Series({"open": 2000.0, "high": 2001.0, "low": 1985.0, "close": 1985.0})
        ts  = pd.Timestamp("2026-06-02 10:00")
        det.update_zones([z], bar, ts)
        assert z.is_mitigated
        assert z.mitigated_at == ts
        assert z.score.fresh == pytest.approx(0.0)

    def test_mitigated_zone_not_updated_again(self):
        z   = _demand_zone_fixture()
        det = ZoneDetector()
        bar_mit = pd.Series({"open": 2000.0, "high": 2001.0, "low": 1985.0, "close": 1985.0})
        bar_ok  = pd.Series({"open": 2001.0, "high": 2001.5, "low": 1999.0, "close": 2000.3})
        ts1 = pd.Timestamp("2026-06-02 10:00")
        ts2 = ts1 + pd.Timedelta(minutes=15)
        det.update_zones([z], bar_mit, ts1)
        count_before = z.touch_count
        det.update_zones([z], bar_ok, ts2)
        assert z.touch_count == count_before   # no further increments

    def test_no_mitigation_on_close_inside_body(self):
        z   = _demand_zone_fixture()
        det = ZoneDetector()
        # close inside body (between zone_bottom=2000 and zone_top=2000.8)
        bar = pd.Series({"open": 2001.0, "high": 2001.5, "low": 1999.0, "close": 2000.4})
        det.update_zones([z], bar, pd.Timestamp("2026-06-02 09:15"))
        assert not z.is_mitigated

    def test_empty_zones_list_no_error(self):
        det = ZoneDetector()
        bar = pd.Series({"open": 2000.0, "high": 2001.0, "low": 1999.0, "close": 2000.5})
        det.update_zones([], bar, pd.Timestamp("2026-06-02 09:15"))  # must not raise


# ── ZoneScore ─────────────────────────────────────────────────────────────────

class TestZoneScore:
    def test_total_sum(self):
        s = ZoneScore(bos=2.0, impulse=1.5, time=1.0, fresh=2.0, sweep=1.0)
        assert s.total == pytest.approx(7.5)

    def test_total_zero(self):
        s = ZoneScore(bos=0.0, impulse=0.0, time=0.0, fresh=0.0, sweep=0.0)
        assert s.total == pytest.approx(0.0)

    def test_total_max(self):
        s = ZoneScore(bos=2.0, impulse=2.0, time=2.0, fresh=2.0, sweep=2.0)
        assert s.total == pytest.approx(10.0)

    def test_str_contains_total(self):
        s = ZoneScore(bos=2.0, impulse=1.5, time=1.0, fresh=2.0, sweep=1.0)
        assert "7.5" in str(s)


# ── SDZone helpers ─────────────────────────────────────────────────────────────

class TestSDZoneHelpers:
    def test_price_in_zone_overlap(self):
        z = _demand_zone_fixture()  # zone 2000.0–2000.8
        assert z.price_in_zone(low=1999.0, high=2000.5)   # overlaps
        assert z.price_in_zone(low=2000.0, high=2000.8)   # exact match
        assert not z.price_in_zone(low=2001.0, high=2002.0)  # above
        assert not z.price_in_zone(low=1988.0, high=1999.0)  # below

    def test_price_closed_through_demand(self):
        z = _demand_zone_fixture()   # wick_extreme=1990.0
        assert z.price_closed_through(1989.0)    # below wick → mitigated
        assert not z.price_closed_through(1991.0) # above wick → safe
        assert not z.price_closed_through(2000.3) # inside body → safe

    def test_midpoint(self):
        z = _demand_zone_fixture()  # zone 2000.0–2000.8
        assert z.midpoint == pytest.approx(2000.4)

    def test_height(self):
        z = _demand_zone_fixture()
        assert z.height == pytest.approx(0.8)
