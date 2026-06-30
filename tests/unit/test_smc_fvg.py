"""
Unit tests for zeus.strategy.smc.fvg.

FVG pattern (bars A, B, C):
    Bullish: C.low > A.high AND B.close > A.high (AND impulse > threshold)
    Bearish: C.high < A.low  AND B.close < A.low  (AND -impulse > threshold)

We disable auto_threshold to make tests deterministic.
"""

import pandas as pd
import pytest

from zeus.strategy.smc.fvg import detect_fvg, get_active_fvgs, FairValueGap
from zeus.strategy.smc.pivot import BULLISH, BEARISH


def _df(highs, lows, closes, opens=None) -> pd.DataFrame:
    h = pd.Series(highs, dtype=float)
    l = pd.Series(lows,  dtype=float)
    c = pd.Series(closes, dtype=float)
    o = pd.Series(opens, dtype=float) if opens else c.shift(1).fillna(c.iloc[0])
    return pd.DataFrame({"high": h, "low": l, "close": c, "open": o})


class TestBullishFVG:
    def test_basic_bullish_fvg_detected(self):
        """
        Classic bullish FVG:
            A: high=100
            B: big bullish candle, close=115, open=101
            C: low=102 (> A.high=100) → gap exists
        """
        df = _df(
            highs  =[100, 120, 118],
            lows   =[95,  101, 102],
            closes =[99,  115, 110],
            opens  =[96,  101, 111],
        )
        fvgs = detect_fvg(df["high"], df["low"], df["close"], df["open"], auto_threshold=False)
        bullish = [f for f in fvgs if f.direction == BULLISH]
        assert len(bullish) == 1
        fvg = bullish[0]
        assert fvg.top    == pytest.approx(102)   # C.low
        assert fvg.bottom == pytest.approx(100)   # A.high
        assert fvg.bar_index == 2                 # detected at bar C

    def test_no_bullish_fvg_when_no_gap(self):
        """No gap when C.low <= A.high."""
        df = _df(
            highs  =[100, 120, 115],
            lows   =[95,  101, 98],   # C.low=98 < A.high=100 → no gap
            closes =[99,  115, 105],
            opens  =[96,  101, 106],
        )
        fvgs = detect_fvg(df["high"], df["low"], df["close"], df["open"], auto_threshold=False)
        bullish = [f for f in fvgs if f.direction == BULLISH]
        assert len(bullish) == 0

    def test_bullish_fvg_mitigated_when_price_drops_below_bottom(self):
        """FVG is mitigated when a later bar's low dips below fvg.bottom."""
        #        A     B      C      mitigating bar
        highs  = [100, 120,   118,   103]
        lows   = [95,  101,   102,   99]   # mitigating: low=99 < bottom=100
        closes = [99,  115,   110,   100]
        opens  = [96,  101,   111,   112]

        df = _df(highs, lows, closes, opens)
        fvgs = detect_fvg(df["high"], df["low"], df["close"], df["open"], auto_threshold=False)
        bullish = [f for f in fvgs if f.direction == BULLISH]
        assert len(bullish) == 1
        assert bullish[0].mitigated_at == 3

    def test_bullish_fvg_not_mitigated_when_low_stays_above_bottom(self):
        """FVG stays active when no bar dips below the gap bottom."""
        highs  = [100, 120, 118, 115, 117]
        lows   = [95,  101, 102, 103, 104]
        closes = [99,  115, 110, 112, 113]
        opens  = [96,  101, 111, 111, 113]

        df = _df(highs, lows, closes, opens)
        fvgs = detect_fvg(df["high"], df["low"], df["close"], df["open"], auto_threshold=False)
        bullish = [f for f in fvgs if f.direction == BULLISH]
        assert len(bullish) == 1
        assert bullish[0].mitigated_at == -1


class TestBearishFVG:
    def test_basic_bearish_fvg_detected(self):
        """
        Classic bearish FVG:
            A: low=100
            B: big bearish candle, close=85, open=99
            C: high=98 (< A.low=100) → gap exists
        """
        df = _df(
            highs  =[105, 99,  98],
            lows   =[100, 80,  82],
            closes =[101, 85,  90],
            opens  =[104, 99,  89],
        )
        fvgs = detect_fvg(df["high"], df["low"], df["close"], df["open"], auto_threshold=False)
        bearish = [f for f in fvgs if f.direction == BEARISH]
        assert len(bearish) == 1
        fvg = bearish[0]
        assert fvg.top    == pytest.approx(100)   # A.low
        assert fvg.bottom == pytest.approx(98)    # C.high
        assert fvg.bar_index == 2

    def test_bearish_fvg_mitigated_when_high_exceeds_top(self):
        """FVG is mitigated when a later bar's high exceeds fvg.top."""
        highs  = [105, 99,  98,  101]  # bar 3 high=101 > top=100
        lows   = [100, 80,  82,  95]
        closes = [101, 85,  90,  98]
        opens  = [104, 99,  89,  97]

        df = _df(highs, lows, closes, opens)
        fvgs = detect_fvg(df["high"], df["low"], df["close"], df["open"], auto_threshold=False)
        bearish = [f for f in fvgs if f.direction == BEARISH]
        assert len(bearish) == 1
        assert bearish[0].mitigated_at == 3


class TestGetActiveFVGs:
    def test_active_fvgs_excludes_mitigated(self):
        """get_active_fvgs must exclude FVGs mitigated at or before at_bar."""
        # Manually construct FVGs
        from zeus.strategy.smc.fvg import FairValueGap
        fvgs = [
            FairValueGap(bar_index=2, top=105, bottom=100, direction=BULLISH, mitigated_at=4),
            FairValueGap(bar_index=3, top=108, bottom=103, direction=BULLISH, mitigated_at=-1),
        ]
        active = get_active_fvgs(fvgs, at_bar=10)
        assert len(active) == 1
        assert active[0].bar_index == 3

    def test_future_fvgs_not_included(self):
        """FVGs created after at_bar must not appear."""
        from zeus.strategy.smc.fvg import FairValueGap
        fvgs = [
            FairValueGap(bar_index=5, top=105, bottom=100, direction=BULLISH, mitigated_at=-1),
        ]
        active = get_active_fvgs(fvgs, at_bar=3)
        assert active == []
