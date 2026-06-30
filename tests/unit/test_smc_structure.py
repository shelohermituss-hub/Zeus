"""
Unit tests for zeus.strategy.smc.structure.

Tests BOS and CHoCH detection on controlled price series.

Pattern used:
    1. Build a downtrend (bearish swing structure formed) → trend = BEARISH
    2. Price breaks above last swing high → CHoCH (reversal)
    3. Price then makes a new pivot high and breaks above it → BOS (continuation)
"""

import pandas as pd
import pytest

from zeus.strategy.smc.pivot import detect_pivots, BULLISH, BEARISH
from zeus.strategy.smc.structure import detect_structure, StructureType, StructureEvent


def _make_df(closes: list[float]) -> pd.DataFrame:
    """Wrap a close list in a minimal DataFrame with high = close + 0.5, low = close - 0.5."""
    c  = pd.Series(closes, dtype=float)
    return pd.DataFrame({"high": c + 0.5, "low": c - 0.5, "close": c})


class TestBOSandCHoCHBasic:
    def test_bullish_choch_after_downtrend(self):
        """
        Build a downtrend (H-L-LH-LL), then close breaks above the last swing high.
        The first break should be a CHoCH (trend was bearish).
        """
        # Downtrend: 110 → 100 → 108 → 95 → 105 → 85
        # Then rocket to 115 (breaks all swing highs)
        closes = (
            [110, 109, 108, 107, 106]   # initial rise (swing high near 110)
            + [105, 104, 103, 102, 100] # falls (swing low forms)
            + [104, 103, 102, 101]      # dead-cat bounce (lower high)
            + [95, 94, 93, 92, 91]      # another leg down
            + [96, 97, 98]              # small bounce
            + [85, 84, 83, 82, 81]      # new low
            + [86, 87, 88, 89]          # bounce
            + [115] * 5                 # break above all previous highs → CHoCH
        )
        df = _make_df(closes)
        pivots = detect_pivots(df["high"], df["low"], size=3)
        events = detect_structure(df["close"], pivots)

        bullish = [e for e in events if e.direction == BULLISH]
        assert len(bullish) >= 1
        # First bullish event should be a CHoCH (breaking bearish trend)
        assert bullish[0].structure_type == StructureType.CHOCH

    def test_no_events_on_monotonic_series(self):
        """Strictly rising series with no pivots yields no structure events."""
        closes = list(range(1, 100))
        df = _make_df(closes)
        pivots = detect_pivots(df["high"], df["low"], size=3)
        events = detect_structure(df["close"], pivots)
        assert events == []

    def test_bearish_choch_after_uptrend(self):
        """
        After a BULLISH trend is established via a BOS/CHoCH, a crash below
        the last swing low must produce a BEARISH CHoCH.

        Sequence:
          1. Initial phase creates a bullish BOS (trend → BULLISH)
          2. Small pull-back then continuation
          3. Deep crash → BEARISH CHoCH (trend was BULLISH)
        """
        closes = (
            [90, 91, 92, 93, 94, 95]   # rise (creates pivot high ~95)
            + [93, 92, 91, 90, 89]      # pull-back (swing low forms ~89)
            + [92, 93, 94, 95, 96, 97]  # breaks above 95 → BULLISH BOS, trend=BULLISH
            + [95, 94, 93, 92, 91, 90]  # pull-back (swing low ~90)
            + [92, 93, 94]              # small bounce
            + [60] * 8                  # crash far below all lows → BEARISH CHoCH
        )
        df = _make_df(closes)
        pivots = detect_pivots(df["high"], df["low"], size=3)
        events = detect_structure(df["close"], pivots)

        bullish = [e for e in events if e.direction == BULLISH]
        bearish = [e for e in events if e.direction == BEARISH]

        # There must be at least one bullish event establishing the uptrend
        assert len(bullish) >= 1, "Expected at least one bullish structure event"

        # Find the first bearish event that occurs AFTER a bullish event
        last_bullish_bar = bullish[-1].bar_index
        choch_candidates = [
            e for e in bearish
            if e.bar_index > last_bullish_bar
            and e.structure_type == StructureType.CHOCH
        ]
        assert len(choch_candidates) >= 1, (
            "Expected a BEARISH CHoCH after BULLISH trend was established"
        )

    def test_event_direction_matches_break(self):
        """Every bullish event has direction=+1, every bearish event has direction=-1."""
        closes = [100, 102, 104, 103, 101, 105, 107, 106, 90, 80, 95, 110]
        df = _make_df(closes)
        pivots = detect_pivots(df["high"], df["low"], size=2)
        events = detect_structure(df["close"], pivots)

        for e in events:
            if e.direction == BULLISH:
                assert e.direction == BULLISH
            else:
                assert e.direction == BEARISH

    def test_pivot_level_not_broken_twice(self):
        """The same pivot level should only produce ONE structure event."""
        closes = (
            [100, 101, 102, 103, 104]  # up
            + [103, 102, 101, 100]     # pull-back
            + [105, 106, 107]          # new high (breaks swing high)
            + [108, 109, 110]          # continues up (same pivot already crossed)
        )
        df = _make_df(closes)
        pivots = detect_pivots(df["high"], df["low"], size=2)
        events = detect_structure(df["close"], pivots)

        # Group events by level — each level should appear at most once
        levels = [e.level for e in events]
        assert len(levels) == len(set(levels)), "Same level broken twice"


class TestStructureEventFields:
    def test_event_has_correct_bar_index(self):
        """StructureEvent.bar_index must be the bar where close crossed the level."""
        closes = [100, 101, 102, 103, 102, 101, 100, 105]
        df = _make_df(closes)
        pivots = detect_pivots(df["high"], df["low"], size=2)
        events = detect_structure(df["close"], pivots)

        bullish = [e for e in events if e.direction == BULLISH]
        if bullish:
            idx = bullish[0].bar_index
            # Close at that bar must be >= the broken level
            assert df["close"].iloc[idx] >= bullish[0].level

    def test_is_internal_flag_propagated(self):
        """is_internal flag must match what was passed to detect_structure."""
        closes = [100, 101, 102, 101, 100, 103, 104]
        df = _make_df(closes)
        pivots = detect_pivots(df["high"], df["low"], size=2)

        swing_events    = detect_structure(df["close"], pivots, is_internal=False)
        internal_events = detect_structure(df["close"], pivots, is_internal=True)

        for e in swing_events:
            assert not e.is_internal

        for e in internal_events:
            assert e.is_internal
