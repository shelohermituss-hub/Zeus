"""
Unit tests for PartialCloseConfig and PartialCloseState.
"""
import pytest

from zeus.backtest.partial_close import (
    PartialCloseConfig,
    PartialCloseLevel,
    PartialCloseState,
    TrailingConfig,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _state(entry=5000.0, sl=4980.0, is_long=True, qty=1.0) -> PartialCloseState:
    """Long state with 20-pip SL ($20 distance at pip_value=$1)."""
    return PartialCloseState(
        entry_price=entry,
        sl_price=sl,
        is_long=is_long,
        original_qty=qty,
    )


def _cfg() -> PartialCloseConfig:
    return PartialCloseConfig.default()


# ──────────────────────────────────────────────────────────────────────────────
# PartialCloseState — math helpers
# ──────────────────────────────────────────────────────────────────────────────

class TestPartialCloseStateMath:
    def test_sl_distance_long(self):
        s = _state(entry=5000, sl=4980)
        assert s.sl_distance == pytest.approx(20.0)

    def test_sl_distance_short(self):
        s = PartialCloseState(entry_price=5000, sl_price=5020, is_long=False, original_qty=1.0)
        assert s.sl_distance == pytest.approx(20.0)

    def test_current_r_at_3r(self):
        s = _state(entry=5000, sl=4980)  # distance=20
        # price at 3R = 5000 + 3*20 = 5060
        assert s.current_r(5060.0) == pytest.approx(3.0)

    def test_current_r_negative_when_in_loss(self):
        s = _state(entry=5000, sl=4980)
        assert s.current_r(4990.0) == pytest.approx(-0.5)

    def test_price_at_r(self):
        s = _state(entry=5000, sl=4980)
        assert s.price_at_r(3.0) == pytest.approx(5060.0)
        assert s.price_at_r(0.0) == pytest.approx(5000.0)
        assert s.price_at_r(-1.0) == pytest.approx(4980.0)

    def test_price_at_r_short(self):
        s = PartialCloseState(entry_price=5000, sl_price=5020, is_long=False, original_qty=1.0)
        assert s.price_at_r(3.0) == pytest.approx(4940.0)


# ──────────────────────────────────────────────────────────────────────────────
# PartialCloseState — partial_close mutation
# ──────────────────────────────────────────────────────────────────────────────

class TestClosePartial:
    def test_close_75_pct_reduces_qty(self):
        s = _state(qty=1.0)
        s.close_partial(5060.0, 0.75)
        assert s.remaining_qty == pytest.approx(0.25)

    def test_pnl_accumulates(self):
        s = _state(entry=5000, sl=4980, qty=1.0)
        pnl = s.close_partial(5060.0, 0.75)   # 60 × 0.75 = 45
        assert pnl == pytest.approx(45.0)
        assert s.partial_pnl == pytest.approx(45.0)

    def test_second_partial_uses_remaining(self):
        s = _state(entry=5000, sl=4980, qty=1.0)
        s.close_partial(5060.0, 0.75)          # remaining = 0.25
        s.close_partial(5100.0, 0.40)          # 40% of 0.25 = 0.10 qty
        assert s.remaining_qty == pytest.approx(0.15)


# ──────────────────────────────────────────────────────────────────────────────
# PartialCloseState — SL move
# ──────────────────────────────────────────────────────────────────────────────

class TestMoveSl:
    def test_move_sl_to_breakeven(self):
        s = _state(entry=5000, sl=4980)
        s.move_sl_to_r(0.0)
        assert s.sl_price == pytest.approx(5000.0)

    def test_effective_sl_without_trailing(self):
        s = _state(entry=5000, sl=4980)
        assert s.effective_sl() == pytest.approx(4980.0)


# ──────────────────────────────────────────────────────────────────────────────
# PartialCloseState — trailing stop
# ──────────────────────────────────────────────────────────────────────────────

class TestTrailingStop:
    def test_trailing_only_moves_forward(self):
        s = _state(entry=5000, sl=4980)
        s.update_trailing(5100.0, trail_r=5.0)  # trail = 5100 - 5*20 = 5000
        first = s.trailing_sl
        s.update_trailing(5090.0, trail_r=5.0)  # would be 5090-100=4990 < 5000 → no move
        assert s.trailing_sl == first

    def test_trailing_advances_with_price(self):
        s = _state(entry=5000, sl=4980)
        s.update_trailing(5100.0, trail_r=5.0)  # trail = 5000
        s.update_trailing(5200.0, trail_r=5.0)  # trail = 5100
        assert s.trailing_sl == pytest.approx(5100.0)

    def test_trailing_short(self):
        s = PartialCloseState(entry_price=5000, sl_price=5020, is_long=False, original_qty=1.0)
        s.update_trailing(4900.0, trail_r=5.0)  # trail = 4900 + 5*20 = 5000
        assert s.trailing_sl == pytest.approx(5000.0)


# ──────────────────────────────────────────────────────────────────────────────
# PartialCloseState.process_bar — full lifecycle
# ──────────────────────────────────────────────────────────────────────────────

class TestProcessBar:
    def test_no_events_below_first_level(self):
        s = _state(entry=5000, sl=4980, qty=1.0)
        events = s.process_bar(5050.0, 4995.0, _cfg())  # high 2.5R, low above SL
        assert events == []

    def test_first_partial_fires_at_3r(self):
        s = _state(entry=5000, sl=4980, qty=1.0)
        # high=5065 (3.25R), low=5001 (above breakeven so no SL after move)
        events = s.process_bar(5065.0, 5001.0, _cfg())
        types = [e[0] for e in events]
        assert "partial" in types
        assert s.levels_hit == 1
        assert s.remaining_qty == pytest.approx(0.25)

    def test_sl_to_breakeven_after_3r(self):
        s = _state(entry=5000, sl=4980, qty=1.0)
        s.process_bar(5065.0, 5001.0, _cfg())  # 3R fires, low above BE
        assert s.sl_price == pytest.approx(5000.0)

    def test_second_partial_fires_at_5r(self):
        s = _state(entry=5000, sl=4980, qty=1.0)
        # Bar 1: 3R fires (high 5065), low=5001 above new BE=5000
        s.process_bar(5065.0, 5001.0, _cfg())
        # Bar 2: 5R fires (high 5110); after 5R, trail_sl=5010; low=5015>trail_sl → no trail yet
        s.process_bar(5110.0, 5015.0, _cfg())
        assert s.levels_hit == 2
        # 75% closed at 3R, then 40% of remaining 25% = 10%; total remaining = 15%
        assert s.remaining_qty == pytest.approx(0.15)

    def test_sl_hit_closes_position(self):
        s = _state(entry=5000, sl=4980, qty=1.0)
        # No partials yet — price drops into SL
        events = s.process_bar(5010.0, 4975.0, _cfg())
        types = [e[0] for e in events]
        assert "sl" in types
        assert s.remaining_qty == pytest.approx(0.0)

    def test_hard_close_at_25r(self):
        s = _state(entry=5000, sl=4980, qty=1.0)
        # Bar 1: 3R fires; low=5001 above new BE=5000
        s.process_bar(5065.0, 5001.0, _cfg())
        # Bar 2: 5R fires; trail_sl=5010 after; low=5015>trail_sl → no trail yet
        s.process_bar(5110.0, 5015.0, _cfg())
        # Bar 3: reaches 25R (5500); hard close fires before trailing SL check
        events = s.process_bar(5510.0, 5300.0, _cfg())
        types = [e[0] for e in events]
        assert "partial" in types
        assert s.remaining_qty == pytest.approx(0.0)

    def test_trailing_sl_closes_runner(self):
        s = _state(entry=5000, sl=4980, qty=1.0)
        # Bar 1: 3R fires (high=5065, low=5001 above BE)
        s.process_bar(5065.0, 5001.0, _cfg())
        # Bar 2: 5R fires + trail activates (high=5150 → trail@5050), low above trail
        s.process_bar(5150.0, 5060.0, _cfg())
        assert s.trailing_sl is not None
        # Bar 3: price reverses below trailing SL (~5050)
        events = s.process_bar(5060.0, 5040.0, _cfg())
        types = [e[0] for e in events]
        assert "trail" in types
        assert s.remaining_qty == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────────────────────
# PartialCloseConfig — default()
# ──────────────────────────────────────────────────────────────────────────────

class TestPartialCloseConfig:
    def test_default_has_two_levels(self):
        cfg = PartialCloseConfig.default()
        assert len(cfg.levels) == 2

    def test_default_first_level_3r_75pct(self):
        cfg = PartialCloseConfig.default()
        lvl = cfg.levels[0]
        assert lvl.r_multiple == pytest.approx(3.0)
        assert lvl.close_fraction == pytest.approx(0.75)
        assert lvl.sl_to_r == pytest.approx(0.0)

    def test_default_second_level_5r_40pct_of_remaining(self):
        cfg = PartialCloseConfig.default()
        lvl = cfg.levels[1]
        assert lvl.r_multiple == pytest.approx(5.0)
        assert lvl.close_fraction == pytest.approx(0.40)
        assert lvl.sl_to_r is None

    def test_default_trailing_activates_at_5r(self):
        cfg = PartialCloseConfig.default()
        assert cfg.trailing.activate_at_r == pytest.approx(5.0)
        assert cfg.trailing.hard_close_r  == pytest.approx(25.0)


# ──────────────────────────────────────────────────────────────────────────────
# PartialCloseConfig — mtf_smc()
# ──────────────────────────────────────────────────────────────────────────────

class TestPartialCloseConfigMtfSmc:
    def test_has_four_levels(self):
        cfg = PartialCloseConfig.mtf_smc()
        assert len(cfg.levels) == 4

    def test_level_0_breakeven_only(self):
        cfg = PartialCloseConfig.mtf_smc()
        lvl = cfg.levels[0]
        assert lvl.r_multiple      == pytest.approx(1.0)
        assert lvl.close_fraction  == pytest.approx(0.0)
        assert lvl.sl_to_r         == pytest.approx(0.0)

    def test_level_1_3r_60pct(self):
        cfg = PartialCloseConfig.mtf_smc()
        lvl = cfg.levels[1]
        assert lvl.r_multiple      == pytest.approx(3.0)
        assert lvl.close_fraction  == pytest.approx(0.60)
        assert lvl.sl_to_r is None

    def test_level_2_5r_625pct_of_remaining(self):
        cfg = PartialCloseConfig.mtf_smc()
        lvl = cfg.levels[2]
        assert lvl.r_multiple      == pytest.approx(5.0)
        assert lvl.close_fraction  == pytest.approx(0.625)
        assert lvl.sl_to_r is None

    def test_level_3_10r_full_exit(self):
        cfg = PartialCloseConfig.mtf_smc()
        lvl = cfg.levels[3]
        assert lvl.r_multiple      == pytest.approx(10.0)
        assert lvl.close_fraction  == pytest.approx(1.0)
        assert lvl.sl_to_r is None

    def test_trailing_effectively_disabled(self):
        cfg = PartialCloseConfig.mtf_smc()
        assert cfg.trailing.activate_at_r > 100.0
        assert cfg.trailing.hard_close_r  > 100.0


# ──────────────────────────────────────────────────────────────────────────────
# process_bar — mtf_smc profile lifecycle
# ──────────────────────────────────────────────────────────────────────────────

def _mtf_cfg() -> PartialCloseConfig:
    return PartialCloseConfig.mtf_smc()


class TestProcessBarMtfSmc:
    """
    Entry=5000, SL=4980 → sl_distance=20 (20 pips), qty=1.0
    R levels:
      1R  = 5020   (BE trigger)
      3R  = 5060   (60 % close)
      5R  = 5100   (62.5 % of remaining close; 85 % total)
      10R = 5200   (100 % close)
    """

    def _long(self, qty: float = 1.0) -> PartialCloseState:
        return PartialCloseState(entry_price=5000, sl_price=4980,
                                 is_long=True, original_qty=qty)

    def _short(self, qty: float = 1.0) -> PartialCloseState:
        return PartialCloseState(entry_price=5000, sl_price=5020,
                                 is_long=False, original_qty=qty)

    # ── BE at 1R ──────────────────────────────────────────────────────────

    def test_be_fires_at_1r(self):
        # low=5001 keeps price above the newly-set BE=5000 within this bar
        s = self._long()
        events = s.process_bar(5025.0, 5001.0, _mtf_cfg())
        types = [e[0] for e in events]
        assert "be" in types

    def test_be_event_qty_is_zero(self):
        s = self._long()
        events = s.process_bar(5025.0, 5001.0, _mtf_cfg())
        be_events = [e for e in events if e[0] == "be"]
        assert len(be_events) == 1
        assert be_events[0][2] == pytest.approx(0.0)

    def test_be_moves_sl_to_entry(self):
        s = self._long()
        s.process_bar(5025.0, 5001.0, _mtf_cfg())
        assert s.sl_price == pytest.approx(5000.0)

    def test_no_qty_closed_at_1r(self):
        # low=5001 keeps price above the newly-set BE=5000 → no SL hit
        s = self._long()
        s.process_bar(5025.0, 5001.0, _mtf_cfg())
        assert s.remaining_qty == pytest.approx(1.0)

    def test_nothing_fires_below_1r(self):
        s = self._long()
        events = s.process_bar(5018.0, 4995.0, _mtf_cfg())  # high < 1R = 5020
        assert events == []
        assert s.sl_price == pytest.approx(4980.0)   # SL unchanged

    # ── 3R: close 60 % ────────────────────────────────────────────────────

    def test_3r_closes_60pct(self):
        s = self._long()
        s.process_bar(5065.0, 5001.0, _mtf_cfg())   # clears 1R and 3R in one bar
        assert s.remaining_qty == pytest.approx(0.40)

    def test_3r_event_type_partial(self):
        s = self._long()
        events = s.process_bar(5065.0, 5001.0, _mtf_cfg())
        types = [e[0] for e in events]
        assert "partial" in types

    def test_3r_partial_pnl_correct(self):
        # 60 % of 1.0 qty closed at 5060; PnL = (5060-5000)*0.6 = 36
        s = self._long()
        s.process_bar(5065.0, 5001.0, _mtf_cfg())
        assert s.partial_pnl == pytest.approx(36.0)

    def test_3r_sl_unchanged_from_be(self):
        """After 3R, SL was already moved to BE at 1R — still at entry."""
        s = self._long()
        s.process_bar(5065.0, 5001.0, _mtf_cfg())
        assert s.sl_price == pytest.approx(5000.0)

    # ── 5R: 62.5 % of remaining → 85 % total ─────────────────────────────

    def test_5r_leaves_15pct_remaining(self):
        s = self._long()
        s.process_bar(5065.0, 5001.0, _mtf_cfg())  # 1R + 3R
        s.process_bar(5105.0, 5055.0, _mtf_cfg())  # 5R (high > 5100)
        assert s.remaining_qty == pytest.approx(0.15)

    def test_5r_levels_hit_is_3(self):
        s = self._long()
        s.process_bar(5065.0, 5001.0, _mtf_cfg())
        s.process_bar(5105.0, 5055.0, _mtf_cfg())
        assert s.levels_hit == 3

    # ── 10R: 100 % → full exit ────────────────────────────────────────────

    def test_10r_closes_all_remaining(self):
        s = self._long()
        s.process_bar(5065.0, 5001.0, _mtf_cfg())   # 1R + 3R
        s.process_bar(5105.0, 5055.0, _mtf_cfg())   # 5R
        s.process_bar(5210.0, 5150.0, _mtf_cfg())   # 10R (high > 5200)
        assert s.remaining_qty == pytest.approx(0.0)

    def test_10r_all_four_levels_hit(self):
        s = self._long()
        s.process_bar(5065.0, 5001.0, _mtf_cfg())
        s.process_bar(5105.0, 5055.0, _mtf_cfg())
        s.process_bar(5210.0, 5150.0, _mtf_cfg())
        assert s.levels_hit == 4

    # ── SL and BE protection ──────────────────────────────────────────────

    def test_sl_hit_before_1r_closes_position(self):
        s = self._long()
        events = s.process_bar(5010.0, 4975.0, _mtf_cfg())  # low < SL=4980
        types = [e[0] for e in events]
        assert "sl" in types
        assert s.remaining_qty == pytest.approx(0.0)

    def test_sl_at_be_protects_after_1r(self):
        """After 1R, SL is at entry (5000). A drop to 4995 must close remaining."""
        s = self._long()
        s.process_bar(5025.0, 5001.0, _mtf_cfg())  # 1R fires, SL → 5000
        events = s.process_bar(5010.0, 4995.0, _mtf_cfg())  # low < BE=5000
        types = [e[0] for e in events]
        assert "sl" in types
        assert s.remaining_qty == pytest.approx(0.0)

    # ── Short position ────────────────────────────────────────────────────

    def test_short_be_fires_at_1r(self):
        """Short: entry=5000, SL=5020, 1R = 4980 (price must drop to 4980)."""
        s = self._short()
        events = s.process_bar(4999.0, 4975.0, _mtf_cfg())  # low < 4980
        types = [e[0] for e in events]
        assert "be" in types
        assert s.sl_price == pytest.approx(5000.0)

    def test_short_3r_closes_60pct(self):
        s = self._short()
        s.process_bar(4999.0, 4935.0, _mtf_cfg())  # 1R + 3R (3R=4940)
        assert s.remaining_qty == pytest.approx(0.40)
