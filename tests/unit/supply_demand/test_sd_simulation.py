"""
Tests for zeus.backtest.sd_simulation — Priority 3 simulation engine.

Verifies:
  - TP recalculated from effective entry (not MSS close)
  - Spread applied at entry (long: open + spread, short: open − spread)
  - Pessimistic SL/TP conflict: SL wins
  - Correct win/loss P&L accounting
  - Expired trade returns None
"""
from __future__ import annotations

import pandas as pd
import pytest

from zeus.backtest.sd_simulation import (
    SPREAD_PER_OZ,
    TradeResult,
    simulate_trade,
)
from zeus.strategy.supply_demand.pivot_candle import PivotCandle, PivotSide
from zeus.strategy.supply_demand.sd_strategy import SDSignal
from zeus.strategy.supply_demand.wyckoff import WyckoffPattern
from zeus.strategy.supply_demand.zone_detector import SDZone, ZoneScore

# ── Helpers ───────────────────────────────────────────────────────────────────

_T0 = pd.Timestamp("2026-06-01 09:00")

SPREAD = 0.30   # match module default


def _pivot(ts: pd.Timestamp = _T0, side: PivotSide = PivotSide.DEMAND) -> PivotCandle:
    return PivotCandle(
        index=ts, open=1900.0, high=1920.0, low=1880.0, close=1910.0,
        side=side, score=8.0,
        body_high=1910.0, body_low=1900.0, wick_high=1920.0, wick_low=1880.0,
        upper_wick_ratio=0.10, lower_wick_ratio=0.80, body_ratio=0.10,
    )


def _demand_zone() -> SDZone:
    return SDZone(
        side=PivotSide.DEMAND,
        zone_top=1910.0, zone_bottom=1900.0, wick_extreme=1880.0,
        pivot=_pivot(side=PivotSide.DEMAND),
        pivot_bar=2, formed_at=_T0, bos_bar=5, bos_level=1920.0,
        base_candles=2,
        score=ZoneScore(bos=2.0, impulse=2.0, time=2.0, fresh=2.0, sweep=0.0),
    )


def _supply_zone() -> SDZone:
    return SDZone(
        side=PivotSide.SUPPLY,
        zone_top=1920.0, zone_bottom=1910.0, wick_extreme=1940.0,
        pivot=_pivot(side=PivotSide.SUPPLY),
        pivot_bar=2, formed_at=_T0, bos_bar=5, bos_level=1900.0,
        base_candles=2,
        score=ZoneScore(bos=2.0, impulse=2.0, time=2.0, fresh=2.0, sweep=0.0),
    )


def _wyckoff_demand(mss_bar: int) -> WyckoffPattern:
    return WyckoffPattern(
        side=PivotSide.DEMAND,
        accum_high=1910.0, accum_low=1900.0, accum_bars=4,
        manip_bar=mss_bar - 1, manip_extreme=1880.0,
        mss_bar=mss_bar, mss_close=1915.0,
        score=7.0, formed_at=_T0,
    )


def _wyckoff_supply(mss_bar: int) -> WyckoffPattern:
    return WyckoffPattern(
        side=PivotSide.SUPPLY,
        accum_high=1920.0, accum_low=1910.0, accum_bars=4,
        manip_bar=mss_bar - 1, manip_extreme=1940.0,
        mss_bar=mss_bar, mss_close=1908.0,
        score=7.0, formed_at=_T0,
    )


def _signal_demand(mss_bar: int = 4) -> SDSignal:
    """
    Long signal: mss_close=1915, sl=1880.
    Original TP = 1915 + 3*(1915-1880) = 1915 + 105 = 2020
    After spread: entry = open+0.30, TP recalculated.
    """
    zone = _demand_zone()
    wy   = _wyckoff_demand(mss_bar)
    entry = wy.mss_close
    sl    = zone.wick_extreme
    risk  = abs(entry - sl)
    tp    = entry + 3.0 * risk
    return SDSignal(
        direction="long", entry_price=entry, stop_loss=sl, take_profit=tp,
        risk_reward=3.0, zone_score=8.0, wyckoff_score=7.0, fib_bonus=0.0,
        composite_score=8.0, zone=zone, wyckoff=wy, fib=None,
        formed_at=_T0, bar_index=mss_bar,
    )


def _signal_supply(mss_bar: int = 4) -> SDSignal:
    zone = _supply_zone()
    wy   = _wyckoff_supply(mss_bar)
    entry = wy.mss_close
    sl    = zone.wick_extreme
    risk  = abs(entry - sl)
    tp    = entry - 3.0 * risk
    return SDSignal(
        direction="short", entry_price=entry, stop_loss=sl, take_profit=tp,
        risk_reward=3.0, zone_score=8.0, wyckoff_score=7.0, fib_bonus=0.0,
        composite_score=8.0, zone=zone, wyckoff=wy, fib=None,
        formed_at=_T0, bar_index=mss_bar,
    )


def _m1(opens: list[float], highs: list[float],
        lows:  list[float], closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range(_T0, periods=len(opens), freq="1min")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=idx,
    )


# ── Spread tests ──────────────────────────────────────────────────────────────

def test_long_entry_is_open_plus_spread():
    """Effective entry for a long must be open + SPREAD_PER_OZ."""
    sig = _signal_demand(mss_bar=4)
    # bar 5 is the entry bar (mss_bar + 1); open = 1910
    m1 = _m1(
        opens  = [0]*5 + [1910.0, 1910.0, 1910.0, 1910.0, 2100.0],
        highs  = [0]*5 + [1920.0, 1920.0, 1920.0, 1920.0, 2200.0],
        lows   = [0]*5 + [1905.0, 1905.0, 1905.0, 1905.0, 2080.0],
        closes = [0]*5 + [1912.0, 1912.0, 1912.0, 1912.0, 2100.0],
    )
    result = simulate_trade(sig, m1, equity=10_000.0, spread=SPREAD)
    assert result is not None
    expected_entry = 1910.0 + SPREAD
    assert result.entry_price == pytest.approx(expected_entry, abs=1e-4)


def test_short_entry_is_open_minus_spread():
    """Effective entry for a short must be open − SPREAD_PER_OZ."""
    sig = _signal_supply(mss_bar=4)
    # bar 5 is the entry bar; open = 1920, SL = 1940, TP recalc below
    m1 = _m1(
        opens  = [0]*5 + [1920.0, 1920.0, 1920.0, 1920.0, 1700.0],
        highs  = [0]*5 + [1925.0, 1925.0, 1925.0, 1925.0, 1730.0],
        lows   = [0]*5 + [1915.0, 1915.0, 1915.0, 1915.0, 1690.0],
        closes = [0]*5 + [1918.0, 1918.0, 1918.0, 1918.0, 1700.0],
    )
    result = simulate_trade(sig, m1, equity=10_000.0, spread=SPREAD)
    assert result is not None
    expected_entry = 1920.0 - SPREAD
    assert result.entry_price == pytest.approx(expected_entry, abs=1e-4)


# ── TP recalculation tests ────────────────────────────────────────────────────

def test_long_tp_recalculated_from_effective_entry():
    """Long TP = effective_entry + 3 × |effective_entry − sl|, NOT signal.take_profit."""
    sig = _signal_demand(mss_bar=4)
    # Put TP far above so only the recalculated TP matters; SL far below
    effective_entry = 1910.0 + SPREAD           # 1910.30
    sl              = sig.stop_loss             # 1880
    expected_tp     = effective_entry + 3 * (effective_entry - sl)  # 1910.30 + 3*30.30

    # Build an m1 that reaches the expected TP on bar 9
    tp_price = expected_tp
    m1 = _m1(
        opens  = [0]*5 + [1910.0] + [1910.0]*3 + [tp_price + 5],
        highs  = [0]*5 + [1920.0] + [1920.0]*3 + [tp_price + 10],
        lows   = [0]*5 + [1905.0] + [1905.0]*3 + [tp_price],
        closes = [0]*5 + [1912.0] + [1912.0]*3 + [tp_price + 2],
    )
    result = simulate_trade(sig, m1, equity=10_000.0, spread=SPREAD)
    assert result is not None
    assert result.outcome == "win"
    assert result.exit_price == pytest.approx(expected_tp, abs=0.01)


def test_signal_original_tp_not_used():
    """
    Verify the trade does NOT exit at signal.take_profit when that differs
    from the recalculated TP.  The signal TP (based on mss_close) is higher
    than the recalculated TP (based on effective entry + spread < mss_close)
    because spread pushes the entry higher for longs → TP also shifts up.
    """
    sig = _signal_demand(mss_bar=4)
    original_tp     = sig.take_profit          # 1915 + 3*(1915-1880) = 2020
    effective_entry = 1910.0 + SPREAD
    recalc_tp       = effective_entry + 3 * (effective_entry - sig.stop_loss)

    # The signal TP (2020) and recalc_tp differ only marginally here; let's
    # test by manufacturing a scenario where mss_close != open of next bar.
    # sig.take_profit = 2020 (from mss_close=1915, sl=1880)
    # recalc_tp = 1910.30 + 3*(1910.30-1880) = 1910.30 + 90.9 = 2001.2
    # So original_tp > recalc_tp.
    # If we set the bar to just touch recalc_tp but NOT original_tp, and
    # the trade exits, it used recalc_tp.
    mid = (recalc_tp + original_tp) / 2
    m1 = _m1(
        opens  = [0]*5 + [1910.0, 1910.0, 1910.0, 1910.0, mid],
        highs  = [0]*5 + [1920.0, 1920.0, 1920.0, 1920.0, recalc_tp + 1],
        lows   = [0]*5 + [1905.0, 1905.0, 1905.0, 1905.0, mid - 5],
        closes = [0]*5 + [1912.0, 1912.0, 1912.0, 1912.0, mid],
    )
    result = simulate_trade(sig, m1, equity=10_000.0, spread=SPREAD)
    assert result is not None
    assert result.outcome == "win"
    # Exit must be at recalc_tp, not original_tp
    assert result.exit_price == pytest.approx(recalc_tp, abs=0.01)
    assert abs(result.exit_price - original_tp) > 1.0  # meaningfully different


# ── Spread cost tests ─────────────────────────────────────────────────────────

def test_spread_cost_deducted_from_pnl():
    """Spread cost is positive and appears in pnl_usd correctly."""
    sig = _signal_demand(mss_bar=4)
    m1 = _m1(
        opens  = [0]*5 + [1910.0, 1910.0, 1910.0, 1910.0, 1870.0],
        highs  = [0]*5 + [1920.0, 1920.0, 1920.0, 1920.0, 1890.0],
        lows   = [0]*5 + [1905.0, 1905.0, 1905.0, 1905.0, 1865.0],
        closes = [0]*5 + [1912.0, 1912.0, 1912.0, 1912.0, 1870.0],
    )
    result = simulate_trade(sig, m1, equity=10_000.0, spread=SPREAD)
    assert result is not None
    assert result.outcome == "loss"
    assert result.spread_cost_usd > 0
    # With spread, actual risk is slightly larger than 1R
    assert result.pnl_usd < -100.0


def test_spread_zero_uses_raw_open():
    """With spread=0, effective_entry equals the raw bar open."""
    sig = _signal_demand(mss_bar=4)
    m1 = _m1(
        opens  = [0]*5 + [1910.0, 1910.0, 1910.0, 1910.0, 1870.0],
        highs  = [0]*5 + [1920.0, 1920.0, 1920.0, 1920.0, 1890.0],
        lows   = [0]*5 + [1905.0, 1905.0, 1905.0, 1905.0, 1865.0],
        closes = [0]*5 + [1912.0, 1912.0, 1912.0, 1912.0, 1870.0],
    )
    result = simulate_trade(sig, m1, equity=10_000.0, spread=0.0)
    assert result is not None
    assert result.entry_price == pytest.approx(1910.0, abs=1e-4)
    assert result.spread_cost_usd == 0.0


# ── Pessimistic conflict ──────────────────────────────────────────────────────

def test_sl_wins_when_both_hit_in_same_bar():
    """If a bar touches both SL and TP, the trade is closed at SL (pessimistic)."""
    sig = _signal_demand(mss_bar=4)
    # Bar 5 is entry (open=1910). Bar 6 touches both SL(1880) and TP.
    effective_entry = 1910.0 + SPREAD
    recalc_tp = effective_entry + 3 * (effective_entry - 1880.0)
    m1 = _m1(
        opens  = [0]*5 + [1910.0, 1900.0],
        highs  = [0]*5 + [1920.0, recalc_tp + 5],   # TP touched
        lows   = [0]*5 + [1905.0, 1875.0],           # SL touched
        closes = [0]*5 + [1912.0, 1895.0],
    )
    result = simulate_trade(sig, m1, equity=10_000.0, spread=SPREAD)
    assert result is not None
    assert result.outcome == "loss"
    assert result.exit_price == pytest.approx(1880.0, abs=1e-4)


# ── Expiry test ───────────────────────────────────────────────────────────────

def test_expired_trade_returns_none():
    """Trade that never reaches SL or TP returns None."""
    sig = _signal_demand(mss_bar=4)
    m1 = _m1(
        opens  = [0]*5 + [1910.0, 1910.0, 1910.0],
        highs  = [0]*5 + [1912.0, 1912.0, 1912.0],
        lows   = [0]*5 + [1908.0, 1908.0, 1908.0],
        closes = [0]*5 + [1910.0, 1910.0, 1910.0],
    )
    result = simulate_trade(sig, m1, equity=10_000.0, spread=SPREAD)
    assert result is None


def test_entry_bar_beyond_data_returns_none():
    """Signal at the last bar of the dataset returns None (no entry bar)."""
    sig = _signal_demand(mss_bar=4)
    m1 = _m1(
        opens=[1910.0]*5, highs=[1920.0]*5,
        lows=[1905.0]*5,  closes=[1912.0]*5,
    )
    result = simulate_trade(sig, m1, equity=10_000.0, spread=SPREAD)
    assert result is None


# ── Non-SDSignal Tradeable tests (F-03) ───────────────────────────────────────

from dataclasses import dataclass as _dc
from zeus.backtest.sd_simulation import simulate_all, Tradeable


@_dc(frozen=True)
class _SimpleSignal:
    """Minimal Tradeable — not an SDSignal."""
    direction:   str
    bar_index:   int
    stop_loss:   float
    risk_reward: float
    formed_at:   pd.Timestamp
    zone_score:  float
    entry_price: float = 0.0  # unused by simulate_trade; documented as duck-typed


def test_tradeable_protocol_accepted():
    """simulate_trade() must accept any Tradeable, not just SDSignal (F-03)."""
    sig = _SimpleSignal(
        direction   = "long",
        bar_index   = 4,
        stop_loss   = 1880.0,
        risk_reward = 2.0,
        formed_at   = _T0,
        zone_score  = 5.0,
    )
    m1 = _m1(
        opens  = [0]*5 + [1910.0, 1910.0, 1910.0, 1910.0, 2000.0],
        highs  = [0]*5 + [1920.0, 1920.0, 1920.0, 1920.0, 2050.0],
        lows   = [0]*5 + [1905.0, 1905.0, 1905.0, 1905.0, 1990.0],
        closes = [0]*5 + [1912.0, 1912.0, 1912.0, 1912.0, 2000.0],
    )
    result = simulate_trade(sig, m1, equity=10_000.0, spread=SPREAD)
    assert result is not None
    assert result.outcome in ("win", "loss", "scratch")
    assert result.entry_price == pytest.approx(1910.0 + SPREAD, abs=1e-4)


def test_simulate_all_with_non_sd_signal():
    """simulate_all() returns correct trade count with non-SDSignal Tradeables (F-02 / F-03)."""
    sig1 = _SimpleSignal(
        direction="long", bar_index=4, stop_loss=1880.0,
        risk_reward=2.0, formed_at=_T0, zone_score=5.0,
    )
    sig2 = _SimpleSignal(
        direction="long", bar_index=8, stop_loss=1850.0,
        risk_reward=2.0, formed_at=_T0 + pd.Timedelta(minutes=60), zone_score=4.0,
    )
    m1 = _m1(
        opens  = [0]*5 + [1910.0]*5 + [1950.0]*5,
        highs  = [0]*5 + [1920.0]*5 + [2100.0]*5,
        lows   = [0]*5 + [1905.0]*5 + [1870.0]*5,
        closes = [0]*5 + [1912.0]*5 + [2080.0]*5,
    )
    results, n_exp = simulate_all(
        [sig1, sig2], m1, risk_pct=0.01, spread=SPREAD,
        initial_equity=10_000.0,
    )
    assert isinstance(results, list)
    assert isinstance(n_exp, int)
    assert len(results) + n_exp == 2


def test_simulate_all_initial_equity_used():
    """simulate_all() must start from the caller-supplied initial_equity (F-02)."""
    sig = _SimpleSignal(
        direction="long", bar_index=4, stop_loss=1880.0,
        risk_reward=2.0, formed_at=_T0, zone_score=5.0,
    )
    m1 = _m1(
        opens  = [0]*5 + [1910.0, 1910.0, 1910.0, 1910.0, 1870.0],
        highs  = [0]*5 + [1920.0, 1920.0, 1920.0, 1920.0, 1890.0],
        lows   = [0]*5 + [1905.0, 1905.0, 1905.0, 1905.0, 1865.0],
        closes = [0]*5 + [1912.0, 1912.0, 1912.0, 1912.0, 1870.0],
    )
    # equity_at_entry should reflect the caller's initial equity, not a hardcoded 10_000
    results_50k, _ = simulate_all([sig], m1, risk_pct=0.01, spread=SPREAD, initial_equity=50_000.0)
    results_10k, _ = simulate_all([sig], m1, risk_pct=0.01, spread=SPREAD, initial_equity=10_000.0)
    if results_50k and results_10k:
        assert results_50k[0].equity_at_entry == pytest.approx(50_000.0)
        assert results_10k[0].equity_at_entry == pytest.approx(10_000.0)
        # 5× equity → 5× USD loss
        assert abs(results_50k[0].pnl_usd) == pytest.approx(
            abs(results_10k[0].pnl_usd) * 5, rel=0.01
        )


# ── Slippage on SL exit (F-09) ───────────────────────────────────────────────

def test_slippage_applied_on_sl_exit():
    """slippage_ticks must worsen the SL fill, increasing the loss (F-09)."""
    sig = _signal_demand(mss_bar=4)
    m1 = _m1(
        opens  = [0]*5 + [1910.0, 1910.0, 1910.0, 1910.0, 1870.0],
        highs  = [0]*5 + [1920.0, 1920.0, 1920.0, 1920.0, 1890.0],
        lows   = [0]*5 + [1905.0, 1905.0, 1905.0, 1905.0, 1865.0],
        closes = [0]*5 + [1912.0, 1912.0, 1912.0, 1912.0, 1870.0],
    )
    res_no = simulate_trade(sig, m1, equity=10_000.0, spread=SPREAD, slippage_ticks=0.0)
    res_sl = simulate_trade(sig, m1, equity=10_000.0, spread=SPREAD, slippage_ticks=0.10)

    assert res_no is not None and res_sl is not None
    assert res_no.outcome == "loss"
    assert res_sl.outcome == "loss"
    # Slipped exit is lower (worse for the long)
    assert res_sl.exit_price < res_no.exit_price
    assert res_no.exit_price - res_sl.exit_price == pytest.approx(0.10, abs=1e-4)
    # Slipped trade loses more money
    assert res_sl.pnl_usd < res_no.pnl_usd


def test_slippage_zero_exit_price_equals_sl():
    """With slippage_ticks=0, SL exit price must equal the exact SL level."""
    sig = _signal_demand(mss_bar=4)
    m1 = _m1(
        opens  = [0]*5 + [1910.0, 1910.0, 1910.0, 1910.0, 1870.0],
        highs  = [0]*5 + [1920.0, 1920.0, 1920.0, 1920.0, 1890.0],
        lows   = [0]*5 + [1905.0, 1905.0, 1905.0, 1905.0, 1865.0],
        closes = [0]*5 + [1912.0, 1912.0, 1912.0, 1912.0, 1870.0],
    )
    result = simulate_trade(sig, m1, equity=10_000.0, spread=SPREAD, slippage_ticks=0.0)
    assert result is not None
    assert result.outcome == "loss"
    assert result.exit_price == pytest.approx(sig.stop_loss, abs=1e-4)
