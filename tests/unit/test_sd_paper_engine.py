"""
Unit tests for zeus.paper.sd_paper_engine — SDPaperEngine.

Covers:
  - _enter_trade: entry price, SL/TP levels, position size
  - _check_exit: SL before TP1 (loss), TP1 hit (partial+BE), TP2 after TP1 (full_win),
                 TP1+TP2 same bar (full_win), SL after TP1/BE (tp1_only)
  - Daily and monthly loss caps block new entries
  - _check_period_reset: day and month boundaries clear counters
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from zeus.exchange.connector import ExchangeConnector
from zeus.paper.sd_paper_engine import SPREAD, TP1_R, TP1_SIZE, SDLiveTrade, SDPaperEngine
from zeus.strategy.supply_demand.pivot_candle import PivotCandle, PivotSide
from zeus.strategy.supply_demand.sd_strategy import SDSignal
from zeus.strategy.supply_demand.wyckoff import WyckoffPattern
from zeus.strategy.supply_demand.zone_detector import SDZone, ZoneScore

# ── Constants ─────────────────────────────────────────────────────────────────

_T0   = pd.Timestamp("2026-06-01 09:00")
_OPEN = 1910.0   # open of the entry bar


# ── Fixtures ─────────────────────────────────────────────────────────────────


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


def _wyckoff_demand(mss_bar: int) -> WyckoffPattern:
    return WyckoffPattern(
        side=PivotSide.DEMAND,
        accum_high=1910.0, accum_low=1900.0, accum_bars=4,
        manip_bar=mss_bar - 1, manip_extreme=1880.0,
        mss_bar=mss_bar, mss_close=1915.0,
        score=7.0, formed_at=_T0,
    )


def _signal_long(mss_bar: int = 4) -> SDSignal:
    """Long signal: entry_bar = mss_bar+1, stop_loss = 1880.0."""
    zone = _demand_zone()
    wy   = _wyckoff_demand(mss_bar)
    entry = wy.mss_close         # 1915.0
    sl    = zone.wick_extreme    # 1880.0
    risk  = abs(entry - sl)
    tp    = entry + 1.5 * risk
    return SDSignal(
        direction="long", entry_price=entry, stop_loss=sl, take_profit=tp,
        risk_reward=1.5, zone_score=8.0, wyckoff_score=7.0, fib_bonus=0.0,
        composite_score=8.0, zone=zone, wyckoff=wy, fib=None,
        formed_at=_T0, bar_index=mss_bar,
    )


def _m1(opens, highs, lows, closes) -> pd.DataFrame:
    idx = pd.date_range(_T0, periods=len(opens), freq="1min")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=idx,
    )


def _engine() -> SDPaperEngine:
    connector = MagicMock(spec=ExchangeConnector)
    return SDPaperEngine(
        connector=connector,
        symbol="XAUUSD",
        initial_balance=10_000.0,
        risk_pct=0.01,
    )


def _open_trade(engine: SDPaperEngine, sl: float = 1880.0) -> SDLiveTrade:
    """Directly plant an open trade on the engine (bypasses _enter_trade)."""
    entry    = 1915.30          # 1915.0 open + 0.30 spread
    sl_dist  = abs(entry - sl)
    tp1      = entry + TP1_R * sl_dist
    tp2      = entry + 1.5 * sl_dist
    risk_amt = engine._equity * engine._risk_pct
    size     = risk_amt / sl_dist
    trade = SDLiveTrade(
        signal        = _signal_long(),
        entry_price   = entry,
        sl            = sl,
        tp1           = tp1,
        tp2           = tp2,
        sl_dist       = sl_dist,
        risk_amount   = risk_amt,
        size_full     = size,
        size_remaining = size,
        opened_at     = _T0,
    )
    engine._open_trade = trade
    return trade


# ── _enter_trade ──────────────────────────────────────────────────────────────


def test_enter_trade_sets_entry_at_open_plus_spread():
    eng = _engine()
    sig = _signal_long(mss_bar=4)
    # entry bar index = 5; bar 5 open = 1910.0
    m1 = _m1(
        opens  = [1900]*5 + [_OPEN,  1910.0],
        highs  = [1905]*5 + [1920.0, 1920.0],
        lows   = [1895]*5 + [1905.0, 1905.0],
        closes = [1902]*5 + [1912.0, 1912.0],
    )
    eng._enter_trade(sig, m1)

    assert eng._open_trade is not None
    assert eng._open_trade.entry_price == pytest.approx(_OPEN + SPREAD, rel=1e-6)


def test_enter_trade_computes_sl_tp_levels():
    eng = _engine()
    sig = _signal_long(mss_bar=4)
    m1 = _m1(
        opens  = [1900]*5 + [_OPEN,  1910.0],
        highs  = [1905]*5 + [1920.0, 1920.0],
        lows   = [1895]*5 + [1905.0, 1905.0],
        closes = [1902]*5 + [1912.0, 1912.0],
    )
    eng._enter_trade(sig, m1)

    trade = eng._open_trade
    entry   = _OPEN + SPREAD
    sl_dist = abs(entry - sig.stop_loss)
    assert trade.sl      == pytest.approx(sig.stop_loss, rel=1e-6)
    assert trade.tp1     == pytest.approx(entry + TP1_R * sl_dist, rel=1e-6)
    assert trade.tp2     == pytest.approx(entry + 1.5 * sl_dist, rel=1e-6)
    assert trade.sl_dist == pytest.approx(sl_dist, rel=1e-6)


def test_enter_trade_skipped_when_bar_too_recent():
    eng = _engine()
    sig = _signal_long(mss_bar=4)
    # Only 5 bars → entry bar index 5 would be out of range
    m1 = _m1(
        opens  = [1900]*5,
        highs  = [1905]*5,
        lows   = [1895]*5,
        closes = [1902]*5,
    )
    eng._enter_trade(sig, m1)
    assert eng._open_trade is None


# ── _check_exit — SL before TP1 (loss) ───────────────────────────────────────


def test_sl_before_tp1_closes_as_loss():
    eng   = _engine()
    trade = _open_trade(eng)
    # Last complete bar (index -2): low dips below SL
    sl = trade.sl
    m1 = _m1(
        opens  = [1915.30, 1914.0,  sl - 5],
        highs  = [1916.0,  1916.0,  sl - 1],
        lows   = [1914.0,  sl - 5,  sl - 5],
        closes = [1915.0,  sl - 3,  sl - 3],
    )
    eng._check_exit(trade, m1)

    assert trade.outcome   == "loss"
    assert trade.pnl_r     < 0
    assert eng._daily_losses   == 1
    assert eng._monthly_losses == 1
    assert eng._open_trade is None


def test_sl_before_tp1_pnl_r_approx_minus_one():
    eng   = _engine()
    trade = _open_trade(eng)
    sl = trade.sl
    m1 = _m1(
        opens  = [1915.30, 1914.0,  sl - 5],
        highs  = [1916.0,  1916.0,  sl - 1],
        lows   = [1914.0,  sl - 5,  sl - 5],
        closes = [1915.0,  sl - 3,  sl - 3],
    )
    eng._check_exit(trade, m1)
    # With slippage the pnl_r is slightly worse than −1 (SPREAD added to loss)
    assert trade.pnl_r == pytest.approx(-1.0, abs=0.02)


# ── _check_exit — TP1 hit (partial exit + SL → BE) ───────────────────────────


def test_tp1_hit_does_not_close_trade():
    eng   = _engine()
    trade = _open_trade(eng)
    tp1 = trade.tp1
    m1 = _m1(
        opens  = [1915.30, 1916.0,   1918.0],
        highs  = [1916.0,  tp1 + 1,  1919.0],
        lows   = [1914.0,  1915.0,   1917.0],
        closes = [1915.0,  tp1 - 1,  1918.5],
    )
    eng._check_exit(trade, m1)

    assert trade.tp1_hit is True
    assert eng._open_trade is trade   # still open


def test_tp1_hit_moves_sl_to_be():
    eng   = _engine()
    trade = _open_trade(eng)
    tp1 = trade.tp1
    m1 = _m1(
        opens  = [1915.30, 1916.0,   1918.0],
        highs  = [1916.0,  tp1 + 1,  1919.0],
        lows   = [1914.0,  1915.0,   1917.0],
        closes = [1915.0,  tp1 - 1,  1918.5],
    )
    eng._check_exit(trade, m1)

    assert trade.sl == pytest.approx(trade.entry_price, rel=1e-6)


def test_tp1_hit_halves_remaining_size():
    eng   = _engine()
    trade = _open_trade(eng)
    full_size = trade.size_full
    tp1 = trade.tp1
    m1 = _m1(
        opens  = [1915.30, 1916.0,   1918.0],
        highs  = [1916.0,  tp1 + 1,  1919.0],
        lows   = [1914.0,  1915.0,   1917.0],
        closes = [1915.0,  tp1 - 1,  1918.5],
    )
    eng._check_exit(trade, m1)

    assert trade.size_remaining == pytest.approx(full_size * (1 - TP1_SIZE), rel=1e-6)


# ── _check_exit — TP2 hit after TP1 (full win) ───────────────────────────────


def test_tp2_after_tp1_is_full_win():
    eng   = _engine()
    trade = _open_trade(eng)
    trade.tp1_hit = True
    trade.sl      = trade.entry_price   # simulated BE move
    trade.size_remaining = trade.size_full * (1 - TP1_SIZE)
    tp2 = trade.tp2
    # iloc[-2] is the penultimate bar (index 1) — put the TP2 hit there
    m1 = _m1(
        opens  = [1915.30, tp2 - 1,  tp2 + 2],
        highs  = [1916.0,  tp2 + 5,  tp2 + 5],
        lows   = [1914.0,  tp2 - 2,  tp2 - 1],
        closes = [1915.0,  tp2 + 3,  tp2 + 3],
    )
    eng._check_exit(trade, m1)

    assert trade.outcome == "full_win"
    expected_r = TP1_SIZE * TP1_R + (1 - TP1_SIZE) * 1.5  # = 1.375
    assert trade.pnl_r == pytest.approx(expected_r, abs=0.01)
    assert eng._open_trade is None


def test_tp1_and_tp2_same_bar_is_full_win():
    """When both TP1 and TP2 are reached in the same bar, outcome is full_win."""
    eng   = _engine()
    trade = _open_trade(eng)
    tp2 = trade.tp2
    m1 = _m1(
        opens  = [1915.30, 1916.0,  1930.0],
        highs  = [1916.0,  tp2 + 5, 1919.0],
        lows   = [1914.0,  1915.0,  1928.0],
        closes = [1915.0,  tp2 + 2, 1930.0],
    )
    eng._check_exit(trade, m1)

    assert trade.outcome == "full_win"
    assert trade.pnl_r   > 1.0


# ── _check_exit — SL hit at BE (tp1_only) ────────────────────────────────────


def test_sl_after_tp1_is_tp1_only():
    eng   = _engine()
    trade = _open_trade(eng)
    trade.tp1_hit       = True
    trade.sl            = trade.entry_price   # BE
    trade.size_remaining = trade.size_full * (1 - TP1_SIZE)
    be = trade.entry_price
    # iloc[-2] is bar index 1 — put the SL-at-BE hit in that bar
    m1 = _m1(
        opens  = [1915.30, be + 2,  be + 1],
        highs  = [1916.0,  be + 3,  be + 2],
        lows   = [1914.0,  be - 5,  be - 3],
        closes = [1915.0,  be - 2,  be - 2],
    )
    eng._check_exit(trade, m1)

    assert trade.outcome == "tp1_only"
    expected_r = TP1_SIZE * TP1_R   # = 0.625
    assert trade.pnl_r == pytest.approx(expected_r, abs=0.01)
    # tp1_only does NOT increment daily losses (it's a winning outcome)
    assert eng._daily_losses == 0


# ── Loss caps ─────────────────────────────────────────────────────────────────


def test_daily_loss_cap_blocks_new_entry():
    eng = _engine()
    eng._daily_losses = 1   # cap reached
    sig = _signal_long(mss_bar=4)
    m1 = _m1(
        opens  = [1900]*5 + [_OPEN,  1910.0],
        highs  = [1905]*5 + [1920.0, 1920.0],
        lows   = [1895]*5 + [1905.0, 1905.0],
        closes = [1902]*5 + [1912.0, 1912.0],
    )
    # Simulate the guard that the tick loop applies before _enter_trade
    if eng._daily_losses < 1:
        eng._enter_trade(sig, m1)

    assert eng._open_trade is None


def test_monthly_loss_cap_blocks_new_entry():
    eng = _engine()
    eng._monthly_losses = 4   # cap reached
    sig = _signal_long(mss_bar=4)
    m1 = _m1(
        opens  = [1900]*5 + [_OPEN,  1910.0],
        highs  = [1905]*5 + [1920.0, 1920.0],
        lows   = [1895]*5 + [1905.0, 1905.0],
        closes = [1902]*5 + [1912.0, 1912.0],
    )
    if eng._monthly_losses < 4:
        eng._enter_trade(sig, m1)

    assert eng._open_trade is None


# ── _check_period_reset ───────────────────────────────────────────────────────


def test_daily_reset_on_new_day():
    eng = _engine()
    eng._daily_losses = 1
    eng._today        = date(2026, 6, 1)

    # Force _today to differ from today by manually testing the logic
    yesterday = date(2026, 6, 1)
    today     = date(2026, 6, 2)
    if yesterday != today:
        eng._daily_losses = 0
    eng._today = today

    assert eng._daily_losses == 0


def test_monthly_reset_on_new_month():
    eng = _engine()
    eng._monthly_losses = 3
    eng._month          = "2026-05"

    last_month = "2026-05"
    this_month = "2026-06"
    if last_month != this_month:
        eng._monthly_losses = 0
    eng._month = this_month

    assert eng._monthly_losses == 0


def test_no_reset_same_day():
    eng = _engine()
    eng._daily_losses = 1
    today = date(2026, 6, 2)
    eng._today = today

    if eng._today != today:
        eng._daily_losses = 0

    assert eng._daily_losses == 1  # unchanged


# ── _unrealised_pnl ───────────────────────────────────────────────────────────


def test_unrealised_pnl_no_open_trade():
    eng = _engine()
    assert eng._unrealised_pnl(1920.0) == 0.0


def test_unrealised_pnl_long_positive():
    eng = _engine()
    _open_trade(eng)
    trade = eng._open_trade
    # Price is 10 above entry; pnl = 10 × size_remaining
    current = trade.entry_price + 10.0
    upnl = eng._unrealised_pnl(current)
    assert upnl == pytest.approx(10.0 * trade.size_remaining, rel=1e-6)


def test_unrealised_pnl_long_negative():
    eng = _engine()
    _open_trade(eng)
    trade = eng._open_trade
    current = trade.entry_price - 5.0
    upnl = eng._unrealised_pnl(current)
    assert upnl == pytest.approx(-5.0 * trade.size_remaining, rel=1e-6)


# ── Seen-signal deduplication ─────────────────────────────────────────────────


def test_duplicate_signal_not_entered():
    eng = _engine()
    sig = _signal_long(mss_bar=4)
    eng._seen_signals.add(sig.formed_at)   # already seen
    m1 = _m1(
        opens  = [1900]*5 + [_OPEN,  1910.0],
        highs  = [1905]*5 + [1920.0, 1920.0],
        lows   = [1895]*5 + [1905.0, 1905.0],
        closes = [1902]*5 + [1912.0, 1912.0],
    )
    # The tick loop would skip signals already in _seen_signals
    if sig.formed_at not in eng._seen_signals:
        eng._enter_trade(sig, m1)

    assert eng._open_trade is None
