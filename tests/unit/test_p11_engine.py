"""Tests du moteur P11 (zeus/live/p11_engine.py) — échelle de sorties, guard, caps."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zeus.live.p11_config import load_p11_config
from zeus.live.p11_engine import P11Engine, P11Trade
from zeus.strategy.supply_demand.sd_strategy import SDSignal

_YAML = Path(__file__).parent.parent.parent / "config" / "p11_v4.yaml"


def _mk_engine(connector=None):
    cfg = load_p11_config(_YAML)
    return P11Engine(cfg, connector)


def _mk_m1(closes: list[float], start="2026-06-01 10:00") -> pd.DataFrame:
    """DataFrame M1 synthétique — high/low élargis autour du close."""
    idx = pd.date_range(start, periods=len(closes), freq="1min")
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": 1.0},
        index=idx,
    )


def _bar(high: float, low: float, ts="2026-06-01 12:00") -> pd.DataFrame:
    idx = pd.date_range(ts, periods=1, freq="1min")
    mid = (high + low) / 2
    return pd.DataFrame(
        {"open": mid, "high": high, "low": low, "close": mid, "volume": 1.0},
        index=idx,
    )


def _open_long(engine, sym="XAUUSD", entry=3000.0, sl=2990.0):
    st = engine.states[sym]
    st.open_trade = P11Trade(
        symbol=sym, direction="long", entry_price=entry, sl=sl,
        sl_dist=entry - sl, exits=st.group.exits,
        opened_at=pd.Timestamp("2026-06-01 11:00"),
    )
    return st


# ── Échelle de sorties XAU V4 (TP1@1R BE · TP2@3R 60% · TP3@8R 85% · Runner@20R) ──

class TestExitsXauV4:
    def test_tp1_moves_sl_to_be_without_closing(self):
        e = _mk_engine()
        st = _open_long(e)                       # 1R = 10 pts
        e._manage_exits("XAUUSD", st, _bar(high=3010.5, low=3005), pd.Timestamp("2026-06-01 12:00"))
        tr = st.open_trade
        assert tr is not None and tr.tp1_done
        assert tr.sl == 3000.0                   # break-even
        assert tr.remaining == 1.0               # tp1_close_pct = 0 en V4

    def test_full_ladder_realizes_expected_r(self):
        e = _mk_engine()
        st = _open_long(e)
        # Une barre géante touche tous les niveaux jusqu'au runner 20R (3200)
        e._manage_exits("XAUUSD", st, _bar(high=3200.5, low=3001),
                        pd.Timestamp("2026-06-01 12:00"))
        assert st.open_trade is None
        trade = e.closed_trades[-1]
        # 60%@3R + 25%@8R + 15%@20R = 1.8 + 2.0 + 3.0 = 6.8R
        assert trade["pnl_r"] == pytest.approx(6.8, abs=1e-6)
        assert trade["reason"] == "RUNNER"
        assert trade["outcome"] == "win"

    def test_sl_before_tp_is_minus_one_r(self):
        e = _mk_engine()
        st = _open_long(e)
        e._manage_exits("XAUUSD", st, _bar(high=3001, low=2989.5),
                        pd.Timestamp("2026-06-01 12:00"))
        assert st.open_trade is None
        trade = e.closed_trades[-1]
        assert trade["pnl_r"] == pytest.approx(-1.0)
        assert trade["outcome"] == "loss"
        # Le compteur de pertes par symbole/direction est incrémenté
        assert st.day_losses["2026-06-01|long"] == 1

    def test_reversal_after_tp2_locks_tp1_level_not_be(self):
        e = _mk_engine()
        st = _open_long(e)                       # 1R = 10 pts, entry 3000, SL 2990
        # Touche TP2 (3030) puis retombe sur l'ancien niveau BE (3000) SANS
        # toucher le nouveau SL ratcheté (3010 = niveau TP1) : le trade doit
        # rester ouvert, protégé par le SL remonté à TP1, pas au BE.
        e._manage_exits("XAUUSD", st, _bar(high=3030.5, low=3025),
                        pd.Timestamp("2026-06-01 12:00"))        # TP1 + TP2
        tr = st.open_trade
        assert tr is not None and tr.tp2_done
        assert tr.sl == pytest.approx(3010.0)     # niveau TP1, pas 3000 (BE)

        e._manage_exits("XAUUSD", st, _bar(high=3015, low=3010.5),
                        pd.Timestamp("2026-06-01 12:30"))         # redescend, SL (3010) pas touché
        assert st.open_trade is not None          # toujours ouvert

        e._manage_exits("XAUUSD", st, _bar(high=3011, low=3009),
                        pd.Timestamp("2026-06-01 13:00"))         # touche le SL ratcheté (3010)
        assert st.open_trade is None
        trade = e.closed_trades[-1]
        # 60%@3R + 40%@1R (reliquat sorti au niveau TP1) = 1.8 + 0.4 = 2.2R
        # (avant le correctif : le reliquat sortait au BE → 1.8 + 0.0 = 1.8R)
        assert trade["pnl_r"] == pytest.approx(2.2, abs=1e-6)
        assert trade["outcome"] == "win"

    def test_be_after_tp1_is_scratch(self):
        e = _mk_engine()
        st = _open_long(e)
        e._manage_exits("XAUUSD", st, _bar(high=3010.5, low=3005),
                        pd.Timestamp("2026-06-01 12:00"))       # TP1 → BE
        e._manage_exits("XAUUSD", st, _bar(high=3005, low=2999.5),
                        pd.Timestamp("2026-06-01 12:30"))       # retour à l'entrée
        assert st.open_trade is None
        trade = e.closed_trades[-1]
        assert trade["pnl_r"] == pytest.approx(0.0)
        assert trade["outcome"] == "scratch"
        assert "2026-06-01|long" not in st.day_losses           # pas une perte

    def test_guard_equity_updated_on_close(self):
        e = _mk_engine()
        st = _open_long(e)
        e._manage_exits("XAUUSD", st, _bar(high=3001, low=2989.5),
                        pd.Timestamp("2026-06-01 12:00"))
        # risque 0.6% de 100k = 600$ → équity 99 400
        assert e.guard.state.equity == pytest.approx(100_000 - 600)


# ── Échelle forex TIERED-5R (TP1@1.5R 33% · TP2@5R cum70% · Runner@10R) ──────

class TestExitsForex:
    def test_full_ladder_forex(self):
        e = _mk_engine()
        st = e.states["EURUSD"]
        st.open_trade = P11Trade(
            symbol="EURUSD", direction="short", entry_price=1.1000, sl=1.1010,
            sl_dist=0.0010, exits=st.group.exits,
            opened_at=pd.Timestamp("2026-06-01 11:00"),
        )
        # Short : runner 10R = 1.1000 − 0.010 = 1.0900
        e._manage_exits("EURUSD", st, _bar(high=1.1000, low=1.0899),
                        pd.Timestamp("2026-06-01 12:00"))
        trade = e.closed_trades[-1]
        # 33%@1.5R + 37%@5R + 30%@10R = 0.495 + 1.85 + 3.0 = 5.345R
        assert trade["pnl_r"] == pytest.approx(5.345, abs=1e-6)


# ── Flux d'entrée : guard et caps bloquent ────────────────────────────────────

class _FakeConnector:
    def __init__(self, m1: pd.DataFrame):
        self._m1 = m1

    def fetch_ohlcv(self, symbol, timeframe, limit):
        return self._m1


class _FakeStrategy:
    def __init__(self, signal):
        self._signal = signal

    def run(self, zone_df, m1_df):
        return [self._signal]


def _fake_signal(m1: pd.DataFrame, direction="long") -> SDSignal:
    ts = m1.index[-2]
    price = float(m1["close"].iloc[-2])
    sl = price - 5.0 if direction == "long" else price + 5.0
    return SDSignal(
        direction=direction, entry_price=price, stop_loss=sl,
        take_profit=price + 100 if direction == "long" else price - 100,
        risk_reward=20.0, zone_score=7.0, wyckoff_score=8.0, fib_bonus=0.0,
        composite_score=7.0, zone=None, wyckoff=None, fib=None,
        formed_at=ts, bar_index=len(m1) - 2,
    )


class TestEntryFlow:
    def _engine_with_signal(self, direction="long"):
        m1 = _mk_m1(list(np.linspace(3000, 3010, 1200)))
        e = _mk_engine(_FakeConnector(m1))
        st = e.states["XAUUSD"]
        st.strategy = _FakeStrategy(_fake_signal(m1, direction))
        return e, st, m1

    def test_signal_opens_trade(self):
        e, st, m1 = self._engine_with_signal()
        e._tick_symbol("XAUUSD", st, None)
        assert st.open_trade is not None
        assert st.open_trade.direction == "long"

    def test_guard_halt_blocks_entry(self):
        e, st, m1 = self._engine_with_signal()
        e.guard.state.halted_permanently = True
        e._tick_symbol("XAUUSD", st, None)
        assert st.open_trade is None

    def test_flat_hour_blocks_entry(self):
        m1 = _mk_m1(list(np.linspace(3000, 3010, 1200)), start="2026-06-01 02:00")
        # dernière barre ≈ 21h59 UTC — après flat_hour 21h
        e = _mk_engine(_FakeConnector(m1))
        st = e.states["XAUUSD"]
        st.strategy = _FakeStrategy(_fake_signal(m1))
        e._tick_symbol("XAUUSD", st, None)
        assert st.open_trade is None

    def test_symbol_daily_cap_blocks_entry(self):
        e, st, m1 = self._engine_with_signal()
        day = m1.index[-2].strftime("%Y-%m-%d")
        st.day_losses[f"{day}|long"] = 1        # cap V4 = 1 perte/jour/direction
        e._tick_symbol("XAUUSD", st, None)
        assert st.open_trade is None

    def test_duplicate_signal_ignored(self):
        e, st, m1 = self._engine_with_signal()
        e._tick_symbol("XAUUSD", st, None)
        first = st.open_trade
        st.open_trade = None                    # trade fermé entre-temps
        e._tick_symbol("XAUUSD", st, None)      # même signal re-proposé
        assert st.open_trade is None            # dédupliqué par formed_at

    def test_engine_boots_from_shipped_config(self):
        e = _mk_engine()
        assert len(e.states) == 11
        assert e.risk_usd == pytest.approx(600.0)   # 0.6% de 100k
