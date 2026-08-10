"""
Tests for zeus.strategy.open_us_strategy.OpenUSStrategy.

Réutilise les mêmes candles synthétiques que test_wyckoff.py (pattern
demand/supply déjà validé : accumulation 4 barres -> spring/upthrust barre 4
-> MSS barre 5) — ces tests ne re-testent PAS le scoring Wyckoff lui-même
(couvert par test_wyckoff.py), seulement le filtre horaire "Open US" et le
mapping des champs Tradeable ajoutés par OpenUSStrategy.

min_score=0.0 est utilisé partout ici pour découpler ces tests du scoring
Wyckoff — seul le comportement propre à OpenUSStrategy (fenêtre horaire,
un signal par jour, mapping des champs) est sous test.

Juin 2026 = heure d'été US (EDT, UTC-4) -> 9h30 ET = 13h30 UTC.
"""
from __future__ import annotations

import pandas as pd
import pytest

from zeus.strategy.open_us_strategy import DEFAULT_WINDOW_MINUTES, OpenUSStrategy
from zeus.strategy.supply_demand.pivot_candle import PivotSide
from zeus.strategy.supply_demand.wyckoff import WyckoffPattern

# ── Helpers ───────────────────────────────────────────────────────────────────

# Pattern demand : accum [1.100,1.110] -> spring barre 4 (low=1.095) -> MSS barre 5 (close=1.113)
_DEMAND_CANDLES = [
    (1.103, 1.108, 1.100, 1.106),
    (1.105, 1.109, 1.102, 1.107),
    (1.104, 1.110, 1.101, 1.105),
    (1.106, 1.109, 1.103, 1.104),
    (1.104, 1.108, 1.095, 1.106),
    (1.108, 1.115, 1.107, 1.113),
]

# Pattern supply : accum [1.100,1.110] -> upthrust barre 4 (high=1.115) -> MSS barre 5 (close=1.097)
_SUPPLY_CANDLES = [
    (1.104, 1.108, 1.101, 1.105),
    (1.105, 1.109, 1.102, 1.106),
    (1.105, 1.110, 1.101, 1.104),
    (1.106, 1.109, 1.100, 1.103),
    (1.106, 1.115, 1.103, 1.104),
    (1.103, 1.104, 1.095, 1.097),
]

# 20 barres neutres (pas de pattern) pour remplir le reste d'une session sans
# déclencher de faux signal.
_FLAT_CANDLES = [(1.100, 1.101, 1.099, 1.100)] * 20


def _df_from(start_utc: str, *blocks: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    candles = [c for block in blocks for c in block]
    idx = pd.date_range(start_utc, periods=len(candles), freq="1min")
    return pd.DataFrame(
        [{"open": o, "high": h, "low": l, "close": c} for o, h, l, c in candles],
        index=idx,
    )


# ── Fenêtre horaire ───────────────────────────────────────────────────────────

class TestWindowGating:
    def test_signal_at_window_open(self):
        # bar 5 (MSS) tombe à 13:30 UTC = 09:30 ET = début exact de la fenêtre
        df = _df_from("2026-06-01 13:25", _DEMAND_CANDLES)
        strat = OpenUSStrategy(min_score=0.0)
        signals = strat.generate_signals(df)
        assert len(signals) == 1
        assert signals[0].direction == "long"

    def test_no_signal_before_open(self):
        # bar 5 tombe à 08:05 ET — bien avant 09:30 ET
        df = _df_from("2026-06-01 12:00", _DEMAND_CANDLES)
        strat = OpenUSStrategy(min_score=0.0)
        signals = strat.generate_signals(df)
        assert signals == []

    def test_no_signal_after_window(self):
        # bar 5 tombe à 09:25 (avant début fenêtre) + décalage : test après la
        # fin de fenêtre (09:30 + DEFAULT_WINDOW_MINUTES) au lieu d'avant.
        start = pd.Timestamp("2026-06-01 13:30") + pd.Timedelta(minutes=DEFAULT_WINDOW_MINUTES + 10)
        df = _df_from(start.strftime("%Y-%m-%d %H:%M"), _DEMAND_CANDLES)
        strat = OpenUSStrategy(min_score=0.0)
        signals = strat.generate_signals(df)
        assert signals == []

    def test_no_signal_on_weekend(self):
        # 2026-06-06 = samedi ; même heure valide (13:25 UTC = 09:25 ET)
        df = _df_from("2026-06-06 13:25", _DEMAND_CANDLES)
        strat = OpenUSStrategy(min_score=0.0)
        signals = strat.generate_signals(df)
        assert signals == []


# ── Un signal par jour ────────────────────────────────────────────────────────

class _AlwaysDemandDetector:
    """Stub détecteur : renvoie systématiquement un pattern DEMAND valide,
    quel que soit le contenu des barres — isole le comportement propre à
    OpenUSStrategy (fenêtre + cooldown journalier) de la géométrie réelle
    du scan Wyckoff (déjà testée dans test_wyckoff.py)."""

    def detect_fast(self, h_arr, l_arr, c_arr, o_arr, df_index, side, end_idx):
        if side != PivotSide.DEMAND:
            return None
        i = end_idx - 1
        return WyckoffPattern(
            side=PivotSide.DEMAND, accum_high=1.110, accum_low=1.100, accum_bars=4,
            manip_bar=max(0, i - 1), manip_extreme=1.095,
            mss_bar=i, mss_close=1.113, score=10.0, formed_at=df_index[i],
        )


class TestOneSignalPerDay:
    def test_second_pattern_same_day_ignored(self):
        # Deux patterns demand valides le même jour, dans la fenêtre —
        # seul le premier doit produire un signal.
        df = _df_from("2026-06-01 13:25", _DEMAND_CANDLES, _FLAT_CANDLES, _DEMAND_CANDLES)
        strat = OpenUSStrategy(min_score=0.0)
        signals = strat.generate_signals(df)
        assert len(signals) == 1

    def test_pattern_next_day_produces_new_signal(self):
        # Détecteur stubbé : isole le cooldown journalier de la logique de
        # scan Wyckoff elle-même (voir _AlwaysDemandDetector ci-dessus).
        idx = pd.date_range("2026-06-01 13:30", periods=5, freq="1min").append(
            pd.date_range("2026-06-02 13:30", periods=5, freq="1min")
        )
        df = pd.DataFrame({"open": 1.1, "high": 1.1, "low": 1.1, "close": 1.1}, index=idx)
        strat = OpenUSStrategy(detector=_AlwaysDemandDetector(), min_score=0.0)
        signals = strat.generate_signals(df)
        assert len(signals) == 2
        assert signals[0].formed_at.date() != signals[1].formed_at.date()


# ── Mapping des champs Tradeable ──────────────────────────────────────────────

class TestSignalFields:
    def test_demand_signal_fields(self):
        df = _df_from("2026-06-01 13:25", _DEMAND_CANDLES)
        strat = OpenUSStrategy(min_score=0.0, risk_reward=2.0)
        sig = strat.generate_signals(df)[0]

        assert sig.direction == "long"
        assert sig.bar_index == 5
        assert sig.stop_loss == pytest.approx(1.095, abs=1e-5)     # spring low
        assert sig.entry_price == pytest.approx(1.113, abs=1e-5)   # mss close
        assert sig.risk_reward == 2.0
        sl_dist = sig.entry_price - sig.stop_loss
        assert sig.take_profit == pytest.approx(sig.entry_price + 2.0 * sl_dist, abs=1e-5)
        assert sig.zone_score == pytest.approx(sig.wyckoff.score, abs=1e-9)
        assert sig.formed_at == df.index[5]

    def test_supply_signal_fields(self):
        df = _df_from("2026-06-01 13:25", _SUPPLY_CANDLES)
        strat = OpenUSStrategy(min_score=0.0, risk_reward=2.0)
        sig = strat.generate_signals(df)[0]

        assert sig.direction == "short"
        assert sig.stop_loss == pytest.approx(1.115, abs=1e-5)     # upthrust high
        assert sig.entry_price == pytest.approx(1.097, abs=1e-5)   # mss close
        sl_dist = sig.stop_loss - sig.entry_price
        assert sig.take_profit == pytest.approx(sig.entry_price - 2.0 * sl_dist, abs=1e-5)


# ── Seuil de score ─────────────────────────────────────────────────────────────

class TestMinScore:
    def test_high_min_score_filters_weak_pattern(self):
        df = _df_from("2026-06-01 13:25", _DEMAND_CANDLES)
        strat = OpenUSStrategy(min_score=10.1)   # au-dessus du max possible (10.0)
        assert strat.generate_signals(df) == []


# ── Validation d'entrée ────────────────────────────────────────────────────────

class TestInputValidation:
    def test_rejects_tz_aware_index(self):
        df = _df_from("2026-06-01 13:25", _DEMAND_CANDLES)
        df.index = df.index.tz_localize("UTC")
        strat = OpenUSStrategy(min_score=0.0)
        with pytest.raises(ValueError):
            strat.generate_signals(df)
