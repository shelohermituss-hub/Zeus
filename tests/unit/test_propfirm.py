"""Tests du garde-fou propfirm (zeus/risk/propfirm.py) — limites strictes fail-closed."""
from __future__ import annotations

import pandas as pd
import pytest

from zeus.risk.propfirm import PropFirmConfig, PropFirmGuard

_BAL = 100_000.0


def _ts(day: int, hour: int = 10) -> pd.Timestamp:
    return pd.Timestamp(f"2026-01-{day:02d} {hour:02d}:00:00")


def _guard(**cfg_overrides) -> PropFirmGuard:
    return PropFirmGuard(PropFirmConfig(**cfg_overrides), initial_balance=_BAL)


# ── Config validation ─────────────────────────────────────────────────────────

class TestConfigValidation:
    def test_defaults_are_strict(self):
        cfg = PropFirmConfig()
        assert cfg.max_daily_loss_pct == 0.03   # < 5% firm
        assert cfg.max_total_dd_pct == 0.06     # < 10% firm
        assert cfg.risk_per_trade_pct == 0.003

    @pytest.mark.parametrize("kwargs", [
        dict(max_daily_loss_pct=0.0),
        dict(max_daily_loss_pct=1.5),
        dict(max_total_dd_pct=0.0),
        dict(risk_per_trade_pct=0.0),
        dict(risk_per_trade_pct=0.05),   # > max_daily_loss_pct
        dict(max_losses_per_day=0),
        dict(flat_hour_utc=24),
    ])
    def test_invalid_config_rejected(self, kwargs):
        with pytest.raises(ValueError):
            PropFirmConfig(**kwargs)

    def test_invalid_balance_rejected(self):
        with pytest.raises(ValueError):
            PropFirmGuard(PropFirmConfig(), initial_balance=0.0)


# ── Autorisations de base ─────────────────────────────────────────────────────

class TestCanTrade:
    def test_fresh_account_can_trade(self):
        assert _guard().can_trade(_ts(5)) is True

    def test_no_entry_at_or_after_flat_hour(self):
        g = _guard(flat_hour_utc=21)
        assert g.can_trade(_ts(5, hour=20)) is True
        assert g.can_trade(_ts(5, hour=21)) is False
        assert g.can_trade(_ts(5, hour=23)) is False

    def test_risk_amount_is_fixed_fraction_of_initial_balance(self):
        g = _guard()
        assert g.risk_amount_usd() == pytest.approx(_BAL * 0.003)
        # Pas de compounding : le montant ne bouge pas avec l'équity
        g.on_trade_closed(_ts(5), +5_000.0)
        assert g.risk_amount_usd() == pytest.approx(_BAL * 0.003)


# ── Limite journalière ────────────────────────────────────────────────────────

class TestDailyLoss:
    def test_daily_loss_halts_for_the_day(self):
        g = _guard()   # daily 3% de 100k = 3 000 $
        g.on_trade_closed(_ts(5), -3_000.0)
        assert g.can_trade(_ts(5, hour=12)) is False

    def test_daily_halt_resets_next_day(self):
        g = _guard()
        g.on_trade_closed(_ts(5), -3_000.0)
        assert g.can_trade(_ts(5, hour=12)) is False
        assert g.can_trade(_ts(6)) is True

    def test_loss_below_daily_limit_keeps_trading(self):
        g = _guard()
        g.on_trade_closed(_ts(5), -1_000.0)
        assert g.can_trade(_ts(5, hour=12)) is True

    def test_daily_loss_measured_from_day_start_equity(self):
        g = _guard()
        # Jour 5 : +4 000 → équity 104 000
        g.on_trade_closed(_ts(5), +4_000.0)
        # Jour 6 : perdre 3 000 depuis le début de jour = limite atteinte,
        # même si l'équity (101 000) reste au-dessus du solde initial.
        g.on_trade_closed(_ts(6), -3_000.0)
        assert g.can_trade(_ts(6, hour=12)) is False


# ── Série de pertes ───────────────────────────────────────────────────────────

class TestLossStreak:
    def test_n_losses_halt_the_day(self):
        g = _guard(max_losses_per_day=3)
        for _ in range(3):
            g.on_trade_closed(_ts(5), -300.0)
        assert g.can_trade(_ts(5, hour=12)) is False

    def test_streak_resets_next_day(self):
        g = _guard(max_losses_per_day=3)
        for _ in range(3):
            g.on_trade_closed(_ts(5), -300.0)
        assert g.can_trade(_ts(6)) is True

    def test_wins_do_not_count_in_streak(self):
        g = _guard(max_losses_per_day=3)
        g.on_trade_closed(_ts(5), -300.0)
        g.on_trade_closed(_ts(5), +900.0)
        g.on_trade_closed(_ts(5), -300.0)
        assert g.can_trade(_ts(5, hour=12)) is True


# ── Drawdown total ────────────────────────────────────────────────────────────

class TestTotalDrawdown:
    def test_total_dd_halts_permanently(self):
        g = _guard()   # total 6% de 100k = 6 000 $
        g.on_trade_closed(_ts(5), -3_000.0)
        g.on_trade_closed(_ts(6), -3_000.0)
        assert g.can_trade(_ts(7)) is False
        assert g.can_trade(_ts(20)) is False
        assert g.challenge_failed() is True

    def test_total_dd_measured_from_initial_balance(self):
        g = _guard()
        # +10 000 de gains puis −6 000 : équity 104 000, DD depuis solde initial
        # négatif → pas d'arrêt (convention statique, pas de trailing).
        g.on_trade_closed(_ts(5), +10_000.0)
        g.on_trade_closed(_ts(6), -3_000.0)
        g.on_trade_closed(_ts(7), -3_000.0)
        assert g.challenge_failed() is False


# ── Statut challenge ──────────────────────────────────────────────────────────

class TestChallengeStatus:
    def test_pass_requires_target_and_min_days(self):
        g = _guard(profit_target_pct=0.10, min_trading_days=4)
        # 3 jours de gains → cible atteinte mais pas les jours minimum
        for d in range(5, 8):
            g.on_trade_closed(_ts(d), +4_000.0)
        assert g.challenge_passed() is False
        g.on_trade_closed(_ts(8), +100.0)   # 4e jour
        assert g.challenge_passed() is True

    def test_below_target_not_passed(self):
        g = _guard(profit_target_pct=0.10, min_trading_days=1)
        g.on_trade_closed(_ts(5), +5_000.0)
        assert g.challenge_passed() is False

    def test_failed_account_never_passes(self):
        g = _guard()
        g.on_trade_closed(_ts(5), -6_000.0)
        assert g.challenge_failed() is True
        assert g.challenge_passed() is False
