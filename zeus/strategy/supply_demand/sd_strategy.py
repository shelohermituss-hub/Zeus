"""
Supply & Demand strategy — main signal generator.

Combines:
    - M15 S&D zone detection  (ZoneDetector)
    - M1 Wyckoff confirmation (WyckoffDetector)
    - Optional Fibonacci OTE  (zone_fib_confluence)

Signal flow (per M1 bar, index i):
    1. Update active zone state (mitigation, freshness, touch count)
    2. For each zone that contains the current price:
       a. Run WyckoffDetector on recent M1 bars
       b. If Wyckoff MSS fires at bar i → score → emit SDSignal

Entry  : Wyckoff mss_close
Stop   : zone.wick_extreme (tight; just beyond manipulation wick)
Target : entry ± risk_reward × |entry − stop|

Break-even (BE) at 1R is an execution concern handled outside this module.
The strategy outputs SL/TP levels only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .fibonacci import FibConfluence, FibLevels, zone_fib_confluence
from .pivot_candle import PivotSide
from .wyckoff import WyckoffDetector, WyckoffPattern
from .zone_detector import SDZone, ZoneDetector


# ── Signal data type ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SDSignal:
    """
    Trading signal produced by the Supply & Demand strategy.

    Attributes
    ----------
    direction        : "long" (demand zone) or "short" (supply zone)
    entry_price      : Wyckoff MSS close — in live, enter at next M1 bar open
    stop_loss        : zone.wick_extreme — tight stop beyond manipulation wick
    take_profit      : entry ± risk_reward × |entry − stop|
    risk_reward      : R:R ratio configured on the strategy
    zone_score       : S&D zone quality 0–10 (at signal time, includes fresh decay)
    wyckoff_score    : Wyckoff pattern quality 0–10
    fib_bonus        : Fibonacci OTE confluence bonus 0–2 (0 if not enabled)
    composite_score  : zone_score + fib_bonus (0–12)
    zone             : the triggering S&D zone
    wyckoff          : the Wyckoff pattern that confirmed entry
    fib              : FibConfluence result (None if Fibonacci disabled)
    formed_at        : M1 timestamp of the MSS bar (signal generation time)
    bar_index        : integer index of the signal bar in m1_df
    """
    direction:       str
    entry_price:     float
    stop_loss:       float
    take_profit:     float
    risk_reward:     float
    zone_score:      float
    wyckoff_score:   float
    fib_bonus:       float
    composite_score: float
    zone:            SDZone
    wyckoff:         WyckoffPattern
    fib:             Optional[FibConfluence]
    formed_at:       pd.Timestamp
    bar_index:       int

    @property
    def risk(self) -> float:
        """Absolute distance from entry to stop-loss (= 1R)."""
        return abs(self.entry_price - self.stop_loss)

    @property
    def reward(self) -> float:
        """Absolute distance from entry to take-profit."""
        return abs(self.take_profit - self.entry_price)


# ── Strategy ──────────────────────────────────────────────────────────────────

class SDStrategy:
    """
    Supply & Demand strategy: M15 zones + M1 Wyckoff entry confirmation.

    Parameters
    ----------
    zone_detector      : ZoneDetector instance (default: ZoneDetector())
    wyckoff_detector   : WyckoffDetector instance (default: WyckoffDetector())
    risk_reward        : R:R target (default 3.0)
    min_zone_score     : minimum S&D zone total score (default 5.0)
    min_wyckoff_score  : minimum Wyckoff pattern score (default 4.0)
    min_composite_score: minimum zone_score + fib_bonus (default 5.0)
    signal_cooldown    : M1 bars before the same zone can re-signal (default 30)
    """

    def __init__(
        self,
        zone_detector:       Optional[ZoneDetector]    = None,
        wyckoff_detector:    Optional[WyckoffDetector] = None,
        risk_reward:         float = 3.0,
        min_zone_score:      float = 5.0,
        min_wyckoff_score:   float = 4.0,
        min_composite_score: float = 5.0,
        signal_cooldown:     int   = 30,
    ) -> None:
        self._zones    = zone_detector    or ZoneDetector()
        self._wyckoff  = wyckoff_detector or WyckoffDetector()
        self.risk_reward          = risk_reward
        self.min_zone_score       = min_zone_score
        self.min_wyckoff_score    = min_wyckoff_score
        self.min_composite_score  = min_composite_score
        self.signal_cooldown      = signal_cooldown

    # ── Public ────────────────────────────────────────────────────────────────

    def run(
        self,
        m15_df:   pd.DataFrame,
        m1_df:    pd.DataFrame,
        htf_fibs: Optional[FibLevels] = None,
    ) -> list[SDSignal]:
        """
        Full historical backtest: detect M15 zones → iterate M1 bars for signals.

        Parameters
        ----------
        m15_df   : M15 OHLCV DataFrame — used for zone detection
        m1_df    : M1  OHLCV DataFrame — used for Wyckoff confirmation
        htf_fibs : optional pre-computed FibLevels for OTE confluence scoring;
                   None → Fibonacci disabled (fib_bonus stays 0)

        Returns
        -------
        list[SDSignal] in chronological order.

        Notes
        -----
        Zone detection (detect_zones) is a historical scan that uses bars on
        both sides of each pivot.  The temporal causality constraint is enforced
        by only making a zone visible once its formed_at timestamp has passed on
        the M1 timeline (``zone.formed_at <= m1_bar_timestamp``).
        """
        all_zones = self._zones.detect_zones(m15_df)
        if not all_zones:
            return []

        signals: list[SDSignal] = []
        # (formed_at, side, zone_bottom) → last M1 bar index that generated a signal
        _last_signal: dict[tuple, int] = {}

        for i in range(len(m1_df)):
            ts  = m1_df.index[i]
            bar = m1_df.iloc[i]

            # Zones that have already formed by this M1 bar
            formed = [z for z in all_zones if z.formed_at <= ts]

            # Update mitigation / freshness state for all formed zones
            self._zones.update_zones(formed, bar, ts)

            for zone in formed:
                if zone.is_mitigated:
                    continue
                if zone.score.total < self.min_zone_score:
                    continue
                if not zone.price_in_zone(float(bar["low"]), float(bar["high"])):
                    continue

                # Per-zone cooldown: avoid rapid re-entries on the same zone
                z_key = (zone.formed_at, zone.side, zone.zone_bottom)
                if i - _last_signal.get(z_key, -(self.signal_cooldown + 1)) < self.signal_cooldown:
                    continue

                # Wyckoff confirmation on M1 bars up to and including bar i
                wyckoff = self._wyckoff.detect(m1_df, zone.side, end_idx=i + 1)
                if wyckoff is None:
                    continue
                if wyckoff.score < self.min_wyckoff_score:
                    continue
                if wyckoff.mss_bar != i:
                    continue  # MSS did not fire on the current bar — not actionable yet

                sig = self._build_signal(i, ts, zone, wyckoff, htf_fibs)
                if sig is None:
                    continue
                if sig.composite_score < self.min_composite_score:
                    continue

                _last_signal[z_key] = i
                signals.append(sig)

        return signals

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_signal(
        self,
        bar_index: int,
        ts:        pd.Timestamp,
        zone:      SDZone,
        wyckoff:   WyckoffPattern,
        htf_fibs:  Optional[FibLevels],
    ) -> Optional[SDSignal]:
        entry = wyckoff.mss_close
        sl    = zone.wick_extreme
        risk  = abs(entry - sl)

        if risk < 1e-8:
            return None

        if zone.side == PivotSide.DEMAND:
            direction = "long"
            tp = entry + self.risk_reward * risk
        else:
            direction = "short"
            tp = entry - self.risk_reward * risk

        # Optional Fibonacci OTE confluence
        fib_bonus: float = 0.0
        fib_conf: Optional[FibConfluence] = None
        if htf_fibs is not None:
            fib_conf  = zone_fib_confluence(zone, htf_fibs)
            fib_bonus = fib_conf.score_bonus

        composite = round(zone.score.total + fib_bonus, 2)

        return SDSignal(
            direction        = direction,
            entry_price      = round(entry, 6),
            stop_loss        = round(sl, 6),
            take_profit      = round(tp, 6),
            risk_reward      = self.risk_reward,
            zone_score       = round(zone.score.total, 2),
            wyckoff_score    = round(wyckoff.score, 2),
            fib_bonus        = round(fib_bonus, 2),
            composite_score  = composite,
            zone             = zone,
            wyckoff          = wyckoff,
            fib              = fib_conf,
            formed_at        = ts,
            bar_index        = bar_index,
        )
