"""
ConfluenceSoftScoredStrategy — N-of-M soft-scoring variant of the "9-10
confirmations" methodology (see confluence_scalp_strategy.py for the
hard-AND-gate version and the full confirmation -> gate mapping).

ConfluenceScalpStrategy requires every documented confirmation to hold
simultaneously (a strict boolean AND across ~10 independent checks).
Diagnostics on real Feb 2026 XAUUSD tick data showed that requirement
produces zero setups even after fixing two real scoring bugs (see
confluence.py's require_entry_gate / zone_tolerance_pct) — the surviving
bottleneck is a genuine, rare simultaneous alignment of every confirmation,
not a bug. This strategy instead counts how many of the 10 confirmations
are true and thresholds on the total, matching the user's own A+ (6-7/10)
/ A++ (8-9/10) grading, so a setup missing one weak confirmation isn't
discarded outright.

No new detection logic is added — every confirmation reuses the exact
same SMC sub-module checks as ConfluenceScalpStrategy/MTFSMCStrategy.
"""
from __future__ import annotations

import pandas as pd

from zeus.strategy.base import Signal, SignalType
from zeus.strategy.confluence import best_confluence
from zeus.strategy.mtf_strategy import MTFSMCStrategy
from zeus.strategy.smc.candle_pattern import detect_entry_candle
from zeus.strategy.smc.fvg import get_active_fvgs
from zeus.strategy.smc.fibonacci import get_latest_fib_zone
from zeus.strategy.smc.ltf_sweep import ltf_liquidity_sweep
from zeus.strategy.smc.pivot import BULLISH
from zeus.strategy.smc.session import (
    detect_session_ranges,
    get_daily_bias,
    get_weekly_bias,
    killzone_name,
    prev_session_liquidity_swept,
)

# Confirmation names in documented order (1-10), for reason strings.
_CONFIRMATION_NAMES = (
    "structure", "ote", "order_block", "external_liquidity", "fvg",
    "poc", "killzone", "entry_model", "confluence_score", "fib_50",
)


class ConfluenceSoftScoredStrategy(MTFSMCStrategy):
    """
    Same 10 documented confirmations as ConfluenceScalpStrategy, scored
    N-of-M instead of hard-AND-gated.

    Args:
        df_htf_1h:          1H OHLCV (HTF zones: OB / FVG / OTE / POC).
        df_daily:           Daily OHLCV (daily + weekly bias).
        min_confirmations:  Minimum count (out of 10) required to emit a
                            signal. 6-7 ~ "A+", 8-9 ~ "A++" per the user's
                            own grading.
    """

    _N_CONFIRMATIONS = len(_CONFIRMATION_NAMES)

    def __init__(
        self,
        df_htf_1h:               pd.DataFrame,
        df_daily:                pd.DataFrame | None = None,
        min_confirmations:       int   = 8,
        sl_pips:                 float = 20.0,
        max_sl_pips:             float = 30.0,
        pip_value:               float = 1.0,
        swing_length:            int   = 20,
        internal_length:         int   = 5,
        atr_period:              int   = 100,
        ltf_lookback:            int   = 15,
        min_wick_ratio:          float = 0.60,
        max_daily_signals:       int   = 3,
        max_signals_per_session: int   = 1,
        session_sweep_lookback:  int   = 30,
        ltf_sweep_lookback:      int   = 3,
        poc_tolerance_pct:       float = 0.003,
    ) -> None:
        super().__init__(
            df_htf=df_htf_1h,
            df_mtf=None,  # 1H/15M MSS gate disabled (run_scalp_backtest.py precedent)
            df_daily=df_daily,
            sl_pips=sl_pips,
            max_sl_pips=max_sl_pips,
            pip_value=pip_value,
            swing_length=swing_length,
            internal_length=internal_length,
            atr_period=atr_period,
            ltf_lookback=ltf_lookback,
            zone_tolerance_pct=0.003,
            sweep_zone_tol_pct=0.005,
            killzone_only=False,               # killzone is confirmation #7, not a hard gate
            require_htf_internal_align=False,
            max_daily_signals=max_daily_signals,
            max_signals_per_session=max_signals_per_session,
        )
        self._min_confirmations      = min_confirmations
        self._soft_session_lookback  = session_sweep_lookback
        self._soft_ltf_sweep_lookback = ltf_sweep_lookback
        self._min_wick_ratio_soft    = min_wick_ratio
        self._poc_tol_soft           = poc_tolerance_pct
        self._session_ranges_soft: list | None = None

    def generate_signal(self, df: pd.DataFrame, bar_index: int) -> Signal:
        ltf_ts   = df.index[bar_index]
        close    = float(df["close"].iloc[bar_index])
        bar_low  = float(df["low"].iloc[bar_index])
        bar_high = float(df["high"].iloc[bar_index])

        htf_bar_idx = self._last_htf_bar(ltf_ts)
        if htf_bar_idx < self._min_htf_bars:
            return Signal(SignalType.NONE, 0.0, "insufficient HTF data", bar_index)

        htf_result = self._htf_analysis(htf_bar_idx)
        direction  = htf_result.swing_bias
        if direction == 0:
            return Signal(SignalType.NONE, 0.0, "no HTF swing bias", bar_index)

        zone_info = self._in_htf_zone(htf_result, htf_bar_idx, close, direction)
        if zone_info is None:
            return Signal(SignalType.NONE, 0.0, "price outside HTF zone", bar_index)
        zone_type, _zone_low, _zone_high = zone_info

        confirmations: list[bool] = []

        # #1 Market structure — HTF swing bias is defined (checked above).
        confirmations.append(True)

        # #2 Fibonacci OTE (61.8-78.6% retracement)
        fib_zone = get_latest_fib_zone(htf_result.fib_zones, htf_bar_idx, direction)
        confirmations.append(fib_zone is not None and fib_zone.is_in_ote(close))

        # #3 Supply/Demand zone (Order Block)
        confirmations.append(zone_type == "OB")

        # #4 External liquidity: weekly bias + prior-session sweep + daily bias
        if self._session_ranges_soft is None:
            self._session_ranges_soft = detect_session_ranges(df)
        session_swept, _reason = prev_session_liquidity_swept(
            self._session_ranges_soft, df, bar_index, direction,
            sweep_lookback=self._soft_session_lookback,
        )
        wb = get_weekly_bias(self._df_daily, ltf_ts) if self._df_daily is not None else 0
        db = get_daily_bias(self._df_daily, ltf_ts) if self._df_daily is not None else 0
        confirmations.append(
            session_swept and (wb == 0 or wb == direction) and (db == 0 or db == direction)
        )

        # #5 FVG / Imbalance
        confirmations.append(any(
            fvg.direction == direction and fvg.bottom <= close <= fvg.top
            for fvg in get_active_fvgs(htf_result.fvgs, htf_bar_idx)
        ))

        # #6 POC (volume Point of Control)
        vp = htf_result.volume_profile
        confirmations.append(vp is not None and vp.is_near_poc(close, self._poc_tol_soft))

        # #7 Kill zone session (London / NY)
        confirmations.append(killzone_name(ltf_ts) is not None)

        # #8 Entry model: LTF internal bias trigger + sweep + entry candle + CHoCH direction
        ltf_entry_ok = self._ltf_entry_confirmed(df, bar_index, direction, close)
        sweep_ok, _reason = ltf_liquidity_sweep(
            df, bar_index, direction, lookback=self._soft_ltf_sweep_lookback,
        )
        pattern_ok, _reason = detect_entry_candle(
            df, bar_index, direction, self._min_wick_ratio_soft,
        )
        bar_open = float(df["open"].iloc[bar_index])
        choch_ok = (close > bar_open) if direction == BULLISH else (close < bar_open)
        confirmations.append(ltf_entry_ok and sweep_ok and pattern_ok and choch_ok)

        # #9 Confluence structure+zone (the underlying 10-factor confluence score)
        cs = best_confluence(
            htf_result, close, htf_bar_idx, min_score=4.0, timestamp=ltf_ts,
            zone_tolerance_pct=self._zone_tol, sweep_zone_tol_pct=self._sweep_zone_tol_pct,
            require_entry_gate=False,
        )
        confirmations.append(cs is not None)

        # #10 Fibonacci 50% (premium/discount)
        pd_ok = False
        if fib_zone is not None:
            level_50 = fib_zone.level_50
            pd_ok = (close <= level_50) if direction == BULLISH else (close >= level_50)
        confirmations.append(pd_ok)

        n_true = sum(confirmations)
        if n_true < self._min_confirmations:
            active = [name for name, ok in zip(_CONFIRMATION_NAMES, confirmations) if ok]
            return Signal(
                SignalType.NONE, 0.0,
                f"{n_true}/{self._N_CONFIRMATIONS} confirmations "
                f"(need {self._min_confirmations}): {active}",
                bar_index,
            )

        sl_pips, _sl_price = self._compute_sl(close, bar_low, bar_high, direction)
        if sl_pips > self._max_sl_pips:
            return Signal(
                SignalType.NONE, 0.0,
                f"SL too wide ({sl_pips:.1f} pips > max {self._max_sl_pips})",
                bar_index,
            )

        today   = ltf_ts.date()
        session = killzone_name(ltf_ts)

        if self._max_daily_signals > 0:
            signals_today = sum(1 for d, *_ in self._signal_log if d == today)
            if signals_today >= self._max_daily_signals:
                return Signal(
                    SignalType.NONE, 0.0,
                    f"daily signal limit reached ({self._max_daily_signals}/day)",
                    bar_index,
                )
        if self._max_signals_per_session > 0 and session is not None:
            signals_session = sum(
                1 for d, s, *_ in self._signal_log if d == today and s == session
            )
            if signals_session >= self._max_signals_per_session:
                return Signal(
                    SignalType.NONE, 0.0,
                    f"{session} session limit reached ({self._max_signals_per_session}/session)",
                    bar_index,
                )
        if any(d == today and h == htf_bar_idx and z == zone_type
               for d, _s, h, z in self._signal_log):
            return Signal(
                SignalType.NONE, 0.0,
                f"zone {zone_type} at HTF bar {htf_bar_idx} already used today",
                bar_index,
            )

        self._signal_log.append((today, session, htf_bar_idx, zone_type))
        stype  = SignalType.LONG if direction == BULLISH else SignalType.SHORT
        reason = (
            f"SOFT {n_true}/{self._N_CONFIRMATIONS} zone={zone_type} SL={sl_pips:.1f}pips"
        )
        return Signal(stype, n_true / self._N_CONFIRMATIONS, reason, bar_index)
