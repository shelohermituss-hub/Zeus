"""
ScalpSMCStrategy — intraday SMC scalping on XAUUSD.

Cascade
-------
1H  (HTF) : SMC analysis — Order Blocks, FVG, OTE.
            1H zones form 4× more often than 4H zones → more setups.
15M (MSS) : Market Structure Shift confirmation (optional).
1M  (LTF) : Entry trigger — 1M internal bias + optional 1M FVG.

Key differences vs MTFSMCStrategy (swing)
------------------------------------------
- HTF = 1H instead of 4H        → ~4× more zone events
- MSS gate on 15M instead of 1H  → faster structure confirmation
- Entry bar = 1M instead of 5M   → tighter SL
- No weekly bias gate             → irrelevant at intraday scale
- Up to 5 signals/day, 2/session → higher frequency
- PartialCloseConfig.scalp()      → 1.5R/2.5R/4R fast ladder

Quality gates (V3 defaults — all ON by default)
------------------------------------------------
- ICT kill zone (07–11 UTC London, 12–15 UTC NY)
- Daily bias alignment (require_daily_bias=True → neutral day = no trade)
- Fibonacci OTE 0.618–0.786 required (require_ote=True)
- Clean approach — price drifts into zone, not impulsively
- LTF liquidity sweep at entry (1M sweep of prev bar extreme)
- HTF confluence score ≥ 5 (3 real factors beyond auto-fire gates)
- Zone re-entry guard (one entry per 1H zone per day)
- Session frequency cap

Stop-loss
---------
SL placed below/above the 1M entry bar's extreme.
Default 20 pips; hard cap 30 pips.  1 pip = $1 for XAUUSD.
"""
from __future__ import annotations

import pandas as pd

from zeus.strategy.base import Signal, SignalType
from zeus.strategy.mtf_strategy import MTFSMCStrategy


class ScalpSMCStrategy(MTFSMCStrategy):
    """
    Scalp variant of MTFSMCStrategy — cascade 1H → 15M → 1M.

    Args:
        df_htf_1h:   1H OHLCV DataFrame (HTF zones).
        df_mtf_15m:  15M OHLCV DataFrame (MSS gate). Pass None to disable.
        df_daily:    Daily OHLCV DataFrame (daily bias gate). Required when
                     require_daily_bias=True (the default).
        sl_pips:     Default SL in pips (1 pip = pip_value). Default 20.
        max_sl_pips: Hard cap — reject setup if SL wider. Default 30.
        pip_value:   Price units per pip (XAUUSD: 1.0).
        min_htf_score: Minimum 1H confluence score. Default 5.0.
        swing_length:  1H pivot lookback (bars). Default 20 ≈ 20 h.
        internal_length: Internal structure lookback. Default 5.
        atr_period:  ATR period for volatility normalisation. Default 100.
        ltf_lookback: 1M bars to scan for LTF entry trigger. Default 15.
        killzone_only: Restrict entries to ICT kill zones. Default True.
        require_entry_fvg: Require price in a 1M FVG at entry. Default True.
        require_choch_candle: Entry bar must close in trade direction. Default False.
        require_ote: Price must be in Fibonacci OTE zone 0.618–0.786. Default True.
        require_daily_bias: Daily bias must be clear and aligned. Default True.
        max_daily_signals: Max trades per calendar day. Default 5.
        max_signals_per_session: Max trades per kill-zone session. Default 2.
    """

    def __init__(
        self,
        df_htf_1h:               pd.DataFrame,
        df_mtf_15m:              pd.DataFrame | None = None,
        df_daily:                pd.DataFrame | None = None,
        sl_pips:                 float = 20.0,
        max_sl_pips:             float = 30.0,
        pip_value:               float = 1.0,
        min_htf_score:           float = 5.0,
        swing_length:            int   = 20,
        internal_length:         int   = 5,
        atr_period:              int   = 100,
        ltf_lookback:            int   = 15,
        killzone_only:           bool  = True,
        require_entry_fvg:       bool  = True,
        require_choch_candle:    bool  = False,
        require_ote:                  bool  = True,
        require_daily_bias:           bool  = True,
        require_clean_approach:       bool  = True,
        approach_lookback:            int   = 5,
        approach_max_momentum:        float = 0.6,
        approach_max_body_atr:        float = 1.5,
        require_poc_zone_confluence:  bool  = False,
        poc_zone_tolerance_pct:       float = 0.005,
        require_session_sweep:        bool  = False,
        session_sweep_lookback:       int   = 30,
        require_ltf_sweep:            bool  = True,
        ltf_sweep_lookback:           int   = 3,
        require_entry_pattern:        bool  = False,
        min_wick_ratio:               float = 0.60,
        max_daily_signals:            int   = 5,
        max_signals_per_session: int   = 2,
        allowed_zones:           list[str] | None = None,
    ) -> None:
        super().__init__(
            df_htf=df_htf_1h,
            df_daily=df_daily,
            df_mtf=df_mtf_15m,
            sl_pips=sl_pips,
            max_sl_pips=max_sl_pips,
            pip_value=pip_value,
            min_htf_score=min_htf_score,
            swing_length=swing_length,
            internal_length=internal_length,
            atr_period=atr_period,
            ltf_lookback=ltf_lookback,
            zone_tolerance_pct=0.003,
            killzone_only=killzone_only,
            sweep_zone_tol_pct=0.005,
            require_entry_fvg=require_entry_fvg,
            require_choch_candle=require_choch_candle,
            require_weekly_bias=False,
            require_asian_sweep=False,
            require_pd_filter=False,
            # Scalp entries happen during 1H pullbacks: internal bias is allowed
            # to be opposite to swing bias (that IS the setup — buy the dip).
            require_htf_internal_align=False,
            require_ote=require_ote,
            require_daily_bias=require_daily_bias,
            require_clean_approach=require_clean_approach,
            approach_lookback=approach_lookback,
            approach_max_momentum=approach_max_momentum,
            approach_max_body_atr=approach_max_body_atr,
            require_poc_zone_confluence=require_poc_zone_confluence,
            poc_zone_tolerance_pct=poc_zone_tolerance_pct,
            require_session_sweep=require_session_sweep,
            session_sweep_lookback=session_sweep_lookback,
            require_ltf_sweep=require_ltf_sweep,
            ltf_sweep_lookback=ltf_sweep_lookback,
            require_entry_pattern=require_entry_pattern,
            min_wick_ratio=min_wick_ratio,
            max_daily_signals=max_daily_signals,
            max_signals_per_session=max_signals_per_session,
            allowed_zones=allowed_zones,
        )

    def generate_signal(self, df: pd.DataFrame, bar_index: int) -> Signal:
        sig = super().generate_signal(df, bar_index)
        if sig.type != SignalType.NONE:
            reason = sig.reason.replace("MTF grade=", "SCALP grade=")
            return Signal(sig.type, sig.confidence, reason, sig.bar_index)
        return sig
