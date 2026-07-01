"""
Multi-timeframe SMC strategy — HTF zone identification + LTF entry confirmation.

Workflow
--------
Higher timeframe (HTF — typically 4H or 1H):
  - SMC analysis identifies active zones: Order Blocks, Fair Value Gaps, OTE zones.
  - The HTF swing bias determines trade direction (factor 1 gate).

Lower timeframe (LTF — typically 1M):
  - When price enters an HTF zone, the LTF is scanned for an entry trigger.
  - Confirmation: the LTF internal bias aligns with the HTF direction (BOS/CHoCH).
  - Entry at the close of the confirming LTF bar.

Stop-loss placement
-------------------
SL is placed *sl_pips* below the entry bar's low (long) or above its high (short).
If the required SL exceeds *max_sl_pips* the setup is rejected (too wide to trade).
SL is expressed in absolute price units where 1 pip = *pip_value* (default $1 for XAUUSD).

Signal metadata
---------------
signal.reason includes:
  - "MTF grade=X score=Y/10" — HTF confluence grade and score
  - The confirmed HTF zone type (OB / FVG / OTE)
  - The computed SL distance in pips
"""
from __future__ import annotations

import pandas as pd

from zeus.strategy.base import Signal, SignalType, Strategy
from zeus.strategy.confluence import PatternGrade, best_confluence
from zeus.strategy.smc.indicator import SMCResult, analyze
from zeus.strategy.smc.session import get_daily_bias, is_in_killzone, killzone_name
from zeus.strategy.smc.order_block import get_active_order_blocks
from zeus.strategy.smc.fvg import get_active_fvgs
from zeus.strategy.smc.fibonacci import get_latest_fib_zone
from zeus.strategy.smc.pivot import BULLISH, BEARISH


class MTFSMCStrategy(Strategy):
    """
    Multi-timeframe SMC strategy.

    Args:
        df_htf:          Pre-computed higher-timeframe OHLCV DataFrame (e.g. 4H).
                         Must share the same timezone as the LTF data.
        min_htf_score:   Minimum HTF confluence score for zone validity (default 4).
        sl_pips:         Default SL distance in pips (1 pip = pip_value in price).
        max_sl_pips:     Hard cap — reject setup if natural SL exceeds this.
        pip_value:       Price units per pip (XAUUSD: 1.0 meaning $1 per pip).
        swing_length:    HTF pivot lookback.
        internal_length: HTF internal structure lookback.
        ltf_lookback:    Number of LTF bars to scan for zone proximity.
        zone_tolerance_pct: ±% band around zone edges for "price inside zone" test.
    """

    # Grade rank used for min_grade comparison (higher = better quality)
    _GRADE_RANK: dict[str, int] = {"A": 3, "B": 2, "C": 1, "F": 0}

    def __init__(
        self,
        df_htf:              pd.DataFrame,
        min_htf_score:       float = 4.0,
        sl_pips:             float = 20.0,
        max_sl_pips:         float = 30.0,
        pip_value:           float = 1.0,
        swing_length:        int   = 50,
        internal_length:     int   = 5,
        atr_period:          int   = 200,
        ltf_lookback:        int   = 20,
        zone_tolerance_pct:  float = 0.003,
        allowed_zones:       list[str] | None  = None,
        min_grade:           PatternGrade       = PatternGrade.C,
        grade_b_threshold:   float              = 6.0,
        grade_a_threshold:   float              = 8.0,
        killzone_only:       bool               = True,
        df_daily:            pd.DataFrame | None = None,
    ) -> None:
        self._df_htf             = df_htf
        self._min_htf_score      = min_htf_score
        self._sl_pips            = sl_pips
        self._max_sl_pips        = max_sl_pips
        self._pip_value          = pip_value
        self._swing_length       = swing_length
        self._internal_length    = internal_length
        self._atr_period         = atr_period
        self._ltf_lookback       = ltf_lookback
        self._zone_tol           = zone_tolerance_pct
        self._allowed_zones      = set(allowed_zones) if allowed_zones is not None else None
        self._min_grade          = min_grade
        self._grade_b_threshold  = grade_b_threshold
        self._grade_a_threshold  = grade_a_threshold
        self._killzone_only      = killzone_only
        self._df_daily           = df_daily
        self._min_htf_bars       = max(swing_length, internal_length) * 2

        # Cache: htf_bar_index → SMCResult  (avoid re-running full analysis each 1M bar)
        self._htf_cache: dict[int, SMCResult] = {}

    # ------------------------------------------------------------------ #
    # Strategy interface                                                   #
    # ------------------------------------------------------------------ #

    def generate_signal(self, df: pd.DataFrame, bar_index: int) -> Signal:
        """
        df is the LTF (e.g. 1M) DataFrame.  Called once per LTF bar.

        Steps:
          1. Find the last closed HTF bar before the current LTF timestamp.
          2. Run (cached) HTF SMC analysis.
          3. Evaluate HTF confluence for both directions.
          4. If tradeable setup: check whether current LTF price is inside an
             active HTF zone aligned with the HTF direction.
          5. Check LTF internal bias confirms direction (BOS/CHoCH trigger).
          6. Compute SL distance; reject if > max_sl_pips.
          7. Emit Signal.
        """
        ltf_ts    = df.index[bar_index]
        close     = float(df["close"].iloc[bar_index])
        bar_low   = float(df["low"].iloc[bar_index])
        bar_high  = float(df["high"].iloc[bar_index])

        # ── 0. Kill Zone gate (cheapest check — runs before HTF analysis) ─
        if self._killzone_only:
            kz = killzone_name(ltf_ts)
            if kz is None:
                return Signal(SignalType.NONE, 0.0, "outside killzone", bar_index)

        # ── 1. Locate last closed HTF bar ────────────────────────────────
        htf_bar_idx = self._last_htf_bar(ltf_ts)
        if htf_bar_idx < self._min_htf_bars:
            return Signal(SignalType.NONE, 0.0, "insufficient HTF data", bar_index)

        # ── 2. HTF SMC analysis (cached) ─────────────────────────────────
        htf_result = self._htf_analysis(htf_bar_idx)

        # ── 3. HTF structure gates ────────────────────────────────────────
        # Gate 1: swing bias must be defined
        direction = htf_result.swing_bias
        if direction == 0:
            return Signal(SignalType.NONE, 0.0, "no HTF swing bias", bar_index)

        # Gate 8: internal bias must confirm swing direction
        if htf_result.internal_bias != direction:
            return Signal(SignalType.NONE, 0.0, "HTF internal bias mismatch", bar_index)

        # ── 3.5. Daily bias alignment gate ───────────────────────────────
        if self._df_daily is not None:
            db = get_daily_bias(self._df_daily, ltf_ts)
            if db != 0 and db != direction:
                db_name  = "bullish" if db == BULLISH else "bearish"
                dir_name = "bullish" if direction == BULLISH else "bearish"
                return Signal(
                    SignalType.NONE, 0.0,
                    f"daily bias {db_name} conflicts with HTF {dir_name}",
                    bar_index,
                )

        # ── 4. Price inside active HTF zone ──────────────────────────────
        # Use LTF (1M) close price against HTF zones — this is the core of
        # sniper entry: price has retraced into an HTF zone after the HTF bar closed
        zone_type = self._in_htf_zone(htf_result, htf_bar_idx, close, direction)
        if zone_type is None:
            return Signal(SignalType.NONE, 0.0, "price outside HTF zone", bar_index)

        # Zone whitelist filter — skip early before expensive confluence scoring
        if self._allowed_zones is not None and zone_type not in self._allowed_zones:
            return Signal(SignalType.NONE, 0.0, f"zone {zone_type} not in allowed_zones", bar_index)

        # ── 5. Full confluence score (LTF price evaluated against HTF zones) ─
        # Gates (1 and 8) already validated above; score only needs 2 more
        # active factors (zone itself counts as one → min_score=3).
        cs = best_confluence(
            htf_result, close, htf_bar_idx,
            min_score=3.0,
            timestamp=ltf_ts,
        )
        if cs is None:
            return Signal(SignalType.NONE, 0.0, "HTF score below threshold", bar_index)

        # Grade filter — applied with configurable A/B thresholds
        grade = cs.grade(
            min_score=3.0,
            b_threshold=self._grade_b_threshold,
            a_threshold=self._grade_a_threshold,
        )
        if self._GRADE_RANK.get(grade.value, 0) < self._GRADE_RANK.get(self._min_grade.value, 0):
            return Signal(
                SignalType.NONE, 0.0,
                f"grade {grade.value} below min {self._min_grade.value} (score={cs.active_count})",
                bar_index,
            )

        # ── 6. LTF internal bias confirmation ────────────────────────────
        if not self._ltf_confirms(df, bar_index, direction):
            return Signal(SignalType.NONE, 0.0, "LTF bias not confirmed", bar_index)

        # ── 7. SL distance check ─────────────────────────────────────────
        sl_pips, sl_price = self._compute_sl(
            close, bar_low, bar_high, direction,
        )
        if sl_pips > self._max_sl_pips:
            return Signal(
                SignalType.NONE, 0.0,
                f"SL too wide ({sl_pips:.1f} pips > max {self._max_sl_pips})",
                bar_index,
            )

        # ── 8. Emit signal ────────────────────────────────────────────────
        stype  = SignalType.LONG if direction == BULLISH else SignalType.SHORT
        reason = (
            f"MTF grade={grade.value} score={cs.active_count}/10 "
            f"zone={zone_type} SL={sl_pips:.1f}pips"
        )
        return Signal(stype, cs.confidence, reason, bar_index)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _last_htf_bar(self, ltf_ts: pd.Timestamp) -> int:
        """Return the index of the last HTF bar whose open <= ltf_ts."""
        idx = self._df_htf.index.searchsorted(ltf_ts, side="right") - 1
        return max(0, int(idx))

    def _htf_analysis(self, htf_bar_idx: int) -> SMCResult:
        """Return cached SMCResult for the HTF window ending at htf_bar_idx."""
        if htf_bar_idx not in self._htf_cache:
            window = self._df_htf.iloc[: htf_bar_idx + 1]
            self._htf_cache[htf_bar_idx] = analyze(
                window,
                swing_length=self._swing_length,
                internal_length=self._internal_length,
                atr_period=self._atr_period,
            )
        return self._htf_cache[htf_bar_idx]

    def _in_htf_zone(
        self,
        result:    SMCResult,
        bar_idx:   int,
        price:     float,
        direction: int,
    ) -> str | None:
        """
        Return the zone type ("OB", "FVG", "OTE") if price is inside an active
        HTF zone aligned with *direction*, else None.
        """
        tol = self._zone_tol

        # Order Blocks
        all_obs = (
            get_active_order_blocks(result.internal_obs, bar_idx)
            + get_active_order_blocks(result.swing_obs, bar_idx)
        )
        for ob in all_obs:
            if ob.direction == direction:
                lo = ob.low  * (1 - tol)
                hi = ob.high * (1 + tol)
                if lo <= price <= hi:
                    return "OB"

        # Fair Value Gaps
        for fvg in get_active_fvgs(result.fvgs, bar_idx):
            if fvg.direction == direction:
                lo = fvg.bottom * (1 - tol)
                hi = fvg.top    * (1 + tol)
                if lo <= price <= hi:
                    return "FVG"

        # OTE Fibonacci zone (61.8–78.6 %)
        fib_zone = get_latest_fib_zone(result.fib_zones, bar_idx, direction)
        if fib_zone is not None and fib_zone.is_in_ote(price):
            return "OTE"

        return None

    def _ltf_confirms(
        self,
        df:        pd.DataFrame,
        bar_index: int,
        direction: int,
    ) -> bool:
        """
        Return True if the LTF internal structure bias at *bar_index* matches
        *direction*.

        Uses a lightweight SMC analysis on the last *ltf_lookback* LTF bars
        to detect BOS/CHoCH that confirms the HTF direction.
        """
        start = max(0, bar_index - self._ltf_lookback + 1)
        window = df.iloc[start : bar_index + 1]
        if len(window) < self._internal_length * 2 + 1:
            return False
        try:
            ltf_result = analyze(
                window,
                swing_length=min(self._swing_length, len(window) // 2),
                internal_length=self._internal_length,
                atr_period=min(self._atr_period, len(window)),
            )
        except Exception:
            return False
        return ltf_result.internal_bias == direction

    def _compute_sl(
        self,
        entry:     float,
        bar_low:   float,
        bar_high:  float,
        direction: int,
    ) -> tuple[float, float]:
        """
        Compute (sl_pips, sl_price) for this entry bar.

        Default SL = _sl_pips below/above entry.
        If the bar's natural range is wider, use the natural range (up to max_sl_pips).
        For longs:  SL below bar_low  (or entry - sl_pips, whichever is lower)
        For shorts: SL above bar_high (or entry + sl_pips, whichever is higher)
        """
        pv = self._pip_value

        if direction == BULLISH:
            natural_sl = bar_low - pv  # 1 pip below the bar low
            default_sl = entry - self._sl_pips * pv
            sl_price   = min(natural_sl, default_sl)
            sl_pips    = (entry - sl_price) / pv
        else:
            natural_sl = bar_high + pv
            default_sl = entry + self._sl_pips * pv
            sl_price   = max(natural_sl, default_sl)
            sl_pips    = (sl_price - entry) / pv

        return sl_pips, sl_price
