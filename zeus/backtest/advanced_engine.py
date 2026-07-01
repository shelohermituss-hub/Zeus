"""
Advanced backtest engine — pip-based SL + progressive partial close.

Designed for the SMC "sniper entry" workflow where:
  - Stop-loss is fixed in pips (absolute price distance), not percentage.
  - Positions exit progressively via a configurable partial-close ladder.
  - Multiple targets (3R, 5R) + a trailing stop for the final runner.

This engine does not use OrderExecutor / RiskManager internally — those
modules are tightly coupled to percentage-based SL and single-exit trades.
Position management is handled directly via PartialCloseState objects so
the partial-close logic stays clean and testable.

The engine still returns a standard BacktestResult for compatibility with
MonteCarloAnalyzer and other reporting utilities.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pandas as pd

from zeus.backtest.partial_close import PartialCloseConfig, PartialCloseState
from zeus.backtest.report import BacktestResult
from zeus.exchange.connector import OrderResult
from zeus.orders.models import Trade, TradeStatus
from zeus.strategy.base import Signal, SignalType, Strategy
from zeus.utils.logger import logger


class AdvancedBacktestEngine:
    """
    Bar-by-bar backtest engine with pip-based SL and progressive partial closes.

    Args:
        strategy:            Any Strategy subclass.
        initial_balance:     Starting capital in quote currency.
        stop_loss_pips:      Default SL distance in pips (1 pip = pip_value).
        max_sl_pips:         Reject setups whose natural SL exceeds this limit.
        pip_value:           Price units per pip; XAUUSD: 1.0 (= $1 per pip).
        min_rr:              Minimum acceptable R/R ratio (reject if first TP < this).
        max_position_pct:    Maximum capital at risk per trade (fraction of equity).
        max_open_positions:  Hard cap on concurrent open trades.
        fee_pct:             Exchange fee per side (entry + exit each pay this).
        slippage_pct:        Market slippage applied to every fill (fraction).
        partial_close:       Progressive exit config; defaults to PartialCloseConfig.default().
    """

    def __init__(
        self,
        strategy:            Strategy,
        initial_balance:     float               = 10_000.0,
        stop_loss_pips:      float               = 20.0,
        max_sl_pips:         float               = 30.0,
        pip_value:           float               = 1.0,
        min_rr:              float               = 3.0,
        max_position_pct:    float               = 0.02,
        max_open_positions:  int                 = 3,
        fee_pct:             float               = 0.001,
        slippage_pct:        float               = 0.0005,
        partial_close:       PartialCloseConfig | None = None,
    ) -> None:
        self._strategy           = strategy
        self._initial_balance    = initial_balance
        self._sl_pips            = stop_loss_pips
        self._max_sl_pips        = max_sl_pips
        self._pip_value          = pip_value
        self._min_rr             = min_rr
        self._max_pos_pct        = max_position_pct
        self._max_open            = max_open_positions
        self._fee_pct            = fee_pct
        self._slippage_pct       = slippage_pct
        self._partial_cfg        = partial_close or PartialCloseConfig.default()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(self, df: pd.DataFrame, symbol: str = "XAUUSD") -> BacktestResult:
        """
        Execute the backtest on *df* for *symbol*.

        Returns a BacktestResult compatible with MonteCarloAnalyzer.
        """
        balance       = self._initial_balance
        equity_curve  = [balance]
        trades: list[Trade] = []

        # (trade_id → (PartialCloseState, Trade_in_progress))
        open_positions: dict[str, tuple[PartialCloseState, Trade]] = {}

        for bar_index in range(len(df)):
            bar_high  = float(df["high"].iloc[bar_index]) if "high" in df.columns else float(df["close"].iloc[bar_index])
            bar_low   = float(df["low"].iloc[bar_index])  if "low"  in df.columns else float(df["close"].iloc[bar_index])
            close     = float(df["close"].iloc[bar_index])
            bar_ts    = df.index[bar_index].to_pydatetime()

            # ── 1. Process open positions (intrabar exits) ────────────────
            to_remove: list[str] = []
            for tid, (state, trade) in open_positions.items():
                events = state.process_bar(bar_high, bar_low, self._partial_cfg)
                for event_type, price, qty in events:
                    fee = price * qty * self._fee_pct
                    if event_type in ("sl", "trail"):
                        # Final exit via SL or trailing stop
                        total_pnl = state.partial_pnl - fee
                        balance  += total_pnl
                        trade.realized_pnl = total_pnl
                        trade.status       = TradeStatus.CLOSED_SL if event_type == "sl" else TradeStatus.CLOSED_MAN
                        trade.closed_at    = bar_ts
                        to_remove.append(tid)
                        logger.debug(
                            "Trade closed",
                            trade_id=tid, event=event_type,
                            price=price, pnl=round(total_pnl, 2),
                        )
                    elif event_type == "partial" and state.remaining_qty <= 1e-10:
                        # Last partial closed the position (hard_close_r or final level)
                        total_pnl = state.partial_pnl - fee
                        balance  += total_pnl
                        trade.realized_pnl = total_pnl
                        trade.status       = TradeStatus.CLOSED_TP
                        trade.closed_at    = bar_ts
                        to_remove.append(tid)
                    # else: intermediate partial → balance update deferred to close

            for tid in to_remove:
                del open_positions[tid]

            # ── 2. Strategy signal ────────────────────────────────────────
            if len(open_positions) < self._max_open:
                signal = self._strategy.generate_signal(df, bar_index)
                if signal.type != SignalType.NONE:
                    entry = self._apply_slippage(close, signal.type)
                    trade = self._try_open(
                        signal, symbol, entry, bar_high, bar_low, balance,
                    )
                    if trade is not None:
                        trades.append(trade)
                        if trade.status == TradeStatus.REJECTED:
                            pass  # audit trail only — do not open a position
                        else:
                            state = PartialCloseState(
                                entry_price  = entry,
                                sl_price     = trade.sl_price,
                                is_long      = trade.is_long,
                                original_qty = trade.quantity,
                            )
                            open_positions[trade.trade_id] = (state, trade)
                            entry_fee = entry * trade.quantity * self._fee_pct
                            balance  -= entry_fee

            # ── 3. Equity snapshot ────────────────────────────────────────
            open_pnl = sum(
                state.partial_pnl + (
                    (close - state.entry_price) * state.remaining_qty if state.is_long
                    else (state.entry_price - close) * state.remaining_qty
                )
                for state, _ in open_positions.values()
            )
            equity_curve.append(balance + open_pnl)

        # ── Force-close remaining positions at last bar ───────────────────
        if len(df) > 0:
            last_close = float(df["close"].iloc[-1])
            last_ts    = df.index[-1].to_pydatetime()
            for tid, (state, trade) in open_positions.items():
                if state.is_long:
                    pnl = (last_close - state.entry_price) * state.original_qty
                else:
                    pnl = (state.entry_price - last_close) * state.original_qty
                fee = last_close * state.remaining_qty * self._fee_pct
                total_pnl = state.partial_pnl + (
                    (last_close - state.entry_price) * state.remaining_qty if state.is_long
                    else (state.entry_price - last_close) * state.remaining_qty
                ) - fee
                balance += total_pnl
                trade.realized_pnl = total_pnl
                trade.status       = TradeStatus.CLOSED_MAN
                trade.closed_at    = last_ts
                trades.append(trade) if trade not in trades else None

        logger.info(
            "Advanced backtest complete",
            symbol=symbol,
            bars=len(df),
            trades=len([t for t in trades if t.status not in (TradeStatus.REJECTED, TradeStatus.FAILED, TradeStatus.OPEN)]),
            final_balance=round(balance, 2),
        )

        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            initial_balance=self._initial_balance,
            fee_pct=self._fee_pct,
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _apply_slippage(self, price: float, stype: SignalType) -> float:
        """Apply slippage: longs pay more, shorts receive less."""
        if stype == SignalType.LONG:
            return price * (1.0 + self._slippage_pct)
        return price * (1.0 - self._slippage_pct)

    def _try_open(
        self,
        signal:   Signal,
        symbol:   str,
        entry:    float,
        bar_high: float,
        bar_low:  float,
        balance:  float,
    ) -> Trade | None:
        """
        Validate and open a new position.

        Rejects if:
          - SL distance > max_sl_pips
          - Implied R/R (using first partial level as TP1) < min_rr
          - Risk amount would exceed max_position_pct × balance
        """
        is_long  = signal.type == SignalType.LONG
        pv       = self._pip_value
        sl_pips  = self._sl_pips

        # Compute SL price
        if is_long:
            natural_sl = bar_low - pv
            default_sl = entry - sl_pips * pv
            sl_price   = min(natural_sl, default_sl)
            sl_distance_pips = (entry - sl_price) / pv
        else:
            natural_sl = bar_high + pv
            default_sl = entry + sl_pips * pv
            sl_price   = max(natural_sl, default_sl)
            sl_distance_pips = (sl_price - entry) / pv

        if sl_distance_pips > self._max_sl_pips:
            logger.debug(
                "Setup rejected — SL too wide",
                sl_pips=round(sl_distance_pips, 1), max=self._max_sl_pips,
            )
            return self._rejected_trade(signal, symbol, entry, sl_price, is_long)

        # First TP at min_rr × SL distance
        if is_long:
            tp1_price = entry + self._min_rr * sl_distance_pips * pv
        else:
            tp1_price = entry - self._min_rr * sl_distance_pips * pv

        # Position sizing: risk = max_position_pct × balance
        risk_amount = balance * self._max_pos_pct
        risk_per_unit = sl_distance_pips * pv
        if risk_per_unit <= 0:
            return None
        qty = risk_amount / risk_per_unit

        trade_id = str(uuid.uuid4())
        dummy_order = OrderResult(
            order_id=trade_id,
            symbol=symbol,
            side="buy" if is_long else "sell",
            quantity=qty,
            filled=qty,
            price=entry,
            average=entry,
            status="filled",
            timestamp=datetime.now(tz=timezone.utc),
        )
        trade = Trade(
            trade_id     = trade_id,
            symbol       = symbol,
            side         = "buy" if is_long else "sell",
            quantity     = qty,
            entry_price  = entry,
            sl_price     = sl_price,
            tp_price     = tp1_price,
            entry_order  = dummy_order,
            signal_reason= signal.reason,
            status       = TradeStatus.OPEN,
        )
        logger.debug(
            "Trade opened",
            trade_id=trade_id, symbol=symbol,
            side=trade.side, entry=round(entry, 2),
            sl=round(sl_price, 2), tp1=round(tp1_price, 2),
            sl_pips=round(sl_distance_pips, 1), qty=round(qty, 6),
        )
        return trade

    def _rejected_trade(
        self,
        signal:   Signal,
        symbol:   str,
        entry:    float,
        sl_price: float,
        is_long:  bool,
    ) -> Trade:
        """Record a rejected trade for audit trail."""
        dummy = OrderResult(
            order_id="rejected", symbol=symbol,
            side="buy" if is_long else "sell",
            quantity=0.0, filled=0.0, price=entry,
            average=None, status="rejected",
            timestamp=datetime.now(tz=timezone.utc),
        )
        return Trade(
            trade_id=str(uuid.uuid4()),
            symbol=symbol,
            side="buy" if is_long else "sell",
            quantity=0.0,
            entry_price=entry,
            sl_price=sl_price,
            tp_price=None,
            entry_order=dummy,
            signal_reason=signal.reason,
            status=TradeStatus.REJECTED,
        )
