"""
Session metrics — running statistics collected throughout a trading session.

SessionMetrics is updated by the caller on each tick and on each trade close.
It never queries the exchange or risk manager directly; all inputs are pushed in.

Usage pattern (paper / live engine):
    metrics = SessionMetrics(initial_balance=10_000.0)

    # on each tick:
    t0 = time.monotonic()
    ... fetch + signal + execute ...
    metrics.record_tick(latency_ms=(time.monotonic() - t0) * 1000, equity=current_equity)

    # when a trade closes:
    metrics.record_trade(closed_trade)

    # on API failure:
    metrics.record_api_error()

    # at any time:
    snap = metrics.snapshot()
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from zeus.orders.models import Trade, TradeStatus
from zeus.utils.logger import logger


@dataclass(frozen=True)
class MetricsSnapshot:
    """
    Point-in-time view of all session metrics.  Immutable once created.
    All monetary values are in the quote currency.
    """
    timestamp:        datetime

    # Profit & Loss
    realized_pnl:     float
    unrealized_pnl:   float
    total_pnl:        float

    # Trade statistics
    n_trades:         int
    n_winning:        int
    n_losing:         int
    win_rate:         float
    avg_win:          float
    avg_loss:         float
    profit_factor:    float

    # Equity and drawdown
    initial_balance:  float
    current_equity:   float
    peak_equity:      float
    drawdown_pct:     float     # current drawdown from peak
    max_drawdown_pct: float     # worst drawdown seen this session
    total_return_pct: float

    # API health
    total_ticks:      int
    api_errors:       int
    error_rate:       float
    avg_latency_ms:   float
    p99_latency_ms:   float

    def to_dict(self) -> dict:
        """Plain dict suitable for logging or serialisation."""
        return {
            "timestamp":        self.timestamp.isoformat(),
            "realized_pnl":     round(self.realized_pnl, 2),
            "unrealized_pnl":   round(self.unrealized_pnl, 2),
            "total_pnl":        round(self.total_pnl, 2),
            "n_trades":         self.n_trades,
            "win_rate":         round(self.win_rate * 100, 1),
            "avg_win":          round(self.avg_win, 2),
            "avg_loss":         round(self.avg_loss, 2),
            "profit_factor":    round(self.profit_factor, 3) if self.profit_factor != float("inf") else "inf",
            "current_equity":   round(self.current_equity, 2),
            "drawdown_pct":     round(self.drawdown_pct * 100, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct * 100, 2),
            "total_return_pct": round(self.total_return_pct * 100, 2),
            "total_ticks":      self.total_ticks,
            "api_errors":       self.api_errors,
            "error_rate":       round(self.error_rate * 100, 1),
            "avg_latency_ms":   round(self.avg_latency_ms, 1),
            "p99_latency_ms":   round(self.p99_latency_ms, 1),
        }


class SessionMetrics:
    """
    Accumulates metrics over the lifetime of a trading session.

    Thread-safety: not guaranteed.  Call from a single loop thread.

    Args:
        initial_balance: Starting capital used to compute return % and drawdown baseline.
    """

    def __init__(self, initial_balance: float) -> None:
        self._initial_balance = initial_balance
        self._closed_trades:  list[Trade] = []
        self._api_errors:     int = 0
        self._total_ticks:    int = 0
        self._latencies:      list[float] = []
        self._last_equity:    float = initial_balance
        self._peak_equity:    float = initial_balance
        self._max_drawdown_pct: float = 0.0

    # ------------------------------------------------------------------ #
    # Event recorders                                                      #
    # ------------------------------------------------------------------ #

    def record_trade(self, trade: Trade) -> None:
        """
        Register a closed trade in the audit trail.

        Open, REJECTED, and FAILED trades are silently ignored — they carry
        no realised P&L and must not distort win-rate statistics.
        """
        if trade.is_open or trade.status in (TradeStatus.REJECTED, TradeStatus.FAILED):
            return
        self._closed_trades.append(trade)
        logger.info(
            "Trade recorded",
            trade_id=trade.trade_id,
            symbol=trade.symbol,
            side=trade.side,
            status=trade.status.value,
            quantity=trade.quantity,
            entry_price=trade.entry_price,
            pnl=round(trade.realized_pnl, 2),
            reason=trade.signal_reason,
        )

    def record_tick(self, latency_ms: float, equity: float) -> None:
        """
        Register a completed tick.

        Args:
            latency_ms: Wall-clock time for the full fetch→signal→execute cycle.
            equity:     Total equity at the end of this tick (including unrealised P&L).
        """
        self._total_ticks += 1
        self._latencies.append(latency_ms)
        self._last_equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity
        if self._peak_equity > 0:
            dd = (self._peak_equity - equity) / self._peak_equity
            if dd > self._max_drawdown_pct:
                self._max_drawdown_pct = dd

    def record_api_error(self) -> None:
        """Register one API error (network timeout, exchange error, etc.)."""
        self._api_errors += 1
        logger.warning(
            "API error recorded",
            total_errors=self._api_errors,
            total_ticks=self._total_ticks,
        )

    # ------------------------------------------------------------------ #
    # Snapshot                                                             #
    # ------------------------------------------------------------------ #

    def snapshot(self, unrealized_pnl: float = 0.0) -> MetricsSnapshot:
        """
        Return a point-in-time view of all metrics.

        Args:
            unrealized_pnl: Open position P&L to add on top of last recorded equity.
                            Pass 0.0 (default) if equity already includes unrealised.
        """
        realized_pnl = sum(t.realized_pnl for t in self._closed_trades)
        current_equity = self._last_equity + unrealized_pnl
        total_pnl = realized_pnl + unrealized_pnl

        # Drawdown including current unrealised position
        peak = max(self._peak_equity, current_equity)
        dd_pct = (peak - current_equity) / peak if peak > 0 else 0.0

        n_trades = len(self._closed_trades)
        n_winning = sum(1 for t in self._closed_trades if t.realized_pnl > 0)
        n_losing  = sum(1 for t in self._closed_trades if t.realized_pnl <= 0)

        wins   = [t.realized_pnl for t in self._closed_trades if t.realized_pnl > 0]
        losses = [t.realized_pnl for t in self._closed_trades if t.realized_pnl < 0]

        avg_win    = sum(wins)   / len(wins)   if wins   else 0.0
        avg_loss   = sum(losses) / len(losses) if losses else 0.0
        gross_profit = sum(wins)
        gross_loss   = abs(sum(losses))
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0
            else (float("inf") if gross_profit > 0 else 0.0)
        )

        avg_latency = sum(self._latencies) / len(self._latencies) if self._latencies else 0.0
        p99_latency = (
            sorted(self._latencies)[int(0.99 * len(self._latencies))]
            if self._latencies else 0.0
        )
        error_rate = (
            self._api_errors / self._total_ticks if self._total_ticks > 0 else 0.0
        )
        total_return = (
            (current_equity - self._initial_balance) / self._initial_balance
            if self._initial_balance > 0 else 0.0
        )

        return MetricsSnapshot(
            timestamp=datetime.now(tz=timezone.utc),
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_pnl=total_pnl,
            n_trades=n_trades,
            n_winning=n_winning,
            n_losing=n_losing,
            win_rate=n_winning / n_trades if n_trades > 0 else 0.0,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            initial_balance=self._initial_balance,
            current_equity=current_equity,
            peak_equity=peak,
            drawdown_pct=dd_pct,
            max_drawdown_pct=self._max_drawdown_pct,
            total_return_pct=total_return,
            total_ticks=self._total_ticks,
            api_errors=self._api_errors,
            error_rate=error_rate,
            avg_latency_ms=avg_latency,
            p99_latency_ms=p99_latency,
        )

    def log_summary(self) -> None:
        """Log a one-line session summary at INFO level."""
        s = self.snapshot()
        logger.info(
            "Session summary",
            n_trades=s.n_trades,
            win_rate=f"{s.win_rate:.1%}",
            realized_pnl=round(s.realized_pnl, 2),
            max_drawdown=f"{s.max_drawdown_pct:.2%}",
            total_return=f"{s.total_return_pct:.2%}",
            api_errors=s.api_errors,
            avg_latency_ms=round(s.avg_latency_ms, 1),
        )
