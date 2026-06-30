"""
BacktestResult — immutable container for all metrics produced by a backtest run.

Metrics are computed lazily from the trade list and equity curve; nothing is
stored redundantly.  Call summary() for a plain dict suitable for logging.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from zeus.orders.models import Trade, TradeStatus


@dataclass
class BacktestResult:
    """
    Output of BacktestEngine.run().

    Args:
        trades:          All trades (open, closed, rejected, failed) in order.
        equity_curve:    Equity value after each bar (length = n_bars + 1;
                         index 0 is the initial balance before bar 0).
        initial_balance: Starting capital in quote currency.
        fee_pct:         Flat fee applied to each side (entry + exit).
    """
    trades:          list[Trade]
    equity_curve:    list[float]
    initial_balance: float
    fee_pct:         float = 0.001

    # ------------------------------------------------------------------ #
    # Trade filtering                                                      #
    # ------------------------------------------------------------------ #

    @property
    def closed_trades(self) -> list[Trade]:
        """Trades that actually executed and reached a closed state."""
        return [
            t for t in self.trades
            if t.status not in (TradeStatus.REJECTED, TradeStatus.FAILED, TradeStatus.OPEN)
        ]

    # ------------------------------------------------------------------ #
    # Counts                                                               #
    # ------------------------------------------------------------------ #

    @property
    def n_trades(self) -> int:
        return len(self.closed_trades)

    @property
    def n_winning(self) -> int:
        return sum(1 for t in self.closed_trades if t.realized_pnl > 0)

    @property
    def n_losing(self) -> int:
        return sum(1 for t in self.closed_trades if t.realized_pnl <= 0)

    # ------------------------------------------------------------------ #
    # Rates and P&L                                                        #
    # ------------------------------------------------------------------ #

    @property
    def win_rate(self) -> float:
        if self.n_trades == 0:
            return 0.0
        return self.n_winning / self.n_trades

    @property
    def total_gross_pnl(self) -> float:
        return sum(t.realized_pnl for t in self.closed_trades)

    @property
    def total_fees(self) -> float:
        """Sum of entry + exit exchange fees for all closed trades."""
        total = 0.0
        for t in self.closed_trades:
            entry_value = t.quantity * t.entry_price
            # Recover exit price from realized P&L
            if t.quantity > 0:
                pnl_per_unit = t.realized_pnl / t.quantity
                exit_price = (
                    t.entry_price + pnl_per_unit
                    if t.is_long
                    else t.entry_price - pnl_per_unit
                )
            else:
                exit_price = t.entry_price
            exit_value = t.quantity * abs(exit_price)
            total += (entry_value + exit_value) * self.fee_pct
        return total

    @property
    def net_pnl(self) -> float:
        return self.total_gross_pnl - self.total_fees

    # ------------------------------------------------------------------ #
    # Equity and drawdown                                                  #
    # ------------------------------------------------------------------ #

    @property
    def final_equity(self) -> float:
        return self.equity_curve[-1] if self.equity_curve else self.initial_balance

    @property
    def total_return_pct(self) -> float:
        if self.initial_balance == 0:
            return 0.0
        return (self.final_equity - self.initial_balance) / self.initial_balance

    @property
    def max_drawdown_pct(self) -> float:
        if not self.equity_curve:
            return 0.0
        peak = self.equity_curve[0]
        max_dd = 0.0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            if peak > 0:
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd
        return max_dd

    # ------------------------------------------------------------------ #
    # Per-trade averages                                                   #
    # ------------------------------------------------------------------ #

    @property
    def avg_win(self) -> float:
        wins = [t.realized_pnl for t in self.closed_trades if t.realized_pnl > 0]
        return sum(wins) / len(wins) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [t.realized_pnl for t in self.closed_trades if t.realized_pnl < 0]
        return sum(losses) / len(losses) if losses else 0.0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.realized_pnl for t in self.closed_trades if t.realized_pnl > 0)
        gross_loss   = abs(sum(t.realized_pnl for t in self.closed_trades if t.realized_pnl < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    # ------------------------------------------------------------------ #
    # Risk-adjusted return                                                 #
    # ------------------------------------------------------------------ #

    @property
    def sharpe_ratio(self) -> float:
        """
        Bar-level Sharpe (not annualised).
        Multiply by sqrt(N) where N = bars per year for the annualised figure.
        """
        if len(self.equity_curve) < 2:
            return 0.0
        returns = []
        for i in range(1, len(self.equity_curve)):
            prev = self.equity_curve[i - 1]
            if prev > 0:
                returns.append((self.equity_curve[i] - prev) / prev)
        if len(returns) < 2:
            return 0.0
        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        std_r = variance ** 0.5
        if std_r == 0:
            return 0.0
        return mean_r / std_r

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #

    def summary(self) -> dict:
        """Return all key metrics as a plain dict (suitable for logging)."""
        pf = self.profit_factor
        return {
            "initial_balance":   self.initial_balance,
            "final_equity":      round(self.final_equity, 2),
            "net_pnl":           round(self.net_pnl, 2),
            "total_return_pct":  round(self.total_return_pct * 100, 2),
            "n_trades":          self.n_trades,
            "win_rate":          round(self.win_rate * 100, 1),
            "avg_win":           round(self.avg_win, 2),
            "avg_loss":          round(self.avg_loss, 2),
            "profit_factor":     round(pf, 4) if pf != float("inf") else "inf",
            "max_drawdown_pct":  round(self.max_drawdown_pct * 100, 2),
            "sharpe_ratio":      round(self.sharpe_ratio, 4),
            "total_fees":        round(self.total_fees, 2),
        }
