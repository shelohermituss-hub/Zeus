"""
Alert manager — checks metrics snapshots against configurable thresholds
and emits structured log entries when conditions are breached.

Alerts are stateless: every call to check() re-evaluates all conditions.
Callers decide how often to run checks (e.g. once per tick, once per minute).
The returned list can be used to drive notifications, dashboards, or kill-switch
logic without coupling the alert system to external I/O.

Alert levels:
    WARNING — degraded but trading continues.
    ERROR   — critical condition; human or automated intervention recommended.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from zeus.monitoring.metrics import MetricsSnapshot
from zeus.risk.models import RiskState
from zeus.utils.logger import logger


@dataclass
class Alert:
    """One triggered alert condition."""
    level:   str   # "WARNING" | "ERROR"
    code:    str   # machine-readable identifier (e.g. "DRAWDOWN_CRITICAL")
    message: str   # human-readable description
    context: dict  = field(default_factory=dict)   # structured key-value data


@dataclass
class AlertConfig:
    """
    Thresholds that govern when alerts fire.

    Defaults are conservative.  Override on construction for tighter or looser policies.
    """
    # Drawdown
    drawdown_warn_pct:   float = 0.05   # 5 % → WARNING
    drawdown_error_pct:  float = 0.10   # 10 % → ERROR

    # Daily loss (requires RiskState)
    daily_loss_warn_pct: float = 0.03   # 3 % of daily_start_equity → WARNING

    # API health
    error_rate_warn_pct: float = 0.10   # 10 % of ticks fail → WARNING
    latency_warn_ms:     float = 5_000  # avg latency > 5 s → WARNING

    # Strategy health
    min_win_rate:              float = 0.30   # < 30 % win rate → WARNING
    min_trades_for_win_rate:   int   = 10     # ignore until this many trades


class AlertManager:
    """
    Stateless alert evaluator.

    Create once per session; call check_and_log() on each tick or on demand.
    """

    def __init__(self, config: AlertConfig | None = None) -> None:
        self._cfg = config or AlertConfig()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def check(
        self,
        snapshot: MetricsSnapshot,
        risk_state: RiskState | None = None,
    ) -> list[Alert]:
        """
        Evaluate all alert conditions against *snapshot* (and optionally *risk_state*).

        Returns a list of triggered alerts (empty = all clear).
        Does NOT write to logs; use check_and_log() for that.
        """
        alerts: list[Alert] = []
        cfg = self._cfg

        # ---- Drawdown -------------------------------------------------
        dd = snapshot.drawdown_pct
        if dd >= cfg.drawdown_error_pct:
            alerts.append(Alert(
                level="ERROR",
                code="DRAWDOWN_CRITICAL",
                message=(
                    f"Drawdown {dd:.2%} reached critical threshold "
                    f"{cfg.drawdown_error_pct:.2%} — consider stopping the bot"
                ),
                context={"drawdown_pct": dd, "threshold": cfg.drawdown_error_pct},
            ))
        elif dd >= cfg.drawdown_warn_pct:
            alerts.append(Alert(
                level="WARNING",
                code="DRAWDOWN_HIGH",
                message=f"Drawdown {dd:.2%} exceeded warning level {cfg.drawdown_warn_pct:.2%}",
                context={"drawdown_pct": dd, "threshold": cfg.drawdown_warn_pct},
            ))

        # ---- API error rate -------------------------------------------
        if snapshot.error_rate >= cfg.error_rate_warn_pct:
            alerts.append(Alert(
                level="WARNING",
                code="HIGH_ERROR_RATE",
                message=(
                    f"API error rate {snapshot.error_rate:.2%} exceeded "
                    f"threshold {cfg.error_rate_warn_pct:.2%}"
                ),
                context={
                    "error_rate": snapshot.error_rate,
                    "api_errors": snapshot.api_errors,
                    "total_ticks": snapshot.total_ticks,
                },
            ))

        # ---- Latency --------------------------------------------------
        if snapshot.avg_latency_ms >= cfg.latency_warn_ms:
            alerts.append(Alert(
                level="WARNING",
                code="HIGH_LATENCY",
                message=(
                    f"Average tick latency {snapshot.avg_latency_ms:.0f} ms "
                    f"exceeded threshold {cfg.latency_warn_ms:.0f} ms"
                ),
                context={
                    "avg_latency_ms": snapshot.avg_latency_ms,
                    "p99_latency_ms": snapshot.p99_latency_ms,
                },
            ))

        # ---- Win rate (only meaningful after enough trades) -----------
        if (
            snapshot.n_trades >= cfg.min_trades_for_win_rate
            and snapshot.win_rate < cfg.min_win_rate
        ):
            alerts.append(Alert(
                level="WARNING",
                code="LOW_WIN_RATE",
                message=(
                    f"Win rate {snapshot.win_rate:.1%} below minimum "
                    f"{cfg.min_win_rate:.1%} after {snapshot.n_trades} trades"
                ),
                context={
                    "win_rate": snapshot.win_rate,
                    "n_trades": snapshot.n_trades,
                    "min_win_rate": cfg.min_win_rate,
                },
            ))

        # ---- Risk state checks (optional) ----------------------------
        if risk_state is not None:
            if risk_state.kill_switch_active:
                alerts.append(Alert(
                    level="ERROR",
                    code="KILL_SWITCH_ACTIVE",
                    message="Kill switch is active — all new orders are blocked",
                    context={"kill_switch": True},
                ))

            if risk_state.daily_start_equity > 0:
                daily_loss_pct = (
                    -risk_state.daily_realized_pnl / risk_state.daily_start_equity
                )
                if daily_loss_pct >= cfg.daily_loss_warn_pct:
                    alerts.append(Alert(
                        level="WARNING",
                        code="DAILY_LOSS_HIGH",
                        message=(
                            f"Daily loss {daily_loss_pct:.2%} exceeded "
                            f"threshold {cfg.daily_loss_warn_pct:.2%}"
                        ),
                        context={
                            "daily_loss_pct": daily_loss_pct,
                            "daily_realized_pnl": risk_state.daily_realized_pnl,
                        },
                    ))

        return alerts

    def check_and_log(
        self,
        snapshot: MetricsSnapshot,
        risk_state: RiskState | None = None,
    ) -> list[Alert]:
        """
        Evaluate all conditions, log every triggered alert, and return the list.

        WARNING-level alerts are logged at WARNING; ERROR-level at ERROR.
        """
        alerts = self.check(snapshot, risk_state)
        for alert in alerts:
            if alert.level == "ERROR":
                logger.error(alert.message, code=alert.code, **alert.context)
            else:
                logger.warning(alert.message, code=alert.code, **alert.context)
        return alerts
