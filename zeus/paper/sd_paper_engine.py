"""
Paper trading engine for the Supply & Demand strategy — V4 bidirectional champion.

Configuration (V4 — validated OOS across 2024/2025/2026):
    SDStrategy:
        risk_reward=1.5, min_zone_score=5.0, min_wyckoff_score=5.9 (long),
        min_wyckoff_score_short=8.5 (bidirectional), trend_slope_lookback=3,
        trend_slope_lookback_short=20, use_price_above_ema=True,
        use_price_above_ema_for_shorts=False, session 07–21h UTC, signal_cooldown=10

    Execution:
        TP1 at 1.25R (50% partial exit), TP2 at 1.5R (remaining 50%)
        SL → break-even after TP1 hit
        Max 1 loss per calendar day (no more entries after first loss)
        Max 4 losses per calendar month
        Long: entry at ask (open + SPREAD), Short: entry at bid (open − SPREAD)

Loop design:
    - Fetches M1 OHLCV from the market connector every `poll_interval` seconds.
    - Resamples M1 → M15 in-process (no separate M15 fetch needed).
    - Executes signal detection on the last COMPLETE M1 bar (penultimate row).
    - Enters at the open of the bar following the MSS bar (best-effort paper fill).
    - Tracks one open trade at a time (long or short, no pyramiding).
    - Applies spread on both entry and stop exit.
    - Implements partial TP1 exit and SL-to-BE management.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd

from zeus.backtest.data_loader import resample_ohlcv
from zeus.exchange.connector import ExchangeConnector
from zeus.monitoring.metrics import SessionMetrics
from zeus.monitoring.telegram import TelegramNotifier
from zeus.strategy.supply_demand.sd_strategy import SDSignal, SDStrategy
from zeus.strategy.supply_demand.wyckoff import WyckoffDetector
from zeus.strategy.supply_demand.zone_detector import ZoneDetector
from zeus.utils.logger import logger

# ── V4 bidirectional champion params ──────────────────────────────────────────
_STRATEGY_PARAMS = dict(
    risk_reward                   = 1.5,
    min_zone_score                = 5.0,
    min_wyckoff_score             = 5.9,
    min_composite_score           = 5.0,
    min_wyckoff_score_short       = 8.5,    # bidirectional: supply zones enabled
    signal_cooldown               = 10,
    use_trend_filter              = True,
    trend_slope_lookback          = 3,
    trend_slope_lookback_short    = 20,     # longer lookback: captures macro downtrend
    use_price_above_ema           = True,
    use_price_above_ema_for_shorts = False,  # supply zones sit above EMA50 — no price filter
    use_session_filter            = True,
    session_start_utc             = 7,
    session_end_utc               = 21,
    max_signals_per_day           = 6,
    use_adx_filter                = False,
    use_h4_trend_filter           = False,
    use_rsi_filter                = False,
)

_WYCKOFF_PARAMS = dict(
    lookback             = 200,
    max_accum_bars       = 20,
    accum_range_mult     = 6.0,
    mss_lookback         = 60,
    min_spring_sweep_pct = 0.05,
    min_mss_strength_pct = 0.03,
)

# Partial exit: take 50% at TP1 (1.25R), let remaining 50% run to TP2 (1.5R)
TP1_R:    float = 1.25
TP1_SIZE: float = 0.50   # fraction closed at TP1

# Spread for XAUUSD CFD (USD per oz)
SPREAD: float = 0.30

# Maximum M1 bars to fetch (3000 ≈ 50h of data = ~200 M15 bars for zone detection)
M1_LIMIT: int = 3_000


# ── Open trade tracker ────────────────────────────────────────────────────────

@dataclass
class SDLiveTrade:
    """State of a single open paper trade."""
    signal:      SDSignal
    entry_price: float           # effective fill (open + spread for longs)
    sl:          float           # current SL (moves to entry after TP1)
    tp1:         float           # TP1 level = entry + TP1_R × sl_dist
    tp2:         float           # TP2 level = entry + 1.5 × sl_dist
    sl_dist:     float           # |entry - original SL|  (= 1R distance)
    risk_amount: float           # equity × risk_pct
    size_full:   float           # total position in oz
    size_remaining: float        # position remaining after TP1 partial exit
    tp1_hit:     bool = False
    opened_at:   Optional[pd.Timestamp] = None
    closed_at:   Optional[pd.Timestamp] = None
    outcome:     str = "open"    # "open" | "tp1_only" | "full_win" | "loss" | "scratch"
    pnl_r:       float = 0.0
    pnl_usd:     float = 0.0

    @property
    def is_open(self) -> bool:
        return self.outcome == "open"


# ── Engine ─────────────────────────────────────────────────────────────────────

class SDPaperEngine:
    """
    Paper trading loop for the V_FINAL_B SD strategy champion.

    Args:
        connector:       Live market data source (MT5 or CCXT connector).
                         Only fetch_ohlcv() is called — no real orders sent.
        symbol:          Instrument symbol (e.g. "XAUUSD" for MT5).
        initial_balance: Starting paper account balance.
        risk_pct:        Fraction of equity risked per trade (default 1%).
        poll_interval:   Seconds between ticks (default 60 = every M1 close).
        notifier:        Optional Telegram notifier for trade alerts.
    """

    def __init__(
        self,
        connector:       ExchangeConnector,
        symbol:          str   = "XAUUSD",
        initial_balance: float = 10_000.0,
        risk_pct:        float = 0.01,
        poll_interval:   float = 60.0,
        notifier:        Optional[TelegramNotifier] = None,
    ) -> None:
        self._connector      = connector
        self._symbol         = symbol
        self._equity         = initial_balance
        self._initial_equity = initial_balance
        self._risk_pct       = risk_pct
        self._poll_interval  = poll_interval
        self._notifier       = notifier or TelegramNotifier(bot_token="", chat_id="")
        self._running        = False

        self._strategy = SDStrategy(
            zone_detector    = ZoneDetector(),
            wyckoff_detector = WyckoffDetector(**_WYCKOFF_PARAMS),
            **_STRATEGY_PARAMS,
        )

        # Signal deduplication — keyed by formed_at timestamp
        self._seen_signals: set[pd.Timestamp] = set()

        # Currently open trade (at most one at a time)
        self._open_trade: Optional[SDLiveTrade] = None

        # Closed trade log
        self._closed_trades: list[SDLiveTrade] = []

        # Loss counters
        self._daily_losses:   int = 0
        self._monthly_losses: int = 0
        self._today:  Optional[date] = None
        self._month:  Optional[str]  = None

        self._metrics = SessionMetrics(initial_balance)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def metrics(self) -> SessionMetrics:
        return self._metrics

    def stop(self) -> None:
        self._running = False

    def run(self, max_ticks: Optional[int] = None) -> None:
        """Start the paper trading loop."""
        tick = 0
        self._running = True
        logger.info(
            "SD paper trading started",
            symbol=self._symbol,
            equity=self._equity,
            config="V4: bidir WS_short≥8.5, TP1@1.25R(50%)→TP2@1.5R, daily_loss_cap=1",
        )
        try:
            while self._running:
                if max_ticks is not None and tick >= max_ticks:
                    break
                self._tick()
                tick += 1
                if self._running and (max_ticks is None or tick < max_ticks):
                    time.sleep(self._poll_interval)
        except KeyboardInterrupt:
            logger.info("SD paper trading stopped by user")
        finally:
            self._running = False
            self._log_summary()

    # ── Tick ─────────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        t0 = time.monotonic()
        try:
            self._check_period_reset()

            # ── Fetch M1 data ─────────────────────────────────────────────
            m1_df = self._connector.fetch_ohlcv(self._symbol, "1m", M1_LIMIT)
            if m1_df is None or len(m1_df) < 202:
                logger.warning("Insufficient M1 data — tick skipped", symbol=self._symbol)
                return

            current_price = float(m1_df["close"].iloc[-1])

            # ── Exit check on open trade ──────────────────────────────────
            if self._open_trade is not None:
                self._check_exit(self._open_trade, m1_df)

            # ── Resample → M15 for zone detection ─────────────────────────
            m15_df = resample_ohlcv(m1_df, "15min")
            if len(m15_df) < 10:
                logger.warning("Insufficient M15 bars — tick skipped")
                return

            # ── Run strategy on all data ──────────────────────────────────
            signals = self._strategy.run(m15_df, m1_df)

            # ── Detect new signal on the last complete M1 bar ─────────────
            if (self._open_trade is None
                    and self._daily_losses < 1
                    and self._monthly_losses < 4
                    and signals):
                last_complete_ts = m1_df.index[-2]
                new_sig = next(
                    (s for s in reversed(signals)
                     if s.formed_at == last_complete_ts
                     and s.formed_at not in self._seen_signals),
                    None,
                )
                if new_sig is not None:
                    self._seen_signals.add(new_sig.formed_at)
                    self._enter_trade(new_sig, m1_df)

            # ── Metrics snapshot ──────────────────────────────────────────
            open_pnl = self._unrealised_pnl(current_price)
            equity   = self._equity + open_pnl
            latency  = (time.monotonic() - t0) * 1000
            self._metrics.record_tick(latency, equity)

            trade_info = (
                f"open trade: entry={self._open_trade.entry_price:.2f} "
                f"tp1_hit={self._open_trade.tp1_hit}"
                if self._open_trade else "no open trade"
            )
            logger.info(
                "Tick",
                symbol=self._symbol,
                price=round(current_price, 2),
                equity=round(equity, 2),
                daily_losses=self._daily_losses,
                monthly_losses=self._monthly_losses,
                trade=trade_info,
                latency_ms=round(latency, 1),
            )

        except Exception as exc:
            self._metrics.record_api_error()
            logger.error("Tick error", symbol=self._symbol, error=str(exc))

    # ── Trade management ──────────────────────────────────────────────────────

    def _enter_trade(self, signal: SDSignal, m1_df: pd.DataFrame) -> None:
        """Open a new paper trade (long or short) at the next bar's open."""
        entry_bar_idx = signal.bar_index + 1
        if entry_bar_idx >= len(m1_df):
            logger.debug("Signal bar too recent — entry skipped", formed_at=str(signal.formed_at))
            return

        is_long     = signal.direction == "long"
        raw_open    = float(m1_df.iloc[entry_bar_idx]["open"])
        # Long: buy at ask (raw + spread). Short: sell at bid (raw − spread).
        entry_price = (raw_open + SPREAD) if is_long else (raw_open - SPREAD)
        sl          = signal.stop_loss
        sl_dist     = abs(entry_price - sl)
        if sl_dist < 1e-6:
            return

        sign = 1.0 if is_long else -1.0
        tp1  = entry_price + sign * TP1_R * sl_dist
        tp2  = entry_price + sign * 1.5   * sl_dist

        risk_amount = self._equity * self._risk_pct
        size_oz     = risk_amount / sl_dist

        trade = SDLiveTrade(
            signal         = signal,
            entry_price    = entry_price,
            sl             = sl,
            tp1            = tp1,
            tp2            = tp2,
            sl_dist        = sl_dist,
            risk_amount    = risk_amount,
            size_full      = size_oz,
            size_remaining = size_oz,
            opened_at      = m1_df.index[entry_bar_idx],
        )
        self._open_trade = trade

        direction_tag   = "LONG"  if is_long else "SHORT"
        direction_emoji = "📈" if is_long else "📉"
        logger.info(
            f"Trade OPENED ({direction_tag})",
            symbol        = self._symbol,
            entry         = round(entry_price, 2),
            sl            = round(sl, 2),
            tp1           = round(tp1, 2),
            tp2           = round(tp2, 2),
            size_oz       = round(size_oz, 4),
            risk_usd      = round(risk_amount, 2),
            zone_score    = round(signal.zone_score, 2),
            wyckoff_score = round(signal.wyckoff_score, 2),
        )
        self._notifier.send(
            f"{direction_emoji} {direction_tag} OPENED {self._symbol} @ {entry_price:.2f}\n"
            f"SL={sl:.2f}  TP1={tp1:.2f}  TP2={tp2:.2f}\n"
            f"Risk={risk_amount:.0f}$ | Zone={signal.zone_score:.1f} Wy={signal.wyckoff_score:.1f}"
        )

    def _check_exit(self, trade: SDLiveTrade, m1_df: pd.DataFrame) -> None:
        """Evaluate the most recent complete M1 bar against the open trade's levels."""
        last_bar  = m1_df.iloc[-2]  # last complete bar
        bar_ts    = m1_df.index[-2]
        bar_lo    = float(last_bar["low"])
        bar_hi    = float(last_bar["high"])

        is_long = trade.signal.direction == "long"

        # Pessimistic conflict: if both SL and TP touched in the same bar → SL wins
        sl_hit  = (bar_lo <= trade.sl)  if is_long else (bar_hi >= trade.sl)
        tp1_hit = (bar_hi >= trade.tp1) if is_long else (bar_lo <= trade.tp1)
        tp2_hit = (bar_hi >= trade.tp2) if is_long else (bar_lo <= trade.tp2)

        # ── SL hit ───────────────────────────────────────────────────────
        if sl_hit and not trade.tp1_hit:
            # Full loss — pessimistic fill: long fills below SL, short fills above SL
            sl_exit = (trade.sl - SPREAD) if is_long else (trade.sl + SPREAD)
            loss_r  = ((sl_exit - trade.entry_price) / trade.sl_dist if is_long
                       else (trade.entry_price - sl_exit) / trade.sl_dist)  # < 0
            pnl_usd = trade.size_full * trade.sl_dist * loss_r - SPREAD * trade.size_full
            self._close_trade(trade, bar_ts, trade.sl, "loss", loss_r, pnl_usd)
            self._daily_losses   += 1
            self._monthly_losses += 1
            return

        if sl_hit and trade.tp1_hit:
            # TP1 already hit; remaining position stopped at BE → scratch outcome
            tp1_portion = TP1_SIZE * TP1_R
            net_r       = tp1_portion  # remaining portion closed at 0R (BE)
            pnl_usd     = (
                TP1_SIZE * trade.size_full * trade.sl_dist * TP1_R
                + (1 - TP1_SIZE) * trade.size_full * 0.0
                - SPREAD * trade.size_full
            )
            self._close_trade(trade, bar_ts, trade.sl, "tp1_only", net_r, pnl_usd)
            return

        # ── TP2 hit (full win) ────────────────────────────────────────────
        if tp2_hit:
            full_r  = TP1_SIZE * TP1_R + (1 - TP1_SIZE) * 1.5
            pnl_usd = (
                TP1_SIZE * trade.size_full * trade.sl_dist * TP1_R
                + (1 - TP1_SIZE) * trade.size_full * trade.sl_dist * 1.5
                - SPREAD * trade.size_full
            )
            self._close_trade(trade, bar_ts, trade.tp2, "full_win", full_r, pnl_usd)
            return

        # ── TP1 hit (partial exit, SL → BE) ──────────────────────────────
        if tp1_hit and not trade.tp1_hit:
            trade.tp1_hit        = True
            trade.sl             = trade.entry_price   # SL → break-even
            trade.size_remaining = trade.size_full * (1 - TP1_SIZE)
            logger.info(
                "TP1 hit — partial exit",
                symbol      = self._symbol,
                tp1         = round(trade.tp1, 2),
                partial_r   = round(TP1_R, 2),
                sl_moved_to = round(trade.entry_price, 2),
                remaining   = round(trade.size_remaining, 4),
            )
            self._notifier.send(
                f"⚡ TP1 HIT {self._symbol} @ {trade.tp1:.2f}\n"
                f"50% closed at +{TP1_R}R  |  SL moved to BE ({trade.entry_price:.2f})"
            )

    def _close_trade(
        self,
        trade:   SDLiveTrade,
        bar_ts:  pd.Timestamp,
        exit_px: float,
        outcome: str,
        pnl_r:   float,
        pnl_usd: float,
    ) -> None:
        trade.closed_at = bar_ts
        trade.outcome   = outcome
        trade.pnl_r     = round(pnl_r, 4)
        trade.pnl_usd   = round(pnl_usd, 2)
        self._equity   += pnl_usd
        self._open_trade = None
        self._closed_trades.append(trade)

        n_closed = len(self._closed_trades)
        n_wins   = sum(1 for t in self._closed_trades if t.outcome in ("full_win", "tp1_only"))
        wr       = n_wins / n_closed * 100 if n_closed else 0.0
        total_r  = sum(t.pnl_r for t in self._closed_trades)

        logger.info(
            f"Trade {outcome.upper()}",
            symbol  = self._symbol,
            exit    = round(exit_px, 2),
            pnl_r   = round(pnl_r, 4),
            pnl_usd = round(pnl_usd, 2),
            equity  = round(self._equity, 2),
            total_r = round(total_r, 2),
            win_rate = round(wr, 1),
        )
        emoji = "✅" if outcome in ("full_win", "tp1_only") else ("🔄" if outcome == "scratch" else "❌")
        self._notifier.send(
            f"{emoji} TRADE {outcome.upper()} {self._symbol} @ {exit_px:.2f}\n"
            f"R={pnl_r:+.2f}  P&L={pnl_usd:+.0f}$\n"
            f"Equity={self._equity:.0f}$  WR={wr:.0f}%  TotalR={total_r:+.2f}"
        )

        self._metrics.record_tick(0.0, self._equity)  # update equity in metrics

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _check_period_reset(self) -> None:
        """Reset daily / monthly loss counters at period boundaries."""
        now   = datetime.now(tz=timezone.utc)
        today = now.date()
        month = now.strftime("%Y-%m")

        if self._today is not None and self._today != today:
            logger.info("Daily loss counter reset", date=str(today))
            self._daily_losses = 0
        self._today = today

        if self._month is not None and self._month != month:
            logger.info("Monthly loss counter reset", month=month)
            self._monthly_losses = 0
        self._month = month

    def _unrealised_pnl(self, current_price: float) -> float:
        """Mark-to-market P&L for the open trade (long or short)."""
        if self._open_trade is None:
            return 0.0
        t = self._open_trade
        is_long    = t.signal.direction == "long"
        price_move = (current_price - t.entry_price) if is_long else (t.entry_price - current_price)
        return price_move * t.size_remaining

    def _log_summary(self) -> None:
        n = len(self._closed_trades)
        if not n:
            logger.info("SD paper trading stopped — no trades closed")
            return
        wins    = sum(1 for t in self._closed_trades if t.outcome in ("full_win", "tp1_only"))
        total_r = sum(t.pnl_r for t in self._closed_trades)
        total_usd = sum(t.pnl_usd for t in self._closed_trades)
        logger.info(
            "SD paper trading summary",
            trades   = n,
            wins     = wins,
            losses   = n - wins,
            win_rate = round(wins / n * 100, 1),
            total_r  = round(total_r, 2),
            total_usd = round(total_usd, 2),
            final_equity = round(self._equity, 2),
        )
        self._metrics.log_summary()
