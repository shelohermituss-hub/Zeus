"""
Paper trading engine for the Universal-D Supply & Demand strategy — Forex multi-pair.

Configuration (Universal-D — validated OOS on 2024 data):
    Bidirectional: wy_long≥7.0, wy_short≥5.9
    zone≥4.0, R:R=1.25, TP1@0.75R(50%)→TP2@1.25R
    Session 07-17h UTC (London only — all 4 pairs are GBP/EUR cluster)
    min_sl_pips=5, min_spring_sweep_pct=0.10, ema_atr_tolerance=0.5

    Execution:
        TP1 at 0.75R (50% partial exit), remaining 50% runs to TP2 at 1.25R.
        SL → break-even after TP1 hit.
        Max 1 loss per pair per calendar day.
        Max 4 losses per pair per calendar month.

    OOS backtest validation (2024 full year):
        GBPUSD: 83.3% WR · DD 2.0%   PASS
        EURUSD: 81.8% WR · DD 1.9%   PASS
        GBPAUD: 77.5% WR · DD 3.5%   PASS
        EURNZD: 78.6% WR · DD 3.2%   PASS

Loop design:
    - Polls MT5 every `poll_interval` seconds.
    - On each tick: iterates all symbols, fetches M1 independently per symbol.
    - Shared equity — all 4 pairs compound from the same balance.
    - Max 1 open trade per symbol at a time (up to 4 concurrent trades total).
    - Bidirectional: long on demand zones, short on supply zones.
    - Spread applied on entry (ask for longs, bid for shorts); pessimistic SL fill.
    - quote_to_usd_rate applied for GBPAUD (AUD-quoted) and EURNZD (NZD-quoted).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
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

# ── Universal-D champion params (frozen from OOS validation) ──────────────────
_UNIVERSAL_D = dict(
    risk_reward             = 1.25,
    min_zone_score          = 4.0,
    min_wyckoff_score       = 7.0,    # longs: strict
    min_wyckoff_score_short = 5.9,    # shorts: normal
    min_composite_score     = 4.0,
    signal_cooldown         = 10,
    use_trend_filter        = True,
    trend_slope_lookback    = 3,
    use_price_above_ema     = True,
    ema_atr_tolerance       = 0.5,    # allow price within 0.5×ATR of EMA50
    use_session_filter      = True,
    max_signals_per_day     = 6,
    use_adx_filter          = False,
    use_h4_trend_filter     = False,
    use_rsi_filter          = False,
    min_sl_pips             = 5,      # reject signals with SL < 5 pips
    pip_size                = 0.0001,
    min_score_product       = 0.0,
)

_WYCKOFF_PARAMS = dict(
    lookback             = 200,
    max_accum_bars       = 20,
    accum_range_mult     = 6.0,
    mss_lookback         = 60,
    min_spring_sweep_pct = 0.10,
    min_mss_strength_pct = 0.03,
)

# ── Partial exit config ────────────────────────────────────────────────────────
TP1_R:    float = 0.75    # first partial at 0.75R
TP2_R:    float = 1.25    # full win at 1.25R (= Universal-D R:R)
TP1_SIZE: float = 0.50    # fraction closed at TP1

# Maximum M1 bars to fetch per symbol
M1_LIMIT: int = 3_000

# ── Instrument config for the OOS-validated paper cluster ─────────────────────

PAPER_CLUSTER: list[str] = ["GBPUSD", "EURUSD", "GBPAUD", "EURNZD"]

# Spread in price units (from INSTRUMENTS registry in OOS validation)
SPREADS: dict[str, float] = {
    "GBPUSD": 0.0001,
    "EURUSD": 0.0001,
    "GBPAUD": 0.0002,
    "EURNZD": 0.0002,
}

# R7: GBP/EUR cluster → London only (07-17h UTC)
SESSION_OVERRIDE: dict[str, tuple[int, int]] = {
    "GBPUSD": (7, 17),
    "EURUSD": (7, 17),
    "GBPAUD": (7, 17),
    "EURNZD": (7, 17),
}

# R5: Quote-to-USD conversion (used in position sizing and P&L reporting)
# GBPUSD, EURUSD → USD-quoted → 1.0
# GBPAUD → P&L in AUD → divide by ~1.52  (AUDUSD ≈ 0.66)
# EURNZD → P&L in NZD → divide by ~1.62  (NZDUSD ≈ 0.62)
QUOTE_TO_USD: dict[str, float] = {
    "GBPUSD": 1.0,
    "EURUSD": 1.0,
    "GBPAUD": 1.52,
    "EURNZD": 1.62,
}


# ── Open trade tracker ────────────────────────────────────────────────────────

@dataclass
class ForexLiveTrade:
    """State of a single open Forex paper trade (long or short)."""
    symbol:            str
    direction:         str             # "long" | "short"
    signal:            SDSignal
    entry_price:       float           # effective fill (raw_open ± spread)
    sl:                float           # current SL (moves to entry after TP1)
    tp1:               float           # TP1 level (0.75R in trade direction)
    tp2:               float           # TP2 level (1.25R in trade direction)
    sl_dist:           float           # |entry - original SL|  (= 1R)
    risk_amount:       float           # equity × risk_pct at entry
    size_full:         float           # total position in units
    size_remaining:    float           # units remaining after TP1 partial exit
    spread:            float           # spread in price units for this symbol
    quote_to_usd_rate: float           # conversion rate from quote ccy to USD
    tp1_hit:           bool = False
    opened_at:         Optional[pd.Timestamp] = None
    closed_at:         Optional[pd.Timestamp] = None
    outcome:           str = "open"    # "open" | "tp1_only" | "full_win" | "loss"
    pnl_r:             float = 0.0
    pnl_usd:           float = 0.0

    @property
    def is_long(self) -> bool:
        return self.direction == "long"

    @property
    def is_open(self) -> bool:
        return self.outcome == "open"


# ── Engine ────────────────────────────────────────────────────────────────────

class ForexPaperEngine:
    """
    Multi-symbol Forex paper trading loop for the Universal-D strategy.

    Args:
        connector:       Live market data source (MT5 connector).
                         Only fetch_ohlcv() is called — no real orders sent.
        symbols:         List of Forex symbols to trade (default: PAPER_CLUSTER).
        initial_balance: Starting paper account balance.
        risk_pct:        Fraction of equity risked per trade (default 0.5%).
        poll_interval:   Seconds between ticks (default 60 = every M1 close).
        notifier:        Optional Telegram notifier for trade alerts.
    """

    def __init__(
        self,
        connector:       ExchangeConnector,
        symbols:         Optional[list[str]]       = None,
        initial_balance: float                     = 10_000.0,
        risk_pct:        float                     = 0.005,
        poll_interval:   float                     = 60.0,
        notifier:        Optional[TelegramNotifier] = None,
    ) -> None:
        self._connector      = connector
        self._symbols        = symbols or PAPER_CLUSTER
        self._equity         = initial_balance
        self._initial_equity = initial_balance
        self._risk_pct       = risk_pct
        self._poll_interval  = poll_interval
        self._notifier       = notifier or TelegramNotifier(bot_token="", chat_id="")
        self._running        = False

        # Per-symbol strategy (independent ZoneDetector + WyckoffDetector per symbol)
        self._strategies: dict[str, SDStrategy] = {}
        for sym in self._symbols:
            sess = SESSION_OVERRIDE.get(sym, (7, 21))
            self._strategies[sym] = SDStrategy(
                zone_detector    = ZoneDetector(),
                wyckoff_detector = WyckoffDetector(**_WYCKOFF_PARAMS),
                **{**_UNIVERSAL_D, "session_start_utc": sess[0], "session_end_utc": sess[1]},
            )

        # Per-symbol trade slot (None = no open trade)
        self._open_trades: dict[str, Optional[ForexLiveTrade]] = {s: None for s in self._symbols}

        # Per-symbol signal deduplication keyed by formed_at timestamp
        self._seen_signals: dict[str, set[pd.Timestamp]] = {s: set() for s in self._symbols}

        # Per-symbol closed trade log
        self._closed_trades: dict[str, list[ForexLiveTrade]] = {s: [] for s in self._symbols}

        # Per-symbol loss counters (reset by calendar day / month)
        self._daily_losses:   dict[str, int] = {s: 0 for s in self._symbols}
        self._monthly_losses: dict[str, int] = {s: 0 for s in self._symbols}
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
            "Forex paper trading started",
            symbols  = self._symbols,
            equity   = self._equity,
            risk_pct = self._risk_pct,
            config   = "Universal-D: bidirectional, TP1@0.75R(50%)→TP2@1.25R, London 07-17h",
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
            logger.info("Forex paper trading stopped by user")
        finally:
            self._running = False
            self._log_summary()

    # ── Tick ─────────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        t0 = time.monotonic()
        self._check_period_reset()

        total_open_pnl = 0.0
        for sym in self._symbols:
            try:
                total_open_pnl += self._tick_symbol(sym)
            except Exception as exc:
                self._metrics.record_api_error()
                logger.error("Tick error", symbol=sym, error=str(exc))

        equity  = self._equity + total_open_pnl
        latency = (time.monotonic() - t0) * 1000
        self._metrics.record_tick(latency, equity)

        open_count = sum(1 for t in self._open_trades.values() if t is not None)
        logger.info(
            "Tick complete",
            equity      = round(equity, 2),
            open_trades = open_count,
            latency_ms  = round(latency, 1),
        )

    def _tick_symbol(self, sym: str) -> float:
        """Process one symbol tick. Returns unrealized P&L (USD) for this symbol."""
        spread       = SPREADS.get(sym, 0.0001)
        quote_to_usd = QUOTE_TO_USD.get(sym, 1.0)
        strategy     = self._strategies[sym]

        # ── Fetch M1 data ─────────────────────────────────────────────────
        m1_df = self._connector.fetch_ohlcv(sym, "1m", M1_LIMIT)
        if m1_df is None or len(m1_df) < 202:
            logger.warning("Insufficient M1 data — tick skipped", symbol=sym)
            return 0.0

        current_price = float(m1_df["close"].iloc[-1])

        # ── Exit check on open trade ──────────────────────────────────────
        if self._open_trades[sym] is not None:
            self._check_exit(self._open_trades[sym], m1_df)

        # ── Resample → M15 for zone detection ─────────────────────────────
        m15_df = resample_ohlcv(m1_df, "15min")
        if len(m15_df) < 10:
            logger.warning("Insufficient M15 bars — tick skipped", symbol=sym)
            return self._unrealised_pnl(sym, current_price)

        # ── Run strategy ──────────────────────────────────────────────────
        signals = strategy.run(m15_df, m1_df)

        # ── Detect new signal on the last complete M1 bar ─────────────────
        if (self._open_trades[sym] is None
                and self._daily_losses[sym] < 1
                and self._monthly_losses[sym] < 4
                and signals):
            last_complete_ts = m1_df.index[-2]
            new_sig = next(
                (s for s in reversed(signals)
                 if s.formed_at == last_complete_ts
                 and s.formed_at not in self._seen_signals[sym]),
                None,
            )
            if new_sig is not None:
                self._seen_signals[sym].add(new_sig.formed_at)
                self._enter_trade(sym, new_sig, m1_df, spread, quote_to_usd)

        return self._unrealised_pnl(sym, current_price)

    # ── Trade management ──────────────────────────────────────────────────────

    def _enter_trade(
        self,
        sym:          str,
        signal:       SDSignal,
        m1_df:        pd.DataFrame,
        spread:       float,
        quote_to_usd: float,
    ) -> None:
        """Open a new paper trade (long or short) at the next bar's open."""
        is_long       = signal.direction == "long"
        entry_bar_idx = signal.bar_index + 1
        if entry_bar_idx >= len(m1_df):
            logger.debug("Signal bar too recent — entry skipped", formed_at=str(signal.formed_at))
            return

        raw_open    = float(m1_df.iloc[entry_bar_idx]["open"])
        # Long: buy at ask; Short: sell at bid
        entry_price = raw_open + spread if is_long else raw_open - spread

        sl      = signal.stop_loss
        sl_dist = abs(entry_price - sl)
        if sl_dist < 1e-8:
            return

        # TP levels in trade direction
        if is_long:
            tp1 = entry_price + TP1_R * sl_dist
            tp2 = entry_price + TP2_R * sl_dist
        else:
            tp1 = entry_price - TP1_R * sl_dist
            tp2 = entry_price - TP2_R * sl_dist

        # size = risk_amount / (sl_dist / quote_to_usd) = risk_amount * quote_to_usd / sl_dist
        risk_amount = self._equity * self._risk_pct
        size_units  = risk_amount * quote_to_usd / sl_dist

        trade = ForexLiveTrade(
            symbol            = sym,
            direction         = signal.direction,
            signal            = signal,
            entry_price       = entry_price,
            sl                = sl,
            tp1               = tp1,
            tp2               = tp2,
            sl_dist           = sl_dist,
            risk_amount       = risk_amount,
            size_full         = size_units,
            size_remaining    = size_units,
            spread            = spread,
            quote_to_usd_rate = quote_to_usd,
            opened_at         = m1_df.index[entry_bar_idx],
        )
        self._open_trades[sym] = trade

        dir_arrow = "↑ LONG" if is_long else "↓ SHORT"
        logger.info(
            f"Trade OPENED — {dir_arrow}",
            symbol        = sym,
            entry         = round(entry_price, 5),
            sl            = round(sl, 5),
            tp1           = round(tp1, 5),
            tp2           = round(tp2, 5),
            size_units    = round(size_units, 0),
            risk_usd      = round(risk_amount, 2),
            zone_score    = round(signal.zone_score, 2),
            wyckoff_score = round(signal.wyckoff_score, 2),
        )
        emoji = "📈" if is_long else "📉"
        self._notifier.send(
            f"{emoji} {signal.direction.upper()} OPENED {sym} @ {entry_price:.5f}\n"
            f"SL={sl:.5f}  TP1={tp1:.5f}  TP2={tp2:.5f}\n"
            f"Risk={risk_amount:.0f}$  |  Zone={signal.zone_score:.1f}  Wy={signal.wyckoff_score:.1f}"
        )

    def _check_exit(self, trade: ForexLiveTrade, m1_df: pd.DataFrame) -> None:
        """Evaluate the most recent complete M1 bar against the open trade's levels."""
        last_bar = m1_df.iloc[-2]
        bar_ts   = m1_df.index[-2]
        bar_lo   = float(last_bar["low"])
        bar_hi   = float(last_bar["high"])
        is_long  = trade.is_long
        spread   = trade.spread

        # Direction-aware hit detection — pessimistic: if both SL and TP in same bar, SL wins
        if is_long:
            sl_hit  = bar_lo <= trade.sl
            tp1_hit = bar_hi >= trade.tp1
            tp2_hit = bar_hi >= trade.tp2
        else:
            sl_hit  = bar_hi >= trade.sl
            tp1_hit = bar_lo <= trade.tp1
            tp2_hit = bar_lo <= trade.tp2

        # ── SL hit (full loss, TP1 not reached) ──────────────────────────
        if sl_hit and not trade.tp1_hit:
            if is_long:
                sl_exit = trade.sl - spread   # pessimistic fill below SL
                loss_r  = (sl_exit - trade.entry_price) / trade.sl_dist
            else:
                sl_exit = trade.sl + spread   # pessimistic fill above SL
                loss_r  = (trade.entry_price - sl_exit) / trade.sl_dist
            pnl_usd = (
                trade.size_full * trade.sl_dist * loss_r - spread * trade.size_full
            ) / trade.quote_to_usd_rate
            self._close_trade(trade, bar_ts, sl_exit, "loss", loss_r, pnl_usd)
            self._daily_losses[trade.symbol]   += 1
            self._monthly_losses[trade.symbol] += 1
            return

        # ── SL hit after TP1 — remaining position exits at break-even ────
        if sl_hit and trade.tp1_hit:
            net_r   = TP1_SIZE * TP1_R   # TP1 portion locked; remainder is 0R at BE
            pnl_usd = (
                TP1_SIZE * trade.size_full * trade.sl_dist * TP1_R
                - spread * trade.size_full
            ) / trade.quote_to_usd_rate
            self._close_trade(trade, bar_ts, trade.entry_price, "tp1_only", net_r, pnl_usd)
            return

        # ── TP2 hit (full win) ────────────────────────────────────────────
        if tp2_hit:
            full_r  = TP1_SIZE * TP1_R + (1 - TP1_SIZE) * TP2_R
            pnl_usd = (
                TP1_SIZE * trade.size_full * trade.sl_dist * TP1_R
                + (1 - TP1_SIZE) * trade.size_full * trade.sl_dist * TP2_R
                - spread * trade.size_full
            ) / trade.quote_to_usd_rate
            self._close_trade(trade, bar_ts, trade.tp2, "full_win", full_r, pnl_usd)
            return

        # ── TP1 hit (partial exit, SL → break-even) ──────────────────────
        if tp1_hit and not trade.tp1_hit:
            trade.tp1_hit        = True
            trade.sl             = trade.entry_price   # SL → BE
            trade.size_remaining = trade.size_full * (1 - TP1_SIZE)
            logger.info(
                "TP1 hit — partial exit",
                symbol      = trade.symbol,
                direction   = trade.direction,
                tp1         = round(trade.tp1, 5),
                partial_r   = TP1_R,
                sl_moved_to = round(trade.entry_price, 5),
                remaining   = round(trade.size_remaining, 0),
            )
            emoji = "⚡📈" if is_long else "⚡📉"
            self._notifier.send(
                f"{emoji} TP1 HIT {trade.symbol} {trade.direction.upper()} @ {trade.tp1:.5f}\n"
                f"50% closed at +{TP1_R}R  |  SL moved to BE ({trade.entry_price:.5f})"
            )

    def _close_trade(
        self,
        trade:   ForexLiveTrade,
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
        self._open_trades[trade.symbol] = None
        self._closed_trades[trade.symbol].append(trade)

        all_closed = [t for trades in self._closed_trades.values() for t in trades]
        n_closed   = len(all_closed)
        n_wins     = sum(1 for t in all_closed if t.outcome in ("full_win", "tp1_only"))
        wr         = n_wins / n_closed * 100 if n_closed else 0.0
        total_r    = sum(t.pnl_r for t in all_closed)

        logger.info(
            f"Trade {outcome.upper()}",
            symbol    = trade.symbol,
            direction = trade.direction,
            exit      = round(exit_px, 5),
            pnl_r     = round(pnl_r, 4),
            pnl_usd   = round(pnl_usd, 2),
            equity    = round(self._equity, 2),
            total_r   = round(total_r, 2),
            win_rate  = round(wr, 1),
        )
        is_win = outcome in ("full_win", "tp1_only")
        emoji  = "✅" if is_win else ("🔄" if outcome == "scratch" else "❌")
        self._notifier.send(
            f"{emoji} TRADE {outcome.upper()} {trade.symbol} {trade.direction.upper()} @ {exit_px:.5f}\n"
            f"R={pnl_r:+.2f}  P&L={pnl_usd:+.0f}$\n"
            f"Equity={self._equity:.0f}$  WR={wr:.0f}%  TotalR={total_r:+.2f}"
        )
        self._metrics.record_tick(0.0, self._equity)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _check_period_reset(self) -> None:
        """Reset daily / monthly loss counters at UTC midnight / month boundary."""
        now   = datetime.now(tz=timezone.utc)
        today = now.date()
        month = now.strftime("%Y-%m")

        if self._today is not None and self._today != today:
            logger.info("Daily loss counters reset", date=str(today))
            for sym in self._symbols:
                self._daily_losses[sym] = 0
        self._today = today

        if self._month is not None and self._month != month:
            logger.info("Monthly loss counters reset", month=month)
            for sym in self._symbols:
                self._monthly_losses[sym] = 0
        self._month = month

    def _unrealised_pnl(self, sym: str, current_price: float) -> float:
        """Mark-to-market P&L (USD) for the open trade on a symbol."""
        trade = self._open_trades.get(sym)
        if trade is None:
            return 0.0
        price_move = (
            current_price - trade.entry_price if trade.is_long
            else trade.entry_price - current_price
        )
        return price_move * trade.size_remaining / trade.quote_to_usd_rate

    def _log_summary(self) -> None:
        all_closed = [t for trades in self._closed_trades.values() for t in trades]
        n = len(all_closed)
        if not n:
            logger.info("Forex paper trading stopped — no trades closed")
            return
        wins      = sum(1 for t in all_closed if t.outcome in ("full_win", "tp1_only"))
        total_r   = sum(t.pnl_r for t in all_closed)
        total_usd = sum(t.pnl_usd for t in all_closed)
        logger.info(
            "Forex paper trading summary",
            trades       = n,
            wins         = wins,
            losses       = n - wins,
            win_rate     = round(wins / n * 100, 1),
            total_r      = round(total_r, 2),
            total_usd    = round(total_usd, 2),
            final_equity = round(self._equity, 2),
        )
        for sym in self._symbols:
            sym_trades = self._closed_trades[sym]
            if not sym_trades:
                continue
            sym_wins = sum(1 for t in sym_trades if t.outcome in ("full_win", "tp1_only"))
            sym_r    = sum(t.pnl_r for t in sym_trades)
            logger.info(
                f"  {sym} summary",
                trades   = len(sym_trades),
                win_rate = round(sym_wins / len(sym_trades) * 100, 1),
                total_r  = round(sym_r, 2),
            )
        self._metrics.log_summary()
