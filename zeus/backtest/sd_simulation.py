"""
Trade simulation engine for the Supply & Demand strategy.

Priority 3 realism fixes applied here (vs the original run_sd_june2026 loop):

1. TP from actual entry
   The signal's take_profit was computed from wyckoff.mss_close, but the
   real fill is the OPEN of the bar after the MSS bar.  We recalculate TP
   using that actual fill price so the R:R target is measured correctly.

2. Spread cost
   XAUUSD CFD spread during active sessions is roughly 0.30 USD/oz.
   - Long  : effective entry = open + SPREAD_PER_OZ  (you buy at ask)
   - Short : effective entry = open − SPREAD_PER_OZ  (you sell at bid)
   The SL stays absolute (zone-derived); only the entry and TP shift.
   Position sizing uses the adjusted entry so risk stays constant at 1R.

3. Pessimistic conflict
   When both SL and TP are touched within the same M1 bar, SL is assumed
   to have been hit first.  This is the conservative standard for tick-less
   backtests and produces a lower-bound on real-world performance.

4. Expired trades
   If a trade has not closed by the last bar of the dataset it is marked
   as expired and excluded from P&L metrics (open risk unknown).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import pandas as pd

# ── Tradeable Protocol ────────────────────────────────────────────────────────
# F-03: removed SDSignal import; simulate_trade / simulate_all accept any object
# that satisfies this structural interface (HarmonicSignal, ICTSignal, SDSignal …).

@runtime_checkable
class Tradeable(Protocol):
    """Structural interface required by simulate_trade() and simulate_all()."""
    direction:   str
    bar_index:   int
    stop_loss:   float
    risk_reward: float
    formed_at:   pd.Timestamp
    zone_score:  float


# ── Cost model ────────────────────────────────────────────────────────────────

SPREAD_PER_OZ: float = 0.30   # USD per oz — typical XAUUSD CFD spread


# ── Trade result ──────────────────────────────────────────────────────────────

@dataclass
class TradeResult:
    """Full record of a simulated trade."""
    signal:           Tradeable      # any Tradeable signal (SDSignal, HarmonicSignal, …)
    entry_price:      float          # effective fill (after spread + slippage)
    exit_price:       float          # SL or TP fill price
    outcome:          Literal["win", "loss", "scratch"]
    pnl_r:            float          # R multiples (+3.0 win / -1.0 loss / 0.0 scratch)
    pnl_usd:          float          # USD P&L after spread cost
    spread_cost_usd:  float          # spread cost deducted from P&L
    entry_bar:        int            # M1 bar index of fill
    exit_bar:         int            # M1 bar index of exit
    bars_held:        int
    equity_at_entry:  float


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_trade(
    signal:            Tradeable,
    df:                pd.DataFrame,
    equity:            float,
    risk_pct:          float = 0.01,
    spread:            float = SPREAD_PER_OZ,
    use_be:            bool  = False,
    tp1_r:             float = 0.0,
    tp1_size:          float = 0.5,
    slippage_ticks:    float = 0.0,
    tick_df:           pd.DataFrame | None = None,
    quote_to_usd_rate: float = 1.0,    # divide P&L by this for non-USD quote pairs (e.g. 149 for CADJPY)
) -> TradeResult | None:
    """
    Simulate one trade on m1_df starting the bar after signal.bar_index.

    tp1_r: if > 0 and < signal.risk_reward, partial take-profit at tp1_r × SL-dist.
           tp1_size fraction of the position exits at TP1; SL moves to break-even;
           the remainder runs to the full TP (signal.risk_reward × SL-dist).
           Outcome = "win" whenever TP1 is hit (price moved in our direction).
           Outcome = "loss" only if original SL is hit before TP1.
           SL check is always done first (pessimistic convention).

    use_be: if True (and tp1_r == 0), move SL to break-even once +1R is reached.
            Ignored when tp1_r > 0 (partial TP takes over the BE role).

    slippage_ticks: applied at BOTH entry and SL exits (stop orders fill at market).
                    Entry: effective_entry += spread + slippage_ticks (long).
                    SL exit: fill at active_sl − slippage_ticks (long) — pessimistic.
                    TP exits use limit orders, so no additional slippage there.

    tick_df: when provided (M1 data), SL/TP are checked bar-by-bar at M1
             resolution — much more realistic than M15-only checking.
             signal.bar_index still refers to the M15 (df) frame.

    Returns None if the trade expires (still open at end of data).
    """
    # F-08: determine the price DataFrame used for bar-by-bar SL/TP scan
    scan_df = tick_df if tick_df is not None else df

    entry_bar_df = signal.bar_index + 1
    if entry_bar_df >= len(df):
        return None

    # Map M15 entry bar to the tick_df (M1) if provided
    if tick_df is not None:
        m15_entry_ts = df.index[entry_bar_df]
        entry_bar = int(scan_df.index.searchsorted(m15_entry_ts))
        if entry_bar >= len(scan_df):
            return None
    else:
        entry_bar = entry_bar_df

    raw_open = float(scan_df.iloc[entry_bar]["open"])
    is_long  = signal.direction == "long"

    # F-09: slippage applied on top of spread
    total_cost      = spread + slippage_ticks
    effective_entry = raw_open + total_cost if is_long else raw_open - total_cost

    sl = signal.stop_loss
    sl_dist = abs(effective_entry - sl)
    if sl_dist < 1e-6:
        return None

    tp = (effective_entry + signal.risk_reward * sl_dist if is_long
          else effective_entry - signal.risk_reward * sl_dist)

    risk_amount = equity * risk_pct
    size_oz     = risk_amount / sl_dist
    spread_cost = spread * size_oz

    # Classic BE state (used only when tp1_r == 0)
    be_active  = False
    be_trigger = (effective_entry + sl_dist if is_long else effective_entry - sl_dist)
    active_sl  = sl

    # Partial TP1 state
    use_tp1  = tp1_r > 0.0 and tp1_r < signal.risk_reward
    tp1      = ((effective_entry + tp1_r * sl_dist) if is_long
                else (effective_entry - tp1_r * sl_dist)) if use_tp1 else 0.0
    tp1_hit  = False

    def _exit(bar_idx: int, exit_px: float, outcome: str, pnl_r: float) -> TradeResult:
        raw_pnl = size_oz * sl_dist * pnl_r
        pnl_usd = (raw_pnl - spread_cost) / quote_to_usd_rate
        sc_usd  = spread_cost / quote_to_usd_rate
        return TradeResult(
            signal          = signal,
            entry_price     = round(effective_entry, 5),
            exit_price      = round(exit_px, 5),
            outcome         = outcome,
            pnl_r           = round(pnl_r, 4),
            pnl_usd         = round(pnl_usd, 2),
            spread_cost_usd = round(sc_usd, 2),
            entry_bar       = entry_bar,
            exit_bar        = bar_idx,
            bars_held       = bar_idx - entry_bar,
            equity_at_entry = equity,
        )

    for bar_idx in range(entry_bar, len(scan_df)):
        bar = scan_df.iloc[bar_idx]
        lo  = float(bar["low"])
        hi  = float(bar["high"])

        # Classic BE activation (only when partial TP not in use)
        if use_be and not use_tp1 and not be_active:
            if (hi >= be_trigger if is_long else lo <= be_trigger):
                be_active = True
                active_sl = effective_entry

        # ── SL check (pessimistic: always evaluated first) ────────────────────
        sl_hit = (lo <= active_sl) if is_long else (hi >= active_sl)
        if sl_hit:
            if use_tp1 and tp1_hit:
                # TP1 was previously hit; remaining position exits at BE (0 P&L)
                # → count as "win" since primary target was reached
                return _exit(bar_idx, active_sl, "win", tp1_size * tp1_r)
            elif be_active:
                return _exit(bar_idx, active_sl, "scratch", 0.0)
            else:
                # F-09: slippage applied on SL exit (stop orders get market-order fill)
                sl_exit    = (active_sl - slippage_ticks) if is_long else (active_sl + slippage_ticks)
                # Direction-aware: long loses when price falls (sl_exit < entry → negative);
                # short loses when price rises (sl_exit > entry → must negate).
                loss_pnl_r = (sl_exit - effective_entry) / sl_dist if is_long else (effective_entry - sl_exit) / sl_dist
                return _exit(bar_idx, sl_exit, "loss", loss_pnl_r)

        # ── TP1 check (partial exit, SL moves to BE for remainder) ───────────
        if use_tp1 and not tp1_hit:
            if (hi >= tp1 if is_long else lo <= tp1):
                tp1_hit   = True
                active_sl = effective_entry  # SL → BE from next evaluation

        # ── TP2 / full TP check ───────────────────────────────────────────────
        tp_hit = (hi >= tp) if is_long else (lo <= tp)
        if tp_hit:
            if use_tp1 and tp1_hit:
                pnl_r = tp1_size * tp1_r + (1.0 - tp1_size) * signal.risk_reward
            else:
                pnl_r = signal.risk_reward
            return _exit(bar_idx, tp, "win", pnl_r)

    return None


def simulate_all(
    signals:            list[Tradeable],
    df:                 pd.DataFrame,
    risk_pct:           float = 0.01,
    spread:             float = SPREAD_PER_OZ,
    max_daily_losses:   int   = 0,
    max_monthly_losses: int   = 0,
    use_be:             bool  = False,
    tp1_r:              float = 0.0,
    tp1_size:           float = 0.5,
    initial_equity:     float = 10_000.0,
    slippage_ticks:     float = 0.0,
    tick_df:            pd.DataFrame | None = None,
    quote_to_usd_rate:  float = 1.0,        # divide P&L by this for non-USD quote pairs
) -> tuple[list[TradeResult], int]:
    """
    Simulate all signals sequentially with compounding equity.

    Returns (results, n_expired).  Expired trades are counted but excluded
    from the result list because their P&L is unknown.

    max_daily_losses:   stop adding signals on a calendar day after N losses.
    max_monthly_losses: stop adding signals in a calendar month after N losses.
    use_be:             enable break-even management at +1R (see simulate_trade).
    """
    results:   list[TradeResult] = []
    n_expired: int               = 0
    equity = initial_equity   # F-02: use caller-specified initial equity

    _day_losses:   dict[str, int] = {}
    _month_losses: dict[str, int] = {}

    for sig in signals:
        day_key   = sig.formed_at.strftime("%Y-%m-%d")
        month_key = sig.formed_at.strftime("%Y-%m")

        if max_monthly_losses > 0 and _month_losses.get(month_key, 0) >= max_monthly_losses:
            n_expired += 1
            continue

        if max_daily_losses > 0 and _day_losses.get(day_key, 0) >= max_daily_losses:
            n_expired += 1
            continue

        result = simulate_trade(
            sig, df, equity, risk_pct, spread, use_be, tp1_r, tp1_size,
            slippage_ticks=slippage_ticks, tick_df=tick_df,
            quote_to_usd_rate=quote_to_usd_rate,
        )
        if result is None:
            n_expired += 1
            continue
        results.append(result)
        equity += result.pnl_usd

        if result.outcome == "loss":
            if max_daily_losses > 0:
                _day_losses[day_key] = _day_losses.get(day_key, 0) + 1
            if max_monthly_losses > 0:
                _month_losses[month_key] = _month_losses.get(month_key, 0) + 1

    return results, n_expired


# ── Reporting ─────────────────────────────────────────────────────────────────

def compute_metrics(
    results:         list[TradeResult],
    initial_equity:  float,
    n_signals:       int,
    n_expired:       int,
) -> dict:
    """Return a dict of standard backtest metrics."""
    n_trades   = len(results)
    n_wins     = sum(1 for r in results if r.outcome == "win")
    n_losses   = sum(1 for r in results if r.outcome == "loss")
    n_scratches = n_trades - n_wins - n_losses
    decided    = n_wins + n_losses   # scratches excluded from WR
    win_rate   = n_wins / decided * 100 if decided else 0.0

    total_r   = sum(r.pnl_r   for r in results)
    total_usd = sum(r.pnl_usd for r in results)
    total_spread = sum(r.spread_cost_usd for r in results)
    final_eq  = initial_equity + total_usd

    avg_bars_win  = (sum(r.bars_held for r in results if r.outcome == "win")  / n_wins
                     if n_wins  else 0)
    avg_bars_loss = (sum(r.bars_held for r in results if r.outcome == "loss") / n_losses
                     if n_losses else 0)

    equity_curve = [initial_equity]
    for r in results:
        equity_curve.append(equity_curve[-1] + r.pnl_usd)
    peak   = initial_equity
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd

    avg_zone_win  = (sum(r.signal.zone_score for r in results if r.outcome == "win")
                     / n_wins  if n_wins  else 0.0)
    avg_zone_loss = (sum(r.signal.zone_score for r in results if r.outcome == "loss")
                     / n_losses if n_losses else 0.0)

    return dict(
        n_signals    = n_signals,
        n_trades     = n_trades,
        n_expired    = n_expired,
        n_wins       = n_wins,
        n_losses     = n_losses,
        n_scratches  = n_scratches,
        win_rate     = win_rate,
        total_r      = total_r,
        total_usd    = total_usd,
        total_spread = total_spread,
        final_eq     = final_eq,
        max_dd       = max_dd,
        avg_bars_win  = avg_bars_win,
        avg_bars_loss = avg_bars_loss,
        avg_zone_win  = avg_zone_win,
        avg_zone_loss = avg_zone_loss,
    )


def print_report(
    results:        list[TradeResult],
    initial_equity: float,
    n_signals:      int,
    n_expired:      int,
    df:             pd.DataFrame,
    label:          str = "",
) -> None:
    """Print a full backtest summary including trade log."""
    m = compute_metrics(results, initial_equity, n_signals, n_expired)

    data_start = df.index[0].strftime("%Y-%m-%d")
    data_end   = df.index[-1].strftime("%Y-%m-%d")
    header = f"S&D Strategy — XAUUSD{' (' + label + ')' if label else ''}"

    print("=" * 62)
    print(f"  {header}")
    print("=" * 62)
    print(f"  Data            : {data_start}  →  {data_end}")
    print(f"  Bars            : {len(df):,}")
    print()
    print(f"  Signals         : {m['n_signals']}")
    print(f"  Trades closed   : {m['n_trades']}")
    print(f"  Expired (open)  : {m['n_expired']}")
    print()
    print(f"  Wins            : {m['n_wins']}   ({m['win_rate']:.1f} %)")
    print(f"  Losses          : {m['n_losses']}")
    print(f"  Total R         : {m['total_r']:+.2f}R")
    print()
    print(f"  Initial equity  : ${initial_equity:,.0f}")
    print(f"  Final equity    : ${m['final_eq']:,.2f}")
    print(f"  Net P&L         : ${m['total_usd']:+,.2f}")
    print(f"  Spread cost     : ${m['total_spread']:,.2f}")
    print(f"  Max drawdown    : {m['max_dd']:.1f} %")
    print()
    print(f"  Avg hold (wins) : {m['avg_bars_win']:.0f} M1 bars")
    print(f"  Avg hold (loss) : {m['avg_bars_loss']:.0f} M1 bars")
    print()
    print(f"  Avg zone score  : win={m['avg_zone_win']:.2f}  loss={m['avg_zone_loss']:.2f}")
    print("=" * 62)

    if results:
        print("\n  Trade log:")
        print(f"  {'#':>3}  {'Date':>11}  {'Dir':>5}  {'Entry':>9}  {'Exit':>9}  "
              f"{'R':>6}  {'P&L $':>8}  {'Sprd':>6}  {'ZSc':>5}  {'WSc':>5}")
        print("  " + "-" * 83)
        for i, r in enumerate(results, 1):
            sig = r.signal
            ts  = sig.formed_at.strftime("%Y-%m-%d")
            wyckoff = getattr(sig, "wyckoff_score", None)
            wyckoff_col = f"  {wyckoff:>5.2f}" if wyckoff is not None else ""
            print(
                f"  {i:>3}  {ts:>11}  {sig.direction:>5}  "
                f"{r.entry_price:>9.3f}  {r.exit_price:>9.3f}  "
                f"{r.pnl_r:>+5.1f}R  {r.pnl_usd:>+8.2f}  "
                f"{r.spread_cost_usd:>6.2f}  "
                f"{sig.zone_score:>5.2f}{wyckoff_col}"
            )
        print()


def print_monthly_breakdown(
    results:        list[TradeResult],
    initial_equity: float,
) -> None:
    """Print a one-line summary per calendar month."""
    from collections import defaultdict

    by_month: dict[str, list[TradeResult]] = defaultdict(list)
    for r in results:
        key = r.signal.formed_at.strftime("%Y-%m")
        by_month[key].append(r)

    if not by_month:
        return

    print("\n  Monthly breakdown:")
    print(f"  {'Month':>8}  {'Sig':>4}  {'W':>3}  {'L':>3}  "
          f"{'WR%':>5}  {'R':>7}  {'P&L $':>9}  {'MaxDD%':>7}")
    print("  " + "-" * 62)

    running_equity = initial_equity
    for month in sorted(by_month):
        month_results = by_month[month]
        mw  = sum(1 for r in month_results if r.outcome == "win")
        ml  = len(month_results) - mw
        mr  = sum(r.pnl_r   for r in month_results)
        mpnl = sum(r.pnl_usd for r in month_results)
        mwr = mw / len(month_results) * 100 if month_results else 0.0

        curve = [running_equity]
        for r in month_results:
            curve.append(curve[-1] + r.pnl_usd)
        peak = running_equity
        mdd  = 0.0
        for eq in curve:
            if eq > peak: peak = eq
            dd = (peak - eq) / peak * 100
            if dd > mdd: mdd = dd

        print(f"  {month:>8}  {len(month_results):>4}  {mw:>3}  {ml:>3}  "
              f"{mwr:>5.1f}  {mr:>+6.2f}R  {mpnl:>+9.2f}  {mdd:>6.1f}%")
        running_equity += mpnl
    print()
