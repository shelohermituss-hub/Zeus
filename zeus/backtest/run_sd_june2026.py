"""
S&D Strategy — XAUUSD June 2026 Backtest

Loads the HistData M1 CSV for June 2026, resamples it to M15 for zone
detection, runs SDStrategy.run() to generate signals, then simulates each
trade on subsequent M1 bars.

Simulation rules
----------------
- Entry  : open of the bar immediately after the MSS bar (bar_index + 1)
- Exit   : first M1 bar where SL or TP is touched (intrabar check)
  LONG  : SL if bar.low  <= stop_loss,  TP if bar.high >= take_profit
  SHORT : SL if bar.high >= stop_loss,  TP if bar.low  <= take_profit
  Same-bar conflict (both touched) → pessimistic: SL hit first
- Expired: trade still open at end of data → excluded from metrics
- Costs  : spread/commission not modelled (raw signal quality focus)

Position sizing (1R = 1% of running equity)
-------------------------------------------
  risk_amount = equity × risk_pct
  size_oz     = risk_amount / abs(entry − stop_loss)
  pnl         = size_oz × |exit − entry| × direction_sign

Usage
-----
    python -m zeus.backtest.run_sd_june2026
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from zeus.backtest.data_loader import parse_histdata_csv, resample_ohlcv
from zeus.strategy.supply_demand.sd_strategy import SDSignal, SDStrategy
from zeus.strategy.supply_demand.zone_detector import ZoneDetector
from zeus.strategy.supply_demand.wyckoff import WyckoffDetector

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT    = Path(__file__).parent.parent.parent
_M1_FILE = _ROOT / "data" / "historical" / "xauusd" / "m1" / "DAT_MT_XAUUSD_M1_202606.csv"

# ── Parameters ────────────────────────────────────────────────────────────────

INITIAL_BALANCE  = 10_000.0   # USD
RISK_PCT         = 0.01       # 1 % equity at risk per trade
RISK_REWARD      = 3.0
MIN_ZONE_SCORE   = 5.0
MIN_WYCKOFF_SCORE = 4.0
MIN_COMPOSITE    = 5.0
SIGNAL_COOLDOWN  = 30         # M1 bars between signals from the same zone


# ── Trade result ──────────────────────────────────────────────────────────────

@dataclass
class TradeResult:
    signal:       SDSignal
    entry_price:  float
    exit_price:   float
    outcome:      Literal["win", "loss"]
    pnl_r:        float          # multiples of 1R (+3.0 for a 3R win, -1.0 for loss)
    pnl_usd:      float          # dollar P&L (position sized at 1 % equity at entry)
    entry_bar:    int            # M1 bar index of actual entry
    exit_bar:     int            # M1 bar index of exit
    bars_held:    int
    equity_at_entry: float


# ── Simulation ────────────────────────────────────────────────────────────────

def _simulate_trade(
    signal:   SDSignal,
    m1_df:    pd.DataFrame,
    equity:   float,
) -> TradeResult | None:
    """
    Simulate one trade starting from the bar after the MSS bar.

    Returns None if the trade is still open at end of data (expired).
    """
    entry_bar = signal.bar_index + 1
    if entry_bar >= len(m1_df):
        return None

    entry  = float(m1_df.iloc[entry_bar]["open"])
    sl     = signal.stop_loss
    tp     = signal.take_profit
    is_long = signal.direction == "long"

    # Position size: risk $X to buy/sell N oz
    risk_amount = equity * RISK_PCT
    sl_dist     = abs(entry - sl)
    if sl_dist < 1e-6:
        return None
    size_oz = risk_amount / sl_dist

    for bar_idx in range(entry_bar, len(m1_df)):
        bar = m1_df.iloc[bar_idx]
        lo  = float(bar["low"])
        hi  = float(bar["high"])

        sl_hit = (lo <= sl) if is_long else (hi >= sl)
        tp_hit = (hi >= tp) if is_long else (lo <= tp)

        if sl_hit or tp_hit:
            if sl_hit:   # pessimistic: SL wins on conflict
                exit_price = sl
                outcome    = "loss"
                pnl_r      = -1.0
            else:
                exit_price = tp
                outcome    = "win"
                pnl_r      = signal.risk_reward

            direction_sign = 1.0 if is_long else -1.0
            pnl_usd = size_oz * (exit_price - entry) * direction_sign

            return TradeResult(
                signal          = signal,
                entry_price     = round(entry, 5),
                exit_price      = round(exit_price, 5),
                outcome         = outcome,
                pnl_r           = pnl_r,
                pnl_usd         = round(pnl_usd, 2),
                entry_bar       = entry_bar,
                exit_bar        = bar_idx,
                bars_held       = bar_idx - entry_bar,
                equity_at_entry = equity,
            )

    return None   # expired — still open at data end


def simulate_all(
    signals: list[SDSignal],
    m1_df:   pd.DataFrame,
) -> list[TradeResult]:
    """
    Simulate all signals sequentially; equity compounds after each closed trade.

    Concurrent positions are not modelled: each signal is evaluated independently
    (the cooldown in SDStrategy already prevents rapid re-entries from the same zone).
    """
    results: list[TradeResult] = []
    equity = INITIAL_BALANCE

    for sig in signals:
        result = _simulate_trade(sig, m1_df, equity)
        if result is None:
            continue   # expired — skip
        results.append(result)
        equity += result.pnl_usd   # compound

    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_funnel(
    signals:       list[SDSignal],
    n_zone_entries: int,
    n_wy_fires:    int,
    n_mss_match:   int,
) -> None:
    """Print the signal generation funnel for diagnostic clarity."""
    print(f"  Signal funnel")
    print(f"    M1 bars in active zone (score ≥ {MIN_ZONE_SCORE}): {n_zone_entries:>6,}")
    print(f"    Bars with Wyckoff pattern in window:       {n_wy_fires:>6,}")
    print(f"    Bars where MSS == current bar:             {n_mss_match:>6,}")
    print(f"    Signals after score + cooldown filters:    {len(signals):>6,}")
    print()


def _print_report(
    signals: list[SDSignal],
    results: list[TradeResult],
    m1_df:   pd.DataFrame,
    label:   str = "",
) -> None:
    n_signals  = len(signals)
    n_trades   = len(results)
    n_expired  = n_signals - n_trades
    n_wins     = sum(1 for r in results if r.outcome == "win")
    n_losses   = n_trades - n_wins
    win_rate   = n_wins / n_trades * 100 if n_trades else 0.0

    total_r    = sum(r.pnl_r   for r in results)
    total_usd  = sum(r.pnl_usd for r in results)
    final_eq   = INITIAL_BALANCE + total_usd

    avg_bars_win  = (sum(r.bars_held for r in results if r.outcome == "win")  / n_wins  if n_wins  else 0)
    avg_bars_loss = (sum(r.bars_held for r in results if r.outcome == "loss") / n_losses if n_losses else 0)

    # Max drawdown on running equity
    equity_curve = [INITIAL_BALANCE]
    for r in results:
        equity_curve.append(equity_curve[-1] + r.pnl_usd)
    peak = INITIAL_BALANCE
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Avg zone and Wyckoff scores on winning vs losing trades
    avg_zone_win  = (sum(r.signal.zone_score for r in results if r.outcome == "win")  / n_wins  if n_wins  else 0)
    avg_zone_loss = (sum(r.signal.zone_score for r in results if r.outcome == "loss") / n_losses if n_losses else 0)

    data_start = m1_df.index[0].strftime("%Y-%m-%d")
    data_end   = m1_df.index[-1].strftime("%Y-%m-%d")

    header = f"S&D Strategy — XAUUSD June 2026{' (' + label + ')' if label else ''}"
    print("=" * 60)
    print(f"  {header}")
    print("=" * 60)
    print(f"  Data            : {data_start}  →  {data_end}")
    print(f"  M1 bars         : {len(m1_df):,}")
    print()
    print(f"  Signals detected: {n_signals}")
    print(f"  Trades closed   : {n_trades}")
    print(f"  Expired (open)  : {n_expired}")
    print()
    print(f"  Wins            : {n_wins}   ({win_rate:.1f} %)")
    print(f"  Losses          : {n_losses}")
    print(f"  Total R         : {total_r:+.2f}R")
    print()
    print(f"  Initial equity  : ${INITIAL_BALANCE:,.0f}")
    print(f"  Final equity    : ${final_eq:,.2f}")
    print(f"  Net P&L         : ${total_usd:+,.2f}")
    print(f"  Max drawdown    : {max_dd:.1f} %")
    print()
    print(f"  Avg hold (wins) : {avg_bars_win:.0f} M1 bars")
    print(f"  Avg hold (loss) : {avg_bars_loss:.0f} M1 bars")
    print()
    print(f"  Avg zone score  : win={avg_zone_win:.2f}  loss={avg_zone_loss:.2f}")
    print("=" * 60)

    if results:
        print("\n  Trade log:")
        print(f"  {'#':>3}  {'Date':>11}  {'Dir':>5}  {'Entry':>9}  {'Exit':>9}  "
              f"{'R':>6}  {'P&L $':>8}  {'ZScore':>7}  {'WScore':>7}")
        print("  " + "-" * 79)
        for i, r in enumerate(results, 1):
            sig  = r.signal
            ts   = sig.formed_at.strftime("%Y-%m-%d")
            pnl_r_str = f"{r.pnl_r:+.1f}R"
            print(
                f"  {i:>3}  {ts:>11}  {sig.direction:>5}  "
                f"{r.entry_price:>9.3f}  {r.exit_price:>9.3f}  "
                f"{pnl_r_str:>6}  {r.pnl_usd:>+8.2f}  "
                f"{sig.zone_score:>7.2f}  {sig.wyckoff_score:>7.2f}"
            )
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

def _run_variant(
    m15_df:   pd.DataFrame,
    m1_df:    pd.DataFrame,
    label:    str,
    wy_accum_mult: float = 3.0,
    wy_mss_lb:     int   = 10,
    cooldown:      int   = 30,
) -> None:
    """Run one parameter variant and print its report."""
    from zeus.strategy.supply_demand.zone_detector import ZoneDetector as ZD
    from zeus.strategy.supply_demand.wyckoff import WyckoffDetector as WD

    zd = ZD()
    wd = WD(accum_range_mult=wy_accum_mult, mss_lookback=wy_mss_lb)

    # Replicate the inner loop to count funnel steps
    all_zones = zd.detect_zones(m15_df)
    n_zone_entries = n_wy_fires = n_mss_match = 0

    strategy = SDStrategy(
        zone_detector       = ZD(),
        wyckoff_detector    = WD(accum_range_mult=wy_accum_mult, mss_lookback=wy_mss_lb),
        risk_reward         = RISK_REWARD,
        min_zone_score      = MIN_ZONE_SCORE,
        min_wyckoff_score   = MIN_WYCKOFF_SCORE,
        min_composite_score = MIN_COMPOSITE,
        signal_cooldown     = cooldown,
    )

    # Manual funnel count on a fresh zone set
    zones_diag = zd.detect_zones(m15_df)
    wd_diag    = WD(accum_range_mult=wy_accum_mult, mss_lookback=wy_mss_lb)
    for i in range(len(m1_df)):
        ts  = m1_df.index[i]
        bar = m1_df.iloc[i]
        formed = [z for z in zones_diag if z.formed_at <= ts]
        zd.update_zones(formed, bar, ts)
        for z in formed:
            if z.is_mitigated: continue
            if z.score.total < MIN_ZONE_SCORE: continue
            if not z.price_in_zone(float(bar["low"]), float(bar["high"])): continue
            n_zone_entries += 1
            wy = wd_diag.detect(m1_df, z.side, end_idx=i + 1)
            if wy is not None:
                n_wy_fires += 1
                if wy.mss_bar == i:
                    n_mss_match += 1

    signals = strategy.run(m15_df, m1_df)
    results = simulate_all(signals, m1_df)

    _print_funnel(signals, n_zone_entries, n_wy_fires, n_mss_match)
    _print_report(signals, results, m1_df, label=label)


def main() -> None:
    print(f"Loading M1 data from {_M1_FILE.name} …")
    m1_df = parse_histdata_csv(_M1_FILE)
    print(f"  {len(m1_df):,} M1 bars  ({m1_df.index[0]}  →  {m1_df.index[-1]})")

    print("Resampling M1 → M15 …")
    m15_df = resample_ohlcv(m1_df, "15min")
    print(f"  {len(m15_df):,} M15 bars")

    # ── Variant A — default (strict) ─────────────────────────────────────────
    print("\nVariant A — default params (accum_mult=3.0, mss_lb=10, cooldown=30) …")
    _run_variant(m15_df, m1_df,
                 label="default: accum_mult=3.0, mss_lb=10",
                 wy_accum_mult=3.0, wy_mss_lb=10, cooldown=30)

    # ── Variant B — relaxed Wyckoff ──────────────────────────────────────────
    print("\nVariant B — relaxed Wyckoff (accum_mult=5.0, mss_lb=15, cooldown=15) …")
    _run_variant(m15_df, m1_df,
                 label="relaxed: accum_mult=5.0, mss_lb=15",
                 wy_accum_mult=5.0, wy_mss_lb=15, cooldown=15)


if __name__ == "__main__":
    main()
