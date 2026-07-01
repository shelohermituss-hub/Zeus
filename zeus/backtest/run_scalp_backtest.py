"""
ScalpSMCStrategy — XAUUSD 2026 H1 Scalp Backtest (Jan–Jun 2026)

Data source : HistData.com MT4 M1 files in data/historical/xauusd/m1/
Cascade     : 1H (HTF zones) → 15M (MSS gate) → 1M (LTF entry)

Strategy differences vs swing (run_2026_backtest.py)
------------------------------------------------------
  HTF      : 1H instead of 4H   → ~4× more zone events per day
  MSS gate : 15M instead of 1H  → faster structure confirmation
  LTF bars : 1M instead of 5M   → tighter entry / smaller SL
  SL       : 6 pips default, 10 pips max  (was 20 / 30)
  Weekly   : disabled            → irrelevant at intraday scale
  CHoCH    : disabled            → strict candle check cuts too many 1M setups
  Signals  : up to 5/day, 2/session (was 2/day, 1/session)
  Exits    : 1.5R BE → 2.5R 60% → 4R 100%  (was 1R/3R/5R/10R)

Risk
----
  1 % of equity at risk per trade (same as swing strategy)
  1 concurrent position max
  Slippage ≈ 0.6 pip spread modelled as 0.02 %

Usage
-----
    python -m zeus.backtest.run_scalp_backtest
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from zeus.backtest.advanced_engine import AdvancedBacktestEngine
from zeus.backtest.data_loader import build_timeframes, load_m1_directory
from zeus.backtest.partial_close import PartialCloseConfig
from zeus.strategy.scalp_strategy import ScalpSMCStrategy
from zeus.utils.logger import setup_logger

# ── Paths ────────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).parent.parent.parent
DATA_DIR    = _ROOT / "data" / "historical" / "xauusd" / "m1"
RESULTS_DIR = _ROOT / "data" / "results"

# ── Parameters ────────────────────────────────────────────────────────────────
SYMBOL        = "XAUUSD"
INITIAL_BAL   = 10_000.0
RISK_PCT      = 0.01
MAX_POSITIONS = 1
SL_PIPS       = 6.0
MAX_SL_PIPS   = 10.0
MIN_RR        = 1.5       # first TP at 1.5R (BE level)
SLIPPAGE_PCT  = 0.0002
START_DATE    = "2026-01-01"
END_DATE      = "2026-06-30"


def _print_sep(char: str = "─", w: int = 62) -> None:
    print(char * w)


def _monthly_breakdown(result) -> None:
    closed = result.closed_trades
    if not closed:
        return
    from collections import defaultdict
    monthly: dict[str, list] = defaultdict(list)
    for t in closed:
        key = t.closed_at.strftime("%Y-%m") if t.closed_at else "open"
        monthly[key].append(t.realized_pnl)
    print("\nMonthly breakdown:")
    _print_sep()
    print(f"  {'Month':<10} {'Trades':>7} {'P&L':>10} {'Win':>6}")
    _print_sep()
    for month in sorted(monthly):
        pnls  = monthly[month]
        wins  = sum(1 for p in pnls if p > 0)
        total = sum(pnls)
        wr    = wins / len(pnls) * 100 if pnls else 0
        print(f"  {month:<10} {len(pnls):>7} {total:>+10.2f} {wr:>5.0f}%")
    _print_sep()


def main() -> None:
    setup_logger("INFO")

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print(f"\nLoading XAUUSD M1 data from {DATA_DIR} …")
    df_m1 = load_m1_directory(DATA_DIR)
    tfs   = build_timeframes(df_m1)

    df_1m   = tfs["m1"]
    df_15m  = tfs["15min"]
    df_1h   = tfs["1h"]
    df_1d   = tfs["1d"]

    # LTF window for backtest: 1M bars, Jan–Jun 2026
    df_ltf = df_1m[START_DATE:END_DATE].copy()

    print(f"  M1  bars total  : {len(df_m1):>8,}")
    print(f"  M1  bars (2026) : {len(df_ltf):>8,}")
    print(f"  15M bars total  : {len(df_15m):>8,}")
    print(f"  1H  bars total  : {len(df_1h):>8,}")
    print(f"  1D  bars total  : {len(df_1d):>8,}")
    print(f"  Price range     : ${float(df_ltf['low'].min()):,.0f} – ${float(df_ltf['high'].max()):,.0f}")

    # ── 2. Build strategy ─────────────────────────────────────────────────────
    strategy = ScalpSMCStrategy(
        df_htf_1h=df_1h,
        df_mtf_15m=df_15m,
        df_daily=df_1d,
        sl_pips=SL_PIPS,
        max_sl_pips=MAX_SL_PIPS,
        pip_value=1.0,
        min_htf_score=3.0,
        swing_length=20,
        internal_length=5,
        atr_period=100,
        ltf_lookback=15,
        killzone_only=True,
        require_entry_fvg=True,
        require_choch_candle=False,
        max_daily_signals=5,
        max_signals_per_session=2,
    )

    # ── 3. Build engine ───────────────────────────────────────────────────────
    engine = AdvancedBacktestEngine(
        strategy=strategy,
        initial_balance=INITIAL_BAL,
        stop_loss_pips=SL_PIPS,
        max_sl_pips=MAX_SL_PIPS,
        pip_value=1.0,
        min_rr=MIN_RR,
        max_position_pct=RISK_PCT,
        max_open_positions=MAX_POSITIONS,
        fee_pct=0.0,
        slippage_pct=SLIPPAGE_PCT,
        partial_close=PartialCloseConfig.scalp(),
    )

    # ── 4. Run ────────────────────────────────────────────────────────────────
    print(f"\nRunning bar-by-bar scalp backtest on {len(df_ltf):,} 1M bars …")
    result  = engine.run(df_ltf, symbol=SYMBOL)
    summary = result.summary()

    # ── 5. Display results ────────────────────────────────────────────────────
    _print_sep("═")
    print(f"  XAUUSD 2026 H1 Scalp Backtest — ScalpSMCStrategy (1H→15M→1M)")
    _print_sep("═")
    print(f"  Period          : {START_DATE}  →  {END_DATE}")
    print(f"  Initial capital : ${INITIAL_BAL:>10,.2f}")
    print(f"  Final equity    : ${summary['final_equity']:>10,.2f}")
    print(f"  Net P&L         : ${summary['net_pnl']:>+10,.2f}")
    print(f"  Return          : {summary['total_return_pct']:>+9.2f} %")
    _print_sep()
    print(f"  Trades          : {summary['n_trades']:>10}")
    print(f"  Win rate        : {summary['win_rate']:>9.1f} %")
    print(f"  Avg win         : ${summary['avg_win']:>10,.2f}")
    print(f"  Avg loss        : ${summary['avg_loss']:>10,.2f}")
    print(f"  Profit factor   : {summary['profit_factor']:>10}")
    _print_sep()
    print(f"  Max drawdown    : {summary['max_drawdown_pct']:>9.2f} %")
    print(f"  Sharpe ratio    : {summary['sharpe_ratio']:>10.4f}")
    _print_sep("═")

    _monthly_breakdown(result)

    # ── 6. Save results ───────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "backtest_scalp_xauusd_2026_h1.json"

    trade_records = []
    for t in result.closed_trades:
        trade_records.append({
            "trade_id":  t.trade_id,
            "side":      t.side,
            "entry":     round(t.entry_price, 3),
            "sl":        round(t.sl_price, 3),
            "qty":       round(t.quantity, 6),
            "pnl":       round(t.realized_pnl, 2),
            "status":    t.status.value,
            "reason":    t.signal_reason,
            "closed_at": t.closed_at.isoformat() if t.closed_at else None,
        })

    output = {
        "meta": {
            "symbol":    SYMBOL,
            "start":     START_DATE,
            "end":       END_DATE,
            "timeframe": "1M → 1H/15M/1M cascade",
            "strategy":  "ScalpSMCStrategy",
            "features":  ["1h_htf_zones", "15m_mss", "1m_fvg_entry",
                          "killzone", "daily_bias", "zone_reentry_guard",
                          "partial_close_scalp", "daily_freq_gate"],
            "disabled":  ["weekly_bias", "choch_candle", "asian_sweep",
                          "premium_discount", "1h_mss_on_4h_strategy"],
            "sl_pips":   SL_PIPS,
            "max_sl_pips": MAX_SL_PIPS,
        },
        "summary": summary,
        "trades":  trade_records,
    }
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2, default=str)
    print(f"\nResults saved → {out_path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
