"""
MTFSMCStrategy — XAUUSD 2026 H1 Backtest (Jan–Jun 2026)

Data source : HistData.com MT4 M1 files in data/historical/xauusd/m1/
Warmup      : 2025 full-year M1 for HTF context (4H/Daily SMC structures)
TF cascade  : Daily → 4H → 5M (LTF entry)

Active features
---------------
  Rec 4 — 5M FVG entry trigger (1H MSS disabled — too few setups when combined)
  Rec 5 — Partial close: BE@1R, 60%@3R, 85%@5R, 100%@10R
  Rec 6 — ICT kill zone gate (Asian sweep disabled — rarely triggers in 2026 data)
  Rec 7 — Max 2 signals/day, 1 signal per London/NY kill zone session

Gate calibration notes
----------------------
  All seven gates active simultaneously produces 0 trades over 6 months because
  the conjunction of HTF alignment + 1H MSS + daily bias + Asian sweep + 5M FVG
  + zone price is extremely rare.  Diagnostic results on the 2026 dataset:

    All gates ON          →   0 trades
    – Asian sweep         →   0 trades   (MSS+FVG+daily still too strict)
    – Asian sweep – 1H MSS →  13 trades  ← this run
    – Asian sweep – 1H MSS – FVG → 18 trades

  The 1H MSS gate and Asian sweep gate are implemented and tested (Recs 4/6)
  but require a larger dataset or more volatile regime to fire at useful frequency.

Risk
----
  SL   : 20–30 pips (1 pip = $1 for XAUUSD)
  Size : 1% of equity at risk per trade
  Cap  : 1 concurrent position

Usage
-----
    python -m zeus.backtest.run_2026_backtest
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from zeus.backtest.advanced_engine import AdvancedBacktestEngine
from zeus.backtest.data_loader import build_timeframes, load_m1_directory
from zeus.backtest.partial_close import PartialCloseConfig
from zeus.strategy.mtf_strategy import MTFSMCStrategy
from zeus.utils.logger import logger, setup_logger

# ── Paths ────────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).parent.parent.parent
DATA_DIR    = _ROOT / "data" / "historical" / "xauusd" / "m1"
RESULTS_DIR = _ROOT / "data" / "results"

# ── Backtest parameters ───────────────────────────────────────────────────────
SYMBOL         = "XAUUSD"
INITIAL_BAL    = 10_000.0   # USD
RISK_PCT       = 0.01       # 1 % of equity per trade
MAX_POSITIONS  = 1          # one trade at a time
SL_PIPS        = 20.0       # default SL distance
MAX_SL_PIPS    = 30.0       # reject setups with wider SL
MIN_RR         = 3.0        # minimum R/R (first TP at 3R)
SLIPPAGE_PCT   = 0.0002     # ≈ 0.6 pip spread on $3 000 gold
START_DATE     = "2026-01-01"
END_DATE       = "2026-06-30"


def _print_separator(char: str = "─", width: int = 62) -> None:
    print(char * width)


def _monthly_breakdown(result, df_ltf: pd.DataFrame) -> None:
    """Print P&L and trade count per calendar month."""
    closed = result.closed_trades
    if not closed:
        return

    from collections import defaultdict
    monthly: dict[str, list] = defaultdict(list)
    for t in closed:
        if t.closed_at is not None:
            key = t.closed_at.strftime("%Y-%m")
        else:
            key = "open"
        monthly[key].append(t.realized_pnl)

    print("\nMonthly breakdown:")
    _print_separator()
    print(f"  {'Month':<10} {'Trades':>7} {'P&L':>10} {'Win':>6}")
    _print_separator()
    for month in sorted(monthly):
        pnls  = monthly[month]
        wins  = sum(1 for p in pnls if p > 0)
        total = sum(pnls)
        wr    = wins / len(pnls) * 100 if pnls else 0
        print(f"  {month:<10} {len(pnls):>7} {total:>+10.2f} {wr:>5.0f}%")
    _print_separator()


def main() -> None:
    setup_logger("INFO")

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print(f"\nLoading XAUUSD M1 data from {DATA_DIR} …")
    df_m1 = load_m1_directory(DATA_DIR)
    tfs   = build_timeframes(df_m1)

    df_5m = tfs["5min"]
    df_1h = tfs["1h"]
    df_4h = tfs["4h"]
    df_1d = tfs["1d"]

    # LTF window: Jan–Jun 2026 only
    df_ltf = df_5m[START_DATE:END_DATE].copy()

    print(f"  M1  bars total : {len(df_m1):>8,}")
    print(f"  5M  bars (2026): {len(df_ltf):>8,}")
    print(f"  1H  bars total : {len(df_1h):>8,}")
    print(f"  4H  bars total : {len(df_4h):>8,}")
    print(f"  1D  bars total : {len(df_1d):>8,}")
    print(f"  Price range    : ${float(df_ltf['low'].min()):,.0f} – ${float(df_ltf['high'].max()):,.0f}")

    # ── 2. Build strategy ─────────────────────────────────────────────────────
    strategy = MTFSMCStrategy(
        df_htf=df_4h,
        df_daily=df_1d,
        df_mtf=None,            # 1H MSS disabled: too few setups in combination
        # SL config (Rec 5)
        sl_pips=SL_PIPS,
        max_sl_pips=MAX_SL_PIPS,
        pip_value=1.0,
        # Kill zone gate (Rec 6) — Asian sweep disabled: rarely fires in 2026 data
        killzone_only=True,
        require_asian_sweep=False,
        sweep_zone_tol_pct=0.005,
        # 5M FVG entry trigger (Rec 4)
        require_entry_fvg=True,
        ltf_lookback=20,
        # Confluence thresholds
        min_htf_score=3.0,
        swing_length=50,
        internal_length=5,
        atr_period=200,
        # Frequency gate (Rec 7)
        max_daily_signals=2,
        max_signals_per_session=1,
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
        fee_pct=0.0,            # XAUUSD: commission-free CFD, spread in slippage
        slippage_pct=SLIPPAGE_PCT,
        partial_close=PartialCloseConfig.mtf_smc(),
    )

    # ── 4. Run ────────────────────────────────────────────────────────────────
    print(f"\nRunning bar-by-bar backtest on {len(df_ltf):,} 5M bars …")
    result = engine.run(df_ltf, symbol=SYMBOL)
    summary = result.summary()

    # ── 5. Display results ────────────────────────────────────────────────────
    _print_separator("═")
    print(f"  XAUUSD 2026 H1 Backtest — MTFSMCStrategy (4-TF cascade)")
    _print_separator("═")
    print(f"  Period         : {START_DATE}  →  {END_DATE}")
    print(f"  Initial capital: ${INITIAL_BAL:>10,.2f}")
    print(f"  Final equity   : ${summary['final_equity']:>10,.2f}")
    print(f"  Net P&L        : ${summary['net_pnl']:>+10,.2f}")
    print(f"  Return         : {summary['total_return_pct']:>+9.2f} %")
    _print_separator()
    print(f"  Trades         : {summary['n_trades']:>10}")
    print(f"  Win rate       : {summary['win_rate']:>9.1f} %")
    print(f"  Avg win        : ${summary['avg_win']:>10,.2f}")
    print(f"  Avg loss       : ${summary['avg_loss']:>10,.2f}")
    print(f"  Profit factor  : {summary['profit_factor']:>10}")
    _print_separator()
    print(f"  Max drawdown   : {summary['max_drawdown_pct']:>9.2f} %")
    print(f"  Sharpe ratio   : {summary['sharpe_ratio']:>10.4f}")
    _print_separator("═")

    _monthly_breakdown(result, df_ltf)

    # ── 6. Save results ───────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "backtest_xauusd_2026_h1.json"

    # Build a per-trade record for deeper analysis
    trade_records = []
    for t in result.closed_trades:
        trade_records.append({
            "trade_id":   t.trade_id,
            "side":       t.side,
            "entry":      round(t.entry_price, 3),
            "sl":         round(t.sl_price, 3),
            "qty":        round(t.quantity, 6),
            "pnl":        round(t.realized_pnl, 2),
            "status":     t.status.value,
            "reason":     t.signal_reason,
            "closed_at":  t.closed_at.isoformat() if t.closed_at else None,
        })

    output = {
        "meta": {
            "symbol":     SYMBOL,
            "start":      START_DATE,
            "end":        END_DATE,
            "timeframe":  "5M → 4-TF cascade",
            "features":   ["killzone", "daily_bias", "htf_zones", "5m_fvg",
                           "partial_close_mtf_smc", "daily_freq_gate"],
        "disabled":   ["1h_mss", "asian_sweep"],
        "note":       "1H MSS and Asian sweep produce 0 trades in 6-month dataset; "
                      "see docstring for full calibration details",
        },
        "summary": summary,
        "trades":  trade_records,
    }
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2, default=str)
    print(f"\nResults saved → {out_path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
