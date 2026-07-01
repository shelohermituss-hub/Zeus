"""
HistData.com MT4/MT5 M1 data loader for XAUUSD backtests.

File format (no header):
    YYYY.MM.DD,HH:MM,open,high,low,close,volume

Timestamps are in New York time (America/New_York) — the standard for HistData
forex/gold files. They are converted to tz-naive UTC on load so they integrate
directly with the strategy's session detection logic.

Typical usage
-------------
    from zeus.backtest.data_loader import load_m1_directory, build_timeframes

    # Load all CSVs in the data directory and resample
    tfs = build_timeframes(load_m1_directory("data/historical/xauusd/m1"))
    df_5m, df_1h, df_4h, df_1d = tfs["5min"], tfs["1h"], tfs["4h"], tfs["1d"]
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_COLUMNS = ["date", "time", "open", "high", "low", "close", "volume"]


def parse_histdata_csv(path: str | Path) -> pd.DataFrame:
    """
    Parse a single HistData MT4 M1 CSV file.

    Returns a tz-naive UTC DatetimeIndex DataFrame with columns:
    open, high, low, close, volume.
    """
    df = pd.read_csv(
        path,
        header=None,
        names=_COLUMNS,
        dtype={"date": str, "time": str},
    )
    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M"
    )
    df = df.set_index("datetime")

    # Localize as New York time (handles DST automatically) then convert to UTC
    df.index = (
        df.index
        .tz_localize(
            "America/New_York",
            ambiguous="infer",
            nonexistent="shift_forward",
        )
        .tz_convert("UTC")
        .tz_localize(None)          # tz-naive UTC for pandas compatibility
    )
    return df[["open", "high", "low", "close", "volume"]].sort_index()


def load_m1_directory(data_dir: str | Path) -> pd.DataFrame:
    """
    Load and concatenate all HistData M1 CSV files found in *data_dir*.

    Files are matched by the glob ``DAT_MT_XAUUSD_M1_*.csv`` and sorted
    chronologically by filename before concatenation.

    Duplicate index entries (weekend/holiday artefacts) are dropped.
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("DAT_MT_XAUUSD_M1_*.csv"))
    if not files:
        raise FileNotFoundError(f"No HistData CSVs found in {data_dir}")

    parts = [parse_histdata_csv(f) for f in files]
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Resample an OHLCV M1 DataFrame to a higher timeframe.

    Args:
        df:   M1 DataFrame with DatetimeIndex.
        freq: Pandas offset alias — e.g. ``"5min"``, ``"1h"``, ``"4h"``, ``"1D"``.

    Returns:
        Resampled DataFrame with NaN rows dropped.
    """
    return (
        df.resample(freq, label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min",
              "close": "last",  "volume": "sum"})
        .dropna(subset=["open"])
    )


def build_timeframes(df_m1: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Resample M1 data to all timeframes used by swing and scalp strategies.

    Returns
    -------
    dict with keys:
        ``"m1"``    — raw M1           (LTF entry bars, tightest SL)
        ``"3min"``  — 3-minute         (ultra-scalp entry)
        ``"5min"``  — 5-minute         (scalp entry / refinement)
        ``"15min"`` — 15-minute        (refinement / MSS gate)
        ``"30min"`` — 30-minute        (intraday HTF / MSS gate)
        ``"1h"``    — 1-hour           (HTF zone analysis)
        ``"4h"``    — 4-hour           (swing HTF analysis)
        ``"1d"``    — daily            (daily bias gate)
    """
    return {
        "m1":    df_m1,
        "3min":  resample_ohlcv(df_m1, "3min"),
        "5min":  resample_ohlcv(df_m1, "5min"),
        "15min": resample_ohlcv(df_m1, "15min"),
        "30min": resample_ohlcv(df_m1, "30min"),
        "1h":    resample_ohlcv(df_m1, "1h"),
        "4h":    resample_ohlcv(df_m1, "4h"),
        "1d":    resample_ohlcv(df_m1, "1D"),
    }
