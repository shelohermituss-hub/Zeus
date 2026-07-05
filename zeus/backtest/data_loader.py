"""
MT4/MT5 M1 data loader — supporte trois formats CSV :

  Format MS (HistData.com MetaStock) — symbole + datetime compact :
    SYMBOL,YYYYMMDDHHMM,open,high,low,close,volume
    Timezone : New York (America/New_York) → converti en UTC tz-naive.

  Format A (HistData.com MetaTrader) — séparateur date/heure en deux colonnes :
    YYYY.MM.DD,HH:MM,open,high,low,close,volume
    Timezone : New York (America/New_York) → converti en UTC tz-naive.

  Format B (autres sources) — datetime en colonne unique :
    YYYY-MM-DD HH:MM,open,high,low,close,volume
    Timezone : UTC supposé (pas de conversion).

Le format est détecté automatiquement à partir de la première ligne.

Typical usage
-------------
    from zeus.backtest.data_loader import load_m1_directory, build_timeframes, parse_m1_csv

    # HistData format (auto-detect)
    df = parse_m1_csv("data/historical/xauusd/m1/DAT_MT_XAUUSD_M1_2024.csv")

    # Load all CSVs in a directory
    tfs = build_timeframes(load_m1_directory("data/historical/xauusd/m1"))
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


def _parse_format_b(path: str | Path) -> pd.DataFrame:
    """Format B : YYYY-MM-DD HH:MM,open,high,low,close,volume — timestamps supposés UTC."""
    df = pd.read_csv(
        path,
        header=None,
        names=["datetime", "open", "high", "low", "close", "volume"],
    )
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M")
    df = df.set_index("datetime")
    return df[["open", "high", "low", "close", "volume"]].sort_index()


def _parse_format_ms(path: str | Path) -> pd.DataFrame:
    """Format MetaStock (HistData MS) : SYMBOL,YYYYMMDDHHMM,open,high,low,close,volume.

    Timezone : New York (America/New_York) → converti en UTC tz-naive.
    """
    df = pd.read_csv(
        path,
        header=None,
        names=["symbol", "datetime", "open", "high", "low", "close", "volume"],
        dtype={"datetime": str},
    )
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%d%H%M")
    df = df.set_index("datetime")
    df.index = (
        df.index
        .tz_localize(
            "America/New_York",
            ambiguous="infer",
            nonexistent="shift_forward",
        )
        .tz_convert("UTC")
        .tz_localize(None)
    )
    return df[["open", "high", "low", "close", "volume"]].sort_index()


def parse_m1_csv(path: str | Path) -> pd.DataFrame:
    """
    Charge un fichier M1 CSV en détectant automatiquement le format :
      - Format MS (HistData MetaStock) : première colonne = symbole (lettre)
        ex. XAUUSD,YYYYMMDDHHMM,open,...  — New York tz → UTC
      - Format A  (HistData MetaTrader)  : première colonne contient '.' → YYYY.MM.DD
        ex. YYYY.MM.DD,HH:MM,open,...    — New York tz → UTC
      - Format B  (autre source)         : première colonne contient '-' → YYYY-MM-DD HH:MM
        ex. YYYY-MM-DD HH:MM,open,...    — UTC direct
    """
    first = Path(path).open().readline().split(",")[0]
    if first and first[0].isalpha():
        return _parse_format_ms(path)
    if "." in first:
        return parse_histdata_csv(path)
    return _parse_format_b(path)


def load_m1_directory(
    data_dir: str | Path,
    glob_pattern: str = "DAT_M[TS]_*_M1_*.csv",
) -> pd.DataFrame:
    """
    Load and concatenate all HistData M1 CSV files found in *data_dir*.

    Files are matched by *glob_pattern* (default ``DAT_M[TS]_*_M1_*.csv``, covers
    both MetaTrader MT and MetaStock MS naming) and sorted chronologically by
    filename before concatenation.

    Duplicate index entries (weekend/holiday artefacts) are dropped.
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No HistData CSVs found in {data_dir}")

    parts = [parse_m1_csv(f) for f in files]
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
