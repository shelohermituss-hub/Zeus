"""
MT5 Tick Data Loader
====================
Parses raw tick exports from MetaTrader 5 and returns a clean DataFrame
with bid/ask prices indexed by UTC timestamp.

Formats supportés (auto-détectés)
----------------------------------
Type A — colonnes séparées date/heure (séparateur , ou ; ou \\t) :
    2024.01.02,00:00:00.133,2062.53,2062.88,0.00,0,6
    <DATE>;<TIME>;<BID>;<ASK>;<LAST>;<VOLUME>;<FLAGS>

Type B — colonne datetime combinée (séparateur espace) :
    2024.01.02 00:00:00.133,2062.53,2062.88,0.00,0,6

Type C — format MT5 avec en-tête :
    Première ligne ignorée si elle contient des lettres/< >.

Le chargeur détecte automatiquement le séparateur et la structure de la
colonne datetime.  Il normalise le fuseau horaire de New York → UTC
(identique à parse_histdata_csv).

Usage typique
-------------
    from zeus.backtest.tick_loader import parse_mt5_ticks, resample_ticks

    tick_df  = parse_mt5_ticks("data/ticks/xauusd/XAUUSD_2024_ticks.csv")
    ohlcv_1s = resample_ticks(tick_df, "1s")   # pour simulate_all tick_df
    ohlcv_5s = resample_ticks(tick_df, "5s")   # résolution légère
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

# ── constantes ───────────────────────────────────────────────────────────────

_HEADER_RE = re.compile(r"[a-zA-Z<>]")   # ligne d'en-tête contient des lettres

# ── fonctions publiques ──────────────────────────────────────────────────────

def parse_mt5_ticks(path: str | Path) -> pd.DataFrame:
    """
    Lit un fichier CSV de ticks MT5 et renvoie un DataFrame avec :
      - Index : DatetimeIndex UTC tz-naive (précision milliseconde)
      - Colonnes : bid (float), ask (float), mid (float)

    Le fichier peut être compressé (.gz / .zip — supporté nativement par pandas).
    Les lignes dupliquées (même timestamp) sont supprimées (keep='last').
    """
    path = Path(path)
    raw  = _peek_lines(path, n=3)

    sep = _detect_sep(raw[0])
    has_header = bool(_HEADER_RE.search(raw[0]))
    split_date = _has_split_datetime(raw[1] if has_header else raw[0], sep)

    if split_date:
        cols = ["date", "time", "bid", "ask", "last", "volume", "flags"]
    else:
        cols = ["datetime", "bid", "ask", "last", "volume", "flags"]

    df = pd.read_csv(
        path,
        sep        = sep,
        header     = 0 if has_header else None,
        names      = None if has_header else cols,
        dtype      = str,
        engine     = "python",
        on_bad_lines = "skip",
    )

    # ── normalise les noms de colonnes ───────────────────────────────────────
    df.columns = [str(c).strip().lstrip("<").rstrip(">").lower() for c in df.columns]

    # ── construit la colonne datetime ────────────────────────────────────────
    if "date" in df.columns and "time" in df.columns:
        df["_dt"] = df["date"].str.strip() + " " + df["time"].str.strip()
    elif "datetime" in df.columns:
        df["_dt"] = df["datetime"].str.strip()
    else:
        # essai : première colonne = datetime combinée
        df["_dt"] = df.iloc[:, 0].str.strip()

    df["_dt"] = pd.to_datetime(df["_dt"], format="mixed", dayfirst=False)

    # ── convertit NY → UTC ───────────────────────────────────────────────────
    df["_dt"] = (
        df["_dt"]
        .dt.tz_localize("America/New_York", ambiguous="infer", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )
    df = df.set_index("_dt")
    df.index.name = "datetime"

    # ── colonnes bid / ask ───────────────────────────────────────────────────
    bid_col = _find_col(df, ("bid", "b", "best_bid"))
    ask_col = _find_col(df, ("ask", "a", "best_ask", "offer"))

    if bid_col is None or ask_col is None:
        raise ValueError(
            f"Impossible de trouver les colonnes bid/ask dans {path}.\n"
            f"Colonnes trouvées : {list(df.columns)}"
        )

    out = pd.DataFrame({
        "bid": pd.to_numeric(df[bid_col], errors="coerce"),
        "ask": pd.to_numeric(df[ask_col], errors="coerce"),
    })
    out["mid"] = (out["bid"] + out["ask"]) * 0.5
    out = out.dropna(subset=["bid", "ask"])
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


def resample_ticks(
    tick_df:     pd.DataFrame,
    freq:        str = "1s",
    price_col:   str = "mid",
) -> pd.DataFrame:
    """
    Rééchantillonne les ticks en OHLCV standard pour passer à simulate_all.

    Parameters
    ----------
    tick_df   : sortie de parse_mt5_ticks (colonnes bid, ask, mid)
    freq      : fréquence Pandas (ex. '1s', '5s', '500ms', '100ms')
    price_col : 'mid' (défaut), 'bid', ou 'ask' — prix utilisé pour OHLCV

    Returns
    -------
    DataFrame OHLCV avec les colonnes open, high, low, close, volume.
    """
    p = tick_df[price_col]
    ohlcv = p.resample(freq).agg(
        open  = "first",
        high  = "max",
        low   = "min",
        close = "last",
    ).dropna()
    ohlcv["volume"] = tick_df["bid"].resample(freq).count()
    return ohlcv


def load_tick_directory(
    data_dir:     str | Path,
    glob_pattern: str = "*.csv",
) -> pd.DataFrame:
    """
    Charge et concatène tous les fichiers de ticks d'un répertoire.
    Utile pour charger une année entière découpée par mois.
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"Aucun fichier trouvé dans {data_dir} ({glob_pattern})")
    parts = [parse_mt5_ticks(f) for f in files]
    df = pd.concat(parts).sort_index()
    return df[~df.index.duplicated(keep="last")]


# ── helpers internes ─────────────────────────────────────────────────────────

def _peek_lines(path: Path, n: int = 3) -> list[str]:
    """Lit les n premières lignes sans décompresser tout le fichier."""
    opener = {"gz": __import__("gzip").open, "zip": None}.get(path.suffix.lstrip("."))
    if opener:
        with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
            return [fh.readline() for _ in range(n)]
    with open(path, encoding="utf-8", errors="replace") as fh:
        return [fh.readline() for _ in range(n)]


def _detect_sep(line: str) -> str:
    """Détecte le séparateur dominant dans une ligne."""
    counts = {s: line.count(s) for s in (",", ";", "\t")}
    return max(counts, key=counts.get)


def _has_split_datetime(line: str, sep: str) -> bool:
    """True si date et heure sont dans deux colonnes séparées."""
    parts = line.split(sep)
    if len(parts) < 2:
        return False
    # La première colonne ressemble à une date seule (YYYY.MM.DD ou YYYY-MM-DD)
    return bool(re.match(r"\d{4}[.\-]\d{2}[.\-]\d{2}$", parts[0].strip()))


def _find_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Retourne le premier nom de colonne qui correspond à un candidat."""
    for c in df.columns:
        if c in candidates:
            return c
    return None
