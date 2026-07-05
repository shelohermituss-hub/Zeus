"""
MT5 / HistData Tick Data Loader
================================
Parses raw tick exports et renvoie un DataFrame bid/ask/mid indexé UTC.

Formats supportés (auto-détectés)
----------------------------------
Type A — MT5, colonnes séparées date/heure (séparateur , ou ; ou \\t) :
    2024.01.02,00:00:00.133,2062.53,2062.88,0.00,0,6
    <DATE>;<TIME>;<BID>;<ASK>;<LAST>;<VOLUME>;<FLAGS>

Type B — MT5, colonne datetime combinée :
    2024.01.02 00:00:00.133,2062.53,2062.88,0.00,0,6

Type C — HistData.com NT tick (3 colonnes) :
    20250101 180000;2625.298000;0
    datetime compacte YYYYMMDD HHMMSS ; prix unique (bid) ; volume
    → ask = bid + synthetic_spread   (défaut : 0.30 USD/oz pour XAUUSD)

Les fichiers .zip et .gz sont supportés.
Timezone : New York → UTC (identique à parse_histdata_csv pour les M1).
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pandas as pd

# ── constantes ───────────────────────────────────────────────────────────────

_HEADER_RE      = re.compile(r"[a-zA-Z<>]")
_COMPACT_DT_RE  = re.compile(r"^\d{8} \d{6}")   # YYYYMMDD HHMMSS


# ── fonctions publiques ──────────────────────────────────────────────────────

def parse_mt5_ticks(
    path:             str | Path,
    synthetic_spread: float = 0.30,   # USD/oz ajouté pour ask quand format 1 prix
) -> pd.DataFrame:
    """
    Lit un fichier de ticks (MT5 ou HistData) et renvoie un DataFrame avec :
      - Index : DatetimeIndex UTC tz-naive
      - Colonnes : bid (float), ask (float), mid (float)

    synthetic_spread
        Utilisé uniquement pour les fichiers HistData à prix unique (3 colonnes).
        Le fichier donne le prix bid ; ask = bid + synthetic_spread.
        Pour XAUUSD : 0.30 USD/oz.  Pour le forex : 0.0001 typiquement.
        Passer 0.0 si le spread est déjà géré en aval.
    """
    path = Path(path)
    raw  = _peek_lines(path, n=3)

    sep        = _detect_sep(raw[0])
    has_header = bool(_HEADER_RE.search(raw[0]))
    first_data = raw[1] if has_header else raw[0]

    # ── Détection format HistData 3-colonnes ──────────────────────────────────
    if _is_histdata_tick(first_data, sep):
        return _parse_histdata_tick(path, sep, synthetic_spread)

    # ── Format MT5 (bid/ask explicites) ──────────────────────────────────────
    split_date = _has_split_datetime(first_data, sep)
    if split_date:
        cols = ["date", "time", "bid", "ask", "last", "volume", "flags"]
    else:
        cols = ["datetime", "bid", "ask", "last", "volume", "flags"]

    df = pd.read_csv(
        path,
        sep          = sep,
        header       = 0 if has_header else None,
        names        = None if has_header else cols,
        dtype        = str,
        engine       = "python",
        on_bad_lines = "skip",
    )
    df.columns = [str(c).strip().lstrip("<").rstrip(">").lower() for c in df.columns]

    if "date" in df.columns and "time" in df.columns:
        df["_dt"] = df["date"].str.strip() + " " + df["time"].str.strip()
    elif "datetime" in df.columns:
        df["_dt"] = df["datetime"].str.strip()
    else:
        df["_dt"] = df.iloc[:, 0].str.strip()

    df["_dt"] = pd.to_datetime(df["_dt"], format="mixed", dayfirst=False)
    df["_dt"] = _ny_to_utc(df["_dt"])
    df = df.set_index("_dt")
    df.index.name = "datetime"

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
    tick_df:   pd.DataFrame,
    freq:      str = "1s",
    price_col: str = "mid",
) -> pd.DataFrame:
    """
    Rééchantillonne les ticks en OHLCV pour passer à simulate_all(tick_df=…).

    freq      : '1s', '5s', '500ms', etc.
    price_col : 'mid' (défaut), 'bid', ou 'ask'
    """
    p     = tick_df[price_col]
    ohlcv = p.resample(freq).agg(
        open  = "first",
        high  = "max",
        low   = "min",
        close = "last",
    ).dropna()
    ohlcv["volume"] = tick_df["bid"].resample(freq).count()
    return ohlcv


def load_tick_directory(
    data_dir:         str | Path,
    synthetic_spread: float = 0.30,
    cache:            bool  = True,
) -> pd.DataFrame:
    """
    Charge et concatène tous les fichiers de ticks (.csv ou .zip) d'un répertoire.
    Utile pour charger une année entière découpée par mois.

    cache : si True (défaut), sauvegarde le résultat en Parquet la première fois
            (_cache.parquet dans le répertoire).  Les chargements suivants utilisent
            ce cache (x10–20 plus rapide).  Le cache est invalidé automatiquement
            si un fichier source est plus récent que lui.
    """
    data_dir = Path(data_dir)
    files    = sorted(data_dir.glob("*.csv")) + sorted(data_dir.glob("*.zip"))
    files    = sorted(f for f in files if f.stem != "_cache")   # exclut le cache lui-même
    if not files:
        raise FileNotFoundError(
            f"Aucun fichier .csv/.zip trouvé dans {data_dir}"
        )

    cache_path = data_dir / "_cache.parquet"
    if cache and cache_path.exists():
        cache_mtime = cache_path.stat().st_mtime
        if all(f.stat().st_mtime <= cache_mtime for f in files):
            return pd.read_parquet(cache_path)

    parts = [parse_mt5_ticks(f, synthetic_spread=synthetic_spread) for f in files]
    df    = pd.concat(parts).sort_index()
    df    = df[~df.index.duplicated(keep="last")]

    if cache:
        df.to_parquet(cache_path)

    return df


# ── parsing HistData 3-colonnes ───────────────────────────────────────────────

def _parse_histdata_tick(
    path:             Path,
    sep:              str,
    synthetic_spread: float,
) -> pd.DataFrame:
    """
    Parse le format HistData NT tick :
        YYYYMMDD HHMMSS ; bid_price ; volume

    bid  = price
    ask  = price + synthetic_spread
    mid  = price + synthetic_spread / 2
    """
    # Pour les .zip multi-fichiers, on extrait explicitement le CSV
    if path.suffix.lower() == ".zip":
        import io as _io
        with zipfile.ZipFile(path) as zf:
            inner = _zip_csv_name(zf)
            with zf.open(inner) as raw:
                source = _io.BytesIO(raw.read())
    else:
        source = path   # type: ignore[assignment]

    df = pd.read_csv(
        source,
        sep          = sep,
        header       = None,
        names        = ["datetime", "price", "volume"],
        dtype        = str,
        engine       = "python",
        on_bad_lines = "skip",
    )

    dt_str = df["datetime"].iloc[0].strip() if len(df) else ""
    if _COMPACT_DT_RE.match(dt_str):
        dt_parsed = pd.to_datetime(df["datetime"].str.strip(), format="%Y%m%d %H%M%S")
    else:
        dt_parsed = pd.to_datetime(df["datetime"].str.strip(), format="mixed", dayfirst=False)

    dt_parsed = _ny_to_utc(dt_parsed)
    df.index  = dt_parsed
    df.index.name = "datetime"

    bid = pd.to_numeric(df["price"], errors="coerce")
    out = pd.DataFrame({
        "bid": bid,
        "ask": bid + synthetic_spread,
        "mid": bid + synthetic_spread / 2.0,
    })
    out = out.dropna(subset=["bid"])
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


# ── helpers internes ─────────────────────────────────────────────────────────

def _zip_csv_name(zf: zipfile.ZipFile) -> str:
    """Retourne le nom du premier .csv dans l'archive (ignore les .txt, etc.)."""
    csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
    if csvs:
        return csvs[0]
    return zf.namelist()[0]   # fallback : premier fichier disponible


def _peek_lines(path: Path, n: int = 3) -> list[str]:
    """Lit les n premières lignes sans décompresser tout le fichier."""
    suffix = path.suffix.lower()
    if suffix == ".gz":
        import gzip
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            return [fh.readline() for _ in range(n)]
    if suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            inner = _zip_csv_name(zf)
            with zf.open(inner) as raw:
                lines, buf = [], b""
                for _ in range(n):
                    while b"\n" not in buf:
                        chunk = raw.read(8192)
                        if not chunk:
                            break
                        buf += chunk
                    line, _sep, buf = buf.partition(b"\n")
                    lines.append(line.decode("utf-8", errors="replace") + "\n")
                return lines
    with open(path, encoding="utf-8", errors="replace") as fh:
        return [fh.readline() for _ in range(n)]


def _detect_sep(line: str) -> str:
    counts = {s: line.count(s) for s in (",", ";", "\t")}
    return max(counts, key=counts.get)


def _is_histdata_tick(first_data_line: str, sep: str) -> bool:
    """True si format HistData 3-colonnes (datetime ; prix ; volume)."""
    parts = first_data_line.strip().split(sep)
    if len(parts) != 3:
        return False
    # La première colonne doit ressembler à une datetime (commence par 8 chiffres)
    return bool(re.match(r"^\d{8}", parts[0].strip()))


def _has_split_datetime(line: str, sep: str) -> bool:
    parts = line.split(sep)
    if len(parts) < 2:
        return False
    return bool(re.match(r"\d{4}[.\-]\d{2}[.\-]\d{2}$", parts[0].strip()))


def _find_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for c in df.columns:
        if c in candidates:
            return c
    return None


def _ny_to_utc(series: pd.Series) -> pd.Series:
    return (
        series
        .dt.tz_localize("America/New_York", ambiguous="infer", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )
