"""
Paper trading signal persistence — SQLite-backed signal log.

Every signal emitted during a paper trading session is recorded with its
outcome once the trade closes.  This enables post-session analysis without
re-running the backtest engine.

Usage::

    store = SignalStore("data/paper_signals.db")
    store.record_signal(signal, strategy_name="harmonic")

    # When trade closes:
    store.update_outcome(signal_id, outcome="win", pnl_r=1.5, pnl_usd=75.0)

    # Analysis:
    df = store.to_dataframe()
    print(df.groupby("strategy")["pnl_r"].agg(["mean", "count", "sum"]))
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at  TEXT    NOT NULL,
    strategy     TEXT    NOT NULL,
    symbol       TEXT    NOT NULL,
    direction    TEXT    NOT NULL,
    entry_price  REAL    NOT NULL,
    stop_loss    REAL    NOT NULL,
    take_profit  REAL    NOT NULL,
    risk_reward  REAL    NOT NULL,
    formed_at    TEXT    NOT NULL,
    bar_index    INTEGER NOT NULL,
    zone_score   REAL    NOT NULL DEFAULT 0.0,
    outcome      TEXT,
    pnl_r        REAL,
    pnl_usd      REAL,
    closed_at    TEXT,
    extra        TEXT
);
"""


class SignalStore:
    """
    SQLite-backed store for paper trading signals.

    Thread-safety: not thread-safe; designed for single-threaded paper engine.
    The DB file is created automatically if it does not exist.
    """

    def __init__(self, db_path: str | Path = "data/paper_signals.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ── Write ────────────────────────────────────────────────────────────────

    def record_signal(
        self,
        signal:        Any,
        strategy_name: str,
        symbol:        str  = "XAUUSD",
        extra:         dict | None = None,
    ) -> int:
        """
        Persist a new signal.  Returns the auto-assigned row id.

        *signal* is any object with the standard Tradeable fields:
        direction, entry_price, stop_loss, take_profit, risk_reward,
        formed_at, bar_index, zone_score.
        """
        import json
        now = datetime.now(tz=timezone.utc).isoformat()
        cur = self._conn.execute(
            """
            INSERT INTO signals
              (recorded_at, strategy, symbol, direction,
               entry_price, stop_loss, take_profit, risk_reward,
               formed_at, bar_index, zone_score, extra)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                now,
                strategy_name,
                symbol,
                getattr(signal, "direction", "long"),
                getattr(signal, "entry_price", 0.0),
                getattr(signal, "stop_loss", 0.0),
                getattr(signal, "take_profit", 0.0),
                getattr(signal, "risk_reward", 0.0),
                str(getattr(signal, "formed_at", now)),
                getattr(signal, "bar_index", -1),
                getattr(signal, "zone_score", 0.0),
                json.dumps(extra) if extra else None,
            ),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def update_outcome(
        self,
        signal_id: int,
        outcome:   str,    # "win" | "loss" | "scratch"
        pnl_r:     float,
        pnl_usd:   float,
    ) -> None:
        """Record the trade outcome once the position closes."""
        closed_at = datetime.now(tz=timezone.utc).isoformat()
        self._conn.execute(
            """
            UPDATE signals
            SET outcome=?, pnl_r=?, pnl_usd=?, closed_at=?
            WHERE id=?
            """,
            (outcome, pnl_r, pnl_usd, closed_at, signal_id),
        )
        self._conn.commit()

    # ── Read ────────────────────────────────────────────────────────────────

    def to_dataframe(self, strategy: str | None = None) -> "pd.DataFrame":
        """Return all signals as a pandas DataFrame, optionally filtered by strategy."""
        import pandas as pd
        query = "SELECT * FROM signals"
        params: tuple = ()
        if strategy:
            query += " WHERE strategy = ?"
            params = (strategy,)
        query += " ORDER BY id"
        rows = self._conn.execute(query, params).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])

    def summary(self) -> dict:
        """Return aggregate statistics per strategy."""
        rows = self._conn.execute(
            """
            SELECT strategy,
                   COUNT(*)                          AS total,
                   SUM(CASE WHEN outcome='win'  THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN outcome='loss' THEN 1 ELSE 0 END) AS losses,
                   ROUND(AVG(pnl_r),  4)             AS avg_r,
                   ROUND(SUM(pnl_usd), 2)            AS total_usd
            FROM   signals
            WHERE  outcome IS NOT NULL
            GROUP  BY strategy
            """
        ).fetchall()
        return {r["strategy"]: dict(r) for r in rows}

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()
