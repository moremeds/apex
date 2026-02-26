"""DuckDB persistence — hot cache for ticks, bars, and signals."""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

import duckdb

from src.domain.events.domain_events import QuoteTick

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ticks (
    symbol VARCHAR,
    price DOUBLE,
    volume BIGINT,
    source VARCHAR,
    ts TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS bars (
    symbol VARCHAR,
    tf VARCHAR,
    o DOUBLE,
    h DOUBLE,
    l DOUBLE,
    c DOUBLE,
    v BIGINT,
    ts TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS signals (
    symbol VARCHAR,
    rule VARCHAR,
    direction VARCHAR,
    strength DOUBLE,
    ts TIMESTAMPTZ
);
"""


class ServerPersistence:
    """DuckDB-backed persistence for the live dashboard server.

    Buffers ticks in memory, periodically flushes to DuckDB.
    Bars and signals are written directly (lower volume).
    """

    def __init__(self, duckdb_path: str = "data/server.duckdb") -> None:
        # Ensure parent directory exists
        from pathlib import Path
        Path(duckdb_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = duckdb.connect(duckdb_path)
        self._db.execute(_SCHEMA_SQL)
        self._tick_buffer: List[tuple] = []
        self._lock = threading.Lock()

    @property
    def pending_tick_count(self) -> int:
        with self._lock:
            return len(self._tick_buffer)

    def buffer_tick(self, tick: QuoteTick) -> None:
        """Buffer a tick for batch flush."""
        with self._lock:
            self._tick_buffer.append((
                tick.symbol,
                tick.last,
                tick.volume,
                tick.source,
                tick.timestamp,
            ))

    def flush_to_duckdb(self) -> int:
        """Flush buffered ticks to DuckDB. Returns number flushed."""
        with self._lock:
            if not self._tick_buffer:
                return 0
            batch = self._tick_buffer.copy()
            self._tick_buffer.clear()

        self._db.executemany(
            "INSERT INTO ticks (symbol, price, volume, source, ts) VALUES (?, ?, ?, ?, ?)",
            batch,
        )
        logger.debug("Flushed %d ticks to DuckDB", len(batch))
        return len(batch)

    def insert_bar(
        self, symbol: str, tf: str,
        o: float, h: float, l: float, c: float, v: int,
        ts: datetime,
    ) -> None:
        """Insert a completed bar."""
        self._db.execute(
            "INSERT INTO bars (symbol, tf, o, h, l, c, v, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [symbol, tf, o, h, l, c, v, ts],
        )

    def insert_signal(
        self, symbol: str, rule: str, direction: str,
        strength: float, ts: datetime,
    ) -> None:
        """Insert a trading signal."""
        self._db.execute(
            "INSERT INTO signals (symbol, rule, direction, strength, ts) VALUES (?, ?, ?, ?, ?)",
            [symbol, rule, direction, strength, ts],
        )

    def query_ticks(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Query recent ticks for a symbol."""
        result = self._db.execute(
            "SELECT symbol, price, volume, source, ts FROM ticks "
            "WHERE symbol = ? ORDER BY ts ASC LIMIT ?",
            [symbol, limit],
        )
        cols = [desc[0] for desc in result.description]
        return [dict(zip(cols, row)) for row in result.fetchall()]

    def query_bars(
        self, symbol: str, tf: str, limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Query recent bars for a symbol and timeframe."""
        result = self._db.execute(
            "SELECT symbol, tf, o, h, l, c, v, ts FROM bars "
            "WHERE symbol = ? AND tf = ? ORDER BY ts ASC LIMIT ?",
            [symbol, tf, limit],
        )
        cols = [desc[0] for desc in result.description]
        return [dict(zip(cols, row)) for row in result.fetchall()]

    def query_signals(
        self, symbol: str = None, limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query recent signals, optionally filtered by symbol."""
        if symbol:
            result = self._db.execute(
                "SELECT symbol, rule, direction, strength, ts FROM signals "
                "WHERE symbol = ? ORDER BY ts DESC LIMIT ?",
                [symbol, limit],
            )
        else:
            result = self._db.execute(
                "SELECT symbol, rule, direction, strength, ts FROM signals "
                "ORDER BY ts DESC LIMIT ?",
                [limit],
            )
        cols = [desc[0] for desc in result.description]
        return [dict(zip(cols, row)) for row in result.fetchall()]

    def close(self) -> None:
        """Flush remaining ticks and close DuckDB connection."""
        self.flush_to_duckdb()
        self._db.close()
        logger.info("ServerPersistence closed")
