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
CREATE INDEX IF NOT EXISTS idx_ticks_symbol_ts ON ticks (symbol, ts);

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
CREATE INDEX IF NOT EXISTS idx_bars_symbol_tf_ts ON bars (symbol, tf, ts);
CREATE UNIQUE INDEX IF NOT EXISTS idx_bars_unique ON bars (symbol, tf, ts);

CREATE TABLE IF NOT EXISTS signals (
    symbol VARCHAR,
    rule VARCHAR,
    direction VARCHAR,
    strength DOUBLE,
    ts TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts ON signals (symbol, ts);

CREATE TABLE IF NOT EXISTS summary (
    symbol VARCHAR PRIMARY KEY,
    close DOUBLE,
    prev_close DOUBLE DEFAULT 0,
    daily_change_pct DOUBLE,
    regime VARCHAR,
    regime_name VARCHAR,
    confidence INTEGER,
    composite_score DOUBLE,
    sector VARCHAR,
    component_states VARCHAR,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS score_history (
    symbol VARCHAR,
    ts TIMESTAMP,
    score DOUBLE,
    trend_state VARCHAR,
    regime VARCHAR,
    PRIMARY KEY (symbol, ts)
);
CREATE INDEX IF NOT EXISTS idx_score_history_ts ON score_history(ts);
"""


class ServerPersistence:
    """DuckDB-backed persistence for the live dashboard server.

    Buffers ticks in memory, periodically flushes to DuckDB.
    Bars and signals are written directly (lower volume).
    """

    # Maximum ticks to buffer before auto-flush to prevent unbounded memory growth
    MAX_BUFFER_SIZE = 50_000

    def __init__(self, duckdb_path: str = "data/server.duckdb") -> None:
        # Ensure parent directory exists
        from pathlib import Path

        Path(duckdb_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = duckdb.connect(duckdb_path)
        self._db.execute(_SCHEMA_SQL)
        self._migrate_signals_columns()
        self._migrate_summary_prev_close()
        self._tick_buffer: List[tuple] = []
        self._lock = threading.Lock()

    def _migrate_signals_columns(self) -> None:
        """Add timeframe/indicator columns and index to signals table if missing."""
        try:
            cols = {
                row[0]
                for row in self._db.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'signals'"
                ).fetchall()
            }
            if "timeframe" not in cols:
                self._db.execute("ALTER TABLE signals ADD COLUMN timeframe VARCHAR DEFAULT ''")
            if "indicator" not in cols:
                self._db.execute("ALTER TABLE signals ADD COLUMN indicator VARCHAR DEFAULT ''")
            # Index creation is safe now — columns are guaranteed to exist
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_signals_symbol_tf ON signals (symbol, timeframe)"
            )
        except Exception:
            logger.debug("Signals column migration skipped", exc_info=True)

    def _migrate_summary_prev_close(self) -> None:
        """Add prev_close column to summary table if missing (for existing DBs)."""
        try:
            cols = {
                row[0]
                for row in self._db.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'summary'"
                ).fetchall()
            }
            if "prev_close" not in cols:
                self._db.execute(
                    "ALTER TABLE summary ADD COLUMN prev_close DOUBLE DEFAULT 0"
                )
        except Exception:
            logger.debug("Summary prev_close migration skipped", exc_info=True)

    @property
    def pending_tick_count(self) -> int:
        with self._lock:
            return len(self._tick_buffer)

    def buffer_tick(self, tick: QuoteTick) -> None:
        """Buffer a tick for batch flush. Auto-flushes if buffer exceeds cap."""
        with self._lock:
            self._tick_buffer.append(
                (
                    tick.symbol,
                    tick.last,
                    tick.volume,
                    tick.source,
                    tick.timestamp,
                )
            )
            needs_flush = len(self._tick_buffer) >= self.MAX_BUFFER_SIZE

        if needs_flush:
            self.flush_to_duckdb()

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

    def bulk_insert_bars(self, symbol: str, tf: str, bars: list[dict]) -> int:
        """Bulk insert historical bars. Skips duplicates via INSERT OR IGNORE."""
        if not bars:
            return 0
        with self._lock:
            self._db.executemany(
                "INSERT OR IGNORE INTO bars (symbol, tf, o, h, l, c, v, ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        symbol,
                        tf,
                        b["open"],
                        b["high"],
                        b["low"],
                        b["close"],
                        b["volume"],
                        b["timestamp"],
                    )
                    for b in bars
                ],
            )
        return len(bars)

    def save_summary(self, summary: dict) -> None:
        """Persist summary tickers to DuckDB."""
        import json

        tickers = summary.get("tickers", [])
        if not tickers:
            return
        with self._lock:
            self._db.execute("DELETE FROM summary")
            self._db.executemany(
                "INSERT INTO summary (symbol, close, prev_close, daily_change_pct, regime, "
                "regime_name, confidence, composite_score, sector, component_states) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        t["symbol"],
                        t.get("close", 0.0),
                        t.get("prev_close", 0.0),
                        t.get("daily_change_pct", 0.0),
                        t.get("regime", "R1"),
                        t.get("regime_name", "Unknown"),
                        t.get("confidence", 50),
                        t.get("composite_score_avg", 0.0),
                        t.get("sector", ""),
                        json.dumps(t.get("component_states", {})),
                    )
                    for t in tickers
                ],
            )
        logger.info("Saved summary for %d tickers to DuckDB", len(tickers))

    def get_summary(self) -> Optional[Dict[str, Any]]:
        """Read summary from DuckDB. Returns dict matching API format or None."""
        import json

        with self._lock:
            result = self._db.execute(
                "SELECT symbol, close, prev_close, daily_change_pct, regime, regime_name, "
                "confidence, composite_score, sector, component_states FROM summary"
            )
            rows = result.fetchall()
        if not rows:
            return None
        tickers = []
        for row in rows:
            comp_states = {}
            if row[9]:
                try:
                    comp_states = json.loads(row[9])
                except Exception:
                    pass
            tickers.append(
                {
                    "symbol": row[0],
                    "close": row[1],
                    "prev_close": row[2],
                    "daily_change_pct": row[3],
                    "regime": row[4],
                    "regime_name": row[5],
                    "confidence": row[6],
                    "composite_score_avg": row[7],
                    "sector": row[8],
                    "component_states": comp_states,
                }
            )
        return {"tickers": tickers, "source": "duckdb"}

    def insert_bar(
        self,
        symbol: str,
        tf: str,
        o: float,
        h: float,
        l: float,
        c: float,
        v: int,
        ts: datetime,
    ) -> None:
        """Insert a completed bar."""
        with self._lock:
            self._db.execute(
                "INSERT INTO bars (symbol, tf, o, h, l, c, v, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [symbol, tf, o, h, l, c, v, ts],
            )

    def insert_signal(
        self,
        symbol: str,
        rule: str,
        direction: str,
        strength: float,
        ts: datetime,
        timeframe: str = "",
        indicator: str = "",
    ) -> None:
        """Insert a trading signal."""
        with self._lock:
            self._db.execute(
                "INSERT INTO signals (symbol, rule, direction, strength, timeframe, indicator, ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [symbol, rule, direction, strength, timeframe, indicator, ts],
            )

    def query_ticks(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Query recent ticks for a symbol."""
        with self._lock:
            result = self._db.execute(
                "SELECT symbol, price, volume, source, ts FROM ticks "
                "WHERE symbol = ? ORDER BY ts ASC LIMIT ?",
                [symbol, limit],
            )
            cols = [desc[0] for desc in result.description]
            rows = result.fetchall()
        return [dict(zip(cols, row)) for row in rows]

    def query_bars(
        self,
        symbol: str,
        tf: str,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Query recent bars for a symbol and timeframe."""
        with self._lock:
            result = self._db.execute(
                "SELECT symbol, tf, o, h, l, c, v, ts FROM bars "
                "WHERE symbol = ? AND tf = ? ORDER BY ts ASC LIMIT ?",
                [symbol, tf, limit],
            )
            cols = [desc[0] for desc in result.description]
            rows = result.fetchall()
        return [dict(zip(cols, row)) for row in rows]

    def query_signals(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query recent signals, optionally filtered by symbol and timeframe."""
        conditions = []
        params: list = []
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if timeframe:
            conditions.append("timeframe = ?")
            params.append(timeframe)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)
        with self._lock:
            result = self._db.execute(
                f"SELECT symbol, rule, direction, strength, timeframe, indicator, ts "
                f"FROM signals{where} ORDER BY ts DESC LIMIT ?",
                params,
            )
            cols = [desc[0] for desc in result.description]
            rows = result.fetchall()
        return [dict(zip(cols, row)) for row in rows]

    def save_score_snapshot(
        self, symbol: str, ts: datetime, score: float, trend_state: str, regime: str
    ) -> None:
        """Insert or replace a score snapshot (for sparklines)."""
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO score_history VALUES (?, ?, ?, ?, ?)",
                [symbol, ts, score, trend_state, regime],
            )

    def get_score_history(self, days: int = 30) -> Dict[str, Any]:
        """Get score history grouped by timestamp for sparklines.

        Returns {"snapshots": [{"scores": {"AAPL": 72.5, ...}}, ...]}
        to match the frontend Overview.tsx contract (ScoreSnapshot[]).
        """
        from collections import OrderedDict
        from datetime import timedelta, timezone

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        with self._lock:
            rows = self._db.execute(
                "SELECT symbol, ts, score "
                "FROM score_history WHERE ts >= ? ORDER BY ts",
                [cutoff],
            ).fetchall()
        # Group by timestamp → {ts: {symbol: score}}
        by_ts: OrderedDict[str, Dict[str, float]] = OrderedDict()
        for symbol, ts, score in rows:
            ts_key = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            if ts_key not in by_ts:
                by_ts[ts_key] = {}
            by_ts[ts_key][symbol] = round(score, 1) if score is not None else 0.0
        # Convert to array of snapshots
        snapshots = [{"scores": scores} for scores in by_ts.values()]
        return {"snapshots": snapshots}

    def close(self) -> None:
        """Flush remaining ticks and close DuckDB connection."""
        self.flush_to_duckdb()
        self._db.close()
        logger.info("ServerPersistence closed")
