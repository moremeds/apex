"""PostgreSQL schema for APEX backend persistence.

Tables: bars, signals, summary, score_history, screener_results, backtest_results.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.infrastructure.persistence.database import Database

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS bars (
    symbol    TEXT NOT NULL,
    tf        TEXT NOT NULL,
    o         DOUBLE PRECISION,
    h         DOUBLE PRECISION,
    l         DOUBLE PRECISION,
    c         DOUBLE PRECISION,
    v         BIGINT,
    ts        TIMESTAMPTZ NOT NULL,
    UNIQUE (symbol, tf, ts)
);
CREATE INDEX IF NOT EXISTS idx_bars_symbol_tf ON bars (symbol, tf, ts);

CREATE TABLE IF NOT EXISTS signals (
    id        BIGSERIAL PRIMARY KEY,
    symbol    TEXT NOT NULL,
    rule      TEXT NOT NULL,
    direction TEXT NOT NULL,
    strength  DOUBLE PRECISION,
    ts        TIMESTAMPTZ NOT NULL,
    timeframe TEXT DEFAULT '',
    indicator TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts ON signals (symbol, ts);

CREATE TABLE IF NOT EXISTS summary (
    symbol           TEXT PRIMARY KEY,
    close            DOUBLE PRECISION,
    prev_close       DOUBLE PRECISION DEFAULT 0,
    daily_change_pct DOUBLE PRECISION,
    regime           TEXT,
    regime_name      TEXT,
    confidence       DOUBLE PRECISION,
    composite_score  DOUBLE PRECISION,
    sector           TEXT,
    component_states JSONB,
    updated_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS score_history (
    symbol      TEXT NOT NULL,
    ts          TIMESTAMPTZ NOT NULL,
    score       DOUBLE PRECISION,
    trend_state TEXT,
    regime      TEXT,
    PRIMARY KEY (symbol, ts)
);
CREATE INDEX IF NOT EXISTS idx_score_history_ts ON score_history (ts);

CREATE TABLE IF NOT EXISTS screener_results (
    id             BIGSERIAL PRIMARY KEY,
    run_id         TEXT NOT NULL,
    screener_type  TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    score          DOUBLE PRECISION,
    metadata       JSONB,
    ts             TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_screener_run ON screener_results (run_id);

CREATE TABLE IF NOT EXISTS backtest_results (
    id        BIGSERIAL PRIMARY KEY,
    run_id    TEXT NOT NULL,
    strategy  TEXT NOT NULL,
    symbols   JSONB,
    metrics   JSONB,
    ts        TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_backtest_run ON backtest_results (run_id);
"""

DROP_SQL = """
DROP TABLE IF EXISTS backtest_results CASCADE;
DROP TABLE IF EXISTS screener_results CASCADE;
DROP TABLE IF EXISTS score_history CASCADE;
DROP TABLE IF EXISTS summary CASCADE;
DROP TABLE IF EXISTS signals CASCADE;
DROP TABLE IF EXISTS bars CASCADE;
"""


async def ensure_schema(db: Database) -> None:
    """Create tables if they don't exist."""
    await db.execute(SCHEMA_SQL)
    logger.info("PG schema ensured (6 tables)")


async def reset_schema(db: Database) -> None:
    """Drop and recreate all tables."""
    await db.execute(DROP_SQL)
    await db.execute(SCHEMA_SQL)
    logger.info("PG schema reset (6 tables)")


async def _cli_main() -> None:
    """CLI entry point for schema management."""
    from config.config_manager import ConfigManager
    from src.infrastructure.persistence.database import Database

    cm = ConfigManager(config_dir="config", env="dev")
    config = cm.load()

    assert config.database is not None, "database config required in config/base.yaml"
    db = Database(config.database)
    await db.connect()

    action = sys.argv[1] if len(sys.argv) > 1 else "--init"
    if action == "--reset":
        await reset_schema(db)
        print("Schema reset complete.")
    else:
        await ensure_schema(db)
        print("Schema init complete.")

    await db.close()


if __name__ == "__main__":
    asyncio.run(_cli_main())
