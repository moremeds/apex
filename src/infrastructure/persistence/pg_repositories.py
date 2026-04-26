"""PostgreSQL repositories for APEX backend persistence."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from src.infrastructure.persistence.database import Database

logger = logging.getLogger(__name__)


class PgRepositories:
    """Unified repository for all APEX PG tables."""

    def __init__(self, db: Database) -> None:
        self._db = db

    # ── Bars ──────────────────────────────────────────────────

    async def insert_bar(
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
        await self._db.execute(
            """INSERT INTO bars (symbol, tf, o, h, l, c, v, ts)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
               ON CONFLICT (symbol, tf, ts) DO UPDATE
               SET o=$3, h=$4, l=$5, c=$6, v=$7""",
            symbol,
            tf,
            o,
            h,
            l,
            c,
            v,
            ts,
        )

    async def bulk_insert_bars(self, symbol: str, tf: str, bars: list[dict]) -> int:
        if not bars:
            return 0
        rows = [
            (
                symbol,
                tf,
                float(b.get("open", 0)),
                float(b.get("high", 0)),
                float(b.get("low", 0)),
                float(b.get("close", 0)),
                int(b.get("volume", 0)) if b.get("volume") else 0,
                b["timestamp"],
            )
            for b in bars
        ]
        async with self._db.transaction() as conn:
            await conn.executemany(
                """INSERT INTO bars (symbol, tf, o, h, l, c, v, ts)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                   ON CONFLICT (symbol, tf, ts) DO NOTHING""",
                rows,
            )
        return len(rows)

    async def query_bars(self, symbol: str, tf: str, limit: int = 500) -> list[dict]:
        rows = await self._db.fetch(
            "SELECT symbol, tf, o, h, l, c, v, ts FROM bars "
            "WHERE symbol=$1 AND tf=$2 ORDER BY ts DESC LIMIT $3",
            symbol,
            tf,
            limit,
        )
        return [dict(r) for r in rows]

    # ── Signals ───────────────────────────────────────────────

    async def insert_signal(
        self,
        symbol: str,
        rule: str,
        direction: str,
        strength: float,
        ts: datetime,
        timeframe: str = "",
        indicator: str = "",
    ) -> None:
        await self._db.execute(
            """INSERT INTO signals (symbol, rule, direction, strength, ts, timeframe, indicator)
               VALUES ($1, $2, $3, $4, $5, $6, $7)""",
            symbol,
            rule,
            direction,
            strength,
            ts,
            timeframe,
            indicator,
        )

    async def query_signals(
        self,
        symbol: str | None = None,
        timeframe: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        conditions: list[str] = []
        params: list[Any] = []
        idx = 1
        if symbol:
            conditions.append(f"symbol=${idx}")
            params.append(symbol)
            idx += 1
        if timeframe:
            conditions.append(f"timeframe=${idx}")
            params.append(timeframe)
            idx += 1
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)
        rows = await self._db.fetch(
            f"SELECT symbol, rule, direction, strength, ts, timeframe, indicator "  # noqa: S608
            f"FROM signals{where} ORDER BY ts DESC LIMIT ${idx}",
            *params,
        )
        return [dict(r) for r in rows]

    # ── Summary ───────────────────────────────────────────────

    async def save_summary(self, summary: dict) -> None:
        tickers = summary.get("tickers", [])
        if not tickers:
            return
        await self._db.execute("DELETE FROM summary")
        for t in tickers:
            comp_states = t.get("component_states", {})
            await self._db.execute(
                """INSERT INTO summary (symbol, close, prev_close, daily_change_pct,
                   regime, regime_name, confidence, composite_score, sector,
                   component_states, updated_at)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,NOW())""",
                t["symbol"],
                t.get("close", 0),
                t.get("prev_close", 0),
                t.get("daily_change_pct", 0),
                t.get("regime", "R1"),
                t.get("regime_name", "Unknown"),
                t.get("confidence", 50),
                t.get("composite_score_avg", 0),
                t.get("sector", ""),
                json.dumps(comp_states),
            )

    async def get_summary(self) -> list[dict]:
        rows = await self._db.fetch("SELECT * FROM summary ORDER BY symbol")
        return [dict(r) for r in rows]

    # ── Score History ─────────────────────────────────────────

    async def save_score_snapshot(
        self,
        symbol: str,
        ts: datetime,
        score: float,
        trend_state: str,
        regime: str,
    ) -> None:
        await self._db.execute(
            """INSERT INTO score_history (symbol, ts, score, trend_state, regime)
               VALUES ($1, $2, $3, $4, $5)
               ON CONFLICT (symbol, ts) DO UPDATE
               SET score=$3, trend_state=$4, regime=$5""",
            symbol,
            ts,
            score,
            trend_state,
            regime,
        )

    async def get_score_history(self, days: int = 30) -> list[dict]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        rows = await self._db.fetch(
            "SELECT symbol, ts, score, trend_state, regime "
            "FROM score_history WHERE ts >= $1 ORDER BY ts",
            cutoff,
        )
        return [dict(r) for r in rows]

    # ── Screener Results ──────────────────────────────────────

    async def insert_screener_results(
        self,
        run_id: str,
        screener_type: str,
        results: list[dict],
    ) -> None:
        if not results:
            return
        rows = [
            (
                run_id,
                screener_type,
                r["symbol"],
                r.get("score", 0),
                json.dumps(r.get("metadata", {})),
            )
            for r in results
        ]
        async with self._db.transaction() as conn:
            await conn.executemany(
                """INSERT INTO screener_results
                   (run_id, screener_type, symbol, score, metadata)
                   VALUES ($1, $2, $3, $4, $5::jsonb)""",
                rows,
            )

    async def query_screener_results(self, run_id: str) -> list[dict]:
        rows = await self._db.fetch(
            "SELECT * FROM screener_results WHERE run_id=$1 ORDER BY score DESC",
            run_id,
        )
        return [dict(r) for r in rows]

    # ── Backtest Results ──────────────────────────────────────

    async def insert_backtest_results(
        self,
        run_id: str,
        strategy: str,
        symbols: list[str],
        metrics: dict,
    ) -> None:
        await self._db.execute(
            """INSERT INTO backtest_results (run_id, strategy, symbols, metrics)
               VALUES ($1, $2, $3::jsonb, $4::jsonb)""",
            run_id,
            strategy,
            json.dumps(symbols),
            json.dumps(metrics),
        )

    async def query_backtest_results(self, run_id: str) -> list[dict]:
        rows = await self._db.fetch(
            "SELECT * FROM backtest_results WHERE run_id=$1",
            run_id,
        )
        return [dict(r) for r in rows]
