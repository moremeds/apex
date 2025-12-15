"""
Repository for backtest results persistence.

Stores backtest runs with parameters, metrics, and trade history.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from asyncpg import Record

from src.infrastructure.persistence.database import Database
from src.infrastructure.persistence.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


@dataclass
class Backtest:
    """Backtest result entity."""

    name: str
    strategy: str
    start_date: date
    end_date: date
    total_return: Optional[Decimal] = None
    sharpe_ratio: Optional[Decimal] = None
    max_drawdown: Optional[Decimal] = None
    win_rate: Optional[Decimal] = None
    parameters: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    trades: Optional[List[Dict[str, Any]]] = None
    equity_curve: Optional[List[Dict[str, Any]]] = None
    status: str = "completed"
    notes: Optional[str] = None
    backtest_id: Optional[uuid.UUID] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    id: Optional[int] = None


class BacktestRepository(BaseRepository[Backtest]):
    """Repository for backtest results."""

    def __init__(self, db: Database):
        super().__init__(db)

    @property
    def table_name(self) -> str:
        return "backtests"

    def _to_entity(self, record: Record) -> Backtest:
        """Convert database record to Backtest entity."""
        return Backtest(
            id=record["id"],
            backtest_id=record["backtest_id"],
            name=record["name"],
            strategy=record["strategy"],
            start_date=record["start_date"],
            end_date=record["end_date"],
            total_return=record["total_return"],
            sharpe_ratio=record["sharpe_ratio"],
            max_drawdown=record["max_drawdown"],
            win_rate=record["win_rate"],
            parameters=self._from_json(record["parameters"]),
            metrics=self._from_json(record["metrics"]),
            trades=self._from_json(record["trades"]),
            equity_curve=self._from_json(record["equity_curve"]),
            status=record["status"],
            notes=record["notes"],
            created_at=record["created_at"],
            updated_at=record["updated_at"],
        )

    def _to_row(self, entity: Backtest) -> Dict[str, Any]:
        """Convert Backtest entity to database row."""
        return {
            "name": entity.name,
            "strategy": entity.strategy,
            "start_date": entity.start_date,
            "end_date": entity.end_date,
            "total_return": entity.total_return,
            "sharpe_ratio": entity.sharpe_ratio,
            "max_drawdown": entity.max_drawdown,
            "win_rate": entity.win_rate,
            "parameters": self._to_json(entity.parameters),
            "metrics": self._to_json(entity.metrics),
            "trades": self._to_json(entity.trades),
            "equity_curve": self._to_json(entity.equity_curve),
            "status": entity.status,
            "notes": entity.notes,
        }

    async def find_by_strategy(
        self,
        strategy: str,
        limit: int = 100,
    ) -> List[Backtest]:
        """Find backtests by strategy name."""
        query = """
            SELECT * FROM backtests
            WHERE strategy = $1
            ORDER BY created_at DESC
            LIMIT $2
        """
        records = await self._db.fetch(query, strategy, limit)
        return [self._to_entity(r) for r in records]

    async def find_by_date_range(
        self,
        start_date: date,
        end_date: date,
        strategy: Optional[str] = None,
    ) -> List[Backtest]:
        """Find backtests that overlap with a date range."""
        if strategy:
            query = """
                SELECT * FROM backtests
                WHERE strategy = $1
                  AND start_date <= $3
                  AND end_date >= $2
                ORDER BY created_at DESC
            """
            records = await self._db.fetch(query, strategy, start_date, end_date)
        else:
            query = """
                SELECT * FROM backtests
                WHERE start_date <= $2
                  AND end_date >= $1
                ORDER BY created_at DESC
            """
            records = await self._db.fetch(query, start_date, end_date)

        return [self._to_entity(r) for r in records]

    async def find_by_backtest_id(
        self, backtest_id: uuid.UUID
    ) -> Optional[Backtest]:
        """Find backtest by UUID."""
        query = """
            SELECT * FROM backtests
            WHERE backtest_id = $1
        """
        record = await self._db.fetchrow(query, backtest_id)
        return self._to_entity(record) if record else None

    async def find_top_performers(
        self,
        strategy: Optional[str] = None,
        metric: str = "sharpe_ratio",
        limit: int = 10,
    ) -> List[Backtest]:
        """Find top performing backtests by a metric."""
        # Validate metric to prevent SQL injection
        valid_metrics = ["sharpe_ratio", "total_return", "win_rate", "max_drawdown"]
        if metric not in valid_metrics:
            metric = "sharpe_ratio"

        # For max_drawdown, lower is better (sort ASC)
        order = "ASC" if metric == "max_drawdown" else "DESC"

        if strategy:
            query = f"""
                SELECT * FROM backtests
                WHERE strategy = $1 AND {metric} IS NOT NULL
                ORDER BY {metric} {order}
                LIMIT $2
            """
            records = await self._db.fetch(query, strategy, limit)
        else:
            query = f"""
                SELECT * FROM backtests
                WHERE {metric} IS NOT NULL
                ORDER BY {metric} {order}
                LIMIT $1
            """
            records = await self._db.fetch(query, limit)

        return [self._to_entity(r) for r in records]

    async def get_strategy_summary(
        self, strategy: str
    ) -> Dict[str, Any]:
        """Get summary statistics for a strategy's backtests."""
        query = """
            SELECT
                COUNT(*) as run_count,
                AVG(total_return) as avg_return,
                AVG(sharpe_ratio) as avg_sharpe,
                AVG(max_drawdown) as avg_drawdown,
                AVG(win_rate) as avg_win_rate,
                MAX(sharpe_ratio) as best_sharpe,
                MIN(max_drawdown) as best_drawdown,
                MAX(total_return) as best_return
            FROM backtests
            WHERE strategy = $1 AND status = 'completed'
        """
        record = await self._db.fetchrow(query, strategy)

        return {
            "strategy": strategy,
            "run_count": record["run_count"] or 0,
            "avg_return": float(record["avg_return"]) if record["avg_return"] else None,
            "avg_sharpe": float(record["avg_sharpe"]) if record["avg_sharpe"] else None,
            "avg_drawdown": float(record["avg_drawdown"]) if record["avg_drawdown"] else None,
            "avg_win_rate": float(record["avg_win_rate"]) if record["avg_win_rate"] else None,
            "best_sharpe": float(record["best_sharpe"]) if record["best_sharpe"] else None,
            "best_drawdown": float(record["best_drawdown"]) if record["best_drawdown"] else None,
            "best_return": float(record["best_return"]) if record["best_return"] else None,
        }

    async def get_all_strategies(self) -> List[str]:
        """Get list of all strategies with backtests."""
        query = """
            SELECT DISTINCT strategy
            FROM backtests
            ORDER BY strategy
        """
        records = await self._db.fetch(query)
        return [r["strategy"] for r in records]

    async def update_status(
        self,
        backtest_id: int,
        status: str,
        notes: Optional[str] = None,
    ) -> bool:
        """Update backtest status."""
        from src.utils.timezone import now_utc

        if notes:
            query = """
                UPDATE backtests
                SET status = $2, notes = $3, updated_at = $4
                WHERE id = $1
            """
            result = await self._db.execute(query, backtest_id, status, notes, now_utc())
        else:
            query = """
                UPDATE backtests
                SET status = $2, updated_at = $3
                WHERE id = $1
            """
            result = await self._db.execute(query, backtest_id, status, now_utc())

        return "UPDATE 1" in result

    async def update_metrics(
        self,
        backtest_id: int,
        metrics: Dict[str, Any],
    ) -> bool:
        """Update backtest metrics."""
        from src.utils.timezone import now_utc

        query = """
            UPDATE backtests
            SET metrics = $2, updated_at = $3
            WHERE id = $1
        """
        result = await self._db.execute(
            query, backtest_id, self._to_json(metrics), now_utc()
        )
        return "UPDATE 1" in result

    async def delete_old_backtests(
        self,
        strategy: str,
        keep_count: int = 100,
    ) -> int:
        """Delete old backtests, keeping the most recent ones."""
        query = """
            DELETE FROM backtests
            WHERE strategy = $1
              AND id NOT IN (
                  SELECT id FROM backtests
                  WHERE strategy = $1
                  ORDER BY created_at DESC
                  LIMIT $2
              )
        """
        result = await self._db.execute(query, strategy, keep_count)

        try:
            return int(result.split()[-1])
        except (IndexError, ValueError):
            return 0
