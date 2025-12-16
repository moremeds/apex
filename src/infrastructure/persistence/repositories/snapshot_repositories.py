"""
Repositories for position, account, and risk snapshot persistence.

Provides warm-start capability by persisting snapshots for recovery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from asyncpg import Record

from src.infrastructure.persistence.database import Database
from src.infrastructure.persistence.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


# =============================================================================
# Position Snapshot Repository
# =============================================================================


@dataclass
class PositionSnapshot:
    """Position snapshot entity for warm-start."""

    snapshot_time: datetime
    broker: str
    account_id: str
    positions: List[Dict[str, Any]]  # Array of Position objects
    position_count: Optional[int] = None
    id: Optional[int] = None


class PositionSnapshotRepository(BaseRepository[PositionSnapshot]):
    """Repository for position snapshots (warm-start)."""

    def __init__(self, db: Database):
        super().__init__(db)

    @property
    def table_name(self) -> str:
        return "position_snapshots"

    @property
    def conflict_columns(self) -> List[str]:
        return ["snapshot_time", "broker", "account_id"]

    def _to_entity(self, record: Record) -> PositionSnapshot:
        """Convert database record to PositionSnapshot entity."""
        return PositionSnapshot(
            id=record["id"],
            snapshot_time=record["snapshot_time"],
            broker=record["broker"],
            account_id=record["account_id"],
            positions=self._from_json(record["positions"]),
            position_count=record["position_count"],
        )

    def _to_row(self, entity: PositionSnapshot) -> Dict[str, Any]:
        """Convert PositionSnapshot entity to database row."""
        return {
            "snapshot_time": entity.snapshot_time,
            "broker": entity.broker,
            "account_id": entity.account_id,
            "positions": self._to_json(entity.positions),
            "position_count": entity.position_count or len(entity.positions),
        }

    async def get_latest(
        self,
        broker: str,
        account_id: str,
    ) -> Optional[PositionSnapshot]:
        """Get the most recent position snapshot."""
        query = """
            SELECT * FROM position_snapshots
            WHERE broker = $1 AND account_id = $2
            ORDER BY snapshot_time DESC
            LIMIT 1
        """
        record = await self._db.fetchrow(query, broker, account_id)
        return self._to_entity(record) if record else None

    async def get_by_time_range(
        self,
        broker: str,
        account_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[PositionSnapshot]:
        """Get snapshots within a time range."""
        query = """
            SELECT * FROM position_snapshots
            WHERE broker = $1 AND account_id = $2
              AND snapshot_time >= $3 AND snapshot_time <= $4
            ORDER BY snapshot_time DESC
        """
        records = await self._db.fetch(query, broker, account_id, start_time, end_time)
        return [self._to_entity(r) for r in records]

    async def cleanup_old(
        self,
        retention_days: int = 365,
    ) -> int:
        """Delete snapshots older than retention period."""
        query = """
            DELETE FROM position_snapshots
            WHERE snapshot_time < NOW() - INTERVAL '%s days'
        """
        result = await self._db.execute(query % retention_days)
        try:
            return int(result.split()[-1])
        except (IndexError, ValueError):
            return 0


# =============================================================================
# Account Snapshot Repository
# =============================================================================


@dataclass
class AccountSnapshot:
    """Account snapshot entity for warm-start."""

    snapshot_time: datetime
    broker: str
    account_id: str
    account_data: Dict[str, Any]  # AccountInfo object
    id: Optional[int] = None


class AccountSnapshotRepository(BaseRepository[AccountSnapshot]):
    """Repository for account snapshots (warm-start)."""

    def __init__(self, db: Database):
        super().__init__(db)

    @property
    def table_name(self) -> str:
        return "account_snapshots"

    @property
    def conflict_columns(self) -> List[str]:
        return ["snapshot_time", "broker", "account_id"]

    def _to_entity(self, record: Record) -> AccountSnapshot:
        """Convert database record to AccountSnapshot entity."""
        return AccountSnapshot(
            id=record["id"],
            snapshot_time=record["snapshot_time"],
            broker=record["broker"],
            account_id=record["account_id"],
            account_data=self._from_json(record["account_data"]),
        )

    def _to_row(self, entity: AccountSnapshot) -> Dict[str, Any]:
        """Convert AccountSnapshot entity to database row."""
        return {
            "snapshot_time": entity.snapshot_time,
            "broker": entity.broker,
            "account_id": entity.account_id,
            "account_data": self._to_json(entity.account_data),
        }

    async def get_latest(
        self,
        broker: str,
        account_id: str,
    ) -> Optional[AccountSnapshot]:
        """Get the most recent account snapshot."""
        query = """
            SELECT * FROM account_snapshots
            WHERE broker = $1 AND account_id = $2
            ORDER BY snapshot_time DESC
            LIMIT 1
        """
        record = await self._db.fetchrow(query, broker, account_id)
        return self._to_entity(record) if record else None

    async def get_by_time_range(
        self,
        broker: str,
        account_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[AccountSnapshot]:
        """Get snapshots within a time range."""
        query = """
            SELECT * FROM account_snapshots
            WHERE broker = $1 AND account_id = $2
              AND snapshot_time >= $3 AND snapshot_time <= $4
            ORDER BY snapshot_time DESC
        """
        records = await self._db.fetch(query, broker, account_id, start_time, end_time)
        return [self._to_entity(r) for r in records]

    async def cleanup_old(
        self,
        retention_days: int = 365,
    ) -> int:
        """Delete snapshots older than retention period."""
        query = """
            DELETE FROM account_snapshots
            WHERE snapshot_time < NOW() - INTERVAL '%s days'
        """
        result = await self._db.execute(query % retention_days)
        try:
            return int(result.split()[-1])
        except (IndexError, ValueError):
            return 0


# =============================================================================
# Risk Snapshot Repository
# =============================================================================


@dataclass
class RiskSnapshotRecord:
    """Risk snapshot entity for time-series history."""

    snapshot_time: datetime
    snapshot_data: Dict[str, Any]  # Full RiskSnapshot object
    portfolio_value: Optional[Decimal] = None
    total_delta: Optional[Decimal] = None
    total_gamma: Optional[Decimal] = None
    total_vega: Optional[Decimal] = None
    total_theta: Optional[Decimal] = None
    unrealized_pnl: Optional[Decimal] = None
    daily_pnl: Optional[Decimal] = None
    position_count: Optional[int] = None
    id: Optional[int] = None


class RiskSnapshotRepository(BaseRepository[RiskSnapshotRecord]):
    """Repository for risk snapshots (full history for analysis)."""

    def __init__(self, db: Database):
        super().__init__(db)

    @property
    def table_name(self) -> str:
        return "risk_snapshots"

    def _to_entity(self, record: Record) -> RiskSnapshotRecord:
        """Convert database record to RiskSnapshotRecord entity."""
        return RiskSnapshotRecord(
            id=record["id"],
            snapshot_time=record["snapshot_time"],
            snapshot_data=self._from_json(record["snapshot_data"]),
            portfolio_value=record["portfolio_value"],
            total_delta=record["total_delta"],
            total_gamma=record["total_gamma"],
            total_vega=record["total_vega"],
            total_theta=record["total_theta"],
            unrealized_pnl=record["unrealized_pnl"],
            daily_pnl=record["daily_pnl"],
            position_count=record["position_count"],
        )

    def _to_row(self, entity: RiskSnapshotRecord) -> Dict[str, Any]:
        """Convert RiskSnapshotRecord entity to database row."""
        return {
            "snapshot_time": entity.snapshot_time,
            "snapshot_data": self._to_json(entity.snapshot_data),
            "portfolio_value": entity.portfolio_value,
            "total_delta": entity.total_delta,
            "total_gamma": entity.total_gamma,
            "total_vega": entity.total_vega,
            "total_theta": entity.total_theta,
            "unrealized_pnl": entity.unrealized_pnl,
            "daily_pnl": entity.daily_pnl,
            "position_count": entity.position_count,
        }

    async def get_latest(self) -> Optional[RiskSnapshotRecord]:
        """Get the most recent risk snapshot."""
        query = """
            SELECT * FROM risk_snapshots
            ORDER BY snapshot_time DESC
            LIMIT 1
        """
        record = await self._db.fetchrow(query)
        return self._to_entity(record) if record else None

    async def get_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        limit: int = 1000,
    ) -> List[RiskSnapshotRecord]:
        """Get snapshots within a time range."""
        query = """
            SELECT * FROM risk_snapshots
            WHERE snapshot_time >= $1 AND snapshot_time <= $2
            ORDER BY snapshot_time DESC
            LIMIT $3
        """
        records = await self._db.fetch(query, start_time, end_time, limit)
        return [self._to_entity(r) for r in records]

    async def get_time_series(
        self,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Get downsampled time series for charting.

        Uses PostgreSQL time_bucket for efficient downsampling.
        """
        query = """
            SELECT
                time_bucket($1 * INTERVAL '1 minute', snapshot_time) AS bucket,
                AVG(portfolio_value)::numeric(16,2) as avg_portfolio_value,
                AVG(total_delta)::numeric(12,2) as avg_total_delta,
                AVG(unrealized_pnl)::numeric(14,2) as avg_unrealized_pnl,
                AVG(daily_pnl)::numeric(14,2) as avg_daily_pnl,
                MAX(position_count) as max_position_count
            FROM risk_snapshots
            WHERE snapshot_time >= $2 AND snapshot_time <= $3
            GROUP BY bucket
            ORDER BY bucket
        """
        records = await self._db.fetch(query, interval_minutes, start_time, end_time)

        return [
            {
                "time": r["bucket"],
                "portfolio_value": float(r["avg_portfolio_value"]) if r["avg_portfolio_value"] else None,
                "total_delta": float(r["avg_total_delta"]) if r["avg_total_delta"] else None,
                "unrealized_pnl": float(r["avg_unrealized_pnl"]) if r["avg_unrealized_pnl"] else None,
                "daily_pnl": float(r["avg_daily_pnl"]) if r["avg_daily_pnl"] else None,
                "position_count": r["max_position_count"],
            }
            for r in records
        ]

    async def get_daily_summary(
        self,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Get daily summary statistics for the past N days."""
        query = """
            SELECT
                date_trunc('day', snapshot_time) AS day,
                MIN(portfolio_value) as min_portfolio,
                MAX(portfolio_value) as max_portfolio,
                AVG(portfolio_value)::numeric(16,2) as avg_portfolio,
                MIN(unrealized_pnl) as min_unrealized_pnl,
                MAX(unrealized_pnl) as max_unrealized_pnl,
                MIN(daily_pnl) as min_daily_pnl,
                MAX(daily_pnl) as max_daily_pnl,
                COUNT(*) as snapshot_count
            FROM risk_snapshots
            WHERE snapshot_time >= NOW() - INTERVAL '%s days'
            GROUP BY day
            ORDER BY day DESC
        """
        records = await self._db.fetch(query % days)

        return [
            {
                "day": r["day"],
                "min_portfolio": float(r["min_portfolio"]) if r["min_portfolio"] else None,
                "max_portfolio": float(r["max_portfolio"]) if r["max_portfolio"] else None,
                "avg_portfolio": float(r["avg_portfolio"]) if r["avg_portfolio"] else None,
                "min_unrealized_pnl": float(r["min_unrealized_pnl"]) if r["min_unrealized_pnl"] else None,
                "max_unrealized_pnl": float(r["max_unrealized_pnl"]) if r["max_unrealized_pnl"] else None,
                "min_daily_pnl": float(r["min_daily_pnl"]) if r["min_daily_pnl"] else None,
                "max_daily_pnl": float(r["max_daily_pnl"]) if r["max_daily_pnl"] else None,
                "snapshot_count": r["snapshot_count"],
            }
            for r in records
        ]

    async def cleanup_old(
        self,
        retention_days: int = 365,
    ) -> int:
        """Delete snapshots older than retention period."""
        query = """
            DELETE FROM risk_snapshots
            WHERE snapshot_time < NOW() - INTERVAL '%s days'
        """
        result = await self._db.execute(query % retention_days)
        try:
            return int(result.split()[-1])
        except (IndexError, ValueError):
            return 0

    @classmethod
    def from_risk_snapshot(
        cls,
        snapshot: Any,  # RiskSnapshot from domain
        snapshot_time: datetime,
    ) -> RiskSnapshotRecord:
        """
        Convert a domain RiskSnapshot to a RiskSnapshotRecord for persistence.

        Maps domain model fields to database columns:
        - Domain: total_net_liquidation → DB: portfolio_value
        - Domain: portfolio_delta → DB: total_delta
        - Domain: portfolio_gamma → DB: total_gamma
        - Domain: portfolio_vega → DB: total_vega
        - Domain: portfolio_theta → DB: total_theta
        - Domain: total_unrealized_pnl → DB: unrealized_pnl
        - Domain: total_daily_pnl → DB: daily_pnl

        Args:
            snapshot: Domain RiskSnapshot object.
            snapshot_time: Time of the snapshot.

        Returns:
            RiskSnapshotRecord ready for persistence.
        """
        # Extract values using correct domain model field names
        portfolio_value = getattr(snapshot, "total_net_liquidation", None)
        total_delta = getattr(snapshot, "portfolio_delta", None)
        total_gamma = getattr(snapshot, "portfolio_gamma", None)
        total_vega = getattr(snapshot, "portfolio_vega", None)
        total_theta = getattr(snapshot, "portfolio_theta", None)
        unrealized_pnl = getattr(snapshot, "total_unrealized_pnl", None)
        daily_pnl = getattr(snapshot, "total_daily_pnl", None)
        position_risks = getattr(snapshot, "position_risks", [])

        # Serialize the full snapshot to dict
        snapshot_dict = {
            "timestamp": str(snapshot.timestamp) if hasattr(snapshot, "timestamp") else str(snapshot_time),
            "total_net_liquidation": float(portfolio_value) if portfolio_value else None,
            "total_unrealized_pnl": float(unrealized_pnl) if unrealized_pnl else None,
            "total_daily_pnl": float(daily_pnl) if daily_pnl else None,
            "portfolio_delta": float(total_delta) if total_delta else None,
            "portfolio_gamma": float(total_gamma) if total_gamma else None,
            "portfolio_vega": float(total_vega) if total_vega else None,
            "portfolio_theta": float(total_theta) if total_theta else None,
            "position_count": len(position_risks),
        }

        return RiskSnapshotRecord(
            snapshot_time=snapshot_time,
            snapshot_data=snapshot_dict,
            portfolio_value=Decimal(str(portfolio_value)) if portfolio_value else None,
            total_delta=Decimal(str(total_delta)) if total_delta else None,
            total_gamma=Decimal(str(total_gamma)) if total_gamma else None,
            total_vega=Decimal(str(total_vega)) if total_vega else None,
            total_theta=Decimal(str(total_theta)) if total_theta else None,
            unrealized_pnl=Decimal(str(unrealized_pnl)) if unrealized_pnl else None,
            daily_pnl=Decimal(str(daily_pnl)) if daily_pnl else None,
            position_count=len(position_risks),
        )
