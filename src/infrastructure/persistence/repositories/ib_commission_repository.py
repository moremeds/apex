"""
Repository for IB raw commissions persistence.

Handles UPSERT operations for commission reports from IB's CommissionReport events.
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


@dataclass
class IbRawCommission:
    """IB raw commission entity."""

    exec_id: str
    account_id: str
    commission: Decimal
    currency: Optional[str]
    realized_pnl: Optional[Decimal]
    yield_: Optional[Decimal]
    yield_redemption_date: Optional[str]
    raw_data: Optional[Dict[str, Any]] = None
    loaded_at: Optional[datetime] = None
    id: Optional[int] = None


class IbCommissionRepository(BaseRepository[IbRawCommission]):
    """
    Repository for IB raw commissions.

    Handles persistence of commission data from IB's CommissionReport events.
    Uses UPSERT pattern with (exec_id, account_id) as the conflict key.
    """

    def __init__(self, db: Database):
        super().__init__(db)

    @property
    def table_name(self) -> str:
        return "ib_raw_commissions"

    @property
    def conflict_columns(self) -> List[str]:
        return ["exec_id", "account_id"]

    def _to_entity(self, record: Record) -> IbRawCommission:
        """Convert database record to IbRawCommission entity."""
        return IbRawCommission(
            id=record["id"],
            exec_id=record["exec_id"],
            account_id=record["account_id"],
            commission=record["commission"],
            currency=record["currency"],
            realized_pnl=record["realized_pnl"],
            yield_=record["yield_"],
            yield_redemption_date=record["yield_redemption_date"],
            raw_data=self._from_json(record["raw_data"]),
            loaded_at=record["loaded_at"],
        )

    def _to_row(self, entity: IbRawCommission) -> Dict[str, Any]:
        """Convert IbRawCommission entity to database row."""
        return {
            "exec_id": entity.exec_id,
            "account_id": entity.account_id,
            "commission": entity.commission,
            "currency": entity.currency,
            "realized_pnl": entity.realized_pnl,
            "yield_": entity.yield_,
            "yield_redemption_date": entity.yield_redemption_date,
            "raw_data": self._to_json(entity.raw_data),
        }

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    async def find_by_exec_id(self, exec_id: str, account_id: str) -> Optional[IbRawCommission]:
        """
        Find commission by exec_id and account_id.

        Args:
            exec_id: IB execution ID.
            account_id: IB account ID.

        Returns:
            IbRawCommission if found, None otherwise.
        """
        query = """
            SELECT * FROM ib_raw_commissions
            WHERE exec_id = $1 AND account_id = $2
        """
        record = await self._db.fetchrow(query, exec_id, account_id)
        return self._to_entity(record) if record else None

    async def find_by_account(self, account_id: str, limit: int = 1000) -> List[IbRawCommission]:
        """
        Find commission records by account.

        Args:
            account_id: IB account ID.
            limit: Maximum number of records.

        Returns:
            List of commission records.
        """
        query = """
            SELECT * FROM ib_raw_commissions
            WHERE account_id = $1
            ORDER BY loaded_at DESC
            LIMIT $2
        """
        records = await self._db.fetch(query, account_id, limit)
        return [self._to_entity(r) for r in records]

    async def find_by_exec_ids(self, exec_ids: List[str], account_id: str) -> List[IbRawCommission]:
        """
        Find commission records for multiple executions.

        Args:
            exec_ids: List of IB execution IDs.
            account_id: IB account ID.

        Returns:
            List of commission records.
        """
        if not exec_ids:
            return []

        query = """
            SELECT * FROM ib_raw_commissions
            WHERE account_id = $1 AND exec_id = ANY($2)
        """
        records = await self._db.fetch(query, account_id, exec_ids)
        return [self._to_entity(r) for r in records]

    async def get_total_commissions(
        self,
        account_id: str,
        exec_ids: Optional[List[str]] = None,
    ) -> Dict[str, Decimal]:
        """
        Calculate total commissions and realized P&L.

        Args:
            account_id: IB account ID.
            exec_ids: Optional list of execution IDs to filter.

        Returns:
            Dictionary with 'total_commission' and 'total_realized_pnl'.
        """
        if exec_ids:
            query = """
                SELECT
                    COALESCE(SUM(commission), 0) as total_commission,
                    COALESCE(SUM(realized_pnl), 0) as total_realized_pnl
                FROM ib_raw_commissions
                WHERE account_id = $1 AND exec_id = ANY($2)
            """
            record = await self._db.fetchrow(query, account_id, exec_ids)
        else:
            query = """
                SELECT
                    COALESCE(SUM(commission), 0) as total_commission,
                    COALESCE(SUM(realized_pnl), 0) as total_realized_pnl
                FROM ib_raw_commissions
                WHERE account_id = $1
            """
            record = await self._db.fetchrow(query, account_id)

        if record is None:
            return {
                "total_commission": Decimal(0),
                "total_realized_pnl": Decimal(0),
            }

        return {
            "total_commission": record["total_commission"] or Decimal(0),
            "total_realized_pnl": record["total_realized_pnl"] or Decimal(0),
        }

    async def get_missing_commission_exec_ids(
        self,
        account_id: str,
        exec_ids: List[str],
    ) -> List[str]:
        """
        Find execution IDs that don't have commission records.

        Used to identify executions missing commission data.

        Args:
            account_id: IB account ID.
            exec_ids: List of execution IDs to check.

        Returns:
            List of execution IDs without commission records.
        """
        if not exec_ids:
            return []

        query = """
            SELECT exec_id FROM ib_raw_commissions
            WHERE account_id = $1 AND exec_id = ANY($2)
        """
        records = await self._db.fetch(query, account_id, exec_ids)
        existing_ids = {r["exec_id"] for r in records}

        return [eid for eid in exec_ids if eid not in existing_ids]

    async def get_commissions_with_executions(
        self,
        account_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get commissions joined with execution details.

        Args:
            account_id: IB account ID.
            limit: Maximum number of records.

        Returns:
            List of dicts with commission and execution data.
        """
        query = """
            SELECT
                c.exec_id,
                c.commission,
                c.currency,
                c.realized_pnl,
                e.symbol,
                e.sec_type,
                e.side,
                e.shares,
                e.price,
                e.exec_time
            FROM ib_raw_commissions c
            JOIN ib_raw_executions e ON c.exec_id = e.exec_id AND c.account_id = e.account_id
            WHERE c.account_id = $1
            ORDER BY e.exec_time DESC
            LIMIT $2
        """
        records = await self._db.fetch(query, account_id, limit)

        return [
            {
                "exec_id": r["exec_id"],
                "commission": r["commission"],
                "currency": r["currency"],
                "realized_pnl": r["realized_pnl"],
                "symbol": r["symbol"],
                "sec_type": r["sec_type"],
                "side": r["side"],
                "shares": r["shares"],
                "price": r["price"],
                "exec_time": r["exec_time"],
            }
            for r in records
        ]

    # -------------------------------------------------------------------------
    # Conversion from IB API
    # -------------------------------------------------------------------------

    @classmethod
    def from_ib_commission_report(
        cls,
        commission_report: Any,
        account_id: str,
    ) -> IbRawCommission:
        """
        Convert IB API CommissionReport to IbRawCommission entity.

        Args:
            commission_report: ib_async CommissionReport object.
            account_id: IB account ID.

        Returns:
            IbRawCommission entity.
        """
        # Build raw data dict for preservation
        raw_data = {
            "execId": commission_report.execId,
            "commission": float(commission_report.commission),
            "currency": commission_report.currency,
            "realizedPNL": (
                float(commission_report.realizedPNL)
                if commission_report.realizedPNL is not None
                else None
            ),
            "yield_": (
                float(commission_report.yield_) if commission_report.yield_ is not None else None
            ),
            "yieldRedemptionDate": commission_report.yieldRedemptionDate,
        }

        return IbRawCommission(
            exec_id=commission_report.execId,
            account_id=account_id,
            commission=Decimal(str(commission_report.commission)),
            currency=commission_report.currency,
            realized_pnl=(
                Decimal(str(commission_report.realizedPNL))
                if commission_report.realizedPNL is not None
                else None
            ),
            yield_=(
                Decimal(str(commission_report.yield_))
                if commission_report.yield_ is not None
                else None
            ),
            yield_redemption_date=commission_report.yieldRedemptionDate,
            raw_data=raw_data,
        )
