"""
Repository for sync state tracking.

Tracks the last sync time and status for each broker/data_type/market
combination to support incremental data loading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from asyncpg import Record

from src.infrastructure.persistence.database import Database
from src.infrastructure.persistence.repositories.base import BaseRepository
from src.utils.timezone import now_utc

logger = logging.getLogger(__name__)


@dataclass
class SyncState:
    """Sync state entity for tracking incremental loads."""

    broker: str
    account_id: str
    data_type: str
    market: Optional[str]
    last_sync_time: Optional[datetime]
    last_record_time: Optional[datetime]
    records_synced: int
    records_total: int
    sync_from_date: Optional[date]
    sync_to_date: Optional[date]
    last_sync_status: Optional[str]
    last_error: Optional[str]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    id: Optional[int] = None


class SyncStateRepository(BaseRepository[SyncState]):
    """
    Repository for sync state tracking.

    Tracks the synchronization state for each broker/account/data_type/market
    combination. Used to support incremental data loading and resumption.
    """

    def __init__(self, db: Database):
        super().__init__(db)

    @property
    def table_name(self) -> str:
        return "sync_state"

    @property
    def conflict_columns(self) -> List[str]:
        # UNIQUE constraint on (broker, account_id, data_type, market)
        return ["broker", "account_id", "data_type", "market"]

    def _to_entity(self, record: Record) -> SyncState:
        """Convert database record to SyncState entity."""
        return SyncState(
            id=record["id"],
            broker=record["broker"],
            account_id=record["account_id"],
            data_type=record["data_type"],
            market=record["market"],
            last_sync_time=record["last_sync_time"],
            last_record_time=record["last_record_time"],
            records_synced=record["records_synced"],
            records_total=record["records_total"],
            sync_from_date=record["sync_from_date"],
            sync_to_date=record["sync_to_date"],
            last_sync_status=record["last_sync_status"],
            last_error=record["last_error"],
            created_at=record["created_at"],
            updated_at=record["updated_at"],
        )

    def _to_row(self, entity: SyncState) -> Dict[str, Any]:
        """Convert SyncState entity to database row."""
        return {
            "broker": entity.broker,
            "account_id": entity.account_id,
            "data_type": entity.data_type,
            "market": entity.market or "",  # Convert None to '' for UNIQUE constraint
            "last_sync_time": entity.last_sync_time,
            "last_record_time": entity.last_record_time,
            "records_synced": entity.records_synced,
            "records_total": entity.records_total,
            "sync_from_date": entity.sync_from_date,
            "sync_to_date": entity.sync_to_date,
            "last_sync_status": entity.last_sync_status,
            "last_error": entity.last_error,
            "updated_at": now_utc(),
        }

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    async def get_state(
        self,
        broker: str,
        account_id: str,
        data_type: str,
        market: Optional[str] = None,
    ) -> Optional[SyncState]:
        """
        Get sync state for a specific broker/account/data_type/market.

        Args:
            broker: Broker identifier ('IB' or 'FUTU').
            account_id: Broker account ID.
            data_type: Data type being synced ('futu_orders', 'ib_executions', etc.).
            market: Optional market code (US, HK, CN) - used for Futu.

        Returns:
            SyncState if found, None otherwise.
        """
        # Use empty string for None market to match UNIQUE constraint behavior
        market_value = market or ""
        query = """
            SELECT * FROM sync_state
            WHERE broker = $1 AND account_id = $2 AND data_type = $3 AND market = $4
        """
        record = await self._db.fetchrow(query, broker, account_id, data_type, market_value)

        return self._to_entity(record) if record else None

    async def get_all_states(
        self,
        broker: Optional[str] = None,
        account_id: Optional[str] = None,
    ) -> List[SyncState]:
        """
        Get all sync states, optionally filtered by broker/account.

        Args:
            broker: Optional broker filter.
            account_id: Optional account filter.

        Returns:
            List of sync states.
        """
        conditions = []
        params = []
        param_idx = 1

        if broker:
            conditions.append(f"broker = ${param_idx}")
            params.append(broker)
            param_idx += 1

        if account_id:
            conditions.append(f"account_id = ${param_idx}")
            params.append(account_id)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        query = f"""
            SELECT * FROM sync_state
            {where_clause}
            ORDER BY broker, account_id, data_type, market
        """
        records = await self._db.fetch(query, *params)
        return [self._to_entity(r) for r in records]

    async def get_last_record_time(
        self,
        broker: str,
        account_id: str,
        data_type: str,
        market: Optional[str] = None,
    ) -> Optional[datetime]:
        """
        Get the last record timestamp for incremental sync.

        Args:
            broker: Broker identifier.
            account_id: Broker account ID.
            data_type: Data type being synced.
            market: Optional market code.

        Returns:
            Last record timestamp, or None if no sync yet.
        """
        state = await self.get_state(broker, account_id, data_type, market)
        return state.last_record_time if state else None

    # -------------------------------------------------------------------------
    # Update Methods
    # -------------------------------------------------------------------------

    async def update_sync_start(
        self,
        broker: str,
        account_id: str,
        data_type: str,
        market: Optional[str] = None,
        sync_from_date: Optional[date] = None,
        sync_to_date: Optional[date] = None,
    ) -> SyncState:
        """
        Record the start of a sync operation.

        Args:
            broker: Broker identifier.
            account_id: Broker account ID.
            data_type: Data type being synced.
            market: Optional market code.
            sync_from_date: Start date for the sync period.
            sync_to_date: End date for the sync period.

        Returns:
            Updated SyncState.
        """
        state = SyncState(
            broker=broker,
            account_id=account_id,
            data_type=data_type,
            market=market,
            last_sync_time=now_utc(),
            last_record_time=None,
            records_synced=0,
            records_total=0,
            sync_from_date=sync_from_date,
            sync_to_date=sync_to_date,
            last_sync_status="IN_PROGRESS",
            last_error=None,
        )
        return await self.upsert(state)

    async def update_sync_progress(
        self,
        broker: str,
        account_id: str,
        data_type: str,
        records_synced: int,
        records_total: int,
        last_record_time: Optional[datetime] = None,
        market: Optional[str] = None,
    ) -> SyncState:
        """
        Update sync progress during an ongoing sync.

        Args:
            broker: Broker identifier.
            account_id: Broker account ID.
            data_type: Data type being synced.
            records_synced: Number of records synced so far.
            records_total: Total records to sync.
            last_record_time: Timestamp of the most recent record synced.
            market: Optional market code.

        Returns:
            Updated SyncState.
        """
        # Get existing state
        state = await self.get_state(broker, account_id, data_type, market)

        if state:
            state.records_synced = records_synced
            state.records_total = records_total
            if last_record_time:
                state.last_record_time = last_record_time
            state.last_sync_status = "IN_PROGRESS"
            return await self.upsert(state)
        else:
            # Create new state if doesn't exist
            return await self.update_sync_start(broker, account_id, data_type, market)

    async def update_sync_complete(
        self,
        broker: str,
        account_id: str,
        data_type: str,
        records_synced: int,
        last_record_time: Optional[datetime] = None,
        market: Optional[str] = None,
    ) -> SyncState:
        """
        Mark a sync operation as complete.

        Args:
            broker: Broker identifier.
            account_id: Broker account ID.
            data_type: Data type being synced.
            records_synced: Final count of records synced.
            last_record_time: Timestamp of the most recent record synced.
            market: Optional market code.

        Returns:
            Updated SyncState.
        """
        state = await self.get_state(broker, account_id, data_type, market)

        if state:
            state.records_synced = records_synced
            state.records_total = records_synced
            state.last_sync_time = now_utc()
            if last_record_time:
                state.last_record_time = last_record_time
            state.last_sync_status = "COMPLETED"
            state.last_error = None
            return await self.upsert(state)
        else:
            # Create new state
            new_state = SyncState(
                broker=broker,
                account_id=account_id,
                data_type=data_type,
                market=market,
                last_sync_time=now_utc(),
                last_record_time=last_record_time,
                records_synced=records_synced,
                records_total=records_synced,
                sync_from_date=None,
                sync_to_date=None,
                last_sync_status="COMPLETED",
                last_error=None,
            )
            return await self.upsert(new_state)

    async def update_sync_failed(
        self,
        broker: str,
        account_id: str,
        data_type: str,
        error_message: str,
        market: Optional[str] = None,
    ) -> SyncState:
        """
        Mark a sync operation as failed.

        Args:
            broker: Broker identifier.
            account_id: Broker account ID.
            data_type: Data type being synced.
            error_message: Error description.
            market: Optional market code.

        Returns:
            Updated SyncState.
        """
        state = await self.get_state(broker, account_id, data_type, market)

        if state:
            state.last_sync_time = now_utc()
            state.last_sync_status = "FAILED"
            state.last_error = error_message[:500]  # Truncate long errors
            return await self.upsert(state)
        else:
            new_state = SyncState(
                broker=broker,
                account_id=account_id,
                data_type=data_type,
                market=market,
                last_sync_time=now_utc(),
                last_record_time=None,
                records_synced=0,
                records_total=0,
                sync_from_date=None,
                sync_to_date=None,
                last_sync_status="FAILED",
                last_error=error_message[:500],
            )
            return await self.upsert(new_state)

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    async def get_sync_summary(self) -> List[Dict[str, Any]]:
        """
        Get a summary of all sync states for monitoring.

        Returns:
            List of dicts with sync state summaries.
        """
        query = """
            SELECT
                broker,
                account_id,
                data_type,
                market,
                last_sync_time,
                last_record_time,
                records_synced,
                last_sync_status,
                CASE WHEN last_error IS NOT NULL THEN LEFT(last_error, 100) ELSE NULL END as last_error
            FROM sync_state
            ORDER BY broker, account_id, data_type, market
        """
        records = await self._db.fetch(query)

        return [
            {
                "broker": r["broker"],
                "account_id": r["account_id"],
                "data_type": r["data_type"],
                "market": r["market"],
                "last_sync_time": r["last_sync_time"],
                "last_record_time": r["last_record_time"],
                "records_synced": r["records_synced"],
                "last_sync_status": r["last_sync_status"],
                "last_error": r["last_error"],
            }
            for r in records
        ]

    async def needs_sync(
        self,
        broker: str,
        account_id: str,
        data_type: str,
        market: Optional[str] = None,
        max_age_hours: float = 24.0,
    ) -> bool:
        """
        Check if data needs to be synced based on last sync time.

        Args:
            broker: Broker identifier.
            account_id: Broker account ID.
            data_type: Data type being synced.
            market: Optional market code.
            max_age_hours: Maximum age in hours before sync needed.

        Returns:
            True if sync is needed, False otherwise.
        """
        state = await self.get_state(broker, account_id, data_type, market)

        if not state or not state.last_sync_time:
            return True

        age_hours = (now_utc() - state.last_sync_time).total_seconds() / 3600
        return age_hours > max_age_hours

    async def clear_state(
        self,
        broker: str,
        account_id: str,
        data_type: Optional[str] = None,
        market: Optional[str] = None,
    ) -> int:
        """
        Clear sync state records.

        Args:
            broker: Broker identifier.
            account_id: Broker account ID.
            data_type: Optional data type filter.
            market: Optional market filter.

        Returns:
            Number of records deleted.
        """
        conditions = ["broker = $1", "account_id = $2"]
        params = [broker, account_id]
        param_idx = 3

        if data_type:
            conditions.append(f"data_type = ${param_idx}")
            params.append(data_type)
            param_idx += 1

        if market:
            conditions.append(f"market = ${param_idx}")
            params.append(market)

        query = f"DELETE FROM sync_state WHERE {' AND '.join(conditions)}"
        result = await self._db.execute(query, *params)

        try:
            return int(result.split()[-1])
        except (IndexError, ValueError):
            return 0
