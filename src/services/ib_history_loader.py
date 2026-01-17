"""
IB History Loader - Load executions and commissions from Interactive Brokers.

Note: IB API only provides current-day executions via reqExecutions().
Historical data requires IB FLEX reports (deferred to v1.2).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from config.models import IbConfig
from src.infrastructure.persistence.repositories import (
    IbCommissionRepository,
    IbExecutionRepository,
    IbRawCommission,
    IbRawExecution,
    SyncStateRepository,
)
from src.utils.timezone import now_utc

logger = logging.getLogger(__name__)


class IbHistoryLoader:
    """
    Loader for IB historical data.

    Handles:
    - Executions from reqExecutions()
    - Commissions from CommissionReport events

    Limitations:
    - IB API only provides current-day executions
    - Historical data requires FLEX reports (deferred to v1.2)
    - Forward capture: System captures executions going forward
    """

    def __init__(
        self,
        exec_repo: IbExecutionRepository,
        comm_repo: IbCommissionRepository,
        sync_state_repo: SyncStateRepository,
        ib_config: IbConfig,
        batch_size: int = 100,
        dry_run: bool = False,
    ):
        """
        Initialize IB history loader.

        Args:
            exec_repo: Repository for executions.
            comm_repo: Repository for commissions.
            sync_state_repo: Repository for sync state tracking.
            ib_config: IB configuration.
            batch_size: Records per batch insert.
            dry_run: If True, don't write to database.
        """
        self._exec_repo = exec_repo
        self._comm_repo = comm_repo
        self._sync_state_repo = sync_state_repo
        self._ib_config = ib_config
        self._batch_size = batch_size
        self._dry_run = dry_run

        self._ib_client = None  # Lazy initialization

    async def _get_ib_client(self):
        """Get or create IB client connection."""
        if self._ib_client is None:
            try:
                from ib_async import IB

                self._ib_client = IB()
                await self._ib_client.connectAsync(
                    host=self._ib_config.host,
                    port=self._ib_config.port,
                    clientId=self._ib_config.client_ids.historical_pool[0],
                )
            except ImportError:
                logger.error("ib_async package not installed")
                raise RuntimeError("ib_async package required for IB history loading")

        return self._ib_client

    async def load_executions(
        self,
        account_id: str,
    ) -> int:
        """
        Load executions from IB.

        Note: IB only provides current-day executions via API.

        Args:
            account_id: IB account ID.

        Returns:
            Number of executions loaded.
        """
        logger.info(f"Loading IB executions for {account_id}")

        await self._sync_state_repo.update_sync_start(
            broker="IB",
            account_id=account_id,
            data_type="ib_executions",
        )

        try:
            executions = await self._fetch_executions(account_id)
            logger.info(f"Fetched {len(executions)} executions from IB API")

            if self._dry_run:
                logger.info(f"DRY RUN: Would insert {len(executions)} executions")
                return len(executions)

            if not executions:
                await self._sync_state_repo.update_sync_complete(
                    broker="IB",
                    account_id=account_id,
                    data_type="ib_executions",
                    records_synced=0,
                )
                return 0

            # Check for existing exec_ids to avoid duplicates
            existing_ids = await self._exec_repo.get_exec_ids(account_id)
            existing_ids_set = set(existing_ids)

            # Filter to new executions only
            new_executions = [e for e in executions if e["exec_id"] not in existing_ids_set]

            if not new_executions:
                logger.info("No new executions to load")
                await self._sync_state_repo.update_sync_complete(
                    broker="IB",
                    account_id=account_id,
                    data_type="ib_executions",
                    records_synced=0,
                )
                return 0

            # Convert to entities
            entities = [
                IbExecutionRepository.from_ib_execution(e["execution"], e["contract"], account_id)
                for e in new_executions
            ]

            # Batch upsert
            total_inserted = 0
            for i in range(0, len(entities), self._batch_size):
                batch = entities[i : i + self._batch_size]
                await self._exec_repo.upsert_many(batch)
                total_inserted += len(batch)

            # Get last exec time
            last_time = None
            if entities:
                last_time = max(e.exec_time for e in entities if e.exec_time)

            await self._sync_state_repo.update_sync_complete(
                broker="IB",
                account_id=account_id,
                data_type="ib_executions",
                records_synced=total_inserted,
                last_record_time=last_time,
            )

            logger.info(f"Loaded {total_inserted} executions for {account_id}")
            return total_inserted

        except Exception as e:
            await self._sync_state_repo.update_sync_failed(
                broker="IB",
                account_id=account_id,
                data_type="ib_executions",
                error_message=str(e),
            )
            raise

    async def load_commissions(
        self,
        account_id: str,
    ) -> int:
        """
        Load commissions for existing executions.

        Note: Commission reports are received asynchronously from IB
        after execution. This method processes any pending reports.

        Args:
            account_id: IB account ID.

        Returns:
            Number of commission records loaded.
        """
        logger.info(f"Loading IB commissions for {account_id}")

        try:
            # Get executions that don't have commission records
            exec_ids = await self._exec_repo.get_exec_ids(account_id)
            missing_ids = await self._comm_repo.get_missing_commission_exec_ids(
                account_id=account_id,
                exec_ids=exec_ids,
            )

            if not missing_ids:
                logger.info("All executions have commission records")
                return 0

            logger.info(f"Found {len(missing_ids)} executions missing commissions")

            # IB sends commission reports automatically with executions
            # In real-time, we capture them via events
            # For history loading, we would need FLEX reports

            if self._dry_run:
                logger.info(f"DRY RUN: Would load commissions for {len(missing_ids)} executions")
                return 0

            # Note: IB API doesn't have a direct way to query historical commissions
            # Commission reports come with execution reports or via FLEX
            # For now, log a warning about missing commissions
            logger.warning(
                f"Missing commission data for {len(missing_ids)} executions. "
                "Commission reports require FLEX queries (not yet implemented)."
            )

            return 0

        except Exception as e:
            logger.error(f"Failed to load commissions: {e}")
            raise

    async def _fetch_executions(
        self,
        account_id: str,
    ) -> List[Dict[str, Any]]:
        """Fetch executions from IB API."""
        try:
            from ib_async import ExecutionFilter

            client = await self._get_ib_client()

            # Create execution filter for specific account
            exec_filter = ExecutionFilter(acctCode=account_id)

            # Request executions
            fills = await client.reqExecutionsAsync(exec_filter)

            # Convert to list of dicts with execution and contract
            results = []
            for fill in fills:
                results.append(
                    {
                        "exec_id": fill.execution.execId,
                        "execution": fill.execution,
                        "contract": fill.contract,
                        "commission_report": fill.commissionReport,
                    }
                )

                # Also capture commission if available
                if fill.commissionReport and fill.commissionReport.execId:
                    await self._save_commission_report(fill.commissionReport, account_id)

            return results

        except ImportError:
            logger.warning("ib_async not available, returning empty list")
            return []
        except Exception as e:
            logger.error(f"Failed to fetch executions: {e}")
            return []

    async def _save_commission_report(
        self,
        commission_report: Any,
        account_id: str,
    ) -> None:
        """Save a commission report to the database."""
        if self._dry_run:
            return

        try:
            entity = IbCommissionRepository.from_ib_commission_report(commission_report, account_id)
            await self._comm_repo.upsert(entity)
        except Exception as e:
            logger.warning(f"Failed to save commission report: {e}")

    async def disconnect(self):
        """Disconnect from IB."""
        if self._ib_client:
            self._ib_client.disconnect()
            self._ib_client = None
