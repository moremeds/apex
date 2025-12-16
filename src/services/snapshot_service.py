"""
Snapshot Service - Periodic capture of position, account, and risk snapshots.

Captures snapshots at configurable intervals for warm-start and analysis.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from config.models import SnapshotConfig
from src.infrastructure.persistence.database import Database
from src.infrastructure.persistence.repositories import (
    AccountSnapshot,
    AccountSnapshotRepository,
    PositionSnapshot,
    PositionSnapshotRepository,
    RiskSnapshotRecord,
    RiskSnapshotRepository,
)
from src.utils.timezone import now_utc

logger = logging.getLogger(__name__)


class SnapshotService:
    """
    Service for periodic snapshot capture.

    Captures:
    - Position snapshots (per broker/account)
    - Account snapshots (per broker/account)
    - Risk snapshots (aggregated risk metrics)

    Features:
    - Configurable capture intervals
    - On-shutdown capture
    - Data cleanup based on retention policy
    """

    def __init__(
        self,
        db: Database,
        config: SnapshotConfig,
        get_positions_callback: Optional[Callable] = None,
        get_account_callback: Optional[Callable] = None,
        get_risk_snapshot_callback: Optional[Callable] = None,
    ):
        """
        Initialize snapshot service.

        Args:
            db: Database connection.
            config: Snapshot configuration.
            get_positions_callback: Callback to get current positions.
            get_account_callback: Callback to get current account info.
            get_risk_snapshot_callback: Callback to get current risk snapshot.
        """
        self._db = db
        self._config = config
        self._get_positions = get_positions_callback
        self._get_account = get_account_callback
        self._get_risk_snapshot = get_risk_snapshot_callback

        # Repositories
        self._position_repo = PositionSnapshotRepository(db)
        self._account_repo = AccountSnapshotRepository(db)
        self._risk_repo = RiskSnapshotRepository(db)

        # Tasks
        self._position_task: Optional[asyncio.Task] = None
        self._account_task: Optional[asyncio.Task] = None
        self._risk_task: Optional[asyncio.Task] = None

        self._running = False

    async def start(self) -> None:
        """Start periodic snapshot capture tasks."""
        if self._running:
            logger.warning("Snapshot service already running")
            return

        self._running = True
        logger.info("Starting snapshot service")

        # Start position snapshot task
        if self._config.position_interval_sec > 0 and self._get_positions:
            self._position_task = asyncio.create_task(
                self._position_capture_loop()
            )
            logger.info(f"Position snapshots: every {self._config.position_interval_sec}s")

        # Start account snapshot task
        if self._config.account_interval_sec > 0 and self._get_account:
            self._account_task = asyncio.create_task(
                self._account_capture_loop()
            )
            logger.info(f"Account snapshots: every {self._config.account_interval_sec}s")

        # Start risk snapshot task
        if self._config.risk_interval_sec > 0 and self._get_risk_snapshot:
            self._risk_task = asyncio.create_task(
                self._risk_capture_loop()
            )
            logger.info(f"Risk snapshots: every {self._config.risk_interval_sec}s")

    async def stop(self) -> None:
        """Stop snapshot service and optionally capture final snapshots."""
        if not self._running:
            return

        logger.info("Stopping snapshot service")
        self._running = False

        # Cancel tasks
        for task in [self._position_task, self._account_task, self._risk_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Capture final snapshots on shutdown
        if self._config.capture_on_shutdown:
            logger.info("Capturing final snapshots on shutdown")
            await self.capture_all_now()

        logger.info("Snapshot service stopped")

    async def capture_all_now(self) -> None:
        """Capture all snapshots immediately."""
        await asyncio.gather(
            self.capture_positions_now(),
            self.capture_accounts_now(),
            self.capture_risk_now(),
            return_exceptions=True,
        )

    async def capture_positions_now(self) -> None:
        """Capture position snapshot immediately."""
        if not self._get_positions:
            return

        try:
            snapshot_time = now_utc()
            positions_by_broker = self._get_positions()

            for (broker, account_id), positions in positions_by_broker.items():
                # Convert positions to serializable format
                positions_data = [self._serialize_position(p) for p in positions]

                snapshot = PositionSnapshot(
                    snapshot_time=snapshot_time,
                    broker=broker,
                    account_id=account_id,
                    positions=positions_data,
                    position_count=len(positions_data),
                )
                await self._position_repo.upsert(snapshot)

            logger.debug(f"Captured position snapshots for {len(positions_by_broker)} broker/accounts")

        except Exception as e:
            logger.error(f"Failed to capture position snapshot: {e}")

    async def capture_accounts_now(self) -> None:
        """Capture account snapshot immediately."""
        if not self._get_account:
            return

        try:
            snapshot_time = now_utc()
            accounts_by_broker = self._get_account()

            for (broker, account_id), account_info in accounts_by_broker.items():
                # Convert account info to serializable format
                account_data = self._serialize_account(account_info)

                snapshot = AccountSnapshot(
                    snapshot_time=snapshot_time,
                    broker=broker,
                    account_id=account_id,
                    account_data=account_data,
                )
                await self._account_repo.upsert(snapshot)

            logger.debug(f"Captured account snapshots for {len(accounts_by_broker)} broker/accounts")

        except Exception as e:
            logger.error(f"Failed to capture account snapshot: {e}")

    async def capture_risk_now(self) -> None:
        """Capture risk snapshot immediately."""
        if not self._get_risk_snapshot:
            return

        try:
            snapshot_time = now_utc()
            risk_snapshot = self._get_risk_snapshot()

            if risk_snapshot:
                record = RiskSnapshotRepository.from_risk_snapshot(
                    risk_snapshot, snapshot_time
                )
                await self._risk_repo.insert(record)
                logger.debug("Captured risk snapshot")

        except Exception as e:
            logger.error(f"Failed to capture risk snapshot: {e}")

    async def cleanup(self) -> Dict[str, int]:
        """
        Cleanup old snapshots based on retention policy.

        Returns:
            Dictionary with count of deleted records per type.
        """
        retention_days = self._config.retention_days
        logger.info(f"Cleaning up snapshots older than {retention_days} days")

        results = {}

        try:
            results["positions"] = await self._position_repo.cleanup_old(retention_days)
            results["accounts"] = await self._account_repo.cleanup_old(retention_days)
            results["risk"] = await self._risk_repo.cleanup_old(retention_days)

            total = sum(results.values())
            logger.info(f"Deleted {total} old snapshot records")

        except Exception as e:
            logger.error(f"Failed to cleanup snapshots: {e}")

        return results

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    async def _position_capture_loop(self) -> None:
        """Background loop for position snapshot capture."""
        while self._running:
            try:
                await asyncio.sleep(self._config.position_interval_sec)
                if self._running:
                    await self.capture_positions_now()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Position capture error: {e}")
                await asyncio.sleep(5)  # Brief pause before retry

    async def _account_capture_loop(self) -> None:
        """Background loop for account snapshot capture."""
        while self._running:
            try:
                await asyncio.sleep(self._config.account_interval_sec)
                if self._running:
                    await self.capture_accounts_now()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Account capture error: {e}")
                await asyncio.sleep(5)

    async def _risk_capture_loop(self) -> None:
        """Background loop for risk snapshot capture."""
        while self._running:
            try:
                await asyncio.sleep(self._config.risk_interval_sec)
                if self._running:
                    await self.capture_risk_now()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Risk capture error: {e}")
                await asyncio.sleep(5)

    def _serialize_position(self, position: Any) -> Dict[str, Any]:
        """
        Serialize a Position object to dictionary.

        Preserves types for proper reconstruction:
        - bools stay as bools (not converted to 1.0)
        - ints stay as ints
        - floats/Decimals become floats
        - datetimes become ISO strings
        - lists are preserved as lists
        """
        from decimal import Decimal

        # Handle both dataclass and dict
        if hasattr(position, "__dict__"):
            data = {}
            for key, value in position.__dict__.items():
                data[key] = self._serialize_value(value)
            return data
        elif isinstance(position, dict):
            return {k: self._serialize_value(v) for k, v in position.items()}
        else:
            return {"value": str(position)}

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a single value preserving types."""
        from decimal import Decimal

        if value is None:
            return None
        elif isinstance(value, bool):
            # Must check bool before int (bool is subclass of int)
            return value
        elif isinstance(value, datetime):
            return {"__type__": "datetime", "value": value.isoformat()}
        elif isinstance(value, Decimal):
            return {"__type__": "Decimal", "value": str(value)}
        elif isinstance(value, (int, float)):
            return value
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif hasattr(value, "__dict__"):
            # Nested dataclass/object
            return {
                "__type__": type(value).__name__,
                **self._serialize_position(value)
            }
        else:
            return str(value)

    def _serialize_account(self, account_info: Any) -> Dict[str, Any]:
        """Serialize an AccountInfo object to dictionary."""
        return self._serialize_position(account_info)
