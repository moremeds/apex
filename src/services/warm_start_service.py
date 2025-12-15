"""
Warm Start Service - Restore state from database snapshots on startup.

Provides fast startup by loading last known positions and account state
instead of waiting for broker connections.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

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


@dataclass
class WarmStartResult:
    """Result of warm start operation."""

    success: bool
    positions_loaded: int = 0
    accounts_loaded: int = 0
    risk_snapshot_loaded: bool = False
    snapshot_age_seconds: Optional[float] = None
    error: Optional[str] = None


class WarmStartService:
    """
    Service for restoring state from database snapshots.

    On startup:
    1. Loads latest position snapshots per broker/account
    2. Loads latest account snapshots per broker/account
    3. Optionally loads latest risk snapshot

    Features:
    - Configurable max snapshot age (stale threshold)
    - Validation of loaded data
    - Graceful degradation if no snapshots available
    """

    # Maximum age of snapshot to consider valid (default: 1 hour)
    DEFAULT_MAX_AGE_SECONDS = 3600

    def __init__(
        self,
        db: Database,
        max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
    ):
        """
        Initialize warm start service.

        Args:
            db: Database connection.
            max_age_seconds: Maximum age of snapshot to load.
        """
        self._db = db
        self._max_age_seconds = max_age_seconds

        # Repositories
        self._position_repo = PositionSnapshotRepository(db)
        self._account_repo = AccountSnapshotRepository(db)
        self._risk_repo = RiskSnapshotRepository(db)

    async def warm_start(
        self,
        brokers: List[Dict[str, Any]],
        on_positions_loaded: Optional[Callable] = None,
        on_account_loaded: Optional[Callable] = None,
        on_risk_loaded: Optional[Callable] = None,
    ) -> WarmStartResult:
        """
        Perform warm start by loading snapshots from database.

        Args:
            brokers: List of broker configs with 'name' and 'account_id' keys.
            on_positions_loaded: Callback when positions are loaded.
            on_account_loaded: Callback when account is loaded.
            on_risk_loaded: Callback when risk snapshot is loaded.

        Returns:
            WarmStartResult with statistics.
        """
        result = WarmStartResult(success=True)
        now = now_utc()

        logger.info("Starting warm start from database snapshots")

        # Load position and account snapshots for each broker
        for broker_config in brokers:
            broker_name = broker_config.get("name", "UNKNOWN")
            account_id = broker_config.get("account_id", "")

            if not account_id:
                logger.warning(f"No account_id for broker {broker_name}, skipping")
                continue

            # Load positions
            try:
                position_snapshot = await self._position_repo.get_latest(
                    broker=broker_name,
                    account_id=account_id,
                )

                if position_snapshot:
                    age = (now - position_snapshot.snapshot_time).total_seconds()

                    if age <= self._max_age_seconds:
                        logger.info(
                            f"Loading {len(position_snapshot.positions)} positions "
                            f"for {broker_name}/{account_id} (age: {age:.0f}s)"
                        )

                        if on_positions_loaded:
                            on_positions_loaded(
                                broker_name,
                                account_id,
                                position_snapshot.positions,
                            )

                        result.positions_loaded += len(position_snapshot.positions)
                        result.snapshot_age_seconds = age
                    else:
                        logger.warning(
                            f"Position snapshot too old for {broker_name}/{account_id} "
                            f"(age: {age:.0f}s > max: {self._max_age_seconds}s)"
                        )
                else:
                    logger.info(f"No position snapshot found for {broker_name}/{account_id}")

            except Exception as e:
                logger.error(f"Failed to load positions for {broker_name}/{account_id}: {e}")
                result.success = False
                result.error = str(e)

            # Load account
            try:
                account_snapshot = await self._account_repo.get_latest(
                    broker=broker_name,
                    account_id=account_id,
                )

                if account_snapshot:
                    age = (now - account_snapshot.snapshot_time).total_seconds()

                    if age <= self._max_age_seconds:
                        logger.info(
                            f"Loading account data for {broker_name}/{account_id} "
                            f"(age: {age:.0f}s)"
                        )

                        if on_account_loaded:
                            on_account_loaded(
                                broker_name,
                                account_id,
                                account_snapshot.account_data,
                            )

                        result.accounts_loaded += 1
                    else:
                        logger.warning(
                            f"Account snapshot too old for {broker_name}/{account_id}"
                        )
                else:
                    logger.info(f"No account snapshot found for {broker_name}/{account_id}")

            except Exception as e:
                logger.error(f"Failed to load account for {broker_name}/{account_id}: {e}")

        # Load latest risk snapshot
        try:
            risk_snapshot = await self._risk_repo.get_latest()

            if risk_snapshot:
                age = (now - risk_snapshot.snapshot_time).total_seconds()

                if age <= self._max_age_seconds:
                    logger.info(f"Loading risk snapshot (age: {age:.0f}s)")

                    if on_risk_loaded:
                        on_risk_loaded(risk_snapshot.snapshot_data)

                    result.risk_snapshot_loaded = True
                else:
                    logger.warning(f"Risk snapshot too old (age: {age:.0f}s)")
            else:
                logger.info("No risk snapshot found")

        except Exception as e:
            logger.error(f"Failed to load risk snapshot: {e}")

        # Summary
        logger.info(
            f"Warm start complete: "
            f"{result.positions_loaded} positions, "
            f"{result.accounts_loaded} accounts, "
            f"risk={'yes' if result.risk_snapshot_loaded else 'no'}"
        )

        return result

    async def get_snapshot_status(
        self,
        brokers: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Get status of available snapshots for warm start.

        Args:
            brokers: List of broker configs.

        Returns:
            Dictionary with snapshot availability info.
        """
        now = now_utc()
        status = {
            "position_snapshots": [],
            "account_snapshots": [],
            "risk_snapshot": None,
        }

        for broker_config in brokers:
            broker_name = broker_config.get("name", "UNKNOWN")
            account_id = broker_config.get("account_id", "")

            if not account_id:
                continue

            # Check position snapshot
            position_snapshot = await self._position_repo.get_latest(
                broker=broker_name, account_id=account_id
            )
            if position_snapshot:
                age = (now - position_snapshot.snapshot_time).total_seconds()
                status["position_snapshots"].append({
                    "broker": broker_name,
                    "account_id": account_id,
                    "snapshot_time": str(position_snapshot.snapshot_time),
                    "age_seconds": age,
                    "position_count": position_snapshot.position_count,
                    "is_valid": age <= self._max_age_seconds,
                })

            # Check account snapshot
            account_snapshot = await self._account_repo.get_latest(
                broker=broker_name, account_id=account_id
            )
            if account_snapshot:
                age = (now - account_snapshot.snapshot_time).total_seconds()
                status["account_snapshots"].append({
                    "broker": broker_name,
                    "account_id": account_id,
                    "snapshot_time": str(account_snapshot.snapshot_time),
                    "age_seconds": age,
                    "is_valid": age <= self._max_age_seconds,
                })

        # Check risk snapshot
        risk_snapshot = await self._risk_repo.get_latest()
        if risk_snapshot:
            age = (now - risk_snapshot.snapshot_time).total_seconds()
            status["risk_snapshot"] = {
                "snapshot_time": str(risk_snapshot.snapshot_time),
                "age_seconds": age,
                "portfolio_value": float(risk_snapshot.portfolio_value) if risk_snapshot.portfolio_value else None,
                "position_count": risk_snapshot.position_count,
                "is_valid": age <= self._max_age_seconds,
            }

        return status

    async def clear_all_snapshots(self) -> Dict[str, int]:
        """
        Clear all snapshots (for testing/reset).

        Returns:
            Count of deleted records per type.
        """
        logger.warning("Clearing all snapshots")

        results = {}
        results["positions"] = await self._position_repo.delete_where()
        results["accounts"] = await self._account_repo.delete_where()
        results["risk"] = await self._risk_repo.delete_where()

        return results
