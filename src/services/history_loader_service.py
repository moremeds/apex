"""
History Loader Service - Orchestrates historical data loading from brokers.

Coordinates loading of orders, executions, and fees from Futu and IB
with rate limiting, progress tracking, and error handling.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

from config.models import AppConfig
from src.infrastructure.persistence.database import Database
from src.infrastructure.persistence.repositories import (
    FutuDealRepository,
    FutuFeeRepository,
    FutuOrderRepository,
    IbCommissionRepository,
    IbExecutionRepository,
    SyncStateRepository,
)
from src.services.futu_history_loader import FutuHistoryLoader
from src.services.ib_history_loader import IbHistoryLoader

logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    """Result of a history load operation."""

    status: str  # SUCCESS, PARTIAL, FAILED
    orders_loaded: int = 0
    deals_loaded: int = 0
    fees_loaded: int = 0
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)


class HistoryLoaderService:
    """
    Service for loading historical trading data from brokers.

    Coordinates:
    - Futu: Orders, Deals, Fees
    - IB: Executions, Commissions

    Features:
    - Rate limiting (Futu: 10 req/30s)
    - Incremental sync (resume from last sync time)
    - Progress tracking via sync_state table
    - Error handling and partial recovery
    """

    def __init__(
        self,
        db: Database,
        config: AppConfig,
        dry_run: bool = False,
    ):
        """
        Initialize history loader service.

        Args:
            db: Database connection.
            config: Application configuration.
            dry_run: If True, don't write to database.
        """
        self._db = db
        self._config = config
        self._dry_run = dry_run

        # Create repositories
        self._sync_state_repo = SyncStateRepository(db)
        self._futu_order_repo = FutuOrderRepository(db)
        self._futu_deal_repo = FutuDealRepository(db)
        self._futu_fee_repo = FutuFeeRepository(db)
        self._ib_exec_repo = IbExecutionRepository(db)
        self._ib_comm_repo = IbCommissionRepository(db)

    async def load_futu_history(
        self,
        account_id: Optional[str] = None,
        market: str = "US",
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        force: bool = False,
    ) -> LoadResult:
        """
        Load historical data from Futu.

        Args:
            account_id: Specific account to load (None for all).
            market: Market filter (US, HK, CN).
            from_date: Start date.
            to_date: End date.
            force: Force full reload ignoring last sync time.

        Returns:
            LoadResult with statistics.
        """
        start_time = time.time()
        result = LoadResult(status="SUCCESS")

        try:
            # Get rate limit config
            rate_limit = self._config.history_loader.futu_rate_limit

            # Create Futu loader
            loader = FutuHistoryLoader(
                order_repo=self._futu_order_repo,
                deal_repo=self._futu_deal_repo,
                fee_repo=self._futu_fee_repo,
                sync_state_repo=self._sync_state_repo,
                futu_config=self._config.futu,
                requests_per_window=rate_limit.requests_per_window,
                window_seconds=rate_limit.window_seconds,
                dry_run=self._dry_run,
            )

            # Determine accounts to load
            accounts = [account_id] if account_id else await self._get_futu_accounts()

            for acc_id in accounts:
                logger.info(f"Loading Futu history for account {acc_id}, market {market}")

                # Check last sync time if not forcing
                if not force:
                    last_sync = await self._sync_state_repo.get_last_record_time(
                        broker="FUTU",
                        account_id=acc_id,
                        data_type="futu_orders",
                        market=market,
                    )
                    if last_sync and from_date:
                        # Adjust from_date to not overlap
                        sync_date = last_sync.date()
                        if sync_date > from_date:
                            logger.info(f"Resuming from last sync: {sync_date}")
                            from_date = sync_date

                # Load orders
                try:
                    orders_count = await loader.load_orders(
                        account_id=acc_id,
                        market=market,
                        from_date=from_date,
                        to_date=to_date,
                    )
                    result.orders_loaded += orders_count
                except Exception as e:
                    logger.error(f"Failed to load orders for {acc_id}: {e}")
                    result.errors.append(f"Orders for {acc_id}: {str(e)}")

                # Load deals
                try:
                    deals_count = await loader.load_deals(
                        account_id=acc_id,
                        market=market,
                        from_date=from_date,
                        to_date=to_date,
                    )
                    result.deals_loaded += deals_count
                except Exception as e:
                    logger.error(f"Failed to load deals for {acc_id}: {e}")
                    result.errors.append(f"Deals for {acc_id}: {str(e)}")

                # Load fees for filled orders
                try:
                    fees_count = await loader.load_fees(
                        account_id=acc_id,
                        market=market,
                    )
                    result.fees_loaded += fees_count
                except Exception as e:
                    logger.error(f"Failed to load fees for {acc_id}: {e}")
                    result.errors.append(f"Fees for {acc_id}: {str(e)}")

        except Exception as e:
            logger.error(f"Futu history load failed: {e}", exc_info=True)
            result.status = "FAILED"
            result.errors.append(str(e))

        result.duration_seconds = time.time() - start_time

        if result.errors and result.status == "SUCCESS":
            result.status = "PARTIAL"

        return result

    async def load_ib_history(
        self,
        account_id: Optional[str] = None,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        force: bool = False,
    ) -> LoadResult:
        """
        Load historical data from IB.

        Note: IB only provides current-day executions via API.
        Historical data requires IB FLEX reports (deferred to v1.2).

        Args:
            account_id: Specific account to load (None for all).
            from_date: Start date (for filtering, IB returns current day).
            to_date: End date.
            force: Force full reload ignoring last sync time.

        Returns:
            LoadResult with statistics.
        """
        start_time = time.time()
        result = LoadResult(status="SUCCESS")

        try:
            # Create IB loader
            loader = IbHistoryLoader(
                exec_repo=self._ib_exec_repo,
                comm_repo=self._ib_comm_repo,
                sync_state_repo=self._sync_state_repo,
                ib_config=self._config.ibkr,
                dry_run=self._dry_run,
            )

            # Determine accounts to load
            accounts = [account_id] if account_id else await self._get_ib_accounts()

            for acc_id in accounts:
                logger.info(f"Loading IB history for account {acc_id}")

                # Load executions (current day only via API)
                try:
                    exec_count = await loader.load_executions(
                        account_id=acc_id,
                    )
                    result.deals_loaded += exec_count
                except Exception as e:
                    logger.error(f"Failed to load executions for {acc_id}: {e}")
                    result.errors.append(f"Executions for {acc_id}: {str(e)}")

                # Load commissions
                try:
                    comm_count = await loader.load_commissions(
                        account_id=acc_id,
                    )
                    result.fees_loaded += comm_count
                except Exception as e:
                    logger.error(f"Failed to load commissions for {acc_id}: {e}")
                    result.errors.append(f"Commissions for {acc_id}: {str(e)}")

        except Exception as e:
            logger.error(f"IB history load failed: {e}", exc_info=True)
            result.status = "FAILED"
            result.errors.append(str(e))

        result.duration_seconds = time.time() - start_time

        if result.errors and result.status == "SUCCESS":
            result.status = "PARTIAL"

        return result

    async def load_all_history(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        force: bool = False,
    ) -> LoadResult:
        """
        Load historical data from all configured brokers.

        Args:
            from_date: Start date.
            to_date: End date.
            force: Force full reload.

        Returns:
            Combined LoadResult.
        """
        start_time = time.time()
        combined = LoadResult(status="SUCCESS")

        # Load from Futu if enabled
        if self._config.futu.enabled:
            market = self._config.futu.filter_trdmarket
            futu_result = await self.load_futu_history(
                market=market,
                from_date=from_date,
                to_date=to_date,
                force=force,
            )
            combined.orders_loaded += futu_result.orders_loaded
            combined.deals_loaded += futu_result.deals_loaded
            combined.fees_loaded += futu_result.fees_loaded
            combined.errors.extend(futu_result.errors)

        # Load from IB if enabled
        if self._config.ibkr.enabled:
            ib_result = await self.load_ib_history(
                from_date=from_date,
                to_date=to_date,
                force=force,
            )
            combined.deals_loaded += ib_result.deals_loaded
            combined.fees_loaded += ib_result.fees_loaded
            combined.errors.extend(ib_result.errors)

        combined.duration_seconds = time.time() - start_time

        if combined.errors:
            combined.status = "PARTIAL"

        return combined

    async def _get_futu_accounts(self) -> List[str]:
        """
        Get list of Futu accounts to load.

        In a full implementation, this would query Futu API.
        For now, returns accounts from sync_state or config.
        """
        # Check sync_state for known accounts
        states = await self._sync_state_repo.get_all_states(broker="FUTU")
        accounts = list(set(s.account_id for s in states))

        if not accounts:
            # Would need to connect to Futu to discover accounts
            logger.warning("No Futu accounts found in sync_state, please specify --account")

        return accounts

    async def _get_ib_accounts(self) -> List[str]:
        """
        Get list of IB accounts to load.

        In a full implementation, this would query IB API.
        For now, returns accounts from sync_state or config.
        """
        # Check sync_state for known accounts
        states = await self._sync_state_repo.get_all_states(broker="IB")
        accounts = list(set(s.account_id for s in states))

        if not accounts:
            logger.warning("No IB accounts found in sync_state, please specify --account")

        return accounts

    async def get_sync_status(self) -> List[Dict[str, Any]]:
        """
        Get current sync status for all data types.

        Returns:
            List of sync state summaries.
        """
        return await self._sync_state_repo.get_sync_summary()
