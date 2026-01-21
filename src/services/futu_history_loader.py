"""
Futu History Loader - Load historical orders, deals, and fees from Futu.

Implements rate-limited loading with incremental sync support.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from config.models import FutuConfig
from src.infrastructure.persistence.repositories import (
    FutuDealRepository,
    FutuFeeRepository,
    FutuOrderRepository,
    SyncStateRepository,
)

logger = logging.getLogger(__name__)

# Maximum days per query (Futu SDK limitation)
MAX_DAYS_PER_QUERY = 90
# Maximum order IDs per fee query (Futu SDK limitation)
MAX_FEE_BATCH_SIZE = 400


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, requests_per_window: int, window_seconds: int):
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.request_times: List[float] = []

    async def acquire(self) -> None:
        """Wait if necessary to stay within rate limit."""
        now = time.time()

        # Remove old requests outside window
        cutoff = now - self.window_seconds
        self.request_times = [t for t in self.request_times if t > cutoff]

        # Check if we need to wait
        if len(self.request_times) >= self.requests_per_window:
            # Wait until oldest request exits window
            oldest = min(self.request_times)
            wait_time = oldest + self.window_seconds - now
            if wait_time > 0:
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)

        # Record this request
        self.request_times.append(time.time())


class FutuHistoryLoader:
    """
    Loader for Futu historical data.

    Handles:
    - Orders from history_order_list_query()
    - Deals from history_deal_list_query()
    - Fees from order_fee_query()

    Features:
    - Rate limiting (default: 10 req/30s)
    - Incremental sync via sync_state
    - Batch inserts for efficiency
    """

    def __init__(
        self,
        order_repo: FutuOrderRepository,
        deal_repo: FutuDealRepository,
        fee_repo: FutuFeeRepository,
        sync_state_repo: SyncStateRepository,
        futu_config: FutuConfig,
        requests_per_window: int = 10,
        window_seconds: int = 30,
        batch_size: int = 100,
        dry_run: bool = False,
    ):
        """
        Initialize Futu history loader.

        Args:
            order_repo: Repository for orders.
            deal_repo: Repository for deals.
            fee_repo: Repository for fees.
            sync_state_repo: Repository for sync state tracking.
            futu_config: Futu configuration.
            requests_per_window: Max API requests per window.
            window_seconds: Rate limit window in seconds.
            batch_size: Records per batch insert.
            dry_run: If True, don't write to database.
        """
        self._order_repo = order_repo
        self._deal_repo = deal_repo
        self._fee_repo = fee_repo
        self._sync_state_repo = sync_state_repo
        self._futu_config = futu_config
        self._batch_size = batch_size
        self._dry_run = dry_run

        self._rate_limiter = RateLimiter(requests_per_window, window_seconds)
        self._futu_client = None  # Lazy initialization
        self._trd_env = None  # Set during client initialization

    def _chunk_date_range(
        self,
        from_date: date,
        to_date: date,
        max_days: int = MAX_DAYS_PER_QUERY,
    ) -> List[tuple]:
        """
        Split a date range into chunks of max_days.

        Futu SDK limits queries to 90-day windows.

        Args:
            from_date: Start date.
            to_date: End date.
            max_days: Maximum days per chunk.

        Returns:
            List of (start_date, end_date) tuples.
        """
        chunks = []
        current_start = from_date

        while current_start <= to_date:
            current_end = min(current_start + timedelta(days=max_days - 1), to_date)
            chunks.append((current_start, current_end))
            current_start = current_end + timedelta(days=1)

        return chunks

    async def _get_futu_client(self) -> Any:
        """Get or create Futu client connection."""
        if self._futu_client is None:
            # Import here to avoid circular imports
            try:
                from futu import OpenSecTradeContext, SecurityFirm, TrdEnv

                # Map config values to Futu enums
                self._trd_env = (
                    TrdEnv.REAL if self._futu_config.trd_env == "REAL" else TrdEnv.SIMULATE
                )
                security_firm_map = {
                    "FUTUSECURITIES": SecurityFirm.FUTUSECURITIES,
                    "FUTUINC": SecurityFirm.FUTUINC,
                    "FUTUSG": SecurityFirm.FUTUSG,
                    "FUTUAU": SecurityFirm.FUTUAU,
                }
                security_firm = security_firm_map.get(
                    self._futu_config.security_firm, SecurityFirm.FUTUSECURITIES
                )

                self._futu_client = OpenSecTradeContext(
                    host=self._futu_config.host,
                    port=self._futu_config.port,
                    security_firm=security_firm,
                )
            except ImportError:
                logger.error("futu-api package not installed")
                raise RuntimeError("futu-api package required for Futu history loading")

        return self._futu_client

    async def load_orders(
        self,
        account_id: str,
        market: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
    ) -> int:
        """
        Load historical orders from Futu.

        Args:
            account_id: Futu account ID.
            market: Market code (US, HK, CN).
            from_date: Start date.
            to_date: End date.

        Returns:
            Number of orders loaded.
        """
        logger.info(f"Loading Futu orders for {account_id}/{market}")

        # Update sync state
        await self._sync_state_repo.update_sync_start(
            broker="FUTU",
            account_id=account_id,
            data_type="futu_orders",
            market=market,
            sync_from_date=from_date,
            sync_to_date=to_date,
        )

        try:
            orders = await self._fetch_orders(account_id, market, from_date, to_date)
            logger.info(f"Fetched {len(orders)} orders from Futu API")

            if self._dry_run:
                logger.info(f"DRY RUN: Would insert {len(orders)} orders")
                return len(orders)

            # Convert to entities and insert
            entities = [FutuOrderRepository.from_futu_order(o, account_id, market) for o in orders]

            # Batch upsert
            total_inserted = 0
            for i in range(0, len(entities), self._batch_size):
                batch = entities[i : i + self._batch_size]
                await self._order_repo.upsert_many(batch)
                total_inserted += len(batch)

                # Update progress
                await self._sync_state_repo.update_sync_progress(
                    broker="FUTU",
                    account_id=account_id,
                    data_type="futu_orders",
                    records_synced=total_inserted,
                    records_total=len(entities),
                    market=market,
                )

            # Get last record time
            last_time = None
            if entities:
                last_time = max(e.updated_time for e in entities if e.updated_time)

            # Mark complete
            await self._sync_state_repo.update_sync_complete(
                broker="FUTU",
                account_id=account_id,
                data_type="futu_orders",
                records_synced=total_inserted,
                last_record_time=last_time,
                market=market,
            )

            logger.info(f"Loaded {total_inserted} orders for {account_id}/{market}")
            return total_inserted

        except Exception as e:
            await self._sync_state_repo.update_sync_failed(
                broker="FUTU",
                account_id=account_id,
                data_type="futu_orders",
                error_message=str(e),
                market=market,
            )
            raise

    async def load_deals(
        self,
        account_id: str,
        market: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
    ) -> int:
        """
        Load historical deals (executions) from Futu.

        Args:
            account_id: Futu account ID.
            market: Market code.
            from_date: Start date.
            to_date: End date.

        Returns:
            Number of deals loaded.
        """
        logger.info(f"Loading Futu deals for {account_id}/{market}")

        await self._sync_state_repo.update_sync_start(
            broker="FUTU",
            account_id=account_id,
            data_type="futu_deals",
            market=market,
            sync_from_date=from_date,
            sync_to_date=to_date,
        )

        try:
            deals = await self._fetch_deals(account_id, market, from_date, to_date)
            logger.info(f"Fetched {len(deals)} deals from Futu API")

            if self._dry_run:
                logger.info(f"DRY RUN: Would insert {len(deals)} deals")
                return len(deals)

            entities = [FutuDealRepository.from_futu_deal(d, account_id, market) for d in deals]

            total_inserted = 0
            for i in range(0, len(entities), self._batch_size):
                batch = entities[i : i + self._batch_size]
                await self._deal_repo.upsert_many(batch)
                total_inserted += len(batch)

            last_time = None
            if entities:
                last_time = max(e.create_time for e in entities if e.create_time)

            await self._sync_state_repo.update_sync_complete(
                broker="FUTU",
                account_id=account_id,
                data_type="futu_deals",
                records_synced=total_inserted,
                last_record_time=last_time,
                market=market,
            )

            logger.info(f"Loaded {total_inserted} deals for {account_id}/{market}")
            return total_inserted

        except Exception as e:
            await self._sync_state_repo.update_sync_failed(
                broker="FUTU",
                account_id=account_id,
                data_type="futu_deals",
                error_message=str(e),
                market=market,
            )
            raise

    async def load_fees(
        self,
        account_id: str,
        market: str,
    ) -> int:
        """
        Load fees for filled orders using batch queries.

        Args:
            account_id: Futu account ID.
            market: Market code.

        Returns:
            Number of fee records loaded.
        """
        logger.info(f"Loading Futu fees for {account_id}/{market}")

        # Get filled orders that don't have fee records yet
        filled_orders = await self._order_repo.find_filled_orders(
            account_id=account_id,
            market=market,
        )

        if not filled_orders:
            logger.info("No filled orders to load fees for")
            return 0

        order_ids = [o.order_id for o in filled_orders]

        # Check which orders are missing fees
        missing_ids = await self._fee_repo.get_missing_fee_order_ids(
            account_id=account_id,
            order_ids=order_ids,
        )

        if not missing_ids:
            logger.info("All filled orders already have fee records")
            return 0

        logger.info(f"Loading fees for {len(missing_ids)} orders")

        if self._dry_run:
            logger.info(f"DRY RUN: Would fetch fees for {len(missing_ids)} orders")
            return len(missing_ids)

        total_loaded = 0

        # Process in batches (SDK limit is 400 per call)
        for i in range(0, len(missing_ids), MAX_FEE_BATCH_SIZE):
            batch_ids = missing_ids[i : i + MAX_FEE_BATCH_SIZE]

            await self._rate_limiter.acquire()

            try:
                fee_records = await self._fetch_order_fees_batch(batch_ids)

                for fee_data in fee_records:
                    try:
                        order_id = str(fee_data.get("order_id", ""))
                        if not order_id:
                            continue

                        entity = FutuFeeRepository.from_futu_fee(fee_data, order_id, account_id)
                        await self._fee_repo.upsert(entity)
                        total_loaded += 1
                    except Exception as e:
                        logger.warning(f"Failed to save fee record: {e}")

            except Exception as e:
                logger.warning(f"Failed to fetch fee batch: {e}")

        logger.info(f"Loaded {total_loaded} fee records")
        return total_loaded

    async def _fetch_orders(
        self,
        account_id: str,
        market: str,
        from_date: Optional[date],
        to_date: Optional[date],
    ) -> List[Dict[str, Any]]:
        """
        Fetch orders from Futu API with date chunking.

        Handles the 90-day query limit by splitting into chunks.
        """
        # Default dates if not provided
        if from_date is None:
            from_date = date.today() - timedelta(days=30)
        if to_date is None:
            to_date = date.today()

        all_orders = []
        chunks = self._chunk_date_range(from_date, to_date)

        try:
            client = await self._get_futu_client()

            for chunk_start, chunk_end in chunks:
                await self._rate_limiter.acquire()

                start_str = chunk_start.strftime("%Y-%m-%d")
                end_str = chunk_end.strftime("%Y-%m-%d")

                logger.debug(f"Fetching orders for {chunk_start} to {chunk_end}")

                # Use correct SDK signature (positional args, not filter_conditions)
                ret, data = client.history_order_list_query(
                    status_filter_list=[],  # Empty = all statuses
                    code="",  # Empty = all symbols
                    start=start_str,
                    end=end_str,
                    trd_env=self._trd_env,
                    acc_id=0,
                    acc_index=0,
                )

                if ret != 0:
                    logger.warning(f"Futu API error for orders {chunk_start}-{chunk_end}: {data}")
                    continue

                # Convert DataFrame to list of dicts
                if data is not None and not data.empty:
                    all_orders.extend(data.to_dict("records"))

            return all_orders

        except ImportError:
            logger.warning("Futu API not available, returning empty list")
            return []

    async def _fetch_deals(
        self,
        account_id: str,
        market: str,
        from_date: Optional[date],
        to_date: Optional[date],
    ) -> List[Dict[str, Any]]:
        """
        Fetch deals from Futu API with date chunking.

        Handles the 90-day query limit by splitting into chunks.
        """
        # Default dates if not provided
        if from_date is None:
            from_date = date.today() - timedelta(days=30)
        if to_date is None:
            to_date = date.today()

        all_deals = []
        chunks = self._chunk_date_range(from_date, to_date)

        try:
            client = await self._get_futu_client()

            for chunk_start, chunk_end in chunks:
                await self._rate_limiter.acquire()

                start_str = chunk_start.strftime("%Y-%m-%d")
                end_str = chunk_end.strftime("%Y-%m-%d")

                logger.debug(f"Fetching deals for {chunk_start} to {chunk_end}")

                # Use correct SDK signature (positional args)
                ret, data = client.history_deal_list_query(
                    code="",  # Empty = all symbols
                    start=start_str,
                    end=end_str,
                    trd_env=self._trd_env,
                    acc_id=0,
                    acc_index=0,
                )

                if ret != 0:
                    logger.warning(f"Futu API error for deals {chunk_start}-{chunk_end}: {data}")
                    continue

                if data is not None and not data.empty:
                    all_deals.extend(data.to_dict("records"))

            return all_deals

        except ImportError:
            logger.warning("Futu API not available, returning empty list")
            return []

    async def _fetch_order_fees_batch(
        self,
        order_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Fetch fee details for a batch of orders.

        SDK expects order_id_list parameter (max 400 per call).
        """
        if not order_ids:
            return []

        try:
            client = await self._get_futu_client()

            # SDK expects order_id_list as a list
            ret, data = client.order_fee_query(
                order_id_list=order_ids,
                trd_env=self._trd_env,
                acc_id=0,
                acc_index=0,
            )

            if ret != 0:
                logger.warning(f"Failed to get fees for batch: {data}")
                return []

            if data is not None and not data.empty:
                return data.to_dict("records")
            return []

        except ImportError:
            return []
        except Exception as e:
            logger.warning(f"Error fetching fees: {e}")
            return []

    def close(self) -> None:
        """Close Futu client connection."""
        if self._futu_client:
            self._futu_client.close()
            self._futu_client = None
