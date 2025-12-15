"""
Futu History Loader - Load historical orders, deals, and fees from Futu.

Implements rate-limited loading with incremental sync support.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from config.models import FutuConfig
from src.infrastructure.persistence.repositories import (
    FutuDealRepository,
    FutuFeeRepository,
    FutuOrderRepository,
    FutuRawDeal,
    FutuRawFee,
    FutuRawOrder,
    SyncStateRepository,
)
from src.utils.timezone import now_utc

logger = logging.getLogger(__name__)


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

    async def _get_futu_client(self):
        """Get or create Futu client connection."""
        if self._futu_client is None:
            # Import here to avoid circular imports
            try:
                from futu import OpenSecTradeContext, TrdEnv, TrdMarket, SecurityFirm

                # Map config values to Futu enums
                trd_env = TrdEnv.REAL if self._futu_config.trd_env == "REAL" else TrdEnv.SIMULATE
                security_firm_map = {
                    "FUTUSECURITIES": SecurityFirm.FUTUSECURITIES,
                    "FUTUINC": SecurityFirm.FUTUINC,
                    "FUTUSG": SecurityFirm.FUTUSG,
                    "FUTUAU": SecurityFirm.FUTUAU,
                }
                security_firm = security_firm_map.get(
                    self._futu_config.security_firm,
                    SecurityFirm.FUTUSECURITIES
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
            entities = [
                FutuOrderRepository.from_futu_order(o, account_id, market)
                for o in orders
            ]

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

            entities = [
                FutuDealRepository.from_futu_deal(d, account_id, market)
                for d in deals
            ]

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
        Load fees for filled orders.

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
        for order_id in missing_ids:
            try:
                # Rate limit before API call
                await self._rate_limiter.acquire()

                fee_data = await self._fetch_order_fee(account_id, order_id)
                if fee_data:
                    entity = FutuFeeRepository.from_futu_fee(fee_data, order_id, account_id)
                    await self._fee_repo.upsert(entity)
                    total_loaded += 1
            except Exception as e:
                logger.warning(f"Failed to load fee for order {order_id}: {e}")

        logger.info(f"Loaded {total_loaded} fee records")
        return total_loaded

    async def _fetch_orders(
        self,
        account_id: str,
        market: str,
        from_date: Optional[date],
        to_date: Optional[date],
    ) -> List[Dict[str, Any]]:
        """Fetch orders from Futu API."""
        await self._rate_limiter.acquire()

        try:
            from futu import TrdMarket

            market_map = {
                "US": TrdMarket.US,
                "HK": TrdMarket.HK,
                "CN": TrdMarket.CN,
                "SG": TrdMarket.SG,
                "JP": TrdMarket.JP,
                "AU": TrdMarket.AU,
            }
            trd_market = market_map.get(market, TrdMarket.US)

            client = await self._get_futu_client()

            # Format dates for Futu API (YYYY-MM-DD)
            start_str = from_date.strftime("%Y-%m-%d") if from_date else None
            end_str = to_date.strftime("%Y-%m-%d") if to_date else None

            ret, data = client.history_order_list_query(
                filter_conditions={
                    "trd_market": trd_market,
                },
                start=start_str,
                end=end_str,
            )

            if ret != 0:
                raise RuntimeError(f"Futu API error: {data}")

            # Convert DataFrame to list of dicts
            if data is not None and not data.empty:
                return data.to_dict("records")
            return []

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
        """Fetch deals from Futu API."""
        await self._rate_limiter.acquire()

        try:
            from futu import TrdMarket

            market_map = {
                "US": TrdMarket.US,
                "HK": TrdMarket.HK,
                "CN": TrdMarket.CN,
            }
            trd_market = market_map.get(market, TrdMarket.US)

            client = await self._get_futu_client()

            start_str = from_date.strftime("%Y-%m-%d") if from_date else None
            end_str = to_date.strftime("%Y-%m-%d") if to_date else None

            ret, data = client.history_deal_list_query(
                filter_conditions={
                    "trd_market": trd_market,
                },
                start=start_str,
                end=end_str,
            )

            if ret != 0:
                raise RuntimeError(f"Futu API error: {data}")

            if data is not None and not data.empty:
                return data.to_dict("records")
            return []

        except ImportError:
            logger.warning("Futu API not available, returning empty list")
            return []

    async def _fetch_order_fee(
        self,
        account_id: str,
        order_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Fetch fee details for a specific order."""
        try:
            client = await self._get_futu_client()

            ret, data = client.order_fee_query(order_id)

            if ret != 0:
                logger.warning(f"Failed to get fee for order {order_id}: {data}")
                return None

            if data is not None and not data.empty:
                return data.to_dict("records")[0] if len(data) > 0 else None
            return None

        except ImportError:
            return None

    def close(self):
        """Close Futu client connection."""
        if self._futu_client:
            self._futu_client.close()
            self._futu_client = None
