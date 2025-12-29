"""
OPT-012: Contract Qualification Service with batching and caching.

IB's contract qualification is rate-limited to ~1 contract/second.
This service batches qualification requests and caches results to
dramatically reduce qualification time for portfolios with many options.

Features:
- Debounced batching: waits for batch window before API call
- Caching: avoids re-qualifying known contracts (24h TTL)
- Non-blocking: returns immediately for cached, queues others
- Fallback: sequential qualification if batch fails
- Stats: tracking for monitoring

Usage:
    service = ContractQualificationService(ib, debounce_ms=500)

    # Batch qualification (blocking, for initial load)
    qualified = await service.qualify_batch(contracts)

    # Single qualification with callback (non-blocking)
    cached = await service.qualify(contract, callback=on_qualified)
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ....utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class QualificationRequest:
    """A request to qualify a contract."""

    contract: Any  # ib_async.Contract
    callback: Optional[Callable[[Any], Awaitable[None]]] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class QualifiedContract:
    """A cached qualified contract."""

    contract: Any  # ib_async.Contract
    qualified_at: datetime
    ttl_hours: int = 24

    @property
    def is_expired(self) -> bool:
        return datetime.now() - self.qualified_at > timedelta(hours=self.ttl_hours)


class ContractQualificationService:
    """
    Batched contract qualification with caching.

    Dramatically reduces IB contract qualification time by:
    1. Caching qualified contracts (24h TTL by default)
    2. Batching multiple requests into single API calls
    3. Debouncing to collect requests before sending batch

    Performance:
    - Before: 10 contracts = 10-30 seconds (sequential, rate-limited)
    - After: 10 contracts = 3-5 seconds (batched) or <100ms (cached)
    """

    def __init__(
        self,
        ib: Any,  # ib_async.IB
        debounce_ms: int = 500,
        max_batch_size: int = 20,
        cache_ttl_hours: int = 24,
        max_concurrent: int = 3,
        qualification_timeout: float = 30.0,
    ):
        """
        Initialize contract qualification service.

        Args:
            ib: Connected IB instance.
            debounce_ms: Wait time before sending batch (default 500ms).
            max_batch_size: Maximum contracts per batch (default 20).
            cache_ttl_hours: Cache TTL in hours (default 24).
            max_concurrent: Max concurrent qualification batches (default 3).
            qualification_timeout: Timeout for batch qualification (default 30s).
        """
        self._ib = ib
        self._debounce_ms = debounce_ms
        self._max_batch_size = max_batch_size
        self._cache_ttl_hours = cache_ttl_hours
        self._max_concurrent = max_concurrent
        self._qualification_timeout = qualification_timeout

        # Queue and cache
        self._pending: Dict[str, QualificationRequest] = OrderedDict()
        self._cache: Dict[str, QualifiedContract] = {}
        self._in_flight: Dict[str, asyncio.Future] = {}

        # Debounce task
        self._debounce_task: Optional[asyncio.Task] = None
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Stats for monitoring
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "batches_sent": 0,
            "contracts_qualified": 0,
            "errors": 0,
        }

    async def qualify(
        self,
        contract: Any,
        callback: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> Optional[Any]:
        """
        Queue a contract for qualification (non-blocking).

        If cached, returns immediately. Otherwise queues and returns None.
        Callback is invoked when qualification completes.

        Args:
            contract: The contract to qualify.
            callback: Optional async callback when qualified.

        Returns:
            Qualified contract if cached, None if queued.
        """
        key = self._contract_key(contract)

        # Check cache
        if key in self._cache:
            cached = self._cache[key]
            if not cached.is_expired:
                self._stats["cache_hits"] += 1
                if callback:
                    await callback(cached.contract)
                return cached.contract

        self._stats["cache_misses"] += 1

        # Check if already in flight
        if key in self._in_flight:
            future = self._in_flight[key]
            if callback:
                asyncio.create_task(self._await_and_callback(future, callback))
            return None

        # Queue request
        self._pending[key] = QualificationRequest(contract=contract, callback=callback)

        # Start or restart debounce timer
        if self._debounce_task is None or self._debounce_task.done():
            self._debounce_task = asyncio.create_task(self._debounce_and_batch())

        return None

    async def qualify_batch(self, contracts: List[Any]) -> List[Any]:
        """
        Qualify a batch of contracts (blocking).

        Used for initial portfolio load where we need all contracts qualified
        before proceeding. Checks cache first, then qualifies remaining.

        Args:
            contracts: List of contracts to qualify.

        Returns:
            List of qualified contracts (may be fewer than input if some fail).
        """
        if not contracts:
            return []

        # Check cache first
        to_qualify = []
        to_qualify_indices = []
        cached = []
        cached_indices = []

        for i, contract in enumerate(contracts):
            key = self._contract_key(contract)
            if key in self._cache and not self._cache[key].is_expired:
                cached.append(self._cache[key].contract)
                cached_indices.append(i)
                self._stats["cache_hits"] += 1
            else:
                to_qualify.append(contract)
                to_qualify_indices.append(i)
                self._stats["cache_misses"] += 1

        if cached:
            logger.debug(f"Contract qualification: {len(cached)} cache hits")

        if not to_qualify:
            return cached

        # Qualify remaining in batches
        qualified = await self._qualify_with_fallback(to_qualify)

        # Update cache
        for contract in qualified:
            self._update_cache(contract)

        # Combine results in original order
        result = [None] * len(contracts)
        for i, contract in zip(cached_indices, cached):
            result[i] = contract
        for i, contract in zip(to_qualify_indices, qualified):
            if contract is not None:
                result[i] = contract

        return [c for c in result if c is not None]

    def get_cached(self, contract: Any) -> Optional[Any]:
        """
        Get cached qualified contract without triggering qualification.

        Args:
            contract: The contract to look up.

        Returns:
            Cached qualified contract or None.
        """
        key = self._contract_key(contract)
        if key in self._cache:
            cached = self._cache[key]
            if not cached.is_expired:
                return cached.contract
        return None

    async def _debounce_and_batch(self) -> None:
        """Wait for debounce window, then process batch."""
        await asyncio.sleep(self._debounce_ms / 1000)

        if not self._pending:
            return

        # Take up to max_batch_size
        batch_keys = list(self._pending.keys())[: self._max_batch_size]
        batch = [self._pending.pop(key) for key in batch_keys]

        # Create futures for in-flight tracking
        futures: Dict[str, asyncio.Future] = {}
        loop = asyncio.get_event_loop()
        for req in batch:
            key = self._contract_key(req.contract)
            futures[key] = loop.create_future()
            self._in_flight[key] = futures[key]

        # Process batch
        contracts = [req.contract for req in batch]
        try:
            qualified = await self._qualify_with_fallback(contracts)

            # Build lookup by contract key
            qualified_by_key = {}
            for contract in qualified:
                if contract is not None:
                    key = self._contract_key(contract)
                    qualified_by_key[key] = contract
                    self._update_cache(contract)

            # Resolve futures and invoke callbacks
            for req in batch:
                key = self._contract_key(req.contract)
                qualified_contract = qualified_by_key.get(key)

                if key in futures:
                    if qualified_contract:
                        futures[key].set_result(qualified_contract)
                    else:
                        futures[key].set_result(None)

                if req.callback and qualified_contract:
                    try:
                        await req.callback(qualified_contract)
                    except Exception as e:
                        logger.error(f"Qualification callback error: {e}")

        except Exception as e:
            logger.error(f"Batch qualification failed: {e}")
            self._stats["errors"] += 1
            for future in futures.values():
                if not future.done():
                    future.set_exception(e)
        finally:
            for key in futures:
                self._in_flight.pop(key, None)

        # Continue if more pending
        if self._pending:
            self._debounce_task = asyncio.create_task(self._debounce_and_batch())

    async def _qualify_with_fallback(self, contracts: List[Any]) -> List[Any]:
        """
        Qualify contracts with fallback to sequential on failure.

        Tries batch qualification first. If that fails (IB sometimes rejects
        large batches), falls back to qualifying one at a time.
        """
        if not contracts:
            return []

        async with self._semaphore:
            try:
                self._stats["batches_sent"] += 1
                logger.debug(f"Qualifying batch of {len(contracts)} contracts...")

                qualified = await asyncio.wait_for(
                    self._ib.qualifyContractsAsync(*contracts),
                    timeout=self._qualification_timeout,
                )

                # Filter out None results (failed qualifications)
                valid = [c for c in qualified if c is not None and c.conId]
                self._stats["contracts_qualified"] += len(valid)

                if len(valid) < len(contracts):
                    logger.warning(
                        f"Batch qualification: {len(valid)}/{len(contracts)} succeeded"
                    )

                return valid

            except asyncio.TimeoutError:
                logger.warning(
                    f"Batch qualification timeout ({self._qualification_timeout}s), "
                    f"falling back to sequential"
                )
                return await self._qualify_sequential(contracts)

            except Exception as e:
                logger.warning(
                    f"Batch qualification failed ({e}), falling back to sequential"
                )
                return await self._qualify_sequential(contracts)

    async def _qualify_sequential(self, contracts: List[Any]) -> List[Any]:
        """Fallback: qualify contracts one at a time."""
        qualified = []
        for contract in contracts:
            try:
                result = await asyncio.wait_for(
                    self._ib.qualifyContractsAsync(contract),
                    timeout=10.0,
                )
                if result and result[0] and result[0].conId:
                    qualified.append(result[0])
                    self._stats["contracts_qualified"] += 1
            except asyncio.TimeoutError:
                logger.warning(f"Timeout qualifying {self._contract_key(contract)}")
                self._stats["errors"] += 1
            except Exception as e:
                logger.warning(f"Failed to qualify {self._contract_key(contract)}: {e}")
                self._stats["errors"] += 1
        return qualified

    def _contract_key(self, contract: Any) -> str:
        """
        Generate cache key for contract.

        For options: symbol:secType:expiry:strike:right:exchange
        For stocks: symbol:secType:exchange:currency
        """
        sec_type = getattr(contract, "secType", "STK")

        if sec_type == "OPT":
            expiry = getattr(contract, "lastTradeDateOrContractMonth", "")
            strike = getattr(contract, "strike", 0)
            right = getattr(contract, "right", "")
            return f"{contract.symbol}:OPT:{expiry}:{strike}:{right}"
        else:
            exchange = getattr(contract, "exchange", "SMART")
            currency = getattr(contract, "currency", "USD")
            return f"{contract.symbol}:{sec_type}:{exchange}:{currency}"

    def _update_cache(self, contract: Any) -> None:
        """Update cache with qualified contract."""
        key = self._contract_key(contract)
        self._cache[key] = QualifiedContract(
            contract=contract,
            qualified_at=datetime.now(),
            ttl_hours=self._cache_ttl_hours,
        )

    async def _await_and_callback(
        self,
        future: asyncio.Future,
        callback: Callable[[Any], Awaitable[None]],
    ) -> None:
        """Await future and invoke callback."""
        try:
            contract = await future
            if contract:
                await callback(contract)
        except Exception as e:
            logger.error(f"Callback error: {e}")

    def get_stats(self) -> Dict[str, int]:
        """Get qualification statistics."""
        return dict(self._stats)

    def get_cache_size(self) -> int:
        """Get number of cached contracts."""
        return len(self._cache)

    def clear_cache(self) -> None:
        """Clear the qualification cache."""
        self._cache.clear()
        logger.info("Contract qualification cache cleared")

    def clear_expired(self) -> int:
        """Remove expired entries from cache. Returns count removed."""
        expired_keys = [k for k, v in self._cache.items() if v.is_expired]
        for key in expired_keys:
            del self._cache[key]
        if expired_keys:
            logger.debug(f"Cleared {len(expired_keys)} expired cache entries")
        return len(expired_keys)
