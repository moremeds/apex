"""
Futu position fetching with caching and rate-limit handling.

Extracted from FutuAdapter for single-responsibility.
"""

from __future__ import annotations
from typing import List, Dict, Optional, TYPE_CHECKING
from datetime import datetime, timedelta
import threading

from ....utils.logging_setup import get_logger
from ....models.position import Position
from .converters import convert_position
from .exceptions import classify_futu_exception, FutuConnectionError, FutuRateLimitError

if TYPE_CHECKING:
    from .adapter import FutuAdapter

logger = get_logger(__name__)


class PositionFetcher:
    """
    Handles position fetching with caching and rate-limit handling.

    Futu rate limit: 10 calls per 30 seconds for position_list_query.
    Uses caching with TTL to avoid hitting limits.
    """

    def __init__(
        self,
        adapter: "FutuAdapter",
        cache_ttl_sec: int = 30,
    ):
        """
        Initialize position fetcher.

        Args:
            adapter: Parent FutuAdapter for connection management.
            cache_ttl_sec: Cache time-to-live in seconds (default 30s).
        """
        self._adapter = adapter
        self._cache_ttl_sec = cache_ttl_sec

        # Cache state
        self._cache: Optional[List[Position]] = None
        self._cache_time: Optional[datetime] = None
        self._cooldown_until: Optional[datetime] = None
        self._lock = threading.Lock()

    def invalidate_cache(self) -> None:
        """Invalidate position cache (e.g., after trade notification)."""
        with self._lock:
            self._cache = None
            self._cache_time = None
            self._cooldown_until = None

    def get_cached(self) -> Optional[List[Position]]:
        """Get cached positions if available."""
        with self._lock:
            return self._cache

    async def fetch(self) -> List[Position]:
        """
        Fetch positions from Futu OpenD.

        Uses caching to avoid rate limits. Returns cached data if:
        - Cache is fresh (within TTL)
        - Rate limit cooldown is active

        Returns:
            List of Position objects.

        Raises:
            Exception: If fetch fails and no cache available.
        """
        now = datetime.now()

        # Check cooldown
        with self._lock:
            if self._cooldown_until and now < self._cooldown_until:
                logger.warning(
                    "Futu position fetch skipped due to rate-limit cooldown "
                    f"(retry after {self._cooldown_until.isoformat(timespec='seconds')})"
                )
                if self._cache is not None:
                    return self._cache
                return []

            # Check cache freshness
            if (
                self._cache is not None
                and self._cache_time is not None
                and (now - self._cache_time).total_seconds() < self._cache_ttl_sec
            ):
                logger.debug("Using cached Futu positions")
                return self._cache

        # Ensure connected
        await self._adapter._ensure_connected()

        from futu import RET_OK, TrdEnv

        positions = []
        try:
            trd_env_enum = getattr(TrdEnv, self._adapter.trd_env, TrdEnv.REAL)

            # Run blocking position_list_query in executor
            ret, data = await self._adapter._run_blocking(
                self._adapter._trd_ctx.position_list_query,
                trd_env=trd_env_enum,
                acc_id=self._adapter._acc_id,
                refresh_cache=False,
            )

            if ret != RET_OK:
                if "disconnect" in str(data).lower() or "connection" in str(data).lower():
                    logger.warning("Futu connection issue detected, reconnecting...")
                    self._adapter._connected = False
                    await self._adapter._ensure_connected()
                    ret, data = await self._adapter._run_blocking(
                        self._adapter._trd_ctx.position_list_query,
                        trd_env=trd_env_enum,
                        acc_id=self._adapter._acc_id,
                        refresh_cache=False,
                    )
                    if ret != RET_OK:
                        raise Exception(f"Position query failed after reconnect: {data}")
                else:
                    if "frequent" not in str(data).lower():
                        logger.error(f"Failed to fetch positions from Futu: {data}")
                    raise Exception(f"Position query failed: {data}")

            if data.empty:
                logger.debug("No positions found in Futu account")
                return []

            for _, row in data.iterrows():
                position = convert_position(row, self._adapter._acc_id)
                if position:
                    positions.append(position)

            logger.debug(f"Fetched {len(positions)} positions from Futu")
            self._adapter._connected = True

            # Update cache
            with self._lock:
                self._cache = positions
                self._cache_time = datetime.now()
                self._cooldown_until = None

        except Exception as e:
            # Classify the exception into typed Futu errors
            futu_error = classify_futu_exception(e)

            if isinstance(futu_error, FutuConnectionError):
                logger.error(f"Futu connection error: {futu_error}")
                self._adapter._connected = False
                raise futu_error

            if isinstance(futu_error, FutuRateLimitError):
                cooldown_seconds = futu_error.cooldown_seconds
                with self._lock:
                    self._cooldown_until = datetime.now() + timedelta(seconds=cooldown_seconds)
                logger.warning(f"Futu rate limit hit; backing off for {cooldown_seconds}s")
                if self._cache is not None:
                    return self._cache
                logger.warning("Futu rate limited and no cached positions available")
                raise futu_error

            logger.error(f"Failed to fetch positions from Futu: {futu_error}")
            raise futu_error

        return positions
