"""
Futu account info fetching with caching and rate-limit handling.

Extracted from FutuAdapter for single-responsibility.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from ....domain.interfaces.event_bus import EventType
from ....models.account import AccountInfo
from ....utils.logging_setup import get_logger
from .exceptions import FutuConnectionError, FutuRateLimitError, classify_futu_exception

if TYPE_CHECKING:
    from .adapter import FutuAdapter

logger = get_logger(__name__)


class AccountFetcher:
    """
    Handles account info fetching with caching and rate-limit handling.

    Futu rate limit: 10 calls per 30 seconds for accinfo_query.
    Uses caching with TTL to avoid hitting limits.
    """

    def __init__(
        self,
        adapter: "FutuAdapter",
        cache_ttl_sec: int = 10,
    ):
        """
        Initialize account fetcher.

        Args:
            adapter: Parent FutuAdapter for connection management.
            cache_ttl_sec: Cache time-to-live in seconds (default 10s).
        """
        self._adapter = adapter
        self._cache_ttl_sec = cache_ttl_sec

        # Cache state
        self._cache: Optional[AccountInfo] = None
        self._cache_time: Optional[datetime] = None

    def get_cached(self) -> Optional[AccountInfo]:
        """Get cached account info if available."""
        return self._cache

    async def fetch(self) -> AccountInfo:
        """
        Fetch account information from Futu OpenD.

        Uses caching to avoid rate limits. Returns cached data if fresh.

        Returns:
            AccountInfo object.

        Raises:
            Exception: If fetch fails and no cache available.
        """
        now = datetime.now()
        if (
            self._cache is not None
            and self._cache_time is not None
            and (now - self._cache_time).total_seconds() < self._cache_ttl_sec
        ):
            logger.debug("Using cached Futu account info")
            return self._cache

        await self._adapter._ensure_connected()

        from futu import RET_OK, Currency, TrdEnv

        if self._adapter._trd_ctx is None:
            raise Exception("Trading context not initialized")

        try:
            trd_env_enum = getattr(TrdEnv, self._adapter.trd_env, TrdEnv.REAL)

            # Run blocking accinfo_query in executor
            ret, data = await self._adapter._run_blocking(
                self._adapter._trd_ctx.accinfo_query,
                trd_env=trd_env_enum,
                acc_id=self._adapter._acc_id,
                refresh_cache=False,
                currency=Currency.USD,
            )

            if ret != RET_OK:
                if "disconnect" in str(data).lower() or "connection" in str(data).lower():
                    logger.warning("Futu connection issue detected, reconnecting...")
                    self._adapter._connected = False
                    await self._adapter._ensure_connected()
                    ret, data = await self._adapter._run_blocking(
                        self._adapter._trd_ctx.accinfo_query,
                        trd_env=trd_env_enum,
                        acc_id=self._adapter._acc_id,
                        refresh_cache=False,
                        currency=Currency.USD,
                    )
                    if ret != RET_OK:
                        raise Exception(f"Account info query failed after reconnect: {data}")
                else:
                    if "frequent" not in str(data).lower():
                        logger.error(f"Failed to fetch account info from Futu: {data}")
                    raise Exception(f"Account info query failed: {data}")

            if data.empty:
                raise Exception("No account info returned")

            row = data.iloc[0]

            def safe_float(key: str, default: float = 0.0) -> float:
                try:
                    value = row.get(key)
                    if value is None or (isinstance(value, float) and value != value):
                        return default
                    return float(value)
                except (ValueError, TypeError):
                    return default

            net_liquidation = safe_float("total_assets")
            total_cash = safe_float("cash")
            buying_power = safe_float("power")
            maintenance_margin = safe_float("maintenance_margin", 0.0)
            init_margin_req = safe_float("initial_margin", 0.0)
            margin_used = init_margin_req
            margin_available = safe_float("available_funds", buying_power)
            excess_liquidity = safe_float("risk_level", 0.0)
            realized_pnl = safe_float("realized_pl", 0.0)
            unrealized_pnl = safe_float("unrealized_pl", 0.0)

            logger.debug(
                f"Fetched Futu account info: TotalAssets=${net_liquidation:,.2f}, "
                f"BuyingPower=${buying_power:,.2f}, Cash=${total_cash:,.2f}"
            )
            self._adapter._connected = True

            account_info = AccountInfo(
                net_liquidation=net_liquidation,
                total_cash=total_cash,
                buying_power=buying_power,
                margin_used=margin_used,
                margin_available=margin_available,
                maintenance_margin=maintenance_margin,
                init_margin_req=init_margin_req,
                excess_liquidity=excess_liquidity,
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl,
                timestamp=datetime.now(),
                account_id=str(self._adapter._acc_id) if self._adapter._acc_id else None,
            )

            # Update cache
            self._cache = account_info
            self._cache_time = datetime.now()

            # Publish event if event bus available
            if self._adapter._event_bus:
                self._adapter._event_bus.publish(
                    EventType.ACCOUNT_UPDATED,
                    {
                        "account": account_info,
                        "source": "FUTU",
                        "timestamp": datetime.now(),
                    },
                )

            return account_info

        except Exception as e:
            # Classify the exception into typed Futu errors
            futu_error = classify_futu_exception(e)

            if isinstance(futu_error, FutuConnectionError):
                logger.error(f"Futu connection error: {futu_error}")
                self._adapter._connected = False
                raise futu_error

            if isinstance(futu_error, FutuRateLimitError):
                if self._cache is not None:
                    logger.debug("Rate limited - returning cached account info")
                    return self._cache
                logger.warning("Futu rate limited and no cached account info available")
                raise futu_error

            logger.error(f"Failed to fetch account info from Futu: {futu_error}")
            raise futu_error
