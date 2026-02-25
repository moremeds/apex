"""REST routes — /api/screeners and /api/backtest (R2 proxy with caching)."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)


class _CachedProxy:
    """Simple TTL cache for R2 JSON fetches."""

    def __init__(self, r2_client: Any, ttl_sec: int = 300):
        self._r2 = r2_client
        self._ttl = ttl_sec
        self._cache: dict[str, tuple[float, Any]] = {}  # key → (expires_at, data)

    def get(self, key: str) -> Any:
        now = time.monotonic()
        if key in self._cache:
            expires_at, data = self._cache[key]
            if now < expires_at:
                return data

        if self._r2 is None:
            return None

        try:
            data = self._r2.get_json(key)
        except Exception as e:
            logger.error("R2 fetch failed for %s: %s", key, e)
            # Return stale cache if available
            if key in self._cache:
                return self._cache[key][1]
            return None

        if data is not None:
            self._cache[key] = (now + self._ttl, data)
        return data


def create_screeners_router(r2_client: Any = None, cache_ttl: int = 300) -> APIRouter:
    """Create router for screener and backtest proxy endpoints.

    Args:
        r2_client: R2Client instance for fetching JSON from Cloudflare R2.
        cache_ttl: Cache TTL in seconds (default 5 minutes).
    """
    router = APIRouter(prefix="/api")
    proxy = _CachedProxy(r2_client, ttl_sec=cache_ttl)

    @router.get("/screeners")
    async def get_screeners() -> dict:
        """Get screener results (proxied from R2, cached)."""
        data = proxy.get("screeners.json")
        if data is None:
            raise HTTPException(status_code=503, detail="Screener data not available")
        return data if isinstance(data, dict) else {"data": data}

    @router.get("/backtest")
    async def get_backtest() -> dict:
        """Get strategy comparison results (proxied from R2, cached)."""
        data = proxy.get("strategies.json")
        if data is None:
            raise HTTPException(status_code=503, detail="Backtest data not available")
        return data if isinstance(data, dict) else {"data": data}

    return router
