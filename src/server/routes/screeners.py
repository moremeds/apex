"""REST routes — /api/screeners and /api/backtest (R2 proxy with caching).

Data resolution order:
1. In-memory cache (5 min TTL)
2. R2 (Cloudflare storage) if credentials configured
3. GitHub Pages static site fallback (always available)
"""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
import time
import urllib.error
import urllib.request
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

logger = logging.getLogger(__name__)

# GitHub Pages static site — always-available fallback
_STATIC_DATA_URL = "https://moremeds.github.io/apex/data"


_ssl_ctx = ssl.create_default_context()
try:
    import certifi
    _ssl_ctx.load_verify_locations(certifi.where())
except ImportError:
    # Fallback: disable verification for GitHub Pages (public, read-only)
    _ssl_ctx.check_hostname = False
    _ssl_ctx.verify_mode = ssl.CERT_NONE


def _fetch_static_sync(filename: str) -> Any | None:
    """Fetch JSON from GitHub Pages static site (sync, run in thread)."""
    url = f"{_STATIC_DATA_URL}/{filename}"
    req = urllib.request.Request(url, headers={"User-Agent": "APEX-Dashboard/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15, context=_ssl_ctx) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        logger.error("Static fetch failed for %s: HTTP %d", filename, e.code)
        return None
    except Exception as e:
        logger.error("Static fetch failed for %s: %s", filename, e)
        return None


class _CachedProxy:
    """TTL cache with R2 primary + GitHub Pages fallback."""

    def __init__(self, r2_client: Any, ttl_sec: int = 300):
        self._r2 = r2_client
        self._ttl = ttl_sec
        self._cache: dict[str, tuple[float, Any]] = {}  # key → (expires_at, data)

    def get(self, key: str, r2_override: Any = None) -> Any:
        now = time.monotonic()
        if key in self._cache:
            expires_at, data = self._cache[key]
            if now < expires_at:
                return data

        r2 = r2_override or self._r2
        if r2 is not None:
            try:
                data = r2.get_json(key)
                if data is not None:
                    self._cache[key] = (now + self._ttl, data)
                    return data
            except Exception as e:
                logger.error("R2 fetch failed for %s: %s", key, e)
                # Return stale cache if available
                if key in self._cache:
                    return self._cache[key][1]

        return None

    async def get_with_fallback(self, key: str, r2_override: Any = None) -> Any:
        """Try cache → R2 → GitHub Pages static site."""
        # 1. Cache
        now = time.monotonic()
        if key in self._cache:
            expires_at, data = self._cache[key]
            if now < expires_at:
                return data

        # 2. R2
        r2 = r2_override or self._r2
        if r2 is not None:
            try:
                data = r2.get_json(key)
                if data is not None:
                    self._cache[key] = (now + self._ttl, data)
                    return data
            except Exception as e:
                logger.error("R2 fetch failed for %s: %s", key, e)

        # 3. GitHub Pages fallback
        try:
            data = await asyncio.to_thread(_fetch_static_sync, key)
            if data is not None:
                self._cache[key] = (now + self._ttl, data)
                return data
        except Exception as e:
            logger.error("Static fallback failed for %s: %s", key, e)

        # 4. Stale cache as last resort
        if key in self._cache:
            return self._cache[key][1]

        return None


def create_screeners_router(r2_client: Any = None, cache_ttl: int = 300) -> APIRouter:
    """Create router for screener and backtest proxy endpoints.

    Args:
        r2_client: R2Client instance for fetching JSON from Cloudflare R2.
        cache_ttl: Cache TTL in seconds (default 5 minutes).
    """
    router = APIRouter(prefix="/api")
    proxy = _CachedProxy(r2_client, ttl_sec=cache_ttl)

    def _r2_from_request(request: Request) -> Any:
        return getattr(request.app.state, "r2_client", None)

    @router.get("/screeners")
    async def get_screeners(request: Request) -> dict:
        """Get screener results (R2 → static fallback)."""
        data = await proxy.get_with_fallback("screeners.json", _r2_from_request(request))
        if data is None:
            raise HTTPException(status_code=503, detail="Screener data not available")
        return data if isinstance(data, dict) else {"data": data}

    @router.get("/backtest")
    async def get_backtest(request: Request) -> dict:
        """Get strategy comparison results (R2 → static fallback)."""
        data = await proxy.get_with_fallback("strategies.json", _r2_from_request(request))
        if data is None:
            raise HTTPException(status_code=503, detail="Backtest data not available")
        return data if isinstance(data, dict) else {"data": data}

    @router.get("/signal-data/{symbol}")
    async def get_signal_data(
        symbol: str, request: Request, tf: str = Query(default="1d")
    ) -> dict:
        """Per-symbol signal data (R2 → static fallback)."""
        key = f"{symbol}_{tf}.json"
        data = await proxy.get_with_fallback(key, _r2_from_request(request))
        if data is None:
            raise HTTPException(status_code=404, detail=f"No signal data for {symbol}/{tf}")
        return data if isinstance(data, dict) else {"data": data}

    @router.get("/summary")
    async def get_summary(request: Request) -> dict:
        """Summary.json — ETF data, regime, generated_at (R2 → static fallback)."""
        data = await proxy.get_with_fallback("summary.json", _r2_from_request(request))
        if data is None:
            raise HTTPException(status_code=503, detail="Summary data not available")
        return data if isinstance(data, dict) else {"data": data}

    @router.get("/score-history")
    async def get_score_history(request: Request) -> dict:
        """Score history for sparklines (R2 → static fallback)."""
        data = await proxy.get_with_fallback("score_history.json", _r2_from_request(request))
        if data is None:
            raise HTTPException(status_code=503, detail="Score history not available")
        return data if isinstance(data, dict) else {"data": data}

    @router.get("/indicators")
    async def get_indicators(request: Request) -> dict:
        """Indicator definitions + rules (R2 → static fallback)."""
        data = await proxy.get_with_fallback("indicators.json", _r2_from_request(request))
        if data is None:
            raise HTTPException(status_code=503, detail="Indicators data not available")
        return data if isinstance(data, dict) else {"data": data}

    @router.get("/universe")
    async def get_universe(request: Request) -> dict:
        """Universe.json — tier, sector enrichment (R2 → static fallback)."""
        data = await proxy.get_with_fallback("universe.json", _r2_from_request(request))
        if data is None:
            raise HTTPException(status_code=503, detail="Universe data not available")
        return data if isinstance(data, dict) else {"data": data}

    return router
