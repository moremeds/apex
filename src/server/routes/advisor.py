"""REST routes for /api/advisor (trading advisor).

Follows the factory function pattern from other route modules.
At request time, prefers ``request.app.state.advisor_service`` (set by
lifespan bootstrap) over the constructor arg (used in tests).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request

logger = logging.getLogger(__name__)


def create_advisor_router(advisor_service: Optional[Any] = None) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/advisor")
    async def get_all_advice(
        request: Request,
        sector: str = Query(default=None),
        action: str = Query(default=None),
    ) -> dict[str, Any]:
        """Get advisor recommendations for all symbols."""
        svc = getattr(request.app.state, "advisor_service", None) or advisor_service
        if svc is None:
            raise HTTPException(status_code=503, detail="Advisor service not available")

        # compute_all() is CPU-bound — run in thread to avoid blocking the event loop
        result = await asyncio.to_thread(svc.compute_all)
        result = _serialize(result)

        if sector:
            result["equity"] = [e for e in result["equity"] if e.get("sector") == sector]
        if action:
            result["equity"] = [e for e in result["equity"] if e.get("action") == action]

        return dict(result)

    @router.get("/advisor/{symbol}")
    async def get_symbol_advice(request: Request, symbol: str) -> dict[str, Any]:
        """Get advisor recommendation for a single symbol."""
        svc = getattr(request.app.state, "advisor_service", None) or advisor_service
        if svc is None:
            raise HTTPException(status_code=503, detail="Advisor service not available")

        result = await asyncio.to_thread(svc.compute_symbol, symbol.upper())
        return dict(_serialize(result))

    return router


def _serialize(obj: Any) -> Any:
    """Recursively convert dataclasses to dicts."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _serialize(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj
