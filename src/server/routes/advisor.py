"""REST routes for /api/advisor (trading advisor).

Follows the factory function pattern from other route modules
(screeners.py, monitor.py, symbols.py): dependencies injected via
constructor args, NOT via app.state.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)


def create_advisor_router(advisor_service: Optional[Any] = None) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/advisor")
    async def get_all_advice(
        sector: str = Query(default=None),
        action: str = Query(default=None),
    ) -> dict[str, Any]:
        """Get advisor recommendations for all symbols."""
        if advisor_service is None:
            raise HTTPException(status_code=503, detail="Advisor service not available")

        result = advisor_service.compute_all()
        result = _serialize(result)

        if sector:
            result["equity"] = [e for e in result["equity"] if e.get("sector") == sector]
        if action:
            result["equity"] = [e for e in result["equity"] if e.get("action") == action]

        return result

    @router.get("/advisor/{symbol}")
    async def get_symbol_advice(symbol: str) -> dict[str, Any]:
        """Get advisor recommendation for a single symbol."""
        if advisor_service is None:
            raise HTTPException(status_code=503, detail="Advisor service not available")

        result = advisor_service.compute_symbol(symbol.upper())
        return _serialize(result)

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
