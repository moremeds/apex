"""Strategy listing and parameter endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.domain.strategy.param_loader import (
    get_strategy_metadata,
    get_strategy_params,
    list_strategies,
    load_strategy_config,
)

router = APIRouter(prefix="/strategy", tags=["strategy"])


@router.get("/list")
async def strategy_list() -> list[dict]:
    """List all registered strategies with tier and param count."""
    result = []
    for name in list_strategies():
        try:
            _module, _cls, params, tier = get_strategy_metadata(name)
            result.append({"name": name, "tier": tier, "param_count": len(params)})
        except Exception:
            result.append({"name": name, "tier": "unknown", "param_count": 0})
    return result


@router.get("/{name}/params")
async def strategy_params(name: str) -> dict:
    """Get current parameters and recent history for a strategy."""
    if name not in list_strategies():
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")

    config = load_strategy_config(name)
    return {
        "name": name,
        "params": get_strategy_params(name),
        "history": (config.get("history") or [])[-5:],
    }
