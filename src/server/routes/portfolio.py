"""REST routes — /api/portfolio for live positions, account, and broker status."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)


def _position_risk_to_dict(pr: Any) -> Dict[str, Any]:
    """Convert PositionRisk to JSON-serializable dict."""
    pos = pr.position
    return {
        "symbol": pos.symbol,
        "underlying": pos.underlying,
        "asset_type": pos.asset_type.value,
        "quantity": pos.quantity,
        "avg_price": pos.avg_price,
        "multiplier": pos.multiplier,
        "mark_price": pr.mark_price,
        "market_value": pr.market_value,
        "unrealized_pnl": pr.unrealized_pnl,
        "daily_pnl": pr.daily_pnl,
        "delta": pr.delta,
        "gamma": pr.gamma,
        "vega": pr.vega,
        "theta": pr.theta,
        "iv": pr.iv,
        "delta_dollars": pr.delta_dollars,
        "notional": pr.notional,
        "expiry": pos.expiry,
        "strike": pos.strike,
        "right": pos.right,
        "days_to_expiry": pos.days_to_expiry(),
        "source": pos.source.value if hasattr(pos.source, "value") else str(pos.source),
        "account_id": pos.account_id,
        "has_market_data": pr.has_market_data,
        "has_greeks": pr.has_greeks,
        "is_stale": pr.is_stale,
    }


def _account_to_dict(account: Any) -> Dict[str, Any]:
    """Convert AccountInfo to JSON-serializable dict."""
    return {
        "net_liquidation": account.net_liquidation,
        "total_cash": account.total_cash,
        "buying_power": account.buying_power,
        "margin_used": account.margin_used,
        "margin_available": account.margin_available,
        "maintenance_margin": account.maintenance_margin,
        "init_margin_req": account.init_margin_req,
        "excess_liquidity": account.excess_liquidity,
        "unrealized_pnl": account.unrealized_pnl,
        "realized_pnl": account.realized_pnl,
        "margin_utilization": account.margin_utilization(),
        "account_id": account.account_id,
        "timestamp": account.timestamp.isoformat() if account.timestamp else None,
    }


def serialize_snapshot(
    snapshot: Any = None,
    account: Any = None,
    broker_manager: Any = None,
) -> Dict[str, Any]:
    """Serialize full portfolio state for WebSocket broadcast or REST response."""
    positions: List[Dict[str, Any]] = []
    greeks = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

    if snapshot is not None:
        positions = [_position_risk_to_dict(pr) for pr in snapshot.position_risks]
        greeks = {
            "delta": snapshot.portfolio_delta,
            "gamma": snapshot.portfolio_gamma,
            "vega": snapshot.portfolio_vega,
            "theta": snapshot.portfolio_theta,
        }

    account_dict: Optional[Dict[str, Any]] = None
    if account is not None:
        account_dict = _account_to_dict(account)

    broker_status: List[Dict[str, Any]] = []
    if broker_manager is not None:
        for name, adapter in broker_manager._adapters.items():
            broker_status.append(
                {
                    "name": name,
                    "connected": adapter.is_connected(),
                    "position_count": 0,
                    "last_error": None,
                }
            )

    pnl = {
        "unrealized": snapshot.total_unrealized_pnl if snapshot else 0.0,
        "daily": snapshot.total_daily_pnl if snapshot else 0.0,
        "net_liquidation": snapshot.total_net_liquidation if snapshot else 0.0,
    }

    return {
        "positions": positions,
        "account": account_dict,
        "greeks": greeks,
        "pnl": pnl,
        "broker_status": broker_status,
        "position_count": len(positions),
        "timestamp": snapshot.timestamp.isoformat() if snapshot else None,
    }


def create_portfolio_router(container: Any = None) -> APIRouter:
    """Create router for portfolio endpoints.

    Dependencies resolved from request.app.state.container at request time.
    """
    router = APIRouter(prefix="/api/portfolio")

    def _get_container(request: Request) -> Any:
        return container or getattr(request.app.state, "container", None)

    @router.get("/positions")
    async def get_positions(request: Request) -> Dict[str, Any]:
        """Get all current positions with risk metrics."""
        c = _get_container(request)
        if c is None or c.orchestrator is None:
            return {"positions": [], "portfolio_enabled": False}

        snapshot = c.orchestrator.get_latest_snapshot()
        if snapshot is None:
            return {"positions": [], "portfolio_enabled": True}

        return {
            "positions": [_position_risk_to_dict(pr) for pr in snapshot.position_risks],
            "portfolio_enabled": True,
            "position_count": len(snapshot.position_risks),
        }

    @router.get("/account")
    async def get_account(request: Request) -> Dict[str, Any]:
        """Get aggregated account info."""
        c = _get_container(request)
        if c is None or c.account_store is None:
            return {"account": None, "portfolio_enabled": False}

        account = c.account_store.get()
        if account is None:
            return {"account": None, "portfolio_enabled": True}

        return {
            "account": _account_to_dict(account),
            "portfolio_enabled": True,
        }

    @router.get("/account/by-broker")
    async def get_account_by_broker(request: Request) -> Dict[str, Any]:
        """Get per-broker account breakdown."""
        c = _get_container(request)
        if c is None or c.account_store is None:
            return {"accounts": {}, "portfolio_enabled": False}

        accounts: Dict[str, Any] = {}
        ib = c.account_store.get_ib_account()
        futu = c.account_store.get_futu_account()
        if ib:
            accounts["ib"] = _account_to_dict(ib)
        if futu:
            accounts["futu"] = _account_to_dict(futu)
        # Include aggregated if available
        agg = c.account_store.get()
        if agg:
            accounts["aggregated"] = _account_to_dict(agg)

        return {"accounts": accounts, "portfolio_enabled": True}

    @router.get("/broker-status")
    async def get_broker_status(request: Request) -> Dict[str, Any]:
        """Get broker connection status."""
        c = _get_container(request)
        if c is None or c.broker_manager is None:
            return {"brokers": [], "portfolio_enabled": False}

        brokers: List[Dict[str, Any]] = []
        for name, adapter in c.broker_manager._adapters.items():
            brokers.append(
                {
                    "name": name,
                    "connected": adapter.is_connected(),
                    "last_error": None,
                }
            )

        return {"brokers": brokers, "portfolio_enabled": True}

    @router.get("/snapshot")
    async def get_snapshot(request: Request) -> Dict[str, Any]:
        """Get full portfolio snapshot (positions + account + Greeks + broker status)."""
        c = _get_container(request)
        if c is None or c.orchestrator is None:
            return {"portfolio_enabled": False, "positions": [], "account": None}

        snapshot = c.orchestrator.get_latest_snapshot()
        account = c.account_store.get() if c.account_store else None

        result = serialize_snapshot(snapshot, account, c.broker_manager)
        result["portfolio_enabled"] = True
        return result

    return router
