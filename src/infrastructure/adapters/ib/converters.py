"""
IB data converters.

Converts IB API responses to internal domain models.
All timestamps are stored as UTC for internal consistency.
IB returns timezone-aware UTC timestamps.
"""

from __future__ import annotations
from typing import Optional
from datetime import datetime
from math import isnan
import logging

from ....models.position import Position, AssetType, PositionSource
from ....models.order import Order, Trade, OrderSource, OrderStatus, OrderSide, OrderType
from ....utils.timezone import now_utc, parse_ib_timestamp


logger = logging.getLogger(__name__)


def convert_position(ib_pos) -> Position:
    """
    Convert ib_async Position to internal Position model.

    Args:
        ib_pos: ib_async Position object.

    Returns:
        Position object.

    Note on avgCost:
        IB's avgCost is the "average cost per share" which for options already
        includes the multiplier. For a PUT sold at $5.00, IB reports avgCost=500.
        We need to divide by multiplier to get the per-contract price for our
        PnL calculation: (mark - avg_price) * quantity * multiplier
    """
    contract = ib_pos.contract

    # Determine asset type
    if contract.secType == "STK":
        asset_type = AssetType.STOCK
    elif contract.secType == "OPT":
        asset_type = AssetType.OPTION
    elif contract.secType == "FUT":
        asset_type = AssetType.FUTURE
    else:
        asset_type = AssetType.CASH

    # For stocks, expiry/strike/right should be None
    if asset_type == AssetType.STOCK:
        expiry = None
        strike = None
        right = None
    else:
        expiry = contract.lastTradeDateOrContractMonth or None
        strike = float(contract.strike) if contract.strike else None
        right = contract.right or None

    # Get multiplier (default 1 for stocks, typically 100 for options)
    multiplier = int(contract.multiplier or 1)

    # IB's avgCost is already multiplied by the contract multiplier for derivatives
    avg_cost = ib_pos.avgCost
    if asset_type in (AssetType.OPTION, AssetType.FUTURE) and multiplier > 1:
        avg_price = avg_cost / multiplier
    else:
        avg_price = avg_cost

    return Position(
        symbol=contract.localSymbol,
        underlying=contract.symbol,
        asset_type=asset_type,
        quantity=float(ib_pos.position),
        strike=strike,
        right=right,
        expiry=expiry,
        avg_price=avg_price,
        multiplier=multiplier,
        source=PositionSource.IB,
        last_updated=now_utc(),
        account_id=ib_pos.account,
    )


def convert_order(ib_order_wrapper) -> Optional[Order]:
    """
    Convert ib_async Trade wrapper to internal Order model.

    Note: In ib_async, the Trade class is a wrapper containing:
    - trade.contract (security)
    - trade.order (the IB Order object)
    - trade.orderStatus (status information)
    It represents an order with its status, NOT a trade/execution.

    Args:
        ib_order_wrapper: ib_async Trade object (order + status wrapper).

    Returns:
        Order object or None if conversion fails.
    """
    try:
        contract = ib_order_wrapper.contract
        order = ib_order_wrapper.order
        order_status = ib_order_wrapper.orderStatus

        # Determine asset type
        if contract.secType == "STK":
            asset_type = "STOCK"
        elif contract.secType == "OPT":
            asset_type = "OPTION"
        elif contract.secType == "FUT":
            asset_type = "FUTURE"
        else:
            asset_type = contract.secType

        # Map IB order type
        order_type_map = {
            "MKT": OrderType.MARKET,
            "LMT": OrderType.LIMIT,
            "STP": OrderType.STOP,
            "STP LMT": OrderType.STOP_LIMIT,
        }
        order_type = order_type_map.get(order.orderType, OrderType.MARKET)

        # Map IB order status
        status_map = {
            "PendingSubmit": OrderStatus.PENDING,
            "PendingCancel": OrderStatus.PENDING,
            "PreSubmitted": OrderStatus.SUBMITTED,
            "Submitted": OrderStatus.SUBMITTED,
            "Cancelled": OrderStatus.CANCELLED,
            "Filled": OrderStatus.FILLED,
            "Inactive": OrderStatus.REJECTED,
        }
        status = status_map.get(order_status.status, OrderStatus.PENDING)

        # Determine side
        side = OrderSide.BUY if order.action == "BUY" else OrderSide.SELL

        # Option-specific fields
        expiry = None
        strike = None
        right = None
        if asset_type == "OPTION":
            expiry = contract.lastTradeDateOrContractMonth or None
            strike = float(contract.strike) if contract.strike else None
            right = contract.right or None

        return Order(
            order_id=str(order.orderId),
            source=OrderSource.IB,
            account_id=order.account or "",
            symbol=contract.localSymbol or contract.symbol,
            underlying=contract.symbol,
            asset_type=asset_type,
            side=side,
            order_type=order_type,
            quantity=float(order.totalQuantity),
            limit_price=float(order.lmtPrice) if order.lmtPrice else None,
            stop_price=float(order.auxPrice) if order.auxPrice else None,
            status=status,
            filled_quantity=float(order_status.filled) if order_status.filled else 0.0,
            avg_fill_price=float(order_status.avgFillPrice) if order_status.avgFillPrice else None,
            commission=float(order_status.commission) if order_status.commission and not isnan(order_status.commission) else 0.0,
            submitted_time=now_utc(),
            filled_time=now_utc() if status == OrderStatus.FILLED else None,
            updated_time=now_utc(),
            expiry=expiry,
            strike=strike,
            right=right,
            broker_order_id=str(order.permId) if order.permId else None,
            exchange=contract.exchange,
            time_in_force=order.tif,
        )

    except Exception as e:
        logger.warning(f"Failed to convert IB order wrapper to order: {e}")
        return None


def convert_fill(ib_fill) -> Optional[Trade]:
    """
    Convert ib_async Fill to internal Trade model.

    Args:
        ib_fill: ib_async Fill object (execution details).

    Returns:
        Trade object or None if conversion fails.
    """
    try:
        contract = ib_fill.contract
        execution = ib_fill.execution
        commission_report = ib_fill.commissionReport

        # Determine asset type
        if contract.secType == "STK":
            asset_type = "STOCK"
        elif contract.secType == "OPT":
            asset_type = "OPTION"
        elif contract.secType == "FUT":
            asset_type = "FUTURE"
        else:
            asset_type = contract.secType

        # Determine side
        side = OrderSide.BUY if execution.side == "BOT" else OrderSide.SELL

        # Get execution time - IB returns UTC-aware timestamps
        trade_time = parse_ib_timestamp(execution.time) if execution.time else now_utc()

        # Option-specific fields
        expiry = None
        strike = None
        right = None
        if asset_type == "OPTION":
            expiry = contract.lastTradeDateOrContractMonth or None
            strike = float(contract.strike) if contract.strike else None
            right = contract.right or None

        # Get commission from report
        commission = 0.0
        if commission_report and commission_report.commission and not isnan(commission_report.commission):
            commission = float(commission_report.commission)

        return Trade(
            trade_id=execution.execId,
            order_id=str(execution.orderId),
            source=OrderSource.IB,
            account_id=execution.acctNumber or "",
            symbol=contract.localSymbol or contract.symbol,
            underlying=contract.symbol,
            asset_type=asset_type,
            side=side,
            quantity=float(execution.shares),
            price=float(execution.price),
            commission=commission,
            trade_time=trade_time,
            expiry=expiry,
            strike=strike,
            right=right,
            exchange=execution.exchange,
            liquidity=execution.liquidation if hasattr(execution, 'liquidation') else None,
        )

    except Exception as e:
        logger.warning(f"Failed to convert IB fill to trade: {e}")
        return None
