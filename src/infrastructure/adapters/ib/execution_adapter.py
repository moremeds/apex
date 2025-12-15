"""
IB Execution Adapter for order management.

Handles:
- Order submission, modification, cancellation
- Trade/fill notifications
- Order status tracking

Uses client_id = base + 2.
"""

from __future__ import annotations
from typing import List, Optional, Callable, Dict
from datetime import datetime, timedelta, timezone

from ....utils.logging_setup import get_logger
from ....domain.interfaces.execution_provider import (
    ExecutionProvider,
    OrderRequest,
    OrderResult,
)
from ....domain.interfaces.event_bus import EventType
from ....domain.events.domain_events import OrderUpdate, TradeFill
from ....models.order import Order, Trade

from .base import IbBaseAdapter
from .converters import convert_order, convert_fill


logger = get_logger(__name__)


class IbExecutionAdapter(IbBaseAdapter, ExecutionProvider):
    """
    IB adapter for order execution.

    Implements ExecutionProvider for order submission and management.
    Uses client_id = base_id + 2 (execution adapter offset).

    Includes safety features:
    - Trading enable/disable (kill switch)
    - Max order size limits
    - Max position size limits
    """

    ADAPTER_TYPE = "execution"

    def __init__(self, *args, **kwargs):
        """Initialize execution adapter."""
        super().__init__(*args, **kwargs)

        # Callbacks
        self._order_callback: Optional[Callable[[OrderUpdate], None]] = None
        self._fill_callback: Optional[Callable[[TradeFill], None]] = None

        # Risk controls
        self._trading_enabled = True
        self._disable_reason: str = ""
        self._max_order_size: Optional[float] = None
        self._max_position_sizes: Dict[str, float] = {}

        # Order tracking
        self._pending_orders: Dict[str, OrderUpdate] = {}

    # -------------------------------------------------------------------------
    # Connection Hooks
    # -------------------------------------------------------------------------

    async def _on_connected(self) -> None:
        """Set up order event handlers after connection."""
        if self.ib:
            self.ib.orderStatusEvent += self._on_order_status
            self.ib.execDetailsEvent += self._on_exec_details
            logger.debug("IbExecutionAdapter: Order event handlers registered")

    async def _on_disconnecting(self) -> None:
        """Clean up before disconnect."""
        if self.ib:
            self.ib.orderStatusEvent -= self._on_order_status
            self.ib.execDetailsEvent -= self._on_exec_details

    # -------------------------------------------------------------------------
    # Order Submission
    # -------------------------------------------------------------------------

    async def submit_order(self, request: OrderRequest) -> OrderResult:
        """Submit a new order."""
        if not self._trading_enabled:
            return OrderResult(
                success=False,
                message=f"Trading disabled: {self._disable_reason}",
                error_code="TRADING_DISABLED",
            )

        # Validate order size
        if self._max_order_size and abs(request.quantity) > self._max_order_size:
            return OrderResult(
                success=False,
                message=f"Order size {request.quantity} exceeds max {self._max_order_size}",
                error_code="MAX_ORDER_SIZE_EXCEEDED",
            )

        # Validate position size
        if request.symbol in self._max_position_sizes:
            max_pos = self._max_position_sizes[request.symbol]
            # Note: Would need to check current position + order quantity
            # For now, just validate order size against max position
            if abs(request.quantity) > max_pos:
                return OrderResult(
                    success=False,
                    message=f"Order would exceed max position size {max_pos} for {request.symbol}",
                    error_code="MAX_POSITION_SIZE_EXCEEDED",
                )

        await self.ensure_connected()

        try:
            from ib_async import (
                Stock, Option, MarketOrder, LimitOrder, StopOrder, StopLimitOrder
            )

            # Create contract
            if request.asset_type == "OPTION":
                expiry = self.format_expiry_for_ib(request.expiry)
                if not expiry:
                    return OrderResult(
                        success=False,
                        message=f"Invalid expiry: {request.expiry}",
                        error_code="INVALID_EXPIRY",
                    )
                contract = Option(
                    symbol=request.underlying or request.symbol,
                    lastTradeDateOrContractMonth=expiry,
                    strike=request.strike,
                    right=request.right,
                    exchange="SMART",
                    multiplier=str(request.multiplier),
                    currency="USD",
                )
            else:
                contract = Stock(request.symbol, "SMART", currency="USD")

            # Qualify contract
            await self.ib.qualifyContractsAsync(contract)

            # Create order
            action = "BUY" if request.side == "BUY" else "SELL"
            quantity = abs(request.quantity)

            if request.order_type == "MARKET":
                ib_order = MarketOrder(action, quantity)
            elif request.order_type == "LIMIT":
                if not request.limit_price:
                    return OrderResult(
                        success=False,
                        message="Limit price required for LIMIT order",
                        error_code="MISSING_LIMIT_PRICE",
                    )
                ib_order = LimitOrder(action, quantity, request.limit_price)
            elif request.order_type == "STOP":
                if not request.stop_price:
                    return OrderResult(
                        success=False,
                        message="Stop price required for STOP order",
                        error_code="MISSING_STOP_PRICE",
                    )
                ib_order = StopOrder(action, quantity, request.stop_price)
            elif request.order_type == "STOP_LIMIT":
                if not request.stop_price or not request.limit_price:
                    return OrderResult(
                        success=False,
                        message="Stop and limit prices required for STOP_LIMIT order",
                        error_code="MISSING_PRICES",
                    )
                ib_order = StopLimitOrder(
                    action, quantity, request.limit_price, request.stop_price
                )
            else:
                return OrderResult(
                    success=False,
                    message=f"Unsupported order type: {request.order_type}",
                    error_code="UNSUPPORTED_ORDER_TYPE",
                )

            # Set time in force
            if request.tif == "GTC":
                ib_order.tif = "GTC"
            elif request.tif == "IOC":
                ib_order.tif = "IOC"
            elif request.tif == "FOK":
                ib_order.tif = "FOK"
            # Default is DAY

            # Place order
            trade = self.ib.placeOrder(contract, ib_order)
            order_id = str(trade.order.orderId)

            logger.info(
                f"Order submitted: {action} {quantity} {request.symbol} "
                f"@ {request.order_type} (order_id={order_id})"
            )

            self.publish_event(EventType.ORDER_SUBMITTED, {
                "order_id": order_id,
                "symbol": request.symbol,
                "side": request.side,
                "quantity": quantity,
                "order_type": request.order_type,
            })

            return OrderResult(
                success=True,
                order_id=order_id,
                message=f"Order submitted successfully",
            )

        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            return OrderResult(
                success=False,
                message=str(e),
                error_code="SUBMISSION_FAILED",
            )

    async def submit_bracket_order(
        self,
        entry: OrderRequest,
        take_profit: OrderRequest,
        stop_loss: OrderRequest,
    ) -> List[OrderResult]:
        """Submit a bracket order."""
        # For simplicity, submit as three separate orders
        # A more sophisticated implementation would use IB's bracket order API
        results = []

        entry_result = await self.submit_order(entry)
        results.append(entry_result)

        if entry_result.success:
            # Submit take profit and stop loss
            tp_result = await self.submit_order(take_profit)
            sl_result = await self.submit_order(stop_loss)
            results.extend([tp_result, sl_result])

        return results

    # -------------------------------------------------------------------------
    # Order Management
    # -------------------------------------------------------------------------

    async def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel an open order."""
        await self.ensure_connected()

        try:
            # Find the trade object
            for trade in self.ib.trades():
                if str(trade.order.orderId) == order_id:
                    self.ib.cancelOrder(trade.order)
                    logger.info(f"Cancel requested for order {order_id}")
                    return OrderResult(
                        success=True,
                        order_id=order_id,
                        message="Cancel requested",
                    )

            return OrderResult(
                success=False,
                order_id=order_id,
                message=f"Order {order_id} not found",
                error_code="ORDER_NOT_FOUND",
            )

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return OrderResult(
                success=False,
                order_id=order_id,
                message=str(e),
                error_code="CANCEL_FAILED",
            )

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        """Cancel all open orders."""
        await self.ensure_connected()

        results = []
        try:
            open_orders = await self.ib.reqOpenOrdersAsync()

            for trade in open_orders:
                # Filter by symbol if specified
                if symbol and trade.contract.symbol != symbol:
                    continue

                try:
                    self.ib.cancelOrder(trade.order)
                    results.append(OrderResult(
                        success=True,
                        order_id=str(trade.order.orderId),
                        message="Cancel requested",
                    ))
                except Exception as e:
                    results.append(OrderResult(
                        success=False,
                        order_id=str(trade.order.orderId),
                        message=str(e),
                        error_code="CANCEL_FAILED",
                    ))

            logger.info(f"Cancelled {len(results)} orders")

        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")

        return results

    async def modify_order(
        self,
        order_id: str,
        new_quantity: Optional[float] = None,
        new_limit_price: Optional[float] = None,
        new_stop_price: Optional[float] = None,
    ) -> OrderResult:
        """Modify an existing order."""
        await self.ensure_connected()

        try:
            for trade in self.ib.trades():
                if str(trade.order.orderId) == order_id:
                    order = trade.order

                    if new_quantity is not None:
                        order.totalQuantity = abs(new_quantity)
                    if new_limit_price is not None:
                        order.lmtPrice = new_limit_price
                    if new_stop_price is not None:
                        order.auxPrice = new_stop_price

                    self.ib.placeOrder(trade.contract, order)
                    logger.info(f"Order {order_id} modified")

                    return OrderResult(
                        success=True,
                        order_id=order_id,
                        message="Order modified",
                    )

            return OrderResult(
                success=False,
                order_id=order_id,
                message=f"Order {order_id} not found",
                error_code="ORDER_NOT_FOUND",
            )

        except Exception as e:
            logger.error(f"Failed to modify order {order_id}: {e}")
            return OrderResult(
                success=False,
                order_id=order_id,
                message=str(e),
                error_code="MODIFY_FAILED",
            )

    # -------------------------------------------------------------------------
    # Order Queries
    # -------------------------------------------------------------------------

    async def get_order(self, order_id: str) -> Optional[OrderUpdate]:
        """Get order status."""
        await self.ensure_connected()

        for trade in self.ib.trades():
            if str(trade.order.orderId) == order_id:
                return self._trade_to_order_update(trade)
        return None

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderUpdate]:
        """Get all open orders."""
        await self.ensure_connected()

        open_trades = await self.ib.reqOpenOrdersAsync()
        orders = []

        for trade in open_trades:
            if symbol and trade.contract.symbol != symbol:
                continue
            orders.append(self._trade_to_order_update(trade))

        return orders

    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        days_back: int = 30,
    ) -> List[OrderUpdate]:
        """Get historical orders."""
        await self.ensure_connected()

        completed = await self.ib.reqCompletedOrdersAsync(apiOnly=False)
        orders = []

        for trade in completed:
            if symbol and trade.contract.symbol != symbol:
                continue
            orders.append(self._trade_to_order_update(trade))

        return orders

    # -------------------------------------------------------------------------
    # Execution Queries
    # -------------------------------------------------------------------------

    async def get_fills(
        self,
        order_id: Optional[str] = None,
        symbol: Optional[str] = None,
        days_back: int = 7,
    ) -> List[TradeFill]:
        """Get trade fills."""
        await self.ensure_connected()

        try:
            from ib_async import ExecutionFilter

            exec_filter = ExecutionFilter()
            start_datetime = datetime.now(timezone.utc) - timedelta(days=days_back + 1)
            exec_filter.time = start_datetime.strftime("%Y%m%d-%H:%M:%S")
            exec_filter.clientId = self.client_id

            fills = await self.ib.reqExecutionsAsync(exec_filter)
            result = []

            for fill in fills:
                if order_id and str(fill.execution.orderId) != order_id:
                    continue
                if symbol and fill.contract.symbol != symbol:
                    continue

                trade_fill = self._fill_to_trade_fill(fill)
                if trade_fill:
                    result.append(trade_fill)

            return result

        except Exception as e:
            logger.error(f"Failed to fetch fills: {e}")
            return []

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def set_order_callback(
        self,
        callback: Optional[Callable[[OrderUpdate], None]]
    ) -> None:
        """Set order status callback."""
        self._order_callback = callback

    def set_fill_callback(
        self,
        callback: Optional[Callable[[TradeFill], None]]
    ) -> None:
        """Set fill callback."""
        self._fill_callback = callback

    def _on_order_status(self, trade) -> None:
        """Handle IB order status event."""
        try:
            order_update = self._trade_to_order_update(trade)

            if self._order_callback:
                self._order_callback(order_update)

            self.publish_event(EventType.ORDER_SUBMITTED, {
                "order_id": order_update.order_id,
                "status": order_update.status,
                "symbol": order_update.symbol,
            })

        except Exception as e:
            logger.error(f"Error handling order status: {e}")

    def _on_exec_details(self, trade, fill) -> None:
        """Handle IB execution details event."""
        try:
            trade_fill = self._fill_to_trade_fill(fill)

            if trade_fill and self._fill_callback:
                self._fill_callback(trade_fill)

            if trade_fill:
                self.publish_event(EventType.ORDER_FILLED, {
                    "exec_id": trade_fill.exec_id,
                    "symbol": trade_fill.symbol,
                    "quantity": trade_fill.quantity,
                    "price": trade_fill.price,
                })

        except Exception as e:
            logger.error(f"Error handling execution: {e}")

    # -------------------------------------------------------------------------
    # Risk Controls
    # -------------------------------------------------------------------------

    def set_max_order_size(self, max_quantity: float) -> None:
        """Set max order size."""
        self._max_order_size = max_quantity
        logger.info(f"Max order size set to {max_quantity}")

    def set_max_position_size(self, symbol: str, max_quantity: float) -> None:
        """Set max position size for symbol."""
        self._max_position_sizes[symbol] = max_quantity
        logger.info(f"Max position size for {symbol} set to {max_quantity}")

    def enable_trading(self) -> None:
        """Enable trading."""
        self._trading_enabled = True
        self._disable_reason = ""
        logger.info("Trading enabled")

    def disable_trading(self, reason: str = "") -> None:
        """Disable trading (kill switch)."""
        self._trading_enabled = False
        self._disable_reason = reason
        logger.warning(f"Trading disabled: {reason}")

    def is_trading_enabled(self) -> bool:
        """Check if trading is enabled."""
        return self._trading_enabled

    # -------------------------------------------------------------------------
    # Converters
    # -------------------------------------------------------------------------

    def _trade_to_order_update(self, trade) -> OrderUpdate:
        """Convert IB trade to OrderUpdate domain event."""
        contract = trade.contract
        order = trade.order
        status = trade.orderStatus

        # Map IB status to our status
        status_map = {
            "PendingSubmit": "PENDING",
            "PreSubmitted": "SUBMITTED",
            "Submitted": "SUBMITTED",
            "Filled": "FILLED",
            "Cancelled": "CANCELLED",
            "ApiCancelled": "CANCELLED",
            "Inactive": "REJECTED",
        }

        return OrderUpdate(
            order_id=str(order.orderId),
            symbol=contract.symbol,
            underlying=contract.symbol,
            side=order.action,
            order_type=order.orderType,
            status=status_map.get(status.status, "PENDING"),
            quantity=float(order.totalQuantity),
            filled_quantity=float(status.filled),
            remaining_quantity=float(status.remaining),
            limit_price=order.lmtPrice if hasattr(order, 'lmtPrice') else None,
            stop_price=order.auxPrice if hasattr(order, 'auxPrice') else None,
            avg_fill_price=float(status.avgFillPrice) if status.avgFillPrice else None,
            asset_type=contract.secType,
            source="IB",
            timestamp=datetime.now(),
        )

    def _fill_to_trade_fill(self, fill) -> Optional[TradeFill]:
        """Convert IB fill to TradeFill domain event."""
        try:
            execution = fill.execution
            contract = fill.contract

            return TradeFill(
                symbol=contract.symbol,
                underlying=contract.symbol,
                side=execution.side,
                quantity=float(execution.shares),
                price=float(execution.price),
                commission=float(fill.commissionReport.commission) if fill.commissionReport else 0.0,
                exec_id=execution.execId,
                order_id=str(execution.orderId),
                asset_type=contract.secType,
                multiplier=int(contract.multiplier) if contract.multiplier else 1,
                source="IB",
                timestamp=datetime.now(),
            )
        except Exception as e:
            logger.error(f"Error converting fill: {e}")
            return None
