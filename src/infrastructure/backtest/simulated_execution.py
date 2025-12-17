"""
Simulated execution engine for backtesting.

Provides order matching and fill simulation for backtests.
Supports different fill models:
- IMMEDIATE: Fill at current price instantly (for unit tests)
- NEXT_BAR: Fill at next bar's open
- SLIPPAGE: Fill with configurable slippage

Usage:
    clock = SimulatedClock(start_time)
    execution = SimulatedExecution(clock, fill_model=FillModel.IMMEDIATE)

    # Update with market data
    execution.update_price(tick)

    # Submit order
    order = OrderRequest(symbol="AAPL", side="BUY", quantity=100)
    broker_id = await execution.submit_order(order)

    # Get fills
    fills = execution.get_pending_fills()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable
from enum import Enum
import uuid
import logging

from ...domain.clock import Clock
from ...domain.events.domain_events import QuoteTick, TradeFill
from ...domain.interfaces.execution_provider import OrderRequest, OrderResult

logger = logging.getLogger(__name__)


class FillModel(Enum):
    """Fill model for order execution."""

    IMMEDIATE = "immediate"  # Fill at current price instantly
    NEXT_BAR = "next_bar"  # Fill at next bar's open
    SLIPPAGE = "slippage"  # Fill with slippage model


@dataclass
class SimulatedOrder:
    """Internal order tracking."""

    request: OrderRequest
    broker_order_id: str
    status: str = "pending"  # pending, filled, cancelled, rejected
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    created_at: Optional[datetime] = None


@dataclass
class SimulatedPosition:
    """Internal position tracking."""

    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    multiplier: int = 1  # Contract multiplier (100 for options, 1 for stocks)


class SimulatedExecution:
    """
    Simulated execution engine for backtesting.

    Handles order submission, matching, and fill generation.
    Tracks positions and P&L internally.
    """

    def __init__(
        self,
        clock: Clock,
        fill_model: FillModel = FillModel.IMMEDIATE,
        slippage_bps: float = 5.0,
        commission_per_share: float = 0.005,
        min_commission: float = 1.0,
    ):
        """
        Initialize simulated execution.

        Args:
            clock: Clock instance for timestamps.
            fill_model: How orders are filled.
            slippage_bps: Slippage in basis points (for SLIPPAGE model).
            commission_per_share: Commission per share.
            min_commission: Minimum commission per order.
        """
        self._clock = clock
        self._fill_model = fill_model
        self._slippage_bps = slippage_bps
        self._commission_per_share = commission_per_share
        self._min_commission = min_commission

        # Order tracking
        self._pending_orders: Dict[str, SimulatedOrder] = {}
        self._filled_orders: Dict[str, SimulatedOrder] = {}
        self._cancelled_orders: Dict[str, SimulatedOrder] = {}

        # Fill queue (to be processed by backtest engine)
        self._pending_fills: List[TradeFill] = []

        # Position tracking
        self._positions: Dict[str, SimulatedPosition] = {}

        # Latest prices
        self._latest_prices: Dict[str, QuoteTick] = {}

        # Callbacks
        self._fill_callback: Optional[Callable[[TradeFill], None]] = None

    def update_price(self, tick: QuoteTick) -> None:
        """
        Update latest price for a symbol.

        Called by backtest engine as ticks/bars are processed.
        Triggers fill matching for pending orders.

        Args:
            tick: Quote tick with latest prices.
        """
        self._latest_prices[tick.symbol] = tick
        self._match_orders(tick.symbol)

    async def submit_order(self, order: OrderRequest) -> OrderResult:
        """
        Submit order for simulated execution (async version).

        Args:
            order: Order request.

        Returns:
            OrderResult with broker order ID.
        """
        return self.submit_order_sync(order)

    def submit_order_sync(self, order: OrderRequest) -> OrderResult:
        """
        Submit order for simulated execution (sync version).

        Use this in backtest context where event loop is already running.

        Args:
            order: Order request.

        Returns:
            OrderResult with broker order ID.
        """
        broker_order_id = f"SIM-{uuid.uuid4().hex[:8]}"

        sim_order = SimulatedOrder(
            request=order,
            broker_order_id=broker_order_id,
            created_at=self._clock.now(),
        )

        self._pending_orders[order.client_order_id or broker_order_id] = sim_order

        logger.debug(
            f"Simulated order submitted: {order.client_order_id} -> {broker_order_id}"
        )

        # Immediate fill for market orders
        if self._fill_model == FillModel.IMMEDIATE and order.order_type == "MARKET":
            self._fill_order(sim_order)

        return OrderResult(
            success=True,
            order_id=broker_order_id,
            message="Order submitted",
        )

    async def cancel_order(self, client_order_id: str) -> OrderResult:
        """
        Cancel a pending order.

        Args:
            client_order_id: Client order ID to cancel.

        Returns:
            OrderResult indicating success/failure.
        """
        if client_order_id in self._pending_orders:
            order = self._pending_orders.pop(client_order_id)
            order.status = "cancelled"
            self._cancelled_orders[client_order_id] = order
            logger.info(f"Order cancelled: {client_order_id}")
            return OrderResult(success=True, message="Order cancelled")

        return OrderResult(success=False, message="Order not found")

    def get_pending_fills(self) -> List[TradeFill]:
        """
        Get and clear pending fills.

        Returns:
            List of fills to be processed.
        """
        fills = self._pending_fills.copy()
        self._pending_fills.clear()
        return fills

    def get_position(self, symbol: str) -> Optional[SimulatedPosition]:
        """Get current position for a symbol."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> Dict[str, SimulatedPosition]:
        """Get all positions."""
        return self._positions.copy()

    def set_fill_callback(self, callback: Callable[[TradeFill], None]) -> None:
        """Set callback for fills."""
        self._fill_callback = callback

    def _match_orders(self, symbol: str) -> None:
        """Match pending orders for a symbol against current price."""
        tick = self._latest_prices.get(symbol)
        if not tick:
            return

        for client_id, order in list(self._pending_orders.items()):
            if order.request.symbol != symbol:
                continue
            if order.status != "pending":
                continue

            should_fill = self._should_fill(order.request, tick)
            if should_fill:
                self._fill_order(order)

    def _should_fill(self, order: OrderRequest, tick: QuoteTick) -> bool:
        """Determine if order should be filled at current tick."""
        if order.order_type == "MARKET":
            return True

        price = tick.last or tick.mid

        if order.order_type == "LIMIT":
            if order.side == "BUY":
                return tick.ask is not None and tick.ask <= (order.limit_price or 0)
            else:
                return tick.bid is not None and tick.bid >= (order.limit_price or 0)

        if order.order_type == "STOP":
            if order.side == "BUY":
                return price is not None and price >= (order.stop_price or 0)
            else:
                return price is not None and price <= (order.stop_price or 0)

        return False

    def _fill_order(self, order: SimulatedOrder) -> None:
        """Execute fill for an order."""
        tick = self._latest_prices.get(order.request.symbol)
        if not tick:
            logger.warning(f"No price for {order.request.symbol}, cannot fill")
            return

        # Determine fill price
        if order.request.side == "BUY":
            base_price = tick.ask or tick.last or 0
        else:
            base_price = tick.bid or tick.last or 0

        if base_price <= 0:
            logger.warning(f"Invalid price for {order.request.symbol}, cannot fill")
            return

        # Apply slippage
        if self._fill_model == FillModel.SLIPPAGE:
            slippage = base_price * (self._slippage_bps / 10000)
            if order.request.side == "BUY":
                fill_price = base_price + slippage
            else:
                fill_price = base_price - slippage
        else:
            fill_price = base_price

        # Calculate commission
        commission = max(
            order.request.quantity * self._commission_per_share,
            self._min_commission,
        )

        # Create fill
        fill = TradeFill(
            symbol=order.request.symbol,
            underlying=order.request.underlying or order.request.symbol,
            side=order.request.side,
            quantity=order.request.quantity,
            price=fill_price,
            commission=commission,
            exec_id=f"EXEC-{uuid.uuid4().hex[:8]}",
            order_id=order.broker_order_id,
            asset_type=order.request.asset_type,
            multiplier=order.request.multiplier,
            source="SIMULATED",
            timestamp=self._clock.now(),
        )

        # Update order state
        order.status = "filled"
        order.filled_quantity = order.request.quantity
        order.avg_fill_price = fill_price
        order.commission = commission

        # Move to filled orders
        client_id = order.request.client_order_id or order.broker_order_id
        self._pending_orders.pop(client_id, None)
        self._filled_orders[client_id] = order

        # Update position
        self._update_position(fill)

        # Add to pending fills queue
        self._pending_fills.append(fill)

        # Call fill callback if set
        if self._fill_callback:
            try:
                self._fill_callback(fill)
            except Exception as e:
                logger.error(f"Fill callback error: {e}")

        logger.info(
            f"Order filled: {order.request.client_order_id} "
            f"{order.request.side} {order.request.quantity} {order.request.symbol} "
            f"@ {fill_price:.4f}"
        )

    def _update_position(self, fill: TradeFill) -> None:
        """Update position from fill."""
        symbol = fill.symbol
        multiplier = getattr(fill, 'multiplier', 1) or 1

        if symbol not in self._positions:
            self._positions[symbol] = SimulatedPosition(symbol=symbol, multiplier=multiplier)

        pos = self._positions[symbol]
        # Update multiplier if not set (first trade for this symbol)
        if pos.multiplier == 1 and multiplier > 1:
            pos.multiplier = multiplier

        qty_delta = fill.quantity if fill.side == "BUY" else -fill.quantity

        if pos.quantity == 0:
            # Opening new position
            pos.quantity = qty_delta
            pos.avg_price = fill.price
        elif (pos.quantity > 0 and fill.side == "BUY") or (
            pos.quantity < 0 and fill.side == "SELL"
        ):
            # Adding to position
            total_cost = (pos.quantity * pos.avg_price) + (qty_delta * fill.price)
            pos.quantity += qty_delta
            pos.avg_price = total_cost / pos.quantity if pos.quantity != 0 else 0
        else:
            # Reducing or closing position - P&L includes multiplier
            if abs(qty_delta) >= abs(pos.quantity):
                # Closing or reversing
                pnl = pos.quantity * (fill.price - pos.avg_price) * pos.multiplier
                pos.realized_pnl += pnl
                remaining = qty_delta + pos.quantity
                if abs(remaining) > 0.0001:
                    pos.quantity = remaining
                    pos.avg_price = fill.price
                else:
                    pos.quantity = 0
                    pos.avg_price = 0
            else:
                # Partial close - P&L includes multiplier
                pnl = (-qty_delta) * (fill.price - pos.avg_price) * pos.multiplier
                pos.realized_pnl += pnl
                pos.quantity += qty_delta

    def get_total_realized_pnl(self) -> float:
        """Get total realized P&L across all positions."""
        return sum(pos.realized_pnl for pos in self._positions.values())

    def get_position_value(self, symbol: str) -> float:
        """Get current market value of a position (includes multiplier for options/futures)."""
        pos = self._positions.get(symbol)
        if not pos or pos.quantity == 0:
            return 0.0

        tick = self._latest_prices.get(symbol)
        if not tick:
            return pos.quantity * pos.avg_price * pos.multiplier

        price = tick.mid or tick.last or pos.avg_price
        return pos.quantity * price * pos.multiplier

    def get_total_position_value(self) -> float:
        """Get total market value of all positions."""
        return sum(self.get_position_value(symbol) for symbol in self._positions)

    def reset(self) -> None:
        """Reset execution state."""
        self._pending_orders.clear()
        self._filled_orders.clear()
        self._cancelled_orders.clear()
        self._pending_fills.clear()
        self._positions.clear()
        self._latest_prices.clear()
        logger.debug("SimulatedExecution reset")
