"""
Simulated execution engine for backtesting.

Provides order matching and fill simulation for backtests.
Supports different fill models:
- IMMEDIATE: Fill at current price instantly (for unit tests)
- NEXT_BAR: Fill at next bar's open
- SLIPPAGE: Fill with configurable slippage

Can also use RealityModelPack for more realistic simulation with:
- FeeModel: Transaction cost calculation
- SlippageModel: Price impact simulation
- FillModel: Order matching logic
- LatencyModel: Execution delay (optional)

Usage:
    clock = SimulatedClock(start_time)

    # Simple mode
    execution = SimulatedExecution(clock, fill_model=FillModel.IMMEDIATE)

    # Realistic mode with reality pack
    from domain.reality import create_ib_pack
    execution = SimulatedExecution(clock, reality_pack=create_ib_pack())

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
from typing import Dict, List, Optional, Callable, TYPE_CHECKING
from enum import Enum
import uuid
import logging

from ...domain.clock import Clock
from ...domain.events.domain_events import QuoteTick, TradeFill
from ...domain.interfaces.execution_provider import OrderRequest, OrderResult

# Reality models (optional)
from ...domain.reality import (
    RealityModelPack,
    FeeModel as RealityFeeModel,
    SlippageModel,
    FillModel as RealityFillModel,
    LatencyModel,
    FeeBreakdown,
    SlippageResult,
    FillResult,
    OrderType as RealityOrderType,
    AssetType,
)

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
        reality_pack: Optional[RealityModelPack] = None,
    ):
        """
        Initialize simulated execution.

        Args:
            clock: Clock instance for timestamps.
            fill_model: How orders are filled (legacy mode, ignored if reality_pack provided).
            slippage_bps: Slippage in basis points (legacy mode).
            commission_per_share: Commission per share (legacy mode).
            min_commission: Minimum commission per order (legacy mode).
            reality_pack: Optional RealityModelPack for realistic simulation.
                         When provided, uses FeeModel, SlippageModel, FillModel
                         from the pack instead of legacy parameters.
        """
        self._clock = clock
        self._reality_pack = reality_pack

        # Legacy parameters (used when reality_pack is None)
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

    @property
    def reality_pack(self) -> Optional[RealityModelPack]:
        """Get the reality model pack if configured."""
        return self._reality_pack

    @reality_pack.setter
    def reality_pack(self, pack: Optional[RealityModelPack]) -> None:
        """Set the reality model pack."""
        self._reality_pack = pack

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
        # If using reality pack with fill model, delegate to it
        if self._reality_pack:
            return self._should_fill_reality(order, tick)

        # Legacy behavior
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

    def _should_fill_reality(self, order: OrderRequest, tick: QuoteTick) -> bool:
        """Check if order should fill using reality pack fill model."""
        fill_model = self._reality_pack.fill_model

        # Map order type to reality OrderType
        order_type_map = {
            "MARKET": RealityOrderType.MARKET,
            "LIMIT": RealityOrderType.LIMIT,
            "STOP": RealityOrderType.STOP,
            "STOP_LIMIT": RealityOrderType.STOP_LIMIT,
        }
        order_type = order_type_map.get(order.order_type, RealityOrderType.MARKET)

        current_price = tick.last or tick.mid or 0

        # Simulate fill to check if it would succeed
        result = fill_model.simulate_fill(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            order_type=order_type,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            current_price=current_price,
            bid=tick.bid,
            ask=tick.ask,
            volume=tick.volume,
        )

        return result.filled

    def _fill_order(self, order: SimulatedOrder) -> None:
        """Execute fill for an order."""
        tick = self._latest_prices.get(order.request.symbol)
        if not tick:
            logger.warning(f"No price for {order.request.symbol}, cannot fill")
            return

        # Use reality pack if available
        if self._reality_pack:
            self._fill_order_reality(order, tick)
            return

        # Legacy fill logic
        self._fill_order_legacy(order, tick)

    def _fill_order_legacy(self, order: SimulatedOrder, tick: QuoteTick) -> None:
        """Legacy fill logic without reality pack."""
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

        self._create_and_record_fill(order, fill_price, order.request.quantity, commission)

    def _fill_order_reality(self, order: SimulatedOrder, tick: QuoteTick) -> None:
        """Fill order using reality pack models."""
        req = order.request
        symbol = req.symbol
        side = req.side
        quantity = req.quantity

        # Get mid/reference price
        mid_price = tick.mid or tick.last or 0
        if mid_price <= 0:
            logger.warning(f"Invalid price for {symbol}, cannot fill")
            return

        # Map order type
        order_type_map = {
            "MARKET": RealityOrderType.MARKET,
            "LIMIT": RealityOrderType.LIMIT,
            "STOP": RealityOrderType.STOP,
            "STOP_LIMIT": RealityOrderType.STOP_LIMIT,
        }
        order_type = order_type_map.get(req.order_type, RealityOrderType.MARKET)

        # 1. Use FillModel to simulate fill
        fill_result = self._reality_pack.fill_model.simulate_fill(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=req.limit_price,
            stop_price=req.stop_price,
            current_price=mid_price,
            bid=tick.bid,
            ask=tick.ask,
            volume=tick.volume,
        )

        if not fill_result.filled:
            logger.debug(f"Order not filled: {fill_result.reject_reason}")
            return

        filled_qty = fill_result.filled_quantity
        base_fill_price = fill_result.fill_price

        # 2. Apply SlippageModel to adjust price
        slippage_result = self._reality_pack.slippage_model.calculate(
            symbol=symbol,
            side=side,
            quantity=filled_qty,
            price=base_fill_price,
            bid=tick.bid,
            ask=tick.ask,
            volume=tick.volume,
        )
        fill_price = slippage_result.adjusted_price

        # 3. Calculate fees using FeeModel
        asset_type_map = {
            "stock": AssetType.STOCK,
            "STK": AssetType.STOCK,
            "STOCK": AssetType.STOCK,
            "option": AssetType.OPTION,
            "OPT": AssetType.OPTION,
            "OPTION": AssetType.OPTION,
            "future": AssetType.FUTURE,
            "FUT": AssetType.FUTURE,
            "FUTURE": AssetType.FUTURE,
        }
        asset_type = asset_type_map.get(req.asset_type or "stock", AssetType.STOCK)

        fee_breakdown = self._reality_pack.fee_model.calculate(
            symbol=symbol,
            quantity=filled_qty,
            price=fill_price,
            side=side,
            asset_type=asset_type,
        )
        commission = fee_breakdown.total

        # Create and record the fill
        self._create_and_record_fill(
            order, fill_price, filled_qty, commission, partial=fill_result.partial
        )

        # If partial fill, update remaining quantity
        if fill_result.partial and fill_result.remaining_quantity > 0:
            # Create a new pending order for remaining quantity
            logger.info(
                f"Partial fill: {filled_qty} filled, {fill_result.remaining_quantity} remaining"
            )

    def _create_and_record_fill(
        self,
        order: SimulatedOrder,
        fill_price: float,
        filled_qty: float,
        commission: float,
        partial: bool = False,
    ) -> None:
        """Create fill event and update internal state."""
        req = order.request

        # Create fill
        fill = TradeFill(
            symbol=req.symbol,
            underlying=req.underlying or req.symbol,
            side=req.side,
            quantity=filled_qty,
            price=fill_price,
            commission=commission,
            exec_id=f"EXEC-{uuid.uuid4().hex[:8]}",
            order_id=order.broker_order_id,
            asset_type=req.asset_type,
            multiplier=req.multiplier,
            source="SIMULATED",
            timestamp=self._clock.now(),
        )

        # Update order state
        order.filled_quantity += filled_qty
        order.avg_fill_price = fill_price
        order.commission += commission

        if order.filled_quantity >= req.quantity:
            order.status = "filled"
            # Move to filled orders
            client_id = req.client_order_id or order.broker_order_id
            self._pending_orders.pop(client_id, None)
            self._filled_orders[client_id] = order
        elif partial:
            order.status = "partial"

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
            f"Order filled: {req.client_order_id} "
            f"{req.side} {filled_qty} {req.symbol} "
            f"@ {fill_price:.4f} (comm: {commission:.2f})"
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
