"""
Order matching logic for backtest execution.

Handles order submission, fill simulation, and matching logic.
Supports both legacy (simple) and reality pack (realistic) modes.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

from ...domain.clock import Clock
from ...domain.events.domain_events import QuoteTick, TradeFill
from ...domain.interfaces.execution_provider import OrderRequest, OrderResult

# Reality models (optional)
from ...domain.reality import (
    AssetType,
)
from ...domain.reality import OrderType as RealityOrderType
from ...domain.reality import (
    RealityModelPack,
)

if TYPE_CHECKING:
    from .ledger import PositionLedger

logger = logging.getLogger(__name__)


class FillModel(Enum):
    """Fill model for order execution."""

    IMMEDIATE = "immediate"  # Fill at current price instantly
    NEXT_BAR = "next_bar"  # Fill at next bar's open
    NEXT_BAR_OPEN = "next_bar_open"  # Fill at next bar's open + slippage
    SLIPPAGE = "slippage"  # Fill with slippage model


@dataclass
class OrderMatcherConfig:
    """Configuration for OrderMatcher execution realism."""

    fill_model: FillModel = FillModel.IMMEDIATE
    slippage_bps: float = 5.0
    commission_per_share: float = 0.005
    min_commission: float = 1.0


# Tier presets for execution realism levels
TIER_A_DEFAULTS = OrderMatcherConfig(
    fill_model=FillModel.IMMEDIATE,
    slippage_bps=5.0,
    commission_per_share=0.005,
)

TIER_B_DEFAULTS = OrderMatcherConfig(
    fill_model=FillModel.NEXT_BAR_OPEN,
    slippage_bps=15.0,
    commission_per_share=0.005,
    min_commission=1.0,
)


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


class OrderMatcher:
    """
    Handles order matching and fill simulation.

    Supports two modes:
    1. Legacy mode: Simple fill with configurable slippage/commission
    2. Reality pack mode: Realistic simulation with fee/slippage/fill models
    """

    def __init__(
        self,
        clock: Clock,
        ledger: "PositionLedger",
        fill_model: FillModel = FillModel.IMMEDIATE,
        slippage_bps: float = 5.0,
        commission_per_share: float = 0.005,
        min_commission: float = 1.0,
        reality_pack: Optional[RealityModelPack] = None,
    ):
        """
        Initialize order matcher.

        Args:
            clock: Clock for timestamps.
            ledger: Position ledger for position updates.
            fill_model: How orders are filled (legacy mode).
            slippage_bps: Slippage in basis points (legacy mode).
            commission_per_share: Commission per share (legacy mode).
            min_commission: Minimum commission per order (legacy mode).
            reality_pack: Optional RealityModelPack for realistic simulation.
        """
        self._clock = clock
        self._ledger = ledger
        self._reality_pack = reality_pack

        # Legacy parameters
        self._fill_model = fill_model
        self._slippage_bps = slippage_bps
        self._commission_per_share = commission_per_share
        self._min_commission = min_commission

        # Order tracking
        self._pending_orders: Dict[str, SimulatedOrder] = {}
        self._filled_orders: Dict[str, SimulatedOrder] = {}
        self._cancelled_orders: Dict[str, SimulatedOrder] = {}

        # Deferred orders for NEXT_BAR_OPEN: filled at the next bar's open
        self._deferred_orders: List[SimulatedOrder] = []

        # Fill queue
        self._pending_fills: List[TradeFill] = []

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

    def submit_order(self, order: OrderRequest) -> OrderResult:
        """
        Submit order for simulated execution.

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

        logger.debug(f"Simulated order submitted: {order.client_order_id} -> {broker_order_id}")

        # Immediate fill for market orders (legacy IMMEDIATE mode only)
        if self._fill_model == FillModel.IMMEDIATE and order.order_type == "MARKET":
            self._fill_order(sim_order)
        elif self._fill_model == FillModel.NEXT_BAR_OPEN and order.order_type == "MARKET":
            # Defer to next bar's open price
            self._deferred_orders.append(sim_order)

        return OrderResult(
            success=True,
            order_id=broker_order_id,
            message="Order submitted",
        )

    def cancel_order(self, client_order_id: str) -> OrderResult:
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
        """Get and clear pending fills."""
        fills = self._pending_fills.copy()
        self._pending_fills.clear()
        return fills

    def set_fill_callback(self, callback: Callable[[TradeFill], None]) -> None:
        """Set callback for fills."""
        self._fill_callback = callback

    def match_orders(self, symbol: str) -> None:
        """Match pending orders for a symbol against current price."""
        tick = self._ledger.get_latest_price(symbol)
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

    def fill_deferred_at_open(self, open_prices: Dict[str, float]) -> None:
        """
        Fill deferred NEXT_BAR_OPEN orders at the given open prices + slippage.

        Called by BacktestEngine at the start of each new bar, before strategy
        processing, with the bar's open prices.

        Args:
            open_prices: Dict of symbol -> bar open price.
        """
        if not self._deferred_orders:
            return

        filled: List[SimulatedOrder] = []
        for order in self._deferred_orders:
            symbol = order.request.symbol
            open_price = open_prices.get(symbol)
            if open_price is None or open_price <= 0:
                logger.warning(f"No open price for {symbol}, cannot fill deferred order")
                continue

            # Apply slippage to open price
            slippage = open_price * (self._slippage_bps / 10000)
            if order.request.side == "BUY":
                fill_price = open_price + slippage
            else:
                fill_price = open_price - slippage

            # Calculate commission
            commission = max(
                order.request.quantity * self._commission_per_share,
                self._min_commission,
            )

            self._create_and_record_fill(order, fill_price, order.request.quantity, commission)
            filled.append(order)

        # Clear processed deferred orders
        self._deferred_orders = [o for o in self._deferred_orders if o not in filled]

    def _should_fill(self, order: OrderRequest, tick: QuoteTick) -> bool:
        """Determine if order should be filled at current tick."""
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
        assert self._reality_pack is not None
        fill_model = self._reality_pack.fill_model

        order_type_map = {
            "MARKET": RealityOrderType.MARKET,
            "LIMIT": RealityOrderType.LIMIT,
            "STOP": RealityOrderType.STOP,
            "STOP_LIMIT": RealityOrderType.STOP_LIMIT,
        }
        order_type = order_type_map.get(order.order_type, RealityOrderType.MARKET)

        current_price = tick.last or tick.mid or 0

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
        tick = self._ledger.get_latest_price(order.request.symbol)
        if not tick:
            logger.warning(f"No price for {order.request.symbol}, cannot fill")
            return

        if self._reality_pack:
            self._fill_order_reality(order, tick)
            return

        self._fill_order_legacy(order, tick)

    def _fill_order_legacy(self, order: SimulatedOrder, tick: QuoteTick) -> None:
        """Legacy fill logic without reality pack."""
        if order.request.side == "BUY":
            base_price = tick.ask or tick.last or 0
        else:
            base_price = tick.bid or tick.last or 0

        if base_price <= 0:
            logger.warning(f"Invalid price for {order.request.symbol}, cannot fill")
            return

        # Apply slippage
        if self._fill_model in (FillModel.SLIPPAGE, FillModel.NEXT_BAR_OPEN):
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

        mid_price = tick.mid or tick.last or 0
        if mid_price <= 0:
            logger.warning(f"Invalid price for {symbol}, cannot fill")
            return

        order_type_map = {
            "MARKET": RealityOrderType.MARKET,
            "LIMIT": RealityOrderType.LIMIT,
            "STOP": RealityOrderType.STOP,
            "STOP_LIMIT": RealityOrderType.STOP_LIMIT,
        }
        order_type = order_type_map.get(req.order_type, RealityOrderType.MARKET)

        # 1. Simulate fill
        assert self._reality_pack is not None
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

        # 2. Apply slippage
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

        # 3. Calculate fees
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

        self._create_and_record_fill(
            order, fill_price, filled_qty, commission, partial=fill_result.partial
        )

        if fill_result.partial and fill_result.remaining_quantity > 0:
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
            client_id = req.client_order_id or order.broker_order_id
            self._pending_orders.pop(client_id, None)
            self._filled_orders[client_id] = order
        elif partial:
            order.status = "partial"

        # Update position ledger
        self._ledger.update_position(fill)

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

    def reset(self) -> None:
        """Reset all order state."""
        self._pending_orders.clear()
        self._filled_orders.clear()
        self._cancelled_orders.clear()
        self._deferred_orders.clear()
        self._pending_fills.clear()
        logger.debug("OrderMatcher reset")
