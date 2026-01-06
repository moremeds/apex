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

Architecture:
    - simulated.py: Main coordinator and public interface
    - ledger.py: Position tracking and P&L calculations
    - order_matching.py: Order submission, matching, and fill logic

Usage:
    clock = SimulatedClock(start_time)

    # Simple mode
    execution = SimulatedExecution(clock, fill_model=FillModel.IMMEDIATE)

    # Realistic mode with reality pack
    from domain.reality import create_ib_pack
    execution = SimulatedExecution(clock, reality_pack=create_ib_pack())

    # Update with market data
    execution.update_price(tick)

    # Submit orders
    order = OrderRequest(symbol="AAPL", side="BUY", quantity=100)
    broker_id = await execution.submit_order(order)

    # Get fills
    fills = execution.get_pending_fills()
"""

from __future__ import annotations
from typing import Dict, List, Optional, Callable
import logging

from ...domain.clock import Clock
from ...domain.events.domain_events import QuoteTick, TradeFill
from ...domain.interfaces.execution_provider import OrderRequest, OrderResult
from ...domain.reality import RealityModelPack

from .ledger import PositionLedger, SimulatedPosition
from .order_matching import OrderMatcher, FillModel, SimulatedOrder

logger = logging.getLogger(__name__)

# Re-export for convenience
__all__ = ["SimulatedExecution", "FillModel", "SimulatedOrder", "SimulatedPosition"]


class SimulatedExecution:
    """
    Simulated execution engine for backtesting.

    Handles order submission, matching, and fill generation.
    Tracks positions and P&L internally.

    This is a thin coordinator that delegates to:
    - PositionLedger: Position state and valuation
    - OrderMatcher: Order submission and fill logic
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

        # Initialize components
        self._ledger = PositionLedger()
        self._matcher = OrderMatcher(
            clock=clock,
            ledger=self._ledger,
            fill_model=fill_model,
            slippage_bps=slippage_bps,
            commission_per_share=commission_per_share,
            min_commission=min_commission,
            reality_pack=reality_pack,
        )

    @property
    def reality_pack(self) -> Optional[RealityModelPack]:
        """Get the reality model pack if configured."""
        return self._matcher.reality_pack

    @reality_pack.setter
    def reality_pack(self, pack: Optional[RealityModelPack]) -> None:
        """Set the reality model pack."""
        self._matcher.reality_pack = pack

    def update_price(self, tick: QuoteTick) -> None:
        """
        Update latest price for a symbol.

        Called by backtest engine as ticks/bars are processed.
        Triggers fill matching for pending orders.

        Args:
            tick: Quote tick with latest prices.
        """
        self._ledger.update_price(tick)
        self._matcher.match_orders(tick.symbol)

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
        return self._matcher.submit_order(order)

    async def cancel_order(self, client_order_id: str) -> OrderResult:
        """
        Cancel a pending order.

        Args:
            client_order_id: Client order ID to cancel.

        Returns:
            OrderResult indicating success/failure.
        """
        return self._matcher.cancel_order(client_order_id)

    def get_pending_fills(self) -> List[TradeFill]:
        """
        Get and clear pending fills.

        Returns:
            List of fills to be processed.
        """
        return self._matcher.get_pending_fills()

    def get_position(self, symbol: str) -> Optional[SimulatedPosition]:
        """Get current position for a symbol."""
        return self._ledger.get_position(symbol)

    def get_all_positions(self) -> Dict[str, SimulatedPosition]:
        """Get all positions."""
        return self._ledger.get_all_positions()

    def set_fill_callback(self, callback: Callable[[TradeFill], None]) -> None:
        """Set callback for fills."""
        self._matcher.set_fill_callback(callback)

    def get_total_realized_pnl(self) -> float:
        """Get total realized P&L across all positions."""
        return self._ledger.get_total_realized_pnl()

    def get_position_value(self, symbol: str) -> float:
        """Get current market value of a position (includes multiplier for options/futures)."""
        return self._ledger.get_position_value(symbol)

    def get_total_position_value(self) -> float:
        """Get total market value of all positions."""
        return self._ledger.get_total_position_value()

    def reset(self) -> None:
        """Reset execution state."""
        self._matcher.reset()
        self._ledger.reset()
        logger.debug("SimulatedExecution reset")
