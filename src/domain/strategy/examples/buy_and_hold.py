"""
Buy and Hold Strategy.

A passive strategy that buys and holds positions for benchmarking.
Useful for comparing active strategy performance against a simple baseline.

This strategy demonstrates:
- on_start() for initial position entry
- Simple position management
- State tracking for first tick
"""

from typing import List, Set
import uuid
import logging

from ..base import Strategy, StrategyContext
from ..registry import register_strategy
from ...events.domain_events import QuoteTick
from ...interfaces.execution_provider import OrderRequest

logger = logging.getLogger(__name__)


@register_strategy(
    "buy_and_hold",
    description="Buy and Hold Benchmark Strategy",
    author="Apex",
    version="1.0",
)
class BuyAndHoldStrategy(Strategy):
    """
    Buy and Hold Strategy.

    Buys the specified quantity of each symbol on first tick and holds.
    Used as a benchmark to compare active strategy performance.

    Parameters:
        position_size: Number of shares to buy per symbol (default: 100)
        buy_on_start: If True, buys on first tick (default: True)

    Example:
        context = StrategyContext(clock=SystemClock())
        strategy = BuyAndHoldStrategy(
            strategy_id="bnh-aapl",
            symbols=["AAPL", "MSFT"],
            context=context,
            position_size=100,
        )
        strategy.start()
    """

    def __init__(
        self,
        strategy_id: str,
        symbols: List[str],
        context: StrategyContext,
        position_size: float = 100,
        buy_on_start: bool = True,
    ):
        """
        Initialize Buy and Hold strategy.

        Args:
            strategy_id: Unique identifier for this strategy instance.
            symbols: List of symbols to buy and hold.
            context: Strategy context with clock, positions, etc.
            position_size: Shares to buy per symbol.
            buy_on_start: Whether to buy on first tick.
        """
        super().__init__(strategy_id, symbols, context)

        self.position_size = position_size
        self.buy_on_start = buy_on_start

        # Track which symbols we've bought
        self._bought: Set[str] = set()

    def on_start(self) -> None:
        """Log strategy start."""
        logger.info(
            f"Buy and Hold Strategy started: {self.symbols} "
            f"(position_size={self.position_size})"
        )

    def on_tick(self, tick: QuoteTick) -> None:
        """
        Process tick and buy if not yet bought.

        Only triggers on first tick for each symbol.
        """
        symbol = tick.symbol

        if not self.buy_on_start:
            return

        if symbol in self._bought:
            return

        # Check if we already have a position
        current_position = self.context.get_position_quantity(symbol)
        if current_position > 0:
            self._bought.add(symbol)
            return

        # Need valid price
        price = tick.last or tick.mid
        if price is None or price <= 0:
            return

        # Buy
        logger.info(
            f"[{self.strategy_id}] Buying {self.position_size} {symbol} @ {price:.2f}"
        )

        order = OrderRequest(
            symbol=symbol,
            side="BUY",
            quantity=self.position_size,
            order_type="MARKET",
            client_order_id=f"{self.strategy_id}-buy-{uuid.uuid4().hex[:8]}",
        )
        self.request_order(order)

        self._bought.add(symbol)

    def on_fill(self, fill) -> None:
        """Log fills."""
        logger.info(
            f"[{self.strategy_id}] FILL: "
            f"{fill.side} {fill.quantity} {fill.symbol} @ {fill.price:.2f}"
        )

    def get_state(self) -> dict:
        """Get current strategy state."""
        return {
            "bought": list(self._bought),
            "pending": [s for s in self.symbols if s not in self._bought],
        }
