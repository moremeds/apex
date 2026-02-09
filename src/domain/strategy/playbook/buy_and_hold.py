"""
Buy and Hold Strategy.

A passive strategy that buys and holds positions for benchmarking.
Uses equal-weight allocation across symbols (portfolio_value / N / price)
for fair comparison against portfolio-fraction strategies.

Parameters (2):
    initial_capital: Portfolio capital for sizing (default 100_000)
    buy_on_start: If True, buys on first tick (default True)
"""

import logging
import uuid
from typing import List, Set

from ...events.domain_events import QuoteTick, TradeFill
from ...interfaces.execution_provider import OrderRequest
from ..base import Strategy, StrategyContext
from ..registry import register_strategy

logger = logging.getLogger(__name__)


@register_strategy(
    "buy_and_hold",
    description="Buy and Hold Benchmark Strategy",
    author="Apex",
    version="1.1",
)
class BuyAndHoldStrategy(Strategy):
    """
    Buy and Hold Strategy.

    Allocates equal dollar weight to each symbol on first tick and holds.
    Used as a benchmark to compare active strategy performance.

    Sizing: qty = int(initial_capital / len(symbols) / price)
    """

    def __init__(
        self,
        strategy_id: str,
        symbols: List[str],
        context: StrategyContext,
        initial_capital: float = 100_000.0,
        buy_on_start: bool = True,
    ):
        super().__init__(strategy_id, symbols, context)

        self.initial_capital = initial_capital
        self.buy_on_start = buy_on_start
        self._target_alloc_pct = 1.0 / max(len(symbols), 1)

        # Track which symbols we've bought
        self._bought: Set[str] = set()

    def on_start(self) -> None:
        logger.info(
            f"Buy and Hold Strategy started: {self.symbols} "
            f"(capital={self.initial_capital:.0f}, "
            f"alloc_per_symbol={self._target_alloc_pct:.1%})"
        )

    def on_tick(self, tick: QuoteTick) -> None:
        """Buy equal-weight allocation on first tick per symbol."""
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

        # Equal-weight sizing: capital * (1/N) / price
        alloc_value = self.initial_capital * self._target_alloc_pct
        qty = int(alloc_value / price)
        if qty < 1:
            logger.warning(
                f"[{self.strategy_id}] {symbol} @ {price:.2f}: "
                f"alloc {alloc_value:.0f} too small for 1 share"
            )
            self._bought.add(symbol)
            return

        logger.info(f"[{self.strategy_id}] Buying {qty} {symbol} @ {price:.2f}")

        order = OrderRequest(
            symbol=symbol,
            side="BUY",
            quantity=qty,
            order_type="MARKET",
            client_order_id=f"{self.strategy_id}-buy-{uuid.uuid4().hex[:8]}",
        )
        self.request_order(order)

        self._bought.add(symbol)

    def on_fill(self, fill: TradeFill) -> None:
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
