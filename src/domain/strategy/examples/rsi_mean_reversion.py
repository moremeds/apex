"""
RSI Mean Reversion Strategy.

A counter-trend strategy that:
- Buys when RSI drops below oversold threshold (e.g., 30)
- Sells when RSI rises above overbought threshold (e.g., 70)
- Uses limit orders for better entry prices

This strategy demonstrates:
- Custom indicator calculation (RSI)
- Limit order submission
- Overbought/oversold signal logic
- Position sizing based on account
"""

from collections import deque
from typing import Deque, Optional, List
import uuid
import logging

from ..base import Strategy, StrategyContext
from ..registry import register_strategy
from ...events.domain_events import QuoteTick, BarData, TradeFill
from ...interfaces.execution_provider import OrderRequest

logger = logging.getLogger(__name__)


@register_strategy(
    "rsi_reversion",
    description="RSI Mean Reversion Strategy with Limit Orders",
    author="Apex",
    version="1.0",
)
class RsiMeanReversionStrategy(Strategy):
    """
    RSI Mean Reversion Strategy.

    Buys oversold conditions (RSI < 30) and sells overbought conditions (RSI > 70).
    Uses limit orders slightly better than market for improved fills.

    Parameters:
        rsi_period: RSI calculation period (default: 14)
        oversold: RSI level considered oversold (default: 30)
        overbought: RSI level considered overbought (default: 70)
        position_size: Number of shares per trade (default: 100)
        limit_offset_pct: Limit price offset from current price (default: 0.1%)

    Features demonstrated:
        - RSI indicator calculation
        - Limit orders with price offset
        - Mean reversion logic
    """

    def __init__(
        self,
        strategy_id: str,
        symbols: List[str],
        context: StrategyContext,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        position_size: float = 100,
        limit_offset_pct: float = 0.1,
    ):
        super().__init__(strategy_id, symbols, context)

        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.position_size = position_size
        self.limit_offset_pct = limit_offset_pct / 100.0  # Convert to decimal

        # Price history for RSI calculation
        self._prices: dict[str, Deque[float]] = {
            symbol: deque(maxlen=rsi_period + 1) for symbol in symbols
        }
        self._rsi: dict[str, Optional[float]] = {s: None for s in symbols}
        self._pending_orders: dict[str, str] = {}  # symbol -> order_id

    def on_start(self) -> None:
        """Initialize strategy."""
        logger.info(
            f"RSI Mean Reversion started: {self.symbols} "
            f"(period={self.rsi_period}, oversold={self.oversold}, overbought={self.overbought})"
        )

    def on_bar(self, bar: BarData) -> None:
        """Process bar data - preferred for daily strategies."""
        symbol = bar.symbol
        price = bar.close

        if price is None or price <= 0:
            return

        self._prices[symbol].append(price)

        # Need enough data for RSI
        if len(self._prices[symbol]) < self.rsi_period + 1:
            return

        # Calculate RSI
        self._rsi[symbol] = self._calculate_rsi(symbol)
        rsi = self._rsi[symbol]

        if rsi is None:
            return

        current_position = self.context.get_position_quantity(symbol)

        # Check for entry/exit signals
        if rsi < self.oversold and current_position <= 0:
            # Oversold - buy with limit order slightly below current price
            if symbol not in self._pending_orders:
                self._enter_long(symbol, price)

        elif rsi > self.overbought and current_position > 0:
            # Overbought - sell with limit order slightly above current price
            if symbol not in self._pending_orders:
                self._exit_position(symbol, price, current_position)

    def _calculate_rsi(self, symbol: str) -> Optional[float]:
        """Calculate RSI using Wilder's smoothing method."""
        prices = list(self._prices[symbol])
        if len(prices) < self.rsi_period + 1:
            return None

        # Calculate price changes
        changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

        # Separate gains and losses
        gains = [max(0, c) for c in changes]
        losses = [max(0, -c) for c in changes]

        # Average gain/loss (simple average for first calculation)
        avg_gain = sum(gains[-self.rsi_period:]) / self.rsi_period
        avg_loss = sum(losses[-self.rsi_period:]) / self.rsi_period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _enter_long(self, symbol: str, current_price: float) -> None:
        """Enter long position with limit order."""
        # Limit price slightly below current for better entry
        limit_price = current_price * (1 - self.limit_offset_pct)

        order_id = f"{self.strategy_id}-{uuid.uuid4().hex[:8]}"

        logger.info(
            f"[{self.strategy_id}] BUY LIMIT: {symbol} @ {limit_price:.2f} "
            f"(current={current_price:.2f}, RSI={self._rsi[symbol]:.1f})"
        )

        order = OrderRequest(
            symbol=symbol,
            side="BUY",
            quantity=self.position_size,
            order_type="LIMIT",
            limit_price=limit_price,
            client_order_id=order_id,
        )
        self.request_order(order)
        self._pending_orders[symbol] = order_id

    def _exit_position(self, symbol: str, current_price: float, quantity: float) -> None:
        """Exit position with limit order."""
        # Limit price slightly above current for better exit
        limit_price = current_price * (1 + self.limit_offset_pct)

        order_id = f"{self.strategy_id}-{uuid.uuid4().hex[:8]}"

        logger.info(
            f"[{self.strategy_id}] SELL LIMIT: {symbol} @ {limit_price:.2f} "
            f"(current={current_price:.2f}, RSI={self._rsi[symbol]:.1f})"
        )

        order = OrderRequest(
            symbol=symbol,
            side="SELL",
            quantity=abs(quantity),
            order_type="LIMIT",
            limit_price=limit_price,
            client_order_id=order_id,
        )
        self.request_order(order)
        self._pending_orders[symbol] = order_id

    def on_fill(self, fill: TradeFill) -> None:
        """Handle fill - clear pending order tracking."""
        symbol = fill.symbol
        if symbol in self._pending_orders:
            del self._pending_orders[symbol]

        logger.info(
            f"[{self.strategy_id}] FILL: "
            f"{fill.side} {fill.quantity} {symbol} @ {fill.price:.2f}"
        )

    def on_tick(self, tick: QuoteTick) -> None:
        """Optional: can also process ticks for intraday."""
        pass  # Using on_bar for this strategy

    def get_state(self) -> dict:
        """Get current strategy state for monitoring."""
        return {
            symbol: {
                "rsi": self._rsi[symbol],
                "price_count": len(self._prices[symbol]),
                "pending_order": self._pending_orders.get(symbol),
            }
            for symbol in self.symbols
        }
