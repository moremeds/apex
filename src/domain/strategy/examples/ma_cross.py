"""
Moving Average Crossover Strategy.

A classic trend-following strategy that:
- Goes long when short MA crosses above long MA
- Closes position when short MA crosses below long MA

This strategy demonstrates:
- Tick-based signal generation
- Position management via context
- Order submission via request_order()
- Signal emission via emit_signal()
"""

import logging
import uuid
from collections import deque
from typing import Deque, List, Optional

from ...events.domain_events import QuoteTick
from ...interfaces.execution_provider import OrderRequest
from ..base import Strategy, StrategyContext, TradingSignal
from ..registry import register_strategy

logger = logging.getLogger(__name__)


@register_strategy(
    "ma_cross",
    description="Moving Average Crossover Strategy",
    author="Apex",
    version="1.0",
)
class MovingAverageCrossStrategy(Strategy):
    """
    Moving Average Crossover Strategy.

    Generates long signals when short MA crosses above long MA.
    Closes positions when short MA crosses below long MA.

    Parameters:
        short_window: Period for short moving average (default: 10)
        long_window: Period for long moving average (default: 50)
        position_size: Number of shares per trade (default: 100)
        use_signals: If True, emit signals; if False, submit orders directly

    Example:
        context = StrategyContext(clock=SystemClock())
        strategy = MovingAverageCrossStrategy(
            strategy_id="ma-aapl",
            symbols=["AAPL"],
            context=context,
            short_window=10,
            long_window=50,
            position_size=100,
        )
        strategy.start()
    """

    def __init__(
        self,
        strategy_id: str,
        symbols: List[str],
        context: StrategyContext,
        short_window: int = 10,
        long_window: int = 50,
        position_size: float = 100,
        use_signals: bool = False,
    ):
        """
        Initialize MA Cross strategy.

        Args:
            strategy_id: Unique identifier for this strategy instance.
            symbols: List of symbols to trade (typically one symbol).
            context: Strategy context with clock, positions, etc.
            short_window: Short MA period.
            long_window: Long MA period.
            position_size: Shares per trade.
            use_signals: Whether to emit signals or submit orders directly.
        """
        super().__init__(strategy_id, symbols, context)

        # Validate parameters
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window")

        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size
        self.use_signals = use_signals

        # Price history per symbol
        self._prices: dict[str, Deque[float]] = {
            symbol: deque(maxlen=long_window) for symbol in symbols
        }

        # MA values for monitoring
        self._short_ma: dict[str, Optional[float]] = {s: None for s in symbols}
        self._long_ma: dict[str, Optional[float]] = {s: None for s in symbols}

        # Track last signal to avoid duplicates
        self._last_signal: dict[str, Optional[str]] = {s: None for s in symbols}

    def on_start(self) -> None:
        """Initialize strategy."""
        logger.info(
            f"MA Cross Strategy started: {self.symbols}"
            f" (short={self.short_window}, long={self.long_window})"
        )

    def on_tick(self, tick: QuoteTick) -> None:
        """
        Process each price tick.

        Updates price history, calculates MAs, and generates signals
        when crossover conditions are met.
        """
        symbol = tick.symbol
        price = tick.last or tick.mid

        if price is None or price <= 0:
            return

        # Update price history
        self._prices[symbol].append(price)

        # Need enough data for both MAs
        if len(self._prices[symbol]) < self.long_window:
            return

        # Calculate MAs
        prices_list = list(self._prices[symbol])
        self._short_ma[symbol] = sum(prices_list[-self.short_window :]) / self.short_window
        self._long_ma[symbol] = sum(prices_list) / self.long_window

        # Check for crossover signals
        current_position = self.context.get_position_quantity(symbol)

        if self._short_ma[symbol] > self._long_ma[symbol]:
            # Bullish: short MA above long MA
            if current_position <= 0 and self._last_signal[symbol] != "LONG":
                self._go_long(symbol, tick)
                self._last_signal[symbol] = "LONG"

        elif self._short_ma[symbol] < self._long_ma[symbol]:
            # Bearish: short MA below long MA
            if current_position > 0 and self._last_signal[symbol] != "FLAT":
                self._go_flat(symbol, tick, current_position)
                self._last_signal[symbol] = "FLAT"

    def _go_long(self, symbol: str, tick: QuoteTick) -> None:
        """Generate long entry signal/order."""
        logger.info(
            f"[{self.strategy_id}] LONG signal: {symbol} @ {tick.last:.2f} "
            f"(short_ma={self._short_ma[symbol]:.2f}, long_ma={self._long_ma[symbol]:.2f})"
        )

        if self.use_signals:
            signal = TradingSignal(
                signal_id=f"{self.strategy_id}-{uuid.uuid4().hex[:8]}",
                symbol=symbol,
                direction="LONG",
                strength=1.0,
                target_quantity=self.position_size,
                target_price=tick.last,
                reason="MA crossover bullish",
                metadata={
                    "short_ma": self._short_ma[symbol],
                    "long_ma": self._long_ma[symbol],
                },
                timestamp=self.context.now(),
            )
            self.emit_signal(signal)
        else:
            order = OrderRequest(
                symbol=symbol,
                side="BUY",
                quantity=self.position_size,
                order_type="MARKET",
                client_order_id=f"{self.strategy_id}-{uuid.uuid4().hex[:8]}",
            )
            self.request_order(order)

    def _go_flat(self, symbol: str, tick: QuoteTick, current_qty: float) -> None:
        """Generate exit signal/order."""
        logger.info(
            f"[{self.strategy_id}] FLAT signal: {symbol} @ {tick.last:.2f} "
            f"(short_ma={self._short_ma[symbol]:.2f}, long_ma={self._long_ma[symbol]:.2f})"
        )

        if self.use_signals:
            signal = TradingSignal(
                signal_id=f"{self.strategy_id}-{uuid.uuid4().hex[:8]}",
                symbol=symbol,
                direction="FLAT",
                strength=1.0,
                target_quantity=0,
                target_price=tick.last,
                reason="MA crossover bearish",
                metadata={
                    "short_ma": self._short_ma[symbol],
                    "long_ma": self._long_ma[symbol],
                },
                timestamp=self.context.now(),
            )
            self.emit_signal(signal)
        else:
            # Sell to close position
            order = OrderRequest(
                symbol=symbol,
                side="SELL",
                quantity=abs(current_qty),
                order_type="MARKET",
                client_order_id=f"{self.strategy_id}-{uuid.uuid4().hex[:8]}",
            )
            self.request_order(order)

    def on_fill(self, fill) -> None:
        """Log fills."""
        logger.info(
            f"[{self.strategy_id}] FILL: "
            f"{fill.side} {fill.quantity} {fill.symbol} @ {fill.price:.2f}"
        )

    def get_state(self) -> dict:
        """
        Get current strategy state for monitoring.

        Returns:
            Dictionary with current MAs and signals per symbol.
        """
        return {
            symbol: {
                "short_ma": self._short_ma[symbol],
                "long_ma": self._long_ma[symbol],
                "last_signal": self._last_signal[symbol],
                "price_count": len(self._prices[symbol]),
            }
            for symbol in self.symbols
        }
