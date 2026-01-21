"""
Momentum Breakout Strategy.

A trend-following strategy that:
- Enters on breakout above N-period high
- Uses ATR for position sizing and stop-loss placement
- Trails stop-loss as position moves in favor

This strategy demonstrates:
- ATR (Average True Range) volatility calculation
- Breakout detection (Donchian channels)
- Dynamic position sizing based on volatility
- Trailing stop management
- on_fill callback for trade tracking
"""

import logging
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

from ...events.domain_events import BarData, QuoteTick, TradeFill
from ...interfaces.execution_provider import OrderRequest
from ..base import Strategy, StrategyContext
from ..registry import register_strategy

logger = logging.getLogger(__name__)


@dataclass
class PositionInfo:
    """Track position entry and stop information."""

    entry_price: float
    quantity: float
    stop_price: float
    highest_price: float  # For trailing stop


@register_strategy(
    "momentum_breakout",
    description="Momentum Breakout Strategy with ATR Stops",
    author="Apex",
    version="1.0",
)
class MomentumBreakoutStrategy(Strategy):
    """
    Momentum Breakout Strategy.

    Enters when price breaks above N-period high.
    Uses ATR multiplier for stop-loss distance.
    Trails stop as price moves higher.

    Parameters:
        lookback: Period for high/low channel (default: 20)
        atr_period: ATR calculation period (default: 14)
        atr_multiplier: ATR multiplier for stop distance (default: 2.0)
        risk_per_trade: Fraction of capital to risk per trade (default: 0.02)
        trail_atr_multiplier: ATR multiplier for trailing stop (default: 1.5)

    Features demonstrated:
        - Donchian channel breakout
        - ATR-based volatility measurement
        - Volatility-adjusted position sizing
        - Trailing stop management
    """

    def __init__(
        self,
        strategy_id: str,
        symbols: List[str],
        context: StrategyContext,
        lookback: int = 20,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        risk_per_trade: float = 0.02,
        trail_atr_multiplier: float = 1.5,
    ):
        super().__init__(strategy_id, symbols, context)

        self.lookback = lookback
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.risk_per_trade = risk_per_trade
        self.trail_atr_multiplier = trail_atr_multiplier

        # Price history for breakout detection
        self._highs: dict[str, Deque[float]] = {
            symbol: deque(maxlen=lookback) for symbol in symbols
        }
        self._lows: dict[str, Deque[float]] = {symbol: deque(maxlen=lookback) for symbol in symbols}
        self._closes: dict[str, Deque[float]] = {
            symbol: deque(maxlen=atr_period + 1) for symbol in symbols
        }

        # True Range history for ATR
        self._true_ranges: dict[str, Deque[float]] = {
            symbol: deque(maxlen=atr_period) for symbol in symbols
        }

        # Position tracking
        self._positions: dict[str, Optional[PositionInfo]] = {s: None for s in symbols}
        self._atr: dict[str, Optional[float]] = {s: None for s in symbols}

    def on_start(self) -> None:
        """Initialize strategy."""
        logger.info(
            f"Momentum Breakout started: {self.symbols} "
            f"(lookback={self.lookback}, atr_period={self.atr_period}, "
            f"atr_mult={self.atr_multiplier})"
        )

    def on_bar(self, bar: BarData) -> None:
        """Process each bar - main strategy logic."""
        symbol = bar.symbol

        # Validate bar data - skip if OHLC values are missing
        if bar.high is None or bar.low is None or bar.close is None:
            return

        high = bar.high
        low = bar.low
        close = bar.close

        # Update price history
        self._highs[symbol].append(high)
        self._lows[symbol].append(low)
        self._closes[symbol].append(close)

        # Calculate True Range
        if len(self._closes[symbol]) >= 2:
            prev_close = list(self._closes[symbol])[-2]
            true_range = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
            self._true_ranges[symbol].append(true_range)

        # Calculate ATR
        if len(self._true_ranges[symbol]) >= self.atr_period:
            self._atr[symbol] = sum(self._true_ranges[symbol]) / len(self._true_ranges[symbol])

        # Need enough data for breakout detection
        if len(self._highs[symbol]) < self.lookback:
            return

        atr = self._atr[symbol]
        if atr is None or atr <= 0:
            return

        current_position = self.context.get_position_quantity(symbol)
        highest_high = max(list(self._highs[symbol])[:-1])  # Exclude current bar

        # Check for breakout entry
        if current_position <= 0 and close > highest_high:
            self._enter_breakout(symbol, close, atr)

        # Manage existing position
        elif current_position > 0 and self._positions[symbol]:
            self._manage_position(symbol, close, atr)

    def _enter_breakout(self, symbol: str, price: float, atr: float) -> None:
        """Enter on breakout with ATR-based position sizing."""
        # Calculate position size based on risk
        stop_distance = atr * self.atr_multiplier
        stop_price = price - stop_distance

        # Risk-based position sizing
        # Position size = (Account Risk) / (Stop Distance per share)
        account_value = 100000  # Could get from context.account if available
        risk_amount = account_value * self.risk_per_trade
        position_size = int(risk_amount / stop_distance)

        if position_size <= 0:
            logger.warning(f"[{self.strategy_id}] Position size too small for {symbol}")
            return

        order_id = f"{self.strategy_id}-{uuid.uuid4().hex[:8]}"

        logger.info(
            f"[{self.strategy_id}] BREAKOUT BUY: {symbol} @ {price:.2f} "
            f"(size={position_size}, stop={stop_price:.2f}, ATR={atr:.2f})"
        )

        order = OrderRequest(
            symbol=symbol,
            side="BUY",
            quantity=position_size,
            order_type="MARKET",
            client_order_id=order_id,
        )
        self.request_order(order)

        # Track position info
        self._positions[symbol] = PositionInfo(
            entry_price=price,
            quantity=position_size,
            stop_price=stop_price,
            highest_price=price,
        )

    def _manage_position(self, symbol: str, price: float, atr: float) -> None:
        """Manage existing position - check stop and trail."""
        pos = self._positions[symbol]
        if pos is None:
            return

        # Update highest price for trailing
        if price > pos.highest_price:
            pos.highest_price = price
            # Trail the stop
            new_stop = price - (atr * self.trail_atr_multiplier)
            if new_stop > pos.stop_price:
                pos.stop_price = new_stop
                logger.debug(
                    f"[{self.strategy_id}] Trail stop: {symbol} " f"new_stop={new_stop:.2f}"
                )

        # Check stop-loss
        if price <= pos.stop_price:
            self._exit_position(symbol, price, pos.quantity, "STOP_LOSS")

    def _exit_position(self, symbol: str, price: float, quantity: float, reason: str) -> None:
        """Exit position."""
        order_id = f"{self.strategy_id}-{uuid.uuid4().hex[:8]}"

        pos = self._positions[symbol]
        pnl = (price - pos.entry_price) * quantity if pos else 0

        logger.info(
            f"[{self.strategy_id}] EXIT {reason}: {symbol} @ {price:.2f} "
            f"(entry={(pos.entry_price if pos else 0):.2f}, pnl=${pnl:.2f})"
        )

        order = OrderRequest(
            symbol=symbol,
            side="SELL",
            quantity=abs(quantity),
            order_type="MARKET",
            client_order_id=order_id,
        )
        self.request_order(order)

        # Clear position tracking
        self._positions[symbol] = None

    def on_fill(self, fill: TradeFill) -> None:
        """Handle fill notification."""
        logger.info(
            f"[{self.strategy_id}] FILL: "
            f"{fill.side} {fill.quantity} {fill.symbol} @ {fill.price:.2f}"
        )

    def on_tick(self, tick: QuoteTick) -> None:
        """Process tick - check stops intraday if needed."""
        symbol = tick.symbol
        price = tick.last or tick.mid

        pos = self._positions[symbol]
        if price is None or pos is None:
            return

        # Real-time stop check
        if price <= pos.stop_price:
            current_qty = self.context.get_position_quantity(symbol)
            if current_qty > 0:
                self._exit_position(symbol, price, current_qty, "STOP_LOSS")

    def get_state(self) -> dict:
        """Get current strategy state for monitoring."""
        result: dict = {}
        for symbol in self.symbols:
            pos = self._positions[symbol]
            result[symbol] = {
                "atr": self._atr[symbol],
                "position": (
                    {
                        "entry": pos.entry_price,
                        "stop": pos.stop_price,
                        "highest": pos.highest_price,
                    }
                    if pos is not None
                    else None
                ),
                "channel_high": max(self._highs[symbol]) if self._highs[symbol] else None,
            }
        return result
