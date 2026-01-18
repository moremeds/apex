"""
Position ledger for backtest execution.

Tracks positions, average prices, and realized P&L.
Pure state container with no side effects beyond state updates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from ...domain.events.domain_events import QuoteTick, TradeFill

logger = logging.getLogger(__name__)


@dataclass
class SimulatedPosition:
    """
    Internal position tracking for backtests.

    Tracks quantity, average price, realized P&L, and contract multiplier.
    """

    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    multiplier: int = 1  # Contract multiplier (100 for options, 1 for stocks)


class PositionLedger:
    """
    Manages position state for simulated execution.

    Provides:
    - Position updates from fills
    - Realized P&L tracking
    - Position valuation with latest prices
    """

    def __init__(self) -> None:
        """Initialize empty ledger."""
        self._positions: Dict[str, SimulatedPosition] = {}
        self._latest_prices: Dict[str, QuoteTick] = {}

    def update_price(self, tick: QuoteTick) -> None:
        """
        Update latest price for a symbol.

        Args:
            tick: Quote tick with latest prices.
        """
        self._latest_prices[tick.symbol] = tick

    def get_latest_price(self, symbol: str) -> Optional[QuoteTick]:
        """Get latest price tick for a symbol."""
        return self._latest_prices.get(symbol)

    def update_position(self, fill: TradeFill) -> None:
        """
        Update position from a fill.

        Handles:
        - Opening new positions
        - Adding to existing positions
        - Reducing/closing positions with P&L calculation

        Args:
            fill: Trade fill event.
        """
        symbol = fill.symbol
        multiplier = getattr(fill, "multiplier", 1) or 1

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

    def get_position(self, symbol: str) -> Optional[SimulatedPosition]:
        """Get current position for a symbol."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> Dict[str, SimulatedPosition]:
        """Get all positions (copy)."""
        return self._positions.copy()

    def get_total_realized_pnl(self) -> float:
        """Get total realized P&L across all positions."""
        return sum(pos.realized_pnl for pos in self._positions.values())

    def get_position_value(self, symbol: str) -> float:
        """
        Get current market value of a position.

        Includes multiplier for options/futures.

        Args:
            symbol: Symbol to value.

        Returns:
            Market value (can be negative for short positions).
        """
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
        """Reset all positions and prices."""
        self._positions.clear()
        self._latest_prices.clear()
        logger.debug("PositionLedger reset")
