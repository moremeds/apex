"""
Trade tracker with entry/exit matching.

Tracks fills and matches them into complete trades (round-trips).
Supports FIFO, LIFO, and average cost matching methods.

Usage:
    tracker = TradeTracker(matching_method="FIFO")

    # Record fills
    tracker.record_fill(fill1)  # Entry
    tracker.record_fill(fill2)  # Exit

    # Get completed trades
    trades = tracker.get_completed_trades()

    # Calculate trade metrics
    metrics = tracker.calculate_metrics()
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
import uuid
import logging

from ...domain.events.domain_events import TradeFill
from ...domain.backtest.backtest_result import TradeRecord, TradeMetrics

logger = logging.getLogger(__name__)


class MatchingMethod(Enum):
    """Position matching method for trade attribution."""
    FIFO = "fifo"  # First-In-First-Out
    LIFO = "lifo"  # Last-In-First-Out
    AVERAGE = "average"  # Average cost


@dataclass
class OpenPosition:
    """Represents an open (partial) position from a fill."""
    fill_id: str
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    remaining_quantity: float
    price: float
    commission: float
    timestamp: datetime
    multiplier: int = 1

    @property
    def is_long(self) -> bool:
        return self.side == "BUY"


class TradeTracker:
    """
    Tracks fills and matches them into completed trades.

    Supports multiple matching methods:
    - FIFO: First entry matched with first exit
    - LIFO: Last entry matched with first exit
    - AVERAGE: Average cost method

    Example:
        tracker = TradeTracker()

        # BUY 100 @ $10 (entry)
        tracker.record_fill(TradeFill(symbol="AAPL", side="BUY", quantity=100, price=10.0, ...))

        # SELL 100 @ $12 (exit)
        tracker.record_fill(TradeFill(symbol="AAPL", side="SELL", quantity=100, price=12.0, ...))

        # Get completed trade
        trades = tracker.get_completed_trades()
        # trades[0].pnl == 200.0 (100 * ($12 - $10))
    """

    def __init__(
        self,
        matching_method: str = "FIFO",
    ):
        """
        Initialize trade tracker.

        Args:
            matching_method: Matching method (FIFO, LIFO, AVERAGE).
        """
        self._method = MatchingMethod(matching_method.lower())

        # Open positions by symbol: List[OpenPosition]
        self._open_positions: Dict[str, List[OpenPosition]] = {}

        # Completed trades
        self._completed_trades: List[TradeRecord] = []

        # All fills (for audit)
        self._all_fills: List[TradeFill] = []

    def record_fill(self, fill: TradeFill) -> Optional[TradeRecord]:
        """
        Record a fill and match against open positions.

        Args:
            fill: Trade fill event.

        Returns:
            TradeRecord if a trade was completed, None otherwise.
        """
        self._all_fills.append(fill)

        symbol = fill.symbol
        if symbol not in self._open_positions:
            self._open_positions[symbol] = []

        open_positions = self._open_positions[symbol]
        multiplier = getattr(fill, 'multiplier', 1) or 1

        # Check if this fill opens or closes a position
        is_opening = self._is_opening_fill(fill, open_positions)

        if is_opening:
            # Add to open positions
            position = OpenPosition(
                fill_id=fill.exec_id or str(uuid.uuid4().hex[:8]),
                symbol=symbol,
                side=fill.side,
                quantity=fill.quantity,
                remaining_quantity=fill.quantity,
                price=fill.price,
                commission=fill.commission,
                timestamp=fill.timestamp,
                multiplier=multiplier,
            )
            open_positions.append(position)
            logger.debug(f"Opened position: {fill.side} {fill.quantity} {symbol} @ {fill.price}")
            return None

        # This fill closes (or partially closes) existing positions
        trade = self._match_and_close(fill, open_positions, multiplier)
        return trade

    def _is_opening_fill(
        self,
        fill: TradeFill,
        open_positions: List[OpenPosition],
    ) -> bool:
        """Determine if fill opens or closes a position."""
        if not open_positions:
            return True  # No open positions, must be opening

        # Check if opposite side exists
        for pos in open_positions:
            if pos.remaining_quantity > 0:
                # If existing position is opposite side, this is closing
                if (pos.side == "BUY" and fill.side == "SELL") or \
                   (pos.side == "SELL" and fill.side == "BUY"):
                    return False

        # Same side as existing positions = adding to position
        return True

    def _match_and_close(
        self,
        fill: TradeFill,
        open_positions: List[OpenPosition],
        multiplier: int,
    ) -> Optional[TradeRecord]:
        """Match fill against open positions and create trade record."""
        remaining_to_close = fill.quantity
        total_pnl = 0.0
        total_entry_commission = 0.0
        total_exit_commission = fill.commission
        entry_qty_weighted_price = 0.0
        total_entry_qty = 0.0
        first_entry_time = None
        trade_side = "LONG" if fill.side == "SELL" else "SHORT"

        # Get positions to match based on method
        if self._method == MatchingMethod.LIFO:
            positions_to_check = reversed(open_positions)
        else:
            positions_to_check = open_positions

        for pos in list(positions_to_check):
            if remaining_to_close <= 0:
                break

            # Skip same-side positions or fully closed
            if pos.remaining_quantity <= 0:
                continue
            if pos.side == fill.side:
                continue

            # Calculate how much to close
            close_qty = min(remaining_to_close, pos.remaining_quantity)

            # Calculate P&L for this portion
            if pos.side == "BUY":  # Long position being closed by SELL
                pnl = close_qty * (fill.price - pos.price) * multiplier
            else:  # Short position being closed by BUY
                pnl = close_qty * (pos.price - fill.price) * multiplier

            total_pnl += pnl
            total_entry_commission += (pos.commission * close_qty / pos.quantity)
            entry_qty_weighted_price += pos.price * close_qty
            total_entry_qty += close_qty

            if first_entry_time is None:
                first_entry_time = pos.timestamp

            # Update remaining quantities
            pos.remaining_quantity -= close_qty
            remaining_to_close -= close_qty

            logger.debug(
                f"Matched {close_qty} {fill.symbol}: "
                f"entry={pos.price:.2f}, exit={fill.price:.2f}, pnl={pnl:.2f}"
            )

        # Clean up fully closed positions
        self._open_positions[fill.symbol] = [
            p for p in open_positions if p.remaining_quantity > 0
        ]

        # If we closed any quantity, create a trade record
        if total_entry_qty > 0:
            avg_entry_price = entry_qty_weighted_price / total_entry_qty
            total_commission = total_entry_commission + total_exit_commission

            # Calculate P&L percentage
            entry_value = total_entry_qty * avg_entry_price * multiplier
            pnl_pct = (total_pnl / entry_value * 100) if entry_value > 0 else 0

            trade = TradeRecord(
                trade_id=f"trade-{uuid.uuid4().hex[:8]}",
                symbol=fill.symbol,
                side=trade_side,
                entry_time=first_entry_time,
                exit_time=fill.timestamp,
                entry_price=avg_entry_price,
                exit_price=fill.price,
                quantity=total_entry_qty,
                pnl=total_pnl - total_commission,  # Net P&L
                pnl_pct=pnl_pct,
                commission=total_commission,
            )

            self._completed_trades.append(trade)
            logger.info(
                f"Completed trade: {trade.side} {trade.quantity} {fill.symbol} "
                f"P&L=${trade.pnl:.2f} ({trade.pnl_pct:+.2f}%)"
            )
            return trade

        # If fill wasn't matched (e.g., opening opposite position), add it
        if remaining_to_close > 0:
            position = OpenPosition(
                fill_id=fill.exec_id or str(uuid.uuid4().hex[:8]),
                symbol=fill.symbol,
                side=fill.side,
                quantity=remaining_to_close,
                remaining_quantity=remaining_to_close,
                price=fill.price,
                commission=fill.commission * (remaining_to_close / fill.quantity),
                timestamp=fill.timestamp,
                multiplier=multiplier,
            )
            self._open_positions[fill.symbol].append(position)

        return None

    def get_completed_trades(self) -> List[TradeRecord]:
        """Get all completed trades."""
        return self._completed_trades.copy()

    def get_open_positions(self) -> Dict[str, List[OpenPosition]]:
        """Get all open positions."""
        return {
            symbol: [p for p in positions if p.remaining_quantity > 0]
            for symbol, positions in self._open_positions.items()
        }

    def get_open_quantity(self, symbol: str) -> float:
        """Get total open quantity for a symbol."""
        positions = self._open_positions.get(symbol, [])
        return sum(p.remaining_quantity for p in positions if p.remaining_quantity > 0)

    def calculate_metrics(self) -> TradeMetrics:
        """Calculate trade metrics from completed trades."""
        trades = self._completed_trades

        if not trades:
            return TradeMetrics()

        total_trades = len(trades)
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]

        winning_trades = len(winners)
        losing_trades = len(losers)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        gross_profit = sum(t.pnl for t in winners)
        gross_loss = abs(sum(t.pnl for t in losers))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        avg_win = (gross_profit / winning_trades) if winning_trades > 0 else 0
        avg_loss = (-gross_loss / losing_trades) if losing_trades > 0 else 0
        avg_trade = sum(t.pnl for t in trades) / total_trades

        largest_win = max((t.pnl for t in winners), default=0)
        largest_loss = min((t.pnl for t in losers), default=0)

        # Calculate average duration
        durations = [t.duration.total_seconds() for t in trades if t.entry_time and t.exit_time]
        avg_duration = sum(durations) / len(durations) if durations else 0

        # Calculate consecutive wins/losses
        max_consecutive_wins, max_consecutive_losses = self._calculate_streaks(trades)

        return TradeMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor if profit_factor != float('inf') else 0,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade=avg_trade,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_trade_duration_seconds=avg_duration,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
        )

    def _calculate_streaks(self, trades: List[TradeRecord]) -> tuple:
        """Calculate max consecutive wins and losses."""
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in trades:
            if trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return max_wins, max_losses

    def reset(self) -> None:
        """Reset tracker state."""
        self._open_positions.clear()
        self._completed_trades.clear()
        self._all_fills.clear()
