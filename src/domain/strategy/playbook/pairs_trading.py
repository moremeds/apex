"""
Pairs Trading Strategy.

A market-neutral strategy that:
- Trades the spread between two correlated assets
- Goes long spread when it's below historical mean (buy A, sell B)
- Goes short spread when it's above historical mean (sell A, buy B)
- Uses z-score for entry/exit signals

This strategy demonstrates:
- Multi-symbol coordination
- Spread calculation and z-score
- Simultaneous long/short positions
- Market-neutral portfolio construction
- on_bar with synchronized data
"""

import logging
import statistics
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from ...events.domain_events import BarData, QuoteTick, TradeFill
from ...interfaces.execution_provider import OrderRequest
from ..base import Strategy, StrategyContext
from ..registry import register_strategy

logger = logging.getLogger(__name__)


@dataclass
class SpreadPosition:
    """Track pairs position."""

    long_symbol: str
    short_symbol: str
    quantity: float
    entry_spread: float
    entry_zscore: float


@register_strategy(
    "pairs_trading",
    description="Statistical Arbitrage Pairs Trading Strategy",
    author="Apex",
    version="1.0",
)
class PairsTradingStrategy(Strategy):
    """
    Pairs Trading Strategy.

    Trades the spread between two correlated symbols using z-score signals.
    Requires exactly 2 symbols to be provided.

    Parameters:
        lookback: Period for mean/std calculation (default: 20)
        entry_zscore: Z-score threshold for entry (default: 2.0)
        exit_zscore: Z-score threshold for exit (default: 0.5)
        position_size: Dollar amount per leg (default: 10000)

    Features demonstrated:
        - Multi-symbol trading
        - Spread calculation
        - Z-score normalization
        - Market-neutral positions
        - Synchronized bar processing
    """

    def __init__(
        self,
        strategy_id: str,
        symbols: List[str],
        context: StrategyContext,
        lookback: int = 20,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        position_size: float = 10000,
    ):
        if len(symbols) != 2:
            raise ValueError("Pairs trading requires exactly 2 symbols")

        super().__init__(strategy_id, symbols, context)

        self.symbol_a = symbols[0]  # First symbol (long leg when spread low)
        self.symbol_b = symbols[1]  # Second symbol (short leg when spread low)
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.position_size = position_size

        # Price history
        self._prices_a: Deque[float] = deque(maxlen=lookback)
        self._prices_b: Deque[float] = deque(maxlen=lookback)
        self._spreads: Deque[float] = deque(maxlen=lookback)

        # Current bar prices (for synchronization)
        self._current_bar: Dict[str, Optional[BarData]] = {
            self.symbol_a: None,
            self.symbol_b: None,
        }

        # Position tracking
        self._position: Optional[SpreadPosition] = None
        self._current_zscore: Optional[float] = None

    def on_start(self) -> None:
        """Initialize strategy."""
        logger.info(
            f"Pairs Trading started: {self.symbol_a} vs {self.symbol_b} "
            f"(lookback={self.lookback}, entry_z={self.entry_zscore}, "
            f"exit_z={self.exit_zscore})"
        )

    def on_bar(self, bar: BarData) -> None:
        """
        Process bars - wait for both symbols before calculating spread.

        This demonstrates synchronization for multi-symbol strategies.
        """
        symbol = bar.symbol
        self._current_bar[symbol] = bar

        # Wait for both symbols to have current bar
        if not all(self._current_bar.values()):
            return

        bar_a = self._current_bar[self.symbol_a]
        bar_b = self._current_bar[self.symbol_b]

        # Safety check - both bars must exist with valid close prices
        if bar_a is None or bar_b is None:
            return
        if bar_a.close is None or bar_b.close is None:
            return

        close_a = bar_a.close
        close_b = bar_b.close

        # Check timestamps match (same bar period)
        # In backtest, bars come in order so we process when both are ready

        # Update price history
        self._prices_a.append(close_a)
        self._prices_b.append(close_b)

        # Calculate spread (price ratio or difference)
        spread = close_a / close_b  # Using ratio
        self._spreads.append(spread)

        # Clear current bar for next period
        self._current_bar = {self.symbol_a: None, self.symbol_b: None}

        # Need enough history
        if len(self._spreads) < self.lookback:
            return

        # Calculate z-score
        spread_mean = statistics.mean(self._spreads)
        spread_std = statistics.stdev(self._spreads)

        if spread_std == 0:
            return

        self._current_zscore = (spread - spread_mean) / spread_std

        # Trading logic
        self._check_signals(close_a, close_b, spread)

    def _check_signals(self, price_a: float, price_b: float, spread: float) -> None:
        """Check for entry/exit signals based on z-score."""
        zscore = self._current_zscore
        if zscore is None:
            return

        # Check exit first
        if self._position is not None:
            if abs(zscore) <= self.exit_zscore:
                self._close_position(price_a, price_b, spread, "MEAN_REVERSION")
            return

        # Check entry
        if zscore < -self.entry_zscore:
            # Spread is low - buy A, sell B (expect spread to increase)
            self._open_position(
                long_symbol=self.symbol_a,
                short_symbol=self.symbol_b,
                long_price=price_a,
                short_price=price_b,
                spread=spread,
            )

        elif zscore > self.entry_zscore:
            # Spread is high - sell A, buy B (expect spread to decrease)
            self._open_position(
                long_symbol=self.symbol_b,
                short_symbol=self.symbol_a,
                long_price=price_b,
                short_price=price_a,
                spread=spread,
            )

    def _open_position(
        self,
        long_symbol: str,
        short_symbol: str,
        long_price: float,
        short_price: float,
        spread: float,
    ) -> None:
        """Open pairs position - long one, short the other."""
        # Calculate quantities for dollar-neutral position
        long_qty = int(self.position_size / long_price)
        short_qty = int(self.position_size / short_price)

        if long_qty <= 0 or short_qty <= 0:
            logger.warning(f"[{self.strategy_id}] Position size too small")
            return

        logger.info(
            f"[{self.strategy_id}] OPEN PAIR: "
            f"LONG {long_qty} {long_symbol} @ {long_price:.2f}, "
            f"SHORT {short_qty} {short_symbol} @ {short_price:.2f} "
            f"(spread={spread:.4f}, z={self._current_zscore:.2f})"
        )

        # Submit both orders
        long_order = OrderRequest(
            symbol=long_symbol,
            side="BUY",
            quantity=long_qty,
            order_type="MARKET",
            client_order_id=f"{self.strategy_id}-long-{uuid.uuid4().hex[:8]}",
        )
        self.request_order(long_order)

        short_order = OrderRequest(
            symbol=short_symbol,
            side="SELL",
            quantity=short_qty,
            order_type="MARKET",
            client_order_id=f"{self.strategy_id}-short-{uuid.uuid4().hex[:8]}",
        )
        self.request_order(short_order)

        # Track position (zscore is guaranteed not None since we get here from _check_signals)
        entry_zscore = self._current_zscore if self._current_zscore is not None else 0.0
        self._position = SpreadPosition(
            long_symbol=long_symbol,
            short_symbol=short_symbol,
            quantity=long_qty,  # Simplified - assumes equal qty
            entry_spread=spread,
            entry_zscore=entry_zscore,
        )

    def _close_position(self, price_a: float, price_b: float, spread: float, reason: str) -> None:
        """Close pairs position - reverse both legs."""
        pos = self._position
        if pos is None:
            return

        # Get current prices for each leg (kept for debugging/extension)
        _long_price = price_a if pos.long_symbol == self.symbol_a else price_b  # noqa: F841
        _short_price = price_b if pos.short_symbol == self.symbol_b else price_a  # noqa: F841

        # Calculate P&L
        spread_pnl = spread - pos.entry_spread
        if pos.long_symbol == self.symbol_b:
            spread_pnl = -spread_pnl  # Reverse if we're short the spread

        logger.info(
            f"[{self.strategy_id}] CLOSE PAIR ({reason}): "
            f"spread_pnl={spread_pnl:.4f}, "
            f"entry_z={pos.entry_zscore:.2f}, exit_z={self._current_zscore:.2f}"
        )

        # Close long leg (sell)
        close_long = OrderRequest(
            symbol=pos.long_symbol,
            side="SELL",
            quantity=pos.quantity,
            order_type="MARKET",
            client_order_id=f"{self.strategy_id}-close-long-{uuid.uuid4().hex[:8]}",
        )
        self.request_order(close_long)

        # Close short leg (buy to cover)
        close_short = OrderRequest(
            symbol=pos.short_symbol,
            side="BUY",
            quantity=pos.quantity,
            order_type="MARKET",
            client_order_id=f"{self.strategy_id}-close-short-{uuid.uuid4().hex[:8]}",
        )
        self.request_order(close_short)

        self._position = None

    def on_fill(self, fill: TradeFill) -> None:
        """Log fills."""
        logger.info(
            f"[{self.strategy_id}] FILL: "
            f"{fill.side} {fill.quantity} {fill.symbol} @ {fill.price:.2f}"
        )

    def on_tick(self, tick: QuoteTick) -> None:
        """Not used - pairs trading typically uses bar data."""

    def get_state(self) -> dict:
        """Get current strategy state for monitoring."""
        return {
            "pair": f"{self.symbol_a}/{self.symbol_b}",
            "current_zscore": self._current_zscore,
            "spread_count": len(self._spreads),
            "position": (
                {
                    "long": self._position.long_symbol,
                    "short": self._position.short_symbol,
                    "entry_spread": self._position.entry_spread,
                    "entry_zscore": self._position.entry_zscore,
                }
                if self._position
                else None
            ),
        }
