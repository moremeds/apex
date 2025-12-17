"""
Scheduled Rebalancing Strategy.

A portfolio management strategy that:
- Maintains target weight allocations across multiple assets
- Rebalances on a scheduled basis (daily/weekly)
- Uses scheduler for time-based actions

This strategy demonstrates:
- Scheduler usage (schedule_daily, schedule_on_bar_close)
- Multi-asset portfolio management
- Target weight rebalancing
- Clock abstraction usage
- on_stop cleanup
"""

from datetime import time
from typing import List, Dict, Optional
import uuid
import logging

from ..base import Strategy, StrategyContext
from ..registry import register_strategy
from ...events.domain_events import QuoteTick, BarData, TradeFill
from ...interfaces.execution_provider import OrderRequest

logger = logging.getLogger(__name__)


@register_strategy(
    "scheduled_rebalance",
    description="Scheduled Portfolio Rebalancing Strategy",
    author="Apex",
    version="1.0",
)
class ScheduledRebalanceStrategy(Strategy):
    """
    Scheduled Portfolio Rebalancing Strategy.

    Maintains target weights across a portfolio of assets.
    Rebalances when allocations drift beyond threshold.

    Parameters:
        target_weights: Dict mapping symbol to target weight (must sum to 1.0)
        rebalance_threshold: Drift threshold to trigger rebalance (default: 0.05 = 5%)
        rebalance_time: Time of day to check rebalance (default: 15:30)
        total_capital: Total portfolio value (default: 100000)

    Features demonstrated:
        - Scheduler for time-based actions
        - Multi-asset weight management
        - Drift detection
        - Clock.now() usage
        - Position sizing across portfolio
    """

    def __init__(
        self,
        strategy_id: str,
        symbols: List[str],
        context: StrategyContext,
        target_weights: Optional[Dict[str, float]] = None,
        rebalance_threshold: float = 0.05,
        rebalance_time: time = time(15, 30),
        total_capital: float = 100000,
    ):
        super().__init__(strategy_id, symbols, context)

        # Set equal weights if not provided
        if target_weights is None:
            weight = 1.0 / len(symbols)
            target_weights = {s: weight for s in symbols}

        # Validate weights
        if set(target_weights.keys()) != set(symbols):
            raise ValueError("target_weights must contain all symbols")
        if abs(sum(target_weights.values()) - 1.0) > 0.01:
            raise ValueError("target_weights must sum to 1.0")

        self.target_weights = target_weights
        self.rebalance_threshold = rebalance_threshold
        self.rebalance_time = rebalance_time
        self.total_capital = total_capital

        # Current prices for each symbol
        self._current_prices: Dict[str, Optional[float]] = {s: None for s in symbols}

        # Track if we've done initial allocation
        self._initialized = False

        # Rebalance count
        self._rebalance_count = 0

    def on_start(self) -> None:
        """Initialize strategy and schedule rebalancing."""
        logger.info(
            f"Scheduled Rebalance started: {self.symbols} "
            f"(threshold={self.rebalance_threshold:.1%}, time={self.rebalance_time})"
        )
        logger.info(f"Target weights: {self.target_weights}")

        # Schedule daily rebalance check
        self.context.scheduler.schedule_daily(
            action_id=f"{self.strategy_id}-daily-rebalance",
            callback=self._scheduled_rebalance_check,
            time_of_day=self.rebalance_time,
        )

        # Also check at bar close for backtest compatibility
        self.context.scheduler.schedule_on_bar_close(
            action_id=f"{self.strategy_id}-bar-close-check",
            callback=self._on_bar_close,
        )

        logger.info(
            f"Scheduled daily rebalance at {self.rebalance_time} "
            f"and on each bar close"
        )

    def on_bar(self, bar: BarData) -> None:
        """Update prices from bar data."""
        self._current_prices[bar.symbol] = bar.close

        # Check if we have all prices for initial allocation
        if not self._initialized and all(self._current_prices.values()):
            self._do_initial_allocation()

    def on_tick(self, tick: QuoteTick) -> None:
        """Update prices from tick data."""
        price = tick.last or tick.mid
        if price:
            self._current_prices[tick.symbol] = price

    def _on_bar_close(self) -> None:
        """Called at bar close - check for rebalance."""
        if not self._initialized:
            return

        # Check drift and rebalance if needed
        drift = self._calculate_max_drift()
        if drift > self.rebalance_threshold:
            logger.info(
                f"[{self.strategy_id}] Drift {drift:.1%} exceeds threshold "
                f"{self.rebalance_threshold:.1%}"
            )
            self._do_rebalance()

    def _scheduled_rebalance_check(self) -> None:
        """Scheduled daily rebalance check."""
        current_time = self.context.now()
        logger.info(
            f"[{self.strategy_id}] Scheduled rebalance check at {current_time}"
        )

        if not self._initialized:
            logger.warning("Portfolio not yet initialized")
            return

        drift = self._calculate_max_drift()
        logger.info(f"[{self.strategy_id}] Current max drift: {drift:.1%}")

        if drift > self.rebalance_threshold:
            self._do_rebalance()
        else:
            logger.info(
                f"[{self.strategy_id}] No rebalance needed "
                f"(drift {drift:.1%} < threshold {self.rebalance_threshold:.1%})"
            )

    def _do_initial_allocation(self) -> None:
        """Perform initial portfolio allocation."""
        logger.info(f"[{self.strategy_id}] Performing initial allocation")

        for symbol in self.symbols:
            target_weight = self.target_weights[symbol]
            price = self._current_prices[symbol]

            if price is None or price <= 0:
                continue

            target_value = self.total_capital * target_weight
            target_qty = int(target_value / price)

            if target_qty > 0:
                order = OrderRequest(
                    symbol=symbol,
                    side="BUY",
                    quantity=target_qty,
                    order_type="MARKET",
                    client_order_id=f"{self.strategy_id}-init-{symbol}-{uuid.uuid4().hex[:8]}",
                )
                self.request_order(order)

                logger.info(
                    f"[{self.strategy_id}] Initial BUY: {target_qty} {symbol} "
                    f"@ {price:.2f} (target weight={target_weight:.1%})"
                )

        self._initialized = True

    def _calculate_max_drift(self) -> float:
        """Calculate maximum weight drift from target."""
        if not all(self._current_prices.values()):
            return 0.0

        # Calculate current portfolio value
        total_value = 0.0
        position_values = {}

        for symbol in self.symbols:
            price = self._current_prices[symbol]
            qty = self.context.get_position_quantity(symbol)
            value = (qty * price) if price and qty else 0
            position_values[symbol] = value
            total_value += value

        if total_value == 0:
            return 1.0  # 100% drift if no positions

        # Calculate drift
        max_drift = 0.0
        for symbol in self.symbols:
            current_weight = position_values[symbol] / total_value
            target_weight = self.target_weights[symbol]
            drift = abs(current_weight - target_weight)
            max_drift = max(max_drift, drift)

        return max_drift

    def _do_rebalance(self) -> None:
        """Execute rebalancing trades."""
        self._rebalance_count += 1
        logger.info(
            f"[{self.strategy_id}] Executing rebalance #{self._rebalance_count}"
        )

        if not all(self._current_prices.values()):
            logger.warning("Missing prices - cannot rebalance")
            return

        # Calculate current portfolio value
        total_value = 0.0
        for symbol in self.symbols:
            price = self._current_prices[symbol]
            qty = self.context.get_position_quantity(symbol)
            total_value += (qty * price) if price and qty else 0

        if total_value == 0:
            logger.warning("Zero portfolio value - using initial capital")
            total_value = self.total_capital

        # Calculate and execute rebalancing trades
        for symbol in self.symbols:
            price = self._current_prices[symbol]
            if price is None or price <= 0:
                continue

            current_qty = self.context.get_position_quantity(symbol)
            target_value = total_value * self.target_weights[symbol]
            target_qty = int(target_value / price)

            diff = target_qty - current_qty

            if abs(diff) < 1:
                continue  # Skip small adjustments

            if diff > 0:
                order = OrderRequest(
                    symbol=symbol,
                    side="BUY",
                    quantity=abs(diff),
                    order_type="MARKET",
                    client_order_id=f"{self.strategy_id}-rebal-{symbol}-{uuid.uuid4().hex[:8]}",
                )
            else:
                order = OrderRequest(
                    symbol=symbol,
                    side="SELL",
                    quantity=abs(diff),
                    order_type="MARKET",
                    client_order_id=f"{self.strategy_id}-rebal-{symbol}-{uuid.uuid4().hex[:8]}",
                )

            self.request_order(order)

            logger.info(
                f"[{self.strategy_id}] Rebalance {'BUY' if diff > 0 else 'SELL'}: "
                f"{abs(diff)} {symbol} @ {price:.2f}"
            )

    def on_fill(self, fill: TradeFill) -> None:
        """Log fills."""
        logger.info(
            f"[{self.strategy_id}] FILL: "
            f"{fill.side} {fill.quantity} {fill.symbol} @ {fill.price:.2f}"
        )

    def on_stop(self) -> None:
        """Cleanup on strategy stop."""
        logger.info(
            f"[{self.strategy_id}] Strategy stopped. "
            f"Total rebalances: {self._rebalance_count}"
        )

        # Log final portfolio state
        if all(self._current_prices.values()):
            logger.info("Final portfolio state:")
            for symbol in self.symbols:
                price = self._current_prices[symbol]
                qty = self.context.get_position_quantity(symbol)
                value = qty * price if price and qty else 0
                logger.info(f"  {symbol}: {qty} shares @ {price:.2f} = ${value:,.2f}")

    def get_state(self) -> dict:
        """Get current strategy state for monitoring."""
        drift = self._calculate_max_drift() if self._initialized else None

        return {
            "initialized": self._initialized,
            "rebalance_count": self._rebalance_count,
            "max_drift": drift,
            "current_prices": self._current_prices.copy(),
            "target_weights": self.target_weights.copy(),
        }
