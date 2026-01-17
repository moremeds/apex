"""
Strategy framework for Apex trading system.

This module provides the core strategy infrastructure:
- Strategy: Abstract base class for all trading strategies
- StrategyContext: Runtime context with clock, positions, execution
- StrategyState: Lifecycle state enumeration
- Scheduler: Time-based action scheduling

Usage:
    from src.domain.strategy import Strategy, StrategyContext
    from src.domain.clock import SystemClock

    class MyStrategy(Strategy):
        def on_tick(self, tick: QuoteTick) -> None:
            if tick.last > 150:
                self.request_order(OrderRequest(...))

    # Create and run
    context = StrategyContext(clock=SystemClock())
    strategy = MyStrategy("my-strat", ["AAPL"], context)
    strategy.start()
"""

from .base import Strategy, StrategyContext, StrategyState, TradingSignal
from .registry import StrategyRegistry, get_strategy_class, register_strategy
from .scheduler import (
    LiveScheduler,
    ScheduledAction,
    ScheduleFrequency,
    Scheduler,
    SimulatedScheduler,
)

__all__ = [
    # Base classes
    "Strategy",
    "StrategyContext",
    "StrategyState",
    "TradingSignal",
    # Scheduler
    "Scheduler",
    "LiveScheduler",
    "SimulatedScheduler",
    "ScheduleFrequency",
    "ScheduledAction",
    # Registry
    "StrategyRegistry",
    "get_strategy_class",
    "register_strategy",
]
