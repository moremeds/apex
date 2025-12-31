"""
Base strategy class and interfaces.

This module defines the core Strategy class that all trading strategies extend.
Strategies are pure signal generators - they receive data, produce signals,
and are unaware of execution details.

Key Principles:
1. Strategy code is identical for live and backtest modes
2. All time operations go through Clock interface
3. All scheduled actions go through Scheduler interface
4. Strategies emit signals/orders but don't handle execution

Lifecycle:
1. __init__: Setup parameters, indicators
2. on_start: Called once when strategy begins
3. on_tick/on_bar: Called on each market data event
4. on_fill: Called when orders are filled
5. on_stop: Called when strategy stops

Usage:
    class MyStrategy(Strategy):
        def __init__(self, strategy_id, symbols, context, **params):
            super().__init__(strategy_id, symbols, context)
            self.threshold = params.get('threshold', 100)

        def on_tick(self, tick: QuoteTick) -> None:
            if tick.last > self.threshold:
                self.request_order(OrderRequest(...))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)
from enum import Enum
import uuid
import logging

from ..events.domain_events import (
    QuoteTick,
    BarData,
    PositionSnapshot,
    AccountSnapshot,
    TradeFill,
    OrderUpdate,
)
from ..interfaces.event_bus import EventBus, EventType
from ..interfaces.execution_provider import OrderRequest

if TYPE_CHECKING:
    from ..clock import Clock
    from .scheduler import Scheduler
    from .cost_estimator import CostEstimator
    from .risk_gate import RiskGate

logger = logging.getLogger(__name__)


class StrategyState(Enum):
    """Strategy lifecycle states."""

    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class TradingSignal:
    """
    Trading signal emitted by strategy.

    Signals express trading intent before being converted to orders.
    They go through risk validation before becoming actual orders.
    """

    signal_id: str
    symbol: str
    direction: str  # "LONG", "SHORT", "FLAT"
    strength: float = 1.0  # Signal strength 0.0 to 1.0
    target_quantity: Optional[float] = None
    target_price: Optional[float] = None

    # Metadata
    strategy_id: str = ""
    timestamp: Optional[datetime] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.signal_id:
            self.signal_id = f"sig-{uuid.uuid4().hex[:8]}"
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class StrategyContext:
    """
    Context available to strategies during execution.

    Provides access to:
    - clock: Time operations (live or simulated)
    - scheduler: Time-based action scheduling
    - positions: Current position state
    - account: Account balances and margin
    - execution: Order submission (optional)
    - market_data: Latest quotes (optional)
    - cost_estimator: Pre-trade cost estimation (optional)
    - risk_gate: Pre-trade risk validation (optional)

    The context is injected by the runner and provides a consistent
    interface regardless of whether the strategy is running live or
    in backtest mode.
    """

    clock: "Clock"
    scheduler: Optional["Scheduler"] = None
    positions: Dict[str, PositionSnapshot] = field(default_factory=dict)
    account: Optional[AccountSnapshot] = None
    execution: Optional[Any] = None  # ExecutionProvider
    market_data: Dict[str, QuoteTick] = field(default_factory=dict)
    cost_estimator: Optional["CostEstimator"] = None
    risk_gate: Optional["RiskGate"] = None

    def now(self) -> datetime:
        """Get current time (live or simulated)."""
        return self.clock.now()

    def get_position(self, symbol: str) -> Optional[PositionSnapshot]:
        """Get position for a symbol."""
        return self.positions.get(symbol)

    def get_position_quantity(self, symbol: str) -> float:
        """Get position quantity (0 if no position)."""
        pos = self.positions.get(symbol)
        return pos.quantity if pos else 0.0

    def get_quote(self, symbol: str) -> Optional[QuoteTick]:
        """Get latest quote for a symbol."""
        return self.market_data.get(symbol)

    def get_mid_price(self, symbol: str) -> Optional[float]:
        """Get mid price for a symbol."""
        quote = self.market_data.get(symbol)
        if quote:
            return quote.mid or quote.last
        return None

    def has_position(self, symbol: str) -> bool:
        """Check if we have a position in symbol."""
        return self.get_position_quantity(symbol) != 0

    def is_long(self, symbol: str) -> bool:
        """Check if we are long in symbol."""
        return self.get_position_quantity(symbol) > 0

    def is_short(self, symbol: str) -> bool:
        """Check if we are short in symbol."""
        return self.get_position_quantity(symbol) < 0


# =============================================================================
# Strategy Protocol (ARCH-004)
# =============================================================================
# This protocol defines the minimal interface required by the backtest engine.
# It ensures clean separation between backtest infrastructure and strategy domain.


@runtime_checkable
class StrategyProtocol(Protocol):
    """
    Protocol defining the strategy interface for backtest and live runners.

    This protocol formalizes the contract between:
    - BacktestEngine (infrastructure) - consumes strategy via this protocol
    - TradingRunner (infrastructure) - consumes strategy via this protocol
    - Strategy (domain) - implements this protocol

    By depending on this protocol rather than the concrete Strategy class,
    runners can work with any object that implements these methods,
    enabling easier testing and alternative implementations.

    Usage:
        def run_strategy(strategy: StrategyProtocol) -> None:
            strategy.start()
            strategy.on_bar(bar_data)
            strategy.stop()
    """

    @property
    def strategy_id(self) -> str:
        """Unique identifier for this strategy instance."""
        ...

    @property
    def symbols(self) -> List[str]:
        """List of symbols this strategy trades."""
        ...

    @property
    def state(self) -> StrategyState:
        """Current lifecycle state."""
        ...

    def start(self) -> None:
        """Start the strategy (transitions to RUNNING state)."""
        ...

    def stop(self) -> None:
        """Stop the strategy (transitions to STOPPED state)."""
        ...

    def on_tick(self, tick: QuoteTick) -> None:
        """Handle incoming tick data."""
        ...

    def on_bar(self, bar: BarData) -> None:
        """Handle incoming bar data."""
        ...

    def on_fill(self, fill: TradeFill) -> None:
        """Handle order fill notification."""
        ...

    def on_order_callback(self, callback: Callable[[OrderRequest], None]) -> None:
        """Register callback for order requests."""
        ...

    def update_position(self, position: PositionSnapshot) -> None:
        """Update position state."""
        ...


class Strategy(ABC):
    """
    Abstract base class for all trading strategies.

    Strategies receive market data events and can emit signals
    or request orders. They should be stateless with respect to
    execution - all execution state is managed externally.

    To create a strategy:
    1. Subclass Strategy
    2. Implement on_tick() and/or on_bar()
    3. Call emit_signal() or request_order() to act
    4. Optionally override lifecycle methods (on_start, on_stop, on_fill)

    Example:
        class MomentumStrategy(Strategy):
            def __init__(self, strategy_id, symbols, context, lookback=20):
                super().__init__(strategy_id, symbols, context)
                self.lookback = lookback
                self.prices = []

            def on_tick(self, tick: QuoteTick) -> None:
                self.prices.append(tick.last)
                if len(self.prices) > self.lookback:
                    momentum = self.prices[-1] / self.prices[-self.lookback]
                    if momentum > 1.05:
                        self.request_order(self._create_buy_order(tick))
    """

    def __init__(
        self,
        strategy_id: str,
        symbols: List[str],
        context: StrategyContext,
    ):
        """
        Initialize strategy.

        Args:
            strategy_id: Unique identifier for this strategy instance.
            symbols: List of symbols this strategy trades.
            context: Runtime context with clock, positions, etc.
        """
        self.strategy_id = strategy_id
        self.symbols = symbols
        self.context = context

        self._state = StrategyState.INITIALIZED
        self._signal_callbacks: List[Callable[[TradingSignal], None]] = []
        self._order_callbacks: List[Callable[[OrderRequest], None]] = []
        self._error_message: Optional[str] = None

    @property
    def state(self) -> StrategyState:
        """Get current strategy state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if strategy is in running state."""
        return self._state == StrategyState.RUNNING

    # -------------------------------------------------------------------------
    # Lifecycle Methods (Override in subclass as needed)
    # -------------------------------------------------------------------------

    def on_start(self) -> None:
        """
        Called once when strategy starts.

        Override to:
        - Initialize indicators
        - Load historical data
        - Set up scheduled actions via context.scheduler
        - Perform any one-time setup

        Example:
            def on_start(self) -> None:
                # Schedule daily rebalance at 3:55 PM
                self.context.scheduler.schedule_daily(
                    action_id="rebalance",
                    callback=self.rebalance,
                    time_of_day=time(15, 55),
                )
        """
        pass

    def on_stop(self) -> None:
        """
        Called once when strategy stops.

        Override to:
        - Cleanup resources
        - Log final state
        - Cancel pending orders (if needed)
        """
        pass

    @abstractmethod
    def on_tick(self, tick: QuoteTick) -> None:
        """
        Called on each market data tick.

        This is the main entry point for strategy logic. Override to
        implement your trading algorithm.

        Args:
            tick: Market data tick with bid/ask/last prices.

        Example:
            def on_tick(self, tick: QuoteTick) -> None:
                if tick.last > self.entry_price:
                    self.request_order(OrderRequest(
                        symbol=tick.symbol,
                        side="BUY",
                        quantity=100,
                        order_type="MARKET",
                    ))
        """
        pass

    def on_bar(self, bar: BarData) -> None:
        """
        Called on each bar close.

        Override for bar-based strategies. By default, converts bar to
        a tick and calls on_tick, so tick-based strategies work with bars.

        Args:
            bar: OHLCV bar data.
        """
        # Convert bar to tick for compatibility
        tick = QuoteTick(
            symbol=bar.symbol,
            bid=bar.close,
            ask=bar.close,
            last=bar.close,
            volume=bar.volume,
            timestamp=bar.bar_end or bar.timestamp,
            source=bar.source,
        )
        self.on_tick(tick)

    def on_greeks(self, symbol: str, greeks: Dict[str, float]) -> None:
        """
        Called on Greeks update (for options strategies).

        Override for strategies that need Greeks (delta, gamma, vega, theta).

        Args:
            symbol: Symbol the Greeks are for.
            greeks: Dictionary with delta, gamma, vega, theta, iv.
        """
        pass

    def on_position(self, position: PositionSnapshot) -> None:
        """
        Called when a position changes.

        Override to react to position changes. Default updates context.

        Args:
            position: Updated position snapshot.
        """
        self.context.positions[position.symbol] = position

    def on_fill(self, fill: TradeFill) -> None:
        """
        Called when an order is filled.

        Override to track execution state, update P&L tracking, etc.

        Args:
            fill: Trade fill/execution details.
        """
        logger.debug(
            f"Strategy {self.strategy_id} received fill: "
            f"{fill.side} {fill.quantity} {fill.symbol} @ {fill.price}"
        )

    def on_order_update(self, update: OrderUpdate) -> None:
        """
        Called when order status changes.

        Override to track order lifecycle.

        Args:
            update: Order status update.
        """
        logger.debug(
            f"Strategy {self.strategy_id} received order update: "
            f"{update.order_id} -> {update.status}"
        )

    def on_account(self, account: AccountSnapshot) -> None:
        """
        Called when account state changes.

        Override to react to margin changes, etc.

        Args:
            account: Updated account snapshot.
        """
        self.context.account = account

    def on_error(self, error: Exception) -> None:
        """
        Called when an error occurs.

        Override to implement custom error handling.

        Args:
            error: The exception that occurred.
        """
        logger.error(f"Strategy {self.strategy_id} error: {error}")
        self._error_message = str(error)

    # -------------------------------------------------------------------------
    # Signal & Order Emission
    # -------------------------------------------------------------------------

    def emit_signal(self, signal: TradingSignal) -> None:
        """
        Emit a trading signal.

        Signals express trading intent and are processed by the risk layer
        before becoming actual orders. Use signals when you want pre-trade
        risk checks.

        Args:
            signal: Trading signal with direction and target.
        """
        signal.strategy_id = self.strategy_id
        if signal.timestamp is None:
            signal.timestamp = self.context.now()

        for callback in self._signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"Signal callback error: {e}")

        logger.debug(
            f"Strategy {self.strategy_id} emitted signal: "
            f"{signal.symbol} {signal.direction}"
        )

    def request_order(self, order: OrderRequest) -> None:
        """
        Request order submission.

        Orders are submitted through the execution layer. They may still
        go through risk validation depending on configuration.

        Args:
            order: Order request with symbol, side, quantity, etc.
        """
        if order.client_order_id is None:
            order.client_order_id = f"{self.strategy_id}-{uuid.uuid4().hex[:8]}"

        for callback in self._order_callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Order callback error: {e}")

        logger.info(
            f"Strategy {self.strategy_id} requested order: "
            f"{order.side} {order.quantity} {order.symbol}"
        )

    def on_signal_callback(
        self, callback: Callable[[TradingSignal], None]
    ) -> None:
        """
        Register callback for signal emission.

        Used by the strategy runner to connect signal output.

        Args:
            callback: Function to call when signals are emitted.
        """
        self._signal_callbacks.append(callback)

    def on_order_callback(
        self, callback: Callable[[OrderRequest], None]
    ) -> None:
        """
        Register callback for order requests.

        Used by the strategy runner to connect order output.

        Args:
            callback: Function to call when orders are requested.
        """
        self._order_callbacks.append(callback)

    # -------------------------------------------------------------------------
    # Lifecycle Control
    # -------------------------------------------------------------------------

    def start(self) -> None:
        """
        Start the strategy.

        Called by the strategy runner. Sets state to RUNNING and calls on_start.
        """
        if self._state not in (StrategyState.INITIALIZED, StrategyState.STOPPED):
            logger.warning(
                f"Cannot start strategy {self.strategy_id} "
                f"from state {self._state}"
            )
            return

        self._state = StrategyState.STARTING
        try:
            self.on_start()
            self._state = StrategyState.RUNNING
            logger.info(f"Strategy {self.strategy_id} started")
        except Exception as e:
            self._state = StrategyState.ERROR
            self._error_message = str(e)
            logger.error(f"Strategy {self.strategy_id} failed to start: {e}")
            raise

    def stop(self) -> None:
        """
        Stop the strategy.

        Called by the strategy runner. Sets state to STOPPED and calls on_stop.
        """
        if self._state == StrategyState.STOPPED:
            return

        self._state = StrategyState.STOPPING
        try:
            self.on_stop()
        except Exception as e:
            logger.error(f"Strategy {self.strategy_id} stop error: {e}")
        finally:
            self._state = StrategyState.STOPPED
            logger.info(f"Strategy {self.strategy_id} stopped")

    def pause(self) -> None:
        """
        Pause the strategy.

        Strategy will stop processing events but not call on_stop.
        Can be resumed with resume().
        """
        if self._state == StrategyState.RUNNING:
            self._state = StrategyState.PAUSED
            logger.info(f"Strategy {self.strategy_id} paused")

    def resume(self) -> None:
        """Resume a paused strategy."""
        if self._state == StrategyState.PAUSED:
            self._state = StrategyState.RUNNING
            logger.info(f"Strategy {self.strategy_id} resumed")

    # -------------------------------------------------------------------------
    # Event Bus Integration
    # -------------------------------------------------------------------------

    def subscribe_to_events(self, event_bus: EventBus) -> None:
        """
        Subscribe strategy to event bus.

        Called by strategy runner to wire up event delivery.
        OPT-007: Strategy callbacks are registered as "heavy" to offload
        to thread pool, preventing event loop blocking.

        Args:
            event_bus: Event bus to subscribe to.
        """
        event_bus.subscribe(EventType.MARKET_DATA_TICK, self._handle_tick_event)
        event_bus.subscribe(EventType.POSITION_UPDATED, self._handle_position_event)
        event_bus.subscribe(EventType.ORDER_FILLED, self._handle_fill_event)
        event_bus.subscribe(EventType.ACCOUNT_UPDATED, self._handle_account_event)

        # OPT-007: Register callbacks as heavy to offload to thread pool
        # This prevents user strategy code from blocking the event loop
        if hasattr(event_bus, 'register_heavy_callback'):
            event_bus.register_heavy_callback(self._handle_tick_event)
            event_bus.register_heavy_callback(self._handle_position_event)
            event_bus.register_heavy_callback(self._handle_fill_event)
            event_bus.register_heavy_callback(self._handle_account_event)
            logger.debug(f"Strategy {self.strategy_id} callbacks registered as heavy")

        logger.debug(f"Strategy {self.strategy_id} subscribed to events")

    def unsubscribe_from_events(self, event_bus: EventBus) -> None:
        """Unsubscribe strategy from event bus."""
        event_bus.unsubscribe(EventType.MARKET_DATA_TICK, self._handle_tick_event)
        event_bus.unsubscribe(EventType.POSITION_UPDATED, self._handle_position_event)
        event_bus.unsubscribe(EventType.ORDER_FILLED, self._handle_fill_event)
        event_bus.unsubscribe(EventType.ACCOUNT_UPDATED, self._handle_account_event)

        # OPT-007: Unregister heavy callbacks
        if hasattr(event_bus, 'unregister_heavy_callback'):
            event_bus.unregister_heavy_callback(self._handle_tick_event)
            event_bus.unregister_heavy_callback(self._handle_position_event)
            event_bus.unregister_heavy_callback(self._handle_fill_event)
            event_bus.unregister_heavy_callback(self._handle_account_event)

    def _handle_tick_event(self, payload: Any) -> None:
        """Handle tick event from bus."""
        if self._state != StrategyState.RUNNING:
            return

        from ..events.domain_events import MarketDataTickEvent

        # C3: Handle MarketDataTickEvent (new standard), QuoteTick, or legacy dict
        tick: Optional[QuoteTick] = None
        if isinstance(payload, MarketDataTickEvent):
            # Convert MarketDataTickEvent to QuoteTick for strategy API
            tick = QuoteTick(
                symbol=payload.symbol,
                bid=payload.bid,
                ask=payload.ask,
                last=payload.last,
                delta=payload.delta,
                gamma=payload.gamma,
                vega=payload.vega,
                theta=payload.theta,
                iv=payload.iv,
                source=payload.source,
            )
        elif isinstance(payload, QuoteTick):
            tick = payload
        elif isinstance(payload, dict):
            # Legacy dict handling
            tick = payload.get("tick")

        if isinstance(tick, QuoteTick) and tick.symbol in self.symbols:
            # Update market data cache
            self.context.market_data[tick.symbol] = tick
            try:
                self.on_tick(tick)
            except Exception as e:
                self.on_error(e)

    def _handle_position_event(self, payload: Any) -> None:
        """Handle position event from bus."""
        if isinstance(payload, dict):
            positions = payload.get("positions", [])
        elif isinstance(payload, list):
            positions = payload
        elif isinstance(payload, PositionSnapshot):
            positions = [payload]
        else:
            return

        for pos in positions:
            if isinstance(pos, PositionSnapshot) and pos.symbol in self.symbols:
                try:
                    self.on_position(pos)
                except Exception as e:
                    self.on_error(e)

    def _handle_fill_event(self, payload: Any) -> None:
        """Handle fill event from bus."""
        if isinstance(payload, dict):
            fill = payload.get("fill")
        elif isinstance(payload, TradeFill):
            fill = payload
        else:
            return

        if isinstance(fill, TradeFill) and fill.symbol in self.symbols:
            try:
                self.on_fill(fill)
            except Exception as e:
                self.on_error(e)

    def _handle_account_event(self, payload: Any) -> None:
        """Handle account event from bus."""
        if isinstance(payload, dict):
            account = payload.get("account")
        elif isinstance(payload, AccountSnapshot):
            account = payload
        else:
            return

        if isinstance(account, AccountSnapshot):
            try:
                self.on_account(account)
            except Exception as e:
                self.on_error(e)

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def log(self, message: str, level: str = "info") -> None:
        """
        Log a message with strategy context.

        Args:
            message: Message to log.
            level: Log level (debug, info, warning, error).
        """
        log_func = getattr(logger, level, logger.info)
        log_func(f"[{self.strategy_id}] {message}")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.strategy_id}, "
            f"symbols={self.symbols}, "
            f"state={self._state.value})"
        )
