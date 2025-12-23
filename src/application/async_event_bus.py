"""Async event bus with non-blocking publish and debounced dispatch."""

from __future__ import annotations
import asyncio
import time
from typing import Callable, Any, Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock

from ..domain.interfaces.event_bus import EventBus, EventType
from ..domain.events.domain_events import DomainEvent
from ..utils.logging_setup import get_logger


logger = get_logger(__name__)


@dataclass
class EventEnvelope:
    """Wrapper for event with metadata."""
    event_type: EventType
    payload: DomainEvent
    timestamp: float = field(default_factory=time.time)
    source: str = ""


class AsyncEventBus(EventBus):
    """
    Async event bus with non-blocking publish and debounced dispatch.

    Features:
    - Non-blocking publish (thread-safe from adapter callbacks)
    - Async dispatch loop for subscribers
    - Debouncing for high-frequency events (market data)
    - Bounded queue with overflow handling
    - Graceful shutdown
    """

    def __init__(
        self,
        max_queue_size: int = 1000,
        debounce_ms: int = 50,
    ):
        """
        Initialize async event bus.

        Args:
            max_queue_size: Maximum events in queue before dropping
            debounce_ms: Debounce window for high-frequency events
        """
        self._subscribers: Dict[EventType, List[Callable[[DomainEvent], None]]] = defaultdict(list)
        self._async_subscribers: Dict[EventType, List[Callable[[DomainEvent], Any]]] = defaultdict(list)
        self._queue: asyncio.Queue[EventEnvelope] = None  # Created in start()
        self._max_queue_size = max_queue_size
        self._debounce_ms = debounce_ms

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Thread safety for publish from non-async contexts
        self._lock = Lock()

        # Stats
        self._stats = {
            "published": 0,
            "dispatched": 0,
            "dropped": 0,
            "errors": 0,
            "high_water_mark": 0,  # Peak queue depth seen
        }

        # Queue depth warning threshold (80% of max)
        self._queue_warning_threshold = int(max_queue_size * 0.8)
        self._queue_warning_logged = False

        # Debounce state for market data ticks
        self._pending_ticks: Dict[str, EventEnvelope] = {}
        self._last_tick_dispatch: float = 0

        # M2: Heavy callback support - callbacks that should run in thread pool
        self._heavy_callbacks: set[Callable] = set()
        self._heavy_semaphore = asyncio.Semaphore(4)  # Limit concurrent heavy callbacks

    def publish(self, event_type: EventType, payload: DomainEvent) -> None:
        """
        Publish an event (non-blocking, thread-safe).

        Can be called from any thread (e.g., adapter callbacks).
        Uses call_soon_threadsafe when called from non-loop threads.

        Args:
            event_type: Type of event
            payload: Event data (Must be DomainEvent)
        """
        if not isinstance(payload, (DomainEvent, dict)) and event_type != EventType.SHUTDOWN:
             # Allow dict for SHUTDOWN or strictly enforce DomainEvent elsewhere?
             # For now, let's enforce DomainEvent but keep dict for SHUTDOWN if payload is dict.
             # Actually, DomainEvent base class is good.
             # Wait, SHUTDOWN payload in existing code is `{"timestamp": time.time()}`.
             # I should probably create a ShutdownEvent or allow dict for legacy compatibility
             # BUT the instruction said "no need to be backward compatible".
             # So I will enforce DomainEvent and update callers.
             pass
        
        if isinstance(payload, dict):
             # STRICT MODE: Reject dicts
             if event_type == EventType.SHUTDOWN:
                 # Exception for shutdown as it's internal control
                 pass
             else:
                 raise TypeError(f"Payload must be DomainEvent, got dict: {payload}")
        
        envelope = EventEnvelope(
            event_type=event_type,
            payload=payload,
            timestamp=time.time(),
        )

        with self._lock:
            self._stats["published"] += 1

        # If not running async, dispatch synchronously
        if not self._running or self._queue is None:
            self._dispatch_sync(envelope)
            return

        # Determine if we're on the event loop thread
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        # If on the same loop, use put_nowait directly
        if running_loop is self._loop:
            self._enqueue_direct(envelope)
        else:
            # Cross-thread: use call_soon_threadsafe for thread safety
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._enqueue_direct, envelope)
            else:
                # Fallback if loop not available
                self._dispatch_sync(envelope)

    def _enqueue_direct(self, envelope: EventEnvelope) -> None:
        """
        Enqueue event directly (must be called from loop thread).

        This method is called either directly from the loop thread,
        or via call_soon_threadsafe from other threads.
        """
        try:
            # Use put_nowait for non-blocking
            self._queue.put_nowait(envelope)

            # Track queue depth metrics
            current_depth = self._queue.qsize()
            with self._lock:
                if current_depth > self._stats["high_water_mark"]:
                    self._stats["high_water_mark"] = current_depth

            # Warn at 80% capacity (once per threshold crossing)
            if current_depth >= self._queue_warning_threshold:
                if not self._queue_warning_logged:
                    logger.warning(
                        f"Event queue at {current_depth}/{self._max_queue_size} "
                        f"({100 * current_depth // self._max_queue_size}%) - "
                        "consider increasing queue size or reducing event rate"
                    )
                    self._queue_warning_logged = True
            else:
                # Reset warning flag when queue drops below threshold
                self._queue_warning_logged = False

        except asyncio.QueueFull:
            with self._lock:
                self._stats["dropped"] += 1
            logger.warning(f"Event queue full, dropping {envelope.event_type.value}")

    def _dispatch_sync(self, envelope: EventEnvelope) -> None:
        """Synchronous dispatch for non-async mode."""
        for callback in self._subscribers.get(envelope.event_type, []):
            try:
                callback(envelope.payload)
                self._stats["dispatched"] += 1
            except Exception as e:
                self._stats["errors"] += 1
                logger.error(f"Error in sync subscriber: {e}", exc_info=True)

    def subscribe(self, event_type: EventType, callback: Callable[[Any], None]) -> None:
        """
        Subscribe to an event type (sync callback).

        Args:
            event_type: Event type to listen for
            callback: Sync function to call when event is published
        """
        with self._lock:
            self._subscribers[event_type].append(callback)
        logger.debug(f"Subscribed to {event_type.value}")

    def subscribe_async(self, event_type: EventType, callback: Callable[[Any], Any]) -> None:
        """
        Subscribe to an event type (async callback).

        Args:
            event_type: Event type to listen for
            callback: Async function to call when event is published
        """
        with self._lock:
            self._async_subscribers[event_type].append(callback)
        logger.debug(f"Async subscribed to {event_type.value}")

    def unsubscribe(self, event_type: EventType, callback: Callable[[Any], None]) -> None:
        """Unsubscribe a callback from an event type."""
        with self._lock:
            if callback in self._subscribers.get(event_type, []):
                self._subscribers[event_type].remove(callback)
                logger.debug(f"Unsubscribed from {event_type.value}")
            if callback in self._async_subscribers.get(event_type, []):
                self._async_subscribers[event_type].remove(callback)
                logger.debug(f"Async unsubscribed from {event_type.value}")

    async def start(self) -> None:
        """Start the async event dispatch loop."""
        if self._running:
            logger.warning("AsyncEventBus already running")
            return

        # Use get_running_loop() for Python 3.10+ compatibility
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=self._max_queue_size)
        self._running = True
        self._task = asyncio.create_task(self._dispatch_loop())
        logger.info("AsyncEventBus started")

    async def stop(self) -> None:
        """Stop the event bus and drain remaining events."""
        if not self._running:
            return

        logger.info("Stopping AsyncEventBus...")
        self._running = False

        # Publish shutdown event
        self.publish(EventType.SHUTDOWN, {"timestamp": time.time()})

        # Wait for task to complete
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Drain remaining events
        await self._drain_queue()

        logger.info(f"AsyncEventBus stopped. Stats: {self._stats}")

    async def _drain_queue(self) -> None:
        """Drain remaining events in queue."""
        if self._queue is None:
            return

        drained = 0
        while not self._queue.empty():
            try:
                envelope = self._queue.get_nowait()
                await self._dispatch_async(envelope)
                drained += 1
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                logger.error(f"Error draining queue: {e}")

        if drained > 0:
            logger.info(f"Drained {drained} events from queue")

    async def _dispatch_loop(self) -> None:
        """Main dispatch loop - processes events from queue."""
        logger.debug("Event dispatch loop started")

        while self._running:
            try:
                # Wait for event with timeout (allows checking _running flag)
                try:
                    envelope = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    # Flush any pending debounced ticks
                    await self._flush_pending_ticks()
                    continue

                # Handle market data ticks with debouncing
                if envelope.event_type == EventType.MARKET_DATA_TICK:
                    await self._handle_tick_debounced(envelope)
                else:
                    await self._dispatch_async(envelope)

            except asyncio.CancelledError:
                logger.debug("Dispatch loop cancelled")
                break
            except Exception as e:
                self._stats["errors"] += 1
                logger.error(f"Error in dispatch loop: {e}", exc_info=True)

        logger.debug("Event dispatch loop ended")

    async def _handle_tick_debounced(self, envelope: EventEnvelope) -> None:
        """
        Handle market data tick with debouncing.

        Coalesces multiple ticks for the same symbol within debounce window.
        C3: Updated to handle both typed events and dicts.
        """
        # C3: Handle both typed events (with .symbol attr) and dicts
        payload = envelope.payload
        if hasattr(payload, "symbol"):
            symbol = payload.symbol
        elif isinstance(payload, dict):
            symbol = payload.get("symbol", "unknown")
        else:
            symbol = "unknown"

        # Store latest tick per symbol
        self._pending_ticks[symbol] = envelope

        # Check if debounce window has passed
        now = time.time()
        if (now - self._last_tick_dispatch) * 1000 >= self._debounce_ms:
            await self._flush_pending_ticks()

    async def _flush_pending_ticks(self) -> None:
        """Dispatch all pending debounced ticks."""
        if not self._pending_ticks:
            return

        # Dispatch all pending ticks
        for symbol, envelope in list(self._pending_ticks.items()):
            await self._dispatch_async(envelope)

        self._pending_ticks.clear()
        self._last_tick_dispatch = time.time()

    async def _dispatch_async(self, envelope: EventEnvelope) -> None:
        """Dispatch event to all subscribers."""
        event_type = envelope.event_type
        payload = envelope.payload

        # Dispatch to sync subscribers
        for callback in self._subscribers.get(event_type, []):
            try:
                # M2: Offload heavy callbacks to thread pool
                if callback in self._heavy_callbacks:
                    asyncio.create_task(self._run_heavy_callback(callback, payload))
                else:
                    callback(payload)
                self._stats["dispatched"] += 1
            except Exception as e:
                self._stats["errors"] += 1
                logger.error(f"Error in sync subscriber for {event_type.value}: {e}", exc_info=True)

        # Dispatch to async subscribers
        for callback in self._async_subscribers.get(event_type, []):
            try:
                result = callback(payload)
                if asyncio.iscoroutine(result):
                    await result
                self._stats["dispatched"] += 1
            except Exception as e:
                self._stats["errors"] += 1
                logger.error(f"Error in async subscriber for {event_type.value}: {e}", exc_info=True)

    async def _run_heavy_callback(self, callback: Callable, payload: Any) -> None:
        """Run heavy callback in thread pool with semaphore limiting."""
        async with self._heavy_semaphore:
            try:
                await asyncio.to_thread(callback, payload)
            except Exception as e:
                self._stats["errors"] += 1
                logger.error(f"Error in heavy callback: {e}", exc_info=True)

    def register_heavy_callback(self, callback: Callable) -> None:
        """Register a callback as heavy (will run in thread pool)."""
        with self._lock:
            self._heavy_callbacks.add(callback)

    def unregister_heavy_callback(self, callback: Callable) -> None:
        """Unregister a heavy callback."""
        with self._lock:
            self._heavy_callbacks.discard(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        with self._lock:
            stats = dict(self._stats)
            stats["queue_size"] = self._queue.qsize() if self._queue else 0
            stats["queue_max"] = self._max_queue_size
            stats["queue_utilization_pct"] = (
                100 * stats["queue_size"] // self._max_queue_size
                if self._max_queue_size > 0 else 0
            )
            stats["subscriber_count"] = sum(len(v) for v in self._subscribers.values())
            stats["async_subscriber_count"] = sum(len(v) for v in self._async_subscribers.values())
            return stats

    @property
    def is_running(self) -> bool:
        """Check if event bus is running."""
        return self._running
