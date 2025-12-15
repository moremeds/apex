"""Priority-based event bus with dual-lane dispatch."""

from __future__ import annotations
import asyncio
import time
from typing import Callable, Any, Dict, List, Optional, TYPE_CHECKING
from collections import defaultdict
from threading import Lock
from dataclasses import dataclass
from abc import ABC

from .event_types import (
    EventType, EventPriority, EVENT_PRIORITY_MAP,
    FAST_LANE_THRESHOLD, PriorityEventEnvelope, DROPPABLE_EVENTS,
    validate_event_payload,
)
from ...utils.logging_setup import get_logger

if TYPE_CHECKING:
    pass  # EventBus only needed for type checking, but we define methods inline

logger = get_logger(__name__)


class PriorityEventBus:
    """
    Priority-based event bus with dual-lane dispatch.

    Fast Lane: Risk, Trading, Market Data, Position events
    - Priority queue for proper ordering
    - No debouncing
    - Immediate dispatch with budget/timeslice limits

    Slow Lane: Snapshot, Diagnostic, UI events
    - Bounded queue
    - Debouncing enabled
    - Batched dispatch with coalescing

    Critical Fix: Fast-lane budget (500 events or 50ms timeslice) before yielding
    to slow lane. Slow lane runs on timer (>=200ms) even when fast lane is busy.
    """

    def __init__(
        self,
        fast_lane_max_size: int = 10000,
        slow_lane_max_size: int = 1000,
        slow_lane_debounce_ms: int = 100,
        fast_budget: int = 500,
        fast_time_slice_ms: int = 50,
        slow_lane_min_interval_ms: int = 200,
        max_pending_slow: int = 100,
        validate_payloads: bool = False,
    ):
        """
        Initialize priority event bus.

        Args:
            fast_lane_max_size: Maximum events in fast queue
            slow_lane_max_size: Maximum events in slow queue
            slow_lane_debounce_ms: Debounce window for slow events
            fast_budget: Max events per fast-lane burst before yielding
            fast_time_slice_ms: Max time (ms) before yielding to slow lane
            slow_lane_min_interval_ms: Minimum interval between slow dispatches
            max_pending_slow: Maximum pending slow events before dropping
            validate_payloads: If True, warn when payloads don't match expected DomainEvent types
        """
        # Queue sizes
        self._fast_lane_max_size = fast_lane_max_size
        self._slow_lane_max_size = slow_lane_max_size

        # Queues (created in start())
        self._fast_queue: Optional[asyncio.PriorityQueue] = None
        self._slow_queue: Optional[asyncio.Queue] = None

        # Subscribers
        self._subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._async_subscribers: Dict[EventType, List[Callable]] = defaultdict(list)

        # State
        self._running = False
        self._fast_task: Optional[asyncio.Task] = None
        self._slow_task: Optional[asyncio.Task] = None
        self._sequence = 0  # Global sequence counter for ordering
        self._lock = Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Slow lane configuration
        self._slow_lane_debounce_ms = slow_lane_debounce_ms
        self._pending_slow: Dict[str, PriorityEventEnvelope] = {}
        self._last_slow_dispatch: float = 0
        self._max_pending_slow = max_pending_slow

        # Fast lane configuration (starvation prevention)
        self._fast_budget = fast_budget
        self._fast_time_slice_ms = fast_time_slice_ms
        self._slow_lane_min_interval_ms = slow_lane_min_interval_ms

        # Payload validation
        self._validate_payloads = validate_payloads
        self._validation_warnings: set = set()  # Track warned event types

        # Stats
        self._stats = {
            "fast_published": 0,
            "fast_dispatched": 0,
            "slow_published": 0,
            "slow_dispatched": 0,
            "dropped": 0,
            "errors": 0,
            "fast_high_water": 0,  # Peak fast queue depth
            "slow_high_water": 0,  # Peak slow pending depth
        }

    def publish(self, event_type: EventType, payload: Any, source: str = "") -> None:
        """
        Publish event to appropriate lane based on priority.

        Thread-safe, can be called from any context.

        Args:
            event_type: Type of event
            payload: Event data (DomainEvent subclass preferred, dict for backward compat)
            source: Optional source identifier
        """
        # Validate payload type if enabled
        if self._validate_payloads and event_type not in self._validation_warnings:
            if not validate_event_payload(event_type, payload):
                self._validation_warnings.add(event_type)
                logger.warning(
                    f"Payload type mismatch for {event_type.value}: "
                    f"got {type(payload).__name__}, expected DomainEvent subclass. "
                    "Consider using typed domain events for type safety."
                )

        priority = EVENT_PRIORITY_MAP.get(event_type, EventPriority.CONTROL)

        with self._lock:
            self._sequence += 1
            envelope = PriorityEventEnvelope(
                priority=priority,
                sequence=self._sequence,
                event_type=event_type,
                payload=payload,
                source=source,
            )

        if not self._running:
            # Sync fallback
            self._dispatch_sync(envelope)
            return

        # Route to appropriate lane
        if priority < FAST_LANE_THRESHOLD:
            self._enqueue_fast(envelope)
        else:
            self._enqueue_slow(envelope)

    def _enqueue_fast(self, envelope: PriorityEventEnvelope) -> None:
        """Enqueue to fast lane (priority queue)."""
        try:
            self._fast_queue.put_nowait(envelope)
            with self._lock:
                self._stats["fast_published"] += 1
                current = self._fast_queue.qsize()
                if current > self._stats["fast_high_water"]:
                    self._stats["fast_high_water"] = current
        except asyncio.QueueFull:
            with self._lock:
                self._stats["dropped"] += 1
            logger.warning(f"Fast lane full, dropped {envelope.event_type.value}")

    def _enqueue_slow(self, envelope: PriorityEventEnvelope) -> None:
        """
        Enqueue to slow lane with debouncing and coalescing.

        Coalesces by (event_type, symbol) to be meaningful.
        Bounds the pending_slow dict and defines drop policy.
        """
        # Coalesce by (event_type, symbol) for meaningful deduplication
        symbol = ""
        if hasattr(envelope.payload, 'symbol'):
            symbol = envelope.payload.symbol
        elif isinstance(envelope.payload, dict):
            symbol = envelope.payload.get('symbol', '')

        key = f"{envelope.event_type.value}:{symbol}"

        with self._lock:
            self._pending_slow[key] = envelope
            self._stats["slow_published"] += 1

            current = len(self._pending_slow)
            if current > self._stats["slow_high_water"]:
                self._stats["slow_high_water"] = current

            # Drop policy: if too many pending, drop lowest priority events first
            if current > self._max_pending_slow:
                self._apply_drop_policy()

    def _apply_drop_policy(self) -> None:
        """Apply drop policy when slow queue is overloaded."""
        # Find and remove droppable events in priority order
        for event_type, _ in sorted(DROPPABLE_EVENTS.items(), key=lambda x: x[1]):
            prefix = event_type.value
            drop_keys = [k for k in self._pending_slow if k.startswith(prefix)]
            for key in drop_keys:
                if len(self._pending_slow) <= self._max_pending_slow:
                    return
                del self._pending_slow[key]
                self._stats["dropped"] += 1

    async def start(self) -> None:
        """Start both dispatch loops."""
        if self._running:
            logger.warning("PriorityEventBus already running")
            return

        self._loop = asyncio.get_event_loop()
        self._fast_queue = asyncio.PriorityQueue(maxsize=self._fast_lane_max_size)
        self._slow_queue = asyncio.Queue(maxsize=self._slow_lane_max_size)
        self._running = True

        self._fast_task = asyncio.create_task(self._fast_dispatch_loop())
        self._slow_task = asyncio.create_task(self._slow_dispatch_loop())

        logger.info("PriorityEventBus started (dual-lane)")

    async def stop(self) -> None:
        """Stop dispatch loops and drain queues."""
        if not self._running:
            return

        logger.info("Stopping PriorityEventBus...")
        self._running = False

        # Publish shutdown event to fast lane (critical priority)
        self.publish(EventType.SHUTDOWN, {"timestamp": time.time()})

        for task in [self._fast_task, self._slow_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Drain remaining events
        await self._drain_queues()

        logger.info(f"PriorityEventBus stopped. Stats: {self._stats}")

    async def _drain_queues(self) -> None:
        """Drain remaining events in both queues."""
        drained = 0

        # Drain fast queue
        if self._fast_queue:
            while not self._fast_queue.empty():
                try:
                    envelope = self._fast_queue.get_nowait()
                    await self._dispatch(envelope)
                    drained += 1
                except asyncio.QueueEmpty:
                    break
                except Exception as e:
                    logger.error(f"Error draining fast queue: {e}")

        # Drain pending slow events
        for envelope in self._pending_slow.values():
            try:
                await self._dispatch(envelope)
                drained += 1
            except Exception as e:
                logger.error(f"Error draining slow queue: {e}")
        self._pending_slow.clear()

        if drained > 0:
            logger.info(f"Drained {drained} events from queues")

    async def _fast_dispatch_loop(self) -> None:
        """
        Fast lane dispatch - processes events with budget to prevent slow-lane starvation.

        Continuous ticks can freeze slow-path (dashboard/health) and can look like a
        dead process to monitoring. We add a budget (500 events or 50ms) before yielding
        to slow lane.
        """
        while self._running:
            try:
                burst_start = time.time()
                burst_count = 0

                # Process fast events with budget/timeslice limit
                while not self._fast_queue.empty():
                    # Check budget limits
                    elapsed_ms = (time.time() - burst_start) * 1000
                    if burst_count >= self._fast_budget or elapsed_ms >= self._fast_time_slice_ms:
                        # Yield to slow lane
                        break

                    try:
                        envelope = self._fast_queue.get_nowait()
                        await self._dispatch(envelope)
                        with self._lock:
                            self._stats["fast_dispatched"] += 1
                        burst_count += 1
                    except asyncio.QueueEmpty:
                        break

                # Allow slow lane to run (even when fast events pending)
                await asyncio.sleep(0.001)  # 1ms yield

            except asyncio.CancelledError:
                break
            except Exception as e:
                with self._lock:
                    self._stats["errors"] += 1
                logger.error(f"Fast dispatch error: {e}", exc_info=True)

    async def _slow_dispatch_loop(self) -> None:
        """
        Slow lane dispatch - debounced, runs on timer even when fast lane busy.

        Ensures slow lane runs on a timer even when fast lane is busy.
        This prevents dashboard/health from appearing dead.
        """
        last_slow_dispatch = time.time()

        while self._running:
            try:
                # Wait for debounce window
                await asyncio.sleep(self._slow_lane_debounce_ms / 1000)

                # Run slow lane periodically regardless of fast lane state
                time_since_last = (time.time() - last_slow_dispatch) * 1000

                # Skip only if within min interval AND fast queue is busy
                if time_since_last < self._slow_lane_min_interval_ms:
                    if self._fast_queue and not self._fast_queue.empty():
                        continue

                # Flush pending slow events (coalesce by event_type + symbol)
                with self._lock:
                    pending = list(self._pending_slow.items())
                    self._pending_slow.clear()

                for key, envelope in pending:
                    await self._dispatch(envelope)
                    with self._lock:
                        self._stats["slow_dispatched"] += 1

                last_slow_dispatch = time.time()

            except asyncio.CancelledError:
                break
            except Exception as e:
                with self._lock:
                    self._stats["errors"] += 1
                logger.error(f"Slow dispatch error: {e}", exc_info=True)

    async def _dispatch(self, envelope: PriorityEventEnvelope) -> None:
        """Dispatch to all subscribers."""
        event_type = envelope.event_type
        payload = envelope.payload

        # Sync subscribers
        for cb in self._subscribers.get(event_type, []):
            try:
                cb(payload)
            except Exception as e:
                with self._lock:
                    self._stats["errors"] += 1
                logger.error(f"Subscriber error for {event_type.value}: {e}")

        # Async subscribers
        for cb in self._async_subscribers.get(event_type, []):
            try:
                result = cb(payload)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                with self._lock:
                    self._stats["errors"] += 1
                logger.error(f"Async subscriber error for {event_type.value}: {e}")

    def _dispatch_sync(self, envelope: PriorityEventEnvelope) -> None:
        """Sync dispatch fallback (when not running)."""
        for cb in self._subscribers.get(envelope.event_type, []):
            try:
                cb(envelope.payload)
            except Exception as e:
                logger.error(f"Sync dispatch error: {e}")

    def subscribe(self, event_type: EventType, callback: Callable) -> None:
        """Subscribe sync callback to event type."""
        with self._lock:
            self._subscribers[event_type].append(callback)
        logger.debug(f"Subscribed to {event_type.value}")

    def subscribe_async(self, event_type: EventType, callback: Callable) -> None:
        """Subscribe async callback to event type."""
        with self._lock:
            self._async_subscribers[event_type].append(callback)
        logger.debug(f"Async subscribed to {event_type.value}")

    def unsubscribe(self, event_type: EventType, callback: Callable) -> None:
        """Unsubscribe a callback from an event type."""
        with self._lock:
            if callback in self._subscribers.get(event_type, []):
                self._subscribers[event_type].remove(callback)
                logger.debug(f"Unsubscribed from {event_type.value}")
            if callback in self._async_subscribers.get(event_type, []):
                self._async_subscribers[event_type].remove(callback)
                logger.debug(f"Async unsubscribed from {event_type.value}")

    def get_stats(self) -> Dict[str, Any]:
        """Get bus statistics."""
        with self._lock:
            return {
                **self._stats,
                "fast_queue_size": self._fast_queue.qsize() if self._fast_queue else 0,
                "fast_queue_max": self._fast_lane_max_size,
                "slow_pending": len(self._pending_slow),
                "slow_pending_max": self._max_pending_slow,
                "running": self._running,
            }

    @property
    def is_running(self) -> bool:
        """Check if event bus is running."""
        return self._running
