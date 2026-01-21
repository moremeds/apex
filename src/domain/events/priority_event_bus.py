"""Priority-based event bus with dual-lane dispatch."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from ...utils.logging_setup import get_logger
from .event_types import (
    DROPPABLE_EVENTS,
    EVENT_PRIORITY_MAP,
    FAST_LANE_THRESHOLD,
    EventPriority,
    EventType,
    PriorityEventEnvelope,
    validate_event_payload,
)

if TYPE_CHECKING:
    pass  # EventBus only needed for type checking, but we define methods inline

logger = get_logger(__name__)


# OPT-008: Atomic counter for lock-free stats updates
class AtomicCounter:
    """Thread-safe counter using fine-grained locking."""

    __slots__ = ("_value", "_lock")

    def __init__(self, initial: int = 0):
        self._value = initial
        self._lock = Lock()

    def increment(self, amount: int = 1) -> int:
        """Atomically increment and return new value."""
        with self._lock:
            self._value += amount
            return self._value

    def max_update(self, value: int) -> None:
        """Atomically update to max of current and new value."""
        with self._lock:
            if value > self._value:
                self._value = value

    @property
    def value(self) -> int:
        """Read current value (lock-free read is safe for int)."""
        return self._value


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
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # HIGH-009: Fine-grained locks to reduce contention
        # Each lock protects a specific data structure
        self._sequence_lock = Lock()  # Protects _sequence counter
        self._subscribers_lock = (
            Lock()
        )  # Protects _subscribers, _async_subscribers, _heavy_callbacks
        self._slow_queue_lock = Lock()  # Protects _pending_slow dict

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

        # OPT-008: Atomic counters for lock-free stats updates
        # Each counter has its own lock, eliminating the global stats lock bottleneck
        self._fast_published = AtomicCounter()
        self._fast_dispatched = AtomicCounter()
        self._slow_published = AtomicCounter()
        self._slow_dispatched = AtomicCounter()
        self._dropped = AtomicCounter()
        self._errors = AtomicCounter()
        self._fast_high_water = AtomicCounter()
        self._slow_high_water = AtomicCounter()
        self._heavy_callbacks_offloaded = AtomicCounter()

        # Heavy callback protection: callbacks registered here run in thread pool
        self._heavy_callbacks: set[Callable] = set()
        self._heavy_semaphore: Optional[asyncio.Semaphore] = None
        self._max_concurrent_heavy: int = 4  # Max concurrent heavy callbacks
        self._heavy_tasks: set[asyncio.Task] = set()  # Track tasks for cleanup on shutdown

    def publish(self, event_type: EventType, payload: Any, source: str = "") -> None:
        """
        Publish event to appropriate lane based on priority.

        Thread-safe, can be called from any context.
        Uses call_soon_threadsafe when called from non-loop threads.

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

        # HIGH-009: Fine-grained lock for sequence counter only
        with self._sequence_lock:
            self._sequence += 1
            seq = self._sequence

        envelope = PriorityEventEnvelope(
            priority=priority,
            sequence=seq,
            event_type=event_type,
            payload=payload,
            source=source,
        )

        if not self._running:
            # Sync fallback
            self._dispatch_sync(envelope)
            return

        # Determine if we're on the event loop thread
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        # Route to appropriate lane with thread-safety
        if running_loop is self._loop:
            # On loop thread: enqueue directly
            self._enqueue_to_lane(envelope, priority)
        else:
            # Cross-thread: use call_soon_threadsafe
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._enqueue_to_lane, envelope, priority)
            else:
                # Fallback if loop not available
                self._dispatch_sync(envelope)

    def _enqueue_to_lane(self, envelope: PriorityEventEnvelope, priority: EventPriority) -> None:
        """
        Enqueue to appropriate lane (must be called from loop thread).

        This method is called either directly from the loop thread,
        or via call_soon_threadsafe from other threads.
        """
        if priority < FAST_LANE_THRESHOLD:
            self._enqueue_fast(envelope)
        else:
            self._enqueue_slow(envelope)

    def _enqueue_fast(self, envelope: PriorityEventEnvelope) -> None:
        """Enqueue to fast lane (priority queue)."""
        if self._fast_queue is None:
            return
        try:
            self._fast_queue.put_nowait(envelope)
            # OPT-008: Lock-free stats update
            self._fast_published.increment()
            current = self._fast_queue.qsize()
            self._fast_high_water.max_update(current)
        except asyncio.QueueFull:
            self._dropped.increment()
            logger.warning(f"Fast lane full, dropped {envelope.event_type.value}")

    def _enqueue_slow(self, envelope: PriorityEventEnvelope) -> None:
        """
        Enqueue to slow lane with debouncing and coalescing.

        Coalesces by (event_type, symbol) to be meaningful.
        Bounds the pending_slow dict and defines drop policy.
        """
        # Coalesce by (event_type, symbol) for meaningful deduplication
        symbol = ""
        if hasattr(envelope.payload, "symbol"):
            symbol = envelope.payload.symbol
        elif isinstance(envelope.payload, dict):
            symbol = envelope.payload.get("symbol", "")

        key = f"{envelope.event_type.value}:{symbol}"

        # HIGH-009: Fine-grained lock for slow queue only
        with self._slow_queue_lock:
            self._pending_slow[key] = envelope
            current = len(self._pending_slow)
            # Drop policy: if too many pending, drop lowest priority events first
            if current > self._max_pending_slow:
                self._apply_drop_policy()

        # Stats update outside lock
        self._slow_published.increment()
        self._slow_high_water.max_update(current)

    def _apply_drop_policy(self) -> None:
        """Apply drop policy when slow queue is overloaded."""
        # Calculate how many to drop
        excess = len(self._pending_slow) - self._max_pending_slow
        if excess <= 0:
            return

        # Optimization: Pre-compute prefixes to drop in priority order
        # Sort droppable events by priority (lower value = higher priority, but here we want to drop low priority first)
        # Wait, DROPPABLE_EVENTS values are: DASHBOARD_UPDATE=1 (First to drop).
        # So we want to drop keys starting with prefixes having lower sort order in DROPPABLE_EVENTS.

        # Build a list of prefixes to drop in order
        drop_order = [e.value for e, _ in sorted(DROPPABLE_EVENTS.items(), key=lambda x: x[1])]

        # We need to drop 'excess' items.
        # Strategy: Iterate through drop_order (types), and for each type, scan keys.
        # This IS O(N*M).

        # To make it O(N):
        # Iterate keys once, categorize them into "buckets" by type.
        # Then empty buckets in drop_order.

        to_drop = []
        buckets = defaultdict(list)

        # Scan once - O(N)
        for key in self._pending_slow:
            # key format is "event_type:symbol"
            type_prefix = key.split(":")[0]
            buckets[type_prefix].append(key)

        # Select keys to drop based on priority - O(M)
        for prefix in drop_order:
            keys = buckets.get(prefix, [])
            if not keys:
                continue

            # Take as many as needed from this bucket
            count = len(keys)
            if count >= excess:
                to_drop.extend(keys[:excess])
                excess = 0
                break
            else:
                to_drop.extend(keys)
                excess -= count

        # If we still have excess (non-droppable types filling queue?),
        # we might need to force drop oldest or just warn.
        # For now, just execute the drops.
        for key in to_drop:
            del self._pending_slow[key]
            self._dropped.increment()

    async def start(self) -> None:
        """Start both dispatch loops."""
        if self._running:
            logger.warning("PriorityEventBus already running")
            return

        # Use get_running_loop() for Python 3.10+ compatibility
        self._loop = asyncio.get_running_loop()
        self._fast_queue = asyncio.PriorityQueue(maxsize=self._fast_lane_max_size)
        self._slow_queue = asyncio.Queue(maxsize=self._slow_lane_max_size)
        self._heavy_semaphore = asyncio.Semaphore(self._max_concurrent_heavy)
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

        # Cancel any outstanding heavy callback tasks
        if self._heavy_tasks:
            logger.debug(f"Cancelling {len(self._heavy_tasks)} outstanding heavy callback tasks")
            for task in list(self._heavy_tasks):
                task.cancel()
            await asyncio.gather(*self._heavy_tasks, return_exceptions=True)
            self._heavy_tasks.clear()

        # Drain remaining events
        await self._drain_queues()

        logger.info(f"PriorityEventBus stopped. Stats: {self.get_stats()}")

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
                    logger.exception(f"Error draining fast queue: {e}")

        # Drain pending slow events
        for envelope in self._pending_slow.values():
            try:
                await self._dispatch(envelope)
                drained += 1
            except Exception as e:
                logger.exception(f"Error draining slow queue: {e}")
        self._pending_slow.clear()

        if drained > 0:
            logger.info(f"Drained {drained} events from queues")

    async def _fast_dispatch_loop(self) -> None:
        """
        Fast lane dispatch - processes events with budget to prevent slow-lane starvation.

        Continuous ticks can freeze slow-path (dashboard/health) and can look like a
        dead process to monitoring. We add a budget (500 events or 50ms) before yielding
        to slow lane.

        M3 fix: Use adaptive backoff - longer sleep when idle to reduce CPU burn.
        """
        idle_backoff_ms = 10  # Sleep longer when queue is empty (saves CPU)
        busy_yield_ms = 2  # Short yield when more events are pending

        while self._running:
            try:
                burst_start = time.perf_counter()
                burst_count = 0
                queue_was_empty = True

                # Process fast events with budget/timeslice limit
                while self._fast_queue is not None and not self._fast_queue.empty():
                    queue_was_empty = False
                    # Check budget limits
                    elapsed_ms = (time.perf_counter() - burst_start) * 1000
                    if burst_count >= self._fast_budget or elapsed_ms >= self._fast_time_slice_ms:
                        # Yield to slow lane
                        break

                    try:
                        envelope = self._fast_queue.get_nowait()
                        await self._dispatch(envelope)
                        # OPT-008: Lock-free stats update
                        self._fast_dispatched.increment()
                        burst_count += 1
                    except asyncio.QueueEmpty:
                        break

                # C8 + M3: Adaptive backoff to balance latency vs CPU usage
                # - When idle (queue empty): longer sleep to save CPU
                # - When busy (budget hit): short yield to let slow lane run
                if queue_was_empty:
                    await asyncio.sleep(idle_backoff_ms / 1000)
                else:
                    await asyncio.sleep(busy_yield_ms / 1000)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._errors.increment()
                logger.error(f"Fast dispatch error: {e}", exc_info=True)

    async def _slow_dispatch_loop(self) -> None:
        """
        Slow lane dispatch - guaranteed minimum interval regardless of fast lane.

        Ensures slow lane runs on a timer even when fast lane is busy.
        This prevents dashboard/health from appearing dead.
        """
        while self._running:
            try:
                # Wait for minimum interval (guaranteed processing)
                await asyncio.sleep(self._slow_lane_min_interval_ms / 1000)

                # Flush pending slow events (coalesce by event_type + symbol)
                # HIGH-009: Fine-grained lock for slow queue only
                with self._slow_queue_lock:
                    pending = list(self._pending_slow.items())
                    self._pending_slow.clear()

                for key, envelope in pending:
                    await self._dispatch(envelope)
                    # OPT-008: Lock-free stats update
                    self._slow_dispatched.increment()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._errors.increment()
                logger.error(f"Slow dispatch error: {e}", exc_info=True)

    async def _dispatch(self, envelope: PriorityEventEnvelope) -> None:
        """
        Dispatch to all subscribers.

        Heavy callbacks (registered via register_heavy_callback) are offloaded
        to a thread pool to prevent blocking the event loop.
        """
        event_type = envelope.event_type
        payload = envelope.payload

        # HIGH-009 FIX: Copy subscriber lists while holding lock to prevent race conditions
        # during concurrent subscribe/unsubscribe operations
        with self._subscribers_lock:
            sync_callbacks = list(self._subscribers.get(event_type, []))
            async_callbacks = list(self._async_subscribers.get(event_type, []))
            heavy_set = set(self._heavy_callbacks)

        # Sync subscribers (iterate over copy)
        for cb in sync_callbacks:
            try:
                if cb in heavy_set:
                    # Offload heavy callback to thread pool with task tracking
                    task = asyncio.create_task(self._run_heavy_callback(cb, payload))
                    self._heavy_tasks.add(task)
                    task.add_done_callback(self._heavy_tasks.discard)
                else:
                    cb(payload)
            except Exception as e:
                self._errors.increment()
                logger.exception(f"Subscriber error for {event_type.value}: {e}")

        # Async subscribers (iterate over copy)
        for cb in async_callbacks:
            try:
                result = cb(payload)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                self._errors.increment()
                logger.exception(f"Async subscriber error for {event_type.value}: {e}")

    async def _run_heavy_callback(self, cb: Callable, payload: Any) -> None:
        """Run a heavy callback in thread pool with semaphore protection."""
        if self._heavy_semaphore is None:
            return
        async with self._heavy_semaphore:
            try:
                await asyncio.to_thread(cb, payload)
                # OPT-008: Lock-free stats update
                self._heavy_callbacks_offloaded.increment()
            except Exception as e:
                self._errors.increment()
                logger.exception(f"Heavy callback error: {e}")

    def _dispatch_sync(self, envelope: PriorityEventEnvelope) -> None:
        """Sync dispatch fallback (when not running)."""
        # HIGH-009 FIX: Copy subscriber list while holding lock
        with self._subscribers_lock:
            callbacks = list(self._subscribers.get(envelope.event_type, []))
        for cb in callbacks:
            try:
                cb(envelope.payload)
            except Exception as e:
                logger.exception(f"Sync dispatch error: {e}")

    def subscribe(self, event_type: EventType, callback: Callable) -> None:
        """Subscribe sync callback to event type."""
        # HIGH-009: Fine-grained lock for subscribers
        with self._subscribers_lock:
            self._subscribers[event_type].append(callback)
        logger.debug(f"Subscribed to {event_type.value}")

    def subscribe_async(self, event_type: EventType, callback: Callable) -> None:
        """Subscribe async callback to event type."""
        # HIGH-009: Fine-grained lock for subscribers
        with self._subscribers_lock:
            self._async_subscribers[event_type].append(callback)
        logger.debug(f"Async subscribed to {event_type.value}")

    def unsubscribe(self, event_type: EventType, callback: Callable) -> None:
        """Unsubscribe a callback from an event type."""
        # HIGH-009: Fine-grained lock for subscribers
        with self._subscribers_lock:
            if callback in self._subscribers.get(event_type, []):
                self._subscribers[event_type].remove(callback)
                logger.debug(f"Unsubscribed from {event_type.value}")
            if callback in self._async_subscribers.get(event_type, []):
                self._async_subscribers[event_type].remove(callback)
                logger.debug(f"Async unsubscribed from {event_type.value}")
            # Also remove from heavy callbacks if registered
            self._heavy_callbacks.discard(callback)

    def register_heavy_callback(self, callback: Callable) -> None:
        """
        Register a callback as "heavy" (will be offloaded to thread pool).

        Heavy callbacks are those that may take >1ms to execute, such as:
        - File I/O (logging, persistence)
        - Database writes
        - Complex computations
        - External API calls

        These will be run in a thread pool to prevent blocking market data processing.

        Args:
            callback: The callback function to mark as heavy
        """
        # HIGH-009: Fine-grained lock for subscribers
        with self._subscribers_lock:
            self._heavy_callbacks.add(callback)
        logger.debug(
            f"Registered heavy callback: {callback.__name__ if hasattr(callback, '__name__') else callback}"
        )

    def unregister_heavy_callback(self, callback: Callable) -> None:
        """Unregister a callback from the heavy callback set."""
        # HIGH-009: Fine-grained lock for subscribers
        with self._subscribers_lock:
            self._heavy_callbacks.discard(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get bus statistics (OPT-008: mostly lock-free reads)."""
        # HIGH-009: Fine-grained lock for slow queue only
        with self._slow_queue_lock:
            slow_pending = len(self._pending_slow)

        return {
            # OPT-008: Lock-free counter reads
            "fast_published": self._fast_published.value,
            "fast_dispatched": self._fast_dispatched.value,
            "slow_published": self._slow_published.value,
            "slow_dispatched": self._slow_dispatched.value,
            "dropped": self._dropped.value,
            "errors": self._errors.value,
            "fast_high_water": self._fast_high_water.value,
            "slow_high_water": self._slow_high_water.value,
            "heavy_callbacks_offloaded": self._heavy_callbacks_offloaded.value,
            "fast_queue_size": self._fast_queue.qsize() if self._fast_queue else 0,
            "fast_queue_max": self._fast_lane_max_size,
            "slow_pending": slow_pending,
            "slow_pending_max": self._max_pending_slow,
            "running": self._running,
        }

    def get_queue_health(self) -> Dict[str, Any]:
        """
        HIGH-012: Get queue health status for monitoring.

        Returns health status with severity levels:
        - healthy: <80% capacity
        - warning: 80-95% capacity
        - critical: >95% capacity

        Returns:
            Dict with health status, utilization percentages, and alerts.
        """
        fast_size = self._fast_queue.qsize() if self._fast_queue else 0
        fast_pct = (
            (fast_size / self._fast_lane_max_size) * 100 if self._fast_lane_max_size > 0 else 0
        )

        with self._slow_queue_lock:
            slow_size = len(self._pending_slow)
        slow_pct = (slow_size / self._max_pending_slow) * 100 if self._max_pending_slow > 0 else 0

        # Determine overall health
        max_pct = max(fast_pct, slow_pct)
        if max_pct >= 95:
            status = "critical"
        elif max_pct >= 80:
            status = "warning"
        else:
            status = "healthy"

        # Log warnings/criticals
        if status == "critical":
            logger.error(f"HIGH-012: Queue CRITICAL - fast={fast_pct:.1f}%, slow={slow_pct:.1f}%")
        elif status == "warning":
            logger.warning(f"HIGH-012: Queue warning - fast={fast_pct:.1f}%, slow={slow_pct:.1f}%")

        return {
            "status": status,
            "fast_queue": {
                "size": fast_size,
                "max": self._fast_lane_max_size,
                "utilization_pct": round(fast_pct, 1),
            },
            "slow_queue": {
                "size": slow_size,
                "max": self._max_pending_slow,
                "utilization_pct": round(slow_pct, 1),
            },
            "dropped_total": self._dropped.value,
            "errors_total": self._errors.value,
        }

    @property
    def is_running(self) -> bool:
        """Check if event bus is running."""
        return self._running
