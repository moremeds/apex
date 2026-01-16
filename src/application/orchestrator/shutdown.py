"""
ShutdownCoordinator - Coordinates graceful shutdown of async resources.

Provides unified shutdown with:
- Configurable timeout per phase
- Ordered resource cleanup:
  1. Stop event bus (prevent new work)
  2. Cancel tasks (with grace period)
  3. Close connections (broker, database)
  4. Run cleanup handlers (flush stores)
- Progress logging for debugging shutdown issues
- Error collection without propagation (best-effort cleanup)
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine, Dict, List, Optional, TYPE_CHECKING

from ...utils.logging_setup import get_logger

if TYPE_CHECKING:
    from ...domain.interfaces.event_bus import EventBus

logger = get_logger(__name__)


class ShutdownCoordinator:
    """
    Coordinates graceful shutdown of all async resources.

    Shutdown order:
    1. Stop event bus (stop accepting new work)
    2. Cancel background tasks with grace period
    3. Close connections (broker, database)
    4. Flush pending writes

    Usage:
        coordinator = ShutdownCoordinator(event_bus=event_bus)
        coordinator.register_task(background_task)
        coordinator.register_connection_closer(ib_pool.disconnect)
        coordinator.register_cleanup(container.cleanup)

        # On shutdown signal:
        await coordinator.shutdown(timeout=10.0)
    """

    def __init__(
        self,
        event_bus: Optional["EventBus"] = None,
    ) -> None:
        """
        Initialize the shutdown coordinator.

        Args:
            event_bus: Event bus to stop on shutdown.
        """
        self._event_bus = event_bus
        self._shutdown_requested = False
        self._shutdown_complete = False

        # Registered resources for cleanup
        self._tasks: List[asyncio.Task] = []
        self._connection_closers: List[Callable[[], Coroutine]] = []
        self._cleanup_handlers: List[Callable[[], Coroutine]] = []
        self._errors: List[str] = []

    @property
    def shutdown_requested(self) -> bool:
        """Whether shutdown has been requested."""
        return self._shutdown_requested

    @property
    def shutdown_complete(self) -> bool:
        """Whether shutdown has completed."""
        return self._shutdown_complete

    def register_task(self, task: asyncio.Task) -> None:
        """Register a background task to cancel on shutdown."""
        self._tasks.append(task)

    def register_connection_closer(
        self, closer: Callable[[], Coroutine]
    ) -> None:
        """Register a connection closer to call on shutdown."""
        self._connection_closers.append(closer)

    def register_cleanup(self, handler: Callable[[], Coroutine]) -> None:
        """Register a cleanup handler to call on shutdown."""
        self._cleanup_handlers.append(handler)

    async def shutdown(self, timeout: float = 10.0) -> Dict[str, Any]:
        """
        Perform graceful shutdown of all resources.

        Args:
            timeout: Maximum time (seconds) for each phase.

        Returns:
            Dict with shutdown results:
            {
                "success": bool,
                "duration_seconds": float,
                "phases_completed": int,
                "errors": List[str],
            }
        """
        if self._shutdown_requested:
            logger.warning("Shutdown already in progress")
            return {"success": False, "errors": ["Shutdown already requested"]}

        self._shutdown_requested = True
        self._errors = []
        start_time = asyncio.get_event_loop().time()

        logger.info("Shutdown coordinator starting graceful shutdown")

        # Phase 1: Stop event bus
        await self._phase_stop_event_bus(timeout)

        # Phase 2: Cancel tasks
        await self._phase_cancel_tasks(timeout)

        # Phase 3: Close connections
        await self._phase_close_connections(timeout)

        # Phase 4: Run cleanup handlers
        await self._phase_run_cleanup(timeout)

        self._shutdown_complete = True
        duration = asyncio.get_event_loop().time() - start_time

        result = {
            "success": len(self._errors) == 0,
            "duration_seconds": duration,
            "phases_completed": 4,
            "errors": self._errors,
        }

        if self._errors:
            logger.warning(
                "Shutdown completed with errors",
                extra={"duration": duration, "errors": self._errors},
            )
        else:
            logger.info(
                "Shutdown completed successfully",
                extra={"duration": duration},
            )

        return result

    async def _phase_stop_event_bus(self, timeout: float) -> None:
        """Phase 1: Stop event bus to prevent new work."""
        if not self._event_bus:
            return

        logger.debug("Shutdown phase 1: Stopping event bus")
        try:
            if hasattr(self._event_bus, "stop"):
                self._event_bus.stop()
            logger.debug("Event bus stopped")
        except Exception as e:
            self._errors.append(f"Event bus stop failed: {e}")
            logger.error(f"Failed to stop event bus: {e}")

    async def _phase_cancel_tasks(self, timeout: float) -> None:
        """Phase 2: Cancel background tasks with grace period."""
        if not self._tasks:
            return

        logger.debug(f"Shutdown phase 2: Cancelling {len(self._tasks)} tasks")
        for task in self._tasks:
            if not task.done():
                task.cancel()

        try:
            await asyncio.wait_for(
                asyncio.gather(*self._tasks, return_exceptions=True),
                timeout=timeout,
            )
            logger.debug("All tasks cancelled")
        except asyncio.TimeoutError:
            self._errors.append(f"Task cancellation timed out after {timeout}s")
            logger.warning(f"Task cancellation timed out after {timeout}s")

    async def _phase_close_connections(self, timeout: float) -> None:
        """Phase 3: Close connections gracefully."""
        if not self._connection_closers:
            return

        logger.debug(f"Shutdown phase 3: Closing {len(self._connection_closers)} connections")
        for closer in self._connection_closers:
            try:
                await asyncio.wait_for(closer(), timeout=timeout)
            except asyncio.TimeoutError:
                self._errors.append(f"Connection close timed out: {closer.__name__}")
                logger.warning(f"Connection close timed out: {closer.__name__}")
            except Exception as e:
                self._errors.append(f"Connection close failed: {e}")
                logger.error(f"Failed to close connection: {e}")

    async def _phase_run_cleanup(self, timeout: float) -> None:
        """Phase 4: Run cleanup handlers."""
        if not self._cleanup_handlers:
            return

        logger.debug(f"Shutdown phase 4: Running {len(self._cleanup_handlers)} cleanup handlers")
        for handler in self._cleanup_handlers:
            try:
                await asyncio.wait_for(handler(), timeout=timeout)
            except asyncio.TimeoutError:
                handler_name = getattr(handler, "__name__", str(handler))
                self._errors.append(f"Cleanup handler timed out: {handler_name}")
                logger.warning(f"Cleanup handler timed out: {handler_name}")
            except Exception as e:
                self._errors.append(f"Cleanup handler failed: {e}")
                logger.error(f"Cleanup handler failed: {e}")


def create_shutdown_handler(
    coordinator: ShutdownCoordinator,
    loop: asyncio.AbstractEventLoop,
) -> Callable:
    """
    Create a signal handler for graceful shutdown.

    Thread-safe: Uses call_soon_threadsafe to schedule shutdown
    from the signal handler context.

    Usage:
        import signal
        handler = create_shutdown_handler(coordinator, loop)
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
    """
    def handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown")
        if coordinator.shutdown_requested:
            return

        # Thread-safe scheduling from signal context
        if loop.is_closed():
            logger.warning("Event loop closed, cannot schedule shutdown")
            return

        def schedule_shutdown():
            if not coordinator.shutdown_requested:
                loop.create_task(coordinator.shutdown())

        try:
            loop.call_soon_threadsafe(schedule_shutdown)
        except RuntimeError:
            logger.warning("Failed to schedule shutdown (loop may be closing)")

    return handler
