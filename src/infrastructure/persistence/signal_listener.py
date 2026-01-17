"""
PostgreSQL LISTEN/NOTIFY listener for real-time signal updates.

Provides push-based notifications from the database to the TUI,
eliminating the need for polling while ensuring no signals are missed.

Uses a dedicated asyncpg connection (not from pool) for LISTEN.
Includes auto-reconnect with exponential backoff on connection loss.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import asyncpg

if TYPE_CHECKING:
    from config.models import DatabaseConfig

logger = logging.getLogger(__name__)


class SignalListener:
    """
    Listens to PostgreSQL NOTIFY channels for real-time signal updates.

    Uses a dedicated asyncpg connection (separate from the pool) for LISTEN.
    Includes auto-reconnect on connection loss with exponential backoff.

    Channels:
    - ta_signal_updates: New trading signals
    - indicator_updates: New indicator values (optional, may be chatty)
    - confluence_updates: New confluence scores

    Usage:
        listener = SignalListener(config)

        # Register callbacks before starting
        listener.on_signal(lambda payload: print(f"Signal: {payload}"))
        listener.on_indicator(lambda payload: print(f"Indicator: {payload}"))
        listener.on_confluence(lambda payload: print(f"Confluence: {payload}"))

        # Start listening (runs in background task)
        await listener.start()

        # ... later ...
        await listener.stop()
    """

    # NOTIFY channel names (must match migration triggers)
    CHANNEL_SIGNALS = "ta_signal_updates"
    CHANNEL_INDICATORS = "indicator_updates"
    CHANNEL_CONFLUENCE = "confluence_updates"

    def __init__(
        self,
        config: "DatabaseConfig",
        initial_reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 30.0,
        listen_to_indicators: bool = False,  # Off by default (chatty)
    ):
        """
        Initialize signal listener.

        Args:
            config: Database configuration with connection details.
            initial_reconnect_delay: Initial delay in seconds before reconnect.
            max_reconnect_delay: Maximum delay between reconnect attempts.
            listen_to_indicators: Whether to listen to indicator updates (chatty).
        """
        self._config = config
        self._initial_delay = initial_reconnect_delay
        self._max_delay = max_reconnect_delay
        self._listen_to_indicators = listen_to_indicators

        self._conn: Optional[asyncpg.Connection] = None
        self._running = False
        self._listen_task: Optional[asyncio.Task] = None

        # Callbacks by channel type
        self._callbacks: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {
            "signal": [],
            "indicator": [],
            "confluence": [],
        }

        # Statistics
        self._connected = False
        self._reconnect_count = 0
        self._messages_received = 0

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def on_signal(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register callback for trading signal notifications.

        Args:
            callback: Function called with signal payload dict.
                     Keys: type, signal_id, symbol, timeframe, direction, indicator, strength, time
        """
        self._callbacks["signal"].append(callback)

    def on_indicator(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register callback for indicator value notifications.

        Args:
            callback: Function called with indicator payload dict.
                     Keys: type, symbol, timeframe, indicator, time
        """
        self._callbacks["indicator"].append(callback)

    def on_confluence(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register callback for confluence score notifications.

        Args:
            callback: Function called with confluence payload dict.
                     Keys: type, symbol, timeframe, alignment_score, dominant_direction, time
        """
        self._callbacks["confluence"].append(callback)

    async def start(self) -> None:
        """
        Start listening for notifications.

        Creates a background task that connects to PostgreSQL and
        listens for NOTIFY messages. Automatically reconnects on failure.
        """
        if self._running:
            logger.warning("SignalListener already running")
            return

        self._running = True
        self._listen_task = asyncio.create_task(self._listen_loop())
        logger.info("SignalListener started")

    async def stop(self) -> None:
        """
        Stop listening and disconnect.

        Cancels the listen task and closes the connection gracefully.
        """
        self._running = False

        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None

        await self._disconnect()
        logger.info(
            f"SignalListener stopped (messages_received={self._messages_received}, "
            f"reconnects={self._reconnect_count})"
        )

    @property
    def is_connected(self) -> bool:
        """Check if currently connected to database."""
        return self._connected and self._conn is not None and not self._conn.is_closed()

    @property
    def stats(self) -> Dict[str, Any]:
        """Get listener statistics."""
        return {
            "connected": self.is_connected,
            "running": self._running,
            "messages_received": self._messages_received,
            "reconnect_count": self._reconnect_count,
            "listen_to_indicators": self._listen_to_indicators,
        }

    # -------------------------------------------------------------------------
    # Internal: Connection Management
    # -------------------------------------------------------------------------

    async def _connect(self) -> bool:
        """
        Establish connection and set up listeners.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            # Build DSN from config
            dsn = self._config.dsn

            # Create dedicated connection (not from pool)
            self._conn = await asyncpg.connect(dsn)

            # Add NOTIFY listeners
            await self._conn.add_listener(self.CHANNEL_SIGNALS, self._on_notify)
            await self._conn.add_listener(self.CHANNEL_CONFLUENCE, self._on_notify)

            if self._listen_to_indicators:
                await self._conn.add_listener(self.CHANNEL_INDICATORS, self._on_notify)

            self._connected = True
            logger.info(
                f"SignalListener connected to PostgreSQL "
                f"(channels: {self.CHANNEL_SIGNALS}, {self.CHANNEL_CONFLUENCE}"
                f"{', ' + self.CHANNEL_INDICATORS if self._listen_to_indicators else ''})"
            )
            return True

        except Exception as e:
            logger.error(f"SignalListener failed to connect: {e}")
            self._connected = False
            return False

    async def _disconnect(self) -> None:
        """Disconnect from database."""
        if self._conn:
            try:
                await self._conn.remove_listener(self.CHANNEL_SIGNALS, self._on_notify)
                await self._conn.remove_listener(self.CHANNEL_CONFLUENCE, self._on_notify)
                if self._listen_to_indicators:
                    await self._conn.remove_listener(self.CHANNEL_INDICATORS, self._on_notify)
                await self._conn.close()
            except Exception as e:
                logger.warning(f"Error during SignalListener disconnect: {e}")
            finally:
                self._conn = None
                self._connected = False

    # -------------------------------------------------------------------------
    # Internal: Listen Loop with Auto-Reconnect
    # -------------------------------------------------------------------------

    async def _listen_loop(self) -> None:
        """
        Main listen loop with auto-reconnect on failure.

        Uses exponential backoff between reconnect attempts.
        """
        delay = self._initial_delay

        while self._running:
            try:
                # Attempt connection
                if not await self._connect():
                    raise asyncpg.PostgresConnectionError("Connection failed")

                # Reset delay on successful connection
                delay = self._initial_delay

                # Keep connection alive - asyncpg handles NOTIFY in background
                # We just need to keep the connection open and handle errors
                while self._running and self._conn and not self._conn.is_closed():
                    # Periodically check connection health
                    await asyncio.sleep(1.0)

                    # Connection check via simple query (optional - adds overhead)
                    # try:
                    #     await self._conn.fetchval("SELECT 1")
                    # except Exception:
                    #     break

            except asyncpg.PostgresConnectionError as e:
                self._connected = False
                self._reconnect_count += 1
                logger.warning(
                    f"SignalListener connection lost: {e}, "
                    f"reconnecting in {delay:.1f}s (attempt #{self._reconnect_count})"
                )
                await self._disconnect()
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._max_delay)  # Exponential backoff

            except asyncio.CancelledError:
                logger.debug("SignalListener listen loop cancelled")
                break

            except Exception as e:
                self._connected = False
                self._reconnect_count += 1
                logger.error(
                    f"SignalListener unexpected error: {e}, " f"reconnecting in {delay:.1f}s"
                )
                await self._disconnect()
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._max_delay)

    # -------------------------------------------------------------------------
    # Internal: NOTIFY Handler
    # -------------------------------------------------------------------------

    def _on_notify(
        self,
        connection: asyncpg.Connection,
        pid: int,
        channel: str,
        payload: str,
    ) -> None:
        """
        Handle incoming NOTIFY message.

        Called by asyncpg when a notification arrives on a listened channel.
        Parses the JSON payload and dispatches to registered callbacks.

        Args:
            connection: The connection that received the notification.
            pid: Process ID of the notifying backend.
            channel: Channel name the notification was sent on.
            payload: JSON string payload from the trigger.
        """
        self._messages_received += 1

        try:
            data = json.loads(payload)
        except json.JSONDecodeError as e:
            logger.error(f"SignalListener failed to parse payload: {e}")
            return

        # Dispatch based on channel
        if channel == self.CHANNEL_SIGNALS:
            self._dispatch("signal", data)
        elif channel == self.CHANNEL_INDICATORS:
            self._dispatch("indicator", data)
        elif channel == self.CHANNEL_CONFLUENCE:
            self._dispatch("confluence", data)
        else:
            logger.warning(f"SignalListener received notification on unknown channel: {channel}")

    def _dispatch(self, callback_type: str, data: Dict[str, Any]) -> None:
        """
        Dispatch notification to registered callbacks.

        Args:
            callback_type: Type of callback ("signal", "indicator", "confluence").
            data: Parsed payload dict.
        """
        callbacks = self._callbacks.get(callback_type, [])
        for callback in callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(
                    f"SignalListener callback error ({callback_type}): {e}",
                    exc_info=True,
                )
