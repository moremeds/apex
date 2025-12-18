"""
MarketDataManager - Unified interface for multiple market data providers.

Manages connections to multiple market data sources (IBKR, Yahoo Finance, CCXT, etc.)
and provides aggregated market data with fallback mechanisms.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING
from datetime import datetime
from dataclasses import dataclass, field
import asyncio
from threading import Lock

from ...utils.logging_setup import get_logger
from ...domain.interfaces.market_data_provider import MarketDataProvider
from ...models.position import Position
from ...models.market_data import MarketData

if TYPE_CHECKING:
    from ...infrastructure.monitoring import HealthMonitor, HealthStatus
    from ...application.event_bus import EventBusProtocol


logger = get_logger(__name__)


@dataclass
class MarketDataProviderStatus:
    """Status of a market data provider connection."""
    name: str
    connected: bool = False
    last_error: Optional[str] = None
    last_updated: datetime = field(default_factory=datetime.now)
    symbols_fetched: int = 0
    supports_streaming: bool = False
    supports_greeks: bool = False


class MarketDataManager(MarketDataProvider):
    """
    Manages multiple market data providers and aggregates their data.

    Provides:
    - Unified connection management
    - Market data fetching with fallback (primary -> secondary providers)
    - Streaming data aggregation
    - Per-provider status tracking
    - Health monitoring integration

    The manager tries providers in priority order and falls back to
    secondary providers when the primary fails.
    """

    def __init__(
        self,
        health_monitor: Optional["HealthMonitor"] = None,
        event_bus: Optional["EventBusProtocol"] = None,
    ):
        """
        Initialize market data manager with empty provider list.

        Args:
            health_monitor: Optional HealthMonitor for reporting provider health.
            event_bus: Optional EventBus for publishing streaming market data.
        """
        self._providers: Dict[str, MarketDataProvider] = {}
        self._provider_priorities: Dict[str, int] = {}  # name -> priority (lower = higher)
        self._priority: List[str] = []  # Provider names sorted by priority
        self._status: Dict[str, MarketDataProviderStatus] = {}
        self._connected = False
        self._health_monitor = health_monitor
        self._event_bus = event_bus
        self._streaming_callback: Optional[Callable[[str, MarketData], None]] = None
        self._latest_data: Dict[str, MarketData] = {}

        # Thread-safety: protect _latest_data access from callback threads
        self._data_lock = Lock()

        # Store loop reference for cross-thread event bus access
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def register_provider(
        self,
        name: str,
        provider: MarketDataProvider,
        priority: int = 100
    ) -> None:
        """
        Register a market data provider.

        Args:
            name: Unique name for the provider (e.g., "ib", "yahoo", "ccxt").
            provider: The provider implementing MarketDataProvider.
            priority: Lower number = higher priority (default 100).
        """
        self._providers[name] = provider
        self._provider_priorities[name] = priority
        self._status[name] = MarketDataProviderStatus(
            name=name,
            supports_streaming=provider.supports_streaming(),
            supports_greeks=provider.supports_greeks(),
        )

        # Update priority list and sort by priority (lower = higher priority)
        self._priority.append(name)
        self._priority.sort(key=lambda n: self._provider_priorities.get(n, 100))

        logger.info(
            f"Registered market data provider: {name} "
            f"(priority={priority}, streaming={provider.supports_streaming()}, "
            f"greeks={provider.supports_greeks()})"
        )

    def get_provider(self, name: str) -> Optional[MarketDataProvider]:
        """Get a specific provider by name."""
        return self._providers.get(name)

    def get_status(self, name: str) -> Optional[MarketDataProviderStatus]:
        """Get status for a specific provider."""
        return self._status.get(name)

    def get_all_status(self) -> Dict[str, MarketDataProviderStatus]:
        """Get status for all providers."""
        return self._status.copy()

    async def connect(self) -> None:
        """
        Connect to all registered market data providers.

        Attempts to connect to each provider independently.
        Failures are logged but don't prevent other providers from connecting.
        """
        # Capture the event loop for cross-thread operations
        self._loop = asyncio.get_running_loop()

        if not self._providers:
            logger.warning("No market data providers registered")
            return

        connect_tasks = []
        for name, provider in self._providers.items():
            connect_tasks.append(self._connect_provider(name, provider))

        await asyncio.gather(*connect_tasks)

        # Set overall connected status if at least one provider connected
        self._connected = any(s.connected for s in self._status.values())

        connected_count = sum(1 for s in self._status.values() if s.connected)
        logger.info(
            f"MarketDataManager connected: {connected_count}/{len(self._providers)} providers"
        )

    async def _connect_provider(self, name: str, provider: MarketDataProvider) -> None:
        """Connect a single provider with error handling."""
        try:
            # Check if provider is already connected (e.g., shared with BrokerManager)
            if provider.is_connected():
                logger.info(f"✓ Market data provider {name} already connected")
                self._status[name].connected = True
                self._status[name].last_error = None
                self._status[name].last_updated = datetime.now()
                self._update_health(name, "HEALTHY", "Connected (shared)")

                # Always wire provider streaming → MarketDataManager._on_provider_data.
                # _on_provider_data publishes to EventBus (and optionally forwards to _streaming_callback).
                if provider.supports_streaming():
                    provider.set_streaming_callback(
                        lambda sym, md, n=name: self._on_provider_data(n, sym, md)
                    )
                return

            # Not connected yet, try to connect
            await provider.connect()
            self._status[name].connected = provider.is_connected()
            self._status[name].last_error = None
            self._status[name].last_updated = datetime.now()

            if self._status[name].connected:
                logger.info(f"✓ Connected to market data provider: {name}")
                self._update_health(name, "HEALTHY", "Connected")

                # Always wire provider streaming → MarketDataManager._on_provider_data.
                # _on_provider_data publishes to EventBus (and optionally forwards to _streaming_callback).
                if provider.supports_streaming():
                    provider.set_streaming_callback(
                        lambda sym, md, n=name: self._on_provider_data(n, sym, md)
                    )
            else:
                logger.warning(f"⚠ Provider {name} connected but reports not connected")
                self._update_health(name, "DEGRADED", "Connection state mismatch")

        except Exception as e:
            self._status[name].connected = False
            self._status[name].last_error = str(e)
            self._status[name].last_updated = datetime.now()
            logger.error(f"✗ Failed to connect to market data provider {name}: {e}")
            self._update_health(name, "UNHEALTHY", f"Connection failed: {str(e)[:50]}")

    async def disconnect(self) -> None:
        """Disconnect from all market data providers."""
        for name, provider in self._providers.items():
            try:
                await provider.disconnect()
                self._status[name].connected = False
                logger.info(f"Disconnected from market data provider: {name}")
            except Exception as e:
                logger.error(f"Error disconnecting from {name}: {e}")

        self._connected = False
        logger.info("MarketDataManager disconnected from all providers")

    def is_connected(self) -> bool:
        """Check if at least one provider is connected."""
        return self._connected

    async def fetch_market_data(self, positions: List[Position]) -> List[MarketData]:
        """
        Fetch market data for positions from all connected providers.

        Tries providers in priority order. For each position, uses data from
        the highest-priority provider that has it.

        Args:
            positions: List of positions to fetch market data for.

        Returns:
            Aggregated list of MarketData from all providers.
        """
        if not positions:
            return []

        # Fast path: single connected provider - skip all iteration overhead
        connected_providers = [
            (name, self._providers[name])
            for name in self._priority
            if self._status[name].connected
        ]

        if len(connected_providers) == 1:
            name, provider = connected_providers[0]
            try:
                market_data_list = await provider.fetch_market_data(positions)
                # Update cache in bulk
                for md in market_data_list:
                    self._latest_data[md.symbol] = md
                self._status[name].symbols_fetched += len(market_data_list)
                self._status[name].last_updated = datetime.now()
                self._status[name].last_error = None
                return market_data_list
            except Exception as e:
                self._status[name].last_error = str(e)
                self._status[name].last_updated = datetime.now()
                logger.error(f"Failed to fetch market data from {name}: {e}")
                return []

        # Multi-provider path: track which symbols we still need
        symbols_needed = {p.symbol for p in positions}
        symbol_to_position = {p.symbol: p for p in positions}
        result: Dict[str, MarketData] = {}

        for name, provider in connected_providers:
            # Get positions we still need
            positions_to_fetch = [
                symbol_to_position[sym]
                for sym in symbols_needed
                if sym in symbol_to_position
            ]

            if not positions_to_fetch:
                break  # Got all data

            try:
                market_data_list = await provider.fetch_market_data(positions_to_fetch)

                for md in market_data_list:
                    if md.symbol in symbols_needed:
                        result[md.symbol] = md
                        symbols_needed.discard(md.symbol)
                        self._latest_data[md.symbol] = md

                self._status[name].symbols_fetched += len(market_data_list)
                self._status[name].last_updated = datetime.now()
                self._status[name].last_error = None

                logger.debug(
                    f"Fetched {len(market_data_list)} symbols from {name}, "
                    f"{len(symbols_needed)} still needed"
                )

            except Exception as e:
                self._status[name].last_error = str(e)
                self._status[name].last_updated = datetime.now()
                logger.error(f"Failed to fetch market data from {name}: {e}")

        if symbols_needed:
            logger.warning(
                f"No market data available for {len(symbols_needed)} symbols: "
                f"{list(symbols_needed)[:5]}..."
            )

        return list(result.values())

    async def fetch_quotes(self, symbols: List[str]) -> Dict[str, MarketData]:
        """
        Fetch quotes for symbols (without position context).

        Useful for market indicators like VIX that aren't positions.

        Args:
            symbols: List of symbols to fetch quotes for.

        Returns:
            Dict mapping symbol to MarketData.
        """
        if not symbols:
            return {}

        symbols_needed = set(symbols)
        result: Dict[str, MarketData] = {}

        # Try each provider in priority order
        for name in self._priority:
            if not self._status[name].connected:
                continue

            provider = self._providers[name]

            try:
                quotes = await provider.fetch_quotes(list(symbols_needed))

                for symbol, md in quotes.items():
                    if symbol in symbols_needed:
                        result[symbol] = md
                        symbols_needed.discard(symbol)
                        self._latest_data[symbol] = md

                if not symbols_needed:
                    break  # Got all data

            except NotImplementedError:
                logger.debug(f"{name} does not support fetch_quotes")
            except Exception as e:
                logger.error(f"Failed to fetch quotes from {name}: {e}")

        return result

    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to real-time updates for symbols via streaming providers."""
        for name in self._priority:
            if not self._status[name].connected:
                continue

            provider = self._providers[name]
            if not provider.supports_streaming():
                continue

            try:
                await provider.subscribe(symbols)
                logger.info(f"Subscribed to {len(symbols)} symbols via {name}")
                return  # Only subscribe via one streaming provider
            except Exception as e:
                logger.error(f"Failed to subscribe via {name}: {e}")

    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from real-time updates."""
        for name, provider in self._providers.items():
            if provider.supports_streaming() and self._status[name].connected:
                try:
                    await provider.unsubscribe(symbols)
                except Exception as e:
                    logger.error(f"Failed to unsubscribe via {name}: {e}")

    def get_latest(self, symbol: str) -> Optional[MarketData]:
        """Get latest cached market data for a symbol (thread-safe)."""
        # First check our aggregated cache (thread-safe read)
        with self._data_lock:
            if symbol in self._latest_data:
                return self._latest_data[symbol]

        # Then check each provider
        for name in self._priority:
            if not self._status[name].connected:
                continue

            provider = self._providers[name]
            md = provider.get_latest(symbol)
            if md:
                with self._data_lock:
                    self._latest_data[symbol] = md
                return md

        return None

    def set_streaming_callback(
        self,
        callback: Optional[Callable[[str, MarketData], None]]
    ) -> None:
        """
        Set callback for streaming market data updates.

        Args:
            callback: Function to call with (symbol, market_data) on updates.
        """
        self._streaming_callback = callback

        # Provider callbacks should always feed into _on_provider_data so EventBus publishing
        # keeps working even when the optional external callback is unset.
        for name, provider in self._providers.items():
            if provider.supports_streaming() and self._status[name].connected:
                provider.set_streaming_callback(
                    lambda sym, md, n=name: self._on_provider_data(n, sym, md)
                )

    def enable_streaming(self) -> None:
        """Enable streaming mode on all providers that support it."""
        for name, provider in self._providers.items():
            if provider.supports_streaming() and self._status[name].connected:
                provider.enable_streaming()
                logger.info(f"Enabled streaming on {name}")

    def disable_streaming(self) -> None:
        """Disable streaming mode on all providers."""
        for name, provider in self._providers.items():
            if provider.supports_streaming():
                provider.disable_streaming()

    def supports_streaming(self) -> bool:
        """Check if any provider supports streaming."""
        return any(
            p.supports_streaming() and self._status[n].connected
            for n, p in self._providers.items()
        )

    def supports_greeks(self) -> bool:
        """Check if any provider supports Greeks."""
        return any(
            p.supports_greeks() and self._status[n].connected
            for n, p in self._providers.items()
        )

    def _on_provider_data(self, provider_name: str, symbol: str, md: MarketData) -> None:
        """
        Handle streaming data from a provider (called from callback threads).

        This is the SINGLE path for streaming market data (Phase 4 optimization):
        IB → MarketDataManager → EventBus (skip Orchestrator hop)

        Thread-safety: This method is called from provider callback threads.
        We use a lock for _latest_data and rely on the thread-safe event bus.
        """
        # Thread-safe cache update
        with self._data_lock:
            self._latest_data[symbol] = md

        # Publish to EventBus (thread-safe after P0-001/P0-002 fixes)
        if self._event_bus:
            # Lazy import to avoid circular dependency
            from ...domain.interfaces.event_bus import EventType
            self._event_bus.publish(EventType.MARKET_DATA_TICK, {
                "symbol": symbol,
                "data": md,
                "source": provider_name,
                "timestamp": datetime.now(),
            })

        # Forward to callback (for stores/orchestrator)
        # Note: Callback should be quick to avoid blocking provider thread
        if self._streaming_callback:
            self._streaming_callback(symbol, md)

    def _update_health(
        self, provider_name: str, status: str, message: str, metadata: Optional[Dict] = None
    ) -> None:
        """
        Update health status for a provider in the health monitor.

        Args:
            provider_name: Name of the provider (e.g., "ib", "yahoo").
            status: Health status string ("HEALTHY", "DEGRADED", "UNHEALTHY").
            message: Status message.
            metadata: Optional additional metadata.
        """
        if self._health_monitor is None:
            return

        from ..monitoring import HealthStatus

        # Map string to enum
        status_map = {
            "HEALTHY": HealthStatus.HEALTHY,
            "DEGRADED": HealthStatus.DEGRADED,
            "UNHEALTHY": HealthStatus.UNHEALTHY,
            "UNKNOWN": HealthStatus.UNKNOWN,
        }
        health_status = status_map.get(status, HealthStatus.UNKNOWN)

        # Use provider-specific component name
        component_name = f"{provider_name}_market_data"

        # Include provider status in metadata
        provider_status = self._status.get(provider_name)
        full_metadata = metadata or {}
        if provider_status:
            full_metadata.update({
                "connected": provider_status.connected,
                "symbols_fetched": provider_status.symbols_fetched,
                "last_error": provider_status.last_error,
                "supports_streaming": provider_status.supports_streaming,
                "supports_greeks": provider_status.supports_greeks,
            })

        self._health_monitor.update_component_health(
            component_name=component_name,
            status=health_status,
            message=message,
            metadata=full_metadata,
        )

    def set_health_monitor(self, health_monitor: "HealthMonitor") -> None:
        """
        Set the health monitor after initialization.

        Args:
            health_monitor: HealthMonitor instance to use.
        """
        self._health_monitor = health_monitor

    async def check_all_health(self) -> Dict[str, MarketDataProviderStatus]:
        """
        Check health of all providers and update health monitor.

        Returns:
            Dict mapping provider name to current status.
        """
        for name, provider in self._providers.items():
            try:
                is_connected = provider.is_connected()
                self._status[name].connected = is_connected

                if is_connected:
                    self._update_health(name, "HEALTHY", "Connected and operational")
                else:
                    self._update_health(name, "UNHEALTHY", "Not connected")
                    self._status[name].last_error = "Connection lost"

            except Exception as e:
                self._status[name].connected = False
                self._status[name].last_error = str(e)
                self._update_health(name, "UNHEALTHY", f"Health check failed: {str(e)[:50]}")

        return self._status.copy()

    def get_providers_with_greeks(self) -> List[str]:
        """Get list of connected provider names that support Greeks."""
        return [
            name for name, provider in self._providers.items()
            if provider.supports_greeks() and self._status[name].connected
        ]

    def get_streaming_providers(self) -> List[str]:
        """Get list of connected provider names that support streaming."""
        return [
            name for name, provider in self._providers.items()
            if provider.supports_streaming() and self._status[name].connected
        ]
