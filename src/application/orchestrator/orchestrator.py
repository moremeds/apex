"""
Orchestrator - Main application coordinator.

Thin coordination layer that:
- Manages lifecycle (start/stop/connect)
- Wires events between components
- Delegates data fetching to DataCoordinator
- Delegates snapshot building to SnapshotCoordinator
"""

from __future__ import annotations
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, TYPE_CHECKING

from ...utils.logging_setup import get_logger
from ...utils.trace_context import new_cycle
from ...utils.market_hours import MarketHours
from ...domain.interfaces.event_bus import EventBus
from ...domain.events import PriorityEventBus
from ...domain.exceptions import RecoverableError, FatalError
from ...models.risk_snapshot import RiskSnapshot
from ...models.risk_signal import RiskSignal
from ...models.account import AccountInfo
from ...models.position import Position
from ...infrastructure.monitoring import HealthStatus

from ..async_event_bus import AsyncEventBus
from .data_coordinator import DataCoordinator
from .snapshot_coordinator import SnapshotCoordinator
from .signal_coordinator import SignalCoordinator

if TYPE_CHECKING:
    from ...domain.services.risk.risk_engine import RiskEngine
    from ...domain.services.pos_reconciler import Reconciler
    from ...domain.services.mdqc import MDQC
    from ...domain.services.risk.rule_engine import RuleEngine
    from ...domain.services.market_alert_detector import MarketAlertDetector
    from ...domain.services.risk.risk_signal_engine import RiskSignalEngine
    from ...domain.services.risk.risk_alert_logger import RiskAlertLogger
    from ...domain.interfaces.signal_persistence import SignalPersistencePort
    from ...infrastructure.stores import PositionStore, MarketDataStore, AccountStore
    from ...infrastructure.monitoring import HealthMonitor, Watchdog
    from ...infrastructure.adapters.broker_manager import BrokerManager
    from ...infrastructure.adapters.market_data_manager import MarketDataManager
    from ...infrastructure.observability import RiskMetrics, HealthMetrics, SignalMetrics
    from ..readiness_manager import ReadinessManager
    from ...services.snapshot_service import SnapshotService
    from ...services.warm_start_service import WarmStartService

logger = get_logger(__name__)


class Orchestrator:
    """
    Main application orchestrator - thin coordination layer.

    Delegates heavy lifting to:
    - DataCoordinator: position/market data fetching
    - SnapshotCoordinator: risk snapshot building/dispatching
    """

    def __init__(
        self,
        broker_manager: BrokerManager,
        market_data_manager: MarketDataManager,
        position_store: PositionStore,
        market_data_store: MarketDataStore,
        account_store: AccountStore,
        risk_engine: RiskEngine,
        reconciler: Reconciler,
        mdqc: MDQC,
        rule_engine: RuleEngine,
        health_monitor: HealthMonitor,
        watchdog: Watchdog,
        event_bus: Union[EventBus, PriorityEventBus],
        config: Dict[str, Any],
        market_alert_detector: Optional[MarketAlertDetector] = None,
        risk_signal_engine: Optional[RiskSignalEngine] = None,
        risk_alert_logger: Optional[RiskAlertLogger] = None,
        risk_metrics: Optional[RiskMetrics] = None,
        health_metrics: Optional[HealthMetrics] = None,
        readiness_manager: Optional[ReadinessManager] = None,
        snapshot_service: Optional[SnapshotService] = None,
        warm_start_service: Optional[WarmStartService] = None,
        signal_metrics: Optional["SignalMetrics"] = None,
        historical_data_manager: Optional[Any] = None,
        signal_persistence: Optional["SignalPersistencePort"] = None,
    ):
        # Core dependencies
        self.broker_manager = broker_manager
        self.market_data_manager = market_data_manager
        self.position_store = position_store
        self.market_data_store = market_data_store
        self.account_store = account_store
        self.risk_engine = risk_engine
        self.rule_engine = rule_engine
        self.health_monitor = health_monitor
        self.watchdog = watchdog
        self.event_bus = event_bus
        self.config = config

        # Optional dependencies
        self.market_alert_detector = market_alert_detector
        self._risk_metrics = risk_metrics
        self._health_metrics = health_metrics
        self._readiness_manager = readiness_manager
        self._snapshot_service = snapshot_service
        self._warm_start_service = warm_start_service
        self._historical_data_manager = historical_data_manager

        # Create coordinators
        self._data_coordinator = DataCoordinator(
            broker_manager=broker_manager,
            market_data_manager=market_data_manager,
            position_store=position_store,
            market_data_store=market_data_store,
            account_store=account_store,
            reconciler=reconciler,
            mdqc=mdqc,
            health_monitor=health_monitor,
            config=config,
            event_bus=event_bus,  # Enable slow-lane MDQC validation
        )

        self._snapshot_coordinator = SnapshotCoordinator(
            risk_engine=risk_engine,
            rule_engine=rule_engine,
            position_store=position_store,
            market_data_store=market_data_store,
            account_store=account_store,
            event_bus=event_bus,
            health_monitor=health_monitor,
            config=config,
            risk_signal_engine=risk_signal_engine,
            risk_alert_logger=risk_alert_logger,
            risk_metrics=risk_metrics,
        )

        # Signal pipeline coordinator (bar aggregation → indicators → signals)
        # With optional persistence for database storage and PostgreSQL NOTIFY
        signals_config = config.get("signals", {})
        self._signal_coordinator = SignalCoordinator(
            event_bus=event_bus,
            timeframes=signals_config.get("timeframes"),
            max_workers=signals_config.get("indicator_max_workers", 4),
            enabled=signals_config.get("enabled", True),
            signal_metrics=signal_metrics,
            historical_data_manager=historical_data_manager,
            preload_config=signals_config.get("preload", {}),
            persistence=signal_persistence,
            exclude_options=signals_config.get("exclude_options", True),
        )

        # State
        self._running = False
        self._timer_task: Optional[asyncio.Task] = None
        self._latest_market_alerts: List[Dict[str, Any]] = []
        self._tick_count: int = 0  # For periodic housekeeping
        self._consecutive_errors: int = 0  # Track timer loop error frequency
        self._bar_preload_pending: bool = True  # Defer bar preload to first timer tick

        # Timer configuration
        self._timer_interval_sec = config.get("timer_interval_sec", 5.0)

    async def start(self) -> None:
        """Start the orchestrator."""
        if self._running:
            logger.warning("Orchestrator already running")
            return

        logger.info("Starting orchestrator...")

        # Start event bus FIRST - must be running before any streaming/callbacks
        # that might publish events (C1: event bus start order fix)
        # Use duck typing: check for start() method rather than concrete types
        if hasattr(self.event_bus, 'start'):
            await self.event_bus.start()
            logger.debug("Event bus started before providers")

        # Subscribe components to events (before streaming starts)
        self._subscribe_components_to_events()

        # Start signal pipeline (bar aggregation → indicators → signals)
        self._signal_coordinator.start()

        # Warm-start: Load state from snapshots
        await self._perform_warm_start()

        # Connect to data sources (streaming callbacks now have event bus ready)
        await self._connect_providers()

        # Start watchdog
        await self.watchdog.start()

        # Start snapshot service
        if self._snapshot_service:
            await self._snapshot_service.start()

        self._running = True

        # Start coordinators
        await self._snapshot_coordinator.start()

        # Start timer loop
        self._timer_task = asyncio.create_task(self._timer_loop())

        # Initial data fetch (loads positions from brokers)
        await self._data_coordinator.fetch_and_reconcile(
            on_dirty_callback=self.risk_engine.mark_dirty
        )

        # NOTE: Bar preload is deferred to first timer tick to ensure TUI is subscribed
        # to CONFLUENCE_UPDATE and ALIGNMENT_UPDATE events before signals are generated.
        # See _timer_loop() for the actual preload trigger.

        logger.info("Orchestrator started")

    async def stop(self) -> None:
        """Stop the orchestrator."""
        logger.info("Stopping orchestrator...")
        self._running = False

        # Stop coordinators
        await self._data_coordinator.cleanup()
        await self._snapshot_coordinator.stop()
        self._signal_coordinator.stop()

        # Cancel timer task
        if self._timer_task:
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass

        # Signal shutdown to ReadinessManager
        if self._readiness_manager:
            self._readiness_manager.shutdown()

        # Stop snapshot service
        if self._snapshot_service:
            await self._snapshot_service.stop()

        # Stop event bus (duck typing - check for stop() method)
        if hasattr(self.event_bus, 'stop'):
            await self.event_bus.stop()

        # Stop watchdog
        await self.watchdog.stop()

        # Disconnect brokers
        await self.broker_manager.disconnect()

        # Close risk engine
        self.risk_engine.close()

        logger.info("Orchestrator stopped")

    async def _connect_providers(self) -> None:
        """Connect to all data providers."""
        # Connect broker manager
        await self.broker_manager.connect()

        # Connect market data manager
        await self.market_data_manager.connect()
        # Enable provider-side streaming callbacks (e.g., IB pendingTickersEvent).
        # Actual subscriptions are created when MarketDataManager.fetch_market_data is called.
        self.market_data_manager.enable_streaming()

        # Subscribe to position updates
        self.broker_manager.set_position_callback(self._on_broker_position_update)
        await self.broker_manager.subscribe_positions()

    def _subscribe_components_to_events(self) -> None:
        """Wire up event subscriptions."""
        # Stores + engines listen to event bus (event-driven mode).
        self.market_data_store.subscribe_to_events(self.event_bus)
        self.risk_engine.subscribe_to_events(self.event_bus)

    async def _timer_loop(self) -> None:
        """Timer loop for periodic data refresh."""
        while self._running:
            try:
                await asyncio.sleep(self._timer_interval_sec)

                if not self._running:
                    break

                new_cycle()  # New trace ID for this cycle

                # Deferred bar preload on first tick (ensures TUI is subscribed to events)
                if self._bar_preload_pending:
                    self._bar_preload_pending = False
                    await self._preload_signal_bars()

                # Fetch and reconcile data
                await self._data_coordinator.fetch_and_reconcile(
                    on_dirty_callback=self.risk_engine.mark_dirty
                )

                # Detect market alerts
                await self._detect_market_alerts()

                # Periodic housekeeping (every ~5 minutes with 5s interval)
                self._tick_count += 1
                if self._tick_count % 60 == 0:
                    self._periodic_cleanup()

                # Reset consecutive error counter on successful iteration
                self._consecutive_errors = 0

            except asyncio.CancelledError:
                break
            except RecoverableError as e:
                logger.warning(f"Recoverable error in timer loop: {e}")
                # Continue loop
            except FatalError as e:
                logger.critical(f"Fatal error in timer loop: {e}", exc_info=True)
                self._running = False
                break
            except (ConnectionError, TimeoutError, OSError) as e:
                # Transient network/IO errors - log and continue
                logger.warning(f"Transient timer loop error (will retry): {e}")
            except Exception as e:
                # Log unexpected errors with full traceback for debugging
                logger.error(f"Unexpected timer loop error: {e}", exc_info=True)
                # Increment error counter for monitoring (if available)
                self._consecutive_errors = getattr(self, '_consecutive_errors', 0) + 1
                if self._consecutive_errors >= 5:
                    logger.critical("Too many consecutive errors in timer loop")
                    # Don't break - let the loop continue but alert is logged

    def _on_broker_position_update(
        self, broker_name: str, positions: List[Position]
    ) -> None:
        """Handle position update from broker."""
        logger.debug(f"Position update from {broker_name}: {len(positions)} positions")
        self.position_store.upsert_positions(positions)
        self.risk_engine.mark_dirty()

    async def _detect_market_alerts(self) -> None:
        """Detect market-wide alerts (VIX regime, etc.)."""
        if not self.market_alert_detector:
            return

        try:
            # Get market indicators from store
            market_data_store = self._data_coordinator.market_data_store
            market_data = {}

            # Get VIX if available
            vix_md = market_data_store.get("VIX")
            if vix_md:
                market_data["vix"] = vix_md.effective_mid() or vix_md.last
                market_data["vix_prev_close"] = vix_md.yesterday_close

            # Get SPY/QQQ for market drop detection
            spy_md = market_data_store.get("SPY")
            if spy_md and spy_md.yesterday_close:
                current = spy_md.effective_mid() or spy_md.last
                if current:
                    market_data["spy_change_pct"] = ((current / spy_md.yesterday_close) - 1) * 100

            qqq_md = market_data_store.get("QQQ")
            if qqq_md and qqq_md.yesterday_close:
                current = qqq_md.effective_mid() or qqq_md.last
                if current:
                    market_data["qqq_change_pct"] = ((current / qqq_md.yesterday_close) - 1) * 100

            alerts = self.market_alert_detector.detect_alerts(market_data)
            self._latest_market_alerts = alerts
        except Exception as e:
            logger.error(f"Market alert detection error: {e}")

    def _periodic_cleanup(self) -> None:
        """Periodic housekeeping tasks (called every ~5 minutes)."""
        # Clean up expired cooldowns in RiskSignalManager to prevent memory leak
        try:
            risk_signal_engine = self._snapshot_coordinator.risk_signal_engine
            if risk_signal_engine and hasattr(risk_signal_engine, 'signal_manager'):
                risk_signal_engine.signal_manager.cleanup_expired()
                logger.debug("RiskSignalManager cleanup completed")
        except Exception as e:
            logger.warning(f"Periodic cleanup error: {e}")

        # Clean up expired signal cooldowns in SignalCoordinator
        try:
            cleared = self._signal_coordinator.clear_cooldowns()
            if cleared > 0:
                logger.debug(f"SignalCoordinator cleared {cleared} expired cooldowns")
        except Exception as e:
            logger.warning(f"Signal cooldown cleanup error: {e}")

    async def _perform_warm_start(self) -> None:
        """Load state from previous snapshots for warm start."""
        if not self._warm_start_service:
            return

        try:
            await self._warm_start_service.restore()
        except Exception as e:
            logger.warning(f"Warm start failed: {e}")

    async def _preload_signal_bars(self) -> None:
        """
        Preload historical bars from Parquet cache for signal warmup.

        Called during startup BEFORE live ticks arrive to ensure indicators
        have sufficient warmup data (e.g., 201 bars for SMA 200).

        This method:
        1. Gets symbols from position store
        2. Calls SignalCoordinator.preload_bar_history() which:
           - Downloads missing bars from IB/Yahoo (gap backfill)
           - Stores to Parquet cache (persistence)
           - Injects into IndicatorEngine (warmup)
        """
        if not self._signal_coordinator.is_started:
            logger.debug("Signal coordinator not started, skipping bar preload")
            return

        if not self._historical_data_manager:
            logger.debug("No historical data manager configured, skipping bar preload")
            return

        # Get symbols from positions (use underlying for options, symbol for stocks)
        positions = self.position_store.get_all()
        symbols = list({
            p.underlying or p.symbol
            for p in positions
            if p.underlying or p.symbol
        })

        if not symbols:
            logger.debug("No symbols to preload (empty position store)")
            return

        logger.info(
            "Initiating bar cache preload for signal pipeline",
            extra={
                "symbols_count": len(symbols),
                "symbols": symbols[:10],  # Log first 10 for brevity
            },
        )

        try:
            results = await self._signal_coordinator.preload_bar_history(symbols)
            logger.info(
                "Bar cache preload completed",
                extra={
                    "symbols_loaded": len(results),
                    "total_bars_injected": sum(results.values()),
                },
            )
        except Exception as e:
            # Don't crash startup on preload failure
            logger.error(f"Bar cache preload failed: {e}", exc_info=True)

    # Public accessors (delegate to coordinators)

    def get_latest_snapshot(self) -> Optional[RiskSnapshot]:
        """Get the latest risk snapshot."""
        return self._snapshot_coordinator.get_latest_snapshot()

    async def wait_for_snapshot(self, timeout: float = 5.0) -> Optional[RiskSnapshot]:
        """Wait for a new snapshot."""
        return await self._snapshot_coordinator.wait_for_snapshot(timeout)

    def get_latest_market_alerts(self) -> List[Dict[str, Any]]:
        """Get the latest market alerts."""
        return self._latest_market_alerts

    def get_latest_risk_signals(self) -> List[RiskSignal]:
        """Get the latest risk signals."""
        return self._snapshot_coordinator.get_latest_risk_signals()

    def get_event_bus_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        if hasattr(self.event_bus, 'get_stats'):
            return self.event_bus.get_stats()
        return {}

    def get_readiness_snapshot(self) -> Optional[Any]:
        """Get readiness manager snapshot."""
        if self._readiness_manager:
            return self._readiness_manager.get_snapshot()
        return None

    def get_positions_preview(self) -> Optional[RiskSnapshot]:
        """
        Get a preview snapshot with position count before full market data is available.

        Returns a minimal RiskSnapshot showing positions loaded, useful for
        early dashboard display while waiting for market data.
        """
        positions = self._data_coordinator.position_store.get_all()
        if not positions:
            return None

        # Create minimal preview snapshot
        return RiskSnapshot(
            total_positions=len(positions),
            position_risks=[],  # No risk calcs yet
            portfolio_delta=0.0,
            portfolio_gamma=0.0,
            portfolio_vega=0.0,
            portfolio_theta=0.0,
        )

    @property
    def signal_coordinator(self) -> SignalCoordinator:
        """Get the signal coordinator for confluence callback wiring."""
        return self._signal_coordinator
