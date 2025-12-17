"""
Orchestrator - Main application workflow control (Event-Driven Architecture).

Coordinates:
- Position/market data fetching via TIMER_TICK events
- Position reconciliation
- Risk calculation triggered by data events
- Limit breach detection
- Dashboard updates via SNAPSHOT_READY events
- Event publishing

The orchestrator operates in a fully event-driven mode:
- TIMER_TICK: Triggers position/market data refresh and reconciliation
- MARKET_DATA_TICK/BATCH: Updates market data store, marks risk engine dirty
- POSITION_UPDATED/BATCH: Updates position store, marks risk engine dirty
- Snapshot dispatcher: Debounced snapshot rebuilds when risk engine is dirty
"""

from __future__ import annotations
import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, TYPE_CHECKING, Union

from ..utils.logging_setup import get_logger, is_console_enabled
from ..utils.trace_context import new_cycle, get_cycle_id
from ..utils.perf_logger import log_timing_async, log_timing
from ..utils.market_hours import MarketHours
from ..domain.interfaces.event_bus import EventBus, EventType
from ..domain.events import PriorityEventBus
from ..domain.services.risk.risk_engine import RiskEngine
from ..domain.services.pos_reconciler import Reconciler
from ..domain.services.mdqc import MDQC
from ..domain.services.risk.rule_engine import RuleEngine
from ..domain.services.market_alert_detector import MarketAlertDetector
from ..domain.services.risk.risk_signal_engine import RiskSignalEngine
from ..domain.services.risk.risk_alert_logger import RiskAlertLogger
from ..models.risk_snapshot import RiskSnapshot
from ..models.account import AccountInfo
from ..models.position import Position
from ..models.risk_signal import RiskSignal
from ..infrastructure.stores import PositionStore, MarketDataStore, AccountStore
from ..infrastructure.monitoring import HealthMonitor, HealthStatus, Watchdog
from ..infrastructure.adapters.broker_manager import BrokerManager
from ..infrastructure.adapters.market_data_manager import MarketDataManager
from .async_event_bus import AsyncEventBus

if TYPE_CHECKING:
    from ..infrastructure.observability import RiskMetrics, HealthMetrics
    from .readiness_manager import ReadinessManager
    from ..services.snapshot_service import SnapshotService
    from ..services.warm_start_service import WarmStartService


logger = get_logger(__name__)


class Orchestrator:
    """
    Main application orchestrator (Event-Driven Architecture).

    Coordinates the entire risk monitoring workflow through events:
    1. TIMER_TICK triggers position/account fetching from all sources
    2. TIMER_TICK triggers market data refresh for stale symbols
    3. Data events (POSITION_*, MARKET_DATA_*, ACCOUNT_*) update stores
    4. Snapshot dispatcher rebuilds risk metrics when data changes
    5. SNAPSHOT_READY triggers dashboard updates
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
        market_alert_detector: MarketAlertDetector | None = None,
        risk_signal_engine: RiskSignalEngine | None = None,
        risk_alert_logger: RiskAlertLogger | None = None,
        risk_metrics: Optional["RiskMetrics"] = None,
        health_metrics: Optional["HealthMetrics"] = None,
        readiness_manager: Optional["ReadinessManager"] = None,
        snapshot_service: Optional["SnapshotService"] = None,
        warm_start_service: Optional["WarmStartService"] = None,
    ):
        """
        Initialize orchestrator with all dependencies.

        Args:
            broker_manager: BrokerManager handling all broker connections.
            market_data_manager: MarketDataManager handling all market data sources.
            position_store: Position store.
            market_data_store: Market data store.
            account_store: Account store.
            risk_engine: Risk calculation engine.
            reconciler: Position reconciliation service.
            mdqc: Market data quality control.
            rule_engine: Risk limit evaluation (legacy).
            health_monitor: Health monitoring service.
            watchdog: Watchdog monitoring service.
            event_bus: Event bus for publish-subscribe.
            config: Application configuration.
            market_alert_detector: Optional market alert detector.
            risk_signal_engine: Multi-layer risk signal engine (optional, replaces rule_engine).
            risk_alert_logger: Optional risk alert logger for audit trail.
            risk_metrics: Optional RiskMetrics for Prometheus observability.
            health_metrics: Optional HealthMetrics for Prometheus observability.
            readiness_manager: Optional ReadinessManager for event-driven readiness gating.
            snapshot_service: Optional SnapshotService for periodic state snapshots.
            warm_start_service: Optional WarmStartService for loading state on startup.
        """
        self.broker_manager = broker_manager
        self.market_data_manager = market_data_manager
        self.position_store = position_store
        self.market_data_store = market_data_store
        self.account_store = account_store
        self.risk_engine = risk_engine
        self.reconciler = reconciler
        self.mdqc = mdqc
        self.rule_engine = rule_engine
        self.health_monitor = health_monitor
        self.watchdog = watchdog
        self.event_bus = event_bus
        self.config = config
        self.market_alert_detector = market_alert_detector
        self.risk_signal_engine = risk_signal_engine
        self.risk_alert_logger = risk_alert_logger
        self._risk_metrics = risk_metrics
        self._health_metrics = health_metrics
        self._readiness_manager = readiness_manager
        self._snapshot_service = snapshot_service
        self._warm_start_service = warm_start_service

        dashboard_cfg = config.get("dashboard", {})
        self.refresh_interval_sec = dashboard_cfg.get("refresh_interval_sec", 2)
        self._running = False

        self._latest_snapshot: RiskSnapshot | None = None
        self._latest_market_alerts: list[dict[str, Any]] = []
        self._latest_risk_signals: List[RiskSignal] = []
        self._last_reconciliation_issue_count: int = 0  # Track to log only on change

        # Event to signal when new snapshot is ready (for dashboard sync)
        self._snapshot_ready: asyncio.Event = asyncio.Event()

        # Event-driven tasks
        self._timer_task: asyncio.Task | None = None
        self._snapshot_dispatcher_task: asyncio.Task | None = None

        # Account TTL caching (account info changes slowly - fetch every 30s, not every tick)
        self._account_ttl_sec: float = config.get("account_ttl_sec", 30.0)
        self._last_account_fetch: Optional[datetime] = None
        self._cached_account_info: Optional[AccountInfo] = None

        # Snapshot readiness gating config (Phase 3 optimization)
        self._snapshot_min_interval_sec: float = dashboard_cfg.get("snapshot_min_interval_sec", 1.0)
        self._snapshot_ready_ratio: float = dashboard_cfg.get("snapshot_ready_ratio", 0.9)
        self._snapshot_ready_timeout_sec: float = dashboard_cfg.get("snapshot_ready_timeout_sec", 30.0)
        self._snapshot_readiness_achieved: bool = False
        self._snapshot_startup_time: Optional[datetime] = None

    async def start(self) -> None:
        """Start the orchestrator (event-driven mode)."""
        if self._running:
            logger.warning("Orchestrator already running")
            return

        logger.info("Starting orchestrator (event-driven mode)...")

        # Warm-start: Load state from snapshots before connecting to brokers
        await self._perform_warm_start()

        # Connect to data sources
        await self._connect_providers()

        # Start watchdog
        await self.watchdog.start()

        # Start snapshot service for periodic state capture
        if self._snapshot_service:
            await self._snapshot_service.start()
            logger.info("Snapshot service started")

        self._running = True

        # Start async event bus (works for both AsyncEventBus and PriorityEventBus)
        if isinstance(self.event_bus, (AsyncEventBus, PriorityEventBus)):
            await self.event_bus.start()
            bus_type = "PriorityEventBus" if isinstance(self.event_bus, PriorityEventBus) else "AsyncEventBus"
            logger.info(f"{bus_type} started")

        # Subscribe all components to events
        self._subscribe_components_to_events()

        # Subscribe to position updates (Phase 6 - event-driven position changes)
        await self._subscribe_to_position_updates()

        # Start snapshot dispatcher (debounced rebuilds)
        self._snapshot_dispatcher_task = asyncio.create_task(self._snapshot_dispatcher())

        # Start timer task for periodic data refresh (includes initial fetch)
        self._timer_task = asyncio.create_task(self._timer_loop())

        logger.info("Orchestrator started (event-driven)")

    async def stop(self) -> None:
        """Stop the orchestrator."""
        logger.info("Stopping orchestrator...")
        self._running = False

        # Cancel timer task
        if self._timer_task:
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass

        # Cancel snapshot dispatcher
        if self._snapshot_dispatcher_task:
            self._snapshot_dispatcher_task.cancel()
            try:
                await self._snapshot_dispatcher_task
            except asyncio.CancelledError:
                pass

        # Signal shutdown to ReadinessManager
        if self._readiness_manager:
            self._readiness_manager.shutdown()

        # Stop snapshot service (captures final snapshots if configured)
        if self._snapshot_service:
            await self._snapshot_service.stop()
            logger.info("Snapshot service stopped")

        # Stop async event bus
        if isinstance(self.event_bus, (AsyncEventBus, PriorityEventBus)):
            await self.event_bus.stop()

        # Stop watchdog
        await self.watchdog.stop()

        # Disconnect all brokers via BrokerManager
        await self.broker_manager.disconnect()

        logger.info("Orchestrator stopped")

    async def _connect_providers(self) -> None:
        """Connect to all data providers via BrokerManager and MarketDataManager."""
        if is_console_enabled():
            print("🔌 Connecting to brokers...", flush=True)
        logger.info("Connecting to all brokers via BrokerManager...")

        # Connect all registered broker adapters
        await self.broker_manager.connect()

        # Get broker status for logging and metrics
        all_status = self.broker_manager.get_all_status()
        for name, status in all_status.items():
            if status.connected:
                if is_console_enabled():
                    print(f"✓ {name} broker connected", flush=True)
                logger.info(f"{name} broker connected successfully")
                if self._health_metrics:
                    self._health_metrics.record_broker_status(name, True)
                    self._health_metrics.record_connection_attempt(name, True)
                # Notify ReadinessManager of broker connection
                if self._readiness_manager:
                    self._readiness_manager.on_broker_connected(name)
            else:
                error_msg = status.last_error or "Unknown error"
                if is_console_enabled():
                    print(f"✗ {name} broker failed: {error_msg}", flush=True)
                logger.error(f"Failed to connect {name} broker: {error_msg}")
                if self._health_metrics:
                    self._health_metrics.record_broker_status(name, False)
                    self._health_metrics.record_connection_attempt(name, False)

        # Connect market data providers
        if is_console_enabled():
            print("📊 Connecting to market data sources...", flush=True)
        logger.info("Connecting to market data sources via MarketDataManager...")

        await self.market_data_manager.connect()

        # Get market data status for logging
        md_status = self.market_data_manager.get_all_status()
        for name, status in md_status.items():
            if status.connected:
                streaming_info = " (streaming)" if status.supports_streaming else ""
                greeks_info = " (greeks)" if status.supports_greeks else ""
                if is_console_enabled():
                    print(f"✓ {name} market data connected{streaming_info}{greeks_info}", flush=True)
                logger.info(f"{name} market data connected successfully")
            else:
                error_msg = status.last_error or "Unknown error"
                if is_console_enabled():
                    print(f"✗ {name} market data failed: {error_msg}", flush=True)
                logger.error(f"Failed to connect {name} market data: {error_msg}")

        # Enable streaming if any provider supports it
        if self.market_data_manager.supports_streaming():
            self.market_data_manager.set_streaming_callback(self._on_streaming_market_data)
            self.market_data_manager.enable_streaming()
            if is_console_enabled():
                print("✓ Market data streaming enabled", flush=True)
            logger.info("Market data streaming enabled")

    async def _subscribe_to_position_updates(self) -> None:
        """
        Subscribe to real-time position updates from brokers.

        Phase 6 optimization: Instead of relying solely on timer-based polling,
        receive push notifications when positions change (trades, fills, closes).
        """
        try:
            self.broker_manager.set_position_callback(self._on_broker_position_update)
            await self.broker_manager.subscribe_positions()
            if is_console_enabled():
                print("✓ Position subscription enabled", flush=True)
            logger.info("Subscribed to position updates from all brokers")
        except Exception as e:
            logger.warning(f"Failed to subscribe to position updates: {e}")
            if is_console_enabled():
                print(f"⚠ Position subscription failed: {e}", flush=True)

    def _on_broker_position_update(self, broker_name: str, positions: List[Position]) -> None:
        """
        Handle real-time position update from a broker.

        Called when positions change (trades, fills, closes) instead of waiting
        for the next timer tick.
        """
        logger.info(f"Position update from {broker_name}: {len(positions)} position(s) changed")

        # Publish event for stores to update
        self.event_bus.publish(EventType.POSITION_UPDATED, {
            "broker": broker_name,
            "positions": positions,
            "source": "push",
            "timestamp": datetime.now(),
        })

        # Mark risk engine as dirty to trigger snapshot rebuild
        self.risk_engine.mark_dirty()

    async def _fetch_and_reconcile(self) -> None:
        """
        Fetch positions/account data from all sources and reconcile.

        This is called:
        1. On startup (initial data load)
        2. On each TIMER_TICK (periodic refresh)

        Optimized for parallelism:
        - Positions and accounts are fetched concurrently
        - Market data subscription runs in background (doesn't block timer loop)
        """
        cycle_id = get_cycle_id()
        logger.debug(f"[{cycle_id}] Starting data fetch and reconciliation")

        # 1. Fetch positions AND accounts concurrently (don't wait serially)
        positions_task = asyncio.create_task(self.broker_manager.fetch_positions_by_broker())
        accounts_task = asyncio.create_task(self.broker_manager.fetch_account_info_by_broker())

        # Wait for both to complete
        positions_by_broker, accounts_by_broker = await asyncio.gather(
            positions_task, accounts_task, return_exceptions=True
        )

        # Handle exceptions
        if isinstance(positions_by_broker, Exception):
            logger.error(f"Failed to fetch positions: {positions_by_broker}")
            positions_by_broker = {}
        if isinstance(accounts_by_broker, Exception):
            logger.error(f"Failed to fetch accounts: {accounts_by_broker}")
            accounts_by_broker = {}

        ib_positions = positions_by_broker.get("ib", [])
        futu_positions = positions_by_broker.get("futu", [])
        manual_positions = positions_by_broker.get("manual", [])

        cached_positions = self.position_store.get_all()

        # 2. Reconcile positions across all sources (IB, Futu, manual) - CPU bound, fast
        issues = self.reconciler.reconcile(
            ib_positions, manual_positions, cached_positions, futu_positions
        )
        issue_count = len(issues) if issues else 0
        # Only log on state change to avoid warning storm
        if issue_count != self._last_reconciliation_issue_count:
            if issue_count > 0:
                logger.warning(f"Reconciliation issues changed: {self._last_reconciliation_issue_count} -> {issue_count}")
            elif self._last_reconciliation_issue_count > 0:
                logger.info(f"Reconciliation issues resolved (was {self._last_reconciliation_issue_count})")
            self._last_reconciliation_issue_count = issue_count
        if issues:
            for issue in issues:
                self.event_bus.publish(EventType.RECONCILIATION_ISSUE, issue)

        # Merge positions from all sources (IB > Futu > Manual precedence)
        merged_positions = self.reconciler.merge_positions(ib_positions, manual_positions, futu_positions)
        merged_positions = self.reconciler.remove_expired_options(merged_positions)

        # Publish event - store will update via subscription (single data path)
        self.event_bus.publish(EventType.POSITIONS_BATCH, {
            "positions": merged_positions,
            "source": "reconciler",
            "timestamp": datetime.now(),
        })

        # Notify ReadinessManager of positions loaded per broker
        # IMPORTANT: Always notify even for empty lists (0 positions) to mark broker as "loaded"
        # Without this, the state machine waits forever for brokers with no positions
        if self._readiness_manager:
            self._readiness_manager.on_positions_loaded("ib", len(ib_positions))
            self._readiness_manager.on_positions_loaded("futu", len(futu_positions))
            self._readiness_manager.on_positions_loaded("manual", len(manual_positions))

        # Log position counts prominently (both to file and console if enabled)
        position_msg = (
            f"📊 Positions: IB={len(ib_positions)}, Futu={len(futu_positions)}, "
            f"Manual={len(manual_positions)} → Merged={len(merged_positions)}"
        )
        logger.info(position_msg)
        if is_console_enabled():
            print(position_msg, flush=True)

        # 3. Publish account info (already fetched concurrently above)
        account_info = await self._fetch_account_info_cached()
        self.event_bus.publish(EventType.ACCOUNT_UPDATED, {
            "account_info": account_info,
            "accounts_by_broker": accounts_by_broker,
            "timestamp": datetime.now(),
        })

        account_msg = f"💰 Account: Total=${account_info.net_liquidation:,.0f}"
        for broker_name, acc in accounts_by_broker.items():
            account_msg += f", {broker_name.upper()}=${acc.net_liquidation:,.0f}"
        logger.info(account_msg)
        if is_console_enabled():
            print(account_msg, flush=True)

        # 4. Subscribe to market data in BACKGROUND (don't block timer loop)
        all_symbols = [p.symbol for p in merged_positions]
        new_symbols = [s for s in all_symbols if not self.market_data_store.has_fresh_data(s)]

        if new_symbols and self.market_data_manager.is_connected():
            positions_to_subscribe = [p for p in merged_positions if p.symbol in new_symbols]
            # Fire and forget - subscription happens in background
            asyncio.create_task(
                self._subscribe_market_data_background(positions_to_subscribe, len(all_symbols))
            )
        elif not self.market_data_manager.is_connected():
            logger.warning("No market data providers connected")
            self.health_monitor.update_component_health(
                "market_data_feed",
                HealthStatus.UNHEALTHY,
                "No market data providers connected"
            )
        else:
            logger.debug(f"All {len(all_symbols)} symbols already subscribed")
            self.health_monitor.update_component_health(
                "market_data_feed",
                HealthStatus.HEALTHY,
                f"Streaming {len(all_symbols)} symbols"
            )

        # 5. Validate market data quality (fast, uses cached data)
        market_data = self.market_data_store.get_all()
        self.mdqc.validate_all(market_data)

        # 6. Fetch market-wide indicators (VIX) in background
        asyncio.create_task(self._detect_market_alerts())

        # Mark risk engine as dirty to trigger snapshot rebuild
        self.risk_engine.mark_dirty()

        logger.debug("Data fetch and reconciliation completed")

    async def _subscribe_market_data_background(
        self,
        positions: List[Position],
        total_symbols: int
    ) -> None:
        """
        Subscribe to market data without blocking the timer loop.

        Runs as a background task so position/account refresh continues
        even if market data subscription is slow.
        """
        try:
            logger.info(f"Background: subscribing to {len(positions)} symbols...")
            market_data_list = await self.market_data_manager.fetch_market_data(positions)

            if market_data_list:
                self.market_data_store.upsert(market_data_list)
                logger.info(f"Background: got initial data for {len(market_data_list)} symbols")
                self.risk_engine.mark_dirty()

            self.health_monitor.update_component_health(
                "market_data_feed",
                HealthStatus.HEALTHY,
                f"Streaming {total_symbols} symbols"
            )
        except Exception as e:
            logger.error(f"Background market data subscription failed: {e}")
            self.health_monitor.update_component_health(
                "market_data_feed",
                HealthStatus.DEGRADED,
                f"Subscription failed: {str(e)[:50]}"
            )

    def get_latest_snapshot(self) -> RiskSnapshot | None:
        """Get the latest risk snapshot."""
        return self._latest_snapshot

    async def wait_for_snapshot(self, timeout: float = 5.0) -> RiskSnapshot | None:
        """
        Wait for a new snapshot to be ready.

        This allows the dashboard to sync with the orchestrator cycle
        instead of polling on a separate timer.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            Latest snapshot, or None if timeout.
        """
        try:
            await asyncio.wait_for(self._snapshot_ready.wait(), timeout=timeout)
            self._snapshot_ready.clear()  # Reset for next cycle
            return self._latest_snapshot
        except asyncio.TimeoutError:
            return self._latest_snapshot  # Return cached if timeout

    def get_latest_market_alerts(self) -> list[dict[str, Any]]:
        """Get the latest market alerts."""
        return self._latest_market_alerts

    def _on_streaming_market_data(self, symbol: str, market_data) -> None:
        """
        Handle streaming market data update from MarketDataManager.

        Note: EventBus publish is handled by MarketDataManager._on_provider_data()
        to avoid duplicate MARKET_DATA_TICK events. This callback only handles
        orchestrator-specific bookkeeping (readiness, metrics).
        """
        # Notify ReadinessManager of tick received
        if self._readiness_manager:
            self._readiness_manager.on_tick_received()

        # Record tick metrics
        if self._health_metrics:
            self._health_metrics.record_tick_received(symbol)

    def get_latest_risk_signals(self) -> List[RiskSignal]:
        """Get the latest risk signals."""
        return self._latest_risk_signals

    async def _fetch_account_info_cached(self) -> AccountInfo:
        """
        Fetch aggregated account info with TTL caching to reduce API calls.

        Account info changes slowly (balance, margin), so we only refresh every 30s.

        Returns:
            Aggregated AccountInfo from all brokers.
        """
        now = datetime.now()

        # Check if cache is still valid
        if self._last_account_fetch and self._cached_account_info:
            elapsed = (now - self._last_account_fetch).total_seconds()
            if elapsed < self._account_ttl_sec:
                logger.debug(f"Using cached account info (age: {elapsed:.1f}s)")
                return self._cached_account_info

        # Cache expired or first fetch - refresh from BrokerManager
        account_info = await self.broker_manager.fetch_account_info()
        logger.debug(f"Account info refreshed: NetLiq=${account_info.net_liquidation:,.2f}")

        # Update cache
        self._last_account_fetch = now
        self._cached_account_info = account_info

        return account_info

    async def _detect_market_alerts(self) -> None:
        """Fetch market indicators and detect alerts like VIX spikes."""
        if not self.market_alert_detector:
            self._latest_market_alerts = []
            return

        # Skip if no market data providers connected
        if not self.market_data_manager.is_connected():
            logger.debug("No market data providers connected - skipping market alerts")
            self._latest_market_alerts = []
            return

        symbols = self.config.get("market_alerts", {}).get("symbols", ["VIX"])
        indicators: dict[str, Any] = {}

        try:
            md_map = await self.market_data_manager.fetch_quotes(symbols)
            if md_map:
                # Publish event - store will update via subscription
                self.event_bus.publish(EventType.MARKET_DATA_BATCH, {
                    "market_data": list(md_map.values()),
                    "source": "MarketDataManager_INDICATORS",
                    "timestamp": datetime.now(),
                })

                # Extract VIX level for alerting
                vix_md = md_map.get("VIX")
                if vix_md:
                    vix_mark = vix_md.effective_mid() or vix_md.last or vix_md.bid or vix_md.ask
                    indicators["vix"] = vix_mark
                    indicators["vix_prev_close"] = vix_md.yesterday_close
                    indicators["timestamp"] = vix_md.timestamp
        except Exception as e:
            logger.warning(f"Failed to fetch market indicators: {e}")

        # Detect alerts (safe on empty data)
        self._latest_market_alerts = self.market_alert_detector.detect_alerts(indicators)

    def _log_risk_alerts(self, snapshot: RiskSnapshot) -> None:
        """
        Log all current alerts and signals to the risk alert logger.

        Only logs if there are active alerts or signals to record.

        Args:
            snapshot: Current risk snapshot with position risks
        """
        if not self.risk_alert_logger:
            return

        # Update market cache in logger
        self.risk_alert_logger.update_market_cache(
            market_data_store=self.market_data_store
        )

        # Only log if there are alerts or signals
        if not self._latest_market_alerts and not self._latest_risk_signals:
            return

        # Log batch with full context
        self.risk_alert_logger.log_batch(
            market_alerts=self._latest_market_alerts,
            risk_signals=self._latest_risk_signals,
            snapshot=snapshot,
            position_risks=snapshot.position_risks if snapshot else None,
            market_data_store=self.market_data_store,
        )

        logger.debug(
            f"Logged {len(self._latest_market_alerts)} market alerts, "
            f"{len(self._latest_risk_signals)} risk signals to audit trail"
        )

    # ========== Event-Driven Core Methods ==========

    def _subscribe_components_to_events(self) -> None:
        """Subscribe all components to the event bus."""
        self.position_store.subscribe_to_events(self.event_bus)
        self.market_data_store.subscribe_to_events(self.event_bus)
        self.account_store.subscribe_to_events(self.event_bus)
        self.risk_engine.subscribe_to_events(self.event_bus)

        logger.info("All components subscribed to events")

    async def _timer_loop(self) -> None:
        """
        Timer loop for periodic data refresh and reconciliation.

        This triggers position/account fetching at regular intervals
        to ensure the system remains consistent. Runs initial fetch
        immediately on startup.

        Each iteration creates a new cycle ID for log correlation.
        """
        first_run = True
        while self._running:
            try:
                # Skip sleep on first run to fetch data immediately
                if first_run:
                    first_run = False
                else:
                    await asyncio.sleep(self.refresh_interval_sec)

                # Create new cycle for this refresh iteration
                with new_cycle() as cycle_id:
                    logger.debug(f"[{cycle_id}] Timer tick started")

                    # Run data fetch and reconciliation with timing
                    async with log_timing_async("fetch_and_reconcile", warn_threshold_ms=2000):
                        await self._fetch_and_reconcile()

                    # Emit TIMER_TICK for any subscribers
                    self.event_bus.publish(EventType.TIMER_TICK, {
                        "timestamp": datetime.now(),
                        "cycle_id": cycle_id,
                    })

            except asyncio.CancelledError:
                logger.debug("Timer loop cancelled")
                break
            except Exception as e:
                logger.error(f"Timer loop error: {e}", exc_info=True)
                await asyncio.sleep(5)  # Back off on error

    async def _snapshot_dispatcher(self) -> None:
        """
        Snapshot dispatcher with readiness gating and debouncing.

        Phase 3 optimization:
        - Waits until positions loaded AND enough market data before first snapshot
        - Uses configurable min_interval to debounce frequent updates
        - Falls back to degraded mode after timeout
        """
        last_snapshot_time: Optional[datetime] = None
        self._snapshot_startup_time = datetime.now()

        while self._running:
            try:
                await asyncio.sleep(self._snapshot_min_interval_sec)

                if not self.risk_engine.needs_rebuild():
                    continue

                # Get current data
                positions = self.position_store.get_all()
                if not positions:
                    self.risk_engine.clear_dirty_state()
                    continue

                account_info = self.account_store.get()
                if not account_info:
                    self.risk_engine.clear_dirty_state()
                    continue

                market_data = self.market_data_store.get_all()

                # Update ReadinessManager with market data coverage
                if self._readiness_manager:
                    position_symbols = {p.symbol for p in positions}
                    symbols_with_data = sum(1 for s in position_symbols if s in market_data)
                    self._readiness_manager.on_market_data_update(
                        total_symbols=len(position_symbols),
                        symbols_with_data=symbols_with_data
                    )

                # Readiness gate: Wait for market data before first snapshot
                # - Market OPEN: Wait for 100% live streaming data
                # - Market CLOSED/EXTENDED: Proceed immediately (options don't trade, use previous close)
                if self._readiness_manager and not self._snapshot_readiness_achieved:
                    market_status = MarketHours.get_market_status()

                    # When market is closed or extended hours, don't wait for live data
                    # Options don't trade outside regular hours - use previous close
                    if market_status in ("CLOSED", "EXTENDED"):
                        elapsed = (datetime.now() - self._snapshot_startup_time).total_seconds()
                        logger.info(
                            f"✓ Market {market_status} - proceeding with last business day's close prices "
                            f"(elapsed={elapsed:.1f}s)"
                        )
                        self._snapshot_readiness_achieved = True
                        if self._health_metrics:
                            self._health_metrics.record_system_ready(True)
                            self._health_metrics.record_startup_duration(elapsed)
                    elif not self._readiness_manager.is_ready():
                        elapsed = (datetime.now() - self._snapshot_startup_time).total_seconds()
                        coverage = self._readiness_manager.coverage_ratio
                        # Log progress every 5 seconds
                        if int(elapsed) % 5 == 0:
                            logger.info(
                                f"Waiting for market data ({market_status}): {coverage:.0%} coverage "
                                f"({symbols_with_data}/{len(position_symbols)} symbols), "
                                f"elapsed={elapsed:.1f}s"
                            )
                        if self._health_metrics:
                            self._health_metrics.record_system_ready(False)
                        continue  # Keep waiting - no timeout fallback
                    else:
                        elapsed = (datetime.now() - self._snapshot_startup_time).total_seconds()
                        logger.info(
                            f"✓ Market data ready: 100% coverage after {elapsed:.1f}s - starting snapshots"
                        )
                        self._snapshot_readiness_achieved = True
                        if self._health_metrics:
                            self._health_metrics.record_system_ready(True)
                            self._health_metrics.record_startup_duration(elapsed)

                elif not self._snapshot_readiness_achieved:
                    elapsed = (datetime.now() - self._snapshot_startup_time).total_seconds()
                    ready = self._check_snapshot_readiness(positions, market_data)

                    if not ready and elapsed < self._snapshot_ready_timeout_sec:
                        # Not ready yet, and haven't timed out - skip this cycle
                        if self._health_metrics:
                            self._health_metrics.record_system_ready(False)
                        continue

                    if not ready:
                        logger.warning(
                            f"Snapshot readiness timeout ({elapsed:.1f}s) - proceeding in degraded mode"
                        )
                    else:
                        logger.info(
                            f"Snapshot readiness achieved after {elapsed:.1f}s "
                            f"({len(market_data)}/{len(positions)} symbols have data)"
                        )

                    self._snapshot_readiness_achieved = True

                    # Record startup metrics
                    if self._health_metrics:
                        self._health_metrics.record_system_ready(True)
                        self._health_metrics.record_startup_duration(elapsed)

                # Debounce: enforce minimum interval between snapshots
                now = datetime.now()
                if last_snapshot_time:
                    elapsed_since_last = (now - last_snapshot_time).total_seconds()
                    if elapsed_since_last < self._snapshot_min_interval_sec:
                        continue

                # Build snapshot with timing
                with log_timing("snapshot_build", warn_threshold_ms=250, extra={"positions": len(positions)}):
                    snapshot = self.risk_engine.build_snapshot(positions, market_data, account_info)

                self._latest_snapshot = snapshot
                self.risk_engine.clear_dirty_state()
                last_snapshot_time = now
                logger.debug(f"[{get_cycle_id()}] Snapshot built: {len(positions)} positions")

                # Record market data coverage metrics
                if self._health_metrics:
                    positions_with_data = snapshot.total_positions - snapshot.positions_with_missing_md
                    self._health_metrics.calculate_coverage(
                        snapshot.total_positions,
                        positions_with_data
                    )

                    # Record event bus queue depth metrics
                    if isinstance(self.event_bus, PriorityEventBus):
                        stats = self.event_bus.get_stats()
                        self._health_metrics.record_queue_size("fast", stats.get("fast_queue_size", 0))
                        self._health_metrics.record_queue_size("slow", stats.get("slow_pending", 0))

                # Evaluate risk signals (RiskSignalEngine is required, no legacy fallback)
                if self.risk_signal_engine:
                    self._latest_risk_signals = self.risk_signal_engine.evaluate(snapshot)
                    for signal in self._latest_risk_signals:
                        self.event_bus.publish(EventType.RISK_SIGNAL, {"signal": signal})

                        # Record breach metrics for each signal
                        if self._risk_metrics:
                            level = 2 if signal.severity.value == "CRITICAL" else (
                                1 if signal.severity.value == "WARNING" else 0
                            )
                            self._risk_metrics.record_breach(signal.trigger_rule, level)

                # Log alerts to audit trail
                if self.risk_alert_logger:
                    self._log_risk_alerts(snapshot)

                # Update watchdog
                self.watchdog.update_snapshot_time(snapshot.timestamp)
                self.watchdog.check_missing_market_data(
                    snapshot.total_positions, snapshot.positions_with_missing_md
                )

                # Signal dashboard
                self.event_bus.publish(EventType.SNAPSHOT_READY, {"snapshot": snapshot})
                self._snapshot_ready.set()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Snapshot dispatcher error: {e}", exc_info=True)

    def _check_snapshot_readiness(
        self,
        positions: List[Position],
        market_data: Dict[str, MarketData]
    ) -> bool:
        """
        Check if we have enough market data to build a meaningful snapshot.

        Returns True if market_data covers at least snapshot_ready_ratio of positions.
        """
        if not positions:
            return True

        position_symbols = {p.symbol for p in positions}
        symbols_with_data = sum(1 for s in position_symbols if s in market_data)
        coverage_ratio = symbols_with_data / len(position_symbols)

        return coverage_ratio >= self._snapshot_ready_ratio

    def get_event_bus_stats(self) -> Dict[str, Any]:
        """Get event bus statistics (if async/priority event bus is used)."""
        if isinstance(self.event_bus, (AsyncEventBus, PriorityEventBus)):
            return self.event_bus.get_stats()
        return {}

    def get_readiness_snapshot(self):
        """Get current readiness state snapshot (if ReadinessManager is used)."""
        if self._readiness_manager:
            return self._readiness_manager.get_snapshot()
        return None

    def get_positions_preview(self) -> "RiskSnapshot":
        """
        Get a preview snapshot with positions but no full risk calculations.

        Used to show positions immediately while waiting for market data.
        Positions are displayed with basic info (symbol, qty, broker) but
        without P&L or Greeks calculations.

        Returns:
            RiskSnapshot with position_risks populated from raw positions.
        """
        from ..models.risk_snapshot import RiskSnapshot
        from ..models.position_risk import PositionRisk

        positions = self.position_store.get_all()
        if not positions:
            return RiskSnapshot()

        # Create lightweight PositionRisk objects without full calculations
        # IMPORTANT: Do NOT use avg_price or any fake data - leave fields blank/zero
        position_risks = []
        for pos in positions:
            # Create minimal PositionRisk with position info only
            # All price/value fields are None/0 - we don't have real market data yet
            pos_risk = PositionRisk(
                position=pos,
                mark_price=None,  # No market data yet - leave blank
                iv=None,
                market_value=0.0,  # Cannot calculate without live price
                unrealized_pnl=0.0,
                daily_pnl=0.0,
                delta=0.0,  # No Greeks without market data
                gamma=0.0,
                vega=0.0,
                theta=0.0,
                beta=0.0,
                delta_dollars=0.0,
                notional=0.0,  # Cannot calculate without live price
                beta_adjusted_delta=0.0,
                has_market_data=False,  # Indicate this is preview
                has_greeks=False,
                is_stale=True,  # Mark as stale to indicate preview
                is_using_close=False,
            )
            position_risks.append(pos_risk)

        # Create preview snapshot
        preview = RiskSnapshot()
        preview.position_risks = position_risks
        preview.total_positions = len(positions)
        preview.positions_with_missing_md = len(positions)  # All positions lack live data
        preview.is_preview = True  # Custom flag for dashboard to show "Loading..." indicator

        return preview

    # ========== Warm Start / Snapshot Service Integration ==========

    async def _perform_warm_start(self) -> None:
        """
        Perform warm start by loading state from database snapshots.

        This is called before connecting to brokers to provide immediate
        position visibility while waiting for live data.

        The warm-start service:
        1. Loads latest position snapshots per broker/account
        2. Loads latest account snapshots per broker/account
        3. Validates snapshot age (rejects stale data)
        4. Populates stores with restored data
        """
        if not self._warm_start_service:
            logger.debug("No warm-start service configured - skipping")
            return

        if is_console_enabled():
            print("🔄 Loading cached state from database...", flush=True)

        try:
            # Get broker configurations for warm-start
            brokers_config = self.config.get("brokers", {})
            broker_list = []

            # Build broker list from config
            for broker_name, broker_cfg in brokers_config.items():
                if isinstance(broker_cfg, dict) and broker_cfg.get("enabled", True):
                    broker_list.append({
                        "name": broker_name.upper(),
                        "account_id": broker_cfg.get("account_id", ""),
                    })

            if not broker_list:
                logger.info("No brokers configured for warm-start")
                return

            # Perform warm start with callbacks to populate stores
            result = await self._warm_start_service.warm_start(
                brokers=broker_list,
                on_positions_loaded=self._on_warm_start_positions,
                on_account_loaded=self._on_warm_start_account,
                on_risk_loaded=None,  # Risk snapshots are informational, don't restore
            )

            if result.success:
                msg = (
                    f"✓ Warm start: {result.positions_loaded} positions, "
                    f"{result.accounts_loaded} accounts"
                )
                if result.snapshot_age_seconds:
                    msg += f" (age: {result.snapshot_age_seconds:.0f}s)"
                logger.info(msg)
                if is_console_enabled():
                    print(msg, flush=True)
            else:
                logger.warning(f"Warm start partial failure: {result.error}")
                if is_console_enabled():
                    print(f"⚠ Warm start: {result.error}", flush=True)

        except Exception as e:
            logger.error(f"Warm start failed: {e}")
            if is_console_enabled():
                print(f"✗ Warm start failed: {e}", flush=True)

    def _on_warm_start_positions(
        self,
        broker_name: str,
        account_id: str,
        position_dicts: list,
    ) -> None:
        """
        Handle positions loaded during warm-start.

        Deserializes position dicts back to Position objects and
        populates the position store.
        """
        if not self._warm_start_service or not position_dicts:
            return

        # Reconstruct Position objects from serialized data
        positions = self._warm_start_service.deserialize_positions(position_dicts)

        if positions:
            # Mark source as CACHED for warm-start positions
            from ..models.position import PositionSource
            for pos in positions:
                pos.source = PositionSource.CACHED

            # Publish event for store to update
            self.event_bus.publish(EventType.POSITIONS_BATCH, {
                "positions": positions,
                "source": "warm_start",
                "broker": broker_name,
                "timestamp": datetime.now(),
            })

            logger.info(f"Warm-start: loaded {len(positions)} positions for {broker_name}/{account_id}")

    def _on_warm_start_account(
        self,
        broker_name: str,
        account_id: str,
        account_dict: dict,
    ) -> None:
        """
        Handle account info loaded during warm-start.

        Deserializes account dict back to AccountInfo object and
        populates the account store.
        """
        if not self._warm_start_service or not account_dict:
            return

        # Reconstruct AccountInfo from serialized data
        account_info = self._warm_start_service.deserialize_account(account_dict)

        if account_info:
            # Publish event for store to update
            self.event_bus.publish(EventType.ACCOUNT_UPDATED, {
                "account_info": account_info,
                "source": "warm_start",
                "broker": broker_name,
                "timestamp": datetime.now(),
            })

            logger.info(
                f"Warm-start: loaded account for {broker_name}/{account_id} "
                f"(NetLiq: ${account_info.net_liquidation:,.2f})"
            )

    def _get_positions_for_snapshot(self) -> Dict[tuple, list]:
        """
        Get current positions grouped by (broker, account_id) for snapshot capture.

        Returns dict keyed by (broker, account_id) with list of positions.
        """
        positions = self.position_store.get_all()
        result = {}

        for pos in positions:
            broker = pos.source.value if pos.source else "UNKNOWN"
            account_id = pos.account_id or ""
            key = (broker, account_id)

            if key not in result:
                result[key] = []
            result[key].append(pos)

        return result

    def _get_accounts_for_snapshot(self) -> Dict[tuple, AccountInfo]:
        """
        Get current account info grouped by (broker, account_id) for snapshot capture.

        Returns dict keyed by (broker, account_id) with AccountInfo.
        """
        # For now, we have a single aggregated account
        # Future: support multiple broker accounts
        account_info = self.account_store.get()
        if not account_info:
            return {}

        # Use account_id from the AccountInfo if available
        broker = "AGGREGATED"
        account_id = account_info.account_id or ""

        return {(broker, account_id): account_info}

    def _get_risk_snapshot_for_capture(self) -> Optional[RiskSnapshot]:
        """Get current risk snapshot for snapshot capture."""
        return self._latest_snapshot

