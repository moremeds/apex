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
from datetime import datetime
from typing import Dict, Any, Optional, List, TYPE_CHECKING
import logging

from ..domain.interfaces.position_provider import PositionProvider
from ..domain.interfaces.market_data_provider import MarketDataProvider
from ..domain.interfaces.event_bus import EventBus, EventType
from src.domain.services.risk.risk_engine import RiskEngine
from ..domain.services.pos_reconciler import Reconciler
from ..domain.services.mdqc import MDQC
from src.domain.services.risk.rule_engine import RuleEngine
from ..domain.services.market_alert_detector import MarketAlertDetector
from src.domain.services.risk.risk_signal_engine import RiskSignalEngine
from src.domain.services.risk.risk_alert_logger import RiskAlertLogger
from ..models.risk_snapshot import RiskSnapshot
from ..models.account import AccountInfo
from ..models.position import Position
from ..models.risk_signal import RiskSignal
from ..infrastructure.stores import PositionStore, MarketDataStore, AccountStore
from ..infrastructure.monitoring import HealthMonitor, HealthStatus, Watchdog
from ..infrastructure.persistence import PersistenceManager
from .async_event_bus import AsyncEventBus

if TYPE_CHECKING:
    from ..infrastructure.adapters.ib_adapter import IBAdapter


logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Main application orchestrator (Event-Driven Architecture).

    Coordinates the entire risk monitoring workflow through events:
    1. TIMER_TICK triggers position/account fetching from all sources
    2. TIMER_TICK triggers market data refresh for stale symbols
    3. Data events (POSITION_*, MARKET_DATA_*, ACCOUNT_*) update stores
    4. Snapshot dispatcher rebuilds risk metrics when data changes
    5. SNAPSHOT_READY triggers dashboard updates and persistence
    """

    def __init__(
        self,
        ib_adapter: Optional["IBAdapter"],
        file_loader: PositionProvider,
        position_store: PositionStore,
        market_data_store: MarketDataStore,
        account_store: AccountStore,
        risk_engine: RiskEngine,
        reconciler: Reconciler,
        mdqc: MDQC,
        rule_engine: RuleEngine,
        health_monitor: HealthMonitor,
        watchdog: Watchdog,
        event_bus: EventBus,
        config: Dict[str, Any],
        market_alert_detector: MarketAlertDetector | None = None,
        futu_adapter: Optional[PositionProvider] = None,
        risk_signal_engine: RiskSignalEngine | None = None,
        risk_alert_logger: RiskAlertLogger | None = None,
        persistence_manager: PersistenceManager | None = None,
    ):
        """
        Initialize orchestrator with all dependencies.

        Args:
            ib_adapter: Interactive Brokers adapter (positions + market data).
            file_loader: Manual position file loader.
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
            futu_adapter: Optional Futu adapter (positions + account).
            risk_signal_engine: Multi-layer risk signal engine (optional, replaces rule_engine).
            risk_alert_logger: Optional risk alert logger for audit trail.
            persistence_manager: Optional persistence manager for database storage.
        """
        self.ib_adapter = ib_adapter
        self.futu_adapter = futu_adapter
        self.file_loader = file_loader
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
        self.persistence_manager = persistence_manager

        self.refresh_interval_sec = config.get("dashboard", {}).get("refresh_interval_sec", 2)
        self._running = False

        self._latest_snapshot: RiskSnapshot | None = None
        self._latest_market_alerts: list[dict[str, Any]] = []
        self._latest_risk_signals: List[RiskSignal] = []

        # Event to signal when new snapshot is ready (for dashboard sync)
        self._snapshot_ready: asyncio.Event = asyncio.Event()

        # Streaming callback registration
        self._streaming_enabled = False

        # Event-driven tasks
        self._timer_task: asyncio.Task | None = None
        self._snapshot_dispatcher_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the orchestrator (event-driven mode)."""
        if self._running:
            logger.warning("Orchestrator already running")
            return

        logger.info("Starting orchestrator (event-driven mode)...")

        # Connect to data sources
        await self._connect_providers()

        # Start watchdog
        await self.watchdog.start()

        self._running = True

        # Start async event bus
        if isinstance(self.event_bus, AsyncEventBus):
            await self.event_bus.start()
            logger.info("AsyncEventBus started")

        # Subscribe all components to events
        self._subscribe_components_to_events()

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

        # Stop async event bus
        if isinstance(self.event_bus, AsyncEventBus):
            await self.event_bus.stop()

        # Stop watchdog
        await self.watchdog.stop()

        # Disconnect providers (if they exist)
        if self.ib_adapter:
            await self.ib_adapter.disconnect()
        if self.futu_adapter:
            await self.futu_adapter.disconnect()
        if self.file_loader:
            await self.file_loader.disconnect()

        # Close persistence manager (flushes pending writes)
        if self.persistence_manager:
            self.persistence_manager.close()

        logger.info("Orchestrator stopped")

    async def _connect_providers(self) -> None:
        """Connect to all data providers."""
        # IB Adapter connection (if enabled)
        if self.ib_adapter:
            try:
                print("🔌 Connecting to IB adapter...", flush=True)
                logger.info("Attempting to connect to IB adapter...")
                await self.ib_adapter.connect()

                # Register streaming callback and enable streaming
                self.ib_adapter.set_market_data_callback(self._on_streaming_market_data)
                self.ib_adapter.enable_streaming()
                self._streaming_enabled = True

                self.health_monitor.update_component_health(
                    "ib_adapter", HealthStatus.HEALTHY, "Connected (streaming)"
                )
                print("✓ IB adapter connected (streaming enabled)", flush=True)
                logger.info("IB adapter connected with streaming enabled")
            except Exception as e:
                print(f"✗ IB adapter failed: {e}", flush=True)
                logger.error(f"Failed to connect IB adapter: {e}")
                self.health_monitor.update_component_health(
                    "ib_adapter", HealthStatus.UNHEALTHY, f"Connection failed: {str(e)[:50]}"
                )
        else:
            print("⏭️  IB adapter disabled (demo mode)", flush=True)
            logger.info("IB adapter disabled - running in demo mode")
            self.health_monitor.update_component_health(
                "ib_adapter", HealthStatus.HEALTHY, "Disabled (demo)"
            )

        # Futu Adapter connection (if configured)
        if self.futu_adapter:
            try:
                print("🔌 Connecting to Futu adapter...", flush=True)
                logger.info("Attempting to connect to Futu adapter...")
                await self.futu_adapter.connect()
                self.health_monitor.update_component_health(
                    "futu_adapter", HealthStatus.HEALTHY, "Connected"
                )
                print("✓ Futu adapter connected", flush=True)
                logger.info("Futu adapter connected successfully")
            except Exception as e:
                print(f"✗ Futu adapter failed: {e}", flush=True)
                logger.error(f"Failed to connect Futu adapter: {e}")
                self.health_monitor.update_component_health(
                    "futu_adapter", HealthStatus.UNHEALTHY, f"Connection failed: {str(e)[:50]}"
                )

        # File loader connection (optional - ignore if empty or fails)
        try:
            logger.info("Loading manual positions from file...")
            await self.file_loader.connect()
            self.health_monitor.update_component_health(
                "file_loader", HealthStatus.HEALTHY, "Loaded"
            )
            logger.info("Manual positions loaded successfully")
        except Exception as e:
            logger.debug(f"Manual positions not loaded (optional): {e}")
            # Don't mark as unhealthy - manual positions are optional

    async def _fetch_and_reconcile(self) -> None:
        """
        Fetch positions/account data from all sources and reconcile.

        This is called:
        1. On startup (initial data load)
        2. On each TIMER_TICK (periodic refresh)

        Market data is handled separately via streaming callbacks.
        Snapshot rebuilds are handled by the snapshot dispatcher.
        """
        logger.debug("Starting data fetch and reconciliation")

        # 1. Fetch positions from all sources (handle disconnected adapters gracefully)
        ib_positions: List[Position] = []
        futu_positions: List[Position] = []
        manual_positions: List[Position] = []

        # Check if demo mode (positions from file only)
        demo_positions_only = self.config.get("demo_positions_only", False)

        # Fetch from IB if enabled and connected (skip in demo mode)
        if demo_positions_only:
            logger.debug("Demo mode - skipping IB positions (using file positions only)")
        elif self.ib_adapter and self.ib_adapter.is_connected():
            try:
                ib_positions = await self.ib_adapter.fetch_positions()
                self.health_monitor.update_component_health(
                    "ib_adapter", HealthStatus.HEALTHY, f"Fetched {len(ib_positions)} positions"
                )
            except Exception as e:
                logger.warning(f"Failed to fetch IB positions: {e}")
                self.health_monitor.update_component_health(
                    "ib_adapter", HealthStatus.DEGRADED, f"Position fetch failed: {str(e)[:50]}"
                )
        elif not self.ib_adapter:
            logger.debug("IB adapter disabled")
        else:
            logger.debug("IB adapter not connected - skipping IB positions")

        # Fetch from Futu if configured and connected (skip in demo mode)
        if demo_positions_only:
            logger.debug("Demo mode - skipping Futu positions")
        elif self.futu_adapter:
            if self.futu_adapter.is_connected():
                try:
                    logger.info("Fetching positions from Futu...")
                    futu_positions = await self.futu_adapter.fetch_positions()
                    logger.info(f"✓ Futu: {len(futu_positions)} positions")
                    self.health_monitor.update_component_health(
                        "futu_adapter", HealthStatus.HEALTHY, f"Fetched {len(futu_positions)} positions"
                    )
                except Exception as e:
                    logger.warning(f"Failed to fetch Futu positions: {e}")
                    self.health_monitor.update_component_health(
                        "futu_adapter", HealthStatus.DEGRADED, f"Position fetch failed: {str(e)[:50]}"
                    )
            else:
                logger.warning("Futu adapter configured but not connected")
                self.health_monitor.update_component_health(
                    "futu_adapter", HealthStatus.UNHEALTHY, "Not connected"
                )
        else:
            logger.debug("Futu adapter not configured")

        # Fetch from file loader if connected (optional - ignore if empty)
        if self.file_loader and self.file_loader.is_connected():
            try:
                manual_positions = await self.file_loader.fetch_positions()
                if manual_positions:
                    logger.debug(f"Loaded {len(manual_positions)} manual positions")
            except Exception as e:
                logger.debug(f"Manual positions not available: {e}")

        cached_positions = self.position_store.get_all()

        # 2. Reconcile positions across all sources (IB, Futu, manual)
        issues = self.reconciler.reconcile(
            ib_positions, manual_positions, cached_positions, futu_positions
        )
        if issues:
            logger.warning(f"Found {len(issues)} reconciliation issues")
            for issue in issues:
                self.event_bus.publish(EventType.RECONCILIATION_ISSUE, issue)

        # Merge positions from all sources (IB > Futu > Manual precedence)
        merged_positions = self.reconciler.merge_positions(ib_positions, manual_positions, futu_positions)
        merged_positions = self.reconciler.remove_expired_options(merged_positions)
        self.position_store.upsert_positions(merged_positions)
        self.event_bus.publish(EventType.POSITIONS_BATCH, {
            "positions": merged_positions,
            "source": "reconciler",
            "timestamp": datetime.now(),
        })

        # Log position counts prominently (both to file and console)
        position_msg = (
            f"📊 Positions: IB={len(ib_positions)}, Futu={len(futu_positions)}, "
            f"Manual={len(manual_positions)} → Merged={len(merged_positions)}"
        )
        logger.info(position_msg)
        print(position_msg, flush=True)  # Ensure visible in console

        # 3. Fetch account info from all brokers
        ib_account: Optional[AccountInfo] = None
        futu_account: Optional[AccountInfo] = None

        # Fetch from IB if enabled and connected
        if self.ib_adapter and self.ib_adapter.is_connected():
            try:
                ib_account = await self.ib_adapter.fetch_account_info()
                logger.debug(f"IB account: NetLiq=${ib_account.net_liquidation:,.2f}")
            except Exception as e:
                logger.warning(f"Failed to fetch account info from IB: {e}")

        # Fetch from Futu if connected
        if self.futu_adapter and self.futu_adapter.is_connected():
            try:
                if hasattr(self.futu_adapter, 'fetch_account_info'):
                    futu_account = await self.futu_adapter.fetch_account_info()
                    logger.debug(f"Futu account: NetLiq=${futu_account.net_liquidation:,.2f}")
            except Exception as e:
                logger.warning(f"Failed to fetch account info from Futu: {e}")

        # Aggregate account info from all sources (single source of truth in Reconciler)
        account_info = self.reconciler.aggregate_account_info(ib_account, futu_account)
        self.account_store.update(account_info)
        self.event_bus.publish(EventType.ACCOUNT_UPDATED, {
            "account_info": account_info,
            "ib_account": ib_account,
            "futu_account": futu_account,
            "timestamp": datetime.now(),
        })

        account_msg = (
            f"💰 Account: IB=${ib_account.net_liquidation if ib_account else 0:,.0f}, "
            f"Futu=${futu_account.net_liquidation if futu_account else 0:,.0f}, "
            f"Total=${account_info.net_liquidation:,.0f}"
        )
        logger.info(account_msg)
        print(account_msg, flush=True)  # Ensure visible in console

        # 4. Fetch market data for new/stale positions
        all_symbols = [p.symbol for p in merged_positions]
        symbols_needing_refresh = set(self.market_data_store.get_symbols_needing_refresh(all_symbols))
        positions_to_fetch = [p for p in merged_positions if p.symbol in symbols_needing_refresh]

        if positions_to_fetch:
            logger.debug(f"Fetching market data for {len(positions_to_fetch)}/{len(merged_positions)} positions")
            try:
                if not self.ib_adapter or not self.ib_adapter.is_connected():
                    logger.error("IB adapter not connected - cannot fetch market data")
                    self.health_monitor.update_component_health(
                        "market_data_feed",
                        HealthStatus.UNHEALTHY,
                        "IB adapter not connected"
                    )
                else:
                    # Fetch market data from IBKR (will emit events via streaming)
                    market_data_list = await self.ib_adapter.fetch_market_data(positions_to_fetch)
                    self.market_data_store.upsert(market_data_list)
                    self.event_bus.publish(EventType.MARKET_DATA_BATCH, {
                        "data": market_data_list,
                        "source": "IB",
                        "timestamp": datetime.now(),
                    })

                    if market_data_list:
                        self.health_monitor.update_component_health(
                            "market_data_feed",
                            HealthStatus.HEALTHY,
                            f"Fetched {len(market_data_list)} symbols"
                        )
                    else:
                        self.health_monitor.update_component_health(
                            "market_data_feed",
                            HealthStatus.DEGRADED,
                            "Fetch returned no data"
                        )
            except Exception as e:
                logger.error(f"Failed to fetch market data: {e}")
                self.health_monitor.update_component_health(
                    "market_data_feed",
                    HealthStatus.UNHEALTHY,
                    f"Fetch failed: {str(e)[:50]}"
                )
        else:
            logger.debug(f"All market data fresh ({len(merged_positions)} positions)")
            self.health_monitor.update_component_health(
                "market_data_feed",
                HealthStatus.HEALTHY,
                "Using cached data (all fresh)"
            )

        # 5. Validate market data quality
        market_data = self.market_data_store.get_all()
        self.mdqc.validate_all(market_data)

        # 6. Fetch market-wide indicators (VIX) and detect alerts
        await self._detect_market_alerts()

        # Mark risk engine as dirty to trigger snapshot rebuild
        if hasattr(self.risk_engine, 'mark_dirty'):
            self.risk_engine.mark_dirty()

        logger.debug("Data fetch and reconciliation completed")

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
        Handle streaming market data update from IB.

        Updates the market data store - snapshot rebuild is handled
        by the snapshot dispatcher when risk engine is marked dirty.
        """
        # Update store
        self.market_data_store.upsert([market_data])

        # Mark risk engine as dirty (snapshot dispatcher will rebuild)
        if hasattr(self.risk_engine, 'mark_dirty'):
            self.risk_engine.mark_dirty()

    def get_latest_risk_signals(self) -> List[RiskSignal]:
        """Get the latest risk signals."""
        return self._latest_risk_signals

    async def _detect_market_alerts(self) -> None:
        """Fetch market indicators and detect alerts like VIX spikes."""
        if not self.market_alert_detector:
            self._latest_market_alerts = []
            return

        # Skip if IB not enabled or not connected
        if not self.ib_adapter or not self.ib_adapter.is_connected():
            logger.debug("IB adapter not available - skipping market alerts")
            self._latest_market_alerts = []
            return

        symbols = self.config.get("market_alerts", {}).get("symbols", ["VIX"])
        indicators: dict[str, Any] = {}

        try:
            md_map = await self.ib_adapter.fetch_market_indicators(symbols)
            if md_map:
                # Cache indicators for reuse / dashboard display
                self.market_data_store.upsert(md_map.values())

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
        # Subscribe stores
        if hasattr(self.position_store, 'subscribe_to_events'):
            self.position_store.subscribe_to_events(self.event_bus)

        if hasattr(self.market_data_store, 'subscribe_to_events'):
            self.market_data_store.subscribe_to_events(self.event_bus)

        if hasattr(self.account_store, 'subscribe_to_events'):
            self.account_store.subscribe_to_events(self.event_bus)

        # Subscribe risk engine
        if hasattr(self.risk_engine, 'subscribe_to_events'):
            self.risk_engine.subscribe_to_events(self.event_bus)

        # Subscribe persistence manager
        if self.persistence_manager and hasattr(self.persistence_manager, 'subscribe_to_events'):
            self.persistence_manager.subscribe_to_events(self.event_bus)

        logger.info("All components subscribed to events")

    async def _timer_loop(self) -> None:
        """
        Timer loop for periodic data refresh and reconciliation.

        This triggers position/account fetching at regular intervals
        to ensure the system remains consistent. Runs initial fetch
        immediately on startup.
        """
        first_run = True
        while self._running:
            try:
                # Skip sleep on first run to fetch data immediately
                if first_run:
                    first_run = False
                else:
                    await asyncio.sleep(self.refresh_interval_sec)

                # Run data fetch and reconciliation
                await self._fetch_and_reconcile()

                # Emit TIMER_TICK for any subscribers
                self.event_bus.publish(EventType.TIMER_TICK, {
                    "timestamp": datetime.now(),
                })

            except asyncio.CancelledError:
                logger.debug("Timer loop cancelled")
                break
            except Exception as e:
                logger.error(f"Timer loop error: {e}", exc_info=True)
                await asyncio.sleep(5)  # Back off on error

    async def _snapshot_dispatcher(self) -> None:
        """
        Snapshot dispatcher with debouncing.

        Rebuilds snapshots when the risk engine is marked dirty,
        with a minimum interval between rebuilds. Also handles:
        - Risk signal evaluation
        - Persistence
        - Watchdog updates
        """
        min_interval_sec = 0.1  # 100ms debounce

        while self._running:
            try:
                await asyncio.sleep(min_interval_sec)

                # Check if risk engine needs rebuild
                if hasattr(self.risk_engine, 'needs_rebuild') and self.risk_engine.needs_rebuild():
                    # Get current data
                    positions = self.position_store.get_all()
                    if not positions:
                        self.risk_engine.clear_dirty_state()
                        continue

                    market_data = self.market_data_store.get_all()
                    account_info = self.account_store.get()

                    if not account_info:
                        self.risk_engine.clear_dirty_state()
                        continue

                    # Build snapshot
                    snapshot = self.risk_engine.build_snapshot(
                        positions, market_data, account_info
                    )
                    self._latest_snapshot = snapshot

                    # Clear dirty state
                    self.risk_engine.clear_dirty_state()

                    # Evaluate risk signals
                    if self.risk_signal_engine:
                        risk_signals = self.risk_signal_engine.evaluate(snapshot)
                        self._latest_risk_signals = risk_signals
                        if risk_signals:
                            logger.warning(f"Found {len(risk_signals)} risk signals")
                            for signal in risk_signals:
                                self.event_bus.publish(EventType.RISK_SIGNAL, {"signal": signal})

                            # Persist risk signals
                            if self.persistence_manager:
                                try:
                                    self.persistence_manager.persist_alerts(risk_signals)
                                except Exception as e:
                                    logger.warning(f"Failed to persist risk signals: {e}")
                    else:
                        # Legacy: Portfolio rule engine only
                        breaches = self.rule_engine.evaluate(snapshot)
                        if breaches:
                            logger.warning(f"Found {len(breaches)} limit breaches")
                            for breach in breaches:
                                self.event_bus.publish(EventType.LIMIT_BREACHED, breach)

                    # Log risk alerts to audit trail
                    if self.risk_alert_logger:
                        self._log_risk_alerts(snapshot)

                    # Persist snapshot
                    if self.persistence_manager:
                        try:
                            self.persistence_manager.persist_snapshot(snapshot)
                        except Exception as e:
                            logger.warning(f"Failed to persist snapshot: {e}")

                    # Update watchdog
                    self.watchdog.update_snapshot_time(snapshot.timestamp)
                    self.watchdog.check_missing_market_data(
                        snapshot.total_positions, snapshot.positions_with_missing_md
                    )

                    # Publish snapshot ready event
                    self.event_bus.publish(EventType.SNAPSHOT_READY, {"snapshot": snapshot})

                    # Signal dashboard
                    self._snapshot_ready.set()

                    logger.debug(f"Snapshot rebuilt: {len(positions)} positions")

            except asyncio.CancelledError:
                logger.debug("Snapshot dispatcher cancelled")
                break
            except Exception as e:
                logger.error(f"Snapshot dispatcher error: {e}", exc_info=True)

    def get_event_bus_stats(self) -> Dict[str, Any]:
        """Get event bus statistics (if async event bus is used)."""
        if isinstance(self.event_bus, AsyncEventBus):
            return self.event_bus.get_stats()
        return {}

