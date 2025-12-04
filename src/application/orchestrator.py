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

from ..domain.interfaces.event_bus import EventBus, EventType
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
from ..infrastructure.persistence import PersistenceManager
from ..infrastructure.adapters.broker_manager import BrokerManager
from ..infrastructure.adapters.market_data_manager import MarketDataManager
from .async_event_bus import AsyncEventBus


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
        event_bus: EventBus,
        config: Dict[str, Any],
        market_alert_detector: MarketAlertDetector | None = None,
        risk_signal_engine: RiskSignalEngine | None = None,
        risk_alert_logger: RiskAlertLogger | None = None,
        persistence_manager: PersistenceManager | None = None,
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
            persistence_manager: Optional persistence manager for database storage.
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
        self.persistence_manager = persistence_manager

        self.refresh_interval_sec = config.get("dashboard", {}).get("refresh_interval_sec", 2)
        self._running = False

        self._latest_snapshot: RiskSnapshot | None = None
        self._latest_market_alerts: list[dict[str, Any]] = []
        self._latest_risk_signals: List[RiskSignal] = []

        # Event to signal when new snapshot is ready (for dashboard sync)
        self._snapshot_ready: asyncio.Event = asyncio.Event()

        # Event-driven tasks
        self._timer_task: asyncio.Task | None = None
        self._snapshot_dispatcher_task: asyncio.Task | None = None

        # Account TTL caching (account info changes slowly - fetch every 30s, not every tick)
        self._account_ttl_sec: float = config.get("account_ttl_sec", 30.0)
        self._last_account_fetch: Optional[datetime] = None
        self._cached_account_info: Optional[AccountInfo] = None

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

        # Disconnect all brokers via BrokerManager
        await self.broker_manager.disconnect()

        # Close persistence manager (flushes pending writes)
        if self.persistence_manager:
            self.persistence_manager.close()

        logger.info("Orchestrator stopped")

    async def _connect_providers(self) -> None:
        """Connect to all data providers via BrokerManager and MarketDataManager."""
        print("🔌 Connecting to brokers...", flush=True)
        logger.info("Connecting to all brokers via BrokerManager...")

        # Connect all registered broker adapters
        await self.broker_manager.connect()

        # Get broker status for logging
        all_status = self.broker_manager.get_all_status()
        for name, status in all_status.items():
            if status.connected:
                print(f"✓ {name} broker connected", flush=True)
                logger.info(f"{name} broker connected successfully")
            else:
                error_msg = status.last_error or "Unknown error"
                print(f"✗ {name} broker failed: {error_msg}", flush=True)
                logger.error(f"Failed to connect {name} broker: {error_msg}")

        # Connect market data providers
        print("📊 Connecting to market data sources...", flush=True)
        logger.info("Connecting to market data sources via MarketDataManager...")

        await self.market_data_manager.connect()

        # Get market data status for logging
        md_status = self.market_data_manager.get_all_status()
        for name, status in md_status.items():
            if status.connected:
                streaming_info = " (streaming)" if status.supports_streaming else ""
                greeks_info = " (greeks)" if status.supports_greeks else ""
                print(f"✓ {name} market data connected{streaming_info}{greeks_info}", flush=True)
                logger.info(f"{name} market data connected successfully")
            else:
                error_msg = status.last_error or "Unknown error"
                print(f"✗ {name} market data failed: {error_msg}", flush=True)
                logger.error(f"Failed to connect {name} market data: {error_msg}")

        # Enable streaming if any provider supports it
        if self.market_data_manager.supports_streaming():
            self.market_data_manager.set_streaming_callback(self._on_streaming_market_data)
            self.market_data_manager.enable_streaming()
            print("✓ Market data streaming enabled", flush=True)
            logger.info("Market data streaming enabled")

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

        # Check if demo mode (positions from file only)
        # 1. Fetch positions from all brokers via BrokerManager
        positions_by_broker = await self.broker_manager.fetch_positions_by_broker()
        ib_positions = positions_by_broker.get("ib", [])
        futu_positions = positions_by_broker.get("futu", [])
        manual_positions = positions_by_broker.get("manual", [])

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
        # Publish event - store will update via subscription (single data path)
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

        # 3. Fetch account info from all brokers (with TTL caching)
        account_info = await self._fetch_account_info_cached()
        accounts_by_broker = await self.broker_manager.fetch_account_info_by_broker()
        # Publish event - store will update via subscription (single data path)
        self.event_bus.publish(EventType.ACCOUNT_UPDATED, {
            "account_info": account_info,
            "accounts_by_broker": accounts_by_broker,
            "timestamp": datetime.now(),
        })

        account_msg = f"💰 Account: Total=${account_info.net_liquidation:,.0f}"
        for broker_name, acc in accounts_by_broker.items():
            account_msg += f", {broker_name.upper()}=${acc.net_liquidation:,.0f}"
        logger.info(account_msg)
        print(account_msg, flush=True)  # Ensure visible in console

        # 4. Fetch market data for new/stale positions via MarketDataManager
        all_symbols = [p.symbol for p in merged_positions]
        symbols_needing_refresh = set(self.market_data_store.get_symbols_needing_refresh(all_symbols))
        positions_to_fetch = [p for p in merged_positions if p.symbol in symbols_needing_refresh]

        if positions_to_fetch:
            logger.debug(f"Fetching market data for {len(positions_to_fetch)}/{len(merged_positions)} positions")
            try:
                if not self.market_data_manager.is_connected():
                    logger.error("No market data providers connected")
                    self.health_monitor.update_component_health(
                        "market_data_feed",
                        HealthStatus.UNHEALTHY,
                        "No market data providers connected"
                    )
                else:
                    # Fetch market data from all available providers
                    market_data_list = await self.market_data_manager.fetch_market_data(positions_to_fetch)
                    # Publish event - store will update via subscription
                    self.event_bus.publish(EventType.MARKET_DATA_BATCH, {
                        "market_data": market_data_list,
                        "source": "MarketDataManager",
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

        # 7. Sync orders and trades from brokers (event-driven persistence)
        await self._sync_orders_and_trades()

        # Mark risk engine as dirty to trigger snapshot rebuild
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
        """Handle streaming market data update from IB via event bus."""
        self.event_bus.publish(EventType.MARKET_DATA_TICK, {
            "symbol": symbol,
            "data": market_data,
            "source": "IB_STREAMING",
            "timestamp": datetime.now(),
        })

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

    async def _sync_orders_and_trades(self) -> None:
        """
        Sync orders and trades from all brokers.

        This triggers ORDERS_BATCH and TRADES_BATCH events which the
        PersistenceManager subscribes to for database persistence.
        """
        # Get sync config
        order_sync_config = self.config.get("order_sync", {})
        enabled = order_sync_config.get("enabled", True)
        days_back = order_sync_config.get("days_back", 7)

        if not enabled:
            return

        try:
            # Fetch orders - BrokerManager publishes ORDERS_BATCH event
            await self.broker_manager.fetch_orders(
                include_open=True,
                include_completed=True,
                days_back=days_back,
                publish_event=True,
            )

            # Fetch trades - BrokerManager publishes TRADES_BATCH event
            await self.broker_manager.fetch_trades(
                days_back=days_back,
                publish_event=True,
            )

        except Exception as e:
            logger.error(f"Failed to sync orders/trades: {e}")

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

        if self.persistence_manager:
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
        """Snapshot dispatcher with debouncing. Rebuilds when risk engine is dirty."""
        min_interval_sec = 0.1  # 100ms debounce

        while self._running:
            try:
                await asyncio.sleep(min_interval_sec)

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

                # Build snapshot
                snapshot = self.risk_engine.build_snapshot(positions, market_data, account_info)
                self._latest_snapshot = snapshot
                self.risk_engine.clear_dirty_state()

                # Evaluate risk signals (RiskSignalEngine is required, no legacy fallback)
                if self.risk_signal_engine:
                    self._latest_risk_signals = self.risk_signal_engine.evaluate(snapshot)
                    for signal in self._latest_risk_signals:
                        self.event_bus.publish(EventType.RISK_SIGNAL, {"signal": signal})

                # Log alerts to audit trail
                if self.risk_alert_logger:
                    self._log_risk_alerts(snapshot)

                # Persistence handled via SNAPSHOT_READY event subscription

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

    def get_event_bus_stats(self) -> Dict[str, Any]:
        """Get event bus statistics (if async event bus is used)."""
        if isinstance(self.event_bus, AsyncEventBus):
            return self.event_bus.get_stats()
        return {}

