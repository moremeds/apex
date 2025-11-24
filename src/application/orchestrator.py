"""
Orchestrator - Main application workflow control.

Coordinates:
- Position/market data fetching
- Position reconciliation
- Risk calculation
- Limit breach detection
- Dashboard updates
- Event publishing
"""

from __future__ import annotations
import asyncio
from typing import Dict, Any, List
import logging

from ..domain.interfaces.position_provider import PositionProvider
from ..domain.interfaces.market_data_provider import MarketDataProvider
from ..domain.interfaces.event_bus import EventBus, EventType
from ..domain.services.risk_engine import RiskEngine
from ..domain.services.reconciler import Reconciler
from ..domain.services.mdqc import MDQC
from ..domain.services.rule_engine import RuleEngine
from ..infrastructure.stores import PositionStore, MarketDataStore, AccountStore
from ..infrastructure.monitoring import HealthMonitor, HealthStatus, Watchdog
from ..models.risk_snapshot import RiskSnapshot
from ..models.account import AccountInfo


logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Main application orchestrator.

    Coordinates the entire risk monitoring workflow:
    1. Fetch positions from IB and manual sources
    2. Reconcile positions across sources
    3. Fetch market data and validate quality
    4. Calculate risk metrics
    5. Evaluate risk limits
    6. Publish events and update dashboard
    """

    def __init__(
        self,
        ib_adapter: PositionProvider | MarketDataProvider,
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
    ):
        """
        Initialize orchestrator with all dependencies.

        Args:
            ib_adapter: Interactive Brokers adapter.
            file_loader: Manual position file loader.
            position_store: Position store.
            market_data_store: Market data store.
            account_store: Account store.
            risk_engine: Risk calculation engine.
            reconciler: Position reconciliation service.
            mdqc: Market data quality control.
            rule_engine: Risk limit evaluation.
            health_monitor: Health monitoring service.
            watchdog: Watchdog monitoring service.
            event_bus: Event bus for publish-subscribe.
            config: Application configuration.
        """
        self.ib_adapter = ib_adapter
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

        self.refresh_interval_sec = config.get("dashboard", {}).get("refresh_interval_sec", 2)
        self._running = False
        self._task: asyncio.Task | None = None

        self._latest_snapshot: RiskSnapshot | None = None

    async def start(self) -> None:
        """Start the orchestrator main loop."""
        if self._running:
            logger.warning("Orchestrator already running")
            return

        logger.info("Starting orchestrator...")

        # Connect to data sources
        await self._connect_providers()

        # Start watchdog
        await self.watchdog.start()

        # Start main loop
        self._running = True
        self._task = asyncio.create_task(self._main_loop())
        logger.info("Orchestrator started")

    async def stop(self) -> None:
        """Stop the orchestrator."""
        logger.info("Stopping orchestrator...")
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Stop watchdog
        await self.watchdog.stop()

        # Disconnect providers
        await self.ib_adapter.disconnect()
        await self.file_loader.disconnect()

        logger.info("Orchestrator stopped")

    async def _connect_providers(self) -> None:
        """Connect to all data providers."""
        # IB Adapter connection
        try:
            logger.info("Attempting to connect to IB adapter...")
            await self.ib_adapter.connect()
            self.health_monitor.update_component_health(
                "ib_adapter", HealthStatus.HEALTHY, "Connected"
            )
            logger.info("IB adapter connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect IB adapter: {e}")
            self.health_monitor.update_component_health(
                "ib_adapter", HealthStatus.UNHEALTHY, f"Connection failed: {str(e)[:50]}"
            )

        # File loader connection
        try:
            logger.info("Loading manual positions from file...")
            await self.file_loader.connect()
            self.health_monitor.update_component_health(
                "file_loader", HealthStatus.HEALTHY, "Loaded"
            )
            logger.info("Manual positions loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load manual positions: {e}")
            self.health_monitor.update_component_health(
                "file_loader", HealthStatus.UNHEALTHY, f"Load failed: {str(e)[:50]}"
            )

    async def _main_loop(self) -> None:
        """Main orchestration loop."""
        while self._running:
            try:
                await self._run_cycle()
                await asyncio.sleep(self.refresh_interval_sec)
            except Exception as e:
                logger.error(f"Orchestrator cycle error: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _run_cycle(self) -> None:
        """
        Run one complete orchestration cycle.

        Steps:
        1. Fetch positions from all sources
        2. Reconcile positions
        3. Fetch market data
        4. Validate market data quality
        5. Calculate risk metrics
        6. Evaluate risk limits
        7. Update watchdog
        8. Publish events
        """
        logger.debug("Starting orchestration cycle")

        # 1. Fetch positions
        ib_positions = await self.ib_adapter.fetch_positions()
        manual_positions = await self.file_loader.fetch_positions()
        cached_positions = self.position_store.get_all()

        # 2. Reconcile positions
        issues = self.reconciler.reconcile(ib_positions, manual_positions, cached_positions)
        if issues:
            logger.warning(f"Found {len(issues)} reconciliation issues")
            for issue in issues:
                self.event_bus.publish(EventType.RECONCILIATION_ISSUE, issue)

        # Merge positions (IB takes precedence, then manual)
        merged_positions = self._merge_positions(ib_positions, manual_positions)
        self.position_store.upsert_positions(merged_positions)
        self.event_bus.publish(EventType.POSITION_CHANGED, {"count": len(merged_positions)})

        # 3. Fetch market data
        symbols = [p.symbol for p in merged_positions]
        try:
            if self.ib_adapter.is_connected():
                market_data_list = await self.ib_adapter.fetch_market_data(merged_positions)
                self.market_data_store.upsert(market_data_list)
            else:
                # IB not connected - use mock data for testing
                #todo add warning at ui
                logger.warning("IB not connected - generating mock market data for testing")
                from ..infrastructure.adapters.mock_market_data import MockMarketDataProvider
                mock_provider = MockMarketDataProvider()
                market_data_list = mock_provider.generate_market_data(merged_positions)
                self.market_data_store.upsert(market_data_list)
        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
            # Try mock data as fallback
            try:
                from ..infrastructure.adapters.mock_market_data import MockMarketDataProvider
                mock_provider = MockMarketDataProvider()
                market_data_list = mock_provider.generate_market_data(merged_positions)
                self.market_data_store.upsert(market_data_list)
                logger.info("Using mock market data as fallback")
            except Exception as e2:
                logger.error(f"Failed to generate mock market data: {e2}")
                # Continue with cached market data

        # 4. Validate market data quality
        market_data = self.market_data_store.get_all()
        self.mdqc.validate_all(market_data)

        # 5. Fetch account info
        account_info = await self.ib_adapter.fetch_account_info()
        self.account_store.update(account_info)

        # 6. Calculate risk metrics
        snapshot = self.risk_engine.build_snapshot(
            merged_positions, market_data, account_info
        )
        self._latest_snapshot = snapshot

        # 7. Evaluate risk limits
        breaches = self.rule_engine.evaluate(snapshot)
        if breaches:
            logger.warning(f"Found {len(breaches)} limit breaches")
            for breach in breaches:
                self.event_bus.publish(EventType.LIMIT_BREACHED, breach)

        # 8. Update watchdog
        self.watchdog.update_snapshot_time(snapshot.timestamp)
        self.watchdog.check_missing_market_data(
            snapshot.total_positions, snapshot.positions_with_missing_md
        )

        logger.debug("Orchestration cycle completed")

    def _merge_positions(self, ib_positions: List, manual_positions: List) -> List:
        """
        Merge positions from multiple sources.

        For positions with the same key:
        - Aggregates quantities across accounts/sources
        - Computes weighted average price
        - IB source takes precedence over manual for metadata

        Args:
            ib_positions: Positions from IB.
            manual_positions: Positions from manual file.

        Returns:
            Merged position list with aggregated quantities.
        """
        merged = {}

        # Process all positions (manual first, then IB)
        for p in manual_positions + ib_positions:
            key = p.key()

            if key not in merged:
                # First time seeing this position
                merged[key] = p
            else:
                # Aggregate with existing position
                existing = merged[key]

                # Compute weighted average price
                total_value = (existing.avg_price * existing.quantity * existing.multiplier +
                              p.avg_price * p.quantity * p.multiplier)
                total_quantity = existing.quantity + p.quantity

                if total_quantity != 0:
                    new_avg_price = total_value / (total_quantity * existing.multiplier)
                else:
                    new_avg_price = existing.avg_price

                # Update existing position
                existing.quantity = total_quantity
                existing.avg_price = new_avg_price

                # IB source takes precedence for metadata
                if p.source.value == "IB":
                    existing.source = p.source
                    existing.last_updated = p.last_updated
                    existing.account_id = p.account_id  # Keep last account_id for reference

        return list(merged.values())

    def get_latest_snapshot(self) -> RiskSnapshot | None:
        """Get the latest risk snapshot."""
        return self._latest_snapshot
