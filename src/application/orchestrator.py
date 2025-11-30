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
from datetime import datetime
from typing import Dict, Any, Optional, List
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


logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Main application orchestrator.

    Coordinates the entire risk monitoring workflow:
    1. Fetch positions from IB, Futu, and manual sources
    2. Reconcile positions across sources
    3. Fetch market data and validate quality
    4. Calculate risk metrics
    5. Evaluate risk limits
    6. Publish events and update dashboard
    """

    def __init__(
        self,
        ib_adapter: Optional[PositionProvider | MarketDataProvider],
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

        self.refresh_interval_sec = config.get("dashboard", {}).get("refresh_interval_sec", 2)
        self._running = False
        self._task: asyncio.Task | None = None

        self._latest_snapshot: RiskSnapshot | None = None
        self._latest_market_alerts: list[dict[str, Any]] = []
        self._latest_risk_signals: List[RiskSignal] = []

        # Event to signal when new snapshot is ready (for dashboard sync)
        self._snapshot_ready: asyncio.Event = asyncio.Event()

        # Streaming mode: rebuild snapshot on market data updates
        self._streaming_enabled = False
        self._market_data_updated: asyncio.Event = asyncio.Event()
        self._last_streaming_rebuild: datetime | None = None

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

        # Disconnect providers (if they exist)
        if self.ib_adapter:
            await self.ib_adapter.disconnect()
        if self.futu_adapter:
            await self.futu_adapter.disconnect()
        if self.file_loader:
            await self.file_loader.disconnect()

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

    async def _main_loop(self) -> None:
        """Main orchestration loop."""
        # Minimum interval between streaming snapshot rebuilds (throttle)
        streaming_throttle_sec = 0.5

        last_full_cycle = datetime.now()

        while self._running:
            try:
                # Wait for either:
                # 1. Streaming market data update (immediate)
                # 2. Timeout for full cycle (refresh_interval_sec)
                time_since_full_cycle = (datetime.now() - last_full_cycle).total_seconds()
                wait_time = max(0.1, self.refresh_interval_sec - time_since_full_cycle)

                try:
                    # Wait for streaming update or timeout
                    await asyncio.wait_for(
                        self._market_data_updated.wait(),
                        timeout=wait_time
                    )
                    # Streaming update received
                    self._market_data_updated.clear()

                    # Throttle streaming rebuilds
                    now = datetime.now()
                    if (
                        self._last_streaming_rebuild is None
                        or (now - self._last_streaming_rebuild).total_seconds() >= streaming_throttle_sec
                    ):
                        self._last_streaming_rebuild = now
                        await self._rebuild_snapshot_from_cache()

                except asyncio.TimeoutError:
                    # Timeout - time for full cycle
                    pass

                # Run full cycle at normal interval
                if (datetime.now() - last_full_cycle).total_seconds() >= self.refresh_interval_sec:
                    await self._run_cycle()
                    last_full_cycle = datetime.now()

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
        self.event_bus.publish(EventType.POSITION_CHANGED, {"count": len(merged_positions)})

        # Log position counts prominently (both to file and console)
        position_msg = (
            f"📊 Positions: IB={len(ib_positions)}, Futu={len(futu_positions)}, "
            f"Manual={len(manual_positions)} → Merged={len(merged_positions)}"
        )
        logger.info(position_msg)
        print(position_msg, flush=True)  # Ensure visible in console

        # 3. Fetch account info EARLY from all brokers (before slow market data fetch)
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

        account_msg = (
            f"💰 Account: IB=${ib_account.net_liquidation if ib_account else 0:,.0f}, "
            f"Futu=${futu_account.net_liquidation if futu_account else 0:,.0f}, "
            f"Total=${account_info.net_liquidation:,.0f}"
        )
        logger.info(account_msg)
        print(account_msg, flush=True)  # Ensure visible in console

        # Note: Early snapshot removed per architecture review - it duplicated work
        # and the final snapshot after market data fetch is the authoritative one

        # 4. Fetch market data (optimized: only fetch stale data)
        # Get symbols that need fresh market data (atomic operation to prevent race conditions)
        all_symbols = [p.symbol for p in merged_positions]
        symbols_needing_refresh = set(self.market_data_store.get_symbols_needing_refresh(all_symbols))
        positions_to_fetch = [p for p in merged_positions if p.symbol in symbols_needing_refresh]

        if positions_to_fetch:
            logger.debug(f"Fetching market data for {len(positions_to_fetch)}/{len(merged_positions)} positions (Greeks cache optimization)")
            try:
                if not self.ib_adapter or not self.ib_adapter.is_connected():
                    # IB not connected - mark as UNHEALTHY and continue with cached data
                    logger.error("IB adapter not connected - cannot fetch market data")
                    self.health_monitor.update_component_health(
                        "market_data_feed",
                        HealthStatus.UNHEALTHY,
                        "IB adapter not connected"
                    )
                else:
                    # Fetch market data from IBKR
                    market_data_list = await self.ib_adapter.fetch_market_data(positions_to_fetch)
                    self.market_data_store.upsert(market_data_list)

                    # Update health status based on fetch success
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
                # Market data fetch failed - mark as UNHEALTHY
                logger.error(f"Failed to fetch market data: {e}")
                self.health_monitor.update_component_health(
                    "market_data_feed",
                    HealthStatus.UNHEALTHY,
                    f"Fetch failed: {str(e)[:50]}"
                )
                # Continue with cached market data (no mock fallback)
                logger.warning("Continuing with cached market data - system may display stale data")
        else:
            logger.debug(f"All market data fresh (Greeks cache hit: {len(merged_positions)} positions)")
            # All data is cached - mark as healthy since we're using valid cached data
            self.health_monitor.update_component_health(
                "market_data_feed",
                HealthStatus.HEALTHY,
                "Using cached data (all fresh)"
            )

        # 4. Validate market data quality
        market_data = self.market_data_store.get_all()
        self.mdqc.validate_all(market_data)

        # 4b. Fetch market-wide indicators for alerts (e.g., VIX) and detect alerts
        await self._detect_market_alerts()

        # 5. Calculate risk metrics (account_info already fetched earlier)
        snapshot = self.risk_engine.build_snapshot(
            merged_positions, market_data, account_info,
            ib_account=ib_account, futu_account=futu_account
        )
        self._latest_snapshot = snapshot

        # Signal that new snapshot is ready (for dashboard sync)
        self._snapshot_ready.set()

        # 7. Evaluate risk limits (use RiskSignalEngine if available, else legacy RuleEngine)
        if self.risk_signal_engine:
            # New: Multi-layer risk signal engine
            risk_signals = self.risk_signal_engine.evaluate(snapshot)
            self._latest_risk_signals = risk_signals
            if risk_signals:
                logger.warning(f"Found {len(risk_signals)} risk signals")
                for signal in risk_signals:
                    self.event_bus.publish(EventType.LIMIT_BREACHED, signal)
        else:
            # Legacy: Portfolio rule engine only
            breaches = self.rule_engine.evaluate(snapshot)
            if breaches:
                logger.warning(f"Found {len(breaches)} limit breaches")
                for breach in breaches:
                    self.event_bus.publish(EventType.LIMIT_BREACHED, breach)

        # 7b. Log all alerts and signals to audit trail
        if self.risk_alert_logger:
            self._log_risk_alerts(snapshot)

        # 8. Update watchdog
        self.watchdog.update_snapshot_time(snapshot.timestamp)
        self.watchdog.check_missing_market_data(
            snapshot.total_positions, snapshot.positions_with_missing_md
        )

        logger.debug("Orchestration cycle completed")

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

        Updates the market data store and signals for snapshot rebuild.
        """
        # Update store
        self.market_data_store.upsert([market_data])

        # Signal main loop to rebuild snapshot (throttled there)
        self._market_data_updated.set()

    async def _rebuild_snapshot_from_cache(self) -> None:
        """
        Rebuild snapshot using cached positions and fresh market data.

        Called on streaming market data updates (throttled).
        Lighter than full _run_cycle - skips position fetching.
        """
        try:
            # Get cached positions and account
            positions = self.position_store.get_all()
            if not positions:
                return

            market_data = self.market_data_store.get_all()
            account_info = self.account_store.get()

            # Rebuild snapshot with fresh market data
            snapshot = self.risk_engine.build_snapshot(
                positions, market_data, account_info
            )
            self._latest_snapshot = snapshot

            # Signal dashboard
            self._snapshot_ready.set()

            logger.debug(f"Streaming snapshot rebuild: {len(positions)} positions")
        except Exception as e:
            logger.warning(f"Streaming snapshot rebuild failed: {e}")

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

