"""
Application Bootstrap - Composition Root for Service Wiring.

This module provides the AppContainer class that encapsulates all service
instantiation and dependency injection, making the initialization order
explicit and testable.

Usage:
    container = AppContainer(config, args)
    await container.initialize()
    orchestrator = container.orchestrator
    dashboard = container.dashboard
    # ... run application ...
    await container.cleanup()
"""

from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..domain.events import PriorityEventBus
from ..domain.services import MarketAlertDetector
from ..domain.services.risk.risk_engine import RiskEngine
from ..domain.services.risk.risk_facade import RiskFacade
from ..domain.services.risk.rule_engine import RuleEngine
from ..domain.services.risk.risk_signal_manager import RiskSignalManager
from ..domain.services.risk.risk_signal_engine import RiskSignalEngine
from ..domain.services.risk.risk_alert_logger import RiskAlertLogger
from ..domain.services.risk.streaming import DeltaPublisher, ShadowValidator
from ..domain.services.pos_reconciler import Reconciler
from ..domain.services.mdqc import MDQC
from ..infrastructure.adapters import FutuAdapter, FileLoader, BrokerManager, MarketDataManager
from ..infrastructure.adapters.ib import IbConnectionPool, ConnectionPoolConfig, IbCompositeAdapter
from ..infrastructure.stores import PositionStore, MarketDataStore, AccountStore
from ..infrastructure.stores.duckdb_coverage_store import DuckDBCoverageStore
from ..infrastructure.persistence.database import Database
from ..infrastructure.persistence.repositories.ta_signal_repository import TASignalRepository
from ..infrastructure.monitoring import HealthMonitor, Watchdog
from ..services.historical_data_manager import HistoricalDataManager
from ..services.bar_persistence_service import BarPersistenceService
from ..utils import StructuredLogger
from ..utils.structured_logger import LogCategory

from . import Orchestrator, ReadinessManager

if TYPE_CHECKING:
    from config.config_manager import Config

# Observability imports (optional)
try:
    from ..infrastructure.observability import (
        MetricsManager, get_metrics_manager, RiskMetrics, HealthMetrics, SignalMetrics
    )
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False
    MetricsManager = None  # type: ignore
    get_metrics_manager = None  # type: ignore
    RiskMetrics = None  # type: ignore
    HealthMetrics = None  # type: ignore
    SignalMetrics = None  # type: ignore


@dataclass
class AppContainer:
    """
    Composition root for all application services.

    Manages service lifecycle: creation, wiring, and cleanup.
    Makes the 14-step initialization order explicit and testable.

    Attributes:
        config: Application configuration.
        env: Environment name (dev, prod, demo).
        metrics_port: Port for Prometheus metrics (0 to disable).
        no_dashboard: Whether to run in headless mode.
    """

    config: Any  # Config object from ConfigManager
    env: str
    metrics_port: int = 8000
    no_dashboard: bool = False

    # Core infrastructure (created during initialize)
    event_bus: Optional[PriorityEventBus] = field(default=None, init=False)
    position_store: Optional[PositionStore] = field(default=None, init=False)
    market_data_store: Optional[MarketDataStore] = field(default=None, init=False)
    account_store: Optional[AccountStore] = field(default=None, init=False)
    health_monitor: Optional[HealthMonitor] = field(default=None, init=False)

    # Managers
    broker_manager: Optional[BrokerManager] = field(default=None, init=False)
    market_data_manager: Optional[MarketDataManager] = field(default=None, init=False)
    historical_data_manager: Optional[HistoricalDataManager] = field(default=None, init=False)

    # Domain services
    risk_engine: Optional[RiskEngine] = field(default=None, init=False)
    risk_facade: Optional[RiskFacade] = field(default=None, init=False)
    delta_publisher: Optional[DeltaPublisher] = field(default=None, init=False)
    shadow_validator: Optional[ShadowValidator] = field(default=None, init=False)
    reconciler: Optional[Reconciler] = field(default=None, init=False)
    mdqc: Optional[MDQC] = field(default=None, init=False)
    rule_engine: Optional[RuleEngine] = field(default=None, init=False)
    risk_signal_engine: Optional[RiskSignalEngine] = field(default=None, init=False)
    risk_alert_logger: Optional[RiskAlertLogger] = field(default=None, init=False)
    market_alert_detector: Optional[MarketAlertDetector] = field(default=None, init=False)
    watchdog: Optional[Watchdog] = field(default=None, init=False)
    readiness_manager: Optional[ReadinessManager] = field(default=None, init=False)

    # Application layer
    orchestrator: Optional[Orchestrator] = field(default=None, init=False)

    # Persistence
    db: Optional[Database] = field(default=None, init=False)
    signal_repo: Optional[TASignalRepository] = field(default=None, init=False)
    coverage_store: Optional[DuckDBCoverageStore] = field(default=None, init=False)
    bar_persistence_service: Optional[BarPersistenceService] = field(default=None, init=False)

    # Observability
    metrics_manager: Optional[Any] = field(default=None, init=False)
    risk_metrics: Optional[Any] = field(default=None, init=False)
    health_metrics: Optional[Any] = field(default=None, init=False)
    signal_metrics: Optional[Any] = field(default=None, init=False)

    # IB-specific
    ib_pool: Optional[IbConnectionPool] = field(default=None, init=False)

    # Internal state
    _logger: Optional[StructuredLogger] = field(default=None, init=False)
    _initialized: bool = field(default=False, init=False)

    async def initialize(self, logger: StructuredLogger) -> None:
        """
        Initialize all services in correct order.

        This is the explicit initialization sequence that was previously
        implicit in main.py. The order matters for dependency resolution.

        Args:
            logger: Structured logger for initialization messages.
        """
        if self._initialized:
            raise RuntimeError("AppContainer already initialized")

        self._logger = logger

        # Phase 1: Core Infrastructure
        self._create_event_bus()
        self._create_observability()
        self._create_data_stores()
        self._create_health_monitor()

        # Phase 2: Adapters and Managers
        self._create_broker_manager()
        self._create_market_data_manager()
        await self._register_adapters()

        # Phase 3: Domain Services
        self._create_domain_services()
        self._create_risk_services()
        self._create_streaming_risk_service()
        self._create_readiness_manager()

        # Phase 4: Historical Data
        self._create_historical_data_manager()
        self._create_bar_persistence_service()

        # Phase 5: Database Persistence
        await self._create_database()

        # Phase 6: Orchestrator (depends on all above)
        self._create_orchestrator()

        # Phase 7: Coverage Store (for TUI Tab 7)
        self._create_coverage_store()

        self._initialized = True
        self._log(LogCategory.SYSTEM, "AppContainer initialization complete")

    def _log(self, category: LogCategory, message: str, extra: Optional[Dict] = None) -> None:
        """Log a message if logger is available."""
        if self._logger:
            self._logger.info(category, message, extra or {})

    def _create_event_bus(self) -> None:
        """Phase 1a: Create priority event bus."""
        self.event_bus = PriorityEventBus()
        self._log(LogCategory.SYSTEM, "PriorityEventBus created (dual-lane)")

    def _create_observability(self) -> None:
        """Phase 1b: Create observability components (optional)."""
        if not OBSERVABILITY_AVAILABLE or self.metrics_port <= 0:
            if not OBSERVABILITY_AVAILABLE:
                self._log(LogCategory.SYSTEM, "Observability not available")
            else:
                self._log(LogCategory.SYSTEM, "Observability disabled (metrics_port=0)")
            return

        try:
            self.metrics_manager = get_metrics_manager(port=self.metrics_port)
            self.metrics_manager.start()

            meter = self.metrics_manager.get_meter("apex")
            self.risk_metrics = RiskMetrics(meter)
            self.health_metrics = HealthMetrics(meter)
            self.signal_metrics = SignalMetrics(meter)

            # Wire perf logger
            from ..utils.perf_logger import set_perf_metrics
            set_perf_metrics(health_metrics=self.health_metrics, risk_metrics=self.risk_metrics)

            self._log(LogCategory.SYSTEM, "Observability enabled", {
                "metrics_port": self.metrics_port,
                "endpoint": f"http://localhost:{self.metrics_port}/metrics"
            })
        except Exception as e:
            self._log(LogCategory.SYSTEM, f"Failed to start metrics server: {e}")

    def _create_data_stores(self) -> None:
        """Phase 1c: Create data stores."""
        self.position_store = PositionStore()
        self.market_data_store = MarketDataStore()
        self.account_store = AccountStore()
        self._log(LogCategory.SYSTEM, "Data stores created")

    def _create_health_monitor(self) -> None:
        """Phase 1d: Create health monitor."""
        self.health_monitor = HealthMonitor()
        self._log(LogCategory.SYSTEM, "HealthMonitor created")

    def _create_broker_manager(self) -> None:
        """Phase 2a: Create broker manager."""
        self.broker_manager = BrokerManager(health_monitor=self.health_monitor)
        self._log(LogCategory.SYSTEM, "BrokerManager created")

    def _create_market_data_manager(self) -> None:
        """Phase 2b: Create market data manager."""
        self.market_data_manager = MarketDataManager(
            health_monitor=self.health_monitor,
            event_bus=self.event_bus,
        )
        self._log(LogCategory.SYSTEM, "MarketDataManager created")

    async def _register_adapters(self) -> None:
        """Phase 2c: Register broker and market data adapters."""
        # IB Adapter
        if self.config.ibkr.enabled:
            pool_config = ConnectionPoolConfig(
                host=self.config.ibkr.host,
                port=self.config.ibkr.port,
                client_ids=self.config.ibkr.client_ids,
            )

            ib_adapter = IbCompositeAdapter(
                pool_config=pool_config,
                event_bus=self.event_bus,
            )

            self.broker_manager.register_adapter("ib", ib_adapter)
            self.market_data_manager.register_provider("ib", ib_adapter, priority=10)

            # Create IB pool for historical data
            self.ib_pool = IbConnectionPool(pool_config)

            self._log(LogCategory.SYSTEM, "IB composite adapter registered", {
                "host": self.config.ibkr.host,
                "port": self.config.ibkr.port,
            })
        else:
            self._log(LogCategory.SYSTEM, "IB adapter DISABLED")

        # Futu Adapter
        if self.config.futu.enabled:
            futu_adapter = FutuAdapter(
                host=self.config.futu.host,
                port=self.config.futu.port,
                security_firm=self.config.futu.security_firm,
                trd_env=self.config.futu.trd_env,
                filter_trading_market=self.config.futu.filter_trdmarket,
                event_bus=self.event_bus,
            )
            self.broker_manager.register_adapter("futu", futu_adapter)
            self._log(LogCategory.SYSTEM, "Futu adapter registered", {
                "host": self.config.futu.host,
                "port": self.config.futu.port,
            })
        else:
            self._log(LogCategory.SYSTEM, "Futu adapter DISABLED")

        # Manual positions file loader
        file_loader = FileLoader(
            file_path=self.config.manual_positions.file,
            reload_interval_sec=self.config.manual_positions.reload_interval_sec,
        )
        self.broker_manager.register_adapter("manual", file_loader)
        self._log(LogCategory.SYSTEM, "Manual positions loader registered")

    def _create_domain_services(self) -> None:
        """Phase 3a: Create domain services."""
        self.risk_engine = RiskEngine(
            config=self.config.raw,
            yahoo_adapter=None,  # Disabled for performance
            risk_metrics=self.risk_metrics,
        )

        self.reconciler = Reconciler(stale_threshold_seconds=300)

        self.mdqc = MDQC(
            stale_seconds=self.config.mdqc.stale_seconds,
            ignore_zero_quotes=self.config.mdqc.ignore_zero_quotes,
            enforce_bid_ask_sanity=self.config.mdqc.enforce_bid_ask_sanity,
        )

        self.rule_engine = RuleEngine(
            risk_limits=self.config.raw.get("risk_limits", {}),
            soft_threshold=self.config.risk_limits.soft_breach_threshold,
        )

        self.market_alert_detector = MarketAlertDetector(
            self.config.raw.get("market_alerts", {})
        )

        self.watchdog = Watchdog(
            health_monitor=self.health_monitor,
            event_bus=self.event_bus,
            config=self.config.raw.get("watchdog", {}),
        )

        self._log(LogCategory.SYSTEM, "Domain services created")

    def _create_risk_services(self) -> None:
        """Phase 3b: Create risk signal engine and alert logger."""
        signal_manager = RiskSignalManager(
            debounce_seconds=self.config.raw.get("risk_signals", {}).get("debounce_seconds", 15),
            cooldown_minutes=self.config.raw.get("risk_signals", {}).get("cooldown_minutes", 5),
        )

        self.risk_signal_engine = RiskSignalEngine(
            config=self.config.raw,
            rule_engine=self.rule_engine,
            signal_manager=signal_manager,
        )

        self.risk_alert_logger = RiskAlertLogger(
            log_dir="./logs",
            env=self.env,
            retention_days=self.config.raw.get("risk_alerts", {}).get("retention_days", 30),
        )

        self._log(LogCategory.SYSTEM, "Risk services created")

    def _create_streaming_risk_service(self) -> None:
        """
        Phase 3d: Create streaming risk components for low-latency TUI updates.

        Creates RiskFacade, DeltaPublisher, and ShadowValidator for the streaming
        hot path. Runs in parallel with existing RiskEngine during shadow mode.
        """
        # Create RiskFacade (manages PortfolioState and TickProcessor internally)
        self.risk_facade = RiskFacade()

        # Create DeltaPublisher (bridges RiskFacade to event bus)
        self.delta_publisher = DeltaPublisher(
            risk_facade=self.risk_facade,
            event_bus=self.event_bus,
            position_store=self.position_store,
        )

        # Create ShadowValidator (compares streaming vs batch calculations)
        self.shadow_validator = ShadowValidator(
            risk_facade=self.risk_facade,
            event_bus=self.event_bus,
        )

        self._log(LogCategory.SYSTEM, "Streaming risk service created (shadow mode)")

    def _create_readiness_manager(self) -> None:
        """Phase 3c: Create readiness manager."""
        required_brokers = []
        if self.config.ibkr.enabled:
            required_brokers.append("ib")
        if self.config.futu.enabled:
            required_brokers.append("futu")
        required_brokers.append("manual")

        self.readiness_manager = ReadinessManager(
            event_bus=self.event_bus,
            required_brokers=required_brokers,
            market_data_coverage_threshold=1.0,
            startup_timeout_sec=self.config.raw.get("dashboard", {}).get("snapshot_ready_timeout_sec", 30.0),
        )

        self._log(LogCategory.SYSTEM, "ReadinessManager created", {
            "required_brokers": required_brokers,
        })

    def _create_historical_data_manager(self) -> None:
        """Phase 4a: Create historical data manager."""
        historical_cfg = self.config.raw.get("historical_data", {})
        storage_cfg = historical_cfg.get("storage", {})

        self.historical_data_manager = HistoricalDataManager(
            base_dir=Path(storage_cfg.get("base_dir", "data/historical")),
            source_priority=["ib", "yahoo"],
        )

        self._log(LogCategory.SYSTEM, "HistoricalDataManager created")

    def _create_bar_persistence_service(self) -> None:
        """Phase 4b: Create bar persistence service."""
        if self.historical_data_manager and self.historical_data_manager._bar_store:
            self.bar_persistence_service = BarPersistenceService(
                event_bus=self.event_bus,
                bar_store=self.historical_data_manager._bar_store,
                flush_threshold_bars=10,
                flush_threshold_sec=60.0,
            )
            self._log(LogCategory.DATA, "BarPersistenceService created")

    async def _create_database(self) -> None:
        """Phase 5: Create database connection and signal repository."""
        if self.config.database.type == "disabled":
            self._log(LogCategory.DATA, "Database DISABLED")
            return

        try:
            self.db = Database(self.config.database)
            await self.db.connect()
            self.signal_repo = TASignalRepository(self.db)
            self._log(LogCategory.DATA, "Signal persistence connected", {
                "database": self.config.database.database,
                "host": self.config.database.host,
            })
        except Exception as e:
            self._log(LogCategory.DATA, f"Signal persistence unavailable: {e}")

    def _create_orchestrator(self) -> None:
        """Phase 6: Create orchestrator (depends on all above)."""
        self.orchestrator = Orchestrator(
            broker_manager=self.broker_manager,
            market_data_manager=self.market_data_manager,
            position_store=self.position_store,
            market_data_store=self.market_data_store,
            account_store=self.account_store,
            risk_engine=self.risk_engine,
            reconciler=self.reconciler,
            mdqc=self.mdqc,
            rule_engine=self.rule_engine,
            health_monitor=self.health_monitor,
            watchdog=self.watchdog,
            event_bus=self.event_bus,
            config=self.config.raw,
            market_alert_detector=self.market_alert_detector,
            risk_signal_engine=self.risk_signal_engine,
            risk_alert_logger=self.risk_alert_logger,
            risk_metrics=self.risk_metrics,
            health_metrics=self.health_metrics,
            readiness_manager=self.readiness_manager,
            signal_metrics=self.signal_metrics,
            historical_data_manager=self.historical_data_manager,
            signal_persistence=self.signal_repo,
            risk_facade=self.risk_facade,
            delta_publisher=self.delta_publisher,
            shadow_validator=self.shadow_validator,
        )

        self._log(LogCategory.SYSTEM, "Orchestrator created")

    def _create_coverage_store(self) -> None:
        """Phase 7: Create DuckDB coverage store for TUI Tab 7."""
        self.coverage_store = DuckDBCoverageStore()
        self._log(LogCategory.DATA, "DuckDB coverage store created")

    async def start(self) -> None:
        """Start all services that need explicit startup."""
        if not self._initialized:
            raise RuntimeError("AppContainer not initialized")

        # Start bar persistence first (to capture all BAR_CLOSE events)
        if self.bar_persistence_service:
            self.bar_persistence_service.start()

        # Start orchestrator
        self._log(LogCategory.SYSTEM, "Starting orchestrator")
        await self.orchestrator.start()

        # Register IB historical source after broker is connected
        if self.config.ibkr.enabled:
            ib_adapter = self.broker_manager.get_adapter("ib")
            if ib_adapter and ib_adapter.is_connected():
                if hasattr(ib_adapter, '_historical_adapter') and ib_adapter._historical_adapter:
                    self.historical_data_manager.set_ib_source(ib_adapter._historical_adapter)
                    self._log(LogCategory.SYSTEM, "IB historical source registered")

    async def cleanup(self) -> None:
        """Clean up all resources in reverse order."""
        self._log(LogCategory.SYSTEM, "Starting cleanup")

        # Stop orchestrator
        if self.orchestrator:
            await self.orchestrator.stop()
            self._log(LogCategory.SYSTEM, "Orchestrator stopped")

        # Stop bar persistence
        if self.bar_persistence_service:
            self.bar_persistence_service.stop()
            self._log(LogCategory.DATA, "BarPersistenceService stopped")

        # Disconnect IB pool
        if self.ib_pool:
            await self.ib_pool.disconnect()
            self._log(LogCategory.SYSTEM, "IB connection pool disconnected")

        # Close database
        if self.db:
            await self.db.close()
            self._log(LogCategory.SYSTEM, "Database connection closed")

        # Shutdown metrics
        if self.metrics_manager:
            self.metrics_manager.shutdown()
            self._log(LogCategory.SYSTEM, "Metrics server stopped")

        self._log(LogCategory.SYSTEM, "Cleanup complete")

    def get_required_brokers(self) -> List[str]:
        """Get list of required broker names for readiness checking."""
        required = []
        if self.config.ibkr.enabled:
            required.append("ib")
        if self.config.futu.enabled:
            required.append("futu")
        required.append("manual")
        return required
