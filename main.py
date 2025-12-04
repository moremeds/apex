"""
Live Risk Management System - Main Entry Point

Usage:
    python main.py --env dev          # Development mode
    python main.py --env prod         # Production mode
    python main.py --config custom.yaml  # Custom config file
"""

from __future__ import annotations
import asyncio
import argparse
import sys

from config.config_manager import ConfigManager
from src.domain.services import MarketAlertDetector
from src.infrastructure.adapters import IbAdapter, FutuAdapter, FileLoader, BrokerManager, MarketDataManager
from src.infrastructure.stores import PositionStore, MarketDataStore, AccountStore
from src.infrastructure.monitoring import HealthMonitor, Watchdog
from src.domain.services.risk.risk_engine import RiskEngine
from src.domain.services.pos_reconciler import Reconciler
from src.domain.services.mdqc import MDQC
from src.domain.services.risk.rule_engine import RuleEngine
from src.domain.services.risk.risk_signal_manager import RiskSignalManager
from src.domain.services.risk.risk_signal_engine import RiskSignalEngine
from src.domain.services.risk.risk_alert_logger import RiskAlertLogger
from src.application import Orchestrator, AsyncEventBus
from src.presentation import TerminalDashboard
from src.tui import TerminalDashboard
from src.models.risk_snapshot import RiskSnapshot
from src.utils import StructuredLogger, flush_all_loggers, set_log_timezone
from src.utils.structured_logger import LogCategory
from src.utils.logging_setup import setup_category_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Live Risk Management System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --env dev              # Run in development mode
  python main.py --env prod             # Run in production mode
  python main.py --config custom.yaml   # Use custom config file
        """
    )

    parser.add_argument(
        "--env",
        type=str,
        default="dev",
        choices=["dev", "prod", "demo"],
        help="Environment to run in (default: dev). Use 'demo' for offline mode with sample positions."
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom config file (overrides env-based loading)"
    )

    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable terminal dashboard (headless mode)"
    )

    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    """Main async entry point."""
    # Load configuration first to get timezone setting
    config_manager = ConfigManager(config_dir="config", env=args.env)
    config = config_manager.load()

    # Set timezone for logging (before creating loggers)
    log_tz = config.logging.timezone
    if log_tz and log_tz.lower() != "local":
        set_log_timezone(log_tz)
    else:
        set_log_timezone(None)  # Use local time

    # Set up category-based logging with environment prefix
    category_loggers = setup_category_logging(env=args.env, log_dir="./logs", level="DEBUG")
    system_logger = category_loggers["system"]
    market_logger = category_loggers["market"]

    # Create structured loggers for each category
    system_structured = StructuredLogger(system_logger)
    market_structured = StructuredLogger(market_logger)

    system_structured.info(
        LogCategory.SYSTEM,
        "Starting Live Risk Management System",
        {"env": args.env, "log_timezone": log_tz}
    )

    orchestrator = None
    dashboard = None

    try:
        system_structured.info(LogCategory.SYSTEM, "Configuration loaded", {"env": args.env})

        # Initialize event bus (always use AsyncEventBus)
        event_bus = AsyncEventBus()
        system_structured.info(LogCategory.SYSTEM, "Using AsyncEventBus")

        # Initialize data stores
        position_store = PositionStore()
        market_data_store = MarketDataStore()
        account_store = AccountStore()

        # Initialize monitoring (needed for BrokerManager)
        health_monitor = HealthMonitor()

        # Initialize BrokerManager to manage all broker connections
        broker_manager = BrokerManager(health_monitor=health_monitor)

        # Register adapters with BrokerManager
        if config.ibkr.enabled:
            ib_adapter = IbAdapter(
                host=config.ibkr.host,
                port=config.ibkr.port,
                client_id=config.ibkr.client_id,
                event_bus=event_bus,
            )
            broker_manager.register_adapter("ib", ib_adapter)
            system_structured.info(LogCategory.SYSTEM, "IB adapter registered", {
                "host": config.ibkr.host,
                "port": config.ibkr.port,
            })
        else:
            system_structured.info(LogCategory.SYSTEM, "IB adapter DISABLED (demo mode)")

        if config.futu.enabled:
            futu_adapter = FutuAdapter(
                host=config.futu.host,
                port=config.futu.port,
                security_firm=config.futu.security_firm,
                trd_env=config.futu.trd_env,
                filter_trdmarket=config.futu.filter_trdmarket,
                event_bus=event_bus,
            )
            broker_manager.register_adapter("futu", futu_adapter)
            system_structured.info(LogCategory.SYSTEM, "Futu adapter registered", {
                "host": config.futu.host,
                "port": config.futu.port,
                "market": config.futu.filter_trdmarket,
            })
        else:
            system_structured.info(LogCategory.SYSTEM, "Futu adapter DISABLED")

        # Register file loader for manual positions
        file_loader = FileLoader(
            file_path=config.manual_positions.file,
            reload_interval_sec=config.manual_positions.reload_interval_sec,
        )
        broker_manager.register_adapter("manual", file_loader)

        # Initialize MarketDataManager to manage all market data sources
        market_data_manager = MarketDataManager(health_monitor=health_monitor)

        # Register IB adapter as market data provider (if enabled)
        if config.ibkr.enabled:
            # IB adapter is already created above, register it for market data too
            market_data_manager.register_provider("ib", ib_adapter, priority=10)
            system_structured.info(LogCategory.SYSTEM, "IB registered as market data provider", {
                "streaming": ib_adapter.supports_streaming(),
                "greeks": ib_adapter.supports_greeks(),
            })
        else:
            system_structured.info(LogCategory.SYSTEM, "IB market data DISABLED (demo mode)")

        # TODO: Register additional market data providers here as needed
        # Example for future Yahoo Finance or CCXT providers:
        # if config.yahoo.enabled:
        #     yahoo_provider = YahooFinanceProvider(...)
        #     market_data_manager.register_provider("yahoo", yahoo_provider, priority=50)
        #
        # if config.ccxt.enabled:
        #     ccxt_provider = CcxtProvider(...)
        #     market_data_manager.register_provider("ccxt", ccxt_provider, priority=60)

        # Initialize domain services
        risk_engine = RiskEngine(config=config.raw)
        reconciler = Reconciler(stale_threshold_seconds=300)
        mdqc = MDQC(
            stale_seconds=config.mdqc.stale_seconds,
            ignore_zero_quotes=config.mdqc.ignore_zero_quotes,
            enforce_bid_ask_sanity=config.mdqc.enforce_bid_ask_sanity,
        )
        rule_engine = RuleEngine(
            risk_limits=config.raw.get("risk_limits", {}),
            soft_threshold=config.risk_limits.soft_breach_threshold,
        )
        market_alert_detector = MarketAlertDetector(config.raw.get("market_alerts", {}))

        # Initialize risk signal engine (Phase 1-4: Multi-layer risk detection)
        signal_manager = RiskSignalManager(
            debounce_seconds=config.raw.get("risk_signals", {}).get("debounce_seconds", 15),
            cooldown_minutes=config.raw.get("risk_signals", {}).get("cooldown_minutes", 5),
        )
        risk_signal_engine = RiskSignalEngine(
            config=config.raw,
            rule_engine=rule_engine,
            signal_manager=signal_manager,
        )
        system_structured.info(LogCategory.SYSTEM, "Risk signal engine initialized")

        # Initialize risk alert logger for audit trail
        risk_alert_logger = RiskAlertLogger(
            log_dir="./logs",
            env=args.env,
            retention_days=config.raw.get("risk_alerts", {}).get("retention_days", 30),
        )
        system_structured.info(LogCategory.SYSTEM, "Risk alert logger initialized", {
            "log_dir": "./logs",
            "retention_days": config.raw.get("risk_alerts", {}).get("retention_days", 30),
        })

        # suggester = SimpleSuggester()  # TODO: Use for breach analysis in dashboard
        # shock_engine = SimpleShockEngine(risk_engine=risk_engine, config=config.raw)  # TODO: Add scenario analysis

      watchdog = Watchdog(
            health_monitor=health_monitor,
            event_bus=event_bus,
            config=config.raw.get("watchdog", {}),
        )

        # Initialize orchestrator with BrokerManager and MarketDataManager
        orchestrator = Orchestrator(
            broker_manager=broker_manager,
            market_data_manager=market_data_manager,
            position_store=position_store,
            market_data_store=market_data_store,
            account_store=account_store,
            risk_engine=risk_engine,
            reconciler=reconciler,
            mdqc=mdqc,
            rule_engine=rule_engine,
            health_monitor=health_monitor,
            watchdog=watchdog,
            event_bus=event_bus,
            config=config.raw,
            market_alert_detector=market_alert_detector,
            risk_signal_engine=risk_signal_engine,
            risk_alert_logger=risk_alert_logger,
        )

        # Initialize dashboard (if not disabled)
        if not args.no_dashboard:
            dashboard = TerminalDashboard(
                config=config.raw.get("dashboard", {}),
                env=args.env,
            )

        # Start dashboard FIRST (non-blocking) so it shows immediately
        if dashboard:
            dashboard.start()
            # Show initial empty state while data loads
            empty_snapshot = RiskSnapshot()
            dashboard.update(empty_snapshot, [], [], [])

        # Start orchestrator (event-driven mode is the default and only mode)
        # Data fetching happens in background, dashboard updates when ready
        system_structured.info(LogCategory.SYSTEM, "Starting orchestrator")
        await orchestrator.start()

        # Debug: Check health components after orchestrator start
        initial_health = health_monitor.get_all_health()
        system_structured.info(
            LogCategory.SYSTEM,
            f"Health components after orchestrator start: {len(initial_health)}",
            {"components": [h.component_name for h in initial_health]}
        )

        # Update loop - event-driven sync with orchestrator
        if dashboard:
            try:
                while True:
                    # Wait for orchestrator to signal new snapshot (instead of polling)
                    snapshot = await orchestrator.wait_for_snapshot(timeout=3.0)
                    health = health_monitor.get_all_health()

                    # Debug: Log health component count
                    if len(health) < 4:
                        system_structured.warning(
                            LogCategory.SYSTEM,
                            f"Health components incomplete: {len(health)}/4",
                            {"components": [h.component_name for h in health]}
                        )

                    if snapshot:
                        # Get risk signals from orchestrator (multi-layer risk detection)
                        risk_signals = orchestrator.get_latest_risk_signals()

                        # Use market alert detector output from orchestrator
                        market_alerts = orchestrator.get_latest_market_alerts()

                        # Dashboard now uses pre-calculated data from snapshot.position_risks
                        # and displays RiskSignals instead of legacy breaches
                        dashboard.update(snapshot, risk_signals, health, market_alerts)

                        # Log market data fetch
                        market_structured.info(
                            LogCategory.DATA,
                            "Risk snapshot refreshed",
                            {"positions": snapshot.total_positions, "position_risks": len(snapshot.position_risks)}
                        )
                    else:
                        # No snapshot yet - show empty snapshot with health status
                        empty_snapshot = RiskSnapshot()
                        dashboard.update(empty_snapshot, [], health, [])
            except KeyboardInterrupt:
                system_structured.info(LogCategory.SYSTEM, "Received shutdown signal")
        else:
            # Headless mode - just wait for interrupt
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                system_structured.info(LogCategory.SYSTEM, "Received shutdown signal")

        system_structured.info(LogCategory.SYSTEM, "System shutdown complete")

    except Exception as e:
        system_structured.error(
            LogCategory.SYSTEM,
            "Fatal error",
            {"error": str(e)}
        )
        system_logger.exception("Fatal error:")
    finally:
        # Always clean up resources
        if dashboard:
            dashboard.stop()
        if orchestrator:
            await orchestrator.stop()
            system_structured.info(LogCategory.SYSTEM, "Orchestrator stopped")

        # Ensure all logs are flushed to disk
        flush_all_loggers()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("Shutdown requested")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
