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
import logging
import sys

from config.config_manager import ConfigManager
from src.domain.services import SimpleSuggester
from src.infrastructure.adapters import IbAdapter, FileLoader
from src.infrastructure.stores import PositionStore, MarketDataStore, AccountStore
from src.infrastructure.monitoring import HealthMonitor, Watchdog
from src.domain.services.risk_engine import RiskEngine
from src.domain.services.reconciler import Reconciler
from src.domain.services.mdqc import MDQC
from src.domain.services.rule_engine import RuleEngine
# from src.domain.services.suggester import SimpleSuggester
# from src.domain.services.shock_engine import SimpleShockEngine
from src.application import Orchestrator, SimpleEventBus
from src.presentation import TerminalDashboard
from src.models.risk_snapshot import RiskSnapshot
from src.utils import StructuredLogger
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
        choices=["dev", "prod"],
        help="Environment to run in (default: dev)"
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
    # Set up category-based logging with environment prefix
    category_loggers = setup_category_logging(env=args.env, log_dir="./logs", level="INFO")
    system_logger = category_loggers["system"]
    market_logger = category_loggers["market"]

    # Create structured loggers for each category
    system_structured = StructuredLogger(system_logger)
    market_structured = StructuredLogger(market_logger)

    system_structured.info(
        LogCategory.SYSTEM,
        "Starting Live Risk Management System",
        {"env": args.env}
    )

    try:
        # Load configuration
        config_manager = ConfigManager(config_dir="config", env=args.env)
        config = config_manager.load()
        system_structured.info(LogCategory.SYSTEM, "Configuration loaded", {"env": args.env})

        # Initialize event bus
        event_bus = SimpleEventBus()

        # Initialize data stores
        position_store = PositionStore()
        market_data_store = MarketDataStore()
        account_store = AccountStore()

        # Initialize adapters
        ib_adapter = IbAdapter(
            host=config.ibkr.host,
            port=config.ibkr.port,
            client_id=config.ibkr.client_id,
        )

        file_loader = FileLoader(
            file_path=config.manual_positions.file,
            reload_interval_sec=config.manual_positions.reload_interval_sec,
        )

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
        # suggester = SimpleSuggester()  # TODO: Use for breach analysis in dashboard
        # shock_engine = SimpleShockEngine(risk_engine=risk_engine, config=config.raw)  # TODO: Add scenario analysis

        # Initialize monitoring
        health_monitor = HealthMonitor()
        watchdog = Watchdog(
            health_monitor=health_monitor,
            event_bus=event_bus,
            config=config.raw.get("watchdog", {}),
        )

        # Initialize orchestrator
        orchestrator = Orchestrator(
            ib_adapter=ib_adapter,
            file_loader=file_loader,
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
        )

        # Initialize dashboard (if not disabled)
        dashboard = None
        if not args.no_dashboard:
            dashboard = TerminalDashboard(config=config.raw.get("dashboard", {}))

        # Start orchestrator
        system_structured.info(LogCategory.SYSTEM, "Starting orchestrator")
        await orchestrator.start()

        # Start dashboard (blocking)
        if dashboard:
            dashboard.start()

            # Update loop
            try:
                while True:
                    snapshot = orchestrator.get_latest_snapshot()
                    health = health_monitor.get_all_health()

                    if snapshot:
                        breaches = rule_engine.evaluate(snapshot)
                        # Get positions and market data from stores
                        positions = position_store.get_all()
                        market_data = market_data_store.get_all()
                        dashboard.update(snapshot, breaches, health, positions, market_data)

                        # Log market data fetch
                        market_structured.info(
                            LogCategory.DATA,
                            "Market data refreshed",
                            {"positions": len(positions), "market_data": len(market_data)}
                        )
                    else:
                        # No snapshot yet - show empty snapshot with health status
                        empty_snapshot = RiskSnapshot()
                        dashboard.update(empty_snapshot, [], health, [], {})

                    await asyncio.sleep(config.dashboard.refresh_interval_sec)
            except KeyboardInterrupt:
                system_structured.info(LogCategory.SYSTEM, "Received shutdown signal")
            finally:
                dashboard.stop()
        else:
            # Headless mode - just wait for interrupt
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                system_structured.info(LogCategory.SYSTEM, "Received shutdown signal")

        # Shutdown
        await orchestrator.stop()
        system_structured.info(LogCategory.SYSTEM, "System shutdown complete")

    except Exception as e:
        system_structured.error(
            LogCategory.SYSTEM,
            "Fatal error",
            {"error": str(e)}
        )
        system_logger.exception("Fatal error:")
        sys.exit(1)


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
