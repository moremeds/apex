"""
Live Risk Management System - Main Entry Point

Usage:
    python orchestrator.py --env dev          # Development mode
    python orchestrator.py --env prod         # Production mode
    python orchestrator.py --config custom.yaml  # Custom config file
"""

from __future__ import annotations
import asyncio
import argparse
import os
import sys
from pathlib import Path

# Change to project root directory (where main.py is located)
# This ensures config files are found regardless of where script is invoked from
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

from config.config_manager import ConfigManager
from src.services import HistoricalDataService, TAService
from src.application import AppContainer
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
  python orchestrator.py --env dev              # Run in development mode (monitor)
  python orchestrator.py --mode monitor         # Explicit monitor mode
  python orchestrator.py --mode backtest --spec config/backtest/ma_cross.yaml
  python orchestrator.py --mode backtest --engine backtrader --strategy ma_cross
        """
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="monitor",
        choices=["monitor", "backtest", "trading"],
        help="Operational mode: monitor (default), backtest, or trading"
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

    parser.add_argument(
        "--metrics-port",
        type=int,
        default=8000,
        help="Port for Prometheus /metrics endpoint (default: 8000, 0 to disable)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level for all categories)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set log level (default: INFO, ignored if --verbose is set)"
    )

    # Backtest mode options
    backtest_group = parser.add_argument_group("Backtest Mode")
    backtest_group.add_argument(
        "--engine",
        type=str,
        default="apex",
        choices=["apex", "backtrader"],
        help="Backtest engine: apex (default) or backtrader"
    )
    backtest_group.add_argument(
        "--spec",
        type=str,
        help="Path to backtest spec YAML file (e.g., config/backtest/ma_cross.yaml)"
    )
    backtest_group.add_argument(
        "--strategy",
        type=str,
        help="Strategy name (if not using --spec)"
    )
    backtest_group.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated symbols (if not using --spec)"
    )
    backtest_group.add_argument(
        "--start",
        type=str,
        help="Start date YYYY-MM-DD (if not using --spec)"
    )
    backtest_group.add_argument(
        "--end",
        type=str,
        help="End date YYYY-MM-DD (if not using --spec)"
    )
    backtest_group.add_argument(
        "--capital",
        type=float,
        default=100000.0,
        help="Initial capital (default: 100000)"
    )

    # Trading mode options
    trading_group = parser.add_argument_group("Trading Mode")
    trading_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Paper trading mode (log orders but don't execute)"
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

    # Set up category-based logging
    category_loggers = setup_category_logging(
        env=args.env,
        log_dir="./logs",
        level=args.log_level,
        console=args.no_dashboard,
        verbose=args.verbose,
    )
    system_logger = category_loggers["system"]
    data_logger = category_loggers["data"]

    system_structured = StructuredLogger(system_logger)
    data_structured = StructuredLogger(data_logger)

    system_structured.info(
        LogCategory.SYSTEM,
        "Starting Live Risk Management System",
        {"env": args.env, "log_timezone": log_tz}
    )

    # Create application container (composition root)
    container = AppContainer(
        config=config,
        env=args.env,
        metrics_port=args.metrics_port,
        no_dashboard=args.no_dashboard,
    )

    dashboard = None
    historical_service = None
    ta_service = None

    try:
        # Initialize all services via container
        await container.initialize(system_structured)

        # Initialize dashboard (if not disabled)
        if not args.no_dashboard:
            dashboard_config = config.raw.get("dashboard", {})
            dashboard_config["display_tz"] = config.display.timezone if config.display else "Asia/Hong_Kong"
            dashboard = TerminalDashboard(
                config=dashboard_config,
                env=args.env,
            )

        # Wire dashboard to container services
        if dashboard:
            event_loop = asyncio.get_event_loop()
            dashboard.set_event_bus(container.event_bus, event_loop)
            dashboard.set_coverage_store(container.coverage_store)
            if container.signal_repo:
                dashboard.set_signal_persistence(container.signal_repo)

        # Start container services
        await container.start()

        # Debug: Check health components after start
        initial_health = container.health_monitor.get_all_health()
        system_structured.info(
            LogCategory.SYSTEM,
            f"Health components after start: {len(initial_health)}",
            {"components": [h.component_name for h in initial_health]}
        )

        # Set trading universe for dashboard
        if dashboard:
            positions = container.position_store.get_all()
            underlyings = {
                pos.underlying if pos.underlying else pos.symbol
                for pos in positions if pos.underlying or pos.symbol
            }

            if underlyings:
                universe_symbols = list(underlyings)
            else:
                universe_symbols = ["AAPL", "TSLA", "NVDA", "AMD", "MSFT", "GOOGL", "AMZN", "META"]
                system_structured.info(LogCategory.DATA, "No positions found, using sample symbols")

            dashboard.set_trading_universe(universe_symbols)
            system_structured.info(
                LogCategory.DATA,
                f"Trading universe set: {len(universe_symbols)} symbols",
                {"symbols": universe_symbols[:10]}
            )

        # Background task: Connect historical IB and pre-fetch daily bars
        prefetch_task = None
        if container.ib_pool:
            async def connect_historical_and_prefetch():
                """Background task to connect historical IB and pre-fetch bars."""
                try:
                    await container.ib_pool.connect_historical()
                    system_structured.info(
                        LogCategory.SYSTEM,
                        "IB historical connection established",
                        {"client_id": config.ibkr.client_ids.historical_pool[0]}
                    )

                    nonlocal historical_service, ta_service
                    historical_service = HistoricalDataService(
                        ib_historical=container.ib_pool.historical,
                        cache_size=512,
                        default_daily_lookback=60,
                    )
                    ta_service = TAService(historical_service)

                    if dashboard:
                        event_loop = asyncio.get_event_loop()
                        dashboard.set_ta_service(
                            ta_service, event_loop, historical_service, container.event_bus
                        )

                    # Pre-fetch daily bars for positions
                    positions = container.position_store.get_all()
                    underlyings = {
                        pos.underlying if pos.underlying else pos.symbol
                        for pos in positions if pos.underlying or pos.symbol
                    }

                    if underlyings:
                        symbols = list(underlyings)
                        system_structured.info(
                            LogCategory.DATA,
                            f"Pre-fetching 60d daily bars for {len(symbols)} symbols",
                            {"symbols": symbols[:10]}
                        )
                        await historical_service.prefetch_daily_bars(symbols, lookback_days=60)
                        system_structured.info(LogCategory.DATA, "Daily bars pre-fetch complete")
                except Exception as e:
                    system_structured.warning(LogCategory.SYSTEM, f"Historical data setup failed: {e}")

            prefetch_task = asyncio.create_task(connect_historical_and_prefetch())

        # Update loop - feeds data from orchestrator to dashboard
        update_task = None
        if dashboard:
            async def update_loop():
                """Background task that feeds orchestrator data to dashboard queue."""
                while dashboard.running:
                    try:
                        snapshot = await container.orchestrator.wait_for_snapshot(timeout=3.0)
                        health = container.health_monitor.get_all_health()

                        if snapshot:
                            risk_signals = container.orchestrator.get_latest_risk_signals()
                            market_alerts = container.orchestrator.get_latest_market_alerts()
                            dashboard.update(snapshot, risk_signals, health, market_alerts)
                            data_structured.info(
                                LogCategory.DATA,
                                "Risk snapshot refreshed",
                                {"positions": snapshot.total_positions, "position_risks": len(snapshot.position_risks)}
                            )
                        else:
                            preview = container.orchestrator.get_positions_preview()
                            if preview and preview.total_positions > 0:
                                dashboard.update(preview, [], health, [])
                            else:
                                dashboard.update(RiskSnapshot(), [], health, [])
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        system_structured.warning(LogCategory.SYSTEM, f"Update loop error: {e}")
                        await asyncio.sleep(1)

            update_task = asyncio.create_task(update_loop())

            try:
                await dashboard.run_async()
            except KeyboardInterrupt:
                system_structured.info(LogCategory.SYSTEM, "Received shutdown signal")
            finally:
                if update_task:
                    update_task.cancel()
                    try:
                        await update_task
                    except asyncio.CancelledError:
                        pass
        else:
            # Headless mode
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                system_structured.info(LogCategory.SYSTEM, "Received shutdown signal")

        system_structured.info(LogCategory.SYSTEM, "System shutdown complete")

    except Exception as e:
        system_structured.error(LogCategory.SYSTEM, "Fatal error", {"error": str(e)})
        system_logger.exception("Fatal error:")
    finally:
        if dashboard:
            dashboard.stop()
        await container.cleanup()
        flush_all_loggers()


async def run_backtest(args: argparse.Namespace) -> int:
    """
    Run backtest mode.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 for success/profitable, 1 for failure/unprofitable).
    """
    from src.runners.backtest_runner import BacktestRunner, BacktraderRunner, BACKTRADER_AVAILABLE
    from src.domain.strategy.registry import list_strategies

    # Configure logging
    import logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Validate required args (if not using spec file)
    if not args.spec:
        if not args.strategy:
            print("Error: --strategy or --spec required for backtest mode")
            print(f"Available strategies: {list_strategies()}")
            return 1
        if not args.symbols:
            print("Error: --symbols required for backtest mode")
            return 1
        if not args.start or not args.end:
            print("Error: --start and --end required for backtest mode")
            return 1

    try:
        # Handle engine selection
        if args.engine == "backtrader":
            if not BACKTRADER_AVAILABLE:
                print("Error: backtrader not installed. Run: pip install backtrader")
                print("Falling back to apex engine.")
                runner = BacktestRunner.from_args(args)
            else:
                print("Using Backtrader engine")
                runner = BacktraderRunner.from_args(args)
        else:
            # Default: apex engine
            runner = BacktestRunner.from_args(args)

        result = await runner.run()
        return 0 if result.is_profitable else 1
    except Exception as e:
        print(f"Backtest failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


async def run_trading(args: argparse.Namespace) -> int:
    """
    Run trading mode.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code.
    """
    from src.runners.trading_runner import TradingRunner
    from src.domain.strategy.registry import list_strategies

    # Configure logging
    import logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Validate required args
    if not args.strategy:
        print("Error: --strategy required for trading mode")
        print(f"Available strategies: {list_strategies()}")
        return 1
    if not args.symbols:
        print("Error: --symbols required for trading mode")
        return 1

    try:
        # Default to dry-run unless explicitly disabled
        dry_run = not getattr(args, "live", False)

        runner = TradingRunner(
            strategy_name=args.strategy,
            symbols=[s.strip() for s in args.symbols.split(",")],
            broker=getattr(args, "broker", "ib"),
            dry_run=dry_run,
        )
        return await runner.run()
    except Exception as e:
        print(f"Trading failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def create_runner(mode: str):
    """
    Factory function to create the appropriate runner for the given mode.

    Args:
        mode: Operational mode (monitor, backtest, trading).

    Returns:
        Async function to run the mode.
    """
    runners = {
        "monitor": main_async,
        "backtest": run_backtest,
        "trading": run_trading,
    }
    return runners.get(mode, main_async)


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Get the appropriate runner for the mode
    runner = create_runner(args.mode)

    try:
        if args.mode == "backtest":
            exit_code = asyncio.run(runner(args))
            sys.exit(exit_code)
        elif args.mode == "trading":
            exit_code = asyncio.run(runner(args))
            sys.exit(exit_code)
        else:
            # Default: monitor mode
            asyncio.run(runner(args))
    except KeyboardInterrupt:
        print("Shutdown requested")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
