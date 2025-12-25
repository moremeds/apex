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
from src.infrastructure.adapters import IbAdapter, FutuAdapter, FileLoader, BrokerManager, MarketDataManager, YahooFinanceAdapter
from src.infrastructure.adapters.ib import IbConnectionPool, ConnectionPoolConfig
from src.services import HistoricalDataService, TAService, BarPeriod
from src.infrastructure.stores import PositionStore, MarketDataStore, AccountStore
from src.infrastructure.monitoring import HealthMonitor, Watchdog
from src.domain.services.risk.risk_engine import RiskEngine
from src.domain.services.pos_reconciler import Reconciler
from src.domain.services.mdqc import MDQC
from src.domain.services.risk.rule_engine import RuleEngine
from src.domain.services.risk.risk_signal_manager import RiskSignalManager
from src.domain.services.risk.risk_signal_engine import RiskSignalEngine
from src.domain.services.risk.risk_alert_logger import RiskAlertLogger
from src.application import Orchestrator, ReadinessManager
from src.domain.events import PriorityEventBus
from src.tui import TerminalDashboard
from src.models.risk_snapshot import RiskSnapshot
from src.utils import StructuredLogger, flush_all_loggers, set_log_timezone
from src.utils.structured_logger import LogCategory
from src.utils.logging_setup import setup_category_logging, is_console_enabled
from src.utils.perf_logger import set_perf_metrics

# Observability imports (optional - graceful fallback if not installed)
try:
    from src.infrastructure.observability import (
        MetricsManager, get_metrics_manager, RiskMetrics, HealthMetrics
    )
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False
    MetricsManager = None  # type: ignore
    get_metrics_manager = None  # type: ignore
    RiskMetrics = None  # type: ignore
    HealthMetrics = None  # type: ignore


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Live Risk Management System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --env dev              # Run in development mode (monitor)
  python main.py --mode monitor         # Explicit monitor mode
  python main.py --mode backtest --spec config/backtest/ma_cross.yaml
  python main.py --mode backtest --engine backtrader --strategy ma_cross
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
    global metrics_manager
    config_manager = ConfigManager(config_dir="config", env=args.env)
    config = config_manager.load()

    # Set timezone for logging (before creating loggers)
    log_tz = config.logging.timezone
    if log_tz and log_tz.lower() != "local":
        set_log_timezone(log_tz)
    else:
        set_log_timezone(None)  # Use local time

    # Set up category-based logging with environment prefix
    # Now creates 5 categories: system, adapter, risk, data, perf
    category_loggers = setup_category_logging(
        env=args.env,
        log_dir="./logs",
        level=args.log_level,
        console=args.no_dashboard,  # Console output when no dashboard
        verbose=args.verbose,
    )
    system_logger = category_loggers["system"]
    data_logger = category_loggers["data"]

    # Create structured loggers for each category
    system_structured = StructuredLogger(system_logger)
    data_structured = StructuredLogger(data_logger)

    system_structured.info(
        LogCategory.SYSTEM,
        "Starting Live Risk Management System",
        {"env": args.env, "log_timezone": log_tz}
    )

    orchestrator = None
    dashboard = None

    try:
        system_structured.info(LogCategory.SYSTEM, "Configuration loaded", {"env": args.env})

        # Initialize event bus (use PriorityEventBus for dual-lane priority dispatch)
        event_bus = PriorityEventBus()
        system_structured.info(LogCategory.SYSTEM, "Using PriorityEventBus (dual-lane)")

        # Initialize observability (Prometheus metrics)
        risk_metrics = None
        health_metrics = None
        metrics_manager = None

        if OBSERVABILITY_AVAILABLE and args.metrics_port > 0:
            try:
                metrics_manager = get_metrics_manager(port=args.metrics_port)
                metrics_manager.start()

                meter = metrics_manager.get_meter("apex")
                risk_metrics = RiskMetrics(meter)
                health_metrics = HealthMetrics(meter)

                # Wire perf logger to send metrics to Prometheus
                set_perf_metrics(health_metrics=health_metrics, risk_metrics=risk_metrics)

                system_structured.info(
                    LogCategory.SYSTEM,
                    "Observability enabled",
                    {"metrics_port": args.metrics_port, "endpoint": f"http://localhost:{args.metrics_port}/metrics"}
                )
                if is_console_enabled():
                    print(f"📊 Metrics available at http://localhost:{args.metrics_port}/metrics", flush=True)
            except Exception as e:
                system_structured.warning(
                    LogCategory.SYSTEM,
                    f"Failed to start metrics server: {e}. Continuing without observability."
                )
        elif not OBSERVABILITY_AVAILABLE:
            system_structured.info(
                LogCategory.SYSTEM,
                "Observability not available (install with: pip install -e '.[observability]')"
            )
        else:
            system_structured.info(LogCategory.SYSTEM, "Observability disabled (--metrics-port 0)")

        # Initialize data stores
        position_store = PositionStore()
        market_data_store = MarketDataStore()
        account_store = AccountStore()

        # Initialize monitoring (needed for BrokerManager)
        health_monitor = HealthMonitor()

        # Initialize BrokerManager to manage all broker connections
        broker_manager = BrokerManager(health_monitor=health_monitor)
        # Initialize MarketDataManager to manage all market data sources
        # EventBus passed for single streaming path (IB → MDManager → EventBus)
        market_data_manager = MarketDataManager(
            health_monitor=health_monitor,
            event_bus=event_bus,
        )

        # IB Connection Pool for historical data (ATR, TA indicators)
        ib_pool = None
        historical_service = None
        ta_service = None

        # Register adapters with BrokerManager
        if config.ibkr.enabled:
            ib_adapter = IbAdapter(
                host=config.ibkr.host,
                port=config.ibkr.port,
                client_id=config.ibkr.client_ids.monitoring,
                event_bus=event_bus,
            )
            broker_manager.register_adapter("ib", ib_adapter)
            system_structured.info(LogCategory.SYSTEM, "IB adapter registered", {
                "host": config.ibkr.host,
                "port": config.ibkr.port,
            })
            market_data_manager.register_provider("ib", ib_adapter, priority=10)
            system_structured.info(LogCategory.SYSTEM, "IB registered as market data provider", {
                "streaming": ib_adapter.supports_streaming(),
                "greeks": ib_adapter.supports_greeks(),
            })

            # Create IB Connection Pool for historical data (separate IB connection)
            # Uses same event loop - no threading issues
            pool_config = ConnectionPoolConfig(
                host=config.ibkr.host,
                port=config.ibkr.port,
                client_ids=config.ibkr.client_ids,
            )
            ib_pool = IbConnectionPool(pool_config)

        else:
            system_structured.info(LogCategory.SYSTEM, "IB adapter DISABLED (demo mode)")

        if config.futu.enabled:
            futu_adapter = FutuAdapter(
                host=config.futu.host,
                port=config.futu.port,
                security_firm=config.futu.security_firm,
                trd_env=config.futu.trd_env,
                filter_trading_market=config.futu.filter_trdmarket,
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

        # TODO: Register additional market data providers here as needed
        # Example for future Yahoo Finance or CCXT providers:
        # if config.yahoo.enabled:
        #     yahoo_provider = YahooFinanceProvider(...)
        #     market_data_manager.register_provider("yahoo", yahoo_provider, priority=50)
        #
        # if config.ccxt.enabled:
        #     ccxt_provider = CcxtProvider(...)
        #     market_data_manager.register_provider("ccxt", ccxt_provider, priority=60)

        # Initialize Yahoo Finance adapter for beta and market data
        # DISABLED: Yahoo API calls were causing 6.5s snapshot delays
        # TODO: Re-enable with async/cached-only mode once performance is verified
        yahoo_adapter = None
        # yahoo_adapter = YahooFinanceAdapter(
        #     price_ttl_seconds=30,
        #     beta_ttl_hours=24,
        # )
        # await yahoo_adapter.connect()
        # system_structured.info(LogCategory.SYSTEM, "Yahoo Finance adapter initialized")
        system_structured.info(LogCategory.SYSTEM, "Yahoo Finance adapter DISABLED (performance)")

        # Initialize domain services
        risk_engine = RiskEngine(
            config=config.raw,
            yahoo_adapter=yahoo_adapter,  # None - uses beta=1.0 for all
            risk_metrics=risk_metrics,
        )
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

        # Initialize ReadinessManager for event-driven system readiness
        # Determine required brokers from config
        required_brokers = []
        if config.ibkr.enabled:
            required_brokers.append("ib")
        if config.futu.enabled:
            required_brokers.append("futu")
        # Manual positions always available
        required_brokers.append("manual")

        # Require 100% market data coverage before starting snapshots
        # User's positions are standard tickers that should all have data
        market_data_threshold = 1.0  # 100% coverage required
        readiness_manager = ReadinessManager(
            event_bus=event_bus,
            required_brokers=required_brokers,
            market_data_coverage_threshold=market_data_threshold,
            startup_timeout_sec=config.raw.get("dashboard", {}).get("snapshot_ready_timeout_sec", 30.0),
        )
        system_structured.info(LogCategory.SYSTEM, "ReadinessManager initialized", {
            "required_brokers": required_brokers,
            "coverage_threshold": market_data_threshold,
        })

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
            risk_metrics=risk_metrics,
            health_metrics=health_metrics,
            readiness_manager=readiness_manager,
        )

        # Initialize dashboard (if not disabled)
        if not args.no_dashboard:
            dashboard = TerminalDashboard(
                config=config.raw.get("dashboard", {}),
                env=args.env,
            )

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

        # Connect historical IB and pre-fetch daily bars (non-blocking background task)
        prefetch_task = None
        if ib_pool:
            async def connect_historical_and_prefetch():
                """Background task to connect historical IB and pre-fetch bars."""
                try:
                    # Connect historical IB connection (on same event loop)
                    await ib_pool.connect_historical()
                    system_structured.info(
                        LogCategory.SYSTEM,
                        "IB historical connection established",
                        {"client_id": config.ibkr.client_ids.historical_pool[0]}
                    )

                    # Create services
                    nonlocal historical_service, ta_service
                    historical_service = HistoricalDataService(
                        ib_historical=ib_pool.historical,
                        cache_size=512,
                        default_daily_lookback=60,
                    )
                    ta_service = TAService(historical_service)

                    # Inject TAService and HistoricalDataService into dashboard
                    if dashboard:
                        event_loop = asyncio.get_event_loop()
                        dashboard.set_ta_service(ta_service, event_loop, historical_service)

                    # Get symbols from positions for pre-fetch
                    positions = position_store.get_all()
                    underlyings = set()
                    for pos in positions:
                        # Use underlying for options, symbol for stocks
                        sym = pos.underlying if pos.underlying else pos.symbol
                        if sym:
                            underlyings.add(sym)

                    if underlyings:
                        symbols = list(underlyings)
                        system_structured.info(
                            LogCategory.DATA,
                            f"Pre-fetching 60d daily bars for {len(symbols)} symbols",
                            {"symbols": symbols[:10]}  # Log first 10
                        )
                        await historical_service.prefetch_daily_bars(symbols, lookback_days=60)
                        system_structured.info(
                            LogCategory.DATA,
                            "Daily bars pre-fetch complete"
                        )
                except Exception as e:
                    system_structured.warning(
                        LogCategory.SYSTEM,
                        f"Historical data setup failed: {e}"
                    )

            # Start pre-fetch in background (non-blocking)
            prefetch_task = asyncio.create_task(connect_historical_and_prefetch())

        # Update loop - feeds data from orchestrator to dashboard via queue
        update_task = None
        if dashboard:
            async def update_loop():
                """Background task that feeds orchestrator data to dashboard queue."""
                while dashboard._running:
                    try:
                        # Wait for orchestrator to signal new snapshot
                        snapshot = await orchestrator.wait_for_snapshot(timeout=3.0)
                        health = health_monitor.get_all_health()

                        if snapshot:
                            # Get risk signals from orchestrator (multi-layer risk detection)
                            risk_signals = orchestrator.get_latest_risk_signals()

                            # Use market alert detector output from orchestrator
                            market_alerts = orchestrator.get_latest_market_alerts()

                            # Queue update to dashboard (thread-safe)
                            dashboard.update(snapshot, risk_signals, health, market_alerts)

                            # Log market data fetch
                            data_structured.info(
                                LogCategory.DATA,
                                "Risk snapshot refreshed",
                                {"positions": snapshot.total_positions, "position_risks": len(snapshot.position_risks)}
                            )
                        else:
                            # No full snapshot yet - show positions preview immediately
                            preview = orchestrator.get_positions_preview()
                            if preview and preview.total_positions > 0:
                                dashboard.update(preview, [], health, [])
                            else:
                                empty_snapshot = RiskSnapshot()
                                dashboard.update(empty_snapshot, [], health, [])
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        system_structured.warning(
                            LogCategory.SYSTEM,
                            f"Update loop error: {e}"
                        )
                        await asyncio.sleep(1)

            # Start update loop as background task
            update_task = asyncio.create_task(update_loop())

            # Run Textual dashboard in main thread (blocks until user quits)
            # The update_task runs concurrently via asyncio
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

        # Disconnect IB connection pool
        if ib_pool:
            await ib_pool.disconnect()
            system_structured.info(LogCategory.SYSTEM, "IB connection pool disconnected")

        # Shutdown metrics server
        if metrics_manager:
            metrics_manager.shutdown()
            system_structured.info(LogCategory.SYSTEM, "Metrics server stopped")

        # Ensure all logs are flushed to disk
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
