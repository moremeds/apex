"""
Trading runner for live strategy execution.

Connects strategies to live market data and execution.
Supports dry-run mode for testing without actual execution.

Usage:
    runner = TradingRunner(
        strategy_name="ma_cross",
        symbols=["AAPL", "MSFT"],
        broker="ib",
        dry_run=True,
    )
    await runner.run()

Safety Features:
- Dry-run mode (default) - logs orders but doesn't execute
- Validation gate - requires ApexEngine validation for live trading
- Confirmation required for live execution
- RiskGate validation before order submission
- Position limits and emergency stop capability
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Type
import logging

import yaml

from ..domain.clock import SystemClock
from ..domain.strategy.base import Strategy, StrategyContext
from ..domain.strategy.scheduler import LiveScheduler
from ..domain.strategy.registry import get_strategy_class, list_strategies
from ..domain.strategy.risk_gate import RiskGate, ValidationResult
from ..domain.strategy.cost_estimator import create_ib_cost_estimator, create_futu_cost_estimator, create_zero_cost_estimator
from ..domain.events.domain_events import QuoteTick, TradeFill
from ..domain.interfaces.execution_provider import OrderRequest

logger = logging.getLogger(__name__)


class StrategyNotValidatedError(Exception):
    """
    Raised when attempting live trading with an unvalidated strategy.

    Strategies must be validated by ApexEngine backtest before live execution.
    This prevents running untested strategies with real money.
    """

    pass


class ManifestLoadError(Exception):
    """Raised when manifest.yaml cannot be loaded or is malformed."""

    pass


def load_strategy_manifest() -> Dict[str, Any]:
    """
    Load the strategy manifest.yaml file.

    Returns:
        Manifest dictionary with strategy configurations.

    Raises:
        ManifestLoadError: If manifest.yaml not found or malformed.
    """
    manifest_path = Path(__file__).parents[1] / "domain/strategy/manifest.yaml"
    if not manifest_path.exists():
        raise ManifestLoadError(f"Strategy manifest not found: {manifest_path}")

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ManifestLoadError(f"Malformed manifest.yaml: {e}") from e

    if not isinstance(manifest, dict):
        raise ManifestLoadError(
            f"Invalid manifest format: expected dict, got {type(manifest).__name__}"
        )

    return manifest


@dataclass
class TradingConfig:
    """Configuration for live trading."""

    strategy_name: str
    symbols: List[str]
    broker: str = "ib"  # "ib" or "futu"

    # Safety
    dry_run: bool = True  # Default to dry run
    require_confirmation: bool = True  # Require user confirmation for live

    # Strategy params
    strategy_params: Dict[str, Any] = field(default_factory=dict)

    # Risk limits
    max_position_size: float = 1000
    max_order_size: float = 100
    max_notional_per_order: float = 50000

    # Execution
    paper_trading: bool = False  # Use paper account if available


class TradingRunner:
    """
    Runner for live strategy execution.

    Connects a strategy to live market data and execution.
    Validates orders through RiskGate before submission.
    """

    def __init__(
        self,
        strategy_name: str,
        symbols: List[str],
        broker: str = "ib",
        dry_run: bool = True,
        strategy_params: Optional[Dict[str, Any]] = None,
        risk_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize trading runner.

        Args:
            strategy_name: Name of strategy in registry.
            symbols: Symbols to trade.
            broker: Broker to use (ib, futu).
            dry_run: If True, log orders but don't execute.
            strategy_params: Parameters for strategy.
            risk_config: Risk gate configuration.
        """
        self.strategy_name = strategy_name
        self.symbols = symbols
        self.broker = broker
        self.dry_run = dry_run
        self.strategy_params = strategy_params or {}
        self.risk_config = risk_config or {}

        # Components (initialized in run())
        self._clock = SystemClock()
        self._scheduler: Optional[LiveScheduler] = None
        self._strategy: Optional[Strategy] = None
        self._risk_gate: Optional[RiskGate] = None
        self._context: Optional[StrategyContext] = None

        # State
        self._running = False
        self._order_count = 0
        self._rejected_count = 0

    def _check_validation_gate(self) -> None:
        """
        Check if strategy is validated for live trading.

        Dry-run mode is always allowed. Live trading requires
        the strategy to have `validated_by_apex: true` in manifest.yaml.

        Raises:
            StrategyNotValidatedError: If live trading attempted without validation.
            ManifestLoadError: If manifest is missing or malformed (fail-closed).
        """
        if self.dry_run:
            return  # Dry run always allowed

        # Fail-closed: if manifest can't be loaded, block live trading
        try:
            manifest = load_strategy_manifest()
        except ManifestLoadError as e:
            raise StrategyNotValidatedError(
                f"\n{'=' * 60}\n"
                f"LIVE TRADING BLOCKED: Cannot verify validation\n"
                f"{'=' * 60}\n"
                f"Error: {e}\n"
                f"\nThe strategy manifest is required for live trading validation.\n"
                f"Ensure src/domain/strategy/manifest.yaml exists and is valid.\n"
                f"{'=' * 60}"
            ) from e

        strategies = manifest.get("strategies", {})
        if not isinstance(strategies, dict):
            raise StrategyNotValidatedError(
                f"\n{'=' * 60}\n"
                f"LIVE TRADING BLOCKED: Invalid manifest format\n"
                f"{'=' * 60}\n"
                f"The 'strategies' section in manifest.yaml is malformed.\n"
                f"{'=' * 60}"
            )

        strategy_config = strategies.get(self.strategy_name, {})
        if not isinstance(strategy_config, dict):
            strategy_config = {}

        validation = strategy_config.get("validation", {})
        if not isinstance(validation, dict):
            validation = {}

        if not validation.get("validated_by_apex", False):
            validation_date = validation.get("validation_date", "never")
            raise StrategyNotValidatedError(
                f"\n{'=' * 60}\n"
                f"LIVE TRADING BLOCKED: Strategy not validated\n"
                f"{'=' * 60}\n"
                f"Strategy:        {self.strategy_name}\n"
                f"Last validated:  {validation_date}\n"
                f"\nStrategies must be validated by ApexEngine before live trading.\n"
                f"This ensures the strategy has been backtested and reviewed.\n"
                f"\nTo validate:\n"
                f"  1. Run backtest with ApexEngine:\n"
                f"     python -m src.backtest.runner --strategy {self.strategy_name} \\\n"
                f"       --symbols AAPL --start 2024-01-01 --end 2024-06-30\n"
                f"\n"
                f"  2. Review results and update manifest.yaml:\n"
                f"     {self.strategy_name}:\n"
                f"       validation:\n"
                f"         validated_by_apex: true\n"
                f"         validation_date: \"YYYY-MM-DD\"\n"
                f"         validation_sharpe: 1.42  # optional\n"
                f"{'=' * 60}"
            )

        logger.info(
            f"Validation gate passed: {self.strategy_name} "
            f"(validated: {validation.get('validation_date', 'unknown')})"
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TradingRunner":
        """
        Create runner from CLI arguments.

        Args:
            args: Parsed CLI arguments.

        Returns:
            TradingRunner instance.
        """
        # Parse symbols
        symbols = [s.strip() for s in args.symbols.split(",")]

        # Parse strategy params if provided
        strategy_params = {}
        if hasattr(args, "params") and args.params:
            for param in args.params:
                key, value = param.split("=")
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                strategy_params[key] = value

        return cls(
            strategy_name=args.strategy,
            symbols=symbols,
            broker=getattr(args, "broker", "ib"),
            dry_run=getattr(args, "dry_run", True),
            strategy_params=strategy_params,
        )

    async def run(self) -> int:
        """
        Run the trading strategy.

        Returns:
            Exit code (0 for success, 1 for error).
        """
        # Validation gate - check strategy is validated before live trading
        try:
            self._check_validation_gate()
        except StrategyNotValidatedError as e:
            print(str(e))
            return 1

        # Safety confirmation for live trading
        if not self.dry_run:
            if not self._confirm_live_trading():
                print("Live trading cancelled.")
                return 1

        self._print_config()

        try:
            # Initialize components
            await self._initialize()

            # Start strategy
            self._strategy.start()
            self._running = True

            logger.info(f"Trading started: {self.strategy_name}")

            if self.dry_run:
                print("\n*** DRY RUN MODE - Orders will be logged but not executed ***\n")

            # Set up signal handlers
            self._setup_signal_handlers()

            # Main loop
            await self._run_loop()

            return 0

        except Exception as e:
            logger.error(f"Trading error: {e}")
            return 1

        finally:
            await self._shutdown()

    async def _initialize(self) -> None:
        """Initialize trading components."""
        # Get strategy class
        strategy_class = get_strategy_class(self.strategy_name)
        if strategy_class is None:
            raise ValueError(
                f"Unknown strategy: {self.strategy_name}. "
                f"Available: {list_strategies()}"
            )

        # Create scheduler
        self._scheduler = LiveScheduler(self._clock)

        # Create cost estimator based on broker
        if self.broker == "ib":
            cost_estimator = create_ib_cost_estimator()
        elif self.broker == "futu":
            cost_estimator = create_futu_cost_estimator()
        else:
            cost_estimator = create_zero_cost_estimator()

        # Create risk gate
        self._risk_gate = RiskGate(config=self.risk_config)

        # Create context
        self._context = StrategyContext(
            clock=self._clock,
            scheduler=self._scheduler,
            positions={},
            account=None,
            cost_estimator=cost_estimator,
            risk_gate=self._risk_gate,
        )

        # Create strategy
        self._strategy = strategy_class(
            strategy_id=f"live-{self.strategy_name}",
            symbols=self.symbols,
            context=self._context,
            **self.strategy_params,
        )

        # Create risk gate
        self._risk_gate = RiskGate(config=self.risk_config)

        # Wire order callback
        self._strategy.on_order_callback(self._handle_order)

        logger.info(f"Initialized strategy: {self._strategy}")

    async def _run_loop(self) -> None:
        """
        Main trading loop.

        LIVE-001: This loop needs broker adapter integration to be production-ready.
        Currently implements:
        - Strategy health monitoring
        - Heartbeat logging
        - Graceful shutdown

        TODO for production readiness:
        1. Initialize broker adapter (IB/Futu) for market data
        2. Subscribe to market data for strategy symbols
        3. Deliver QuoteTick events to strategy.on_quote()
        4. Wire order execution to broker adapter
        5. Monitor broker connection health and reconnect
        6. Handle broker disconnection gracefully
        """
        logger.info("Main trading loop started")

        # Health tracking
        last_heartbeat = 0
        heartbeat_interval = 60  # seconds
        loop_count = 0
        consecutive_errors = 0
        max_consecutive_errors = 10

        while self._running:
            try:
                loop_count += 1

                # 1. Check strategy health
                if self._strategy.state.value == "error":
                    logger.error(f"Strategy entered error state: {self._strategy._error_message}")
                    self._running = False
                    break

                # 2. Check for stuck loop (no progress)
                # TODO: Add actual market data check when broker integration is added

                # 3. Log heartbeat periodically
                now = int(self._clock.now().timestamp())
                if now - last_heartbeat >= heartbeat_interval:
                    last_heartbeat = now
                    logger.info(
                        f"Trading heartbeat: loop={loop_count}, orders={self._order_count}, "
                        f"rejected={self._rejected_count}, state={self._strategy.state.value}"
                    )

                # 4. Wait for next iteration
                # TODO: Replace with event-driven market data reception
                await asyncio.sleep(1.0)

                # Reset error counter on successful iteration
                consecutive_errors = 0

            except asyncio.CancelledError:
                logger.info("Trading loop cancelled")
                break
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error in trading loop (count={consecutive_errors}): {e}")

                # Stop if too many consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Too many consecutive errors ({consecutive_errors}), stopping")
                    self._running = False
                    break

                await asyncio.sleep(5.0)  # Cool down on error

    def _handle_order(self, order: OrderRequest) -> None:
        """Handle order from strategy."""
        # Validate through risk gate
        result = self._risk_gate.validate(order, self._context)

        if not result.approved:
            self._rejected_count += 1
            logger.warning(
                f"Order REJECTED by RiskGate: {result.message} "
                f"[{order.side} {order.quantity} {order.symbol}]"
            )
            return

        self._order_count += 1

        if self.dry_run:
            # Log order but don't execute
            self._log_dry_run_order(order)
        else:
            # Submit to broker
            self._submit_order(order)

    def _log_dry_run_order(self, order: OrderRequest) -> None:
        """Log order in dry-run mode."""
        logger.info(
            f"[DRY RUN] Order #{self._order_count}: "
            f"{order.side} {order.quantity} {order.symbol} "
            f"@ {order.order_type}"
            f"{f' limit={order.limit_price}' if order.limit_price else ''}"
        )
        print(
            f"[DRY RUN] {self._clock.now().strftime('%H:%M:%S')} "
            f"ORDER: {order.side} {order.quantity} {order.symbol}"
        )

    def _submit_order(self, order: OrderRequest) -> None:
        """Submit order to broker (live mode)."""
        # In a real implementation, this would submit to broker adapter
        logger.info(
            f"[LIVE] Submitting order #{self._order_count}: "
            f"{order.side} {order.quantity} {order.symbol}"
        )
        print(
            f"[LIVE] {self._clock.now().strftime('%H:%M:%S')} "
            f"SUBMITTED: {order.side} {order.quantity} {order.symbol}"
        )
        # TODO: Wire to actual broker execution adapter

    def _confirm_live_trading(self) -> bool:
        """Get user confirmation for live trading."""
        print("\n" + "=" * 60)
        print("WARNING: LIVE TRADING MODE")
        print("=" * 60)
        print(f"Strategy:  {self.strategy_name}")
        print(f"Symbols:   {', '.join(self.symbols)}")
        print(f"Broker:    {self.broker}")
        print("\nThis will execute REAL orders with REAL money.")
        print("=" * 60)

        try:
            response = input("\nType 'CONFIRM' to proceed with live trading: ")
            return response.strip().upper() == "CONFIRM"
        except (EOFError, KeyboardInterrupt):
            return False

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def handle_signal(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self._running = False

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

    async def _shutdown(self) -> None:
        """Shutdown trading."""
        self._running = False

        if self._strategy:
            self._strategy.stop()

        if self._scheduler:
            self._scheduler.stop()

        logger.info(
            f"Trading stopped: orders={self._order_count}, "
            f"rejected={self._rejected_count}"
        )

    def _print_config(self) -> None:
        """Print trading configuration."""
        mode = "DRY RUN" if self.dry_run else "LIVE"
        print(f"\n{'=' * 60}")
        print(f"TRADING CONFIGURATION ({mode})")
        print(f"{'=' * 60}")
        print(f"Strategy:     {self.strategy_name}")
        print(f"Symbols:      {', '.join(self.symbols)}")
        print(f"Broker:       {self.broker}")
        print(f"Mode:         {mode}")
        if self.strategy_params:
            print(f"Parameters:   {self.strategy_params}")
        print(f"{'=' * 60}\n")

    def stop(self) -> None:
        """Stop trading."""
        self._running = False


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for trading runner."""
    parser = argparse.ArgumentParser(description="Run live trading strategy")

    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="Strategy name from registry",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        required=True,
        help="Comma-separated list of symbols",
    )
    parser.add_argument(
        "--broker",
        type=str,
        default="ib",
        choices=["ib", "futu"],
        help="Broker to use",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Dry run mode (default: True)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live trading (overrides --dry-run)",
    )
    parser.add_argument(
        "--params",
        nargs="*",
        help="Strategy parameters (key=value)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    return parser


async def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Override dry_run if --live is specified
    if args.live:
        args.dry_run = False

    runner = TradingRunner.from_args(args)
    exit_code = await runner.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
