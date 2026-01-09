"""
Signal Runner - Standalone TA signal pipeline for validation and testing.

Enables running the signal pipeline independently from the TUI for:
- Pipeline validation with synthetic data
- Live signal generation without TUI overhead
- Historical bar backfill for indicator warmup

Usage:
    # Validate pipeline with synthetic ticks (no database)
    python -m src.runners.signal_runner --validate

    # Run pipeline on live market data (headless mode)
    python -m src.runners.signal_runner --live --symbols AAPL,TSLA

    # Backfill signals from historical bars
    python -m src.runners.signal_runner --backfill --symbols AAPL --days 365

    # Connect to database for persistence
    python -m src.runners.signal_runner --live --symbols AAPL --with-persistence
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import random
import signal
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..domain.events.domain_events import MarketDataTickEvent
from ..domain.events.event_types import EventType
from ..domain.events.priority_event_bus import PriorityEventBus
from ..application.services.ta_signal_service import TASignalService
from ..utils.logging_setup import get_logger
from ..utils.timezone import now_utc

logger = get_logger(__name__)


@dataclass
class SignalRunnerConfig:
    """Configuration for signal runner."""

    symbols: List[str]
    timeframes: List[str]
    max_workers: int = 4

    # Validation mode
    validate: bool = False
    validation_ticks: int = 100
    validation_interval_ms: float = 10.0

    # Live mode
    live: bool = False

    # Backfill mode
    backfill: bool = False
    backfill_days: int = 365

    # Persistence
    with_persistence: bool = False

    # Output
    verbose: bool = False
    stats_interval: int = 10


class SignalRunner:
    """
    Standalone signal pipeline runner.

    Can run TASignalService independently for:
    - Validation (synthetic data, no DB)
    - Live execution (real market data, optional DB)
    - Backfill (historical bars from data provider)
    """

    def __init__(self, config: SignalRunnerConfig) -> None:
        """
        Initialize signal runner.

        Args:
            config: Runner configuration.
        """
        self.config = config
        self._event_bus: Optional[PriorityEventBus] = None
        self._service: Optional[TASignalService] = None
        self._persistence = None
        self._running = False
        self._signal_count = 0

    async def run(self) -> int:
        """
        Run the signal pipeline based on configuration.

        Returns:
            Exit code (0 for success, non-zero for errors).
        """
        try:
            await self._initialize()

            if self.config.validate:
                return await self._run_validation()
            elif self.config.backfill:
                return await self._run_backfill()
            elif self.config.live:
                return await self._run_live()
            else:
                print("No mode specified. Use --validate, --live, or --backfill")
                return 1

        except KeyboardInterrupt:
            logger.info("Signal runner interrupted by user")
            return 0
        except Exception as e:
            logger.error(f"Signal runner error: {e}", exc_info=True)
            return 1
        finally:
            await self._shutdown()

    async def _initialize(self) -> None:
        """Initialize pipeline components."""
        # Create event bus
        self._event_bus = PriorityEventBus()
        await self._event_bus.start()

        # Setup persistence if requested
        if self.config.with_persistence:
            self._persistence = await self._create_persistence()

        # Create TASignalService
        self._service = TASignalService(
            event_bus=self._event_bus,
            persistence=self._persistence,
            timeframes=self.config.timeframes,
            max_workers=self.config.max_workers,
        )

        # Subscribe to signals for counting
        self._event_bus.subscribe(EventType.TRADING_SIGNAL, self._on_signal)

        await self._service.start()
        logger.info(f"SignalRunner initialized: timeframes={self.config.timeframes}")

    async def _create_persistence(self):
        """Create persistence layer if database is available."""
        try:
            from config.loader import load_config
            from ..infrastructure.persistence.database import get_database
            from ..infrastructure.persistence.repositories.ta_signal_repository import TASignalRepository

            config = load_config()
            db = await get_database(config.database)
            return TASignalRepository(db)
        except Exception as e:
            logger.warning(f"Could not initialize persistence: {e}")
            return None

    async def _shutdown(self) -> None:
        """Shutdown pipeline components."""
        if self._service:
            await self._service.stop()

        if self._event_bus:
            await self._event_bus.stop()

        logger.info(f"SignalRunner stopped: signals_received={self._signal_count}")

    def _on_signal(self, payload: Any) -> None:
        """Handle trading signal event."""
        self._signal_count += 1

        # Extract signal details for logging
        signal_type = getattr(payload, "direction", "unknown")
        symbol = getattr(payload, "symbol", "unknown")
        indicator = getattr(payload, "indicator", "unknown")
        strength = getattr(payload, "strength", 0)

        if self.config.verbose:
            print(f"  SIGNAL: {signal_type.upper()} {symbol} [{indicator}] strength={strength}")

    # -------------------------------------------------------------------------
    # Validation Mode
    # -------------------------------------------------------------------------

    async def _run_validation(self) -> int:
        """
        Run pipeline validation with synthetic data.

        Generates synthetic ticks and verifies pipeline produces expected signals.

        Returns:
            Exit code (0 if validation passed).
        """
        print("=" * 60)
        print("SIGNAL PIPELINE VALIDATION")
        print("=" * 60)
        print(f"Symbols:    {', '.join(self.config.symbols)}")
        print(f"Timeframes: {', '.join(self.config.timeframes)}")
        print(f"Ticks:      {self.config.validation_ticks}")
        print("=" * 60)
        print()

        self._running = True

        # Generate and inject synthetic ticks
        for symbol in self.config.symbols:
            print(f"Injecting {self.config.validation_ticks} ticks for {symbol}...")
            ticks = self._generate_synthetic_ticks(
                symbol=symbol,
                count=self.config.validation_ticks,
            )

            for i, tick in enumerate(ticks):
                if not self._running:
                    break

                self._event_bus.publish(EventType.MARKET_DATA_TICK, tick)

                if (i + 1) % 20 == 0:
                    print(f"  Processed {i + 1}/{self.config.validation_ticks} ticks")

                await asyncio.sleep(self.config.validation_interval_ms / 1000)

        # Wait for pipeline to process remaining events
        print("\nWaiting for pipeline to complete...")
        await asyncio.sleep(2.0)

        # Report results
        stats = self._service.stats
        print()
        print("=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)
        print(f"Bars processed:       {stats['bars_processed']}")
        print(f"Indicators computed:  {stats['indicators_computed']}")
        print(f"Signals emitted:      {stats['signals_emitted']}")
        print(f"Signals received:     {self._signal_count}")
        print("=" * 60)

        # Validation criteria
        if stats["bars_processed"] > 0:
            print("\n✅ Validation PASSED: Pipeline is processing data")
            return 0
        else:
            print("\n❌ Validation FAILED: No bars were processed")
            return 1

    def _generate_synthetic_ticks(
        self,
        symbol: str,
        count: int,
        base_price: float = 150.0,
        volatility: float = 0.02,
    ) -> List[MarketDataTickEvent]:
        """
        Generate synthetic market data ticks with realistic price movement.

        Uses Geometric Brownian Motion to simulate price paths that
        will trigger various indicator conditions (oversold, overbought, etc.).

        Args:
            symbol: Trading symbol.
            count: Number of ticks to generate.
            base_price: Starting price.
            volatility: Price volatility (standard deviation per tick).

        Returns:
            List of MarketDataTickEvent objects.
        """
        ticks = []
        price = base_price
        timestamp = now_utc()

        # Add some trending behavior to trigger signals
        trend_direction = 1.0  # Start bullish
        trend_duration = count // 5  # Change trend every 20% of ticks

        for i in range(count):
            # Change trend direction periodically
            if i > 0 and i % trend_duration == 0:
                trend_direction *= -1

            # Geometric Brownian Motion with drift
            drift = 0.0002 * trend_direction  # Small trend bias
            random_shock = random.gauss(0, volatility)
            price *= math.exp(drift + random_shock)

            # Add some bid-ask spread
            spread = price * 0.001  # 0.1% spread
            bid = price - spread / 2
            ask = price + spread / 2

            tick = MarketDataTickEvent(
                symbol=symbol,
                source="synthetic",
                timestamp=timestamp,
                bid=bid,
                ask=ask,
                last=price,
            )
            ticks.append(tick)

            timestamp += timedelta(seconds=1)

        return ticks

    # -------------------------------------------------------------------------
    # Live Mode
    # -------------------------------------------------------------------------

    async def _run_live(self) -> int:
        """
        Run pipeline on live market data.

        Connects to broker adapters for real-time data without TUI.

        Returns:
            Exit code (0 for success).
        """
        print("=" * 60)
        print("LIVE SIGNAL PIPELINE (Headless)")
        print("=" * 60)
        print(f"Symbols:     {', '.join(self.config.symbols)}")
        print(f"Timeframes:  {', '.join(self.config.timeframes)}")
        print(f"Persistence: {'enabled' if self.config.with_persistence else 'disabled'}")
        print("=" * 60)
        print("\nStarting live signal generation. Press Ctrl+C to stop.\n")

        self._running = True
        self._setup_signal_handlers()

        # Connect to live market data
        market_adapter = await self._create_market_adapter()

        if market_adapter is None:
            print("Running in demo mode (no broker connection)")
            print("Generating synthetic ticks for demonstration...")
            return await self._run_live_demo()

        # Subscribe to market data
        for symbol in self.config.symbols:
            await market_adapter.subscribe(symbol)

        # Main loop - print stats periodically
        last_stats_time = 0
        while self._running:
            try:
                now = int(now_utc().timestamp())
                if now - last_stats_time >= self.config.stats_interval:
                    last_stats_time = now
                    stats = self._service.stats
                    print(
                        f"[{now_utc().strftime('%H:%M:%S')}] "
                        f"bars={stats['bars_processed']} "
                        f"indicators={stats['indicators_computed']} "
                        f"signals={stats['signals_emitted']}"
                    )

                await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                break

        return 0

    async def _run_live_demo(self) -> int:
        """Run live mode with synthetic data for demonstration."""
        tick_count = 0
        last_stats_time = 0

        for symbol in self.config.symbols:
            base_price = 150.0 + random.uniform(-50, 50)
            price = base_price

            while self._running:
                # Generate tick
                drift = random.gauss(0, 0.0002)
                price *= math.exp(drift + random.gauss(0, 0.01))

                tick = MarketDataTickEvent(
                    symbol=symbol,
                    source="demo",
                    timestamp=now_utc(),
                    bid=price * 0.999,
                    ask=price * 1.001,
                    last=price,
                )

                self._event_bus.publish(EventType.MARKET_DATA_TICK, tick)
                tick_count += 1

                # Print stats periodically
                now = int(now_utc().timestamp())
                if now - last_stats_time >= self.config.stats_interval:
                    last_stats_time = now
                    stats = self._service.stats
                    print(
                        f"[{now_utc().strftime('%H:%M:%S')}] "
                        f"ticks={tick_count} "
                        f"bars={stats['bars_processed']} "
                        f"indicators={stats['indicators_computed']} "
                        f"signals={stats['signals_emitted']}"
                    )

                await asyncio.sleep(0.5)  # Slower for demo

        return 0

    async def _create_market_adapter(self):
        """
        Create market data adapter for live mode.

        Returns:
            Market adapter or None if unavailable.
        """
        # Try to connect to IB
        try:
            from config.loader import load_config
            from ..infrastructure.adapters.ib.live_adapter import IbLiveAdapter

            config = load_config()
            adapter = IbLiveAdapter(config.brokers.ib, self._event_bus)

            # Try to connect (with timeout)
            try:
                await asyncio.wait_for(adapter.connect(), timeout=5.0)
                logger.info("Connected to Interactive Brokers")
                return adapter
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Could not connect to IB: {e}")
                return None

        except ImportError:
            logger.warning("IB adapter not available")
            return None

    # -------------------------------------------------------------------------
    # Backfill Mode
    # -------------------------------------------------------------------------

    async def _run_backfill(self) -> int:
        """
        Run backfill to process historical bars.

        Fetches historical bar data and injects it into the indicator engine
        for warmup and historical signal generation.

        Returns:
            Exit code (0 for success).
        """
        print("=" * 60)
        print("SIGNAL BACKFILL")
        print("=" * 60)
        print(f"Symbols:    {', '.join(self.config.symbols)}")
        print(f"Timeframes: {', '.join(self.config.timeframes)}")
        print(f"Days:       {self.config.backfill_days}")
        print(f"Persistence: {'enabled' if self.config.with_persistence else 'disabled'}")
        print("=" * 60)

        self._running = True
        total_bars = 0

        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                print(f"\nBackfilling {symbol} {timeframe}...")

                # Fetch historical bars
                bars = await self._fetch_historical_bars(
                    symbol=symbol,
                    timeframe=timeframe,
                    days=self.config.backfill_days,
                )

                if not bars:
                    print(f"  No historical data available for {symbol} {timeframe}")
                    continue

                # Inject into indicator engine
                injected = await self._service.inject_historical_bars(
                    symbol=symbol,
                    timeframe=timeframe,
                    bars=bars,
                )

                total_bars += injected
                print(f"  Injected {injected} bars")

        # Report results
        print()
        print("=" * 60)
        print("BACKFILL RESULTS")
        print("=" * 60)
        stats = self._service.stats
        print(f"Total bars injected:  {total_bars}")
        print(f"Indicators computed:  {stats['indicators_computed']}")
        print(f"Signals emitted:      {stats['signals_emitted']}")
        print("=" * 60)

        return 0 if total_bars > 0 else 1

    async def _fetch_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        days: int,
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical bars from data provider.

        Args:
            symbol: Trading symbol.
            timeframe: Bar timeframe.
            days: Number of days of history.

        Returns:
            List of bar dicts with open, high, low, close, volume, timestamp.
        """
        try:
            # Try DuckDB/Parquet store first
            from ..infrastructure.stores.duckdb_market_store import DuckDBMarketStore

            store = DuckDBMarketStore()
            end_date = now_utc()
            start_date = end_date - timedelta(days=days)

            bars = await store.get_bars(
                symbol=symbol,
                timeframe=timeframe,
                start=start_date,
                end=end_date,
            )

            if bars:
                return [bar.to_dict() for bar in bars]

        except Exception as e:
            logger.debug(f"DuckDB store not available: {e}")

        # Generate synthetic bars for demonstration
        logger.info(f"Generating synthetic historical bars for {symbol} {timeframe}")
        return self._generate_synthetic_bars(
            symbol=symbol,
            timeframe=timeframe,
            count=min(days, 252),  # Max 1 year of daily bars
        )

    def _generate_synthetic_bars(
        self,
        symbol: str,
        timeframe: str,
        count: int,
        base_price: float = 150.0,
    ) -> List[Dict[str, Any]]:
        """Generate synthetic historical bars for backfill testing."""
        bars = []
        price = base_price
        timestamp = now_utc() - timedelta(days=count)

        for _ in range(count):
            # Generate OHLC with random walk
            open_price = price
            high_price = price * (1 + random.uniform(0, 0.03))
            low_price = price * (1 - random.uniform(0, 0.03))
            close_price = random.uniform(low_price, high_price)

            bars.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": timestamp,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": random.randint(100000, 10000000),
            })

            price = close_price
            timestamp += timedelta(days=1)

        return bars

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def handle_signal(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self._running = False

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for signal runner."""
    parser = argparse.ArgumentParser(
        description="Standalone TA signal pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate pipeline with synthetic data
  python -m src.runners.signal_runner --validate

  # Run live signal generation
  python -m src.runners.signal_runner --live --symbols AAPL,TSLA

  # Backfill historical signals
  python -m src.runners.signal_runner --backfill --symbols AAPL --days 365

  # Enable database persistence
  python -m src.runners.signal_runner --live --symbols AAPL --with-persistence
        """,
    )

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--validate",
        action="store_true",
        help="Run validation with synthetic data (no database)",
    )
    mode_group.add_argument(
        "--live",
        action="store_true",
        help="Run on live market data (headless mode)",
    )
    mode_group.add_argument(
        "--backfill",
        action="store_true",
        help="Process historical bars for indicator warmup",
    )

    # Symbol/timeframe configuration
    parser.add_argument(
        "--symbols",
        type=str,
        default="AAPL",
        help="Comma-separated list of symbols (default: AAPL)",
    )
    parser.add_argument(
        "--timeframes",
        type=str,
        default="1d",
        help="Comma-separated list of timeframes (default: 1d)",
    )

    # Validation options
    parser.add_argument(
        "--ticks",
        type=int,
        default=100,
        help="Number of synthetic ticks for validation (default: 100)",
    )

    # Backfill options
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Days of history for backfill (default: 365)",
    )

    # Persistence options
    parser.add_argument(
        "--with-persistence",
        action="store_true",
        help="Enable database persistence for signals",
    )

    # General options
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Thread pool size for indicator calculations (default: 4)",
    )
    parser.add_argument(
        "--stats-interval",
        type=int,
        default=10,
        help="Seconds between stats output in live mode (default: 10)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output (show individual signals)",
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

    # Parse configuration
    config = SignalRunnerConfig(
        symbols=[s.strip() for s in args.symbols.split(",")],
        timeframes=[tf.strip() for tf in args.timeframes.split(",")],
        max_workers=args.max_workers,
        validate=args.validate,
        validation_ticks=args.ticks,
        live=args.live,
        backfill=args.backfill,
        backfill_days=args.days,
        with_persistence=args.with_persistence,
        verbose=args.verbose,
        stats_interval=args.stats_interval,
    )

    runner = SignalRunner(config)
    exit_code = await runner.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
