"""
Signal Runner - Standalone TA signal pipeline for live and backfill processing.

Enables running the signal pipeline independently from the TUI for:
- Live signal generation without TUI overhead
- Historical bar backfill for indicator warmup

Usage:
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
import signal
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from ..domain.events.event_types import EventType
from ..domain.events.priority_event_bus import PriorityEventBus
from ..application.services.ta_signal_service import TASignalService
from ..utils.logging_setup import get_logger
from ..utils.timezone import now_utc, DisplayTimezone

logger = get_logger(__name__)


@dataclass
class SignalRunnerConfig:
    """Configuration for signal runner."""

    symbols: List[str]
    timeframes: List[str]
    max_workers: int = 4

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
    html_output: Optional[str] = None  # Path for HTML report generation


class SignalRunner:
    """
    Standalone signal pipeline runner.

    Can run TASignalService independently for:
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
        self._display_tz: Optional[DisplayTimezone] = None

    async def run(self) -> int:
        """
        Run the signal pipeline based on configuration.

        Returns:
            Exit code (0 for success, non-zero for errors).
        """
        try:
            await self._initialize()

            if self.config.backfill:
                return await self._run_backfill()
            elif self.config.live:
                return await self._run_live()
            else:
                print("No mode specified. Use --live or --backfill")
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
        # Initialize display timezone from config (same pattern as header.py)
        try:
            from config.config_manager import ConfigManager
            config = ConfigManager().load()
            display_tz = config.display.get("timezone", "Asia/Hong_Kong")
        except Exception as e:
            logger.warning(f"Failed to load config for display timezone: {e}")
            display_tz = "Asia/Hong_Kong"
        self._display_tz = DisplayTimezone(display_tz)

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
            from config.config_manager import ConfigManager
            from ..infrastructure.persistence.database import get_database
            from ..infrastructure.persistence.repositories.ta_signal_repository import TASignalRepository

            config = ConfigManager().load()
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
        timestamp = getattr(payload, "timestamp", None)

        if self.config.verbose:
            # Format timestamp with timezone (same pattern as header.py)
            if timestamp and self._display_tz:
                ts_str = self._display_tz.format_with_tz(timestamp, "%H:%M:%S %Z")
            else:
                ts_str = now_utc().strftime("%H:%M:%S")
            print(f"  SIGNAL [{ts_str}]: {signal_type.upper()} {symbol} [{indicator}] strength={strength}")

    # -------------------------------------------------------------------------
    # Live Mode
    # -------------------------------------------------------------------------

    async def _run_live(self) -> int:
        """
        Run pipeline on historical bars from IB/Yahoo.

        Uses HistoricalDataManager to fetch past session bars,
        injects them into IndicatorEngine, and computes signals.

        Returns:
            Exit code (0 for success).
        """
        from pathlib import Path
        from ..services.historical_data_manager import HistoricalDataManager
        from ..application.orchestrator.signal_pipeline import BarPreloader

        print("=" * 60)
        print("SIGNAL PIPELINE (Historical Bars)")
        print("=" * 60)
        print(f"Symbols:     {', '.join(self.config.symbols)}")
        print(f"Timeframes:  {', '.join(self.config.timeframes)}")
        print(f"Persistence: {'enabled' if self.config.with_persistence else 'disabled'}")
        print("=" * 60)

        self._running = True
        self._setup_signal_handlers()

        # Create historical data manager
        historical_manager = HistoricalDataManager(
            base_dir=Path("data/historical"),
            source_priority=["ib", "yahoo"],
        )

        # Try to set IB source for better data quality
        try:
            from config.config_manager import ConfigManager
            from ..infrastructure.adapters.ib.historical_adapter import IbHistoricalAdapter

            config = ConfigManager().load()
            ib_config = config.ibkr

            if ib_config.enabled:
                ib_adapter = IbHistoricalAdapter(
                    host=ib_config.host,
                    port=ib_config.port,
                    client_id=ib_config.client_ids.historical_pool[0] if ib_config.client_ids.historical_pool else 3,
                )
                await ib_adapter.connect()
                historical_manager.set_ib_source(ib_adapter)
                print(f"Connected to IB at {ib_config.host}:{ib_config.port} for historical data")
        except Exception as e:
            logger.warning(f"IB not available, using Yahoo only: {e}")
            print("Using Yahoo Finance for historical data (IB unavailable)")

        # Create bar preloader with indicator engine from TASignalService
        bar_preloader = BarPreloader(
            historical_data_manager=historical_manager,
            indicator_engine=self._service._indicator_engine,
            timeframes=self.config.timeframes,
            preload_config={
                "lookback_days": 365,
                "slow_preload_warn_sec": 30,
            },
        )

        # Preload historical bars and compute indicators
        print(f"\nLoading historical bars for {len(self.config.symbols)} symbols...")
        results = await bar_preloader.preload_startup(self.config.symbols)

        total_bars = sum(results.values())
        print(f"\nLoaded {total_bars} bars across {len(results)} symbols")
        for symbol, count in results.items():
            print(f"  {symbol}: {count} bars")

        # Allow event bus to dispatch INDICATOR_UPDATE -> RuleEngine -> TRADING_SIGNAL
        print("\nProcessing signals...")
        # Yield to event loop multiple times to ensure dispatch tasks run
        for _ in range(20):
            await asyncio.sleep(0.1)

        # Show final stats
        stats = self._service.stats
        time_str = self._display_tz.format_with_tz(now_utc(), "%H:%M:%S %Z") if self._display_tz else now_utc().strftime("%H:%M:%S")
        print(f"\n[{time_str}] Pipeline complete:")
        print(f"  Bars processed:      {stats['bars_processed']}")
        print(f"  Indicators computed: {stats['indicators_computed']}")
        print(f"  Signals emitted:     {stats['signals_emitted']}")
        print(f"  Signals received:    {self._signal_count}")

        # Generate HTML report if requested
        if self.config.html_output:
            await self._generate_html_report(historical_manager, self.config.html_output)

        return 0

    # -------------------------------------------------------------------------
    # Backfill Mode
    # -------------------------------------------------------------------------

    async def _run_backfill(self) -> int:
        """
        Run backfill to process historical bars.

        Fetches historical bar data and processes each bar through the
        indicator engine, triggering persistence for each.

        Returns:
            Exit code (0 for success).
        """
        from ..domain.events.domain_events import BarCloseEvent

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
        total_processed = 0

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

                total_bars += len(bars)
                print(f"  Found {len(bars)} bars, processing...")

                # First inject all bars for warmup (needed for indicator lookback)
                await self._service.inject_historical_bars(
                    symbol=symbol,
                    timeframe=timeframe,
                    bars=bars,
                )

                # Now emit BAR_CLOSE for each bar to trigger indicator computation
                for i, bar_dict in enumerate(bars):
                    if not self._running:
                        break

                    # Convert timestamp string to datetime if needed
                    timestamp = bar_dict.get("timestamp")
                    if isinstance(timestamp, str):
                        from datetime import datetime as dt
                        timestamp = dt.fromisoformat(timestamp.replace("Z", "+00:00"))

                    bar_event = BarCloseEvent(
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=timestamp,
                        open=float(bar_dict.get("open", 0)),
                        high=float(bar_dict.get("high", 0)),
                        low=float(bar_dict.get("low", 0)),
                        close=float(bar_dict.get("close", 0)),
                        volume=int(bar_dict.get("volume", 0)),
                    )

                    self._event_bus.publish(EventType.BAR_CLOSE, bar_event)
                    total_processed += 1

                    # Progress update
                    if (i + 1) % 50 == 0 or i == len(bars) - 1:
                        print(f"    Processed {i + 1}/{len(bars)} bars")

                    # Small delay to allow event processing
                    await asyncio.sleep(0.001)

        # Wait for pipeline to finish processing
        print("\nWaiting for pipeline to complete...")
        await asyncio.sleep(2.0)

        # Report results
        print()
        print("=" * 60)
        print("BACKFILL RESULTS")
        print("=" * 60)
        stats = self._service.stats
        print(f"Total bars found:     {total_bars}")
        print(f"Bars processed:       {total_processed}")
        print(f"Indicators computed:  {stats['indicators_computed']}")
        print(f"Signals emitted:      {stats['signals_emitted']}")
        print("=" * 60)

        return 0 if total_processed > 0 else 1

    async def _fetch_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        days: int,
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical bars from data provider using HistoricalDataManager.

        Uses cache-first approach: reads from Parquet cache, downloads missing
        data from IB/Yahoo if needed, and saves to cache for future use.

        Args:
            symbol: Trading symbol.
            timeframe: Bar timeframe.
            days: Number of days of history.

        Returns:
            List of bar dicts with open, high, low, close, volume, timestamp.

        Raises:
            ValueError: If no historical data is available after attempting download.
        """
        from pathlib import Path
        from ..services.historical_data_manager import HistoricalDataManager

        end_date = now_utc()
        start_date = end_date - timedelta(days=days)

        # Use HistoricalDataManager for cache-first + download flow
        manager = HistoricalDataManager(
            base_dir=Path("data/historical"),
            source_priority=["ib", "yahoo"],
        )

        # Try to set up IB source for better data quality
        try:
            from config.config_manager import ConfigManager
            from ..infrastructure.adapters.ib.historical_adapter import IbHistoricalAdapter

            config = ConfigManager().load()
            ib_config = config.ibkr

            if ib_config.enabled:
                ib_adapter = IbHistoricalAdapter(
                    host=ib_config.host,
                    port=ib_config.port,
                    client_id=ib_config.client_ids.historical_pool[0] if ib_config.client_ids.historical_pool else 3,
                )
                await ib_adapter.connect()
                manager.set_ib_source(ib_adapter)
        except Exception as e:
            logger.warning(f"IB not available for backfill, using Yahoo: {e}")

        # ensure_data: check cache -> download gaps -> save to parquet -> return bars
        bars = await manager.ensure_data(
            symbol=symbol,
            timeframe=timeframe,
            start=start_date.replace(tzinfo=None),
            end=end_date.replace(tzinfo=None),
        )

        if not bars:
            raise ValueError(
                f"No historical data available for {symbol}/{timeframe}. "
                f"Ensure IB is connected or data exists in data/historical/{symbol}/"
            )

        return [bar.to_dict() for bar in bars]

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def handle_signal(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self._running = False

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

    # -------------------------------------------------------------------------
    # HTML Report Generation
    # -------------------------------------------------------------------------

    async def _generate_html_report(
        self,
        historical_manager: Any,
        output_path: str,
    ) -> None:
        """
        Generate HTML report with charts for all symbols/timeframes.

        Loads historical data into DataFrames, computes indicators,
        and generates an interactive HTML report.

        Args:
            historical_manager: HistoricalDataManager instance with IB/Yahoo sources.
            output_path: Path to save the HTML report.
        """
        from pathlib import Path
        from typing import Tuple
        from ..domain.signals.reporting import SignalReportGenerator
        from ..domain.signals.rules import ALL_RULES

        print(f"\nGenerating HTML report...")

        # Load historical data into DataFrames
        data: Dict[Tuple[str, str], pd.DataFrame] = {}
        end = now_utc().replace(tzinfo=None)  # Naive for historical manager

        for symbol in self.config.symbols:
            for tf in self.config.timeframes:
                start = end - timedelta(days=365)  # 1 year of history
                try:
                    bars = await historical_manager.ensure_data(symbol, tf, start, end)
                    if bars:
                        records = [
                            {
                                "timestamp": getattr(b, "bar_start", None) or b.timestamp,
                                "open": b.open,
                                "high": b.high,
                                "low": b.low,
                                "close": b.close,
                                "volume": b.volume,
                            }
                            for b in bars
                        ]
                        df = pd.DataFrame(records)
                        df.set_index("timestamp", inplace=True)
                        df = df.tail(100)  # Last 100 bars for report
                        data[(symbol, tf)] = df
                except Exception as e:
                    logger.warning(f"Failed to load {symbol}/{tf} for report: {e}")

        if not data:
            print("  No data available for HTML report")
            return

        # Compute indicators on DataFrames
        indicators = list(self._service._indicator_engine._indicators) if self._service._indicator_engine else []
        for key, df in data.items():
            data[key] = self._compute_indicators_on_df(df, indicators)

        # Generate report
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        generator = SignalReportGenerator(theme="dark")
        report_path = generator.generate(
            data=data,
            indicators=indicators,
            rules=ALL_RULES,
            output_path=output,
        )
        print(f"  Report saved: {report_path}")

    def _compute_indicators_on_df(
        self,
        df: pd.DataFrame,
        indicators: List[Any],
    ) -> pd.DataFrame:
        """
        Compute all indicators on a DataFrame and merge columns.

        Each indicator's columns are prefixed with indicator name.
        """
        indicator_dfs = []

        for indicator in indicators:
            if len(df) < indicator.warmup_periods:
                continue
            try:
                ind_df = indicator.calculate(df, indicator.default_params)
                prefixed = ind_df.add_prefix(f"{indicator.name}_")
                indicator_dfs.append(prefixed)
            except Exception as e:
                logger.debug(f"Failed to compute {indicator.name}: {e}")

        if indicator_dfs:
            return pd.concat([df] + indicator_dfs, axis=1)
        return df.copy()


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for signal runner."""
    parser = argparse.ArgumentParser(
        description="Standalone TA signal pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with HTML report (default output: results/signals/signal_report.html)
  python -m src.runners.signal_runner --live --symbols AAPL TSLA QQQ

  # Custom HTML output path
  python -m src.runners.signal_runner --live --symbols AAPL --html-output my_report.html

  # Skip HTML report
  python -m src.runners.signal_runner --live --symbols AAPL --no-html

  # Backfill historical signals
  python -m src.runners.signal_runner --backfill --symbols AAPL --days 365
        """,
    )

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
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
        nargs="+",
        default=["AAPL"],
        help="Space-separated symbols (default: AAPL)",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["1d"],
        help="Space-separated timeframes (default: 1d)",
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

    # HTML report generation (default: enabled)
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip HTML report generation",
    )
    parser.add_argument(
        "--html-output",
        type=str,
        default="results/signals/signal_report.html",
        metavar="PATH",
        help="HTML report output path (default: results/signals/signal_report.html)",
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
        symbols=args.symbols,
        timeframes=args.timeframes,
        max_workers=args.max_workers,
        live=args.live,
        backfill=args.backfill,
        backfill_days=args.days,
        with_persistence=args.with_persistence,
        verbose=args.verbose,
        stats_interval=args.stats_interval,
        html_output=None if args.no_html else args.html_output,
    )

    runner = SignalRunner(config)
    exit_code = await runner.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
