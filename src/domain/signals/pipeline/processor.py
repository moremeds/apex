"""
Signal Pipeline Processor.

Handles live and backfill signal processing.
"""

from __future__ import annotations

import asyncio
import signal
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

import pandas as pd

from src.application.services.ta_signal_service import TASignalService
from src.domain.events.event_types import EventType
from src.domain.events.priority_event_bus import PriorityEventBus
from src.domain.services.bar_count_calculator import BarCountCalculator
from src.utils.logging_setup import get_logger
from src.utils.timezone import DisplayTimezone, now_utc

from .config import SignalPipelineConfig

# Project root for resolving relative paths (works regardless of cwd)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

if TYPE_CHECKING:
    from src.application.orchestrator.signal_pipeline import BarPreloader
    from src.domain.interfaces.event_bus import EventBus
    from src.services.historical_data_manager import HistoricalDataManager

logger = get_logger(__name__)


class SignalPipelineProcessor:
    """
    Handles live and backfill signal processing.

    Responsibilities:
    - Initialize TASignalService and event bus
    - Run live signal processing on historical bars
    - Run backfill processing
    - Generate HTML reports
    - Deploy to GitHub Pages
    """

    def __init__(
        self,
        config: SignalPipelineConfig,
        persistence: Any = None,
    ) -> None:
        """
        Initialize processor.

        Args:
            config: Pipeline configuration.
            persistence: SignalPersistencePort implementation (injected).
                         If None and config.with_persistence, creation is skipped.
        """
        self.config = config
        self._event_bus: Optional[PriorityEventBus] = None
        self._service: Optional[TASignalService] = None
        self._persistence = persistence
        self._running = False
        self._signal_count = 0
        self._display_tz: Optional[DisplayTimezone] = None

    async def initialize(self) -> None:
        """Initialize pipeline components."""
        # Initialize display timezone from config
        try:
            from config.config_manager import ConfigManager

            config = ConfigManager().load()
            display_tz = config.display.timezone if config.display else "Asia/Hong_Kong"
        except Exception as e:
            logger.warning(f"Failed to load config for display timezone: {e}")
            display_tz = "Asia/Hong_Kong"
        self._display_tz = DisplayTimezone(display_tz)

        # Create event bus
        self._event_bus = PriorityEventBus()
        await self._event_bus.start()

        # Persistence is injected via constructor (if needed)

        # Create TASignalService
        self._service = TASignalService(
            event_bus=cast("EventBus", self._event_bus),
            persistence=self._persistence,
            timeframes=self.config.timeframes,
            max_workers=self.config.max_workers,
        )

        # Subscribe to signals for counting
        self._event_bus.subscribe(EventType.TRADING_SIGNAL, self._on_signal)

        await self._service.start()
        logger.info(f"SignalPipelineProcessor initialized: timeframes={self.config.timeframes}")

    async def shutdown(self) -> None:
        """Shutdown pipeline components."""
        if self._service:
            await self._service.stop()

        if self._event_bus:
            await self._event_bus.stop()

        logger.info(f"SignalPipelineProcessor stopped: signals_received={self._signal_count}")

    def _on_signal(self, payload: Any) -> None:
        """Handle trading signal event."""
        self._signal_count += 1

        signal_type = getattr(payload, "direction", "unknown")
        symbol = getattr(payload, "symbol", "unknown")
        indicator = getattr(payload, "indicator", "unknown")
        strength = getattr(payload, "strength", 0)
        timestamp = getattr(payload, "timestamp", None)

        if self.config.verbose:
            if timestamp and self._display_tz:
                ts_str = self._display_tz.format_with_tz(timestamp, "%H:%M:%S %Z")
            else:
                ts_str = now_utc().strftime("%H:%M:%S")
            print(
                f"  SIGNAL [{ts_str}]: {signal_type.upper()} {symbol} [{indicator}] strength={strength}"
            )

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        from types import FrameType

        def handle_signal(signum: int, frame: Optional[FrameType]) -> None:
            logger.info(f"Received signal {signum}, shutting down...")
            self._running = False

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

    def _get_last_trading_day(self) -> pd.Timestamp:
        """
        Get the last complete trading day's close time.

        Uses BarCountCalculator with NYSE calendar to get the actual
        market close time (handles DST correctly).

        Returns:
            pd.Timestamp: UTC-aware market close of the previous trading day.
        """
        calculator = BarCountCalculator("NYSE")
        prev_date = calculator.get_previous_trading_day()
        sessions = calculator.get_trading_sessions(prev_date, prev_date)
        if sessions:
            return pd.Timestamp(sessions[0].market_close)
        # Fallback: assume 21:00 UTC (EST close)
        return pd.Timestamp(prev_date, tz="UTC") + pd.Timedelta(hours=21)

    def _get_intraday_end(self) -> pd.Timestamp:
        """
        Get the end timestamp for data fetching, including today's bars when available.

        Logic:
        - Not a trading day → previous trading day close
        - Before market open → previous trading day close
        - After market close → today's close
        - During session → current time (floor to last completed bar)

        Returns:
            pd.Timestamp: UTC-aware end timestamp.
        """
        calculator = BarCountCalculator("NYSE")
        today = date.today()

        if not calculator.is_trading_day(today):
            return self._get_last_trading_day()

        sessions = calculator.get_trading_sessions(today, today)
        if not sessions:
            return self._get_last_trading_day()

        session = sessions[0]
        now = datetime.now(timezone.utc)

        if now < session.market_open:
            return self._get_last_trading_day()
        elif now >= session.market_close:
            return pd.Timestamp(session.market_close)
        else:
            return pd.Timestamp(now)

    # -------------------------------------------------------------------------
    # Live Mode
    # -------------------------------------------------------------------------

    async def run_live(self) -> int:
        """
        Run pipeline on historical bars from IB/Yahoo.

        Uses HistoricalDataManager to fetch past session bars,
        injects them into IndicatorEngine, and computes signals.

        Returns:
            Exit code (0 for success).
        """
        self._print_banner()
        self._running = True
        self._setup_signal_handlers()

        # Initialize data sources and preloader
        historical_manager = await self._create_historical_manager()
        bar_preloader = self._create_bar_preloader(historical_manager)

        # Run pipeline
        preload_started = time.monotonic()
        _ = await self._preload_bars(bar_preloader)
        preload_seconds = time.monotonic() - preload_started
        await self._wait_for_signals()
        self._print_stats()
        print(f"  historical_preload_seconds: {preload_seconds:.2f}")

        return 0

    def _print_banner(self) -> None:
        """Print pipeline startup banner."""
        print("=" * 60)
        print("SIGNAL PIPELINE (Historical Bars)")
        print("=" * 60)
        print(f"Symbols:     {', '.join(self.config.symbols)}")
        print(f"Timeframes:  {', '.join(self.config.timeframes)}")
        print(f"Persistence: {'enabled' if self.config.with_persistence else 'disabled'}")
        print("=" * 60)

    async def _create_historical_manager(
        self, ib_adapter: Any = None
    ) -> "HistoricalDataManager":
        """Create and configure historical data manager with config-driven source priority.

        Args:
            ib_adapter: Optional pre-configured IB historical adapter (injected from
                        application layer to avoid domain→infrastructure import).
        """
        from src.services.historical_data_manager import HistoricalDataManager

        # Read source priority from config
        try:
            from config.config_manager import ConfigManager

            config = ConfigManager().load()
            source_priority = (
                config.historical_data.source_priority
                if config.historical_data
                else ["fmp", "yahoo"]
            )
        except Exception:
            source_priority = ["fmp", "yahoo"]

        historical_manager = HistoricalDataManager(
            base_dir=PROJECT_ROOT / "data/historical",
            source_priority=source_priority,
        )

        # Use injected IB adapter if provided
        if ib_adapter and "ib" in source_priority:
            try:
                await ib_adapter.connect()
                historical_manager.set_ib_source(ib_adapter)
                logger.info("IB historical adapter connected")
            except Exception as e:
                logger.warning(f"IB not available: {e}")

        logger.info("Historical data sources: %s", source_priority)

        return historical_manager

    def _create_bar_preloader(self, historical_manager: "HistoricalDataManager") -> "BarPreloader":
        """Create bar preloader with indicator engine from TASignalService."""
        from src.application.orchestrator.signal_pipeline import BarPreloader

        assert self._service is not None, "TASignalService not initialized"
        assert self._service._indicator_engine is not None, "IndicatorEngine not initialized"

        return BarPreloader(
            historical_data_manager=historical_manager,
            indicator_engine=self._service._indicator_engine,
            timeframes=self.config.timeframes,
            preload_config={
                "lookback_days": 365,
                "slow_preload_warn_sec": 30,
                "preload_concurrency": self.config.preload_concurrency,
            },
        )

    async def _preload_bars(self, bar_preloader: "BarPreloader") -> Dict[str, int]:
        """Preload historical bars and return results per symbol."""
        print(f"\nLoading historical bars for {len(self.config.symbols)} symbols...")
        results = await bar_preloader.preload_startup(self.config.symbols)

        total_bars = sum(results.values())
        print(f"\nLoaded {total_bars} bars across {len(results)} symbols")
        for symbol, count in results.items():
            print(f"  {symbol}: {count} bars")

        return results

    async def _wait_for_signals(self) -> None:
        """Allow event bus to dispatch signals."""
        print("\nProcessing signals...")
        for _ in range(20):
            await asyncio.sleep(0.1)

    def _print_stats(self) -> None:
        """Print final pipeline statistics."""
        if not self._service:
            return
        stats = self._service.stats
        time_str = (
            self._display_tz.format_with_tz(now_utc(), "%H:%M:%S %Z")
            if self._display_tz
            else now_utc().strftime("%H:%M:%S")
        )
        print(f"\n[{time_str}] Pipeline complete:")
        print(f"  Bars processed:      {stats['bars_processed']}")
        print(f"  Indicators computed: {stats['indicators_computed']}")
        print(f"  Signals emitted:     {stats['signals_emitted']}")
        print(f"  Signals received:    {self._signal_count}")

    # -------------------------------------------------------------------------
    # Backfill Mode
    # -------------------------------------------------------------------------

    async def run_backfill(self) -> int:
        """
        Run backfill to process historical bars.

        Fetches historical bar data and processes each bar through the
        indicator engine, triggering persistence for each.

        Returns:
            Exit code (0 for success).
        """
        from datetime import datetime as dt

        from src.domain.events.domain_events import BarCloseEvent

        assert self._service is not None, "TASignalService not initialized"
        assert self._event_bus is not None, "EventBus not initialized"

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

                # First inject all bars for warmup
                await self._service.inject_historical_bars(
                    symbol=symbol,
                    timeframe=timeframe,
                    bars=bars,
                )

                # Emit BAR_CLOSE for each bar
                for i, bar_dict in enumerate(bars):
                    if not self._running:
                        break

                    raw_timestamp = bar_dict.get("timestamp")
                    if isinstance(raw_timestamp, str):
                        timestamp = dt.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
                    elif isinstance(raw_timestamp, dt):
                        timestamp = raw_timestamp
                    else:
                        continue

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

                    if (i + 1) % 50 == 0 or i == len(bars) - 1:
                        print(f"    Processed {i + 1}/{len(bars)} bars")

                    await asyncio.sleep(0.001)

        # Wait for pipeline to finish
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
        Fetch historical bars from data provider.

        Uses cache-first approach with HistoricalDataManager.
        """
        from src.services.historical_data_manager import HistoricalDataManager

        end_date = self._get_intraday_end()
        start_date = end_date - timedelta(days=days)

        # Read source priority from config
        try:
            from config.config_manager import ConfigManager

            config = ConfigManager().load()
            source_priority = (
                config.historical_data.source_priority
                if config.historical_data
                else ["fmp", "yahoo"]
            )
        except Exception:
            source_priority = ["fmp", "yahoo"]

        manager = HistoricalDataManager(
            base_dir=PROJECT_ROOT / "data/historical",
            source_priority=source_priority,
        )

        # IB adapter should be injected from the application layer if needed
        # (domain code does not import infrastructure adapters directly)

        bars = await manager.ensure_data(
            symbol=symbol,
            timeframe=timeframe,
            start=start_date,
            end=end_date,
        )

        if not bars:
            raise ValueError(
                f"No historical data available for {symbol}/{timeframe}. "
                f"Ensure IB is connected or data exists in data/historical/{symbol}/"
            )

        return [bar.to_dict() for bar in bars]

    @property
    def service(self) -> Optional[TASignalService]:
        """Get the TASignalService instance."""
        return self._service

    @property
    def event_bus(self) -> Optional[PriorityEventBus]:
        """Get the event bus instance."""
        return self._event_bus

    @property
    def signal_count(self) -> int:
        """Get the count of signals received."""
        return self._signal_count
