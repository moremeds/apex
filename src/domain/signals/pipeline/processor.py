"""
Signal Pipeline Processor.

Handles live and backfill signal processing, report generation, and GitHub deployment.
Extracted from signal_runner.py for better modularity.
"""

from __future__ import annotations

import asyncio
import shutil
import signal
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import pandas as pd

from src.application.services.ta_signal_service import TASignalService
from src.domain.events.event_types import EventType
from src.domain.events.priority_event_bus import PriorityEventBus
from src.domain.services.bar_count_calculator import BarCountCalculator
from src.utils.logging_setup import get_logger
from src.utils.timezone import DisplayTimezone, now_utc

from .config import SignalPipelineConfig

if TYPE_CHECKING:
    from src.domain.interfaces.event_bus import EventBus

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

    def __init__(self, config: SignalPipelineConfig) -> None:
        """
        Initialize processor.

        Args:
            config: Pipeline configuration.
        """
        self.config = config
        self._event_bus: Optional[PriorityEventBus] = None
        self._service: Optional[TASignalService] = None
        self._persistence = None
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

        # Setup persistence if requested
        if self.config.with_persistence:
            self._persistence = await self._create_persistence()

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

    async def _create_persistence(self) -> Any:
        """Create persistence layer if database is available."""
        try:
            from config.config_manager import ConfigManager
            from src.infrastructure.persistence.database import get_database
            from src.infrastructure.persistence.repositories.ta_signal_repository import (
                TASignalRepository,
            )

            config = ConfigManager().load()
            db = await get_database(config.database)
            return TASignalRepository(db)
        except Exception as e:
            logger.warning(f"Could not initialize persistence: {e}")
            return None

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
        Get the last complete trading day.

        Uses BarCountCalculator with NYSE calendar.

        Returns:
            pd.Timestamp: Last complete trading day as naive datetime (UTC).
        """
        calculator = BarCountCalculator("NYSE")
        last_trading_date = calculator.get_previous_trading_day()
        return pd.Timestamp(
            year=last_trading_date.year,
            month=last_trading_date.month,
            day=last_trading_date.day,
            hour=21,
            minute=0,
            second=0,
        )

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
        from src.application.orchestrator.signal_pipeline import BarPreloader
        from src.services.historical_data_manager import HistoricalDataManager

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
            from src.infrastructure.adapters.ib.historical_adapter import IbHistoricalAdapter

            config = ConfigManager().load()
            ib_config = config.ibkr

            if ib_config.enabled:
                ib_adapter = IbHistoricalAdapter(
                    host=ib_config.host,
                    port=ib_config.port,
                    client_id=(
                        ib_config.client_ids.historical_pool[0]
                        if ib_config.client_ids.historical_pool
                        else 3
                    ),
                )
                await ib_adapter.connect()
                historical_manager.set_ib_source(ib_adapter)
                print(f"Connected to IB at {ib_config.host}:{ib_config.port} for historical data")
        except Exception as e:
            logger.warning(f"IB not available, using Yahoo only: {e}")
            print("Using Yahoo Finance for historical data (IB unavailable)")

        # Create bar preloader with indicator engine from TASignalService
        assert self._service is not None, "TASignalService not initialized"
        assert self._service._indicator_engine is not None, "IndicatorEngine not initialized"
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

        # Allow event bus to dispatch
        print("\nProcessing signals...")
        for _ in range(20):
            await asyncio.sleep(0.1)

        # Show final stats
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

        # Generate HTML report if requested
        if self.config.html_output:
            await self._generate_html_report(historical_manager, self.config.html_output)

        return 0

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

        end_date = self._get_last_trading_day()
        start_date = end_date - timedelta(days=days)

        manager = HistoricalDataManager(
            base_dir=Path("data/historical"),
            source_priority=["ib", "yahoo"],
        )

        # Try to set up IB source
        try:
            from config.config_manager import ConfigManager
            from src.infrastructure.adapters.ib.historical_adapter import IbHistoricalAdapter

            config = ConfigManager().load()
            ib_config = config.ibkr

            if ib_config.enabled:
                ib_adapter = IbHistoricalAdapter(
                    host=ib_config.host,
                    port=ib_config.port,
                    client_id=(
                        ib_config.client_ids.historical_pool[0]
                        if ib_config.client_ids.historical_pool
                        else 3
                    ),
                )
                await ib_adapter.connect()
                manager.set_ib_source(ib_adapter)
        except Exception as e:
            logger.warning(f"IB not available for backfill, using Yahoo: {e}")

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

        Args:
            historical_manager: HistoricalDataManager instance.
            output_path: Path to save the HTML report.
        """
        from src.domain.signals.indicators.registry import get_indicator_registry
        from src.domain.signals.reporting import SignalReportGenerator
        from src.domain.signals.reporting.package_builder import PackageBuilder
        from src.domain.signals.rules import ALL_RULES

        print(f"\nGenerating HTML report...")

        # Load historical data into DataFrames
        data: Dict[Tuple[str, str], pd.DataFrame] = {}
        end = self._get_last_trading_day()

        for symbol in self.config.symbols:
            for tf in self.config.timeframes:
                start = end - timedelta(days=550)
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
                        df = df.tail(350)  # Last 350 bars
                        data[(symbol, tf)] = df
                except Exception as e:
                    logger.warning(f"Failed to load {symbol}/{tf} for report: {e}")

        if not data:
            print("  No data available for HTML report")
            return

        # Get indicators
        indicators = []
        if self._service and self._service._indicator_engine:
            indicators = list(self._service._indicator_engine._indicators)
        if not indicators:
            logger.info("Indicator engine empty, loading from registry")
            registry = get_indicator_registry()
            indicators = registry.get_all()
            logger.info(f"Loaded {len(indicators)} indicators from registry")

        # Compute indicators on DataFrames
        for key, df in data.items():
            data[key] = self._compute_indicators_on_df(df, indicators)

        # Generate report based on format
        output = Path(output_path)

        if self.config.output_format == "package":
            # PR-02: Package format with lazy loading
            from src.application.services.regime_service import RegimeService

            # Calculate regime for each symbol
            regime_outputs = {}
            regime_service = RegimeService()
            market_benchmarks = {"QQQ", "SPY", "IWM", "DIA"}

            for symbol in self.config.symbols:
                daily_key = (symbol, "1d")
                if daily_key not in data:
                    for tf in ["1d", "1h", "4h"]:
                        if (symbol, tf) in data:
                            daily_key = (symbol, tf)
                            break

                if daily_key in data:
                    df_for_regime = data[daily_key]
                    warmup_needed = regime_service._regime_detector.warmup_periods
                    if len(df_for_regime) >= warmup_needed:
                        try:
                            regime_output = regime_service.calculate_regime(
                                symbol=symbol,
                                data=df_for_regime,
                                params=None,
                                is_market_level=(symbol in market_benchmarks),
                            )
                            regime_outputs[symbol] = regime_output
                            logger.info(
                                f"Calculated regime for {symbol}: {regime_output.final_regime.value} "
                                f"({regime_output.regime_name}, confidence={regime_output.confidence})"
                            )
                        except Exception as e:
                            logger.warning(f"Failed to calculate regime for {symbol}: {e}")
                    else:
                        logger.warning(
                            f"Insufficient data for {symbol} regime: {len(df_for_regime)} bars "
                            f"(need {warmup_needed})"
                        )

            print(f"  Calculated regime for {len(regime_outputs)} symbols")

            package_dir = output.with_suffix("")
            if package_dir.suffix == ".html":
                package_dir = package_dir.with_suffix("")

            builder = PackageBuilder(theme="dark")
            package_path = builder.build(
                data=data,
                indicators=indicators,
                rules=ALL_RULES,
                output_dir=package_dir,
                regime_outputs=regime_outputs,
                validation_url="validation.html",  # Link to validation summary
            )
            print(f"  Package saved: {package_path}")
            print(f"  To view: cd {package_path} && python -m http.server 8080")
            print(f"  Then open: http://localhost:8080")

            # Deploy to GitHub Pages if requested
            if self.config.deploy_github:
                self._deploy_to_github(package_path)
        else:
            # Legacy: Single HTML file
            output.parent.mkdir(parents=True, exist_ok=True)

            generator = SignalReportGenerator(theme="dark")
            report_path = generator.generate(
                data=data,
                indicators=indicators,
                rules=ALL_RULES,
                output_path=output,
            )
            print(f"  Report saved: {report_path}")

    def _deploy_to_github(self, package_path: Path) -> None:
        """
        Deploy package to GitHub Pages.

        Uses gh CLI if available, falls back to git commands.
        """
        print("\nDeploying to GitHub Pages...")

        repo = self.config.github_repo

        if repo:
            repo_url = f"https://github.com/{repo}.git"
            pages_url = f"https://{repo.split('/')[0]}.github.io/{repo.split('/')[1]}/"
        else:
            try:
                result = subprocess.run(
                    ["git", "remote", "get-url", "origin"],
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=package_path.parent.parent,
                )
                repo_url = result.stdout.strip()
                if "github.com" in repo_url:
                    if repo_url.startswith("git@"):
                        repo = repo_url.split(":")[1].replace(".git", "")
                    else:
                        repo = "/".join(repo_url.split("/")[-2:]).replace(".git", "")
                    pages_url = f"https://{repo.split('/')[0]}.github.io/{repo.split('/')[1]}/"
                else:
                    pages_url = "(check your repo settings)"
            except subprocess.CalledProcessError:
                print("  Error: Could not determine git remote. Use --github-repo to specify.")
                return

        with tempfile.TemporaryDirectory() as tmpdir:
            deploy_dir = Path(tmpdir) / "deploy"
            shutil.copytree(package_path, deploy_dir)

            try:
                subprocess.run(["git", "init"], cwd=deploy_dir, check=True, capture_output=True)
                subprocess.run(
                    ["git", "checkout", "-b", "gh-pages"],
                    cwd=deploy_dir,
                    check=True,
                    capture_output=True,
                )
                subprocess.run(["git", "add", "."], cwd=deploy_dir, check=True, capture_output=True)

                commit_msg = f"Deploy signal report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

                subprocess.run(
                    ["git", "commit", "-m", commit_msg],
                    cwd=deploy_dir,
                    check=True,
                    capture_output=True,
                )

                subprocess.run(
                    ["git", "push", "-f", repo_url, "gh-pages"],
                    cwd=deploy_dir,
                    check=True,
                    capture_output=True,
                )

                print(f"  Deployed to: {pages_url}")
                print("  Note: May take 1-2 minutes for GitHub Pages to update.")

            except subprocess.CalledProcessError as e:
                print(f"  Deployment failed: {e}")
                if e.stderr:
                    print(
                        f"  Error: {e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr}"
                    )
                print("  Tip: Make sure you have push access to the repository.")

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
        computed_count = 0
        skipped_count = 0

        for indicator in indicators:
            if len(df) < indicator.warmup_periods:
                logger.warning(
                    f"Skipping {indicator.name}: need {indicator.warmup_periods} bars, "
                    f"have {len(df)}"
                )
                skipped_count += 1
                continue
            try:
                ind_df = indicator.calculate(df, indicator.default_params)
                prefixed = ind_df.add_prefix(f"{indicator.name}_")
                indicator_dfs.append(prefixed)
                computed_count += 1
            except Exception as e:
                logger.warning(f"Failed to compute {indicator.name}: {e}")

        logger.info(
            f"Computed {computed_count} indicators, skipped {skipped_count} "
            f"(insufficient warmup)"
        )

        if indicator_dfs:
            return pd.concat([df] + indicator_dfs, axis=1)
        return df.copy()

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
