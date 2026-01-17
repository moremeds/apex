"""
Bar Replay Service - Replays historical bars through the signal pipeline.

Emits BAR_CLOSE events to PriorityEventBus to trigger:
IndicatorEngine -> RuleEngine -> TRADING_SIGNAL flow

Supports speed control, gap detection, and progress tracking.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from ...domain.events.domain_events import BarCloseEvent, BarData
from ...domain.events.event_types import EventType
from ...infrastructure.stores.parquet_historical_store import ParquetHistoricalStore
from ...utils.logging_setup import get_logger
from .bar_count_calculator import BarCountCalculator
from .data_validator import DataValidator, ValidationResult

logger = get_logger(__name__)


class ReplaySpeed(str, Enum):
    """Replay speed modes."""

    REALTIME = "realtime"  # 1x wall clock (actual bar intervals)
    FAST_FORWARD = "fast"  # Configurable multiplier (10x-100x)
    MAX_SPEED = "max"  # No delays (as fast as possible)
    STEP = "step"  # Manual single-bar advance


@dataclass
class ReplayProgress:
    """Current replay progress."""

    symbol: str
    timeframe: str
    total_bars: int
    replayed_bars: int
    current_timestamp: Optional[datetime]
    gaps_detected: int
    start_time: float
    elapsed_seconds: float = 0

    @property
    def progress_pct(self) -> float:
        """Progress percentage."""
        return (self.replayed_bars / self.total_bars * 100) if self.total_bars > 0 else 0

    @property
    def bars_per_second(self) -> float:
        """Replay rate in bars per second."""
        if self.elapsed_seconds > 0:
            return self.replayed_bars / self.elapsed_seconds
        return 0

    @property
    def remaining_bars(self) -> int:
        """Bars remaining to replay."""
        return self.total_bars - self.replayed_bars

    @property
    def eta_seconds(self) -> Optional[float]:
        """Estimated time remaining in seconds."""
        if self.bars_per_second > 0:
            return self.remaining_bars / self.bars_per_second
        return None


@dataclass
class ReplayGapEvent:
    """Detected gap during replay."""

    before_timestamp: datetime
    after_timestamp: datetime
    expected_bars: int
    gap_duration: timedelta

    @property
    def gap_hours(self) -> float:
        """Gap duration in hours."""
        return self.gap_duration.total_seconds() / 3600


class BarReplayService:
    """
    Replays historical bars through the signal pipeline via event bus.

    Reads bars from ParquetHistoricalStore and emits BAR_CLOSE events,
    triggering IndicatorEngine -> RuleEngine -> TRADING_SIGNAL flow.

    Features:
    - Configurable replay speed (realtime, fast-forward, max, step)
    - Gap detection during replay
    - Progress callbacks
    - Graceful pause/resume/stop

    Example:
        replay = BarReplayService(event_bus)
        progress = await replay.replay("AAPL", "5m", start, end, speed=ReplaySpeed.FAST_FORWARD)
        print(f"Replayed {progress.replayed_bars} bars")
    """

    # Timeframe intervals in seconds for gap detection
    TIMEFRAME_SECONDS: Dict[str, int] = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "4h": 14400,
        "1d": 86400,
        "1w": 604800,
    }

    def __init__(
        self,
        event_bus: Optional[object] = None,
        bar_store: Optional[ParquetHistoricalStore] = None,
        bar_calculator: Optional[BarCountCalculator] = None,
        base_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize replay service.

        Args:
            event_bus: PriorityEventBus for BAR_CLOSE events.
            bar_store: Source of historical bars.
            bar_calculator: For gap detection.
            base_dir: Base directory for bar store.
        """
        self._event_bus = event_bus
        self._bar_store = bar_store or ParquetHistoricalStore(
            base_dir=base_dir or Path("data/historical")
        )
        self._calculator = bar_calculator or BarCountCalculator()

        # State
        self._is_paused = False
        self._is_stopped = False
        self._current_bars: List[BarData] = []
        self._current_index = 0

    async def replay(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[date | datetime] = None,
        end: Optional[date | datetime] = None,
        speed: ReplaySpeed = ReplaySpeed.MAX_SPEED,
        speed_multiplier: float = 10.0,
        on_progress: Optional[Callable[[ReplayProgress], None]] = None,
        on_gap: Optional[Callable[[ReplayGapEvent], None]] = None,
        progress_interval: int = 100,
    ) -> ReplayProgress:
        """
        Replay bars for a symbol/timeframe range.

        Args:
            symbol: Ticker symbol.
            timeframe: Bar timeframe.
            start: Start date (default: all data).
            end: End date (default: all data).
            speed: Replay speed mode.
            speed_multiplier: Speed factor for FAST_FORWARD mode.
            on_progress: Progress callback (called every progress_interval bars).
            on_gap: Gap detection callback.
            progress_interval: Bars between progress callbacks.

        Returns:
            Final ReplayProgress with summary statistics.
        """
        self._is_stopped = False
        self._is_paused = False

        # Convert dates for filtering
        start_dt = None
        end_dt = None
        if start:
            start_dt = (
                datetime.combine(start, datetime.min.time())
                if isinstance(start, date) and not isinstance(start, datetime)
                else start
            )
        if end:
            end_dt = (
                datetime.combine(end, datetime.max.time())
                if isinstance(end, date) and not isinstance(end, datetime)
                else end
            )

        # Load bars
        bars = self._bar_store.read_bars(symbol, timeframe, start=start_dt, end=end_dt)
        if not bars:
            logger.warning(f"No bars found for {symbol}/{timeframe}")
            return ReplayProgress(
                symbol=symbol,
                timeframe=timeframe,
                total_bars=0,
                replayed_bars=0,
                current_timestamp=None,
                gaps_detected=0,
                start_time=time.time(),
            )

        self._current_bars = bars
        self._current_index = 0

        # Initialize progress
        start_time = time.time()
        progress = ReplayProgress(
            symbol=symbol,
            timeframe=timeframe,
            total_bars=len(bars),
            replayed_bars=0,
            current_timestamp=None,
            gaps_detected=0,
            start_time=start_time,
        )

        # Get timeframe interval for delays and gap detection
        interval_seconds = self.TIMEFRAME_SECONDS.get(timeframe, 86400)
        gap_tolerance = interval_seconds * 2.0  # Allow 2x interval before flagging gap

        last_bar: Optional[BarData] = None

        logger.info(f"Starting replay: {symbol}/{timeframe}, {len(bars)} bars, speed={speed.value}")

        # Replay loop
        for i, bar in enumerate(bars):
            if self._is_stopped:
                break

            # Handle pause
            while self._is_paused and not self._is_stopped:
                await asyncio.sleep(0.1)

            if self._is_stopped:
                break

            # Get timestamp
            bar_ts = bar.bar_start or bar.timestamp

            # Gap detection
            if last_bar:
                last_ts = last_bar.bar_start or last_bar.timestamp
                if bar_ts and last_ts:
                    actual_gap = (bar_ts - last_ts).total_seconds()
                    if actual_gap > gap_tolerance:
                        gap_event = ReplayGapEvent(
                            before_timestamp=last_ts,
                            after_timestamp=bar_ts,
                            expected_bars=max(1, int(actual_gap / interval_seconds) - 1),
                            gap_duration=timedelta(seconds=actual_gap),
                        )
                        progress.gaps_detected += 1
                        if on_gap:
                            on_gap(gap_event)

            # Emit BAR_CLOSE event
            self._emit_bar_close(bar)

            # Update progress
            progress.replayed_bars = i + 1
            progress.current_timestamp = bar_ts
            progress.elapsed_seconds = time.time() - start_time

            # Progress callback
            if on_progress and (i + 1) % progress_interval == 0:
                on_progress(progress)

            # Speed control
            if speed == ReplaySpeed.REALTIME and last_bar:
                last_ts = last_bar.bar_start or last_bar.timestamp
                if bar_ts and last_ts:
                    delay = (bar_ts - last_ts).total_seconds()
                    if delay > 0:
                        await asyncio.sleep(delay)
            elif speed == ReplaySpeed.FAST_FORWARD and last_bar:
                last_ts = last_bar.bar_start or last_bar.timestamp
                if bar_ts and last_ts:
                    delay = (bar_ts - last_ts).total_seconds() / speed_multiplier
                    if delay > 0:
                        await asyncio.sleep(delay)
            elif speed == ReplaySpeed.STEP:
                # In step mode, caller must call step() to advance
                self._is_paused = True
                await asyncio.sleep(0)

            last_bar = bar
            self._current_index = i + 1

        # Final progress update
        progress.elapsed_seconds = time.time() - start_time
        if on_progress:
            on_progress(progress)

        logger.info(
            f"Replay complete: {progress.replayed_bars}/{progress.total_bars} bars, "
            f"{progress.gaps_detected} gaps, {progress.elapsed_seconds:.1f}s"
        )

        return progress

    async def replay_all(
        self,
        symbols: List[str],
        timeframes: List[str],
        start: Optional[date | datetime] = None,
        end: Optional[date | datetime] = None,
        **kwargs,
    ) -> Dict[Tuple[str, str], ReplayProgress]:
        """
        Replay multiple symbol/timeframe combinations.

        Args:
            symbols: List of symbols.
            timeframes: List of timeframes.
            start: Start date.
            end: End date.
            **kwargs: Additional arguments for replay().

        Returns:
            Dict mapping (symbol, timeframe) to ReplayProgress.
        """
        results = {}
        for symbol in symbols:
            for tf in timeframes:
                results[(symbol, tf)] = await self.replay(symbol, tf, start, end, **kwargs)
        return results

    def pause(self) -> None:
        """Pause replay (can resume)."""
        self._is_paused = True
        logger.debug("Replay paused")

    def resume(self) -> None:
        """Resume paused replay."""
        self._is_paused = False
        logger.debug("Replay resumed")

    async def stop(self) -> None:
        """Stop replay."""
        self._is_stopped = True
        self._is_paused = False
        logger.debug("Replay stopped")

    async def step(self) -> Optional[BarCloseEvent]:
        """
        Step mode: emit exactly one bar and pause.

        Returns:
            The emitted BarCloseEvent, or None if no more bars.
        """
        if self._current_index >= len(self._current_bars):
            return None

        bar = self._current_bars[self._current_index]
        event = self._emit_bar_close(bar)
        self._current_index += 1
        self._is_paused = True
        return event

    def _emit_bar_close(self, bar: BarData) -> BarCloseEvent:
        """
        Convert BarData to BarCloseEvent and publish to event bus.

        Args:
            bar: Bar data to emit.

        Returns:
            The emitted BarCloseEvent.
        """
        bar_ts = bar.bar_start or bar.timestamp or datetime.now()

        event = BarCloseEvent(
            timestamp=bar_ts,
            symbol=bar.symbol,
            timeframe=bar.timeframe,
            open=bar.open or 0,
            high=bar.high or 0,
            low=bar.low or 0,
            close=bar.close or 0,
            volume=bar.volume or 0,
            bar_end=bar.bar_end or bar_ts,
        )

        # Publish to event bus if available
        if self._event_bus is not None:
            try:
                self._event_bus.publish(EventType.BAR_CLOSE, event)
            except Exception as e:
                logger.error(f"Failed to publish BAR_CLOSE: {e}")

        return event

    def get_validation(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[date | datetime] = None,
        end: Optional[date | datetime] = None,
    ) -> ValidationResult:
        """
        Get validation result for a symbol/timeframe before replay.

        Args:
            symbol: Ticker symbol.
            timeframe: Bar timeframe.
            start: Start date.
            end: End date.

        Returns:
            ValidationResult from DataValidator.
        """
        validator = DataValidator(
            bar_store=self._bar_store,
            bar_calculator=self._calculator,
        )
        return validator.validate(symbol, timeframe, start, end)
