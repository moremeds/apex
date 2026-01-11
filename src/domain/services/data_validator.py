"""
Data Validator - Validates historical bar data against expected counts and quality standards.

Compares actual bar data from Parquet storage against expected counts from BarCountCalculator.
Detects gaps, duplicates, out-of-order bars, and price anomalies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from ...domain.events.domain_events import BarData
from ...infrastructure.stores.parquet_historical_store import ParquetHistoricalStore
from ...utils.logging_setup import get_logger
from .bar_count_calculator import BarCountCalculator

logger = get_logger(__name__)


class ValidationStatus(str, Enum):
    """Validation status levels."""

    PASS = "PASS"         # >= 98% coverage
    WARN = "WARN"         # 95-98% coverage
    CAUTION = "CAUTION"   # 90-95% coverage
    FAIL = "FAIL"         # < 90% coverage


@dataclass
class ValidationGap:
    """A detected gap in bar data."""

    start: datetime
    end: datetime
    expected_bars: int
    gap_type: Literal["missing", "weekend", "holiday", "after_hours"]

    @property
    def duration(self) -> timedelta:
        """Gap duration."""
        return self.end - self.start

    @property
    def duration_hours(self) -> float:
        """Gap duration in hours."""
        return self.duration.total_seconds() / 3600


@dataclass
class ValidationAnomaly:
    """Data quality anomaly."""

    anomaly_type: Literal["duplicate", "out_of_order", "price_spike", "zero_volume", "ohlc_invalid"]
    timestamp: datetime
    details: str
    severity: Literal["low", "medium", "high"]


@dataclass
class ValidationResult:
    """Complete validation result for a symbol/timeframe."""

    symbol: str
    timeframe: str
    start: date
    end: date

    # Counts
    expected_bars: int
    actual_bars: int
    trading_days: int
    early_close_days: int

    # Coverage
    coverage_pct: float
    status: ValidationStatus

    # Gaps
    gaps: List[ValidationGap] = field(default_factory=list)

    # Anomalies
    anomalies: List[ValidationAnomaly] = field(default_factory=list)

    # Timing
    validated_at: datetime = field(default_factory=datetime.now)
    validation_duration_ms: float = 0

    @property
    def gap_count(self) -> int:
        """Number of gaps detected."""
        return len(self.gaps)

    @property
    def total_gap_bars(self) -> int:
        """Total bars missing across all gaps."""
        return sum(g.expected_bars for g in self.gaps)

    @property
    def duplicate_count(self) -> int:
        """Number of duplicate anomalies."""
        return sum(1 for a in self.anomalies if a.anomaly_type == "duplicate")

    @property
    def out_of_order_count(self) -> int:
        """Number of out-of-order anomalies."""
        return sum(1 for a in self.anomalies if a.anomaly_type == "out_of_order")

    @property
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return self.status in (ValidationStatus.PASS, ValidationStatus.WARN)

    def __str__(self) -> str:
        """Human-readable summary."""
        return (
            f"{self.symbol}/{self.timeframe}: {self.status.value} "
            f"({self.coverage_pct:.1f}% coverage, {self.actual_bars}/{self.expected_bars} bars)"
        )


class DataValidator:
    """
    Validates historical bar data against expected counts and quality standards.

    Uses BarCountCalculator for expected bar counts and ParquetHistoricalStore
    for actual bar data. Detects:
    - Missing bars (gaps)
    - Duplicate timestamps
    - Out-of-order bars
    - Price anomalies (spikes, zeros)
    - OHLC integrity violations

    Example:
        validator = DataValidator()
        result = validator.validate("AAPL", "1d", date(2024, 1, 1), date(2024, 12, 31))
        print(result)
    """

    # Coverage thresholds
    COMPLETE_THRESHOLD = 0.98
    ACCEPTABLE_THRESHOLD = 0.95
    CAUTION_THRESHOLD = 0.90

    # Price spike threshold (10% change in one bar)
    PRICE_SPIKE_THRESHOLD = 0.10

    # Timeframe intervals in seconds
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
        bar_store: Optional[ParquetHistoricalStore] = None,
        bar_calculator: Optional[BarCountCalculator] = None,
        base_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize validator.

        Args:
            bar_store: Parquet store for bar data.
            bar_calculator: Calculator for expected bars.
            base_dir: Base directory for bar store (if bar_store not provided).
        """
        self._bar_store = bar_store or ParquetHistoricalStore(
            base_dir=base_dir or Path("data/historical")
        )
        self._calculator = bar_calculator or BarCountCalculator()

    def validate(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[date | datetime] = None,
        end: Optional[date | datetime] = None,
    ) -> ValidationResult:
        """
        Validate data coverage and quality for a symbol/timeframe.

        Args:
            symbol: Ticker symbol.
            timeframe: Bar timeframe.
            start: Start date (default: from actual data).
            end: End date (default: from actual data).

        Returns:
            ValidationResult with coverage metrics and detected issues.
        """
        start_time = time.time()

        # Convert dates for filtering
        start_dt = None
        end_dt = None
        if start:
            start_dt = datetime.combine(start, datetime.min.time()) if isinstance(start, date) and not isinstance(start, datetime) else start
        if end:
            end_dt = datetime.combine(end, datetime.max.time()) if isinstance(end, date) and not isinstance(end, datetime) else end

        # Get bars filtered by date range
        bars = self._bar_store.read_bars(symbol, timeframe, start=start_dt, end=end_dt)

        if not bars:
            return ValidationResult(
                symbol=symbol,
                timeframe=timeframe,
                start=start.date() if isinstance(start, datetime) else (start or date.today()),
                end=end.date() if isinstance(end, datetime) else (end or date.today()),
                expected_bars=0,
                actual_bars=0,
                trading_days=0,
                early_close_days=0,
                coverage_pct=0,
                status=ValidationStatus.FAIL,
                gaps=[],
                anomalies=[],
                validation_duration_ms=(time.time() - start_time) * 1000,
            )

        # Determine date range from data if not provided
        first_bar = bars[0]
        last_bar = bars[-1]
        first_ts = first_bar.bar_start or first_bar.timestamp
        last_ts = last_bar.bar_start or last_bar.timestamp

        start_date = start.date() if isinstance(start, datetime) else (start or first_ts.date())
        end_date = end.date() if isinstance(end, datetime) else (end or last_ts.date())

        # Calculate expected bars
        expected = self._calculator.calculate(symbol, timeframe, start_date, end_date)

        # Calculate coverage
        actual_count = len(bars)
        if expected.expected_bars > 0:
            coverage_pct = (actual_count / expected.expected_bars) * 100
        else:
            coverage_pct = 0 if actual_count == 0 else 100

        # Determine status
        status = self._determine_status(coverage_pct)

        # Detect gaps
        gaps = self.detect_gaps(bars, timeframe)

        # Detect anomalies
        anomalies = self.detect_anomalies(bars)

        return ValidationResult(
            symbol=symbol,
            timeframe=timeframe,
            start=start_date,
            end=end_date,
            expected_bars=expected.expected_bars,
            actual_bars=actual_count,
            trading_days=expected.trading_days,
            early_close_days=expected.early_close_days,
            coverage_pct=coverage_pct,
            status=status,
            gaps=gaps,
            anomalies=anomalies,
            validation_duration_ms=(time.time() - start_time) * 1000,
        )

    def validate_all(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
    ) -> Dict[Tuple[str, str], ValidationResult]:
        """
        Validate all symbol/timeframe combinations.

        Args:
            symbols: List of symbols (default: all in store).
            timeframes: List of timeframes (default: all standard).

        Returns:
            Dict mapping (symbol, timeframe) to ValidationResult.
        """
        symbols = symbols or self._bar_store.list_symbols()
        timeframes = timeframes or ["1d", "4h", "1h", "30m", "15m", "5m", "1m"]

        results = {}
        for symbol in symbols:
            available_timeframes = self._bar_store.list_timeframes(symbol)
            for tf in timeframes:
                if tf in available_timeframes:
                    results[(symbol, tf)] = self.validate(symbol, tf)

        return results

    def detect_gaps(
        self,
        bars: List[BarData],
        timeframe: str,
    ) -> List[ValidationGap]:
        """
        Detect gaps in bar sequence.

        Uses timeframe interval to identify missing bars.
        Excludes expected gaps (weekends, after-hours).

        Args:
            bars: List of bars sorted by timestamp.
            timeframe: Bar timeframe.

        Returns:
            List of ValidationGap.
        """
        if len(bars) < 2:
            return []

        gaps = []
        expected_interval = self.TIMEFRAME_SECONDS.get(timeframe, 86400)

        # Tolerance: allow 50% extra for market close gaps
        tolerance_factor = 1.5 if timeframe in ("1d", "1w") else 2.0

        for i in range(len(bars) - 1):
            current_bar = bars[i]
            next_bar = bars[i + 1]

            current_ts = current_bar.bar_start or current_bar.timestamp
            next_ts = next_bar.bar_start or next_bar.timestamp

            if current_ts is None or next_ts is None:
                continue

            actual_gap_seconds = (next_ts - current_ts).total_seconds()

            # Check if gap is significant
            if actual_gap_seconds > expected_interval * tolerance_factor:
                # Classify gap type
                gap_type = self._classify_gap(current_ts, next_ts, timeframe)

                # Only report non-expected gaps
                if gap_type == "missing":
                    expected_bars = max(1, int(actual_gap_seconds / expected_interval) - 1)
                    gaps.append(ValidationGap(
                        start=current_ts,
                        end=next_ts,
                        expected_bars=expected_bars,
                        gap_type=gap_type,
                    ))

        return gaps

    def detect_anomalies(
        self,
        bars: List[BarData],
    ) -> List[ValidationAnomaly]:
        """
        Detect data quality anomalies.

        Checks for:
        - Duplicate timestamps
        - Out-of-order bars
        - Price spikes (> 10% in one bar)
        - Zero/negative volume
        - OHLC integrity (H >= L, etc.)

        Args:
            bars: List of bars.

        Returns:
            List of ValidationAnomaly.
        """
        if not bars:
            return []

        anomalies = []
        seen_timestamps = set()
        last_ts = None
        last_close = None

        for bar in bars:
            ts = bar.bar_start or bar.timestamp

            # Check for duplicates
            if ts:
                ts_key = ts.isoformat()
                if ts_key in seen_timestamps:
                    anomalies.append(ValidationAnomaly(
                        anomaly_type="duplicate",
                        timestamp=ts,
                        details=f"Duplicate timestamp: {ts}",
                        severity="medium",
                    ))
                else:
                    seen_timestamps.add(ts_key)

            # Check for out-of-order
            if last_ts and ts and ts < last_ts:
                anomalies.append(ValidationAnomaly(
                    anomaly_type="out_of_order",
                    timestamp=ts,
                    details=f"Out of order: {ts} < {last_ts}",
                    severity="high",
                ))

            # Check for price spike
            if last_close and bar.close and last_close > 0:
                change_pct = abs(bar.close - last_close) / last_close
                if change_pct > self.PRICE_SPIKE_THRESHOLD:
                    anomalies.append(ValidationAnomaly(
                        anomaly_type="price_spike",
                        timestamp=ts or datetime.now(),
                        details=f"Price spike: {change_pct*100:.1f}% change ({last_close} -> {bar.close})",
                        severity="medium" if change_pct < 0.20 else "high",
                    ))

            # Check for zero volume (for intraday)
            if bar.volume is not None and bar.volume == 0:
                anomalies.append(ValidationAnomaly(
                    anomaly_type="zero_volume",
                    timestamp=ts or datetime.now(),
                    details="Zero volume bar",
                    severity="low",
                ))

            # Check OHLC integrity
            if bar.high is not None and bar.low is not None:
                if bar.high < bar.low:
                    anomalies.append(ValidationAnomaly(
                        anomaly_type="ohlc_invalid",
                        timestamp=ts or datetime.now(),
                        details=f"High ({bar.high}) < Low ({bar.low})",
                        severity="high",
                    ))

            if bar.open is not None and bar.high is not None and bar.low is not None:
                if bar.open > bar.high or bar.open < bar.low:
                    anomalies.append(ValidationAnomaly(
                        anomaly_type="ohlc_invalid",
                        timestamp=ts or datetime.now(),
                        details=f"Open ({bar.open}) outside H/L range ({bar.low}-{bar.high})",
                        severity="medium",
                    ))

            if bar.close is not None and bar.high is not None and bar.low is not None:
                if bar.close > bar.high or bar.close < bar.low:
                    anomalies.append(ValidationAnomaly(
                        anomaly_type="ohlc_invalid",
                        timestamp=ts or datetime.now(),
                        details=f"Close ({bar.close}) outside H/L range ({bar.low}-{bar.high})",
                        severity="medium",
                    ))

            last_ts = ts
            last_close = bar.close

        return anomalies

    def _determine_status(self, coverage_pct: float) -> ValidationStatus:
        """Determine validation status from coverage percentage."""
        if coverage_pct >= self.COMPLETE_THRESHOLD * 100:
            return ValidationStatus.PASS
        elif coverage_pct >= self.ACCEPTABLE_THRESHOLD * 100:
            return ValidationStatus.WARN
        elif coverage_pct >= self.CAUTION_THRESHOLD * 100:
            return ValidationStatus.CAUTION
        else:
            return ValidationStatus.FAIL

    def _classify_gap(
        self,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> Literal["missing", "weekend", "holiday", "after_hours"]:
        """
        Classify a gap between bars.

        Args:
            start: Gap start timestamp.
            end: Gap end timestamp.
            timeframe: Bar timeframe.

        Returns:
            Gap classification.
        """
        # Check if gap spans a weekend
        if start.weekday() == 4 and end.weekday() == 0:  # Friday to Monday
            return "weekend"

        # For daily bars, check if gap is more than 3 days (holiday)
        if timeframe in ("1d", "1w"):
            gap_days = (end.date() - start.date()).days
            if gap_days > 3:
                return "holiday"

        # For intraday, check if gap is overnight
        if timeframe in ("1m", "5m", "15m", "30m", "1h", "4h"):
            # If end time is morning and start time is afternoon, it's overnight
            if end.hour < 10 and start.hour >= 15:
                return "after_hours"

        return "missing"

    def generate_report(
        self,
        results: Dict[Tuple[str, str], ValidationResult],
        format: Literal["text", "json"] = "text",
    ) -> str:
        """
        Generate summary report from validation results.

        Args:
            results: Dict of validation results.
            format: Output format.

        Returns:
            Formatted report string.
        """
        if format == "json":
            import json
            return json.dumps({
                f"{sym}/{tf}": {
                    "status": r.status.value,
                    "coverage_pct": r.coverage_pct,
                    "expected_bars": r.expected_bars,
                    "actual_bars": r.actual_bars,
                    "gap_count": r.gap_count,
                }
                for (sym, tf), r in results.items()
            }, indent=2)

        # Text format
        lines = ["=" * 60, "Historical Data Validation Report", "=" * 60, ""]

        # Group by status
        by_status = {s: [] for s in ValidationStatus}
        for (sym, tf), r in results.items():
            by_status[r.status].append((sym, tf, r))

        for status in [ValidationStatus.FAIL, ValidationStatus.CAUTION, ValidationStatus.WARN, ValidationStatus.PASS]:
            items = by_status[status]
            if items:
                lines.append(f"\n{status.value} ({len(items)} items)")
                lines.append("-" * 40)
                for sym, tf, r in items:
                    lines.append(
                        f"  {sym}/{tf}: {r.coverage_pct:.1f}% "
                        f"({r.actual_bars}/{r.expected_bars} bars)"
                    )
                    if r.gaps:
                        lines.append(f"    Gaps: {len(r.gaps)} ({r.total_gap_bars} bars missing)")

        # Summary
        total = len(results)
        pass_count = len(by_status[ValidationStatus.PASS]) + len(by_status[ValidationStatus.WARN])
        fail_count = len(by_status[ValidationStatus.FAIL]) + len(by_status[ValidationStatus.CAUTION])

        lines.extend([
            "",
            "=" * 60,
            f"Summary: {pass_count}/{total} passed, {fail_count}/{total} need attention",
            "=" * 60,
        ])

        return "\n".join(lines)
