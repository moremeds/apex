"""
Bar Validator.

Validates bar counts and generates BarValidationReport for PR-01 compliance.
Solves the "350 vs 252" mystery by providing full transparency on bar counts.

Extracted from signal_runner.py for better modularity.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

import pandas as pd

from src.domain.services.bar_count_calculator import BarCountCalculator
from src.domain.signals.schemas import (
    BarReductionReason,
    BarValidationBuilder,
    BarValidationReport,
)
from src.utils.logging_setup import get_logger

from .config import SignalPipelineConfig

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Configuration constants
LOOKBACK_DAYS = 550  # Calendar days to request (~1.5 years)
WARMUP_REQUIRED = 252  # For regime detector (SMA200 + margin)
WARMUP_INDICATOR = "SMA(200) + regime warmup"


class BarValidator:
    """
    Validates bar counts and outputs BarValidationReport.

    This solves the "350 vs 252" mystery by providing full transparency
    on how many bars were requested, loaded, trimmed, and validated.

    Usage:
        validator = BarValidator(config)
        exit_code = await validator.validate()

    Output format:
        BarValidationReport for AAPL (1d):
          requested_bars:  550  (calendar days requested)
          loaded_bars:     389  (actual data fetched)
          usable_bars:     340  (after NaN/gap trim)
          validated_bars:  140  (after SMA(200) warmup)

          coverage: 25.5% | warmup_satisfied: True

          reasons:
            - Weekends/holidays removed: 161 bars
            - NaN gaps trimmed: 49 bars
            - Warmup for SMA(200): 200 bars
    """

    def __init__(self, config: SignalPipelineConfig) -> None:
        """
        Initialize bar validator.

        Args:
            config: Pipeline configuration.
        """
        self.config = config

    async def validate(self) -> int:
        """
        Validate bar counts for all configured symbols/timeframes.

        Returns:
            Exit code (0 for success).
        """
        from src.services.historical_data_manager import HistoricalDataManager

        print("=" * 60)
        print("BAR VALIDATION REPORT")
        print("=" * 60)
        print(f"Symbols:    {', '.join(self.config.symbols)}")
        print(f"Timeframes: {', '.join(self.config.timeframes)}")
        print("=" * 60)

        # Create historical data manager
        historical_manager = HistoricalDataManager(
            base_dir=Path("data/historical"),
            source_priority=["ib", "yahoo"],
        )

        # Try to set IB source for better data quality
        ib_adapter = None
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
                print(f"Connected to IB at {ib_config.host}:{ib_config.port}")
        except Exception as e:
            logger.warning(f"IB not available, using Yahoo: {e}")
            print("Using Yahoo Finance (IB unavailable)")

        reports: List[BarValidationReport] = []

        for symbol in self.config.symbols:
            for tf in self.config.timeframes:
                report = await self._validate_bars_for_symbol(historical_manager, symbol, tf)
                reports.append(report)

        # Output reports
        print("\n")
        if self.config.json_output:
            # JSON output for machine parsing
            output = {
                "reports": [r.to_dict() for r in reports],
                "summary": {
                    "total_symbols": len(self.config.symbols),
                    "total_timeframes": len(self.config.timeframes),
                    "all_valid": all(r.is_valid for r in reports),
                    "warmup_failures": sum(1 for r in reports if not r.warmup_satisfied),
                },
            }
            print(json.dumps(output, indent=2, default=str))
        else:
            # Human-readable output
            for report in reports:
                print(report.format_report())
                print("-" * 60)

            # Summary
            print("\nSUMMARY:")
            print(f"  Symbols validated: {len(self.config.symbols)}")
            valid_count = sum(1 for r in reports if r.is_valid)
            print(f"  Valid reports:     {valid_count}/{len(reports)}")
            warmup_failures = [r for r in reports if not r.warmup_satisfied]
            if warmup_failures:
                print(f"  Warmup failures:   {len(warmup_failures)}")
                for r in warmup_failures:
                    print(f"    - {r.symbol}: need {r.warmup_required}, have {r.usable_bars}")

        return 0

    async def _validate_bars_for_symbol(
        self,
        historical_manager: Any,
        symbol: str,
        timeframe: str,
    ) -> BarValidationReport:
        """
        Validate bars for a single symbol/timeframe.

        Args:
            historical_manager: HistoricalDataManager instance.
            symbol: Symbol to validate.
            timeframe: Timeframe to validate.

        Returns:
            BarValidationReport with full breakdown.
        """
        builder = BarValidationBuilder(symbol=symbol, timeframe=timeframe)
        builder.set_requested_bars(LOOKBACK_DAYS)

        # Use last trading day to avoid holiday issues
        end = self._get_last_trading_day()
        start = end - timedelta(days=LOOKBACK_DAYS)

        try:
            # Load bars from historical manager
            bars = await historical_manager.ensure_data(symbol, timeframe, start, end)

            if not bars:
                builder.set_loaded_bars(0, source="none")
                builder.set_usable_bars(0)
                builder.set_validated_bars(0, warmup_required=WARMUP_REQUIRED)
                builder.add_reduction(
                    BarReductionReason.MISSING_DATA,
                    LOOKBACK_DAYS,
                    "No data available",
                )
                return builder.build()

            # Convert to DataFrame for analysis
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

            # Determine data source
            source = "cache"
            if hasattr(historical_manager, "_last_source"):
                source = historical_manager._last_source

            loaded_bars = len(df)
            first_date = df["timestamp"].min() if not df.empty else None
            last_date = df["timestamp"].max() if not df.empty else None

            builder.set_loaded_bars(
                loaded_bars,
                source=source,
                start=first_date,
                end=last_date,
            )

            # Calculate weekend/holiday reduction
            weekend_holiday_removed = LOOKBACK_DAYS - loaded_bars
            if weekend_holiday_removed > 0:
                builder.add_reduction(
                    BarReductionReason.WEEKEND_HOLIDAY,
                    weekend_holiday_removed,
                    "Weekends/holidays removed",
                )

            # Check for NaN values
            nan_rows = df[["open", "high", "low", "close"]].isna().any(axis=1).sum()
            usable_bars = loaded_bars - nan_rows

            builder.set_usable_bars(usable_bars)

            if nan_rows > 0:
                builder.add_reduction(
                    BarReductionReason.NAN_GAP,
                    nan_rows,
                    "NaN values/gaps trimmed",
                )

            # Calculate validated bars (after warmup)
            validated_bars = max(0, usable_bars - WARMUP_REQUIRED)

            builder.set_validated_bars(
                validated_bars,
                warmup_required=WARMUP_REQUIRED,
                warmup_indicator=WARMUP_INDICATOR,
            )

            if usable_bars >= WARMUP_REQUIRED:
                builder.add_reduction(
                    BarReductionReason.WARMUP,
                    WARMUP_REQUIRED,
                    f"Warmup for {WARMUP_INDICATOR}",
                )

            return builder.build()

        except Exception as e:
            logger.error(f"Failed to validate bars for {symbol}/{timeframe}: {e}")
            builder.set_loaded_bars(0, source="error")
            builder.set_usable_bars(0)
            builder.set_validated_bars(0, warmup_required=WARMUP_REQUIRED)
            builder.add_reduction(
                BarReductionReason.MISSING_DATA,
                LOOKBACK_DAYS,
                f"Error: {str(e)[:50]}",
            )
            return builder.build()

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
