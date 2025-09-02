"""Yahoo Finance data provider implementation."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from typing import Optional

import polars as pl
import structlog
import yfinance as yf

from apex.core.types import ProviderConfig
from apex.data.providers.base import BaseDataProvider

logger = structlog.get_logger(__name__)

# Security patterns
VALID_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9\.\-\^=]{1,12}$", re.IGNORECASE)
MAX_SYMBOL_LENGTH = 12
MIN_SYMBOL_LENGTH = 1


class YahooDataProvider(BaseDataProvider):
    """Yahoo Finance data provider using yfinance library."""

    def __init__(self, config: Optional[ProviderConfig] = None) -> None:
        """Initialize Yahoo Finance provider."""
        if config is None:
            config = ProviderConfig(name="yahoo_finance")
        super().__init__(config)

    async def _fetch_raw_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pl.DataFrame:
        """Fetch raw data from Yahoo Finance.
        
        Validates input parameters for security and fetches historical market data
        from Yahoo Finance with proper error handling and data validation.
        
        Args:
            symbol: Stock symbol to fetch (validated for security)
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            pl.DataFrame: Historical market data with OHLCV columns
            
        Raises:
            ValueError: If symbol format is invalid or data is unavailable
            SecurityError: If input validation fails for security reasons
        """
        # Security validation
        self._validate_symbol_security(symbol)
        self._validate_date_range(start_date, end_date)
        
        try:
            # Run yfinance in thread pool since it's not async
            loop = asyncio.get_event_loop()
            ticker = yf.Ticker(symbol)
            
            # Download historical data
            hist_data = await loop.run_in_executor(
                None,
                lambda: ticker.history(
                    start=start_date.date(),
                    end=end_date.date(),
                    auto_adjust=True,
                    prepost=False,
                ),
            )

            if hist_data.empty:
                raise ValueError(f"No data available for symbol {symbol}")

            # Convert pandas DataFrame to polars
            hist_data = hist_data.reset_index()
            
            # Rename columns to lowercase
            hist_data.columns = [col.lower() for col in hist_data.columns]
            
            # Convert to polars DataFrame
            df = pl.from_pandas(hist_data)
            
            # Rename date column to datetime
            if "date" in df.columns:
                df = df.rename({"date": "datetime"})
            
            # Ensure datetime column is datetime type
            df = df.with_columns([
                pl.col("datetime").cast(pl.Datetime)
            ])

            # Validate and select required columns
            df = self._ensure_required_columns(df)
            
            logger.info(
                "Successfully fetched data from Yahoo Finance",
                symbol=symbol,
                rows=len(df),
                start_date=start_date,
                end_date=end_date,
            )
            
            return df

        except Exception as e:
            logger.error(
                "Failed to fetch data from Yahoo Finance",
                symbol=symbol,
                error=str(e),
            )
            raise

    def _validate_symbol_security(self, symbol: str) -> None:
        """Validate symbol for security and format requirements.
        
        Performs comprehensive validation to prevent injection attacks and
        ensure symbol conforms to expected format.
        
        Args:
            symbol: Stock symbol to validate
            
        Raises:
            ValueError: If symbol fails security or format validation
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
            
        # Length validation
        if not (MIN_SYMBOL_LENGTH <= len(symbol) <= MAX_SYMBOL_LENGTH):
            raise ValueError(
                f"Symbol length must be between {MIN_SYMBOL_LENGTH} and {MAX_SYMBOL_LENGTH} characters"
            )
            
        # Format validation using regex
        if not VALID_SYMBOL_PATTERN.match(symbol):
            raise ValueError(
                f"Symbol '{symbol}' contains invalid characters. "
                "Only letters, numbers, dots, hyphens, carets, and equals signs are allowed."
            )
            
        # Sanitize for logging (prevent log injection)
        sanitized_symbol = re.sub(r'[^\w\.\-\^=]', '_', symbol)
        logger.debug(f"Symbol validation passed for: {sanitized_symbol}")

    def _validate_date_range(self, start_date: datetime, end_date: datetime) -> None:
        """Validate date range for reasonable bounds.
        
        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch
            
        Raises:
            ValueError: If date range is invalid or unreasonable
        """
        if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
            raise ValueError("Start and end dates must be datetime objects")
            
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")
            
        # Prevent fetching data from too far in the future
        if start_date > datetime.now():
            raise ValueError("Start date cannot be in the future")
            
        # Prevent excessive historical data requests (before 1970)
        if start_date.year < 1970:
            raise ValueError("Start date cannot be before 1970")
            
        # Limit maximum date range to 50 years
        max_range_days = 50 * 365
        if (end_date - start_date).days > max_range_days:
            raise ValueError(f"Date range cannot exceed {max_range_days} days")

    def _validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol exists in Yahoo Finance.
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            bool: True if symbol exists and is valid
        """
        try:
            # First run security validation
            self._validate_symbol_security(symbol)
            
            # Then check if symbol exists
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return bool(info.get("symbol") or info.get("shortName"))
        except ValueError:
            # Security validation failed
            return False
        except Exception as e:
            logger.warning(f"Symbol validation failed for {symbol}", error=str(e))
            return False