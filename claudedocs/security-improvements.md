# Security Improvements for Apex Backtesting System

## Current Security Issues Identified

### 1. Yahoo Finance Provider - Input Validation Missing

**File**: `src/apex/data/providers/yahoo.py`
**Method**: `_fetch_raw_data`
**Risk Level**: HIGH

**Issues**:
1. No symbol format validation - could allow injection attacks
2. No date range validation - could cause resource exhaustion
3. Direct string interpolation in error messages - potential info disclosure
4. No rate limiting or request throttling

**Current Code**:
```python
async def _fetch_raw_data(
    self,
    symbol: str,  # ← No validation
    start_date: datetime,  # ← No range checks
    end_date: datetime,
) -> pl.DataFrame:
    try:
        ticker = yf.Ticker(symbol)  # ← Direct usage without validation
        hist_data = await loop.run_in_executor(
            None,
            lambda: ticker.history(
                start=start_date.date(),  # ← No date validation
                end=end_date.date(),
            ),
        )
        if hist_data.empty:
            raise ValueError(f"No data available for symbol {symbol}")  # ← Info disclosure
```

## Proposed Security Enhancements

### 1. Input Validation Framework

```python
# New file: src/apex/data/validation.py
"""
Data validation utilities for secure input handling.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Tuple

class DataValidationError(ValueError):
    """Custom exception for data validation errors."""
    pass

class InputValidator:
    """Validates inputs for data providers to prevent security issues."""
    
    # Valid symbol patterns
    SYMBOL_PATTERN = re.compile(r'^[A-Za-z0-9._-]{1,12}$')
    CRYPTO_SYMBOL_PATTERN = re.compile(r'^[A-Za-z0-9_-]{1,20}$')
    
    # Limits
    MAX_DATE_RANGE_DAYS = 365 * 10  # 10 years max
    MIN_DATE_RANGE_DAYS = 1  # Minimum 1 day
    
    @classmethod
    def validate_symbol(cls, symbol: str, allow_crypto: bool = False) -> str:
        """Validate trading symbol format.
        
        Args:
            symbol: Trading symbol to validate
            allow_crypto: Whether to allow cryptocurrency symbols
            
        Returns:
            Validated and normalized symbol
            
        Raises:
            DataValidationError: If symbol format is invalid
        """
        if not symbol or not isinstance(symbol, str):
            raise DataValidationError("Symbol must be a non-empty string")
        
        # Normalize symbol
        symbol = symbol.strip().upper()
        
        # Check length
        if len(symbol) > (20 if allow_crypto else 12):
            raise DataValidationError(f"Symbol too long: {len(symbol)} characters")
        
        # Check pattern
        pattern = cls.CRYPTO_SYMBOL_PATTERN if allow_crypto else cls.SYMBOL_PATTERN
        if not pattern.match(symbol):
            raise DataValidationError("Invalid symbol format")
        
        # Block potentially dangerous patterns
        dangerous_patterns = ['..', '//', '\\\\', '<', '>', '&', '|']
        if any(pattern in symbol for pattern in dangerous_patterns):
            raise DataValidationError("Symbol contains invalid characters")
            
        return symbol
    
    @classmethod
    def validate_date_range(
        cls, 
        start_date: datetime, 
        end_date: datetime
    ) -> Tuple[datetime, datetime]:
        """Validate date range for data requests.
        
        Args:
            start_date: Start date for data request
            end_date: End date for data request
            
        Returns:
            Tuple of validated start and end dates
            
        Raises:
            DataValidationError: If date range is invalid
        """
        if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
            raise DataValidationError("Dates must be datetime objects")
        
        # Check date order
        if start_date >= end_date:
            raise DataValidationError("Start date must be before end date")
        
        # Check range limits
        date_range = (end_date - start_date).days
        
        if date_range < cls.MIN_DATE_RANGE_DAYS:
            raise DataValidationError(f"Date range too small: {date_range} days")
        
        if date_range > cls.MAX_DATE_RANGE_DAYS:
            raise DataValidationError(f"Date range too large: {date_range} days (max: {cls.MAX_DATE_RANGE_DAYS})")
        
        # Check if dates are reasonable (not too far in future)
        now = datetime.now()
        future_limit = now + timedelta(days=30)  # Allow 30 days in future
        
        if end_date > future_limit:
            raise DataValidationError("End date too far in future")
        
        # Check if start date is too old (more than 50 years)
        past_limit = now - timedelta(days=365 * 50)
        if start_date < past_limit:
            raise DataValidationError("Start date too far in past")
            
        return start_date, end_date
    
    @classmethod
    def sanitize_error_message(cls, message: str, symbol: str) -> str:
        """Sanitize error messages to prevent information disclosure.
        
        Args:
            message: Original error message
            symbol: Symbol that caused the error
            
        Returns:
            Sanitized error message
        """
        # Remove potentially sensitive information
        sanitized_symbol = cls._sanitize_symbol_for_logging(symbol)
        
        # Generic error messages to prevent info disclosure
        if "no data" in message.lower():
            return f"Data not available for symbol {sanitized_symbol}"
        elif "invalid" in message.lower():
            return "Invalid request parameters"
        else:
            return "Data request failed"
    
    @classmethod
    def _sanitize_symbol_for_logging(cls, symbol: str) -> str:
        """Sanitize symbol for safe logging.
        
        Args:
            symbol: Original symbol
            
        Returns:
            Sanitized symbol safe for logging
        """
        if not symbol:
            return "[empty]"
        
        # Only allow alphanumeric and basic punctuation for logging
        sanitized = re.sub(r'[^A-Za-z0-9._-]', '_', str(symbol))
        
        # Truncate if too long
        if len(sanitized) > 20:
            sanitized = sanitized[:17] + "..."
            
        return sanitized
```

### 2. Enhanced Yahoo Provider with Security

```python
# Enhanced version of yahoo.py with security improvements
"""
Yahoo Finance data provider implementation with security enhancements.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

import polars as pl
import structlog
import yfinance as yf

from apex.core.types import ProviderConfig
from apex.data.providers.base import BaseDataProvider
from apex.data.validation import InputValidator, DataValidationError

logger = structlog.get_logger(__name__)


class YahooDataProvider(BaseDataProvider):
    """Yahoo Finance data provider with enhanced security validation."""

    def __init__(self, config: Optional[ProviderConfig] = None) -> None:
        """Initialize Yahoo Finance provider with security settings."""
        if config is None:
            config = ProviderConfig(name="yahoo_finance")
        super().__init__(config)
        
        # Security settings
        self._validator = InputValidator()
        self._request_timeout = 30  # seconds
        self._max_retries = 3

    async def _fetch_raw_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pl.DataFrame:
        """Fetch raw data from Yahoo Finance with security validation.
        
        Args:
            symbol: Trading symbol (will be validated)
            start_date: Start date for data (will be validated)
            end_date: End date for data (will be validated)
            
        Returns:
            Polars DataFrame with validated data
            
        Raises:
            DataValidationError: If inputs are invalid
            ValueError: If data fetch fails after validation
        """
        # Security: Validate inputs before processing
        try:
            validated_symbol = self._validator.validate_symbol(symbol)
            validated_start, validated_end = self._validator.validate_date_range(start_date, end_date)
        except DataValidationError as e:
            logger.warning(
                "Input validation failed",
                error=str(e),
                symbol_length=len(symbol) if symbol else 0
            )
            raise
        
        try:
            # Create ticker with timeout protection
            ticker = yf.Ticker(validated_symbol)
            
            # Run yfinance in thread pool with timeout
            loop = asyncio.get_event_loop()
            
            # Use asyncio.wait_for for timeout protection
            hist_data = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: ticker.history(
                        start=validated_start.date(),
                        end=validated_end.date(),
                        auto_adjust=True,
                        prepost=False,
                        timeout=self._request_timeout
                    )
                ),
                timeout=self._request_timeout + 5
            )

            if hist_data.empty:
                # Security: Use sanitized error message
                sanitized_msg = self._validator.sanitize_error_message(
                    "No data available", validated_symbol
                )
                raise ValueError(sanitized_msg)

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
            
            # Security: Log success without sensitive data
            logger.info(
                "Successfully fetched data from Yahoo Finance",
                symbol_length=len(validated_symbol),
                rows=len(df),
                date_range_days=(validated_end - validated_start).days
            )
            
            return df

        except asyncio.TimeoutError:
            logger.error(
                "Yahoo Finance request timed out",
                timeout=self._request_timeout,
                symbol_length=len(validated_symbol)
            )
            raise ValueError("Data request timed out")
            
        except Exception as e:
            # Security: Sanitize error message
            sanitized_msg = self._validator.sanitize_error_message(str(e), validated_symbol)
            
            logger.error(
                "Failed to fetch data from Yahoo Finance",
                error_type=type(e).__name__,
                symbol_length=len(validated_symbol)
            )
            
            raise ValueError(sanitized_msg) from e
```

### 3. Rate Limiting Enhancement

```python
# New file: src/apex/data/rate_limiting.py
"""
Rate limiting for data providers to prevent abuse.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional

class RateLimiter:
    """Rate limiter for data provider requests."""
    
    def __init__(
        self, 
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_limit: int = 10
    ):
        """Initialize rate limiter with configurable limits.
        
        Args:
            requests_per_minute: Max requests per minute
            requests_per_hour: Max requests per hour  
            burst_limit: Max burst requests
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_limit = burst_limit
        
        self._minute_requests: Dict[str, list] = {}
        self._hour_requests: Dict[str, list] = {}
        self._burst_requests: Dict[str, list] = {}
        self._lock = asyncio.Lock()
    
    async def acquire(self, key: str = "default") -> None:
        """Acquire rate limit permission.
        
        Args:
            key: Rate limiting key (e.g., IP address, user ID)
            
        Raises:
            ValueError: If rate limit exceeded
        """
        async with self._lock:
            now = datetime.now()
            
            # Clean old requests
            self._cleanup_old_requests(key, now)
            
            # Check limits
            if self._check_burst_limit(key):
                raise ValueError("Burst rate limit exceeded")
            
            if self._check_minute_limit(key):
                raise ValueError("Per-minute rate limit exceeded")
            
            if self._check_hour_limit(key):
                raise ValueError("Per-hour rate limit exceeded")
            
            # Record request
            self._record_request(key, now)
    
    def _cleanup_old_requests(self, key: str, now: datetime) -> None:
        """Remove old request records."""
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        burst_window = now - timedelta(seconds=10)
        
        # Clean minute requests
        if key in self._minute_requests:
            self._minute_requests[key] = [
                req_time for req_time in self._minute_requests[key] 
                if req_time > minute_ago
            ]
        
        # Clean hour requests
        if key in self._hour_requests:
            self._hour_requests[key] = [
                req_time for req_time in self._hour_requests[key] 
                if req_time > hour_ago
            ]
        
        # Clean burst requests
        if key in self._burst_requests:
            self._burst_requests[key] = [
                req_time for req_time in self._burst_requests[key] 
                if req_time > burst_window
            ]
    
    def _check_burst_limit(self, key: str) -> bool:
        """Check if burst limit would be exceeded."""
        burst_count = len(self._burst_requests.get(key, []))
        return burst_count >= self.burst_limit
    
    def _check_minute_limit(self, key: str) -> bool:
        """Check if per-minute limit would be exceeded."""
        minute_count = len(self._minute_requests.get(key, []))
        return minute_count >= self.requests_per_minute
    
    def _check_hour_limit(self, key: str) -> bool:
        """Check if per-hour limit would be exceeded."""
        hour_count = len(self._hour_requests.get(key, []))
        return hour_count >= self.requests_per_hour
    
    def _record_request(self, key: str, now: datetime) -> None:
        """Record a new request."""
        if key not in self._minute_requests:
            self._minute_requests[key] = []
        if key not in self._hour_requests:
            self._hour_requests[key] = []
        if key not in self._burst_requests:
            self._burst_requests[key] = []
        
        self._minute_requests[key].append(now)
        self._hour_requests[key].append(now)
        self._burst_requests[key].append(now)
```

## Implementation Priority

### Phase 1: Critical Security (Day 1)
1. Create `validation.py` with input validation framework
2. Update Yahoo provider with validation calls
3. Add comprehensive error handling

### Phase 2: Rate Limiting (Day 2)
1. Implement `rate_limiting.py`
2. Integrate rate limiter into providers
3. Add configuration options

### Phase 3: Security Testing (Day 3)
1. Add security-focused unit tests
2. Test injection attack prevention
3. Validate error message sanitization

## Security Testing Plan

```python
# tests/unit/test_data/test_security.py
"""
Security tests for data providers.
"""
import pytest
from datetime import datetime, timedelta

from apex.data.validation import InputValidator, DataValidationError
from apex.data.providers.yahoo import YahooDataProvider


class TestInputValidation:
    """Test input validation security."""
    
    def test_symbol_injection_prevention(self):
        """Test prevention of symbol injection attacks."""
        validator = InputValidator()
        
        # Test various injection attempts
        malicious_symbols = [
            "AAPL'; DROP TABLE--",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "AAPL|rm -rf /",
            "AAPL&whoami",
            "../../../secret"
        ]
        
        for symbol in malicious_symbols:
            with pytest.raises(DataValidationError):
                validator.validate_symbol(symbol)
    
    def test_date_range_limits(self):
        """Test date range validation."""
        validator = InputValidator()
        now = datetime.now()
        
        # Test oversized range
        with pytest.raises(DataValidationError):
            validator.validate_date_range(
                now - timedelta(days=365 * 20),  # 20 years ago
                now
            )
        
        # Test future dates
        with pytest.raises(DataValidationError):
            validator.validate_date_range(
                now,
                now + timedelta(days=365)  # 1 year in future
            )
    
    def test_error_message_sanitization(self):
        """Test error message sanitization."""
        validator = InputValidator()
        
        # Test sanitization of potentially sensitive info
        original = "Database error: user 'admin' password '12345'"
        sanitized = validator.sanitize_error_message(original, "AAPL")
        
        assert "admin" not in sanitized
        assert "12345" not in sanitized
        assert "Database error" not in sanitized
```

## Benefits

### Security Benefits:
- **Injection Prevention**: Validates all inputs to prevent code injection
- **Resource Protection**: Limits request size and frequency
- **Information Security**: Sanitizes error messages
- **DoS Prevention**: Rate limiting prevents abuse

### Operational Benefits:
- **Better Debugging**: Structured error handling
- **Performance**: Early validation prevents unnecessary processing
- **Compliance**: Meets security best practices
- **Monitoring**: Enhanced logging for security events

This security enhancement addresses all identified vulnerabilities while maintaining backward compatibility and improving overall system robustness.