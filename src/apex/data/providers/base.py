"""Base data provider implementation."""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import asyncio

import polars as pl
import structlog

from apex.core.types import (
    DataProviderProtocol,
    MarketDataFrame,
    ProviderConfig,
)

logger = structlog.get_logger(__name__)


class BaseDataProvider(ABC):
    """Base class for all data providers."""

    def __init__(self, config: ProviderConfig) -> None:
        """Initialize the provider with configuration."""
        self.config = config
        self.cache_dir = config.cache_config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    async def _fetch_raw_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pl.DataFrame:
        """Fetch raw data from the provider source.
        
        This method should be implemented by concrete providers.
        """
        pass

    async def fetch_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> MarketDataFrame:
        """Fetch market data with caching and quality validation."""
        cache_key = self.get_cache_key(symbol, start_date, end_date)
        
        # Try to load from cache first
        if self.config.cache_config.enabled:
            cached_data = await self._load_from_cache(cache_key)
            if cached_data is not None:
                logger.info(
                    "Data loaded from cache",
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                )
                return cached_data

        # Fetch fresh data
        logger.info(
            "Fetching data from source",
            provider=self.config.name,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )
        
        raw_data = await self._fetch_raw_data(symbol, start_date, end_date)
        
        market_data = MarketDataFrame(
            data=raw_data,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            source=self.config.name,
            cached_at=datetime.now(),
        )

        # Save to cache
        if self.config.cache_config.enabled:
            await self._save_to_cache(cache_key, market_data)

        return market_data

    async def validate_data(self, data: MarketDataFrame) -> MarketDataFrame:
        """Validate data quality - to be implemented by quality scorer."""
        # This will be implemented when we create the DataQualityScorer
        return data

    def get_cache_key(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> str:
        """Generate a unique cache key for the data request.
        
        Creates a secure MD5 hash-based cache key from request parameters
        to prevent path traversal attacks and ensure consistent caching.
        
        Args:
            symbol: Trading symbol (will be sanitized)
            start_date: Start date for the data request
            end_date: End date for the data request
            
        Returns:
            str: Secure MD5 hash suitable for use as filename
            
        Raises:
            ValueError: If input parameters are invalid
        """
        # Sanitize inputs for security
        sanitized_symbol = self._sanitize_cache_input(symbol)
        
        key_data = {
            "provider": self.config.name,
            "symbol": sanitized_symbol,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }
        key_string = json.dumps(key_data, sort_keys=True)
        cache_key = hashlib.md5(key_string.encode()).hexdigest()
        
        # Ensure key is safe for filesystem use
        if not self._is_safe_cache_key(cache_key):
            raise ValueError(f"Generated cache key is not filesystem safe: {cache_key}")
            
        return cache_key

    def _sanitize_cache_input(self, symbol: str) -> str:
        """Sanitize symbol for cache key generation.
        
        Args:
            symbol: Raw symbol input
            
        Returns:
            str: Sanitized symbol safe for cache operations
        """
        if not isinstance(symbol, str):
            symbol = str(symbol)
            
        # Remove any potentially dangerous characters
        sanitized = re.sub(r'[^\w\-\.\^=]', '_', symbol)
        return sanitized[:50]  # Limit length
        
    def _is_safe_cache_key(self, cache_key: str) -> bool:
        """Validate that cache key is safe for filesystem operations.
        
        Args:
            cache_key: Generated cache key to validate
            
        Returns:
            bool: True if key is safe for filesystem use
        """
        # Should be hex string of fixed length
        if len(cache_key) != 32:
            return False
            
        # Should only contain hex characters
        try:
            int(cache_key, 16)
            return True
        except ValueError:
            return False

    async def _load_from_cache(self, cache_key: str) -> Optional[MarketDataFrame]:
        """Load data from cache if available and not expired.
        
        Args:
            cache_key: Validated cache key (should be hex string)
            
        Returns:
            Optional[MarketDataFrame]: Cached data if available and valid
            
        Raises:
            ValueError: If cache key fails security validation
        """
        # Security validation of cache key
        if not self._is_safe_cache_key(cache_key):
            raise ValueError(f"Invalid cache key format: {cache_key}")
            
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Ensure the resolved path is within the cache directory
        try:
            cache_file.resolve().relative_to(self.cache_dir.resolve())
        except ValueError:
            raise ValueError("Cache file path traversal detected")
        
        if not cache_file.exists():
            return None

        try:
            # Check if cache is expired
            age_hours = (
                datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            ).total_seconds() / 3600
            
            if age_hours > self.config.cache_config.ttl_hours:
                logger.info("Cache expired, removing", cache_key=cache_key)
                cache_file.unlink()
                return None

            # Use asyncio for non-blocking file I/O
            loop = asyncio.get_event_loop()
            with open(cache_file, "rb") as f:
                data = await loop.run_in_executor(None, pickle.load, f)
                return data
                
        except Exception as e:
            logger.warning("Failed to load from cache", error=str(e))
            return None

    async def _save_to_cache(self, cache_key: str, data: MarketDataFrame) -> None:
        """Save data to cache with security validation.
        
        Args:
            cache_key: Validated cache key (should be hex string)
            data: Market data to cache
            
        Raises:
            ValueError: If cache key fails security validation
        """
        # Security validation of cache key
        if not self._is_safe_cache_key(cache_key):
            raise ValueError(f"Invalid cache key format: {cache_key}")
            
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Ensure the resolved path is within the cache directory
        try:
            cache_file.resolve().parent.relative_to(self.cache_dir.resolve())
        except ValueError:
            raise ValueError("Cache file path traversal detected")
        
        try:
            # Use asyncio for non-blocking file I/O
            loop = asyncio.get_event_loop()
            with open(cache_file, "wb") as f:
                await loop.run_in_executor(None, pickle.dump, data, f)
                
            logger.debug("Data saved to cache", cache_key=cache_key)
            
        except Exception as e:
            logger.warning("Failed to save to cache", error=str(e))

    def _ensure_required_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ensure the DataFrame has required OHLCV columns."""
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        return df.select([
            "datetime",
            "open",
            "high", 
            "low",
            "close",
            "volume",
        ])