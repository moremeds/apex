"""Core types and data structures for Apex."""

from __future__ import annotations

import asyncio
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

import polars as pl
from pydantic import BaseModel, Field


class DataQualityDimension(str, Enum):
    """Data quality scoring dimensions."""

    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    TIMELINESS = "timeliness"
    UNIQUENESS = "uniqueness"


class DataQualityScore(BaseModel):
    """Data quality score for a specific dimension."""

    dimension: DataQualityDimension
    score: float = Field(ge=0.0, le=1.0, description="Score between 0.0 and 1.0")
    details: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)


class MarketDataFrame(BaseModel):
    """Wrapper for market data with metadata."""

    data: pl.DataFrame
    symbol: str
    start_date: datetime
    end_date: datetime
    source: str
    quality_scores: Dict[DataQualityDimension, DataQualityScore] = Field(
        default_factory=dict
    )
    cached_at: Optional[datetime] = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def overall_quality_score(self) -> float:
        """Calculate overall quality score as average of dimension scores."""
        if not self.quality_scores:
            return 0.0
        return sum(score.score for score in self.quality_scores.values()) / len(
            self.quality_scores
        )


class CacheConfig(BaseModel):
    """Configuration for data caching."""

    enabled: bool = True
    cache_dir: Path = Field(default=Path.home() / ".apex" / "cache")
    ttl_hours: int = Field(default=24, description="Time to live in hours")
    max_size_mb: int = Field(default=1000, description="Max cache size in MB")


class ProviderConfig(BaseModel):
    """Base configuration for data providers."""

    name: str
    cache_config: CacheConfig = Field(default_factory=CacheConfig)
    quality_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum quality score"
    )
    enforcement_level: str = Field(
        default="warn", description="warn, block, or fix"
    )


class DataProviderProtocol(Protocol):
    """Protocol defining the interface for data providers."""

    async def fetch_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> MarketDataFrame:
        """Fetch market data for the given symbol and date range."""
        ...

    async def validate_data(self, data: MarketDataFrame) -> MarketDataFrame:
        """Validate and score data quality."""
        ...

    def get_cache_key(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> str:
        """Generate cache key for the given parameters."""
        ...