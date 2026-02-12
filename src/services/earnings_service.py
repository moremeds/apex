"""Earnings data caching service.

Cache-first pattern mirroring MarketCapService:
1. Cache-first: never block screening on external API calls
2. Explicit updates: earnings cache refreshed via dedicated command
3. Graceful degradation: missing symbols return empty

Cache file: data/cache/earnings.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CACHE_PATH = PROJECT_ROOT / "data/cache/earnings.json"
CACHE_VERSION = "1.0"


@dataclass
class EarningsCache:
    """Cached earnings data with metadata."""

    version: str = CACHE_VERSION
    updated_at: datetime | None = None
    source: str = "fmp"
    earnings: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    skipped_count: int = 0
    errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "version": self.version,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "source": self.source,
            "earnings": self.earnings,
            "skipped_count": self.skipped_count,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EarningsCache:
        """Deserialize from dictionary."""
        updated_at = None
        if data.get("updated_at"):
            try:
                updated_at = datetime.fromisoformat(data["updated_at"])
            except (ValueError, TypeError):
                pass

        return cls(
            version=data.get("version", CACHE_VERSION),
            updated_at=updated_at,
            source=data.get("source", "fmp"),
            earnings=data.get("earnings", {}),
            skipped_count=data.get("skipped_count", 0),
            errors=data.get("errors", {}),
        )


class EarningsService:
    """Service for managing cached earnings data.

    Mirrors the MarketCapService pattern:
    - Reading cached earnings (for screening)
    - Bulk updating from FMP (for cron/CLI)
    - Graceful handling of missing data
    """

    def __init__(self, cache_path: Path | None = None) -> None:
        self._cache_path = cache_path or DEFAULT_CACHE_PATH
        self._cache: EarningsCache | None = None

    def _load_cache(self) -> EarningsCache:
        """Load cache from disk, creating empty if not exists."""
        if self._cache is not None:
            return self._cache

        if not self._cache_path.exists():
            logger.info(f"Earnings cache not found at {self._cache_path}, using empty")
            self._cache = EarningsCache()
            return self._cache

        try:
            with open(self._cache_path) as f:
                data = json.load(f)
            self._cache = EarningsCache.from_dict(data)
            symbol_count = sum(len(v) for v in self._cache.earnings.values())
            logger.info(
                f"Loaded earnings cache: {len(self._cache.earnings)} symbols, "
                f"{symbol_count} events, updated {self._cache.updated_at}"
            )
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load earnings cache: {e}")
            self._cache = EarningsCache()

        return self._cache

    def _save_cache(self, cache: EarningsCache) -> bool:
        """Save cache to disk."""
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "w") as f:
                json.dump(cache.to_dict(), f, indent=2)
            self._cache = cache
            logger.info(f"Saved earnings cache: {len(cache.earnings)} symbols")
            return True
        except IOError as e:
            logger.error(f"Failed to save earnings cache: {e}")
            return False

    def get_recent_earnings(self, lookback_days: int = 10) -> list[dict[str, Any]]:
        """Read recent earnings from cache.

        Returns all cached earning events (caller filters by date as needed).
        """
        cache = self._load_cache()
        all_earnings: list[dict[str, Any]] = []
        for earnings_list in cache.earnings.values():
            all_earnings.extend(earnings_list)
        return all_earnings

    def get_cache_metadata(self) -> dict[str, Any]:
        """Get cache metadata."""
        cache = self._load_cache()
        return {
            "version": cache.version,
            "updated_at": cache.updated_at.isoformat() if cache.updated_at else None,
            "source": cache.source,
            "symbol_count": len(cache.earnings),
            "skipped_count": cache.skipped_count,
            "error_count": len(cache.errors),
        }

    def update_earnings(
        self,
        symbols: list[str],
        lookback_days: int = 10,
        api_key: str | None = None,
    ) -> EarningsCache:
        """Update earnings cache from FMP + yfinance.

        Args:
            symbols: Universe symbols to check.
            lookback_days: How many calendar days back to fetch.
            api_key: Optional FMP API key (overrides env/secrets).

        Returns:
            Updated EarningsCache.
        """
        from src.infrastructure.adapters.earnings.fmp_earnings import FMPEarningsAdapter

        logger.info(f"Updating earnings for {len(symbols)} symbols (lookback={lookback_days}d)")

        adapter = FMPEarningsAdapter(api_key=api_key)
        earnings_data, skipped = adapter.fetch_recent_earnings(symbols, lookback_days)

        # Build earnings dict keyed by symbol
        earnings_by_symbol: dict[str, list[dict[str, Any]]] = {}
        for e in earnings_data:
            sym = e.get("symbol", "?")
            earnings_by_symbol.setdefault(sym, []).append(e)

        cache = EarningsCache(
            version=CACHE_VERSION,
            updated_at=datetime.now(),
            source="fmp",
            earnings=earnings_by_symbol,
            skipped_count=skipped,
        )

        self._save_cache(cache)
        return cache

    def clear_cache(self) -> None:
        """Clear in-memory cache (forces reload on next access)."""
        self._cache = None
