"""
Market Cap Caching Service.

PR-C Deliverable: Provides cached market capitalization data for heatmap sizing.

Core Principles:
1. Cache-first - Never fetch market caps at report generation time
2. Explicit updates - Market caps updated via dedicated command only
3. Graceful degradation - Missing caps return 0 with cap_missing=true

Usage:
    # Update cache (run separately, e.g., daily cron)
    python -m src.runners.signal_runner update-market-caps --universe config/signals/universe.yaml

    # Read cache (during report generation)
    service = MarketCapService()
    caps = service.get_market_caps(["AAPL", "MSFT", "GOOG"])
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Default cache file location
DEFAULT_CACHE_PATH = Path("data/cache/market_caps.json")

# Cache schema version
CACHE_VERSION = "1.0"


@dataclass
class MarketCapEntry:
    """Single market cap entry with metadata."""

    symbol: str
    market_cap: float  # USD
    source: str = "yfinance"
    updated_at: Optional[datetime] = None


@dataclass
class MarketCapCache:
    """Cached market cap data with metadata."""

    version: str = CACHE_VERSION
    updated_at: Optional[datetime] = None
    source: str = "yfinance"
    caps: Dict[str, float] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)  # Symbols that failed to fetch

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "version": self.version,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "source": self.source,
            "caps": self.caps,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketCapCache":
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
            source=data.get("source", "yfinance"),
            caps=data.get("caps", {}),
            errors=data.get("errors", {}),
        )


@dataclass
class MarketCapResult:
    """Result of market cap lookup for a single symbol."""

    symbol: str
    market_cap: float
    cap_missing: bool = False
    error: Optional[str] = None


class MarketCapService:
    """
    Service for managing cached market capitalization data.

    This service provides:
    1. Reading cached market caps (for report generation)
    2. Bulk updating market caps from yfinance (for cron jobs)
    3. Graceful handling of missing data

    Design decisions:
    - Cache-first: Never blocks report generation on external API calls
    - Explicit updates: Market caps only refresh via dedicated command
    - Transparent: Missing caps flagged with cap_missing=true
    """

    def __init__(self, cache_path: Optional[Path] = None) -> None:
        """
        Initialize market cap service.

        Args:
            cache_path: Path to cache file. Defaults to data/cache/market_caps.json
        """
        self._cache_path = cache_path or DEFAULT_CACHE_PATH
        self._cache: Optional[MarketCapCache] = None

    def _load_cache(self) -> MarketCapCache:
        """Load cache from disk, creating empty cache if not exists."""
        if self._cache is not None:
            return self._cache

        if not self._cache_path.exists():
            logger.info(f"Market cap cache not found at {self._cache_path}, using empty cache")
            self._cache = MarketCapCache()
            return self._cache

        try:
            with open(self._cache_path, "r") as f:
                data = json.load(f)
            self._cache = MarketCapCache.from_dict(data)
            logger.info(
                f"Loaded market cap cache: {len(self._cache.caps)} symbols, "
                f"updated {self._cache.updated_at}"
            )
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load market cap cache: {e}")
            self._cache = MarketCapCache()

        return self._cache

    def _save_cache(self, cache: MarketCapCache) -> bool:
        """Save cache to disk."""
        try:
            # Ensure parent directory exists
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self._cache_path, "w") as f:
                json.dump(cache.to_dict(), f, indent=2)

            self._cache = cache
            logger.info(f"Saved market cap cache: {len(cache.caps)} symbols")
            return True

        except IOError as e:
            logger.error(f"Failed to save market cap cache: {e}")
            return False

    def get_market_cap(self, symbol: str) -> MarketCapResult:
        """
        Get market cap for a single symbol from cache.

        Args:
            symbol: Stock symbol (e.g., "AAPL")

        Returns:
            MarketCapResult with market_cap=0 and cap_missing=True if not cached
        """
        cache = self._load_cache()

        if symbol in cache.caps:
            return MarketCapResult(
                symbol=symbol,
                market_cap=cache.caps[symbol],
                cap_missing=False,
            )

        # Check if we have a recorded error for this symbol
        error = cache.errors.get(symbol)

        return MarketCapResult(
            symbol=symbol,
            market_cap=0.0,
            cap_missing=True,
            error=error,
        )

    def get_market_caps(self, symbols: List[str]) -> Dict[str, MarketCapResult]:
        """
        Get market caps for multiple symbols from cache.

        Args:
            symbols: List of stock symbols

        Returns:
            Dictionary mapping symbol to MarketCapResult
        """
        return {symbol: self.get_market_cap(symbol) for symbol in symbols}

    def get_all_cached_caps(self) -> Dict[str, float]:
        """Get all cached market caps as a simple dict."""
        cache = self._load_cache()
        return dict(cache.caps)

    def get_cache_metadata(self) -> Dict[str, Any]:
        """Get cache metadata (updated_at, source, count)."""
        cache = self._load_cache()
        return {
            "version": cache.version,
            "updated_at": cache.updated_at.isoformat() if cache.updated_at else None,
            "source": cache.source,
            "symbol_count": len(cache.caps),
            "error_count": len(cache.errors),
        }

    def update_market_caps(
        self,
        symbols: List[str],
        source: str = "yfinance",
    ) -> Dict[str, MarketCapResult]:
        """
        Update market caps from external source (yfinance).

        This method fetches fresh market cap data and updates the cache.
        Should be called from a dedicated update command, not during report generation.

        Args:
            symbols: List of symbols to update
            source: Data source (currently only "yfinance" supported)

        Returns:
            Dictionary of results for each symbol
        """
        if source != "yfinance":
            raise ValueError(f"Unsupported market cap source: {source}")

        logger.info(f"Updating market caps for {len(symbols)} symbols from {source}")

        # Import yfinance only when needed (optional dependency)
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed. Run: pip install yfinance")
            raise ImportError("yfinance required for market cap updates")

        # Load existing cache to preserve symbols not in current update
        cache = self._load_cache()

        results: Dict[str, MarketCapResult] = {}
        updated_caps: Dict[str, float] = dict(cache.caps)  # Start with existing
        errors: Dict[str, str] = dict(cache.errors)

        # Fetch in batches to avoid rate limiting
        batch_size = 50
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            logger.info(f"Fetching batch {i // batch_size + 1}: {len(batch)} symbols")

            try:
                # Use download with group_by to get info
                tickers = yf.Tickers(" ".join(batch))

                for symbol in batch:
                    try:
                        ticker = tickers.tickers.get(symbol)
                        if ticker is None:
                            errors[symbol] = "Ticker not found"
                            results[symbol] = MarketCapResult(
                                symbol=symbol,
                                market_cap=0.0,
                                cap_missing=True,
                                error="Ticker not found",
                            )
                            continue

                        info = ticker.info
                        market_cap = info.get("marketCap", 0)

                        if market_cap and market_cap > 0:
                            updated_caps[symbol] = float(market_cap)
                            # Remove from errors if previously failed
                            errors.pop(symbol, None)
                            results[symbol] = MarketCapResult(
                                symbol=symbol,
                                market_cap=float(market_cap),
                                cap_missing=False,
                            )
                        else:
                            errors[symbol] = "No market cap in response"
                            results[symbol] = MarketCapResult(
                                symbol=symbol,
                                market_cap=0.0,
                                cap_missing=True,
                                error="No market cap in response",
                            )

                    except Exception as e:
                        error_msg = str(e)[:100]
                        logger.warning(f"Failed to fetch {symbol}: {error_msg}")
                        errors[symbol] = error_msg
                        results[symbol] = MarketCapResult(
                            symbol=symbol,
                            market_cap=0.0,
                            cap_missing=True,
                            error=error_msg,
                        )

            except Exception as e:
                logger.error(f"Batch fetch failed: {e}")
                for symbol in batch:
                    if symbol not in results:
                        error_msg = f"Batch error: {str(e)[:50]}"
                        errors[symbol] = error_msg
                        results[symbol] = MarketCapResult(
                            symbol=symbol,
                            market_cap=0.0,
                            cap_missing=True,
                            error=error_msg,
                        )

        # Create updated cache
        new_cache = MarketCapCache(
            version=CACHE_VERSION,
            updated_at=datetime.now(),
            source=source,
            caps=updated_caps,
            errors=errors,
        )

        # Save to disk
        self._save_cache(new_cache)

        # Log summary
        success_count = sum(1 for r in results.values() if not r.cap_missing)
        logger.info(
            f"Market cap update complete: {success_count}/{len(symbols)} successful, "
            f"{len(errors)} errors"
        )

        return results

    def get_missing_symbols(self, symbols: List[str]) -> List[str]:
        """Get list of symbols not in cache."""
        cache = self._load_cache()
        return [s for s in symbols if s not in cache.caps]

    def clear_cache(self) -> None:
        """Clear the in-memory cache (forces reload on next access)."""
        self._cache = None

    def invalidate_symbols(self, symbols: List[str]) -> int:
        """
        Remove specific symbols from cache.

        Args:
            symbols: Symbols to remove

        Returns:
            Number of symbols removed
        """
        cache = self._load_cache()
        removed = 0

        for symbol in symbols:
            if symbol in cache.caps:
                del cache.caps[symbol]
                removed += 1
            cache.errors.pop(symbol, None)

        if removed > 0:
            self._save_cache(cache)

        return removed


def load_universe_symbols(universe_path: Path) -> List[str]:
    """
    Load symbol list from universe YAML file.

    Handles multiple formats:
    - Flat list: [AAPL, MSFT, ...]
    - Dict with lists: {tech: [AAPL, ...], finance: [JPM, ...]}
    - Nested groups: {groups: {tech_megacap: {symbols: [AAPL, ...]}}}

    Args:
        universe_path: Path to universe.yaml file

    Returns:
        List of symbols
    """
    import yaml

    try:
        with open(universe_path, "r") as f:
            data = yaml.safe_load(f)

        symbols: Set[str] = set()

        def extract_symbols(obj: Any) -> None:
            """Recursively extract symbols from nested structure."""
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, str):
                        symbols.add(item)
                    else:
                        extract_symbols(item)
            elif isinstance(obj, dict):
                # Check for 'symbols' key (e.g., {symbols: [AAPL, ...]})
                if "symbols" in obj:
                    extract_symbols(obj["symbols"])
                # Check for 'groups' key (e.g., {groups: {tech: {symbols: [...]}}})
                elif "groups" in obj:
                    extract_symbols(obj["groups"])
                else:
                    # Recurse into all dict values
                    for key, value in obj.items():
                        # Skip non-symbol keys like 'version', 'defaults', 'provider'
                        if key in ("version", "defaults", "provider", "overrides", "enabled"):
                            continue
                        extract_symbols(value)

        extract_symbols(data)

        return sorted(symbols)

    except Exception as e:
        logger.error(f"Failed to load universe from {universe_path}: {e}")
        raise
