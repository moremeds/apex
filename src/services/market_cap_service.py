"""
Market Cap Caching Service.

PR-C Deliverable: Provides cached market capitalization data for heatmap sizing.

Core Principles:
1. Cache-first - Never fetch market caps at report generation time
2. Explicit updates - Market caps updated via dedicated command only
3. Graceful degradation - Missing caps return 0 with cap_missing=true

Usage:
    # Update cache (run separately, e.g., daily cron)
    python -m src.runners.signal_runner update-market-caps --universe config/universe.yaml

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

# Project root for resolving relative paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Default cache file location
DEFAULT_CACHE_PATH = PROJECT_ROOT / "data/cache/market_caps.json"

# Cache schema version - bumped for sector info
CACHE_VERSION = "2.0"


@dataclass
class SymbolInfo:
    """Complete symbol information from yfinance."""

    symbol: str
    market_cap: float  # USD
    sector: Optional[str] = None  # e.g., "Technology", "Financial Services"
    industry: Optional[str] = None  # e.g., "Consumer Electronics"
    quote_type: str = "EQUITY"  # EQUITY, ETF, INDEX, etc.
    short_name: Optional[str] = None  # Company short name


@dataclass
class MarketCapCache:
    """Cached market cap and sector data with metadata."""

    version: str = CACHE_VERSION
    updated_at: Optional[datetime] = None
    source: str = "yfinance"
    caps: Dict[str, float] = field(default_factory=dict)
    sectors: Dict[str, str] = field(default_factory=dict)  # symbol -> sector
    industries: Dict[str, str] = field(default_factory=dict)  # symbol -> industry
    quote_types: Dict[str, str] = field(default_factory=dict)  # symbol -> EQUITY/ETF/etc
    short_names: Dict[str, str] = field(default_factory=dict)  # symbol -> short name
    errors: Dict[str, str] = field(default_factory=dict)  # Symbols that failed to fetch

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "version": self.version,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "source": self.source,
            "caps": self.caps,
            "sectors": self.sectors,
            "industries": self.industries,
            "quote_types": self.quote_types,
            "short_names": self.short_names,
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
            sectors=data.get("sectors", {}),
            industries=data.get("industries", {}),
            quote_types=data.get("quote_types", {}),
            short_names=data.get("short_names", {}),
            errors=data.get("errors", {}),
        )


@dataclass
class MarketCapResult:
    """Result of market cap lookup for a single symbol."""

    symbol: str
    market_cap: float
    sector: Optional[str] = None
    industry: Optional[str] = None
    quote_type: str = "EQUITY"
    short_name: Optional[str] = None
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
        Get market cap and sector info for a single symbol from cache.

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
                sector=cache.sectors.get(symbol),
                industry=cache.industries.get(symbol),
                quote_type=cache.quote_types.get(symbol, "EQUITY"),
                short_name=cache.short_names.get(symbol),
                cap_missing=False,
            )

        # Check if we have a recorded error for this symbol
        error = cache.errors.get(symbol)

        return MarketCapResult(
            symbol=symbol,
            market_cap=0.0,
            sector=cache.sectors.get(symbol),
            industry=cache.industries.get(symbol),
            quote_type=cache.quote_types.get(symbol, "EQUITY"),
            short_name=cache.short_names.get(symbol),
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
        Update market caps and sector info from yfinance.

        Fetches:
        - Market cap
        - Sector (e.g., "Technology", "Financial Services")
        - Industry (e.g., "Consumer Electronics")
        - Quote type (EQUITY, ETF, etc.)
        - Short name

        Args:
            symbols: List of symbols to update
            source: Data source (currently only "yfinance" supported)

        Returns:
            Dictionary of results for each symbol
        """
        if source != "yfinance":
            raise ValueError(f"Unsupported market cap source: {source}")

        logger.info(f"Updating market caps and sector info for {len(symbols)} symbols")

        # Import yfinance only when needed (optional dependency)
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed. Run: pip install yfinance")
            raise ImportError("yfinance required for market cap updates")

        # Load existing cache to preserve symbols not in current update
        cache = self._load_cache()

        results: Dict[str, MarketCapResult] = {}
        updated_caps: Dict[str, float] = dict(cache.caps)
        updated_sectors: Dict[str, str] = dict(cache.sectors)
        updated_industries: Dict[str, str] = dict(cache.industries)
        updated_quote_types: Dict[str, str] = dict(cache.quote_types)
        updated_short_names: Dict[str, str] = dict(cache.short_names)
        errors: Dict[str, str] = dict(cache.errors)

        # Fetch in batches to avoid rate limiting
        batch_size = 50
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            logger.info(f"Fetching batch {i // batch_size + 1}: {len(batch)} symbols")

            try:
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
                        sector = info.get("sector")
                        industry = info.get("industry")
                        quote_type = info.get("quoteType", "EQUITY")
                        short_name = info.get("shortName")

                        # Store all info
                        if sector:
                            updated_sectors[symbol] = sector
                        if industry:
                            updated_industries[symbol] = industry
                        if quote_type:
                            updated_quote_types[symbol] = quote_type
                        if short_name:
                            updated_short_names[symbol] = short_name

                        if market_cap and market_cap > 0:
                            updated_caps[symbol] = float(market_cap)
                            errors.pop(symbol, None)
                            results[symbol] = MarketCapResult(
                                symbol=symbol,
                                market_cap=float(market_cap),
                                sector=sector,
                                industry=industry,
                                quote_type=quote_type,
                                short_name=short_name,
                                cap_missing=False,
                            )
                        else:
                            # ETFs may not have market cap but still have useful info
                            if quote_type == "ETF":
                                errors.pop(symbol, None)
                                results[symbol] = MarketCapResult(
                                    symbol=symbol,
                                    market_cap=0.0,
                                    sector=sector,
                                    industry=industry,
                                    quote_type=quote_type,
                                    short_name=short_name,
                                    cap_missing=True,  # No market cap but not an error
                                )
                            else:
                                errors[symbol] = "No market cap in response"
                                results[symbol] = MarketCapResult(
                                    symbol=symbol,
                                    market_cap=0.0,
                                    sector=sector,
                                    industry=industry,
                                    quote_type=quote_type,
                                    short_name=short_name,
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

        # Create updated cache with all info
        new_cache = MarketCapCache(
            version=CACHE_VERSION,
            updated_at=datetime.now(),
            source=source,
            caps=updated_caps,
            sectors=updated_sectors,
            industries=updated_industries,
            quote_types=updated_quote_types,
            short_names=updated_short_names,
            errors=errors,
        )

        # Save to disk
        self._save_cache(new_cache)

        # Log summary
        success_count = sum(1 for r in results.values() if not r.cap_missing)
        sector_count = len(updated_sectors)
        logger.info(
            f"Update complete: {success_count}/{len(symbols)} with market cap, "
            f"{sector_count} with sector info"
        )

        return results

    def get_missing_symbols(self, symbols: List[str]) -> List[str]:
        """Get list of symbols not in cache."""
        cache = self._load_cache()
        return [s for s in symbols if s not in cache.caps]

    def ensure_market_caps(
        self,
        symbols: List[str],
        auto_fetch: bool = True,
    ) -> Dict[str, MarketCapResult]:
        """
        Get market caps, auto-fetching missing ones if needed.

        This is the recommended method for report generation - it ensures
        all requested symbols have market cap data by fetching any missing
        ones from yfinance.

        Args:
            symbols: List of symbols to get market caps for
            auto_fetch: If True, fetch missing caps from yfinance

        Returns:
            Dictionary mapping symbol to MarketCapResult
        """
        # First check what's missing
        missing = self.get_missing_symbols(symbols)

        if missing and auto_fetch:
            logger.info(f"Auto-fetching {len(missing)} missing market caps...")
            try:
                self.update_market_caps(missing)
            except ImportError:
                logger.warning("yfinance not installed, cannot auto-fetch market caps")
            except Exception as e:
                logger.warning(f"Failed to auto-fetch market caps: {e}")

        # Return all results (including any that failed to fetch)
        return self.get_market_caps(symbols)

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
