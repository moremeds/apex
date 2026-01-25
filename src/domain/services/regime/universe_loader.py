"""
Universe Loader - Single Source of Truth for Sector/Stock Mappings.

Loads the regime verification universe YAML and derives all mappings
from it, eliminating the need for manual synchronization.

The YAML defines sectors with ETFs and stocks, and this loader provides:
- UniverseConfig: The full parsed configuration
- all_symbols: All unique symbols (market ETFs + sector ETFs + stocks)
- stock_to_sector: Map from stock symbol to sector ETF
- sector_etfs: List of all sector ETFs

Usage:
    from .universe_loader import load_universe, DEFAULT_UNIVERSE_PATH

    universe = load_universe(DEFAULT_UNIVERSE_PATH)
    print(universe.all_symbols)  # ["QQQ", "SPY", "XLK", "AAPL", ...]
    print(universe.stock_to_sector["AAPL"])  # "XLK"
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

# Default path to universe YAML (relative to project root)
DEFAULT_UNIVERSE_PATH = Path("config/universe.yaml")


class UniverseLoadError(Exception):
    """Raised when universe YAML fails validation."""

    pass


@dataclass
class MarketSymbol:
    """Market-level ETF configuration."""

    symbol: str
    name: str
    role: str


@dataclass
class SectorConfig:
    """Configuration for a single sector."""

    name: str  # Sector name (e.g., "technology")
    etf: str  # Sector ETF symbol (e.g., "XLK")
    stocks: List[str]  # Stocks in this sector

    def __post_init__(self) -> None:
        """Validate sector configuration."""
        if not self.etf:
            raise UniverseLoadError(f"Sector '{self.name}' missing ETF")
        if not self.stocks:
            raise UniverseLoadError(f"Sector '{self.name}' has no stocks")

        # Ensure uppercase
        self.etf = self.etf.upper()
        self.stocks = [s.upper() for s in self.stocks]


@dataclass
class UniverseConfig:
    """
    Complete universe configuration loaded from YAML.

    Provides derived mappings as cached properties for efficient access.
    """

    market_symbols: List[MarketSymbol]
    sectors: Dict[str, SectorConfig]  # sector_name -> SectorConfig
    quick_test: List[str]
    model_training: List[str]
    pr_validation: List[str]

    @cached_property
    def all_symbols(self) -> List[str]:
        """
        Derive all unique symbols from the universe.

        Includes: market ETFs + sector ETFs + all sector stocks.
        Replaces the deprecated 'full_test' list in YAML.

        Returns:
            Sorted list of unique symbols.
        """
        symbols: Set[str] = set()

        # Market-level ETFs
        for m in self.market_symbols:
            symbols.add(m.symbol.upper())

        # Sector ETFs and stocks
        for sector in self.sectors.values():
            symbols.add(sector.etf)
            symbols.update(sector.stocks)

        return sorted(symbols)

    @cached_property
    def stock_to_sector(self) -> Dict[str, str]:
        """
        Derive stock-to-sector ETF mapping.

        Maps each stock symbol to its sector ETF.
        Note: Stocks appearing in multiple sectors (e.g., NVDA in tech + semi)
        will be mapped to the last sector processed.

        Returns:
            Dict mapping stock symbol to sector ETF.
        """
        mapping: Dict[str, str] = {}
        for sector in self.sectors.values():
            for stock in sector.stocks:
                mapping[stock] = sector.etf
        return mapping

    @cached_property
    def sector_etfs(self) -> List[str]:
        """
        Get all sector ETF symbols.

        Returns:
            Sorted list of sector ETF symbols.
        """
        return sorted(sector.etf for sector in self.sectors.values())

    @cached_property
    def sector_names(self) -> Dict[str, str]:
        """
        Get mapping from sector ETF to human-readable sector name.

        Returns:
            Dict mapping ETF symbol to sector name.
        """
        return {
            sector.etf: sector.name.replace("_", " ").title() for sector in self.sectors.values()
        }

    @cached_property
    def market_benchmarks(self) -> Set[str]:
        """
        Get market benchmark symbols.

        Returns:
            Set of market benchmark symbols (QQQ, SPY, etc.).
        """
        return {m.symbol.upper() for m in self.market_symbols}

    def get_sector_for_symbol(self, symbol: str) -> Optional[str]:
        """
        Get the sector ETF for a given symbol.

        Args:
            symbol: Stock or ETF symbol.

        Returns:
            Sector ETF if found, None otherwise.
        """
        return self.stock_to_sector.get(symbol.upper())


def _validate_no_duplicate_symbols(sectors: Dict[str, SectorConfig]) -> None:
    """
    Validate that no symbol appears in multiple sectors.

    Note: Some symbols (like NVDA) may legitimately appear in multiple sectors
    (technology + semiconductors). This validation is currently disabled but
    can be enabled for stricter checking.

    Args:
        sectors: Dict of sector configurations.

    Raises:
        UniverseLoadError: If duplicate symbols found (when strict mode enabled).
    """
    seen: Dict[str, str] = {}  # symbol -> sector_name
    duplicates: List[str] = []

    for sector_name, sector in sectors.items():
        for stock in sector.stocks:
            if stock in seen:
                # Track duplicates but allow them (common in real data)
                duplicates.append(f"{stock} in {seen[stock]} and {sector_name}")
            else:
                seen[stock] = sector_name

    # Note: We allow duplicates for now (NVDA in tech + semi is valid)
    # Uncomment below to enforce strict uniqueness:
    # if duplicates:
    #     raise UniverseLoadError(f"Duplicate symbols across sectors: {duplicates}")


def _validate_yaml_schema(data: Dict[str, Any]) -> None:
    """
    Validate YAML schema structure.

    Args:
        data: Parsed YAML data.

    Raises:
        UniverseLoadError: If schema validation fails.
    """
    if "sectors" not in data:
        raise UniverseLoadError("Missing 'sectors' key in universe YAML")

    sectors = data["sectors"]
    if not isinstance(sectors, dict):
        raise UniverseLoadError("'sectors' must be a dictionary")

    for sector_name, sector_data in sectors.items():
        if not isinstance(sector_data, dict):
            raise UniverseLoadError(f"Sector '{sector_name}' must be a dictionary")

        if "etf" not in sector_data:
            raise UniverseLoadError(f"Sector '{sector_name}' missing 'etf' key")

        if "stocks" not in sector_data:
            raise UniverseLoadError(f"Sector '{sector_name}' missing 'stocks' key")

        stocks = sector_data["stocks"]
        if not isinstance(stocks, list):
            raise UniverseLoadError(f"Sector '{sector_name}' stocks must be a list")

        # Validate stock symbols are uppercase (or will be normalized)
        for stock in stocks:
            if not isinstance(stock, str):
                raise UniverseLoadError(f"Sector '{sector_name}' has non-string stock: {stock}")


def load_universe(path: Optional[Path] = None) -> UniverseConfig:
    """
    Load and validate universe configuration from YAML.

    Args:
        path: Path to YAML file. If None, uses DEFAULT_UNIVERSE_PATH.

    Returns:
        Validated UniverseConfig instance.

    Raises:
        UniverseLoadError: If YAML is missing, invalid, or fails validation.
        FileNotFoundError: If YAML file doesn't exist.
    """
    if path is None:
        path = DEFAULT_UNIVERSE_PATH

    # Resolve relative path from project root
    if not path.is_absolute():
        # Try from current directory first, then from project root
        if not path.exists():
            project_root = Path(__file__).parent.parent.parent.parent.parent
            path = project_root / path

    if not path.exists():
        raise FileNotFoundError(f"Universe YAML not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if data is None:
        raise UniverseLoadError(f"Empty YAML file: {path}")

    # Validate schema
    _validate_yaml_schema(data)

    # Parse market symbols
    market_symbols: List[MarketSymbol] = []
    if "market" in data:
        for m in data["market"]:
            market_symbols.append(
                MarketSymbol(
                    symbol=m["symbol"].upper(),
                    name=m.get("name", ""),
                    role=m.get("role", ""),
                )
            )

    # Parse sectors
    sectors: Dict[str, SectorConfig] = {}
    for sector_name, sector_data in data["sectors"].items():
        sectors[sector_name] = SectorConfig(
            name=sector_name,
            etf=sector_data["etf"],
            stocks=sector_data["stocks"],
        )

    # Validate no duplicates (currently just logs, doesn't fail)
    _validate_no_duplicate_symbols(sectors)

    # Parse subsets (quick_test, pr_validation, model_training, etc.)
    subsets = data.get("subsets", {})
    quick_test = [s.upper() for s in subsets.get("quick_test", data.get("quick_test", []))]
    model_training = [s.upper() for s in subsets.get("model_training", [])]
    pr_validation = [s.upper() for s in subsets.get("pr_validation", [])]

    return UniverseConfig(
        market_symbols=market_symbols,
        sectors=sectors,
        quick_test=quick_test,
        model_training=model_training,
        pr_validation=pr_validation,
    )


# Module-level singleton for convenience (lazy-loaded)
_universe_cache: Optional[UniverseConfig] = None


def get_universe() -> UniverseConfig:
    """
    Get the default universe configuration (cached).

    Returns:
        Cached UniverseConfig instance.
    """
    global _universe_cache
    if _universe_cache is None:
        _universe_cache = load_universe()
    return _universe_cache


def clear_universe_cache() -> None:
    """Clear the cached universe (useful for testing)."""
    global _universe_cache
    _universe_cache = None
