"""
Universe Consistency Tests (Drift Guard).

These tests ensure that the universe YAML remains the single source of truth
and that derived mappings are consistent. Run these tests after any changes
to config/signals/regime_verification_universe.yaml.

Tests verify:
1. All YAML symbols have valid sector assignments
2. No duplicate symbols across mutually exclusive sectors
3. all_symbols is correctly derived (no manual full_test list)
4. YAML schema validation catches malformed data
5. Sector ETF to name mapping is complete

Usage:
    pytest tests/unit/test_universe_consistency.py -v
"""

import pytest

from src.domain.services.regime.universe_loader import (
    SectorConfig,
    UniverseConfig,
    UniverseLoadError,
    clear_universe_cache,
    get_universe,
    load_universe,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear universe cache before each test."""
    clear_universe_cache()
    yield
    clear_universe_cache()


class TestUniverseLoading:
    """Test universe YAML loading and parsing."""

    def test_load_default_universe(self):
        """Default universe YAML loads successfully."""
        universe = load_universe()
        assert isinstance(universe, UniverseConfig)
        assert len(universe.sectors) > 0

    def test_load_universe_has_market_symbols(self):
        """Universe includes market benchmark symbols."""
        universe = load_universe()
        assert len(universe.market_symbols) >= 2  # QQQ, SPY at minimum
        symbols = {m.symbol for m in universe.market_symbols}
        assert "QQQ" in symbols
        assert "SPY" in symbols

    def test_load_universe_has_all_sp500_sectors(self):
        """Universe covers all 11 S&P 500 sectors."""
        universe = load_universe()
        sector_names = set(universe.sectors.keys())

        # 11 S&P sectors (semiconductors is a sub-sector but included)
        expected_sectors = {
            "technology",
            "semiconductors",
            "financials",
            "healthcare",
            "energy",
            "consumer_discretionary",
            "industrials",
            "consumer_staples",
            "utilities",
            "materials",
            "real_estate",
        }

        for sector in expected_sectors:
            assert sector in sector_names, f"Missing sector: {sector}"


class TestDerivedMappings:
    """Test that derived mappings are correct."""

    def test_all_yaml_symbols_have_sector_mapping(self):
        """Every stock in YAML has a sector assignment in stock_to_sector."""
        universe = load_universe()

        # Known overlaps: stocks that appear in multiple sectors
        # These map to the LAST sector processed (dict iteration order)
        # NVDA/AMD appear in both technology and semiconductors
        known_overlaps = {"NVDA", "AMD"}

        for sector in universe.sectors.values():
            for stock in sector.stocks:
                assert stock in universe.stock_to_sector, (
                    f"Stock {stock} from sector {sector.name} " f"not in stock_to_sector mapping"
                )
                # Verify it maps to the sector ETF (skip overlap stocks)
                if stock not in known_overlaps:
                    assert universe.stock_to_sector[stock] == sector.etf, (
                        f"Stock {stock} maps to {universe.stock_to_sector[stock]} "
                        f"but expected {sector.etf}"
                    )

    def test_full_symbols_list_derived_correctly(self):
        """all_symbols includes all ETFs and stocks without duplicates."""
        universe = load_universe()
        all_symbols = universe.all_symbols

        # Should be sorted
        assert all_symbols == sorted(all_symbols)

        # Should have no duplicates
        assert len(all_symbols) == len(set(all_symbols))

        # Should include market ETFs
        for m in universe.market_symbols:
            assert m.symbol in all_symbols

        # Should include sector ETFs
        for sector in universe.sectors.values():
            assert sector.etf in all_symbols

        # Should include all stocks
        for sector in universe.sectors.values():
            for stock in sector.stocks:
                assert stock in all_symbols

    def test_sector_etfs_list_complete(self):
        """sector_etfs includes all sector ETFs from YAML."""
        universe = load_universe()
        sector_etfs = set(universe.sector_etfs)

        expected_etfs = {sector.etf for sector in universe.sectors.values()}
        assert sector_etfs == expected_etfs

    def test_sector_names_mapping_complete(self):
        """sector_names maps all sector ETFs to readable names."""
        universe = load_universe()

        for sector in universe.sectors.values():
            assert sector.etf in universe.sector_names
            # Name should be properly formatted
            name = universe.sector_names[sector.etf]
            assert len(name) > 0
            assert name[0].isupper()  # Should be title case

    def test_market_benchmarks_complete(self):
        """market_benchmarks includes all market-level ETFs."""
        universe = load_universe()

        for m in universe.market_symbols:
            assert m.symbol in universe.market_benchmarks


class TestDuplicateDetection:
    """Test duplicate symbol detection."""

    def test_stocks_within_sector_no_duplicates(self):
        """No stock appears twice within the same sector."""
        universe = load_universe()

        for sector in universe.sectors.values():
            unique_stocks = set(sector.stocks)
            assert len(unique_stocks) == len(
                sector.stocks
            ), f"Sector {sector.name} has duplicate stocks"

    def test_etf_symbols_unique(self):
        """Each sector has a unique ETF symbol."""
        universe = load_universe()

        etf_to_sector = {}
        for sector in universe.sectors.values():
            if sector.etf in etf_to_sector:
                pytest.fail(
                    f"ETF {sector.etf} used by multiple sectors: "
                    f"{etf_to_sector[sector.etf]} and {sector.name}"
                )
            etf_to_sector[sector.etf] = sector.name


class TestSchemaValidation:
    """Test YAML schema validation catches errors."""

    def test_missing_sectors_key_raises_error(self, tmp_path):
        """YAML without 'sectors' key raises UniverseLoadError."""
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("market:\n  - symbol: QQQ\n")

        with pytest.raises(UniverseLoadError, match="Missing 'sectors' key"):
            load_universe(invalid_yaml)

    def test_sector_missing_etf_raises_error(self, tmp_path):
        """Sector without 'etf' key raises UniverseLoadError."""
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("""
sectors:
  tech:
    stocks:
      - AAPL
""")

        with pytest.raises(UniverseLoadError, match="missing 'etf' key"):
            load_universe(invalid_yaml)

    def test_sector_missing_stocks_raises_error(self, tmp_path):
        """Sector without 'stocks' key raises UniverseLoadError."""
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("""
sectors:
  tech:
    etf: XLK
""")

        with pytest.raises(UniverseLoadError, match="missing 'stocks' key"):
            load_universe(invalid_yaml)

    def test_empty_yaml_raises_error(self, tmp_path):
        """Empty YAML file raises UniverseLoadError."""
        empty_yaml = tmp_path / "empty.yaml"
        empty_yaml.write_text("")

        with pytest.raises(UniverseLoadError, match="Empty YAML"):
            load_universe(empty_yaml)

    def test_nonexistent_file_raises_error(self, tmp_path):
        """Nonexistent file raises FileNotFoundError."""
        missing = tmp_path / "missing.yaml"

        with pytest.raises(FileNotFoundError):
            load_universe(missing)


class TestSectorConfigValidation:
    """Test SectorConfig dataclass validation."""

    def test_sector_config_normalizes_uppercase(self):
        """SectorConfig normalizes symbols to uppercase."""
        sector = SectorConfig(
            name="test",
            etf="xlk",
            stocks=["aapl", "msft"],
        )

        assert sector.etf == "XLK"
        assert sector.stocks == ["AAPL", "MSFT"]

    def test_sector_config_empty_etf_raises_error(self):
        """SectorConfig with empty ETF raises UniverseLoadError."""
        with pytest.raises(UniverseLoadError, match="missing ETF"):
            SectorConfig(name="test", etf="", stocks=["AAPL"])

    def test_sector_config_empty_stocks_raises_error(self):
        """SectorConfig with empty stocks raises UniverseLoadError."""
        with pytest.raises(UniverseLoadError, match="has no stocks"):
            SectorConfig(name="test", etf="XLK", stocks=[])


class TestCacheFunction:
    """Test universe cache functionality."""

    def test_get_universe_returns_cached(self):
        """get_universe returns same instance on subsequent calls."""
        u1 = get_universe()
        u2 = get_universe()
        assert u1 is u2

    def test_clear_cache_resets(self):
        """clear_universe_cache allows fresh load."""
        u1 = get_universe()
        clear_universe_cache()
        u2 = get_universe()
        # Not the same object after cache clear
        assert u1 is not u2
        # But same content
        assert u1.all_symbols == u2.all_symbols


class TestIntegrationWithModels:
    """Test that models.py correctly uses derived mappings."""

    def test_models_stock_to_sector_matches_loader(self):
        """STOCK_TO_SECTOR in models matches loader derivation."""
        from src.domain.services.regime.models import STOCK_TO_SECTOR

        universe = load_universe()
        assert STOCK_TO_SECTOR == universe.stock_to_sector

    def test_models_sector_names_matches_loader(self):
        """SECTOR_NAMES in models matches loader derivation."""
        from src.domain.services.regime.models import SECTOR_NAMES

        universe = load_universe()
        assert SECTOR_NAMES == universe.sector_names

    def test_models_sector_etfs_matches_loader(self):
        """SECTOR_ETFS in models matches loader derivation."""
        from src.domain.services.regime.models import SECTOR_ETFS

        universe = load_universe()
        assert SECTOR_ETFS == set(universe.sector_etfs)

    def test_models_market_benchmarks_matches_loader(self):
        """MARKET_BENCHMARKS in models matches loader derivation."""
        from src.domain.services.regime.models import MARKET_BENCHMARKS

        universe = load_universe()
        assert MARKET_BENCHMARKS == universe.market_benchmarks


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_get_sector_for_known_stock(self):
        """get_sector_for_symbol returns correct sector."""
        universe = load_universe()

        # Test a few known stocks
        assert universe.get_sector_for_symbol("AAPL") == "XLK"
        assert universe.get_sector_for_symbol("JPM") == "XLF"
        assert universe.get_sector_for_symbol("XOM") == "XLE"

    def test_get_sector_for_unknown_stock(self):
        """get_sector_for_symbol returns None for unknown symbol."""
        universe = load_universe()
        assert universe.get_sector_for_symbol("UNKNOWN") is None

    def test_get_sector_case_insensitive(self):
        """get_sector_for_symbol is case-insensitive."""
        universe = load_universe()
        assert universe.get_sector_for_symbol("aapl") == "XLK"
        assert universe.get_sector_for_symbol("Aapl") == "XLK"

    def test_quick_test_symbols_all_exist(self):
        """All quick_test symbols exist in all_symbols."""
        universe = load_universe()
        all_set = set(universe.all_symbols)

        for symbol in universe.quick_test:
            assert symbol in all_set, f"quick_test symbol {symbol} not in all_symbols"
