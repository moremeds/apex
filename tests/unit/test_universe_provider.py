"""
Unit tests for UniverseProvider.

Tests:
- YamlUniverseProvider loading
- Symbol config retrieval
- Timeframe filtering
- Group management
- Dynamic add/remove
"""

from pathlib import Path

import pytest
import yaml

from src.domain.signals.universe.base import SymbolConfig
from src.domain.signals.universe.yaml_provider import YamlUniverseProvider


class TestYamlUniverseProvider:
    """Tests for YamlUniverseProvider."""

    @pytest.fixture
    def sample_config_file(self, tmp_path: Path) -> Path:
        """Create a temporary config file for testing."""
        config = {
            "version": 1,
            "defaults": {
                "timeframes": ["1h", "4h", "1d"],
                "min_volume_usd": 1000000,
            },
            "groups": {
                "tech": {
                    "symbols": ["AAPL", "MSFT", "GOOGL"],
                    "timeframes": ["15m", "1h", "4h"],
                },
                "etf": {
                    "symbols": ["SPY", "QQQ"],
                    "timeframes": ["5m", "15m", "1h"],
                },
            },
            "overrides": {
                "AAPL": {
                    "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
                },
                "TSLA": {
                    "timeframes": ["1h", "4h"],
                },
            },
        }

        config_path = tmp_path / "universe.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        return config_path

    def test_load_config(self, sample_config_file: Path) -> None:
        """Test loading configuration from YAML."""
        provider = YamlUniverseProvider(str(sample_config_file))

        # Should have symbols from groups and overrides
        symbols = provider.get_symbols()
        assert "AAPL" in symbols
        assert "MSFT" in symbols
        assert "GOOGL" in symbols
        assert "SPY" in symbols
        assert "QQQ" in symbols
        assert "TSLA" in symbols  # From overrides

    def test_get_symbol_config(self, sample_config_file: Path) -> None:
        """Test retrieving individual symbol configuration."""
        provider = YamlUniverseProvider(str(sample_config_file))

        # AAPL has override
        aapl = provider.get_symbol_config("AAPL")
        assert aapl is not None
        assert "1m" in aapl.timeframes  # From override
        assert aapl.group == "tech"

        # MSFT uses group defaults
        msft = provider.get_symbol_config("MSFT")
        assert msft is not None
        assert msft.timeframes == ["15m", "1h", "4h"]
        assert msft.group == "tech"

        # Non-existent symbol
        unknown = provider.get_symbol_config("UNKNOWN")
        assert unknown is None

    def test_get_symbols_for_timeframe(self, sample_config_file: Path) -> None:
        """Test filtering symbols by timeframe."""
        provider = YamlUniverseProvider(str(sample_config_file))

        # 1h should include all symbols with 1h
        hourly = provider.get_symbols_for_timeframe("1h")
        assert "AAPL" in hourly
        assert "MSFT" in hourly
        assert "SPY" in hourly
        assert "TSLA" in hourly

        # 1m should only include AAPL (from override)
        minute = provider.get_symbols_for_timeframe("1m")
        assert "AAPL" in minute
        assert len(minute) == 1

        # 5m should include AAPL and ETFs
        five_min = provider.get_symbols_for_timeframe("5m")
        assert "AAPL" in five_min
        assert "SPY" in five_min
        assert "QQQ" in five_min

    def test_get_all_timeframes(self, sample_config_file: Path) -> None:
        """Test getting all unique timeframes sorted by duration."""
        provider = YamlUniverseProvider(str(sample_config_file))

        timeframes = provider.get_all_timeframes()

        # Should be sorted by duration (smallest to largest)
        assert timeframes[0] == "1m"
        assert "1d" in timeframes
        assert timeframes.index("1m") < timeframes.index("5m")
        assert timeframes.index("5m") < timeframes.index("15m")
        assert timeframes.index("1h") < timeframes.index("4h")

    def test_get_groups(self, sample_config_file: Path) -> None:
        """Test getting group names."""
        provider = YamlUniverseProvider(str(sample_config_file))

        groups = provider.get_groups()
        assert "tech" in groups
        assert "etf" in groups

    def test_get_symbols_in_group(self, sample_config_file: Path) -> None:
        """Test getting symbols in a specific group."""
        provider = YamlUniverseProvider(str(sample_config_file))

        tech_symbols = provider.get_symbols_in_group("tech")
        assert "AAPL" in tech_symbols
        assert "MSFT" in tech_symbols
        assert "GOOGL" in tech_symbols
        assert "SPY" not in tech_symbols

    def test_add_symbol_dynamic(self, sample_config_file: Path) -> None:
        """Test dynamically adding a symbol."""
        provider = YamlUniverseProvider(str(sample_config_file))

        initial_count = len(provider)
        provider.add_symbol("NVDA", timeframes=["1h", "4h"])

        assert len(provider) == initial_count + 1
        assert "NVDA" in provider
        nvda = provider.get_symbol_config("NVDA")
        assert nvda is not None
        assert nvda.timeframes == ["1h", "4h"]

    def test_remove_symbol_dynamic(self, sample_config_file: Path) -> None:
        """Test dynamically removing a symbol."""
        provider = YamlUniverseProvider(str(sample_config_file))

        initial_count = len(provider)
        result = provider.remove_symbol("AAPL")

        assert result is True
        assert len(provider) == initial_count - 1
        assert "AAPL" not in provider

        # Removing non-existent returns False
        result = provider.remove_symbol("NONEXISTENT")
        assert result is False

    def test_nonexistent_config_file(self, tmp_path: Path) -> None:
        """Test handling of non-existent config file."""
        provider = YamlUniverseProvider(str(tmp_path / "nonexistent.yaml"))

        # Should initialize empty but not crash
        assert len(provider) == 0

    def test_refresh_reloads_on_change(self, sample_config_file: Path) -> None:
        """Test that refresh() reloads when file changes."""
        import asyncio
        import os

        provider = YamlUniverseProvider(str(sample_config_file))
        initial_count = len(provider)
        initial_mtime = sample_config_file.stat().st_mtime

        # Modify the file
        with open(sample_config_file) as f:
            config = yaml.safe_load(f)
        config["groups"]["new_group"] = {
            "symbols": ["NEW1", "NEW2"],
            "timeframes": ["1h"],
        }
        with open(sample_config_file, "w") as f:
            yaml.dump(config, f)

        # Force mtime to be different (works on all filesystems)
        os.utime(sample_config_file, (initial_mtime + 2, initial_mtime + 2))

        # Refresh should reload
        asyncio.run(provider.refresh())

        assert len(provider) == initial_count + 2
        assert "NEW1" in provider

    def test_symbol_config_has_timeframe(self) -> None:
        """Test SymbolConfig.has_timeframe method."""
        config = SymbolConfig(
            symbol="TEST",
            timeframes=["1h", "4h", "1d"],
        )

        assert config.has_timeframe("1h") is True
        assert config.has_timeframe("4h") is True
        assert config.has_timeframe("1m") is False

    def test_contains_operator(self, sample_config_file: Path) -> None:
        """Test __contains__ operator."""
        provider = YamlUniverseProvider(str(sample_config_file))

        assert "AAPL" in provider
        assert "UNKNOWN" not in provider

    def test_len_operator(self, sample_config_file: Path) -> None:
        """Test __len__ operator."""
        provider = YamlUniverseProvider(str(sample_config_file))

        # 3 tech + 2 etf + 1 TSLA from overrides = 6
        assert len(provider) == 6
