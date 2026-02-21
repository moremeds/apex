"""Tests for source_priority config parsing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


class TestSourcePriorityConfig:
    def test_config_parses_source_priority(self) -> None:
        """Load base.yaml, verify config.historical_data.source_priority."""
        from config.config_manager import ConfigManager

        config = ConfigManager().load()
        assert config.historical_data is not None
        assert config.historical_data.source_priority == ["fmp", "yahoo"]

    def test_config_default_when_missing(self) -> None:
        """HistoricalDataConfig defaults to ["fmp", "yahoo"] when not specified."""
        from config.models import HistoricalDataConfig

        hdc = HistoricalDataConfig()
        assert hdc.source_priority == ["fmp", "yahoo"]

    def test_historical_data_manager_default_priority(self) -> None:
        """HistoricalDataManager constructor defaults to ["fmp", "yahoo"]."""
        from src.services.historical_data_manager import HistoricalDataManager

        with patch.object(HistoricalDataManager, "_init_sources"):
            manager = HistoricalDataManager(base_dir=Path("/tmp/test_hdm"))
            assert manager._source_priority == ["fmp", "yahoo"]

    def test_historical_data_manager_respects_explicit_priority(self) -> None:
        """HistoricalDataManager uses explicitly passed priority."""
        from src.services.historical_data_manager import HistoricalDataManager

        with patch.object(HistoricalDataManager, "_init_sources"):
            manager = HistoricalDataManager(
                base_dir=Path("/tmp/test_hdm"),
                source_priority=["yahoo"],
            )
            assert manager._source_priority == ["yahoo"]
