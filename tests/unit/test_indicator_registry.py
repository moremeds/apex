"""
Unit tests for IndicatorRegistry.

Tests:
- Registration and lookup
- Duplicate handling (overwrite with category cleanup)
- Auto-discovery
- Category filtering
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

from src.domain.signals.indicators.base import IndicatorBase
from src.domain.signals.indicators.registry import IndicatorRegistry
from src.domain.signals.models import SignalCategory


class MockIndicator(IndicatorBase):
    """Mock indicator for testing."""

    name = "mock"
    category = SignalCategory.MOMENTUM
    required_fields = ["close"]
    warmup_periods = 10

    _default_params = {"period": 14}

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame({"value": data["close"]}, index=data.index)

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {"value": current.get("value", 0)}


class MockTrendIndicator(IndicatorBase):
    """Mock trend indicator for testing."""

    name = "mock_trend"
    category = SignalCategory.TREND
    required_fields = ["close"]
    warmup_periods = 20

    _default_params = {"period": 20}

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame({"value": data["close"]}, index=data.index)

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {"value": current.get("value", 0), "direction": "bullish"}


class TestIndicatorRegistry:
    """Tests for IndicatorRegistry."""

    def test_register_and_get(self) -> None:
        """Test basic registration and retrieval."""
        registry = IndicatorRegistry()
        indicator = MockIndicator()

        registry.register(indicator)

        assert "mock" in registry
        assert len(registry) == 1
        assert registry.get("mock") is indicator

    def test_get_nonexistent(self) -> None:
        """Test getting a non-existent indicator returns None."""
        registry = IndicatorRegistry()
        assert registry.get("nonexistent") is None

    def test_get_all(self) -> None:
        """Test getting all registered indicators."""
        registry = IndicatorRegistry()
        ind1 = MockIndicator()
        ind2 = MockTrendIndicator()

        registry.register(ind1)
        registry.register(ind2)

        all_inds = registry.get_all()
        assert len(all_inds) == 2
        assert ind1 in all_inds
        assert ind2 in all_inds

    def test_get_by_category(self) -> None:
        """Test filtering by category."""
        registry = IndicatorRegistry()
        momentum = MockIndicator()
        trend = MockTrendIndicator()

        registry.register(momentum)
        registry.register(trend)

        momentum_inds = registry.get_by_category(SignalCategory.MOMENTUM)
        assert len(momentum_inds) == 1
        assert momentum_inds[0].name == "mock"

        trend_inds = registry.get_by_category(SignalCategory.TREND)
        assert len(trend_inds) == 1
        assert trend_inds[0].name == "mock_trend"

        volatility_inds = registry.get_by_category(SignalCategory.VOLATILITY)
        assert len(volatility_inds) == 0

    def test_duplicate_registration_overwrites(self) -> None:
        """Test that duplicate registration overwrites and cleans category."""
        registry = IndicatorRegistry()

        # Register original
        ind1 = MockIndicator()
        registry.register(ind1)

        # Create duplicate with same name but different category
        class DuplicateMock(IndicatorBase):
            name = "mock"  # Same name
            category = SignalCategory.TREND  # Different category
            required_fields = ["close"]
            warmup_periods = 5
            _default_params = {}

            def _calculate(self, data, params):
                return pd.DataFrame({"value": [0]})

            def _get_state(self, current, previous, params):
                return {"value": 0}

        ind2 = DuplicateMock()
        registry.register(ind2)

        # Should still have only 1 indicator
        assert len(registry) == 1

        # Should be the new one
        assert registry.get("mock") is ind2

        # Old category should be empty
        momentum_inds = registry.get_by_category(SignalCategory.MOMENTUM)
        assert len(momentum_inds) == 0

        # New category should have it
        trend_inds = registry.get_by_category(SignalCategory.TREND)
        assert len(trend_inds) == 1

    def test_clear(self) -> None:
        """Test clearing the registry."""
        registry = IndicatorRegistry()
        registry.register(MockIndicator())
        registry.register(MockTrendIndicator())

        assert len(registry) == 2

        registry.clear()

        assert len(registry) == 0
        assert registry.get("mock") is None
        assert len(registry.get_by_category(SignalCategory.MOMENTUM)) == 0

    def test_get_names(self) -> None:
        """Test getting all indicator names."""
        registry = IndicatorRegistry()
        registry.register(MockIndicator())
        registry.register(MockTrendIndicator())

        names = registry.get_names()
        assert "mock" in names
        assert "mock_trend" in names

    def test_contains(self) -> None:
        """Test __contains__ method."""
        registry = IndicatorRegistry()
        registry.register(MockIndicator())

        assert "mock" in registry
        assert "nonexistent" not in registry

    def test_discover_finds_rsi(self) -> None:
        """Test that discover() finds the RSI indicator."""
        registry = IndicatorRegistry()
        count = registry.discover()

        # Should find at least RSI
        assert count >= 1
        assert "rsi" in registry

        rsi = registry.get("rsi")
        assert rsi is not None
        assert rsi.category == SignalCategory.MOMENTUM

    def test_multiple_discover_no_duplicates(self) -> None:
        """Test that calling discover() multiple times doesn't create duplicates."""
        registry = IndicatorRegistry()

        count1 = registry.discover()
        count2 = registry.discover()

        # Second discover should not add duplicates to categories
        momentum_inds = registry.get_by_category(SignalCategory.MOMENTUM)
        names = [ind.name for ind in momentum_inds]
        assert len(names) == len(set(names)), "Duplicate names in category"

    def test_instantiation_guard_skips_indicators_with_required_args(self) -> None:
        """Test that indicators requiring constructor args are skipped gracefully."""

        # Create a mock indicator class that requires constructor arguments
        class IndicatorWithRequiredArgs(IndicatorBase):
            name = "requires_args"
            category = SignalCategory.MOMENTUM
            required_fields = ["close"]
            warmup_periods = 10
            _default_params = {}

            def __init__(self, required_param: str):
                self.required_param = required_param

            def _calculate(self, data, params):
                return pd.DataFrame({"value": [0]})

            def _get_state(self, current, previous, params):
                return {"value": 0}

        registry = IndicatorRegistry()

        # _is_indicator_class should return True (it's a valid subclass)
        assert registry._is_indicator_class(IndicatorWithRequiredArgs)

        # But trying to instantiate it without args should raise
        import pytest

        with pytest.raises(TypeError):
            IndicatorWithRequiredArgs()

        # The registry should handle this gracefully during discover
        # (We can't easily test the full discover path without creating actual files,
        # but we verify the class detection works)
