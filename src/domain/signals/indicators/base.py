"""
Indicator Protocol and Base Class.

Defines the unified interface for all technical indicators in the trading signal engine.
All indicators must implement the Indicator protocol to be auto-discovered and used
by the IndicatorEngine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import pandas as pd

from ..models import SignalCategory


@runtime_checkable
class Indicator(Protocol):
    """
    Protocol for all technical indicators.

    Each indicator must define:
    - name: Unique identifier (e.g., "rsi", "macd")
    - category: SignalCategory (MOMENTUM, TREND, etc.)
    - required_fields: OHLCV fields needed (e.g., ["close"] or ["high", "low", "close"])
    - warmup_periods: Minimum bars needed before valid output
    - default_params: Default calculation parameters

    And implement:
    - calculate(): Compute indicator values from OHLCV data
    - get_state(): Extract state dict for rule evaluation
    """

    name: str
    category: SignalCategory
    required_fields: List[str]
    warmup_periods: int

    @property
    def default_params(self) -> Dict[str, Any]:
        """Default parameters for this indicator."""
        ...

    def calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate indicator values from OHLCV data.

        Args:
            data: DataFrame with OHLCV columns (open, high, low, close, volume)
            params: Calculation parameters (merged with default_params)

        Returns:
            DataFrame with indicator columns, same index as input
        """
        ...

    def get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract state dictionary for rule evaluation.

        The state dict should contain:
        - "value": Primary indicator value
        - Any additional fields needed by rules (e.g., "zone", "direction")

        Args:
            current: Current row of indicator values
            previous: Previous row (None if first evaluation)
            params: Optional parameters for state extraction (e.g., custom thresholds).
                    Merged with default_params. Allows rules to use different thresholds
                    for the same indicator calculation.

        Returns:
            State dictionary for rule evaluation
        """
        ...


class IndicatorBase(ABC):
    """
    Abstract base class for indicators with common functionality.

    Provides:
    - Parameter merging with defaults
    - Data validation
    - Common state extraction patterns

    Subclasses must implement:
    - _calculate(): Core calculation logic
    - _get_state(): State extraction logic
    """

    name: str = ""
    category: SignalCategory = SignalCategory.MOMENTUM
    required_fields: List[str] = ["close"]
    warmup_periods: int = 14

    _default_params: Dict[str, Any] = {}

    @property
    def default_params(self) -> Dict[str, Any]:
        """Default parameters for this indicator."""
        return self._default_params.copy()

    def calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate indicator with parameter merging and validation.

        Args:
            data: OHLCV DataFrame
            params: User-provided parameters

        Returns:
            DataFrame with indicator columns
        """
        # Merge with defaults
        merged_params = {**self.default_params, **params}

        # Validate required fields
        self._validate_data(data)

        # Delegate to subclass
        return self._calculate(data, merged_params)

    def get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract state with parameter merging.

        Args:
            current: Current indicator values
            previous: Previous indicator values (may be None)
            params: Optional parameters for state extraction (merged with defaults)

        Returns:
            State dictionary
        """
        merged_params = {**self.default_params, **(params or {})}
        return self._get_state(current, previous, merged_params)

    @abstractmethod
    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Core calculation logic - must be implemented by subclasses.

        Args:
            data: Validated OHLCV DataFrame
            params: Merged parameters

        Returns:
            DataFrame with indicator columns
        """
        ...

    @abstractmethod
    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        State extraction logic - must be implemented by subclasses.

        Args:
            current: Current indicator values
            previous: Previous indicator values
            params: Merged parameters (defaults + any overrides)

        Returns:
            State dictionary with at minimum {"value": ...}
        """
        ...

    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate that required fields are present.

        Args:
            data: Input DataFrame

        Raises:
            ValueError: If required fields are missing
        """
        missing = [f for f in self.required_fields if f not in data.columns]
        if missing:
            raise ValueError(
                f"Indicator {self.name} requires fields {self.required_fields}, "
                f"missing: {missing}"
            )

    def _safe_get(
        self,
        series: Optional[pd.Series],
        field: str,
        default: Any = None,
    ) -> Any:
        """
        Safely get value from series with default.

        Args:
            series: Pandas Series (may be None)
            field: Field name to get
            default: Default value if missing

        Returns:
            Field value or default
        """
        if series is None:
            return default
        return series.get(field, default)
