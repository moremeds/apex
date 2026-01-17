"""
Parameter Provenance Tracking for Regime Classification.

Provides auditability for regime parameters:
- Track parameter sources (symbol-specific, group, default)
- Store training validation metrics (PBO, DSR, OOS Sharpe)
- Enable parameter lineage queries

This is critical for answering "why these parameters?" questions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Optional


@dataclass
class ParamProvenance:
    """
    Track parameter sources for auditability.

    Each parameter set has a stable ID (hash of params + training metadata)
    that enables tracking across time and reproducibility.

    Attributes:
        param_set_id: Hash of param dict + training metadata (stable identifier)
        source: "symbol-specific" | "group" | "default"
        symbol: Symbol these params apply to
        group: Optional group name (e.g., "semiconductors" -> SMH params)
        trained_data_end: Last date of training data
        trainer_version: Version of training algorithm (e.g., "optuna-tpe-v2.1")
        walk_forward_folds: Number of WFO folds used in validation
        pbo_value: Probability of Backtest Overfitting (lower is better)
        pbo_threshold: Max acceptable PBO (default 0.5)
        dsr_value: Deflated Sharpe Ratio (higher is better)
        dsr_threshold: Min acceptable DSR (default 1.0)
        oos_sharpe: Out-of-sample Sharpe ratio
        validation_passed: Whether all validation metrics met thresholds
    """

    param_set_id: str = ""
    source: str = "default"  # "symbol-specific" | "group" | "default"
    symbol: str = ""
    group: Optional[str] = None

    # Training metadata
    trained_data_end: Optional[date] = None
    trainer_version: Optional[str] = None

    # Validation results from last training
    walk_forward_folds: Optional[int] = None
    pbo_value: Optional[float] = None
    pbo_threshold: float = 0.5
    dsr_value: Optional[float] = None
    dsr_threshold: float = 1.0
    oos_sharpe: Optional[float] = None
    validation_passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "param_set_id": self.param_set_id,
            "source": self.source,
            "symbol": self.symbol,
            "group": self.group,
            "trained_data_end": (
                self.trained_data_end.isoformat() if self.trained_data_end else None
            ),
            "trainer_version": self.trainer_version,
            "walk_forward_folds": self.walk_forward_folds,
            "pbo_value": self.pbo_value,
            "pbo_threshold": self.pbo_threshold,
            "dsr_value": self.dsr_value,
            "dsr_threshold": self.dsr_threshold,
            "oos_sharpe": self.oos_sharpe,
            "validation_passed": self.validation_passed,
        }

    @property
    def is_validated(self) -> bool:
        """Check if params have been properly validated."""
        return (
            self.walk_forward_folds is not None
            and self.pbo_value is not None
            and self.dsr_value is not None
        )

    @property
    def pbo_ok(self) -> bool:
        """Check if PBO is below threshold."""
        return self.pbo_value is not None and self.pbo_value < self.pbo_threshold

    @property
    def dsr_ok(self) -> bool:
        """Check if DSR is above threshold."""
        return self.dsr_value is not None and self.dsr_value > self.dsr_threshold

    @property
    def oos_ok(self) -> bool:
        """Check if OOS Sharpe is positive."""
        return self.oos_sharpe is not None and self.oos_sharpe > 0

    @classmethod
    def compute_param_set_id(
        cls,
        params: Dict[str, Any],
        symbol: str,
        trained_data_end: Optional[date] = None,
    ) -> str:
        """
        Compute a stable hash for a parameter set.

        The hash is based on:
        - Parameter values (sorted for stability)
        - Symbol
        - Training end date (if available)

        Args:
            params: Parameter dictionary
            symbol: Symbol the params apply to
            trained_data_end: Optional training end date

        Returns:
            8-character hex hash
        """
        # Create stable string representation
        data = {
            "params": dict(sorted(params.items())),
            "symbol": symbol,
            "trained_data_end": (
                trained_data_end.isoformat() if trained_data_end else None
            ),
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:8]

    @classmethod
    def from_params(
        cls,
        params: Dict[str, Any],
        symbol: str,
        source: str = "default",
        group: Optional[str] = None,
        trained_data_end: Optional[date] = None,
        trainer_version: Optional[str] = None,
    ) -> "ParamProvenance":
        """
        Create provenance from a parameter dictionary.

        Args:
            params: Parameter dictionary
            symbol: Symbol the params apply to
            source: Parameter source type
            group: Optional group name
            trained_data_end: Training data cutoff date
            trainer_version: Training algorithm version

        Returns:
            ParamProvenance instance
        """
        param_set_id = cls.compute_param_set_id(params, symbol, trained_data_end)
        return cls(
            param_set_id=param_set_id,
            source=source,
            symbol=symbol,
            group=group,
            trained_data_end=trained_data_end,
            trainer_version=trainer_version,
        )


@dataclass
class ParamSource:
    """
    Tracks the source of a single parameter value.

    Used for displaying parameter tables showing which params
    came from which source (symbol-specific, group, default).
    """

    param_name: str
    value: Any
    source: str  # "symbol-specific" | "group" | "default"
    trained_on: Optional[date] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "param_name": self.param_name,
            "value": self.value,
            "source": self.source,
            "trained_on": self.trained_on.isoformat() if self.trained_on else None,
        }


@dataclass
class ParamProvenanceSet:
    """
    Collection of parameter sources for a symbol.

    Aggregates individual param sources with overall provenance metadata.
    """

    symbol: str
    provenance: ParamProvenance
    param_sources: Dict[str, ParamSource] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "symbol": self.symbol,
            "provenance": self.provenance.to_dict(),
            "param_sources": {k: v.to_dict() for k, v in self.param_sources.items()},
        }

    def get_params_by_source(self, source: str) -> Dict[str, ParamSource]:
        """Get all parameters from a specific source."""
        return {k: v for k, v in self.param_sources.items() if v.source == source}
