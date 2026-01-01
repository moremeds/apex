"""
Trial specification - single parameter combination to test.

A trial represents one point in the parameter space being tested
across multiple symbols and time windows.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .hashing import generate_trial_id


class TrialSpec(BaseModel):
    """
    Specification for a single trial (parameter combination).

    A trial tests one specific set of parameter values across:
    - Multiple symbols in the universe
    - Multiple time windows (walk-forward folds)
    - Different execution profiles (screening vs validation)

    The trial_id is deterministically generated from the experiment_id
    and parameter values, ensuring reproducibility.
    """

    experiment_id: str = Field(description="Parent experiment ID")
    params: Dict[str, Any] = Field(description="Parameter values for this trial")

    # Optional metadata
    trial_index: Optional[int] = Field(default=None, description="Index in grid search")
    suggested_by: Optional[str] = Field(
        default=None, description="Optimization method that suggested this (e.g., 'TPE')"
    )

    # Computed
    trial_id: Optional[str] = Field(default=None, description="Generated trial ID")

    def model_post_init(self, __context) -> None:
        """Generate trial ID after initialization."""
        if self.trial_id is None:
            self.trial_id = generate_trial_id(
                experiment_id=self.experiment_id,
                params=self.params,
            )

    def get_param(self, name: str, default: Any = None) -> Any:
        """Get a parameter value with optional default."""
        return self.params.get(name, default)

    def to_strategy_params(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for strategy initialization."""
        return dict(self.params)

    @classmethod
    def from_grid(
        cls,
        experiment_id: str,
        param_combinations: List[Dict[str, Any]],
    ) -> List["TrialSpec"]:
        """Create trial specs from list of parameter combinations."""
        return [
            cls(
                experiment_id=experiment_id,
                params=params,
                trial_index=i,
            )
            for i, params in enumerate(param_combinations)
        ]
