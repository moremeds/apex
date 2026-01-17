"""
Experiment specification - top-level experiment definition.

An experiment defines:
- Strategy to test
- Parameter search space
- Universe of symbols
- Temporal splitting rules
- Optimization objectives
- Execution profiles
- Reproducibility settings
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, model_validator

from .hashing import generate_experiment_id


class ParameterDef(BaseModel):
    """Definition of a single parameter in the search space."""

    name: str = Field(description="Parameter name")
    type: Literal["range", "categorical", "fixed"] = Field(description="Parameter type")

    # For range type
    min: Optional[float] = Field(default=None, description="Minimum value (range)")
    max: Optional[float] = Field(default=None, description="Maximum value (range)")
    step: Optional[float] = Field(default=None, description="Step size (range)")

    # For categorical type
    values: Optional[List[Any]] = Field(default=None, description="Allowed values (categorical)")

    # For fixed type
    value: Optional[Any] = Field(default=None, description="Fixed value")

    @model_validator(mode="after")
    def validate_type_fields(self) -> "ParameterDef":
        if self.type == "range":
            if self.min is None or self.max is None:
                raise ValueError("Range parameters require min and max")
        elif self.type == "categorical":
            if not self.values:
                raise ValueError("Categorical parameters require values list")
        elif self.type == "fixed":
            if self.value is None:
                raise ValueError("Fixed parameters require a value")
        return self

    def expand(self) -> List[Any]:
        """Expand parameter to list of concrete values."""
        if self.type == "fixed":
            return [self.value]
        elif self.type == "categorical":
            return self.values or []
        elif self.type == "range":
            if self.min is None or self.max is None:
                return []
            values: List[Any] = []
            current: float = self.min
            step = self.step or 1.0
            while current <= self.max:
                # Use int if step is integer-like
                if step == int(step) and self.min == int(self.min):
                    values.append(int(current))
                else:
                    values.append(current)
                current += step
            return values
        return []


class UniverseConfig(BaseModel):
    """Configuration for the symbol universe."""

    type: Literal["static", "dynamic", "index"] = Field(
        default="static", description="Universe type"
    )
    symbols: Optional[List[str]] = Field(default=None, description="Static list of symbols")
    rules: Optional[str] = Field(
        default=None, description="Dynamic selection rules (e.g., 'top_500_by_adv')"
    )
    as_of: Optional[str] = Field(
        default=None, description="Point-in-time reference for dynamic universe"
    )
    index: Optional[str] = Field(default=None, description="Index name for index-based universe")

    @model_validator(mode="after")
    def validate_universe(self) -> "UniverseConfig":
        if self.type == "static" and not self.symbols:
            raise ValueError("Static universe requires symbols list")
        if self.type == "dynamic" and not self.rules:
            raise ValueError("Dynamic universe requires rules")
        if self.type == "index" and not self.index:
            raise ValueError("Index universe requires index name")
        return self


class TemporalConfig(BaseModel):
    """Configuration for temporal splits (walk-forward, CPCV)."""

    # Primary method
    primary_method: Literal["walk_forward", "expanding", "sliding"] = Field(
        default="walk_forward", description="Primary splitting method"
    )
    train_days: int = Field(default=252, description="Training window in trading days")
    test_days: int = Field(default=63, description="Test window in trading days")
    step_days: Optional[int] = Field(
        default=None, description="Step size between windows (defaults to test_days)"
    )
    folds: int = Field(default=5, description="Number of walk-forward folds")

    # Purge and embargo for preventing look-ahead bias
    purge_days: int = Field(default=5, description="Gap between train and test to prevent leakage")
    embargo_days: int = Field(default=2, description="Gap after test for model decay")

    # Secondary validation method
    secondary_method: Optional[Literal["cpcv", "monte_carlo", "none"]] = Field(
        default=None, description="Secondary validation method"
    )
    cpcv_groups: int = Field(default=6, description="Number of CPCV groups")
    cpcv_test_groups: int = Field(default=2, description="Number of test groups per CPCV path")

    # Date range
    start_date: Optional[date] = Field(default=None, description="Start date")
    end_date: Optional[date] = Field(default=None, description="End date")


class OptimizationConfig(BaseModel):
    """Configuration for optimization/search."""

    method: Literal["grid", "bayesian", "random"] = Field(
        default="bayesian", description="Optimization method (Bayesian TPE is default)"
    )

    # For Bayesian optimization
    sampler: Optional[str] = Field(default="TPE", description="Optuna sampler")
    pruner: Optional[str] = Field(default="ASHA", description="Optuna pruner")
    n_trials: Optional[int] = Field(default=None, description="Max trials for Bayesian")

    # Objective
    metric: str = Field(default="sharpe", description="Optimization metric")
    direction: Literal["maximize", "minimize"] = Field(
        default="maximize", description="Optimization direction"
    )

    # Constraints
    constraints: List[Dict[str, Any]] = Field(
        default_factory=list, description="Optimization constraints"
    )


class DataConfig(BaseModel):
    """Configuration for data loading and timeframes."""

    primary_timeframe: str = Field(
        default="1d", description="Primary bar timeframe (e.g., '1d', '1h', '5m')"
    )
    secondary_timeframes: List[str] = Field(
        default_factory=list,
        description="Secondary timeframes for multi-timeframe strategies (e.g., ['1h', '4h'])",
    )

    @property
    def all_timeframes(self) -> List[str]:
        """Get all timeframes (primary + secondary) in order."""
        return [self.primary_timeframe] + self.secondary_timeframes

    @property
    def is_multi_timeframe(self) -> bool:
        """Check if this is a multi-timeframe configuration."""
        return len(self.secondary_timeframes) > 0


class ProfileConfig(BaseModel):
    """Execution profile configuration."""

    name: str = Field(description="Profile name")
    version: str = Field(default="v1", description="Profile version")

    # Cost settings
    slippage_bps: float = Field(default=5.0, description="Slippage in basis points")
    commission_per_share: float = Field(default=0.005, description="Per-share commission")
    min_commission: float = Field(default=1.0, description="Minimum commission per trade")

    # Execution settings
    fill_model: str = Field(default="close", description="Fill price model")
    volume_limit_pct: float = Field(default=0.1, description="Max volume participation")

    # Market impact
    market_impact_model: Optional[str] = Field(default=None, description="Market impact model name")


class ReproducibilityConfig(BaseModel):
    """Configuration for reproducibility."""

    random_seed: int = Field(default=42, description="Random seed")
    data_version: str = Field(description="Data version identifier")
    code_version: Optional[str] = Field(default=None, description="Code/strategy version")


class ExperimentSpec(BaseModel):
    """
    Top-level experiment specification.

    An experiment defines everything needed to run a systematic backtest:
    - Strategy and parameter space
    - Symbol universe
    - Temporal splits
    - Optimization settings
    - Execution profiles
    - Reproducibility settings
    """

    name: str = Field(description="Experiment name")
    description: Optional[str] = Field(default=None, description="Experiment description")
    strategy: str = Field(description="Strategy name (registered in strategy registry)")

    # Parameter space
    parameters: Dict[str, Union[ParameterDef, Dict[str, Any]]] = Field(
        description="Parameter definitions"
    )

    # Universe
    universe: UniverseConfig = Field(description="Symbol universe configuration")

    # Temporal
    temporal: TemporalConfig = Field(description="Temporal split configuration")

    # Data configuration (timeframes, sources)
    data: DataConfig = Field(default_factory=DataConfig, description="Data loading configuration")

    # Optimization
    optimization: OptimizationConfig = Field(
        default_factory=OptimizationConfig, description="Optimization configuration"
    )

    # Profiles (screening vs validation)
    profiles: Dict[str, ProfileConfig] = Field(
        default_factory=dict, description="Execution profiles"
    )

    # Reproducibility
    reproducibility: ReproducibilityConfig = Field(description="Reproducibility settings")

    # Computed fields
    experiment_id: Optional[str] = Field(default=None, description="Generated experiment ID")

    @model_validator(mode="after")
    def generate_id(self) -> "ExperimentSpec":
        """Generate experiment ID after validation."""
        if self.experiment_id is None:
            self.experiment_id = generate_experiment_id(
                name=self.name,
                strategy=self.strategy,
                parameters={
                    k: v.model_dump() if isinstance(v, ParameterDef) else v
                    for k, v in self.parameters.items()
                },
                universe=self.universe.model_dump(),
                temporal=self.temporal.model_dump(),
                data_version=self.reproducibility.data_version,
                profile_version=(
                    next(iter(self.profiles.values()), ProfileConfig(name="default")).version
                    if self.profiles
                    else None
                ),
                code_version=self.reproducibility.code_version,
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def normalize_parameters(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Convert dict-style parameters to ParameterDef objects."""
        if "parameters" in values:
            params = values["parameters"]
            normalized = {}
            for name, defn in params.items():
                if isinstance(defn, dict):
                    defn["name"] = name
                    normalized[name] = defn
                else:
                    normalized[name] = defn
            values["parameters"] = normalized
        return values

    def get_parameter_defs(self) -> Dict[str, ParameterDef]:
        """Get parameter definitions as ParameterDef objects."""
        result = {}
        for name, defn in self.parameters.items():
            if isinstance(defn, ParameterDef):
                result[name] = defn
            else:
                result[name] = ParameterDef(name=name, **defn)
        return result

    def expand_parameter_grid(self) -> List[Dict[str, Any]]:
        """Expand parameters to full grid of combinations."""
        from itertools import product

        param_defs = self.get_parameter_defs()

        # Get all possible values for each parameter
        param_values = {name: defn.expand() for name, defn in param_defs.items()}

        # Generate all combinations
        if not param_values:
            return [{}]

        keys = list(param_values.keys())
        values_list = [param_values[k] for k in keys]

        combinations = []
        for combo in product(*values_list):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def get_symbols(self) -> List[str]:
        """Get list of symbols from universe configuration."""
        if self.universe.type == "static":
            return self.universe.symbols or []
        # For dynamic/index, would need to resolve at runtime
        return []

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExperimentSpec":
        """Load experiment spec from YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save experiment spec to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
