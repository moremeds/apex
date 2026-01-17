"""
Grid search optimizer - exhaustive parameter space search.
"""

from itertools import product
from typing import Any, Dict, Iterator, List

from ..core import ExperimentSpec


class GridOptimizer:
    """
    Grid search optimizer.

    Generates all combinations from the parameter space.
    Best for small parameter spaces or initial screening.

    Example:
        optimizer = GridOptimizer(experiment_spec)
        for params in optimizer.generate_params():
            # params = {"fast_period": 10, "slow_period": 50}
            ...
    """

    def __init__(self, spec: ExperimentSpec):
        """
        Initialize grid optimizer.

        Args:
            spec: Experiment specification with parameter definitions
        """
        self.spec = spec
        self.param_defs = spec.get_parameter_defs()

    def get_total_combinations(self) -> int:
        """Get total number of parameter combinations."""
        total = 1
        for defn in self.param_defs.values():
            total *= len(defn.expand())
        return total

    def generate_params(self) -> Iterator[Dict[str, Any]]:
        """
        Generate all parameter combinations.

        Yields:
            Dictionary of parameter name → value
        """
        if not self.param_defs:
            yield {}
            return

        # Get parameter names and value lists
        names = list(self.param_defs.keys())
        value_lists = [self.param_defs[name].expand() for name in names]

        # Generate Cartesian product
        for values in product(*value_lists):
            yield dict(zip(names, values))

    def generate_params_list(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations as a list."""
        return list(self.generate_params())
