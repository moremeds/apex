"""
Bayesian optimization using Optuna.

Uses TPE (Tree-structured Parzen Estimator) for efficient
parameter search in large spaces.
"""

from typing import Any, Callable, Dict, Iterator, Optional

from ..core import ExperimentSpec, ParameterDef

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import HyperbandPruner

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class BayesianOptimizer:
    """
    Bayesian optimization using Optuna.

    Uses TPE sampler for efficient exploration of large parameter spaces.
    Supports pruning of unpromising trials.

    Example:
        optimizer = BayesianOptimizer(experiment_spec, n_trials=100)
        for params in optimizer.generate_params():
            # params = {"fast_period": 12, "slow_period": 48}
            result = run_trial(params)
            optimizer.report_result(params, result.trial_score)
    """

    def __init__(
        self,
        spec: ExperimentSpec,
        n_trials: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Initialize Bayesian optimizer.

        Args:
            spec: Experiment specification
            n_trials: Maximum number of trials (default from spec)
            seed: Random seed
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "optuna is required for Bayesian optimization. "
                "Install with: pip install optuna"
            )

        self.spec = spec
        self.param_defs = spec.get_parameter_defs()
        self.n_trials = n_trials or spec.optimization.n_trials or 100
        self.seed = seed

        # Create Optuna study
        self.sampler = TPESampler(seed=seed)
        self.pruner = HyperbandPruner()

        direction = (
            "maximize"
            if spec.optimization.direction == "maximize"
            else "minimize"
        )

        self.study = optuna.create_study(
            direction=direction,
            sampler=self.sampler,
            pruner=self.pruner,
        )

        self._trial_count = 0
        self._current_trial: Optional[optuna.Trial] = None

    def generate_params(self) -> Iterator[Dict[str, Any]]:
        """
        Generate parameter suggestions using Bayesian optimization.

        Yields parameter dictionaries. After each iteration, call
        report_result() to update the optimizer.

        Yields:
            Dictionary of parameter name → value
        """
        while self._trial_count < self.n_trials:
            trial = self.study.ask()
            self._current_trial = trial

            params = {}
            for name, defn in self.param_defs.items():
                if defn.type == "range":
                    # Use int if step is 1 and bounds are integers
                    if (
                        defn.step == 1
                        and defn.min == int(defn.min)
                        and defn.max == int(defn.max)
                    ):
                        params[name] = trial.suggest_int(
                            name, int(defn.min), int(defn.max)
                        )
                    else:
                        params[name] = trial.suggest_float(
                            name, defn.min, defn.max, step=defn.step
                        )
                elif defn.type == "categorical":
                    params[name] = trial.suggest_categorical(name, defn.values)
                elif defn.type == "fixed":
                    params[name] = defn.value

            self._trial_count += 1
            yield params

    def report_result(
        self,
        params: Dict[str, Any],
        score: float,
        pruned: bool = False,
    ) -> None:
        """
        Report trial result to update the optimizer.

        Args:
            params: Parameters that were tested
            score: Resulting score
            pruned: Whether trial was pruned early
        """
        if self._current_trial is not None:
            if pruned:
                self.study.tell(self._current_trial, state=optuna.trial.TrialState.PRUNED)
            else:
                self.study.tell(self._current_trial, score)
            self._current_trial = None

    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters found so far."""
        return self.study.best_params

    def get_best_score(self) -> float:
        """Get best score found so far."""
        return self.study.best_value

    def get_trials_dataframe(self):
        """Get DataFrame of all trials."""
        return self.study.trials_dataframe()
