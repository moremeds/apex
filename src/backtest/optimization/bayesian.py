"""
Bayesian optimization using Optuna.

Uses TPE (Tree-structured Parzen Estimator) for efficient
parameter search in large spaces.

Supports batched suggestions for parallel trial execution.
"""

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

from ..core import ExperimentSpec, ParameterDef


class BayesianOptimizer:
    """
    Bayesian optimization using Optuna with batched parallel support.

    Uses TPE sampler for efficient exploration of large parameter spaces.
    Supports batched suggestions for parallel trial execution.

    Now the DEFAULT optimization method - automatically selected when
    spec.optimization.method is 'bayesian' or not specified.

    Example (sequential):
        optimizer = BayesianOptimizer(experiment_spec, n_trials=100)
        for params in optimizer.generate_params():
            result = run_trial(params)
            optimizer.report_result(params, result.trial_score)

    Example (batched parallel):
        optimizer = BayesianOptimizer(experiment_spec, n_trials=100, batch_size=10)
        for batch in optimizer.generate_batches():
            # batch = [(trial_obj, params), ...]
            results = parallel_run(batch)
            optimizer.report_batch_results(results)
    """

    def __init__(
        self,
        spec: ExperimentSpec,
        n_trials: Optional[int] = None,
        seed: int = 42,
        batch_size: int = 1,
    ):
        """
        Initialize Bayesian optimizer.

        Args:
            spec: Experiment specification
            n_trials: Maximum number of trials (default from spec)
            seed: Random seed
            batch_size: Number of trials to suggest at once (1 = sequential)
        """
        self.spec = spec
        self.param_defs = spec.get_parameter_defs()
        self.n_trials = n_trials or spec.optimization.n_trials or 100
        self.seed = seed
        self.batch_size = max(1, batch_size)

        # Create Optuna study
        self.sampler = TPESampler(seed=seed, n_startup_trials=max(10, batch_size))
        self.pruner = HyperbandPruner()

        direction = "maximize" if spec.optimization.direction == "maximize" else "minimize"

        self.study = optuna.create_study(
            direction=direction,
            sampler=self.sampler,
            pruner=self.pruner,
        )

        self._trial_count = 0
        self._current_trial: Optional[optuna.Trial] = None
        self._pending_trials: Dict[int, optuna.Trial] = {}  # trial_number -> trial

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest parameters for a single trial."""
        params = {}
        for name, defn in self.param_defs.items():
            if defn.type == "range":
                # Use int if step is 1 and bounds are integers
                if defn.step == 1 and defn.min == int(defn.min) and defn.max == int(defn.max):
                    params[name] = trial.suggest_int(name, int(defn.min), int(defn.max))
                else:
                    params[name] = trial.suggest_float(name, defn.min, defn.max, step=defn.step)
            elif defn.type == "categorical":
                params[name] = trial.suggest_categorical(name, defn.values)
            elif defn.type == "fixed":
                params[name] = defn.value
        return params

    def generate_params(self) -> Iterator[Dict[str, Any]]:
        """
        Generate parameter suggestions using Bayesian optimization (sequential).

        Yields parameter dictionaries. After each iteration, call
        report_result() to update the optimizer.

        Yields:
            Dictionary of parameter name → value
        """
        while self._trial_count < self.n_trials:
            trial = self.study.ask()
            self._current_trial = trial
            params = self._suggest_params(trial)
            self._trial_count += 1
            yield params

    def generate_batches(self) -> Iterator[List[Tuple[int, Dict[str, Any]]]]:
        """
        Generate batches of parameter suggestions for parallel execution.

        Each batch contains up to batch_size trials that can be run in parallel.
        After running the batch, call report_batch_results() with the scores.

        Yields:
            List of (trial_number, params) tuples
        """
        while self._trial_count < self.n_trials:
            batch: List[Tuple[int, Dict[str, Any]]] = []
            remaining = self.n_trials - self._trial_count
            batch_count = min(self.batch_size, remaining)

            for _ in range(batch_count):
                trial = self.study.ask()
                params = self._suggest_params(trial)
                self._pending_trials[trial.number] = trial
                batch.append((trial.number, params))
                self._trial_count += 1

            yield batch

    def report_result(
        self,
        params: Dict[str, Any],
        score: float,
        pruned: bool = False,
    ) -> None:
        """
        Report trial result to update the optimizer (sequential mode).

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

    def report_batch_results(
        self,
        results: List[Tuple[int, float]],
    ) -> None:
        """
        Report batch results to update the optimizer (parallel mode).

        Args:
            results: List of (trial_number, score) tuples
        """
        for trial_number, score in results:
            if trial_number in self._pending_trials:
                trial = self._pending_trials.pop(trial_number)
                self.study.tell(trial, score)

    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters found so far."""
        return self.study.best_params

    def get_best_score(self) -> float:
        """Get best score found so far."""
        return self.study.best_value

    def get_trials_dataframe(self):
        """Get DataFrame of all trials."""
        return self.study.trials_dataframe()
