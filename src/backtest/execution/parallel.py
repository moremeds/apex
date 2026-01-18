"""
Parallel execution engine using ProcessPoolExecutor.

Scales backtesting across multiple CPU cores with progress tracking,
graceful shutdown, and automatic retry for transient failures.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import random
import time
from concurrent.futures import (
    BrokenExecutor,
    Future,
    ProcessPoolExecutor,
)
from concurrent.futures import TimeoutError as FuturesTimeoutError
from concurrent.futures import (
    as_completed,
)
from dataclasses import dataclass, field
from threading import Event, Thread
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type

from ..core import RunResult, RunSpec, RunStatus

logger = logging.getLogger(__name__)

# Error types that are considered transient and should be retried
# These are typically resource-related or timing issues
TRANSIENT_ERROR_TYPES: Tuple[Type[BaseException], ...] = (
    TimeoutError,
    FuturesTimeoutError,
    ConnectionError,
    BrokenExecutor,
    OSError,  # Includes "Too many open files", "Resource temporarily unavailable"
)

# Error messages that indicate transient failures (checked in exception message)
TRANSIENT_ERROR_PATTERNS: Tuple[str, ...] = (
    "resource temporarily unavailable",
    "too many open files",
    "connection reset",
    "broken pipe",
    "temporary failure",
    "timed out",
    "memory allocation failed",  # Sometimes transient under load
)

# Error types that should NEVER be retried (deterministic failures)
DETERMINISTIC_ERROR_TYPES: Tuple[Type[BaseException], ...] = (
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    AssertionError,
    NotImplementedError,
    ZeroDivisionError,
)


def is_transient_error(error: BaseException) -> bool:
    """
    Determine if an error is transient and should be retried.

    Transient errors are typically:
    - Resource exhaustion (memory, file descriptors)
    - Timeout issues
    - Connection problems
    - Temporary system load issues

    Args:
        error: The exception to classify

    Returns:
        True if the error is transient and retryable
    """
    # Check if it's a deterministic error type (never retry)
    if isinstance(error, DETERMINISTIC_ERROR_TYPES):
        return False

    # Check if it's a known transient error type
    if isinstance(error, TRANSIENT_ERROR_TYPES):
        return True

    # Check error message for transient patterns
    error_msg = str(error).lower()
    for pattern in TRANSIENT_ERROR_PATTERNS:
        if pattern in error_msg:
            return True

    return False


@dataclass
class ParallelConfig:
    """Configuration for parallel execution."""

    max_workers: int = field(default_factory=lambda: max(1, mp.cpu_count() - 1))
    chunk_size: int = 10  # Runs to batch per worker
    timeout_per_run: int = 300  # Seconds
    max_retries: int = 2  # Maximum retry attempts for transient failures
    retry_base_delay: float = 1.0  # Base delay in seconds for exponential backoff
    retry_max_delay: float = 30.0  # Maximum delay between retries
    retry_jitter: float = 0.1  # Jitter factor (0.1 = ±10% randomness)
    graceful_shutdown_timeout: int = 30

    # Multiprocessing worker initialization (for pickling workaround)
    initializer: Optional[Callable[..., None]] = None  # Worker init function
    initargs: Tuple[Any, ...] = ()  # Arguments for initializer
    mp_context: Optional[str] = None  # "spawn", "fork", or "forkserver"

    def calculate_retry_delay(self, attempt: int) -> float:
        """
        Calculate delay before retry using exponential backoff with jitter.

        Args:
            attempt: The retry attempt number (1-indexed)

        Returns:
            Delay in seconds before the next retry
        """
        # Exponential backoff: base_delay * 2^(attempt-1)
        delay = self.retry_base_delay * (2 ** (attempt - 1))

        # Cap at max delay
        delay = min(delay, self.retry_max_delay)

        # Add jitter: ±jitter_factor of the delay
        jitter_range = delay * self.retry_jitter
        delay += random.uniform(-jitter_range, jitter_range)

        return float(max(0, delay))


@dataclass
class ExecutionProgress:
    """Progress tracking for parallel execution."""

    total_runs: int = 0
    completed_runs: int = 0
    failed_runs: int = 0
    skipped_runs: int = 0
    retried_runs: int = 0  # Runs that succeeded after retry
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def runs_per_second(self) -> float:
        if self.elapsed_seconds == 0:
            return 0.0
        return self.completed_runs / self.elapsed_seconds

    @property
    def eta_seconds(self) -> float:
        if self.runs_per_second == 0:
            return float("inf")
        remaining = self.total_runs - self.completed_runs - self.failed_runs - self.skipped_runs
        return remaining / self.runs_per_second

    @property
    def completion_pct(self) -> float:
        if self.total_runs == 0:
            return 100.0
        return (self.completed_runs + self.failed_runs + self.skipped_runs) / self.total_runs * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total_runs,
            "completed": self.completed_runs,
            "failed": self.failed_runs,
            "skipped": self.skipped_runs,
            "retried": self.retried_runs,
            "elapsed_s": round(self.elapsed_seconds, 1),
            "runs_per_s": round(self.runs_per_second, 2),
            "eta_s": round(self.eta_seconds, 1) if self.eta_seconds != float("inf") else None,
            "pct": round(self.completion_pct, 1),
        }


class ParallelRunner:
    """
    Parallel execution engine using ProcessPoolExecutor.

    Features:
    - Configurable worker count
    - Progress tracking with ETA
    - Graceful shutdown
    - Automatic retry for transient failures
    - Result streaming
    - Persistent executor mode (reuse across multiple run_all calls)

    Example (one-shot):
        runner = ParallelRunner(ParallelConfig(max_workers=4))
        results = runner.run_all(run_specs, backtest_fn)

    Example (persistent - reuse across trials):
        runner = ParallelRunner(ParallelConfig(max_workers=4), persistent=True)
        runner.start()  # Create executor once
        for trial in trials:
            results = runner.run_all(run_specs, backtest_fn)
        runner.stop()   # Cleanup
    """

    def __init__(self, config: Optional[ParallelConfig] = None, persistent: bool = False):
        self.config = config or ParallelConfig()
        self.persistent = persistent
        self.progress = ExecutionProgress()
        self._shutdown_event = Event()
        self._executor: Optional[ProcessPoolExecutor] = None
        self._executor_kwargs: Dict[str, Any] = {}
        self._started = False

    def start(self, execute_fn: Optional[Callable[[RunSpec], RunResult]] = None) -> None:
        """
        Start the executor in persistent mode (call once, reuse across run_all calls).

        Args:
            execute_fn: Optional function to extract initializer settings from
        """
        if self._started:
            return

        # Build executor kwargs
        self._executor_kwargs = {"max_workers": self.config.max_workers}

        if execute_fn:
            init_fn = getattr(execute_fn, "_mp_initializer", None) or self.config.initializer
            init_args = getattr(execute_fn, "_mp_initargs", None) or self.config.initargs
            mp_context_name = getattr(execute_fn, "_mp_context", None) or self.config.mp_context

            if init_fn is not None:
                self._executor_kwargs["initializer"] = init_fn
                self._executor_kwargs["initargs"] = init_args or ()
            if mp_context_name:
                self._executor_kwargs["mp_context"] = mp.get_context(mp_context_name)

        self._executor = ProcessPoolExecutor(**self._executor_kwargs)
        self._started = True
        logger.debug(f"Started persistent executor with {self.config.max_workers} workers")

    def stop(self) -> None:
        """Stop the persistent executor."""
        if self._executor and self._started:
            self._executor.shutdown(wait=True)
            self._executor = None
            self._started = False
            logger.debug("Stopped persistent executor")

    def run_all(
        self,
        run_specs: List[RunSpec],
        execute_fn: Callable[[RunSpec], RunResult],
        on_result: Optional[Callable[[RunResult], None]] = None,
    ) -> List[RunResult]:
        """
        Execute all runs in parallel with automatic retry for transient failures.

        In persistent mode, reuses the executor created by start().
        In one-shot mode, creates and destroys executor per call.

        Args:
            run_specs: List of run specifications
            execute_fn: Function to execute a single run
            on_result: Optional callback for each result

        Returns:
            List of all results
        """
        self.progress = ExecutionProgress(total_runs=len(run_specs))
        results: List[RunResult] = []
        retry_queue: List[Tuple[RunSpec, int, BaseException]] = []

        logger.debug(
            f"Parallel execution: {len(run_specs)} runs, " f"{self.config.max_workers} workers"
        )

        # Persistent mode: use existing executor
        if self.persistent and self._started and self._executor:
            return self._run_with_executor(
                self._executor, run_specs, execute_fn, on_result, results, retry_queue
            )

        # One-shot mode: create executor just for this call
        init_fn = getattr(execute_fn, "_mp_initializer", None) or self.config.initializer
        init_args = getattr(execute_fn, "_mp_initargs", None) or self.config.initargs
        mp_context_name = getattr(execute_fn, "_mp_context", None) or self.config.mp_context

        executor_kwargs: Dict[str, Any] = {"max_workers": self.config.max_workers}
        if init_fn is not None:
            executor_kwargs["initializer"] = init_fn
            executor_kwargs["initargs"] = init_args or ()
        if mp_context_name:
            executor_kwargs["mp_context"] = mp.get_context(mp_context_name)

        with ProcessPoolExecutor(**executor_kwargs) as executor:
            self._executor = executor
            return self._run_with_executor(
                executor, run_specs, execute_fn, on_result, results, retry_queue
            )

    def _run_with_executor(
        self,
        executor: ProcessPoolExecutor,
        run_specs: List[RunSpec],
        execute_fn: Callable[[RunSpec], RunResult],
        on_result: Optional[Callable[[RunResult], None]],
        results: List[RunResult],
        retry_queue: List[Tuple[RunSpec, int, BaseException]],
    ) -> List[RunResult]:
        """Execute runs using the provided executor."""
        # CRIT-003: Track (spec, attempt) pairs to preserve retry count
        pending_specs: List[Tuple[RunSpec, int]] = [(spec, 0) for spec in run_specs]

        while pending_specs or retry_queue:
            if self._shutdown_event.is_set():
                break

            # Process retry queue first - apply backoff delay
            if retry_queue and not pending_specs:
                spec, attempt, last_error = retry_queue.pop(0)
                delay = self.config.calculate_retry_delay(attempt)
                logger.info(
                    f"Retrying {spec.run_id} (attempt {attempt}/{self.config.max_retries}) "
                    f"after {delay:.1f}s delay"
                )
                time.sleep(delay)
                # CRIT-003: Preserve attempt count from retry_queue
                pending_specs = [(spec, attempt)]

            # Submit runs
            future_to_spec: Dict[Future, Tuple[RunSpec, int]] = {}
            for spec, attempt in pending_specs:
                if self._shutdown_event.is_set():
                    break
                # CRIT-003: Use actual attempt count, not always 0
                future = executor.submit(execute_fn, spec)
                future_to_spec[future] = (spec, attempt)

            pending_specs = []

            # Collect results as they complete
            for future in as_completed(future_to_spec):
                if self._shutdown_event.is_set():
                    break

                spec, attempt = future_to_spec[future]
                try:
                    result = future.result(timeout=self.config.timeout_per_run)
                    results.append(result)

                    if result.status == RunStatus.SUCCESS:
                        self.progress.completed_runs += 1
                        if attempt > 0:
                            self.progress.retried_runs += 1
                            logger.info(f"Run {spec.run_id} succeeded after {attempt} retry(ies)")
                    else:
                        self.progress.failed_runs += 1

                    if on_result:
                        on_result(result)

                except Exception as e:
                    # Determine if we should retry
                    next_attempt = attempt + 1
                    should_retry = (
                        is_transient_error(e)
                        and next_attempt <= self.config.max_retries
                        and not self._shutdown_event.is_set()
                    )

                    if should_retry:
                        logger.warning(
                            f"Run {spec.run_id} failed with transient error: {e}. "
                            f"Scheduling retry {next_attempt}/{self.config.max_retries}"
                        )
                        retry_queue.append((spec, next_attempt, e))
                    else:
                        # No more retries - record failure
                        if next_attempt > self.config.max_retries:
                            logger.error(
                                f"Run {spec.run_id} failed after {self.config.max_retries} retries: {e}"
                            )
                        else:
                            logger.error(f"Run {spec.run_id} failed with deterministic error: {e}")

                        self.progress.failed_runs += 1

                        # Create failure result
                        result = RunResult(
                            run_id=spec.run_id or "",
                            trial_id=spec.trial_id,
                            experiment_id=spec.experiment_id or "",
                            symbol=spec.symbol,
                            window_id=spec.window.window_id,
                            profile_version=spec.profile_version,
                            data_version=spec.data_version,
                            status=RunStatus.FAIL_STRATEGY,
                            error=str(e),
                            is_train=spec.window.is_train,
                            is_oos=spec.window.is_oos,
                        )
                        results.append(result)

                        if on_result:
                            on_result(result)

                # Log progress periodically
                processed = self.progress.completed_runs + self.progress.failed_runs
                if processed % 100 == 0:
                    self._log_progress()

        self._log_progress()  # Final progress
        return results

    def run_streaming(
        self, run_specs: Iterator[RunSpec], execute_fn: Callable[[RunSpec], RunResult]
    ) -> Iterator[RunResult]:
        """
        Stream results as they complete.

        Useful for large experiments where you want to process
        results incrementally.
        """
        batch: List[RunSpec] = []
        for spec in run_specs:
            batch.append(spec)

            if len(batch) >= self.config.chunk_size * self.config.max_workers:
                yield from self.run_all(batch, execute_fn)
                batch = []

        if batch:
            yield from self.run_all(batch, execute_fn)

    def shutdown(self, wait: bool = True) -> None:
        """
        Gracefully shutdown the executor.

        Args:
            wait: If True, wait for running tasks to complete
        """
        logger.info("Initiating graceful shutdown...")
        self._shutdown_event.set()

        if self._executor:
            self._executor.shutdown(wait=wait, cancel_futures=not wait)

    def _log_progress(self) -> None:
        """Log current progress (per-trial, use DEBUG to reduce noise)."""
        p = self.progress
        eta_str = f"{p.eta_seconds:.0f}s" if p.eta_seconds != float("inf") else "unknown"
        logger.debug(
            f"Runs: {p.completed_runs}/{p.total_runs} ({p.completion_pct:.1f}%) | "
            f"Failed: {p.failed_runs} | "
            f"Speed: {p.runs_per_second:.1f} runs/s | "
            f"ETA: {eta_str}"
        )


class ProgressMonitor:
    """
    Real-time progress monitoring for long experiments.

    Runs in a separate thread, periodically logs progress
    and can trigger callbacks.

    Example:
        monitor = ProgressMonitor(runner.progress, interval_seconds=10)
        with monitor:
            runner.run_all(specs, fn)
    """

    def __init__(
        self,
        progress: ExecutionProgress,
        interval_seconds: float = 10.0,
        on_update: Optional[Callable[[ExecutionProgress], None]] = None,
    ):
        self.progress = progress
        self.interval = interval_seconds
        self.on_update = on_update
        self._stop_event = Event()
        self._thread: Optional[Thread] = None

    def start(self) -> None:
        """Start monitoring thread."""
        self._stop_event.clear()
        self._thread = Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop monitoring thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            if self.on_update:
                self.on_update(self.progress)

            self._stop_event.wait(self.interval)

    def __enter__(self) -> "ProgressMonitor":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
