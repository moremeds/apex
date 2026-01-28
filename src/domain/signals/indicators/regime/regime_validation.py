"""
Regime Validation Framework.

Phase 5: Quality and usefulness validation with explicit failure criteria.
Tests both classification quality (occupancy, stability) and
practical usefulness (does R2 actually precede drawdowns?).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class QualityMetrics:
    """Quality metrics for regime classification."""

    # Occupancy: % time in each regime
    occupancy: Dict[str, float] = field(default_factory=dict)

    # Mean duration per regime (bars)
    mean_duration: Dict[str, float] = field(default_factory=dict)

    # Transition frequency (transitions / total bars)
    transition_frequency: float = 0.0

    # R1 dominance (% in R1 - should be < 70%)
    r1_dominance: float = 0.0

    # Min/max duration per regime
    min_duration: Dict[str, int] = field(default_factory=dict)
    max_duration: Dict[str, int] = field(default_factory=dict)


@dataclass
class UsefulnessMetrics:
    """Usefulness metrics for regime classification."""

    # Mean return by regime
    conditional_returns: Dict[str, float] = field(default_factory=dict)

    # Mean drawdown by regime
    conditional_drawdown: Dict[str, float] = field(default_factory=dict)

    # Risk-off hit rate: % of R2 periods that precede drawdowns
    risk_off_hit_rate: float = 0.0

    # Whipsaw rate in R1 (false signals)
    whipsaw_rate_r1: float = 0.0

    # Return separation (R0 return - R2 return)
    return_separation: float = 0.0


@dataclass
class FailureCriteria:
    """Explicit failure criteria for validation."""

    max_r1_occupancy: float = 0.70  # FAIL if R1 > 70%
    min_risk_off_hit_rate: float = 0.50  # FAIL if R2 hit rate < 50%
    min_regime_duration_bars: int = 5  # FAIL if avg duration < 5 bars
    max_transition_frequency: float = 0.10  # FAIL if transitions > 10%
    min_return_separation: float = 0.001  # FAIL if R0 return not > R2


@dataclass
class ValidationResult:
    """Result of regime validation."""

    passed: bool
    quality: QualityMetrics
    usefulness: UsefulnessMetrics
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class RegimeQualityValidator:
    """Validate regime classification quality."""

    def compute_metrics(self, regimes: pd.Series) -> QualityMetrics:
        """
        Compute quality metrics for regime series.

        Args:
            regimes: Series of regime labels (R0, R1, R2)

        Returns:
            QualityMetrics with occupancy, duration, transitions
        """
        regimes = regimes.dropna()
        if len(regimes) == 0:
            return QualityMetrics()

        # Occupancy
        occupancy = regimes.value_counts(normalize=True).to_dict()

        # Duration analysis per regime
        mean_duration = {}
        min_duration = {}
        max_duration = {}

        for regime in regimes.unique():
            mask = regimes == regime
            # Find runs
            runs = self._find_runs(mask)
            if runs:
                mean_duration[regime] = float(np.mean(runs))
                min_duration[regime] = int(min(runs))
                max_duration[regime] = int(max(runs))

        # Transition frequency
        transitions = (regimes != regimes.shift(1)).sum()
        transition_frequency = transitions / len(regimes)

        # R1 dominance
        r1_dominance = occupancy.get("R1", 0.0)

        return QualityMetrics(
            occupancy=occupancy,
            mean_duration=mean_duration,
            transition_frequency=transition_frequency,
            r1_dominance=r1_dominance,
            min_duration=min_duration,
            max_duration=max_duration,
        )

    def _find_runs(self, mask: pd.Series) -> List[int]:
        """Find consecutive run lengths where mask is True."""
        runs = []
        current_run = 0

        for val in mask:
            if val:
                current_run += 1
            elif current_run > 0:
                runs.append(current_run)
                current_run = 0

        if current_run > 0:
            runs.append(current_run)

        return runs


class RegimeUsefulnessValidator:
    """Validate if regimes actually improve strategy decisions."""

    def __init__(self, forward_bars: int = 10) -> None:
        self.forward_bars = forward_bars

    def compute_metrics(
        self,
        regimes: pd.Series,
        returns: pd.Series,
        prices: Optional[pd.Series] = None,
    ) -> UsefulnessMetrics:
        """
        Compute usefulness metrics.

        Args:
            regimes: Series of regime labels
            returns: Series of period returns
            prices: Optional price series for drawdown calculation

        Returns:
            UsefulnessMetrics
        """
        # Align series
        aligned = pd.DataFrame({"regime": regimes, "returns": returns}).dropna()
        if len(aligned) == 0:
            return UsefulnessMetrics()

        # Conditional returns by regime
        conditional_returns = aligned.groupby("regime")["returns"].mean().to_dict()

        # Conditional drawdown (if prices provided)
        conditional_drawdown = {}
        if prices is not None:
            dd = self._compute_drawdowns(prices)
            aligned["drawdown"] = dd
            conditional_drawdown = aligned.groupby("regime")["drawdown"].mean().to_dict()

        # Risk-off hit rate
        risk_off_hit_rate = self._compute_risk_off_hit_rate(
            regimes, returns if prices is None else -dd
        )

        # Whipsaw rate in R1
        whipsaw_rate = self._compute_whipsaw_rate(regimes, returns)

        # Return separation
        r0_ret = conditional_returns.get("R0", 0.0)
        r2_ret = conditional_returns.get("R2", 0.0)
        return_separation = r0_ret - r2_ret

        return UsefulnessMetrics(
            conditional_returns=conditional_returns,
            conditional_drawdown=conditional_drawdown,
            risk_off_hit_rate=risk_off_hit_rate,
            whipsaw_rate_r1=whipsaw_rate,
            return_separation=return_separation,
        )

    def _compute_drawdowns(self, prices: pd.Series) -> pd.Series:
        """Compute drawdown series from prices."""
        rolling_max = prices.expanding().max()
        drawdown = (rolling_max - prices) / rolling_max
        return drawdown

    def _compute_risk_off_hit_rate(
        self,
        regimes: pd.Series,
        returns: pd.Series,
    ) -> float:
        """Compute what % of R2 periods precede negative returns."""
        r2_mask = regimes == "R2"
        if not r2_mask.any():
            return 0.0

        # Forward returns after R2
        forward_ret = returns.shift(-self.forward_bars)
        r2_forward = forward_ret[r2_mask].dropna()

        if len(r2_forward) == 0:
            return 0.0

        # Hit = R2 correctly predicted negative forward period
        hits = (r2_forward < 0).sum()
        return float(hits / len(r2_forward))

    def _compute_whipsaw_rate(
        self,
        regimes: pd.Series,
        returns: pd.Series,
    ) -> float:
        """Compute whipsaw rate in R1 (regime changes that hurt)."""
        r1_mask = regimes == "R1"
        transitions = regimes != regimes.shift(1)

        # Transitions INTO R1
        into_r1 = transitions & r1_mask

        if not into_r1.any():
            return 0.0

        # Forward returns after entering R1
        forward_ret = returns.shift(-self.forward_bars)
        r1_forward = forward_ret[into_r1].dropna()

        if len(r1_forward) == 0:
            return 0.0

        # Whipsaw = entered R1 but forward return was strongly positive (should've stayed R0)
        # or strongly negative (should've gone R2)
        threshold = returns.std() * 1.5
        whipsaws = ((r1_forward > threshold) | (r1_forward < -threshold)).sum()
        return float(whipsaws / len(r1_forward))


class RegimeValidator:
    """
    Combined validator with explicit failure criteria.

    Validates both quality and usefulness with pass/fail gates.
    """

    def __init__(self, criteria: Optional[FailureCriteria] = None) -> None:
        self.criteria = criteria or FailureCriteria()
        self.quality_validator = RegimeQualityValidator()
        self.usefulness_validator = RegimeUsefulnessValidator()

    def validate(
        self,
        regimes: pd.Series,
        returns: pd.Series,
        prices: Optional[pd.Series] = None,
    ) -> ValidationResult:
        """
        Run full validation with pass/fail criteria.

        Args:
            regimes: Series of regime labels
            returns: Series of period returns
            prices: Optional price series

        Returns:
            ValidationResult with pass/fail and metrics
        """
        quality = self.quality_validator.compute_metrics(regimes)
        usefulness = self.usefulness_validator.compute_metrics(regimes, returns, prices)

        failures = []
        warnings = []

        # Check failure criteria
        if quality.r1_dominance > self.criteria.max_r1_occupancy:
            failures.append(
                f"R1 occupancy {quality.r1_dominance:.1%} > {self.criteria.max_r1_occupancy:.1%}"
            )

        if quality.transition_frequency > self.criteria.max_transition_frequency:
            failures.append(
                f"Transition freq {quality.transition_frequency:.1%} > "
                f"{self.criteria.max_transition_frequency:.1%}"
            )

        for regime, duration in quality.mean_duration.items():
            if duration < self.criteria.min_regime_duration_bars:
                failures.append(
                    f"{regime} avg duration {duration:.1f} < "
                    f"{self.criteria.min_regime_duration_bars} bars"
                )

        if usefulness.risk_off_hit_rate < self.criteria.min_risk_off_hit_rate:
            failures.append(
                f"Risk-off hit rate {usefulness.risk_off_hit_rate:.1%} < "
                f"{self.criteria.min_risk_off_hit_rate:.1%}"
            )

        if usefulness.return_separation < self.criteria.min_return_separation:
            warnings.append(
                f"Return separation {usefulness.return_separation:.4f} < "
                f"{self.criteria.min_return_separation:.4f}"
            )

        return ValidationResult(
            passed=len(failures) == 0,
            quality=quality,
            usefulness=usefulness,
            failures=failures,
            warnings=warnings,
        )

    def validate_walk_forward(
        self,
        data: pd.DataFrame,
        regime_classifier: Any,
        train_months: int = 18,
        test_months: int = 3,
    ) -> Dict[str, Any]:
        """
        Walk-forward validation across multiple windows.

        Args:
            data: DataFrame with OHLCV
            regime_classifier: Callable that takes df and returns regime series
            train_months: Training window size
            test_months: Test window size

        Returns:
            Dict with window results and aggregate metrics
        """
        results = []
        all_failures = []

        # Generate windows
        windows = self._generate_windows(data, train_months, test_months)

        for train_start, train_end, test_start, test_end in windows:
            train_data = data.loc[train_start:train_end]
            test_data = data.loc[test_start:test_end]

            # Classify test period (classifier may use train for calibration)
            regimes = regime_classifier(train_data, test_data)

            # Compute returns
            returns = test_data["close"].pct_change()

            # Validate
            result = self.validate(regimes, returns, test_data["close"])

            results.append(
                {
                    "period": (test_start, test_end),
                    "passed": result.passed,
                    "quality": result.quality,
                    "usefulness": result.usefulness,
                    "failures": result.failures,
                }
            )

            all_failures.extend(result.failures)

        # Aggregate
        pass_rate = sum(1 for r in results if r["passed"]) / len(results) if results else 0

        return {
            "windows": results,
            "pass_rate": pass_rate,
            "total_failures": len(all_failures),
            "unique_failures": list(set(all_failures)),
            "passed": len(all_failures) == 0,
        }

    def _generate_windows(
        self,
        data: pd.DataFrame,
        train_months: int,
        test_months: int,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generate train/test window boundaries."""
        windows = []
        total_months = train_months + test_months

        start = data.index[0]
        end = data.index[-1]

        current = start
        while current + pd.DateOffset(months=total_months) <= end:
            train_start = current
            train_end = current + pd.DateOffset(months=train_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=test_months)

            windows.append((train_start, train_end, test_start, test_end))
            current = test_end  # Non-overlapping windows

        return windows
