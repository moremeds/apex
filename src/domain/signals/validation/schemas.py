"""
Output JSON Schemas for Validation Results.

Defines the unified output format for validation runs,
combining horizon config, labeler thresholds, and validation results.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .confirmation import ConfirmationResult
from .earliness import EarlinessResult
from .statistics import StatisticalResult


@dataclass
class HorizonConfig:
    """Horizon configuration for validation output."""

    horizon_calendar_days: int
    bars_per_day: Dict[str, float]
    horizon_bars_by_tf: Dict[str, int]

    def to_dict(self) -> dict:
        return {
            "horizon_calendar_days": self.horizon_calendar_days,
            "bars_per_day": self.bars_per_day,
            "horizon_bars_by_tf": self.horizon_bars_by_tf,
        }


@dataclass
class LabelerThreshold:
    """Labeler thresholds for a single timeframe."""

    version: str
    trending_forward_return_min: float
    trending_sharpe_min: float
    choppy_volatility_min: float
    choppy_drawdown_max: float

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "trending_forward_return_min": self.trending_forward_return_min,
            "trending_sharpe_min": self.trending_sharpe_min,
            "choppy_volatility_min": self.choppy_volatility_min,
            "choppy_drawdown_max": self.choppy_drawdown_max,
        }


@dataclass
class SplitConfig:
    """CV split configuration."""

    outer_folds: int
    inner_folds: int
    purge_bars_by_tf: Dict[str, int]
    embargo_bars_by_tf: Dict[str, int]

    def to_dict(self) -> dict:
        return {
            "outer_folds": self.outer_folds,
            "inner_folds": self.inner_folds,
            "purge_bars_by_tf": self.purge_bars_by_tf,
            "embargo_bars_by_tf": self.embargo_bars_by_tf,
        }


@dataclass
class BarValidation:
    """Bar validation details for a timeframe."""

    requested_bars: int
    loaded_bars: int
    usable_bars: int
    validated_bars: int
    max_lookback_indicator: str
    reasons: List[str]

    def to_dict(self) -> dict:
        return {
            "requested_bars": self.requested_bars,
            "loaded_bars": self.loaded_bars,
            "usable_bars": self.usable_bars,
            "validated_bars": self.validated_bars,
            "max_lookback_indicator": self.max_lookback_indicator,
            "reasons": self.reasons,
        }


@dataclass
class GateResult:
    """Result of a single gate check."""

    gate_name: str
    passed: bool
    value: float
    threshold: float
    message: str

    def to_dict(self) -> dict:
        return {
            "gate_name": self.gate_name,
            "passed": self.passed,
            "value": self.value,
            "threshold": self.threshold,
            "message": self.message,
        }


@dataclass
class ValidationOutput:
    """
    Complete validation output with all required JSON fields.

    This is the top-level structure for validation results.
    """

    # Metadata
    version: str = "m2_v2.0"
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    mode: str = "full"  # "fast", "full", "holdout"

    # Configuration
    horizon_config: Optional[HorizonConfig] = None
    labeler_thresholds_by_tf: Dict[str, LabelerThreshold] = field(default_factory=dict)
    split_config: Optional[SplitConfig] = None

    # Core results
    statistical_result: Optional[StatisticalResult] = None

    # Multi-TF results
    earliness_stats_by_tf_pair: Dict[str, EarlinessResult] = field(default_factory=dict)
    confirmation_result: Optional[ConfirmationResult] = None

    # Bar validation
    bar_validation_by_tf: Dict[str, BarValidation] = field(default_factory=dict)

    # Gate results
    gate_results: List[GateResult] = field(default_factory=list)
    all_gates_passed: bool = False

    # Universe info
    training_symbols: List[str] = field(default_factory=list)
    holdout_symbols: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON output."""
        result: Dict[str, Any] = {
            "version": self.version,
            "generated_at": self.generated_at.isoformat(),
            "mode": self.mode,
        }

        if self.horizon_config:
            result["horizon_config"] = self.horizon_config.to_dict()

        if self.labeler_thresholds_by_tf:
            result["labeler_thresholds_by_tf"] = {
                tf: t.to_dict() for tf, t in self.labeler_thresholds_by_tf.items()
            }

        if self.split_config:
            result["split_config"] = self.split_config.to_dict()

        if self.statistical_result:
            result["statistical_result"] = self.statistical_result.to_dict()

        if self.earliness_stats_by_tf_pair:
            result["earliness_stats_by_tf_pair"] = {
                tf: e.to_dict() for tf, e in self.earliness_stats_by_tf_pair.items()
            }

        if self.confirmation_result:
            result["confirmation_result"] = self.confirmation_result.to_dict()

        if self.bar_validation_by_tf:
            result["bar_validation_by_tf"] = {
                tf: b.to_dict() for tf, b in self.bar_validation_by_tf.items()
            }

        result["gate_results"] = [g.to_dict() for g in self.gate_results]
        result["all_gates_passed"] = self.all_gates_passed

        result["universe"] = {
            "n_training_symbols": len(self.training_symbols),
            "n_holdout_symbols": len(self.holdout_symbols),
            "training_symbols": self.training_symbols,
            "holdout_symbols": self.holdout_symbols,
        }

        return result


def create_fast_validation_output(
    trending_r0_rate: float,
    choppy_r0_rate: float,
    causality_passed: bool,
    symbols: List[str],
) -> ValidationOutput:
    """
    Create validation output for fast (PR) gate.

    Args:
        trending_r0_rate: R0 rate for trending periods
        choppy_r0_rate: R0 rate for choppy periods
        causality_passed: Whether causality test passed
        symbols: Symbols used

    Returns:
        ValidationOutput for fast gate
    """
    gates = [
        GateResult(
            gate_name="trending_r0",
            passed=trending_r0_rate > 0.50,
            value=trending_r0_rate,
            threshold=0.50,
            message=f"Trending R0 rate: {trending_r0_rate:.3f}",
        ),
        GateResult(
            gate_name="choppy_r0",
            passed=choppy_r0_rate < 0.40,
            value=choppy_r0_rate,
            threshold=0.40,
            message=f"Choppy R0 rate: {choppy_r0_rate:.3f}",
        ),
        GateResult(
            gate_name="causality_g7",
            passed=causality_passed,
            value=1.0 if causality_passed else 0.0,
            threshold=1.0,
            message="Causality test: " + ("PASS" if causality_passed else "FAIL"),
        ),
    ]

    all_passed = all(g.passed for g in gates)

    return ValidationOutput(
        mode="fast",
        gate_results=gates,
        all_gates_passed=all_passed,
        training_symbols=symbols,
    )


def create_full_validation_output(
    statistical_result: StatisticalResult,
    earliness_by_tf: Dict[str, EarlinessResult],
    confirmation_result: Optional[ConfirmationResult],
    horizon_config: HorizonConfig,
    split_config: SplitConfig,
    labeler_thresholds: Dict[str, LabelerThreshold],
    training_symbols: List[str],
    holdout_symbols: List[str],
) -> ValidationOutput:
    """
    Create validation output for full (nightly) gate.

    Args:
        statistical_result: Symbol-level statistics
        earliness_by_tf: Earliness results by TF pair
        confirmation_result: Multi-TF confirmation result
        horizon_config: Horizon configuration
        split_config: CV split configuration
        labeler_thresholds: Labeler thresholds by TF
        training_symbols: Training universe
        holdout_symbols: Holdout universe

    Returns:
        ValidationOutput for full gate
    """
    gates = []

    # Statistical gates
    stat_passes, stat_failures = statistical_result.passes_gates()
    gates.append(
        GateResult(
            gate_name="cohens_d",
            passed=statistical_result.effect_size_cohens_d >= 0.8,
            value=statistical_result.effect_size_cohens_d,
            threshold=0.8,
            message=f"Cohen's d: {statistical_result.effect_size_cohens_d:.3f}",
        )
    )
    gates.append(
        GateResult(
            gate_name="p_value",
            passed=statistical_result.p_value < 0.01,
            value=statistical_result.p_value,
            threshold=0.01,
            message=f"p-value: {statistical_result.p_value:.4f}",
        )
    )
    gates.append(
        GateResult(
            gate_name="trending_ci_lower",
            passed=statistical_result.trending_ci_lower >= 0.60,
            value=statistical_result.trending_ci_lower,
            threshold=0.60,
            message=f"Trending CI lower: {statistical_result.trending_ci_lower:.3f}",
        )
    )
    gates.append(
        GateResult(
            gate_name="choppy_ci_upper",
            passed=statistical_result.choppy_ci_upper <= 0.25,
            value=statistical_result.choppy_ci_upper,
            threshold=0.25,
            message=f"Choppy CI upper: {statistical_result.choppy_ci_upper:.3f}",
        )
    )

    # Earliness gates (for 4h vs 1d)
    if "4h_vs_1d" in earliness_by_tf:
        earl = earliness_by_tf["4h_vs_1d"]
        gates.append(
            GateResult(
                gate_name="median_earliness_4h",
                passed=earl.median_earliness_days >= 1.0,
                value=earl.median_earliness_days,
                threshold=1.0,
                message=f"4h earliness: {earl.median_earliness_days:.2f} days",
            )
        )
        gates.append(
            GateResult(
                gate_name="pct_earlier_4h",
                passed=earl.pct_earlier_than_baseline >= 0.60,
                value=earl.pct_earlier_than_baseline,
                threshold=0.60,
                message=f"4h pct earlier: {earl.pct_earlier_than_baseline:.1%}",
            )
        )

    # Confirmation gates
    if confirmation_result:
        gates.append(
            GateResult(
                gate_name="fp_rate_reduction",
                passed=-confirmation_result.delta_fp_rate >= 0.05,
                value=-confirmation_result.delta_fp_rate,
                threshold=0.05,
                message=f"FP reduction: {-confirmation_result.delta_fp_rate:.1%}",
            )
        )
        gates.append(
            GateResult(
                gate_name="precision_drop",
                passed=-confirmation_result.delta_precision <= 0.02,
                value=-confirmation_result.delta_precision,
                threshold=0.02,
                message=f"Precision drop: {-confirmation_result.delta_precision:.1%}",
            )
        )

    all_passed = all(g.passed for g in gates)

    return ValidationOutput(
        mode="full",
        horizon_config=horizon_config,
        labeler_thresholds_by_tf=labeler_thresholds,
        split_config=split_config,
        statistical_result=statistical_result,
        earliness_stats_by_tf_pair=earliness_by_tf,
        confirmation_result=confirmation_result,
        gate_results=gates,
        all_gates_passed=all_passed,
        training_symbols=training_symbols,
        holdout_symbols=holdout_symbols,
    )
