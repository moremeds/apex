"""
Tests for two-stage StrategyObjective (Optuna anti-overfit).

Validates:
- Stage 1 gates: MaxDD cap, min trades, positive Sharpe, min exposure
- Stage 2 ranking: composite = Sharpe - turnover_penalty
- Stress results stored as badges (not gates)
- Parameter budget enforcement
- TrendPulse frozen params (cooldown_bars, risk_per_trade_pct)
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
from optuna import create_study
from optuna.trial import Trial

from src.backtest.optimization.strategy_objective import (
    FROZEN_PARAMS,
    BacktestResult,
    StrategyObjective,
)


def _make_result(
    sharpe: float = 1.0,
    total_return: float = 0.10,
    max_drawdown: float = -0.15,
    trade_count: int = 30,
    exposure_pct: float = 0.50,
    total_cost: float = 0.0,
    win_rate: float = 0.55,
) -> BacktestResult:
    """Create a BacktestResult with given metrics."""
    return BacktestResult(
        sharpe=sharpe,
        total_return=total_return,
        max_drawdown=max_drawdown,
        trade_count=trade_count,
        exposure_pct=exposure_pct,
        total_cost=total_cost,
        win_rate=win_rate,
    )


class TestStage1Gates:
    """Test Stage 1 non-negotiable gates (prune on violation)."""

    def test_max_drawdown_gate(self) -> None:
        """Prune trials with MaxDD exceeding cap."""
        result = _make_result(max_drawdown=-0.35)  # Exceeds -30% cap
        objective = StrategyObjective(
            strategy_name="trend_pulse",
            run_fn=lambda _: result,
        )
        study = create_study(direction="maximize")
        study.optimize(objective, n_trials=1)
        assert study.trials[0].state.name == "PRUNED"

    def test_max_drawdown_at_cap_passes(self) -> None:
        """MaxDD exactly at cap should pass."""
        result = _make_result(max_drawdown=-0.30)  # Exactly at cap
        objective = StrategyObjective(
            strategy_name="trend_pulse",
            run_fn=lambda _: result,
        )
        study = create_study(direction="maximize")
        study.optimize(objective, n_trials=1)
        assert study.trials[0].state.name == "COMPLETE"

    def test_min_trades_gate(self) -> None:
        """Prune trials with too few trades."""
        result = _make_result(trade_count=10)  # Below 20 min
        objective = StrategyObjective(
            strategy_name="trend_pulse",
            run_fn=lambda _: result,
        )
        study = create_study(direction="maximize")
        study.optimize(objective, n_trials=1)
        assert study.trials[0].state.name == "PRUNED"

    def test_min_trades_at_threshold_passes(self) -> None:
        """Exactly at min trades should pass."""
        result = _make_result(trade_count=20)
        objective = StrategyObjective(
            strategy_name="trend_pulse",
            run_fn=lambda _: result,
        )
        study = create_study(direction="maximize")
        study.optimize(objective, n_trials=1)
        assert study.trials[0].state.name == "COMPLETE"

    def test_negative_sharpe_gate(self) -> None:
        """Prune trials with negative Sharpe."""
        result = _make_result(sharpe=-0.5)
        objective = StrategyObjective(
            strategy_name="trend_pulse",
            run_fn=lambda _: result,
        )
        study = create_study(direction="maximize")
        study.optimize(objective, n_trials=1)
        assert study.trials[0].state.name == "PRUNED"

    def test_zero_sharpe_passes(self) -> None:
        """Zero Sharpe should pass (gate is < 0, not <= 0)."""
        result = _make_result(sharpe=0.0)
        objective = StrategyObjective(
            strategy_name="trend_pulse",
            run_fn=lambda _: result,
        )
        study = create_study(direction="maximize")
        study.optimize(objective, n_trials=1)
        assert study.trials[0].state.name == "COMPLETE"

    def test_min_exposure_gate(self) -> None:
        """Prune trials with insufficient exposure (win by being flat)."""
        result = _make_result(exposure_pct=0.10)  # 10% < 30% min
        objective = StrategyObjective(
            strategy_name="trend_pulse",
            run_fn=lambda _: result,
        )
        study = create_study(direction="maximize")
        study.optimize(objective, n_trials=1)
        assert study.trials[0].state.name == "PRUNED"

    def test_exposure_at_threshold_passes(self) -> None:
        """Exactly at min exposure should pass."""
        result = _make_result(exposure_pct=0.30)
        objective = StrategyObjective(
            strategy_name="trend_pulse",
            run_fn=lambda _: result,
        )
        study = create_study(direction="maximize")
        study.optimize(objective, n_trials=1)
        assert study.trials[0].state.name == "COMPLETE"

    def test_custom_gates(self) -> None:
        """Custom gate thresholds are respected."""
        result = _make_result(max_drawdown=-0.22, trade_count=8, exposure_pct=0.15)
        objective = StrategyObjective(
            strategy_name="trend_pulse",
            run_fn=lambda _: result,
            max_drawdown_cap=-0.25,  # Looser cap
            min_trades=5,
            min_exposure_pct=0.10,
        )
        study = create_study(direction="maximize")
        study.optimize(objective, n_trials=1)
        assert study.trials[0].state.name == "COMPLETE"


class TestStage2Ranking:
    """Test Stage 2 survivor ranking."""

    def test_composite_score(self) -> None:
        """Score = Sharpe - turnover_penalty."""
        result = _make_result(sharpe=1.5, total_cost=2.0)
        objective = StrategyObjective(
            strategy_name="trend_pulse",
            run_fn=lambda _: result,
            turnover_penalty_weight=0.1,
        )
        study = create_study(direction="maximize")
        study.optimize(objective, n_trials=1)
        trial = study.trials[0]
        expected_score = 1.5 - (2.0 * 0.1)  # 1.3
        assert abs(trial.value - expected_score) < 0.01

    def test_stress_as_badge_not_gate(self) -> None:
        """Stress results are stored as user_attrs, not used for pruning."""
        result = _make_result(sharpe=0.8)
        # Even if stress would fail, trial should COMPLETE
        objective = StrategyObjective(
            strategy_name="trend_pulse",
            run_fn=lambda _: result,
        )
        study = create_study(direction="maximize")
        study.optimize(objective, n_trials=1)
        trial = study.trials[0]
        assert trial.state.name == "COMPLETE"
        # Stress result should be stored (or None if no equity curve)

    def test_user_attrs_stored(self) -> None:
        """User attributes are stored for dashboard display."""
        result = _make_result(
            sharpe=1.2,
            total_return=0.15,
            max_drawdown=-0.10,
            trade_count=40,
            win_rate=0.60,
            exposure_pct=0.45,
        )
        objective = StrategyObjective(
            strategy_name="trend_pulse",
            run_fn=lambda _: result,
        )
        study = create_study(direction="maximize")
        study.optimize(objective, n_trials=1)
        trial = study.trials[0]

        assert trial.user_attrs["total_return"] == 0.15
        assert trial.user_attrs["max_drawdown"] == -0.10
        assert trial.user_attrs["trade_count"] == 40
        assert trial.user_attrs["win_rate"] == 0.60
        assert abs(trial.user_attrs["exposure_pct"] - 0.45) < 0.01

    def test_higher_sharpe_wins(self) -> None:
        """Between two passing strategies, higher Sharpe wins."""
        results = [
            _make_result(sharpe=0.5),
            _make_result(sharpe=1.5),
        ]
        call_count = [0]

        def run_fn(_: Dict[str, Any]) -> BacktestResult:
            idx = call_count[0] % 2
            call_count[0] += 1
            return results[idx]

        objective = StrategyObjective(
            strategy_name="trend_pulse",
            run_fn=run_fn,
        )
        study = create_study(direction="maximize")
        study.optimize(objective, n_trials=2)
        assert study.best_value > 1.0  # The 1.5 Sharpe trial should win


class TestParameterSpaces:
    """Test parameter suggestion for different strategies."""

    def test_trend_pulse_params(self) -> None:
        """TrendPulse suggests correct parameter space."""
        result = _make_result()
        objective = StrategyObjective(
            strategy_name="trend_pulse",
            run_fn=lambda _: result,
        )
        study = create_study(direction="maximize")
        study.optimize(objective, n_trials=1)
        trial = study.trials[0]
        assert "atr_stop_mult" in trial.params
        assert "min_confidence" in trial.params
        assert "zig_threshold_pct" in trial.params
        assert "trend_strength_moderate" in trial.params
        assert "hard_stop_pct" in trial.params
        assert "exit_bearish_bars" in trial.params
        assert "adx_entry_min" in trial.params
        # cooldown_bars is frozen (structural)
        assert "cooldown_bars" not in trial.params
        # risk_per_trade_pct is frozen (structural)
        assert "risk_per_trade_pct" not in trial.params

    def test_trend_pulse_has_7_params(self) -> None:
        """TrendPulse should have exactly 7 optimizable params."""
        result = _make_result()
        objective = StrategyObjective(
            strategy_name="trend_pulse",
            run_fn=lambda _: result,
        )
        study = create_study(direction="maximize")
        study.optimize(objective, n_trials=1)
        trial = study.trials[0]
        assert len(trial.params) == 7

    def test_unknown_strategy_empty_params(self) -> None:
        """Unknown strategy returns empty params (uses YAML defaults)."""
        result = _make_result()
        objective = StrategyObjective(
            strategy_name="unknown_strategy",
            run_fn=lambda _: result,
        )
        study = create_study(direction="maximize")
        study.optimize(objective, n_trials=1)
        trial = study.trials[0]
        assert len(trial.params) == 0


class TestBacktestResult:
    """Test BacktestResult dataclass."""

    def test_calmar_ratio(self) -> None:
        """Calmar = |total_return / max_drawdown|."""
        result = BacktestResult(total_return=0.30, max_drawdown=-0.10)
        assert abs(result.calmar - 3.0) < 0.01

    def test_calmar_zero_drawdown(self) -> None:
        """Calmar is 0 when no drawdown."""
        result = BacktestResult(total_return=0.10, max_drawdown=0.0)
        assert result.calmar == 0.0

    def test_exposure_pct_field(self) -> None:
        """Exposure percentage field exists and defaults to 0."""
        result = BacktestResult()
        assert result.exposure_pct == 0.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_backtest_error_pruned(self) -> None:
        """Backtest exceptions are caught and trial is pruned."""

        def failing_run(_: Dict[str, Any]) -> BacktestResult:
            raise RuntimeError("Backtest crashed")

        objective = StrategyObjective(
            strategy_name="trend_pulse",
            run_fn=failing_run,
        )
        study = create_study(direction="maximize")
        study.optimize(objective, n_trials=1)
        assert study.trials[0].state.name == "PRUNED"

    def test_all_gates_fail_simultaneously(self) -> None:
        """When multiple gates fail, trial is still pruned (first check wins)."""
        result = _make_result(
            sharpe=-1.0,
            max_drawdown=-0.50,
            trade_count=5,
            exposure_pct=0.05,
        )
        objective = StrategyObjective(
            strategy_name="trend_pulse",
            run_fn=lambda _: result,
        )
        study = create_study(direction="maximize")
        study.optimize(objective, n_trials=1)
        assert study.trials[0].state.name == "PRUNED"


class TestTier2ParamSpaces:
    """Test Tier 2 strategy param spaces (RegimeFlex, SectorPulse)."""

    def test_regime_flex_params(self) -> None:
        """RegimeFlex suggests 4 params matched to constructor."""
        result = _make_result()
        objective = StrategyObjective(
            strategy_name="regime_flex",
            run_fn=lambda _: result,
        )
        study = create_study(direction="maximize")
        study.optimize(objective, n_trials=1)
        trial = study.trials[0]
        assert "r0_gross_pct" in trial.params
        assert "r1_gross_pct" in trial.params
        assert "r3_gross_pct" in trial.params
        assert "ramp_bars" in trial.params
        assert len(trial.params) == 4

    def test_sector_pulse_params(self) -> None:
        """SectorPulse suggests 5 params (no frozen params after dead-code removal)."""
        result = _make_result()
        objective = StrategyObjective(
            strategy_name="sector_pulse",
            run_fn=lambda _: result,
        )
        study = create_study(direction="maximize")
        study.optimize(objective, n_trials=1)
        trial = study.trials[0]
        assert "top_n_sectors" in trial.params
        assert "confidence_threshold" in trial.params
        assert "drift_threshold_pct" in trial.params
        assert "max_turnover_pct" in trial.params
        assert "risk_per_sector_pct" in trial.params
        assert len(trial.params) == 5

    def test_regime_flex_param_ranges(self) -> None:
        """RegimeFlex params are within expected ranges."""
        result = _make_result()
        objective = StrategyObjective(
            strategy_name="regime_flex",
            run_fn=lambda _: result,
        )
        study = create_study(direction="maximize")
        study.optimize(objective, n_trials=1)
        trial = study.trials[0]
        assert 0.7 <= trial.params["r0_gross_pct"] <= 1.0
        assert 0.2 <= trial.params["r1_gross_pct"] <= 0.7
        assert 0.1 <= trial.params["r3_gross_pct"] <= 0.5
        assert 3 <= trial.params["ramp_bars"] <= 20


class TestConstructorParamValidation:
    """Test that Optuna param names are validated against strategy constructors."""

    def test_valid_params_pass(self) -> None:
        """Params matching constructor should pass validation."""
        from src.domain.strategy.playbook.trend_pulse import TrendPulseStrategy

        result = _make_result()
        objective = StrategyObjective(
            strategy_name="trend_pulse",
            run_fn=lambda _: result,
            strategy_class=TrendPulseStrategy,
        )
        study = create_study(direction="maximize")
        study.optimize(objective, n_trials=1)
        assert study.trials[0].state.name == "COMPLETE"

    def test_invalid_params_raise(self) -> None:
        """Params NOT in constructor should raise ValueError."""
        from src.domain.strategy.playbook.trend_pulse import TrendPulseStrategy

        # Create a custom objective that suggests a param not in the constructor
        class BadObjective(StrategyObjective):
            def _build_param_space(self, trial: Trial) -> Dict[str, Any]:
                return {"nonexistent_param": trial.suggest_float("nonexistent_param", 0, 1)}

        result = _make_result()
        objective = BadObjective(
            strategy_name="trend_pulse",
            run_fn=lambda _: result,
            strategy_class=TrendPulseStrategy,
        )
        study = create_study(direction="maximize")
        # Should raise ValueError (converted to TrialPruned by Optuna's error handling)
        with pytest.raises(ValueError, match="not in trend_pulse constructor"):
            objective(study.ask())

    def test_no_class_skips_validation(self) -> None:
        """Without strategy_class, validation is skipped."""
        result = _make_result()
        objective = StrategyObjective(
            strategy_name="trend_pulse",
            run_fn=lambda _: result,
            # No strategy_class provided
        )
        study = create_study(direction="maximize")
        study.optimize(objective, n_trials=1)
        assert study.trials[0].state.name == "COMPLETE"


class TestFrozenParams:
    """Test frozen params registry."""

    def test_sector_pulse_frozen(self) -> None:
        """SectorPulse has no frozen params (dead params removed from constructor)."""
        assert FROZEN_PARAMS["sector_pulse"] == set()

    def test_trend_pulse_frozen(self) -> None:
        """TrendPulse has 2 frozen params."""
        assert FROZEN_PARAMS["trend_pulse"] == {"risk_per_trade_pct", "cooldown_bars"}
