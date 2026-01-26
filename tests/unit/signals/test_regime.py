"""
Tests for Regime Detection and Reporting.

Covers:
1. Unit tests for component state classifications
2. Snapshot test for RegimeOutput serialization
3. HTML smoke test for regime report sections
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from src.domain.services.regime import (
    AnalysisMetrics,
    ParamProvenance,
    RecommenderResult,
    get_regime_params,
)
from src.domain.signals.indicators.regime import (
    ChopState,
    ComponentStates,
    ComponentValues,
    DataQuality,
    DataWindow,
    ExtState,
    IVState,
    MarketRegime,
    RegimeOutput,
    RegimeTransitionState,
    RuleTrace,
    TrendState,
    VolState,
    generate_counterfactual,
)

# =============================================================================
# Unit Tests: Component State Classifications
# =============================================================================


class TestTrendState:
    """Test TrendState enum and classification."""

    def test_trend_up_values(self) -> None:
        """TrendState.UP should have expected value."""
        assert TrendState.UP.value == "trend_up"

    def test_trend_down_values(self) -> None:
        """TrendState.DOWN should have expected value."""
        assert TrendState.DOWN.value == "trend_down"

    def test_trend_neutral_values(self) -> None:
        """TrendState.NEUTRAL should have expected value."""
        assert TrendState.NEUTRAL.value == "neutral"


class TestVolState:
    """Test VolState enum and classification."""

    def test_vol_high_values(self) -> None:
        """VolState.HIGH should have expected value."""
        assert VolState.HIGH.value == "vol_high"

    def test_vol_normal_values(self) -> None:
        """VolState.NORMAL should have expected value."""
        assert VolState.NORMAL.value == "vol_normal"

    def test_vol_low_values(self) -> None:
        """VolState.LOW should have expected value."""
        assert VolState.LOW.value == "vol_low"


class TestChopState:
    """Test ChopState enum and classification."""

    def test_chop_choppy_values(self) -> None:
        """ChopState.CHOPPY should have expected value."""
        assert ChopState.CHOPPY.value == "choppy"

    def test_chop_trending_values(self) -> None:
        """ChopState.TRENDING should have expected value."""
        assert ChopState.TRENDING.value == "trending"


class TestMarketRegime:
    """Test MarketRegime enum properties."""

    def test_regime_display_names(self) -> None:
        """Each regime should have a display name."""
        assert MarketRegime.R0_HEALTHY_UPTREND.display_name == "Healthy Uptrend"
        assert MarketRegime.R1_CHOPPY_EXTENDED.display_name == "Choppy/Extended"
        assert MarketRegime.R2_RISK_OFF.display_name == "Risk-Off"
        assert MarketRegime.R3_REBOUND_WINDOW.display_name == "Rebound Window"

    def test_regime_severity_ordering(self) -> None:
        """Regime severity should be ordered R0 < R1 < R3 < R2."""
        assert MarketRegime.R0_HEALTHY_UPTREND.severity < MarketRegime.R1_CHOPPY_EXTENDED.severity
        assert MarketRegime.R1_CHOPPY_EXTENDED.severity < MarketRegime.R3_REBOUND_WINDOW.severity
        assert MarketRegime.R3_REBOUND_WINDOW.severity < MarketRegime.R2_RISK_OFF.severity


# =============================================================================
# Unit Tests: RuleTrace
# =============================================================================


class TestRuleTrace:
    """Test RuleTrace dataclass."""

    def test_rule_trace_creation(self) -> None:
        """RuleTrace should be created with expected fields."""
        trace = RuleTrace(
            rule_id="r0_trend_up",
            description="Trend is UP",
            passed=True,
            evidence={"trend_state": "trend_up", "close": 150.0},
            regime_target="R0",
            category="trend",
            priority=1,
        )
        assert trace.rule_id == "r0_trend_up"
        assert trace.passed is True
        assert trace.regime_target == "R0"

    def test_rule_trace_to_dict(self) -> None:
        """RuleTrace.to_dict should serialize correctly."""
        trace = RuleTrace(
            rule_id="r2_vol_high",
            description="Vol is HIGH",
            passed=False,
            evidence={"vol_state": "vol_normal"},
            regime_target="R2",
            category="vol",
            priority=2,
        )
        result = trace.to_dict()
        assert result["rule_id"] == "r2_vol_high"
        assert result["passed"] is False
        assert result["regime_target"] == "R2"


# =============================================================================
# Snapshot Test: RegimeOutput Serialization
# =============================================================================


class TestRegimeOutputSerialization:
    """Test RegimeOutput to_dict() stable serialization."""

    def test_to_dict_produces_valid_json(self) -> None:
        """to_dict should produce JSON-serializable output."""
        output = RegimeOutput(
            schema_version="regime_output@1.0",
            symbol="TEST",
            asof_ts=datetime(2026, 1, 17, 12, 0, 0),
            final_regime=MarketRegime.R1_CHOPPY_EXTENDED,
            decision_regime=MarketRegime.R0_HEALTHY_UPTREND,
            confidence=75,
        )

        result = output.to_dict(precision=4)

        # Should be JSON serializable
        json_str = json.dumps(result, sort_keys=True)
        assert json_str is not None

    def test_to_dict_stable_ordering(self) -> None:
        """to_dict should produce stable key ordering."""
        # Use fixed timestamp to ensure deterministic results
        fixed_ts = datetime(2026, 1, 17, 12, 0, 0)
        fixed_window = DataWindow(start_ts=fixed_ts, end_ts=fixed_ts, bars=100)

        output1 = RegimeOutput(
            symbol="TEST",
            final_regime=MarketRegime.R1_CHOPPY_EXTENDED,
            asof_ts=fixed_ts,
            data_window=fixed_window,
        )
        output2 = RegimeOutput(
            symbol="TEST",
            final_regime=MarketRegime.R1_CHOPPY_EXTENDED,
            asof_ts=fixed_ts,
            data_window=fixed_window,
        )

        # Same inputs should produce same serialization
        json1 = json.dumps(output1.to_dict(precision=4), sort_keys=True)
        json2 = json.dumps(output2.to_dict(precision=4), sort_keys=True)
        assert json1 == json2

    def test_to_dict_rounds_floats(self) -> None:
        """to_dict should round floats to specified precision."""
        output = RegimeOutput(
            symbol="TEST",
            component_values=ComponentValues(
                close=150.123456789,
                atr20=5.987654321,
            ),
        )

        result = output.to_dict(precision=2)
        # Check that floats are rounded
        assert result["component_values"]["close"] == 150.12
        assert result["component_values"]["atr20"] == 5.99

    def test_to_dict_handles_enums(self) -> None:
        """to_dict should convert enums to values."""
        output = RegimeOutput(
            symbol="TEST",
            final_regime=MarketRegime.R2_RISK_OFF,
            component_states=ComponentStates(
                trend_state=TrendState.DOWN,
                vol_state=VolState.HIGH,
            ),
        )

        result = output.to_dict()
        assert result["final_regime"] == "R2"
        assert result["component_states"]["trend_state"] == "trend_down"
        assert result["component_states"]["vol_state"] == "vol_high"


# =============================================================================
# Unit Tests: ParamProvenance
# =============================================================================


class TestParamProvenance:
    """Test ParamProvenance dataclass."""

    def test_compute_param_set_id_deterministic(self) -> None:
        """Same params should produce same ID."""
        params = {"vol_high_short_pct": 80, "chop_high_pct": 70}

        id1 = ParamProvenance.compute_param_set_id(params, "TEST")
        id2 = ParamProvenance.compute_param_set_id(params, "TEST")

        assert id1 == id2
        assert len(id1) == 8  # 8-char hex hash

    def test_compute_param_set_id_different_for_different_symbols(self) -> None:
        """Different symbols should produce different IDs."""
        params = {"vol_high_short_pct": 80}

        id1 = ParamProvenance.compute_param_set_id(params, "AAPL")
        id2 = ParamProvenance.compute_param_set_id(params, "NVDA")

        assert id1 != id2

    def test_from_params_creates_provenance(self) -> None:
        """from_params should create a valid provenance."""
        params = {"vol_high_short_pct": 85, "chop_high_pct": 70}
        provenance = ParamProvenance.from_params(
            params=params,
            symbol="NVDA",
            source="symbol-specific",
        )

        assert provenance.symbol == "NVDA"
        assert provenance.source == "symbol-specific"
        assert len(provenance.param_set_id) == 8

    def test_provenance_validation_flags(self) -> None:
        """Validation flags should work correctly."""
        prov = ParamProvenance(
            pbo_value=0.3,  # Good (< 0.5)
            dsr_value=1.2,  # Good (> 1.0)
            oos_sharpe=0.5,  # Good (> 0)
            walk_forward_folds=5,
        )

        assert prov.is_validated is True
        assert prov.pbo_ok is True
        assert prov.dsr_ok is True
        assert prov.oos_ok is True


# =============================================================================
# Unit Tests: AnalysisMetrics
# =============================================================================


class TestAnalysisMetrics:
    """Test AnalysisMetrics dataclass."""

    def test_analysis_metrics_to_dict(self) -> None:
        """to_dict should serialize metrics correctly."""
        metrics = AnalysisMetrics(
            vol_threshold=80.0,
            vol_boundary_density=0.15,
            vol_above_threshold_pct=0.25,
            vol_proxy_mean=55.5,
            vol_proxy_current=60.0,
            chop_threshold=70.0,
            chop_boundary_density=0.10,
        )

        result = metrics.to_dict()
        assert result["vol_threshold"] == 80.0
        assert result["vol_boundary_density"] == 0.15
        assert result["vol_proxy_mean"] == 55.5


# =============================================================================
# Unit Tests: RecommenderResult
# =============================================================================


class TestRecommenderResult:
    """Test RecommenderResult dataclass."""

    def test_no_recommendations_result(self) -> None:
        """Result with no recommendations should be valid."""
        from datetime import date

        result = RecommenderResult(
            symbol="AAPL",
            analysis_date=date(2026, 1, 17),
            lookback_days=63,
            has_recommendations=False,
            no_change_reason="Parameters appear well-calibrated",
            boundary_density_ok=True,
        )

        assert result.has_recommendations is False
        assert result.no_change_reason == "Parameters appear well-calibrated"

    def test_result_to_dict(self) -> None:
        """to_dict should serialize correctly."""
        from datetime import date

        result = RecommenderResult(
            symbol="NVDA",
            analysis_date=date(2026, 1, 17),
            lookback_days=63,
            has_recommendations=False,
            current_params={"vol_high_short_pct": 88},
            analysis_metrics=AnalysisMetrics(vol_threshold=88.0),
        )

        d = result.to_dict()
        assert d["symbol"] == "NVDA"
        assert d["current_params"]["vol_high_short_pct"] == 88


# =============================================================================
# Unit Tests: Counterfactual Generation
# =============================================================================


class TestCounterfactual:
    """Test counterfactual generation."""

    def test_generate_counterfactual_returns_failures(self) -> None:
        """generate_counterfactual should return failed conditions."""
        from src.domain.signals.indicators.regime.rule_trace import ThresholdInfo

        rules = [
            RuleTrace(
                rule_id="r0_trend_up",
                description="Trend is UP",
                passed=False,
                evidence={},
                regime_target="R0",
                category="trend",
                priority=1,
                failed_conditions=[
                    ThresholdInfo(
                        metric_name="trend_state",
                        current_value=0,
                        threshold=1,
                        operator="==",
                        gap=1.0,
                    )
                ],
            ),
            RuleTrace(
                rule_id="r0_chop_low",
                description="Chop is LOW",
                passed=False,
                evidence={},
                regime_target="R0",
                category="chop",
                priority=2,
                failed_conditions=[
                    ThresholdInfo(
                        metric_name="chop_pctile",
                        current_value=75.0,
                        threshold=30.0,
                        operator="<",
                        gap=45.0,
                    )
                ],
            ),
        ]

        failures = generate_counterfactual(rules, "R0")
        assert len(failures) > 0
        # Should be sorted by gap (smallest first)
        assert failures[0].metric_name == "trend_state"


# =============================================================================
# Smoke Test: HTML Report Rendering
# =============================================================================


class TestHTMLReportSmoke:
    """Smoke tests for HTML report generation."""

    def test_escape_html_prevents_xss(self) -> None:
        """escape_html should prevent XSS attacks."""
        from src.infrastructure.reporting.value_card import escape_html

        malicious = "<script>alert('xss')</script>"
        escaped = escape_html(malicious)

        assert "<script>" not in escaped
        assert "&lt;script&gt;" in escaped

    def test_regime_styles_returns_css(self) -> None:
        """generate_regime_styles should return CSS string."""
        from src.infrastructure.reporting.regime_report import generate_regime_styles

        css = generate_regime_styles()
        assert ".regime-dashboard" in css
        assert ".report-header-section" in css

    def test_report_header_renders(self) -> None:
        """generate_report_header_html should render without errors."""
        from src.infrastructure.reporting.regime_report import generate_report_header_html

        output = RegimeOutput(
            symbol="TEST",
            asof_ts=datetime(2026, 1, 17, 12, 0),
            final_regime=MarketRegime.R1_CHOPPY_EXTENDED,
        )

        html = generate_report_header_html(output)
        assert "TEST" in html
        assert "R1" in html
        assert "report-header-section" in html

    def test_one_liner_renders(self) -> None:
        """generate_regime_one_liner_html should render without errors."""
        from src.infrastructure.reporting.regime_report import generate_regime_one_liner_html

        output = RegimeOutput(
            symbol="TEST",
            decision_regime=MarketRegime.R0_HEALTHY_UPTREND,
            final_regime=MarketRegime.R1_CHOPPY_EXTENDED,
        )

        html = generate_regime_one_liner_html(output)
        assert "R0" in html or "R1" in html

    def test_decision_tree_renders(self) -> None:
        """generate_decision_tree_html should render without errors."""
        from src.infrastructure.reporting.regime_report import generate_decision_tree_html

        output = RegimeOutput(
            symbol="TEST",
            final_regime=MarketRegime.R1_CHOPPY_EXTENDED,
            rules_fired_decision=[
                RuleTrace(
                    rule_id="r1_choppy",
                    description="Choppy condition",
                    passed=True,
                    evidence={"chop_state": "choppy"},
                    regime_target="R1",
                    category="chop",
                    priority=3,
                )
            ],
        )

        html = generate_decision_tree_html(output)
        assert "decision-tree" in html or "Decision" in html

    def test_recommendations_renders_no_changes(self) -> None:
        """generate_recommendations_html should render 'no changes' case."""
        from datetime import date

        from src.infrastructure.reporting.regime_report import generate_recommendations_html

        result = RecommenderResult(
            symbol="TEST",
            analysis_date=date(2026, 1, 17),
            lookback_days=63,
            has_recommendations=False,
            no_change_reason="Well calibrated",
            analysis_metrics=AnalysisMetrics(
                vol_threshold=80.0,
                vol_boundary_density=0.10,
            ),
            current_params={"vol_high_short_pct": 80},
        )

        html = generate_recommendations_html(result)
        assert "NO CHANGES SUGGESTED" in html
        assert "Well calibrated" in html


# =============================================================================
# Snapshot Test: Golden File Comparison
# =============================================================================


class TestRegimeOutputSnapshot:
    """Snapshot tests for stable serialization across versions."""

    @staticmethod
    def _create_deterministic_output() -> RegimeOutput:
        """Create a RegimeOutput with fully deterministic values."""
        from src.domain.signals.indicators.regime.rule_trace import ThresholdInfo

        fixed_ts = datetime(2026, 1, 17, 12, 0, 0)

        return RegimeOutput(
            schema_version="regime_output@1.0",
            symbol="NVDA",
            asof_ts=fixed_ts,
            bar_interval="1d",
            data_window=DataWindow(
                start_ts=datetime(2025, 1, 1, 0, 0, 0),
                end_ts=fixed_ts,
                bars=252,
            ),
            decision_regime=MarketRegime.R0_HEALTHY_UPTREND,
            final_regime=MarketRegime.R1_CHOPPY_EXTENDED,
            regime_name="Choppy/Extended",
            confidence=72,
            component_states=ComponentStates(
                trend_state=TrendState.UP,
                vol_state=VolState.NORMAL,
                chop_state=ChopState.CHOPPY,
                ext_state=ExtState.NEUTRAL,
                iv_state=IVState.NA,
            ),
            component_values=ComponentValues(
                close=145.50,
                ma20=142.30,
                ma50=140.25,
                ma200=135.10,
                ma50_slope=0.0125,
                atr20=3.85,
                atr_pct=2.65,
                atr_pct_63=65.0,
                atr_pct_252=58.0,
                chop=55.5,
                chop_pct_252=78.0,
                ma20_crosses=3,
                ext=0.83,
            ),
            rules_fired_decision=[
                RuleTrace(
                    rule_id="r2_trend_down",
                    description="R2: Trend is DOWN",
                    passed=False,
                    evidence={"trend_state": "trend_up", "close": 145.50},
                    regime_target="R2",
                    category="trend",
                    priority=1,
                    failed_conditions=[
                        ThresholdInfo(
                            metric_name="trend_state",
                            current_value=0,
                            threshold=1,
                            operator="==",
                            gap=1.0,
                        )
                    ],
                ),
                RuleTrace(
                    rule_id="r1_chop_high",
                    description="R1: Choppiness is HIGH",
                    passed=True,
                    evidence={"chop_pctile": 78.0, "threshold": 70.0},
                    regime_target="R1",
                    category="chop",
                    priority=3,
                ),
            ],
            quality=DataQuality(
                warmup_ok=True,
                warmup_bars_needed=252,
                warmup_bars_available=312,
                component_validity={
                    "trend": True,
                    "vol": True,
                    "chop": True,
                    "ext": True,
                    "iv": False,
                },
                component_issues={"iv": "not available for single-name"},
            ),
            transition=RegimeTransitionState(
                pending_regime=MarketRegime.R0_HEALTHY_UPTREND,
                pending_count=1,
                entry_threshold=2,
                exit_threshold=2,
                bars_in_current=5,
                last_transition_ts=datetime(2026, 1, 12, 16, 0, 0),
                transition_reason="CHOP_pct crossed above 70",
            ),
            regime_changed=False,
            previous_regime=MarketRegime.R0_HEALTHY_UPTREND,
        )

    def test_snapshot_matches_golden_file(self, tmp_path: Any) -> None:
        """Verify serialized output matches expected golden file."""
        output = self._create_deterministic_output()
        result = output.to_dict(precision=4)

        # Golden reference - these are the expected keys at top level
        # Note: indicator_traces added in Phase 3 for observability
        # Note: turning_point added in Phase 4 for turning point detection
        expected_keys = {
            "schema_version",
            "symbol",
            "asof_ts",
            "bar_interval",
            "data_window",
            "decision_regime",
            "final_regime",
            "regime_name",
            "confidence",
            "component_states",
            "component_values",
            "inputs_used",
            "derived_metrics",
            "rules_fired_decision",
            "rules_fired_hysteresis",
            "quality",
            "transition",
            "regime_changed",
            "previous_regime",
            "indicator_traces",  # Phase 3: Observability
            "turning_point",  # Phase 4: Turning Point Detection
        }

        assert set(result.keys()) == expected_keys, f"Key mismatch: {set(result.keys())}"

        # Verify specific values
        assert result["schema_version"] == "regime_output@1.0"
        assert result["symbol"] == "NVDA"
        assert result["decision_regime"] == "R0"
        assert result["final_regime"] == "R1"
        assert result["confidence"] == 72
        assert result["regime_changed"] is False
        assert result["previous_regime"] == "R0"

        # Verify nested structures
        assert result["component_states"]["trend_state"] == "trend_up"
        assert result["component_states"]["chop_state"] == "choppy"
        assert result["component_values"]["close"] == 145.5
        assert result["transition"]["pending_count"] == 1
        assert result["quality"]["warmup_ok"] is True

    def test_serialization_is_idempotent(self) -> None:
        """Multiple serializations produce identical output."""
        output = self._create_deterministic_output()

        result1 = json.dumps(output.to_dict(precision=4), sort_keys=True)
        result2 = json.dumps(output.to_dict(precision=4), sort_keys=True)
        result3 = json.dumps(output.to_dict(precision=4), sort_keys=True)

        assert result1 == result2 == result3

    def test_precision_affects_output(self) -> None:
        """Different precision values produce different outputs."""
        output = self._create_deterministic_output()

        result_p2 = output.to_dict(precision=2)
        result_p4 = output.to_dict(precision=4)

        # ma50_slope should differ based on precision
        # 0.0125 rounded to 2 decimal places = 0.01
        # 0.0125 rounded to 4 decimal places = 0.0125
        assert result_p2["component_values"]["ma50_slope"] == 0.01
        assert result_p4["component_values"]["ma50_slope"] == 0.0125


# =============================================================================
# Integration Test: get_regime_params
# =============================================================================


class TestGetRegimeParams:
    """Test get_regime_params function."""

    def test_get_default_params(self) -> None:
        """get_regime_params should return default params for unknown symbol."""
        params = get_regime_params("UNKNOWN_SYMBOL")
        assert "vol_high_short_pct" in params
        assert "chop_high_pct" in params
        assert "ma50_period" in params

    def test_get_known_symbol_params(self) -> None:
        """get_regime_params should return params for known symbol."""
        # NVDA has custom params
        params = get_regime_params("NVDA")
        assert "vol_high_short_pct" in params
        # NVDA params might differ from default
        assert params is not None
