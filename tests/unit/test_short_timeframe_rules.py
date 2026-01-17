"""
Unit tests for short-timeframe trading signal rules.

Tests:
- Rule count validation (16 rules)
- All rules target ("1m", "5m") timeframes only
- All rules use ALERT direction (informational)
- Cooldowns are in appropriate range (60-300 seconds)
- Strength values are calibrated for short timeframes (30-50)
- Naming convention follows *_st_* pattern
- Category distribution across indicators
"""

from src.domain.signals.models import (
    ConditionType,
    SignalCategory,
    SignalDirection,
    SignalPriority,
)
from src.domain.signals.rules import (
    ALL_RULES,
    SHORT_TIMEFRAME_RULES,
)


class TestShortTimeframeRulesStructure:
    """Tests for rule collection structure and configuration."""

    def test_rule_count(self) -> None:
        """Verify expected number of short-timeframe rules."""
        assert len(SHORT_TIMEFRAME_RULES) == 16

    def test_rules_in_all_rules(self) -> None:
        """Verify short-timeframe rules are included in ALL_RULES."""
        all_rule_names = {r.name for r in ALL_RULES}
        short_rule_names = {r.name for r in SHORT_TIMEFRAME_RULES}
        assert short_rule_names.issubset(all_rule_names)

    def test_all_rules_target_short_timeframes(self) -> None:
        """All rules should only target 1m and 5m timeframes."""
        for rule in SHORT_TIMEFRAME_RULES:
            assert rule.timeframes == (
                "1m",
                "5m",
            ), f"{rule.name} has wrong timeframes: {rule.timeframes}"

    def test_all_rules_are_alerts(self) -> None:
        """All rules should use ALERT direction initially."""
        for rule in SHORT_TIMEFRAME_RULES:
            assert (
                rule.direction == SignalDirection.ALERT
            ), f"{rule.name} should be ALERT, got {rule.direction}"

    def test_cooldown_ranges(self) -> None:
        """Cooldowns should be between 60-300 seconds."""
        for rule in SHORT_TIMEFRAME_RULES:
            assert (
                60 <= rule.cooldown_seconds <= 300
            ), f"{rule.name} cooldown {rule.cooldown_seconds}s out of range [60-300]"

    def test_strength_ranges(self) -> None:
        """Strength should be between 30-50 for short-term signals."""
        for rule in SHORT_TIMEFRAME_RULES:
            assert (
                30 <= rule.strength <= 50
            ), f"{rule.name} strength {rule.strength} out of range [30-50]"

    def test_naming_convention(self) -> None:
        """All rules should follow *_st_* naming convention."""
        for rule in SHORT_TIMEFRAME_RULES:
            assert "_st_" in rule.name, f"{rule.name} doesn't follow _st_ naming convention"

    def test_all_rules_enabled(self) -> None:
        """All rules should be enabled by default."""
        for rule in SHORT_TIMEFRAME_RULES:
            assert rule.enabled is True, f"{rule.name} is disabled"


class TestShortTimeframeRulesCategories:
    """Tests for category distribution and indicator coverage."""

    def test_momentum_rule_count(self) -> None:
        """Verify momentum category rule count."""
        momentum_rules = [r for r in SHORT_TIMEFRAME_RULES if r.category == SignalCategory.MOMENTUM]
        assert len(momentum_rules) == 8

    def test_trend_rule_count(self) -> None:
        """Verify trend category rule count."""
        trend_rules = [r for r in SHORT_TIMEFRAME_RULES if r.category == SignalCategory.TREND]
        assert len(trend_rules) == 2

    def test_volatility_rule_count(self) -> None:
        """Verify volatility category rule count."""
        volatility_rules = [
            r for r in SHORT_TIMEFRAME_RULES if r.category == SignalCategory.VOLATILITY
        ]
        assert len(volatility_rules) == 2

    def test_volume_rule_count(self) -> None:
        """Verify volume category rule count."""
        volume_rules = [r for r in SHORT_TIMEFRAME_RULES if r.category == SignalCategory.VOLUME]
        assert len(volume_rules) == 4

    def test_indicator_coverage(self) -> None:
        """Verify all expected indicators are covered."""
        expected_indicators = {
            "rsi",
            "williams_r",
            "roc",  # Momentum
            "supertrend",  # Trend
            "atr",  # Volatility
            "obv",
            "cvd",  # Volume
        }
        actual_indicators = {r.indicator for r in SHORT_TIMEFRAME_RULES}
        assert actual_indicators == expected_indicators


class TestShortTimeframeRulesConditions:
    """Tests for condition types and configurations."""

    def test_state_change_rules_have_from_to_config(self) -> None:
        """STATE_CHANGE rules must have 'from' and 'to' in config."""
        for rule in SHORT_TIMEFRAME_RULES:
            if rule.condition_type == ConditionType.STATE_CHANGE:
                assert (
                    "field" in rule.condition_config
                ), f"{rule.name} missing 'field' in condition_config"
                assert (
                    "from" in rule.condition_config
                ), f"{rule.name} missing 'from' in condition_config"
                assert (
                    "to" in rule.condition_config
                ), f"{rule.name} missing 'to' in condition_config"

    def test_threshold_rules_have_threshold_config(self) -> None:
        """THRESHOLD_CROSS_* rules must have 'threshold' in config."""
        threshold_types = {
            ConditionType.THRESHOLD_CROSS_UP,
            ConditionType.THRESHOLD_CROSS_DOWN,
        }
        for rule in SHORT_TIMEFRAME_RULES:
            if rule.condition_type in threshold_types:
                assert (
                    "field" in rule.condition_config
                ), f"{rule.name} missing 'field' in condition_config"
                assert (
                    "threshold" in rule.condition_config
                ), f"{rule.name} missing 'threshold' in condition_config"

    def test_unique_rule_names(self) -> None:
        """All rule names must be unique."""
        names = [r.name for r in SHORT_TIMEFRAME_RULES]
        assert len(names) == len(set(names)), "Duplicate rule names found"


class TestShortTimeframeRulesPriority:
    """Tests for priority assignment."""

    def test_extreme_conditions_have_medium_priority(self) -> None:
        """Extreme condition rules should have MEDIUM priority."""
        extreme_rules = [r for r in SHORT_TIMEFRAME_RULES if "extreme" in r.name]
        for rule in extreme_rules:
            assert rule.priority == SignalPriority.MEDIUM, f"{rule.name} should be MEDIUM priority"

    def test_regular_rules_have_low_priority(self) -> None:
        """Regular short-term rules should have LOW priority."""
        regular_rules = [r for r in SHORT_TIMEFRAME_RULES if "extreme" not in r.name]
        for rule in regular_rules:
            assert rule.priority == SignalPriority.LOW, f"{rule.name} should be LOW priority"


class TestShortTimeframeRulesMessages:
    """Tests for message templates."""

    def test_all_rules_have_message_template(self) -> None:
        """All rules should have a message template."""
        for rule in SHORT_TIMEFRAME_RULES:
            assert rule.message_template, f"{rule.name} missing message_template"

    def test_message_templates_contain_symbol_placeholder(self) -> None:
        """Message templates should contain {symbol} placeholder."""
        for rule in SHORT_TIMEFRAME_RULES:
            assert (
                "{symbol}" in rule.message_template
            ), f"{rule.name} message template missing {{symbol}}"

    def test_message_templates_indicate_short_term(self) -> None:
        """Message templates should indicate short-term context."""
        for rule in SHORT_TIMEFRAME_RULES:
            msg = rule.message_template.lower()
            assert (
                "short" in msg or "extreme" in msg
            ), f"{rule.name} message should indicate short-term context"
