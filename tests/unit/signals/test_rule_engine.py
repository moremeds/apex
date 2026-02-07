"""
Unit tests for RuleEngine and RuleRegistry.

Tests the core rule evaluation engine including:
- Start/stop lifecycle
- RuleRegistry rule management
- INDICATOR_UPDATE event handling
- Rule condition evaluation
- Cooldown logic
- Signal emission
- Trace mode and evaluation history
"""

from datetime import datetime, timedelta, timezone

from src.domain.events.event_types import EventType
from src.domain.signals.models import (
    ConditionType,
    SignalCategory,
    SignalDirection,
    SignalPriority,
)
from src.domain.signals.rule_engine import RuleEngine, RuleRegistry

from .conftest import MockEventBus, make_indicator_update_event, make_signal_rule

# =============================================================================
# RuleRegistry Tests
# =============================================================================


class TestRuleRegistry:
    """Test RuleRegistry functionality."""

    def test_registry_starts_empty(self) -> None:
        """New registry should be empty."""
        registry = RuleRegistry()
        assert len(registry) == 0
        assert registry.get_all_rules() == []

    def test_add_single_rule(self) -> None:
        """Adding a rule should make it retrievable."""
        registry = RuleRegistry()
        rule = make_signal_rule(name="test_rule", indicator="rsi")

        registry.add_rule(rule)

        assert len(registry) == 1
        assert registry.get_rule_by_name("test_rule") == rule

    def test_add_multiple_rules(self) -> None:
        """add_rules should add multiple rules at once."""
        registry = RuleRegistry()
        rules = [
            make_signal_rule(name="rule1", indicator="rsi"),
            make_signal_rule(name="rule2", indicator="macd"),
        ]

        registry.add_rules(rules)

        assert len(registry) == 2
        assert registry.get_rule_by_name("rule1") is not None
        assert registry.get_rule_by_name("rule2") is not None

    def test_get_rules_for_indicator(self) -> None:
        """Should return only rules matching the indicator."""
        registry = RuleRegistry()
        registry.add_rules(
            [
                make_signal_rule(name="rsi_rule1", indicator="rsi"),
                make_signal_rule(name="rsi_rule2", indicator="rsi"),
                make_signal_rule(name="macd_rule", indicator="macd"),
            ]
        )

        rsi_rules = registry.get_rules_for_indicator("rsi")

        assert len(rsi_rules) == 2
        assert all(r.indicator == "rsi" for r in rsi_rules)

    def test_get_rules_for_unknown_indicator(self) -> None:
        """Should return empty list for unknown indicator."""
        registry = RuleRegistry()
        registry.add_rule(make_signal_rule(indicator="rsi"))

        rules = registry.get_rules_for_indicator("unknown")
        assert rules == []

    def test_get_rule_by_name_not_found(self) -> None:
        """Should return None for unknown rule name."""
        registry = RuleRegistry()
        result = registry.get_rule_by_name("nonexistent")
        assert result is None

    def test_clear_removes_all_rules(self) -> None:
        """clear() should remove all rules."""
        registry = RuleRegistry()
        registry.add_rules(
            [
                make_signal_rule(name="rule1"),
                make_signal_rule(name="rule2"),
            ]
        )

        registry.clear()

        assert len(registry) == 0
        assert registry.get_all_rules() == []


class TestRuleRegistryFromConfig:
    """Test RuleRegistry.from_config() factory method."""

    def test_from_empty_config(self) -> None:
        """Empty config should create empty registry."""
        registry = RuleRegistry.from_config({})
        assert len(registry) == 0

    def test_from_config_with_rules(self) -> None:
        """Should parse rules from config dict."""
        config = {
            "rules": {
                "rsi_oversold": {
                    "enabled": True,
                    "indicator": "rsi",
                    "category": "momentum",
                    "direction": "buy",
                    "strength": 70,
                    "priority": "high",
                    "condition": {
                        "type": "state_change",
                        "field": "zone",
                        "from": ["oversold"],
                        "to": ["neutral"],
                    },
                    "timeframes": ["1h", "4h"],
                    "cooldown_seconds": 7200,
                    "message": "{symbol} RSI exited oversold",
                }
            }
        }

        registry = RuleRegistry.from_config(config)

        assert len(registry) == 1
        rule = registry.get_rule_by_name("rsi_oversold")
        assert rule is not None
        assert rule.indicator == "rsi"
        assert rule.category == SignalCategory.MOMENTUM
        assert rule.direction == SignalDirection.BUY
        assert rule.strength == 70
        assert rule.priority == SignalPriority.HIGH
        assert rule.condition_type == ConditionType.STATE_CHANGE
        assert "1h" in rule.timeframes
        assert "4h" in rule.timeframes
        assert rule.cooldown_seconds == 7200

    def test_from_config_skips_disabled_rules(self) -> None:
        """Disabled rules should not be loaded."""
        config = {
            "rules": {
                "enabled_rule": {
                    "enabled": True,
                    "indicator": "rsi",
                    "category": "momentum",
                    "direction": "buy",
                    "condition": {"type": "custom"},
                },
                "disabled_rule": {
                    "enabled": False,
                    "indicator": "macd",
                    "category": "momentum",
                    "direction": "sell",
                    "condition": {"type": "custom"},
                },
            }
        }

        registry = RuleRegistry.from_config(config)

        assert len(registry) == 1
        assert registry.get_rule_by_name("enabled_rule") is not None
        assert registry.get_rule_by_name("disabled_rule") is None

    def test_from_config_handles_invalid_rule(self) -> None:
        """Invalid rules should be skipped with error logging."""
        config = {
            "rules": {
                "valid_rule": {
                    "enabled": True,
                    "indicator": "rsi",
                    "category": "momentum",
                    "direction": "buy",
                    "condition": {"type": "custom"},
                },
                "invalid_rule": {
                    # Missing required fields
                    "enabled": True,
                },
            }
        }

        registry = RuleRegistry.from_config(config)

        # Should load valid rule and skip invalid
        assert len(registry) == 1
        assert registry.get_rule_by_name("valid_rule") is not None


# =============================================================================
# RuleEngine Lifecycle Tests
# =============================================================================


class TestRuleEngineLifecycle:
    """Test RuleEngine start/stop behavior."""

    def test_start_subscribes_to_indicator_update(self, mock_event_bus: MockEventBus) -> None:
        """Engine should subscribe to INDICATOR_UPDATE on start."""
        engine = RuleEngine(mock_event_bus)

        engine.start()

        assert EventType.INDICATOR_UPDATE in mock_event_bus.subscriptions
        assert len(mock_event_bus.subscriptions[EventType.INDICATOR_UPDATE]) == 1

    def test_start_is_idempotent(self, mock_event_bus: MockEventBus) -> None:
        """Multiple start() calls should not create duplicate subscriptions."""
        engine = RuleEngine(mock_event_bus)

        engine.start()
        engine.start()

        assert len(mock_event_bus.subscriptions[EventType.INDICATOR_UPDATE]) == 1

    def test_stop_sets_started_false(self, mock_event_bus: MockEventBus) -> None:
        """stop() should set _started to False."""
        engine = RuleEngine(mock_event_bus)
        engine.start()

        engine.stop()

        assert engine._started is False


# =============================================================================
# RuleEngine Properties Tests
# =============================================================================


class TestRuleEngineProperties:
    """Test RuleEngine property accessors."""

    def test_signals_emitted_starts_at_zero(self, mock_event_bus: MockEventBus) -> None:
        """signals_emitted should start at 0."""
        engine = RuleEngine(mock_event_bus)
        assert engine.signals_emitted == 0

    def test_rules_evaluated_starts_at_zero(self, mock_event_bus: MockEventBus) -> None:
        """rules_evaluated should start at 0."""
        engine = RuleEngine(mock_event_bus)
        assert engine.rules_evaluated == 0

    def test_registry_property(self, mock_event_bus: MockEventBus) -> None:
        """registry property should return the registry."""
        registry = RuleRegistry()
        engine = RuleEngine(mock_event_bus, registry=registry)
        # Note: RuleEngine stores a reference to the registry, verify contents match
        assert len(engine.registry) == len(registry)

    def test_trace_mode_property(self, mock_event_bus: MockEventBus) -> None:
        """trace_mode should be gettable and settable."""
        engine = RuleEngine(mock_event_bus, trace_mode=False)
        assert engine.trace_mode is False

        engine.trace_mode = True
        assert engine.trace_mode is True


# =============================================================================
# Cooldown Tests
# =============================================================================


class TestRuleEngineCooldown:
    """Test cooldown functionality."""

    def test_no_cooldown_initially(self, mock_event_bus: MockEventBus) -> None:
        """New engine should have no active cooldowns."""
        engine = RuleEngine(mock_event_bus)
        cooldowns = engine.get_all_cooldowns()
        assert cooldowns == []

    def test_cooldown_status_for_nonexistent(self, mock_event_bus: MockEventBus) -> None:
        """get_cooldown_status should return None for non-cooldown signal."""
        engine = RuleEngine(mock_event_bus)

        status = engine.get_cooldown_status("momentum", "rsi", "AAPL", "1d")
        assert status is None

    def test_clear_cooldowns_returns_count(self, mock_event_bus: MockEventBus) -> None:
        """clear_cooldowns should return number of cleared entries."""
        engine = RuleEngine(mock_event_bus)

        # No cooldowns to clear
        cleared = engine.clear_cooldowns()
        assert cleared == 0

    def test_cooldown_blocks_duplicate_signals(self, mock_event_bus: MockEventBus) -> None:
        """Signal should be blocked if in cooldown window."""
        registry = RuleRegistry()
        rule = make_signal_rule(
            name="test_rule",
            indicator="rsi",
            condition_type=ConditionType.STATE_CHANGE,
            condition_config={"field": "zone", "from": ["oversold"], "to": ["neutral"]},
            cooldown_seconds=3600,
        )
        registry.add_rule(rule)

        engine = RuleEngine(mock_event_bus, registry=registry)
        engine.start()

        timestamp = datetime.now(timezone.utc)

        # First event - should trigger signal
        event1 = make_indicator_update_event(
            indicator="rsi",
            state={"zone": "neutral"},
            previous_state={"zone": "oversold"},
            timestamp=timestamp,
        )
        mock_event_bus.publish(EventType.INDICATOR_UPDATE, event1)

        first_signal_count = len(mock_event_bus.get_events(EventType.TRADING_SIGNAL))

        # Second event with same condition (within cooldown)
        event2 = make_indicator_update_event(
            indicator="rsi",
            state={"zone": "neutral"},
            previous_state={"zone": "oversold"},
            timestamp=timestamp + timedelta(seconds=60),
        )
        mock_event_bus.publish(EventType.INDICATOR_UPDATE, event2)

        second_signal_count = len(mock_event_bus.get_events(EventType.TRADING_SIGNAL))

        # Second event should be blocked by cooldown
        assert second_signal_count == first_signal_count


# =============================================================================
# Rule Evaluation Tests
# =============================================================================


class TestRuleEvaluation:
    """Test rule condition evaluation."""

    def test_no_rules_for_indicator(self, mock_event_bus: MockEventBus) -> None:
        """Events for indicators without rules should be ignored."""
        engine = RuleEngine(mock_event_bus)  # Empty registry
        engine.start()

        event = make_indicator_update_event(indicator="unknown_indicator")
        mock_event_bus.publish(EventType.INDICATOR_UPDATE, event)

        signals = mock_event_bus.get_events(EventType.TRADING_SIGNAL)
        assert len(signals) == 0

    def test_disabled_rule_skipped(self, mock_event_bus: MockEventBus) -> None:
        """Disabled rules should not trigger signals."""
        registry = RuleRegistry()
        rule = make_signal_rule(
            indicator="rsi",
            enabled=False,
            condition_type=ConditionType.STATE_CHANGE,
            condition_config={"field": "zone", "from": ["oversold"], "to": ["neutral"]},
        )
        registry.add_rule(rule)

        engine = RuleEngine(mock_event_bus, registry=registry)
        engine.start()

        event = make_indicator_update_event(
            indicator="rsi",
            state={"zone": "neutral"},
            previous_state={"zone": "oversold"},
        )
        mock_event_bus.publish(EventType.INDICATOR_UPDATE, event)

        signals = mock_event_bus.get_events(EventType.TRADING_SIGNAL)
        assert len(signals) == 0

    def test_timeframe_filter(self, mock_event_bus: MockEventBus) -> None:
        """Rules should only trigger for matching timeframes."""
        registry = RuleRegistry()
        rule = make_signal_rule(
            indicator="rsi",
            timeframes=("1h", "4h"),  # Only hourly timeframes
            condition_type=ConditionType.STATE_CHANGE,
            condition_config={"field": "zone", "from": ["oversold"], "to": ["neutral"]},
        )
        registry.add_rule(rule)

        engine = RuleEngine(mock_event_bus, registry=registry)
        engine.start()

        # Event with non-matching timeframe
        event = make_indicator_update_event(
            indicator="rsi",
            timeframe="1d",  # Not in rule's timeframes
            state={"zone": "neutral"},
            previous_state={"zone": "oversold"},
        )
        mock_event_bus.publish(EventType.INDICATOR_UPDATE, event)

        signals = mock_event_bus.get_events(EventType.TRADING_SIGNAL)
        assert len(signals) == 0

    def test_state_change_condition_triggers(self, mock_event_bus: MockEventBus) -> None:
        """STATE_CHANGE condition should trigger on matching transition."""
        registry = RuleRegistry()
        rule = make_signal_rule(
            name="rsi_oversold_exit",
            indicator="rsi",
            timeframes=("1d",),
            condition_type=ConditionType.STATE_CHANGE,
            condition_config={"field": "zone", "from": ["oversold"], "to": ["neutral"]},
        )
        registry.add_rule(rule)

        engine = RuleEngine(mock_event_bus, registry=registry)
        engine.start()

        event = make_indicator_update_event(
            indicator="rsi",
            timeframe="1d",
            state={"zone": "neutral", "value": 35},
            previous_state={"zone": "oversold", "value": 25},
        )
        mock_event_bus.publish(EventType.INDICATOR_UPDATE, event)

        signals = mock_event_bus.get_events(EventType.TRADING_SIGNAL)
        assert len(signals) == 1

    def test_condition_not_met(self, mock_event_bus: MockEventBus) -> None:
        """No signal when condition is not met."""
        registry = RuleRegistry()
        rule = make_signal_rule(
            indicator="rsi",
            timeframes=("1d",),
            condition_type=ConditionType.STATE_CHANGE,
            condition_config={"field": "zone", "from": ["oversold"], "to": ["neutral"]},
        )
        registry.add_rule(rule)

        engine = RuleEngine(mock_event_bus, registry=registry)
        engine.start()

        # No state change (both neutral)
        event = make_indicator_update_event(
            indicator="rsi",
            timeframe="1d",
            state={"zone": "neutral"},
            previous_state={"zone": "neutral"},
        )
        mock_event_bus.publish(EventType.INDICATOR_UPDATE, event)

        signals = mock_event_bus.get_events(EventType.TRADING_SIGNAL)
        assert len(signals) == 0


# =============================================================================
# Signal Building Tests
# =============================================================================


class TestSignalBuilding:
    """Test TradingSignal construction from rules."""

    def test_signal_has_correct_fields(self, mock_event_bus: MockEventBus) -> None:
        """Built signal should have correct fields from rule and event."""
        registry = RuleRegistry()
        rule = make_signal_rule(
            name="test_signal_rule",
            indicator="rsi",
            category=SignalCategory.MOMENTUM,
            direction=SignalDirection.BUY,
            strength=75,
            priority=SignalPriority.HIGH,
            timeframes=("1d",),
            condition_type=ConditionType.STATE_CHANGE,
            condition_config={"field": "zone", "from": ["oversold"], "to": ["neutral"]},
            message_template="{symbol} RSI signal at {value}",
        )
        registry.add_rule(rule)

        engine = RuleEngine(mock_event_bus, registry=registry)
        engine.start()

        event = make_indicator_update_event(
            symbol="AAPL",
            indicator="rsi",
            timeframe="1d",
            value=35.0,
            state={"zone": "neutral", "value": 35.0},
            previous_state={"zone": "oversold", "value": 25.0},
        )
        mock_event_bus.publish(EventType.INDICATOR_UPDATE, event)

        signals = mock_event_bus.get_events(EventType.TRADING_SIGNAL)
        assert len(signals) == 1

        # TradingSignalEvent is a flat event with copied fields, not a wrapper
        signal_event = signals[0]

        assert signal_event.symbol == "AAPL"
        assert signal_event.indicator == "rsi"
        assert signal_event.category == "momentum"  # String value, not enum
        assert signal_event.direction == "LONG"  # Mapped from BUY
        assert signal_event.strength == 75.0  # Converted to float
        assert signal_event.priority == "high"  # String value
        assert signal_event.timeframe == "1d"
        assert signal_event.trigger_rule == "test_signal_rule"


# =============================================================================
# Trace Mode Tests
# =============================================================================


class TestTraceMode:
    """Test trace mode and evaluation history."""

    def test_trace_mode_disabled_no_history(self, mock_event_bus: MockEventBus) -> None:
        """With trace_mode=False, evaluation history should be empty."""
        registry = RuleRegistry()
        rule = make_signal_rule(indicator="rsi", timeframes=("1d",))
        registry.add_rule(rule)

        engine = RuleEngine(mock_event_bus, registry=registry, trace_mode=False)
        engine.start()

        event = make_indicator_update_event(indicator="rsi", timeframe="1d")
        mock_event_bus.publish(EventType.INDICATOR_UPDATE, event)

        history = engine.get_evaluation_history()
        assert history == []

    def test_trace_mode_enabled_records_history(self, mock_event_bus: MockEventBus) -> None:
        """With trace_mode=True, evaluations should be recorded."""
        registry = RuleRegistry()
        rule = make_signal_rule(
            indicator="rsi",
            timeframes=("1d",),
            condition_type=ConditionType.STATE_CHANGE,
            condition_config={"field": "zone", "from": ["oversold"], "to": ["neutral"]},
        )
        registry.add_rule(rule)

        engine = RuleEngine(mock_event_bus, registry=registry, trace_mode=True)
        engine.start()

        event = make_indicator_update_event(
            indicator="rsi",
            timeframe="1d",
            state={"zone": "neutral"},
            previous_state={"zone": "oversold"},
        )
        mock_event_bus.publish(EventType.INDICATOR_UPDATE, event)

        history = engine.get_evaluation_history()
        assert len(history) > 0

    def test_get_evaluation_history_triggered_only(self, mock_event_bus: MockEventBus) -> None:
        """triggered_only=True should filter to only triggered evaluations."""
        registry = RuleRegistry()
        rule = make_signal_rule(
            indicator="rsi",
            timeframes=("1d",),
            condition_type=ConditionType.STATE_CHANGE,
            condition_config={"field": "zone", "from": ["oversold"], "to": ["neutral"]},
        )
        registry.add_rule(rule)

        engine = RuleEngine(mock_event_bus, registry=registry, trace_mode=True)
        engine.start()

        # Event that doesn't trigger
        event1 = make_indicator_update_event(
            indicator="rsi",
            timeframe="1d",
            state={"zone": "neutral"},
            previous_state={"zone": "neutral"},  # No change
        )
        mock_event_bus.publish(EventType.INDICATOR_UPDATE, event1)

        # Event that triggers
        event2 = make_indicator_update_event(
            indicator="rsi",
            timeframe="1d",
            state={"zone": "neutral"},
            previous_state={"zone": "oversold"},  # Change!
        )
        mock_event_bus.publish(EventType.INDICATOR_UPDATE, event2)

        all_history = engine.get_evaluation_history()
        triggered_only = engine.get_evaluation_history(triggered_only=True)

        assert len(all_history) >= len(triggered_only)
        assert all(e["triggered"] for e in triggered_only)


# =============================================================================
# Coercion Tests
# =============================================================================


class TestIndicatorUpdateCoercion:
    """Test IndicatorUpdateEvent coercion."""

    def test_coerce_event_passthrough(self, mock_event_bus: MockEventBus) -> None:
        """Should pass through IndicatorUpdateEvent unchanged."""
        engine = RuleEngine(mock_event_bus)
        event = make_indicator_update_event()

        result = engine._coerce_indicator_update(event)
        assert result is event

    def test_coerce_dict(self, mock_event_bus: MockEventBus) -> None:
        """Should convert dict to IndicatorUpdateEvent."""
        engine = RuleEngine(mock_event_bus)
        event_dict = {
            "symbol": "AAPL",
            "timeframe": "1d",
            "indicator": "rsi",
            "value": 50.0,
            "state": {"zone": "neutral"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        result = engine._coerce_indicator_update(event_dict)

        assert result is not None
        assert result.symbol == "AAPL"
        assert result.indicator == "rsi"

    def test_coerce_invalid(self, mock_event_bus: MockEventBus) -> None:
        """Should return None for invalid payload."""
        engine = RuleEngine(mock_event_bus)

        assert engine._coerce_indicator_update("invalid") is None
        assert engine._coerce_indicator_update(None) is None


# =============================================================================
# Helper Method Tests
# =============================================================================


class TestHelperMethods:
    """Test static helper methods."""

    def test_build_evaluation_reason_triggered(self) -> None:
        """Should return 'signal emitted' for triggered."""
        reason = RuleEngine._build_evaluation_reason(
            triggered=True, blocked_by_cooldown=False, condition_met=True
        )
        assert reason == "signal emitted"

    def test_build_evaluation_reason_error(self) -> None:
        """Should return 'evaluation error' for error."""
        reason = RuleEngine._build_evaluation_reason(
            triggered=False, blocked_by_cooldown=False, condition_met=False, error=True
        )
        assert reason == "evaluation error"

    def test_build_evaluation_reason_cooldown(self) -> None:
        """Should return 'blocked by cooldown' for cooldown."""
        reason = RuleEngine._build_evaluation_reason(
            triggered=False, blocked_by_cooldown=True, condition_met=True
        )
        assert reason == "blocked by cooldown"

    def test_build_evaluation_reason_not_met(self) -> None:
        """Should return 'condition not met' when condition fails."""
        reason = RuleEngine._build_evaluation_reason(
            triggered=False, blocked_by_cooldown=False, condition_met=False
        )
        assert reason == "condition not met"

    def test_parse_cooldown_key(self) -> None:
        """Should parse cooldown key into components."""
        category, indicator, symbol, timeframe = RuleEngine._parse_cooldown_key(
            "momentum:rsi:AAPL:1d"
        )
        assert category == "momentum"
        assert indicator == "rsi"
        assert symbol == "AAPL"
        assert timeframe == "1d"

    def test_parse_cooldown_key_incomplete(self) -> None:
        """Should handle incomplete keys gracefully."""
        category, indicator, symbol, timeframe = RuleEngine._parse_cooldown_key("momentum:rsi")
        assert category == "momentum"
        assert indicator == "rsi"
        assert symbol == ""
        assert timeframe == ""

    def test_compute_remaining_seconds_active(self) -> None:
        """Should compute remaining seconds for active cooldown."""
        now = datetime.now(timezone.utc)
        last_time = now - timedelta(seconds=30)

        remaining = RuleEngine._compute_remaining_seconds(last_time, 60.0, now)
        assert remaining == 30

    def test_compute_remaining_seconds_expired(self) -> None:
        """Should return None for expired cooldown."""
        now = datetime.now(timezone.utc)
        last_time = now - timedelta(seconds=120)

        remaining = RuleEngine._compute_remaining_seconds(last_time, 60.0, now)
        assert remaining is None
