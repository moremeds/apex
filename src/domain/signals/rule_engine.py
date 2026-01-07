"""
RuleEngine - Evaluates SignalRules on indicator updates and emits trading signals.

Subscribes to INDICATOR_UPDATE events, evaluates matching rules using
SignalRule.check_condition(), and publishes TRADING_SIGNAL events when
conditions are met (respecting per-signal cooldowns).

Cooldowns are tracked per signal_id (category:indicator:symbol:timeframe),
allowing the same rule to trigger independently on different symbols/timeframes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from src.domain.events.domain_events import IndicatorUpdateEvent, TradingSignalEvent
from src.domain.events.event_types import EventType
from src.utils.logging_setup import get_logger
from src.utils.timezone import now_local

from .models import SignalRule, TradingSignal

logger = get_logger(__name__)


class EventBusProtocol(Protocol):
    """Protocol for event bus compatibility."""

    def publish(self, event_type: EventType, payload: Any) -> None:
        """Publish an event."""
        ...

    def subscribe(self, event_type: EventType, callback: Callable[[Any], None]) -> None:
        """Subscribe to an event type."""
        ...


@dataclass
class RuleRegistry:
    """
    Registry for SignalRule lookup by indicator name.

    Provides efficient O(1) lookup of rules that apply to a given indicator,
    enabling fast rule evaluation on indicator updates.

    Example:
        registry = RuleRegistry()
        registry.add_rules(MOMENTUM_RULES)
        registry.add_rules(TREND_RULES)

        rules = registry.get_rules_for_indicator("rsi")
    """

    _by_indicator: Dict[str, List[SignalRule]] = field(default_factory=dict)
    _by_name: Dict[str, SignalRule] = field(default_factory=dict)

    def add_rule(self, rule: SignalRule) -> None:
        """
        Register a single rule.

        Args:
            rule: SignalRule to register
        """
        # Index by indicator
        if rule.indicator not in self._by_indicator:
            self._by_indicator[rule.indicator] = []
        self._by_indicator[rule.indicator].append(rule)

        # Index by name
        self._by_name[rule.name] = rule

    def add_rules(self, rules: List[SignalRule]) -> None:
        """
        Register multiple rules.

        Args:
            rules: List of SignalRules to register
        """
        for rule in rules:
            self.add_rule(rule)

    def get_rules_for_indicator(self, indicator_name: str) -> List[SignalRule]:
        """
        Get all rules that match an indicator.

        Args:
            indicator_name: Name of the indicator (e.g., "rsi", "macd")

        Returns:
            List of matching SignalRules (empty if none)
        """
        return list(self._by_indicator.get(indicator_name, []))

    def get_rule_by_name(self, rule_name: str) -> Optional[SignalRule]:
        """Get a rule by its name."""
        return self._by_name.get(rule_name)

    def get_all_rules(self) -> List[SignalRule]:
        """Get all registered rules."""
        return list(self._by_name.values())

    def clear(self) -> None:
        """Clear all registered rules."""
        self._by_indicator.clear()
        self._by_name.clear()

    def __len__(self) -> int:
        """Return total number of registered rules."""
        return len(self._by_name)


class RuleEngine:
    """
    Evaluates SignalRules on indicator updates and emits trading signals.

    The engine:
    1. Subscribes to INDICATOR_UPDATE events
    2. Finds rules matching the updated indicator
    3. Evaluates rule conditions using SignalRule.check_condition()
    4. Checks per-signal cooldowns to prevent signal spam
    5. Publishes TRADING_SIGNAL events (as TradingSignalEvent) for triggered rules

    Cooldowns are tracked per signal_id (category:indicator:symbol:timeframe),
    so the same rule can trigger independently on different symbols/timeframes.

    Example:
        registry = RuleRegistry()
        registry.add_rules(MOMENTUM_RULES)

        engine = RuleEngine(event_bus, registry)
        engine.start()
    """

    def __init__(
        self,
        event_bus: EventBusProtocol,
        registry: Optional[RuleRegistry] = None,
    ) -> None:
        """
        Initialize the rule engine.

        Args:
            event_bus: Event bus for subscriptions and publishing
            registry: RuleRegistry with rules to evaluate (creates empty if None)
        """
        self._event_bus = event_bus
        self._registry = registry or RuleRegistry()

        # Per-signal cooldown tracking: signal_id -> (last_triggered_time, cooldown_seconds)
        # signal_id format: "{category}:{indicator}:{symbol}:{timeframe}"
        self._last_triggered: Dict[str, Tuple[datetime, int]] = {}
        self._lock = RLock()

        self._started = False
        self._signals_emitted = 0
        self._rules_evaluated = 0

    @property
    def registry(self) -> RuleRegistry:
        """Access the rule registry."""
        return self._registry

    @property
    def signals_emitted(self) -> int:
        """Total signals emitted since start."""
        return self._signals_emitted

    @property
    def rules_evaluated(self) -> int:
        """Total rule evaluations since start."""
        return self._rules_evaluated

    def start(self) -> None:
        """Start the engine by subscribing to INDICATOR_UPDATE events."""
        if self._started:
            return

        self._event_bus.subscribe(EventType.INDICATOR_UPDATE, self._on_indicator_update)
        self._started = True
        logger.info(f"RuleEngine started with {len(self._registry)} rules")

    def stop(self) -> None:
        """Stop the engine."""
        self._started = False
        logger.info("RuleEngine stopped")

    def clear_cooldowns(self) -> int:
        """
        Clear expired cooldowns. Returns count cleared.

        Should be called periodically to prevent memory growth.
        """
        with self._lock:
            now = now_local()  # Use timezone-aware timestamp
            expired = []

            for signal_id, (last_time, cooldown_seconds) in self._last_triggered.items():
                # Handle both timezone-aware and naive datetimes
                try:
                    elapsed = (now - last_time).total_seconds()
                except TypeError:
                    # Naive datetime - force expire
                    elapsed = cooldown_seconds + 1

                if elapsed > cooldown_seconds:
                    expired.append(signal_id)

            for signal_id in expired:
                del self._last_triggered[signal_id]

            return len(expired)

    def _on_indicator_update(self, payload: Any) -> None:
        """Handle INDICATOR_UPDATE event (sync entry point)."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._evaluate_rules_async(payload))
        except RuntimeError:
            # No event loop running - run synchronously
            asyncio.run(self._evaluate_rules_async(payload))

    async def _evaluate_rules_async(self, payload: Any) -> None:
        """Evaluate rules for an indicator update."""
        event = self._coerce_indicator_update(payload)
        if event is None:
            return

        rules = self._registry.get_rules_for_indicator(event.indicator)
        if not rules:
            return

        curr_state = event.state or {}
        prev_state = event.previous_state

        for rule in rules:
            self._rules_evaluated += 1

            # Skip disabled rules
            if not rule.enabled:
                continue

            # Check timeframe filter
            if event.timeframe not in rule.timeframes:
                continue

            # Evaluate condition
            try:
                triggered = rule.check_condition(prev_state, curr_state)
            except Exception as e:
                logger.error(f"Rule {rule.name} check_condition failed: {e}")
                continue

            if not triggered:
                continue

            # Check cooldown using signal_id (category:indicator:symbol:timeframe)
            if self._is_in_cooldown(rule, event):
                logger.debug(
                    f"Rule {rule.name} in cooldown for {event.symbol}/{event.timeframe}"
                )
                continue

            # Build and emit signal
            signal = self._build_signal(rule, event, curr_state, prev_state)
            self._emit_signal(signal)

    def _is_in_cooldown(self, rule: SignalRule, event: IndicatorUpdateEvent) -> bool:
        """
        Check if signal is in cooldown window.

        Cooldowns are tracked per signal_id (category:indicator:symbol:timeframe),
        not per rule name. This allows the same rule to trigger independently
        on different symbols/timeframes.

        If not in cooldown, updates last_triggered time.
        """
        with self._lock:
            # Build signal_id matching _build_signal format
            signal_id = (
                f"{rule.category.value}:{rule.indicator}:{event.symbol}:{event.timeframe}"
            )
            cooldown_entry = self._last_triggered.get(signal_id)

            if cooldown_entry is not None:
                last_time, cooldown_seconds = cooldown_entry
                elapsed = (event.timestamp - last_time).total_seconds()
                if elapsed < cooldown_seconds:
                    return True

            # Not in cooldown - record trigger with rule's cooldown duration
            self._last_triggered[signal_id] = (event.timestamp, rule.cooldown_seconds)
            return False

    def _build_signal(
        self,
        rule: SignalRule,
        event: IndicatorUpdateEvent,
        curr_state: Dict[str, Any],
        prev_state: Optional[Dict[str, Any]],
    ) -> TradingSignal:
        """Build TradingSignal from rule and indicator state."""
        # Extract value
        value = curr_state.get("value")
        if value is None:
            value = event.value

        # Extract threshold from rule config if applicable
        threshold = rule.condition_config.get("threshold")

        # Format message
        message = rule.format_message(
            symbol=event.symbol,
            value=value,
            threshold=threshold,
        )

        # Calculate cooldown expiry
        cooldown_until = None
        if rule.cooldown_seconds > 0:
            cooldown_until = event.timestamp + timedelta(seconds=rule.cooldown_seconds)

        # Build unique signal ID
        signal_id = (
            f"{rule.category.value}:{rule.indicator}:{event.symbol}:{event.timeframe}"
        )

        return TradingSignal(
            signal_id=signal_id,
            symbol=event.symbol,
            category=rule.category,
            indicator=rule.indicator,
            direction=rule.direction,
            strength=rule.strength,
            priority=rule.priority,
            timeframe=event.timeframe,
            trigger_rule=rule.name,
            current_value=value if value is not None else 0.0,
            threshold=threshold,
            previous_value=event.previous_value,
            timestamp=event.timestamp,
            cooldown_until=cooldown_until,
            message=message,
            metadata={
                "state": curr_state,
                "previous_state": prev_state,
            },
        )

    def _emit_signal(self, signal: TradingSignal) -> None:
        """Publish TRADING_SIGNAL event wrapped in TradingSignalEvent."""
        try:
            # Wrap TradingSignal in TradingSignalEvent for consistent payloads
            event = TradingSignalEvent.from_signal(signal, source="rule_engine")
            self._event_bus.publish(EventType.TRADING_SIGNAL, event)
            self._signals_emitted += 1
            logger.info(f"Signal emitted: {signal}")
        except Exception as e:
            logger.error(f"Failed to emit signal {signal.signal_id}: {e}")

    @staticmethod
    def _coerce_indicator_update(payload: Any) -> Optional[IndicatorUpdateEvent]:
        """Coerce payload to IndicatorUpdateEvent."""
        if isinstance(payload, IndicatorUpdateEvent):
            return payload

        if isinstance(payload, dict):
            try:
                return IndicatorUpdateEvent.from_dict(payload)
            except Exception:
                return None

        return None
