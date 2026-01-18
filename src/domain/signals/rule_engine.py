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
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List, Optional, Protocol, Tuple

from src.domain.events.domain_events import IndicatorUpdateEvent, TradingSignalEvent
from src.domain.events.event_types import EventType
from src.utils.logging_setup import get_logger
from src.utils.timezone import now_local

from .models import SignalRule, TradingSignal

if TYPE_CHECKING:
    from src.infrastructure.observability import SignalMetrics

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

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RuleRegistry":
        """
        Create a registry from YAML configuration.

        Expected config structure:
            rules:
              rule_name:
                enabled: true
                indicator: rsi
                category: momentum
                direction: buy
                strength: 70
                priority: high
                condition:
                  type: state_change
                  field: zone
                  from: [oversold]
                  to: [neutral]
                timeframes: [1h, 4h, 1d]
                cooldown_seconds: 3600
                message: "{symbol} RSI signal"

        Args:
            config: Configuration dict with 'rules' key

        Returns:
            Populated RuleRegistry
        """
        from .models import (
            ConditionType,
            SignalCategory,
            SignalDirection,
            SignalPriority,
            SignalRule,
        )

        registry = cls()
        rules_config = config.get("rules", {})

        for rule_name, settings in rules_config.items():
            if not settings.get("enabled", True):
                logger.debug(f"Skipping disabled rule: {rule_name}")
                continue

            try:
                # Parse condition
                condition = settings.get("condition", {})
                condition_type_str = condition.get("type", "custom").upper()
                condition_type = ConditionType[condition_type_str]

                # Build condition_config from non-type fields
                condition_config = {k: v for k, v in condition.items() if k != "type"}

                # Parse enums
                category = SignalCategory[settings["category"].upper()]
                direction = SignalDirection[settings["direction"].upper()]
                priority = SignalPriority[settings.get("priority", "medium").upper()]

                # Parse timeframes as tuple
                timeframes = tuple(settings.get("timeframes", ["1h"]))

                rule = SignalRule(
                    name=rule_name,
                    indicator=settings["indicator"],
                    category=category,
                    direction=direction,
                    strength=settings.get("strength", 50),
                    priority=priority,
                    condition_type=condition_type,
                    condition_config=condition_config,
                    timeframes=timeframes,
                    cooldown_seconds=settings.get("cooldown_seconds", 3600),
                    enabled=True,
                    message_template=settings.get("message", ""),
                )
                registry.add_rule(rule)
                logger.debug(f"Loaded rule: {rule_name} for indicator={rule.indicator}")

            except KeyError as e:
                logger.error(f"Invalid rule config '{rule_name}': missing {e}")
            except Exception as e:
                logger.error(f"Failed to load rule '{rule_name}': {e}")

        logger.info(f"Loaded {len(registry)} rules from config")
        return registry

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
        signal_metrics: Optional["SignalMetrics"] = None,
        trace_mode: bool = False,
    ) -> None:
        """
        Initialize the rule engine.

        Args:
            event_bus: Event bus for subscriptions and publishing
            registry: RuleRegistry with rules to evaluate (creates empty if None)
            signal_metrics: Metrics collector for instrumentation
            trace_mode: If True, enable verbose logging of rule evaluations for debugging
        """
        self._event_bus = event_bus
        self._registry = registry or RuleRegistry()
        self._metrics = signal_metrics
        self._trace_mode = trace_mode

        # Per-signal cooldown tracking: signal_id -> (last_triggered_time, cooldown_seconds)
        # signal_id format: "{category}:{indicator}:{symbol}:{timeframe}"
        self._last_triggered: Dict[str, Tuple[datetime, int]] = {}
        self._lock = RLock()

        self._started = False
        self._signals_emitted = 0
        self._rules_evaluated = 0

        # Evaluation history (ring buffer) - only populated when trace_mode enabled
        # Stores recent rule evaluations for introspection/debugging
        self._evaluation_history: Deque[Dict[str, Any]] = deque(maxlen=200)

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

    @property
    def trace_mode(self) -> bool:
        """Whether trace mode is enabled."""
        return self._trace_mode

    @trace_mode.setter
    def trace_mode(self, value: bool) -> None:
        """Enable or disable trace mode at runtime."""
        self._trace_mode = value
        logger.info(f"RuleEngine trace_mode {'enabled' if value else 'disabled'}")

    def start(self) -> None:
        """Start the engine by subscribing to INDICATOR_UPDATE events."""
        if self._started:
            return

        self._event_bus.subscribe(EventType.INDICATOR_UPDATE, self._on_indicator_update)
        self._started = True

        # Count rules by indicator for logging
        all_rules = self._registry.get_all_rules()
        indicators_with_rules = set(r.indicator for r in all_rules)
        short_tf_rules = [r.name for r in all_rules if "1m" in r.timeframes or "5m" in r.timeframes]

        logger.info(
            f"RuleEngine started: {len(all_rules)} rules across {len(indicators_with_rules)} indicators, "
            f"{len(short_tf_rules)} short-timeframe rules",
        )
        if short_tf_rules:
            logger.info(f"Short-timeframe rules: {short_tf_rules}")

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
            logger.warning(f"RuleEngine: Failed to coerce payload type={type(payload).__name__}")
            return

        rules = self._registry.get_rules_for_indicator(event.indicator)
        if not rules:
            # No rules for this indicator - this is expected for many indicators
            return

        curr_state = event.state or {}
        prev_state = event.previous_state

        start_time = time.perf_counter()
        rules_checked = 0
        rules_matched_timeframe = 0
        rules_triggered = 0

        for rule in rules:
            self._rules_evaluated += 1
            rules_checked += 1

            # Record rule evaluation metric
            if self._metrics:
                self._metrics.record_rule_evaluated(rule.name)

            # Skip disabled rules
            if not rule.enabled:
                continue

            # Check timeframe filter
            if event.timeframe not in rule.timeframes:
                continue

            rules_matched_timeframe += 1

            # Evaluate condition
            condition_met = False
            try:
                condition_met = rule.check_condition(prev_state, curr_state)
            except Exception as e:
                if self._metrics:
                    self._metrics.record_error("rule_engine", "check_condition")
                logger.error(
                    f"Rule check_condition failed: {rule.name} error={e!r}",
                    exc_info=True,
                )
                self._record_evaluation(rule, event, False, False, False, error=True)
                continue

            # Trace mode: log detailed rule evaluation values
            if self._trace_mode:
                field = rule.condition_config.get("field", "value")
                threshold = rule.condition_config.get("threshold")
                prev_val = prev_state.get(field) if prev_state else None
                curr_val = curr_state.get(field) if curr_state else None
                logger.info(
                    f"TRACE Rule {rule.name}: "
                    f"prev={prev_val}, curr={curr_val}, threshold={threshold} -> "
                    f"{'TRIGGERED' if condition_met else 'NO_MATCH'}"
                )

            if not condition_met:
                self._record_evaluation(rule, event, False, False, False)
                continue

            rules_triggered += 1

            # Check cooldown using signal_id (category:indicator:symbol:timeframe)
            blocked_by_cooldown = self._is_in_cooldown(rule, event)
            if blocked_by_cooldown:
                if self._metrics:
                    self._metrics.record_signal_blocked(rule.name, "cooldown")
                logger.debug(
                    f"Signal blocked by cooldown: {rule.name} symbol={event.symbol} tf={event.timeframe}",
                )
                self._record_evaluation(rule, event, False, True, True)
                continue

            # Build and emit signal
            signal = self._build_signal(rule, event, curr_state, prev_state)
            self._emit_signal(signal)
            self._record_evaluation(rule, event, True, False, True)

        # Log rule evaluation summary for debugging
        if rules_matched_timeframe > 0:
            logger.debug(
                f"RuleEngine evaluated: indicator={event.indicator} symbol={event.symbol} "
                f"tf={event.timeframe} rules_checked={rules_checked} "
                f"matched_tf={rules_matched_timeframe} triggered={rules_triggered}"
            )

        # Record batch evaluation latency
        duration_ms = (time.perf_counter() - start_time) * 1000
        if self._metrics:
            self._metrics.record_rule_evaluation_latency(duration_ms)

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
            signal_id = f"{rule.category.value}:{rule.indicator}:{event.symbol}:{event.timeframe}"
            cooldown_entry = self._last_triggered.get(signal_id)

            if cooldown_entry is not None:
                last_time, cooldown_seconds = cooldown_entry
                elapsed = (event.timestamp - last_time).total_seconds()
                if elapsed < cooldown_seconds:
                    return True

            # Not in cooldown - record trigger with rule's cooldown duration
            self._last_triggered[signal_id] = (event.timestamp, rule.cooldown_seconds)

            # Update cooldown entries gauge
            if self._metrics:
                self._metrics.set_cooldown_entries(len(self._last_triggered))

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
        signal_id = f"{rule.category.value}:{rule.indicator}:{event.symbol}:{event.timeframe}"

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

            # Record signal emission metric
            if self._metrics:
                self._metrics.record_signal_emitted(signal.trigger_rule, signal.direction.value)

            # Structured info log for signal emission (most important event)
            logger.info(
                "Trading signal emitted",
                extra={
                    "signal_id": signal.signal_id,
                    "symbol": signal.symbol,
                    "timeframe": signal.timeframe,
                    "direction": signal.direction.value,
                    "indicator": signal.indicator,
                    "rule": signal.trigger_rule,
                    "value": signal.current_value,
                    "threshold": signal.threshold,
                    "strength": signal.strength,
                    "priority": signal.priority,
                },
            )

        except Exception as e:
            if self._metrics:
                self._metrics.record_error("rule_engine", "emit_signal")
            logger.error(
                "Failed to emit signal",
                extra={"signal_id": signal.signal_id, "error": str(e)},
            )

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

    # -------------------------------------------------------------------------
    # Introspection Methods (for SignalIntrospectionPort)
    # -------------------------------------------------------------------------

    def _record_evaluation(
        self,
        rule: SignalRule,
        event: IndicatorUpdateEvent,
        triggered: bool,
        blocked_by_cooldown: bool,
        condition_met: bool,
        error: bool = False,
    ) -> None:
        """
        Record a rule evaluation for introspection.

        Only records when trace_mode is enabled (zero overhead when off).
        This enables debugging and TUI visibility into rule evaluations.
        """
        if not self._trace_mode:
            return

        reason = self._build_evaluation_reason(triggered, blocked_by_cooldown, condition_met, error)

        entry = {
            "rule_name": rule.name,
            "indicator": rule.indicator,
            "category": rule.category.value,
            "symbol": event.symbol,
            "timeframe": event.timeframe,
            "triggered": triggered,
            "blocked_by_cooldown": blocked_by_cooldown,
            "condition_met": condition_met,
            "error": error,
            "reason": reason,
            "timestamp": event.timestamp,
        }

        # Thread-safe append (same lock as reads)
        with self._lock:
            self._evaluation_history.append(entry)

    @staticmethod
    def _build_evaluation_reason(
        triggered: bool,
        blocked_by_cooldown: bool,
        condition_met: bool,
        error: bool = False,
    ) -> str:
        """Build human-readable reason for evaluation outcome."""
        if triggered:
            return "signal emitted"
        if error:
            return "evaluation error"
        if blocked_by_cooldown:
            return "blocked by cooldown"
        if not condition_met:
            return "condition not met"
        return "unknown"

    @staticmethod
    def _compute_remaining_seconds(
        last_time: datetime, cooldown_seconds: float, now: datetime
    ) -> Optional[int]:
        """
        Compute remaining cooldown seconds.

        Returns None if cooldown expired or timestamp comparison fails (naive datetime).
        """
        try:
            elapsed = (now - last_time).total_seconds()
        except TypeError:
            return None

        remaining = cooldown_seconds - elapsed
        if remaining <= 0:
            return None

        return int(remaining)

    @staticmethod
    def _parse_cooldown_key(key: str) -> Tuple[str, str, str, str]:
        """Parse cooldown key into (category, indicator, symbol, timeframe)."""
        parts = key.split(":", 3)
        while len(parts) < 4:
            parts.append("")
        return parts[0], parts[1], parts[2], parts[3]

    def get_evaluation_history(
        self,
        limit: int = 50,
        triggered_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get recent rule evaluation history.

        Returns empty list if trace_mode was not enabled during evaluations.

        Args:
            limit: Maximum number of evaluations to return.
            triggered_only: If True, only return evaluations that triggered.

        Returns:
            List of evaluation dicts, most recent first.
        """
        with self._lock:
            history = list(self._evaluation_history)

        # Most recent first
        history = history[-limit:][::-1]

        if triggered_only:
            history = [e for e in history if e["triggered"]]

        return history

    def get_cooldown_status(
        self,
        category: str,
        indicator: str,
        symbol: str,
        timeframe: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get cooldown status for a specific category:indicator:symbol:timeframe.

        Args:
            category: Signal category (e.g., "momentum", "trend").
            indicator: Indicator name (e.g., "rsi").
            symbol: Trading symbol.
            timeframe: Bar timeframe.

        Returns:
            Cooldown info dict if actively in cooldown, None otherwise.
        """
        key = f"{category}:{indicator}:{symbol}:{timeframe}"

        with self._lock:
            entry = self._last_triggered.get(key)
            if entry is None:
                return None

            last_time, cooldown_seconds = entry
            remaining_seconds = self._compute_remaining_seconds(
                last_time, cooldown_seconds, now_local()
            )
            if remaining_seconds is None:
                return None

            return {
                "category": category,
                "indicator": indicator,
                "symbol": symbol,
                "timeframe": timeframe,
                "last_triggered": last_time,
                "cooldown_seconds": cooldown_seconds,
                "remaining_seconds": remaining_seconds,
                "active": True,
            }

    def get_all_cooldowns(self) -> List[Dict[str, Any]]:
        """
        Get all active cooldowns.

        Returns:
            List of active cooldown dicts.
        """
        now = now_local()
        result: List[Dict[str, Any]] = []

        with self._lock:
            for key, (last_time, cooldown_seconds) in self._last_triggered.items():
                remaining_seconds = self._compute_remaining_seconds(
                    last_time, cooldown_seconds, now
                )
                if remaining_seconds is None:
                    continue

                category, indicator, symbol, timeframe = self._parse_cooldown_key(key)
                result.append(
                    {
                        "key": key,
                        "category": category,
                        "indicator": indicator,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "last_triggered": last_time,
                        "cooldown_seconds": cooldown_seconds,
                        "remaining_seconds": remaining_seconds,
                    }
                )

        return result
