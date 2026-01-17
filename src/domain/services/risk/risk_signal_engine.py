"""
Risk Signal Engine - Orchestrates all risk rule evaluation layers.

Coordinates:
- Layer 1: Portfolio hard limits (RuleEngine)
- Layer 2a: Position-level rules (PositionRiskAnalyzer)
- Layer 2b: Strategy-level rules (StrategyRiskAnalyzer)
- Layer 2c: Correlation/sector risk (CorrelationAnalyzer)
- Layer 3: VIX regime (MarketAlertDetector)
- Layer 4: Event risk (EventRiskDetector)

Applies debounce/cooldown filtering via RiskSignalManager.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.domain.exceptions import RecoverableError
from src.domain.services.correlation_analyzer import CorrelationAnalyzer
from src.domain.services.event_risk_detector import EventRiskDetector
from src.domain.services.position_risk_analyzer import PositionRiskAnalyzer
from src.domain.services.strategy_detector import StrategyDetector
from src.domain.services.strategy_risk_analyzer import StrategyRiskAnalyzer
from src.models.position_risk import PositionRisk
from src.models.risk_signal import RiskSignal
from src.models.risk_snapshot import RiskSnapshot
from src.utils.logging_setup import get_logger

from .risk_signal_manager import RiskSignalManager
from .rule_engine import RuleEngine

logger = get_logger(__name__)


class RiskSignalEngine:
    """
    Orchestrates all risk rule evaluation layers.

    Runs multi-layer risk checks and returns filtered, deduplicated signals.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        rule_engine: RuleEngine,
        signal_manager: RiskSignalManager,
    ):
        """
        Initialize risk signal engine.

        Args:
            config: Configuration dictionary
            rule_engine: Existing portfolio rule engine (Layer 1)
            signal_manager: Signal debounce/cooldown manager
        """
        self.config = config
        self.rule_engine = rule_engine
        self.signal_manager = signal_manager

        # Initialize analyzers
        self.position_analyzer = PositionRiskAnalyzer(config)
        self.strategy_detector = StrategyDetector()
        self.strategy_analyzer = StrategyRiskAnalyzer(config)
        self.correlation_analyzer = CorrelationAnalyzer(config)
        self.event_detector = EventRiskDetector(config)

        # Statistics
        self._stats = {
            "total_evaluations": 0,
            "raw_signals": 0,
            "filtered_signals": 0,
            "layer1_signals": 0,
            "layer2_signals": 0,
            # Note: layer3_signals reserved for future VIX regime integration
            # Currently handled separately by MarketAlertDetector
            "layer4_signals": 0,
        }

        logger.info("RiskSignalEngine initialized with all analyzers")

    def evaluate(self, snapshot: RiskSnapshot) -> List[RiskSignal]:
        """
        Run all risk checks and return filtered signals.

        Args:
            snapshot: Current risk snapshot

        Returns:
            List of RiskSignal objects ready for dashboard display
        """
        self._stats["total_evaluations"] += 1
        raw_signals = []

        # Use position_risks from snapshot - single source of truth for all calculations
        # This contains pre-calculated P&L, Greeks, market data from RiskEngine
        position_risks = snapshot.position_risks

        # Layer 1: Portfolio hard limits (existing RuleEngine)
        try:
            breaches = self.rule_engine.evaluate(snapshot)
            layer1_signals = [RiskSignal.from_breach(breach, layer=1) for breach in breaches]
            raw_signals.extend(layer1_signals)
            self._stats["layer1_signals"] += len(layer1_signals)
        except RecoverableError as e:
            logger.warning(f"Recoverable error in Layer 1 (RuleEngine): {e}")

        # Layer 2a: Position-level rules (uses pre-calculated PositionRisk)
        try:
            position_signals = self._check_position_rules(position_risks)
            raw_signals.extend(position_signals)
            self._stats["layer2_signals"] += len(position_signals)
        except RecoverableError as e:
            logger.warning(f"Recoverable error in Layer 2a (Position rules): {e}")

        # Layer 2b: Strategy-level rules (uses pre-calculated PositionRisk)
        try:
            strategy_signals = self._check_strategy_rules(position_risks)
            raw_signals.extend(strategy_signals)
            self._stats["layer2_signals"] += len(strategy_signals)
        except RecoverableError as e:
            logger.warning(f"Recoverable error in Layer 2b (Strategy rules): {e}")

        # Layer 2c: Correlation & sector concentration
        try:
            correlation_signals = self.correlation_analyzer.check(snapshot)
            raw_signals.extend(correlation_signals)
            self._stats["layer2_signals"] += len(correlation_signals)
        except RecoverableError as e:
            logger.warning(f"Recoverable error in Layer 2c (Correlation): {e}")

        # Layer 3: VIX regime (handled by MarketAlertDetector separately)
        # Could integrate here in future

        # Layer 4: Event risk
        try:
            event_signals = self.event_detector.check(snapshot)
            raw_signals.extend(event_signals)
            self._stats["layer4_signals"] += len(event_signals)
        except RecoverableError as e:
            logger.warning(f"Recoverable error in Layer 4 (Event risk): {e}")

        self._stats["raw_signals"] = len(raw_signals)

        # Filter through debounce/cooldown
        # Note: process() is pure logic (datetime/dict ops) - failures are bugs, not recoverable errors
        filtered_signals = []
        for signal in raw_signals:
            result = self.signal_manager.process(signal)
            filtered_signals.extend(result)

        self._stats["filtered_signals"] = len(filtered_signals)

        # Only log when there are signals (reduces log volume 90%+)
        if filtered_signals:
            logger.info(f"Risk signals: {len(raw_signals)} raw → {len(filtered_signals)} filtered")

        return filtered_signals

    def _check_position_rules(
        self,
        position_risks: List[PositionRisk],
    ) -> List[RiskSignal]:
        """Check all positions for position-level rules."""
        signals = []
        for pos_risk in position_risks:
            signals.extend(self.position_analyzer.check(pos_risk))
        return signals

    def _check_strategy_rules(
        self,
        position_risks: List[PositionRisk],
    ) -> List[RiskSignal]:
        """Detect strategies and check strategy-specific rules."""
        if not position_risks:
            return []

        position_risk_map = {pr.symbol: pr for pr in position_risks}
        positions = [pr.position for pr in position_risks]
        strategies = self.strategy_detector.detect(positions)

        signals = []
        for strategy in strategies:
            signals.extend(self.strategy_analyzer.check(strategy, position_risk_map))
        return signals

    def get_stats(self) -> Dict[str, Any]:
        """
        Get evaluation statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            **self._stats,
            "signal_manager_stats": self.signal_manager.get_stats(),
            "strategy_detector_stats": self.strategy_detector.get_stats(),
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self._stats = {
            "total_evaluations": 0,
            "raw_signals": 0,
            "filtered_signals": 0,
            "layer1_signals": 0,
            "layer2_signals": 0,
            # Note: layer3_signals reserved for future VIX regime integration
            "layer4_signals": 0,
        }
        self.signal_manager.reset_stats()
        logger.info("RiskSignalEngine statistics reset")

    def __repr__(self) -> str:
        """Debug representation."""
        return (
            f"RiskSignalEngine(evaluations={self._stats['total_evaluations']}, "
            f"filtered_signals={self._stats['filtered_signals']})"
        )
