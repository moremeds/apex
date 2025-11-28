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
from typing import List, Dict, Any, Optional
import logging

from src.models.risk_snapshot import RiskSnapshot
from src.models.risk_signal import RiskSignal
from src.models.position import Position
from src.models.market_data import MarketData

from .rule_engine import RuleEngine
from .risk_signal_manager import RiskSignalManager
from src.domain.services.position_risk_analyzer import PositionRiskAnalyzer
from src.domain.services.strategy_detector import StrategyDetector
from src.domain.services.strategy_risk_analyzer import StrategyRiskAnalyzer
from src.domain.services.correlation_analyzer import CorrelationAnalyzer
from src.domain.services.event_risk_detector import EventRiskDetector


logger = logging.getLogger(__name__)


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
        position_store=None,
        market_data_store=None,
    ):
        """
        Initialize risk signal engine.

        Args:
            config: Configuration dictionary
            rule_engine: Existing portfolio rule engine (Layer 1)
            signal_manager: Signal debounce/cooldown manager
            position_store: Position store for position lookups
            market_data_store: Market data store for market data lookups
        """
        self.config = config
        self.rule_engine = rule_engine
        self.signal_manager = signal_manager
        self.position_store = position_store
        self.market_data_store = market_data_store

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
            "layer3_signals": 0,
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

        # Layer 1: Portfolio hard limits (existing RuleEngine)
        try:
            breaches = self.rule_engine.evaluate(snapshot)
            layer1_signals = [RiskSignal.from_breach(breach, layer=1) for breach in breaches]
            raw_signals.extend(layer1_signals)
            self._stats["layer1_signals"] += len(layer1_signals)
        except Exception as e:
            logger.error(f"Error in Layer 1 (RuleEngine): {e}", exc_info=True)

        # Layer 2a: Position-level rules
        try:
            position_signals = self._check_position_rules(snapshot)
            raw_signals.extend(position_signals)
            self._stats["layer2_signals"] += len(position_signals)
        except Exception as e:
            logger.error(f"Error in Layer 2a (Position rules): {e}", exc_info=True)

        # Layer 2b: Strategy-level rules
        try:
            strategy_signals = self._check_strategy_rules(snapshot)
            raw_signals.extend(strategy_signals)
            self._stats["layer2_signals"] += len(strategy_signals)
        except Exception as e:
            logger.error(f"Error in Layer 2b (Strategy rules): {e}", exc_info=True)

        # Layer 2c: Correlation & sector concentration
        try:
            correlation_signals = self.correlation_analyzer.check(snapshot)
            raw_signals.extend(correlation_signals)
            self._stats["layer2_signals"] += len(correlation_signals)
        except Exception as e:
            logger.error(f"Error in Layer 2c (Correlation): {e}", exc_info=True)

        # Layer 3: VIX regime (handled by MarketAlertDetector separately)
        # Could integrate here in future

        # Layer 4: Event risk
        try:
            event_signals = self.event_detector.check(snapshot)
            raw_signals.extend(event_signals)
            self._stats["layer4_signals"] += len(event_signals)
        except Exception as e:
            logger.error(f"Error in Layer 4 (Event risk): {e}", exc_info=True)

        self._stats["raw_signals"] = len(raw_signals)

        # Filter through debounce/cooldown
        filtered_signals = []
        for signal in raw_signals:
            try:
                result = self.signal_manager.process(signal)
                filtered_signals.extend(result)
            except Exception as e:
                logger.error(f"Error processing signal {signal.signal_id}: {e}", exc_info=True)

        self._stats["filtered_signals"] = len(filtered_signals)

        logger.info(
            f"Risk evaluation complete: {len(raw_signals)} raw signals → "
            f"{len(filtered_signals)} filtered signals"
        )

        return filtered_signals

    def _check_position_rules(self, snapshot: RiskSnapshot) -> List[RiskSignal]:
        """
        Check all positions for position-level rules.

        Args:
            snapshot: Risk snapshot

        Returns:
            List of position-level risk signals
        """
        signals = []

        # Need position store and market data store
        if not self.position_store or not self.market_data_store:
            logger.debug("Position/market data stores not available, skipping position rules")
            return signals

        # Get all positions
        positions = self.position_store.get_all()

        for position in positions:
            # Get market data
            market_data = self.market_data_store.get(position.symbol)

            # Check position
            pos_signals = self.position_analyzer.check(position, market_data)
            signals.extend(pos_signals)

            # Update max profit watermark
            if market_data:
                current_price = market_data.effective_mid()
                if current_price and position.avg_price:
                    pnl_pct = (current_price - position.avg_price) / position.avg_price
                    self.position_analyzer.update_max_profit(position, pnl_pct)

        return signals

    def _check_strategy_rules(self, snapshot: RiskSnapshot) -> List[RiskSignal]:
        """
        Detect strategies and check strategy-specific rules.

        Args:
            snapshot: Risk snapshot

        Returns:
            List of strategy-level risk signals
        """
        signals = []

        # Need position store
        if not self.position_store:
            logger.debug("Position store not available, skipping strategy rules")
            return signals

        # Get all positions
        positions = self.position_store.get_all()

        # Detect strategies
        strategies = self.strategy_detector.detect(positions)

        # Build market data map
        market_data_map = {}
        if self.market_data_store:
            for position in positions:
                md = self.market_data_store.get(position.symbol)
                if md:
                    market_data_map[position.symbol] = md

        # Check each strategy
        for strategy in strategies:
            strategy_signals = self.strategy_analyzer.check(strategy, market_data_map)
            signals.extend(strategy_signals)

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
            "layer3_signals": 0,
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
