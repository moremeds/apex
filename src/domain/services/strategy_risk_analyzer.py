"""
Strategy Risk Analyzer - Strategy-specific risk rules.

Implements risk checks for multi-leg strategies:
- Diagonal delta flip (critical risk)
- Credit spread R-multiple stops
- Calendar spread IV crush detection
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from ...models.position_risk import PositionRisk
from ...models.risk_signal import (
    RiskSignal,
    SignalLevel,
    SignalSeverity,
    SuggestedAction,
)
from ...utils.logging_setup import get_logger
from .strategy_detector import DetectedStrategy

logger = get_logger(__name__)


class StrategyRiskAnalyzer:
    """
    Analyzes detected strategies for strategy-specific risk conditions.

    Rules:
    - Diagonal delta flip: Short leg delta > long leg delta (critical)
    - Credit spread stop: Loss > R-multiple × premium (1.5x-2x)
    - Calendar spread: IV crush > 30% on long leg
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize strategy risk analyzer.

        Args:
            config: Configuration dictionary with strategy_rules section
        """
        self.config = config
        strategy_rules = config.get("risk_signals", {}).get("strategy_rules", {})

        # Thresholds
        self.credit_spread_r_multiple = strategy_rules.get("credit_spread_r_multiple", 1.5)
        self.diagonal_delta_flip_warning = strategy_rules.get("diagonal_delta_flip_warning", True)
        self.calendar_iv_crush_threshold = strategy_rules.get("calendar_iv_crush_threshold", 0.30)

        logger.info(
            f"StrategyRiskAnalyzer initialized: "
            f"credit_r_multiple={self.credit_spread_r_multiple}, "
            f"iv_crush_threshold={self.calendar_iv_crush_threshold:.0%}"
        )

    def check(
        self,
        strategy: DetectedStrategy,
        position_risk_map: Dict[str, PositionRisk],
    ) -> List[RiskSignal]:
        """
        Check strategy for risk conditions.

        Args:
            strategy: Detected strategy to analyze
            position_risk_map: Map of symbol -> PositionRisk (pre-calculated by RiskEngine)

        Returns:
            List of risk signals
        """
        signals: List[RiskSignal] = []

        # Route to specific strategy checker
        if strategy.strategy_type == "DIAGONAL_SPREAD":
            signals.extend(self._check_diagonal_spread(strategy, position_risk_map))

        elif "CREDIT" in strategy.strategy_type:
            signals.extend(self._check_credit_spread(strategy, position_risk_map))

        elif strategy.strategy_type == "CALENDAR_SPREAD":
            signals.extend(self._check_calendar_spread(strategy, position_risk_map))

        elif strategy.strategy_type == "IRON_CONDOR":
            signals.extend(self._check_iron_condor(strategy, position_risk_map))

        return signals

    def _check_diagonal_spread(
        self,
        strategy: DetectedStrategy,
        position_risk_map: Dict[str, PositionRisk],
    ) -> List[RiskSignal]:
        """
        Check diagonal spread for delta flip.

        Critical risk: Short leg delta > long leg delta
        This indicates the short leg is now ITM and exposed to assignment risk.
        """
        signals: List[RiskSignal] = []

        if not self.diagonal_delta_flip_warning:
            return signals

        # Get long and short legs
        long_leg = next((p for p in strategy.positions if p.quantity > 0), None)
        short_leg = next((p for p in strategy.positions if p.quantity < 0), None)

        if not long_leg or not short_leg:
            return signals

        # Get pre-calculated PositionRisk
        long_pr = position_risk_map.get(long_leg.symbol)
        short_pr = position_risk_map.get(short_leg.symbol)

        if not long_pr or not short_pr:
            return signals

        # Check delta flip using pre-calculated delta from RiskEngine
        # Note: PositionRisk.delta is position-level (qty*multiplier applied), so use per-share
        long_delta = (
            abs(long_pr.delta / (abs(long_leg.quantity) * long_leg.multiplier))
            if long_leg.quantity != 0
            else 0.0
        )
        short_delta = (
            abs(short_pr.delta / (abs(short_leg.quantity) * short_leg.multiplier))
            if short_leg.quantity != 0
            else 0.0
        )

        if short_delta > long_delta:
            # Delta flip detected - critical risk
            signals.append(
                RiskSignal(
                    signal_id=f"STRATEGY:{strategy.underlying}:Diagonal_Delta_Flip",
                    timestamp=datetime.now(),
                    level=SignalLevel.STRATEGY,
                    severity=SignalSeverity.CRITICAL,
                    symbol=strategy.underlying,
                    strategy_type=strategy.strategy_type,
                    trigger_rule="Diagonal_Delta_Flip",
                    current_value=short_delta,
                    threshold=long_delta,
                    breach_pct=(
                        ((short_delta - long_delta) / long_delta * 100) if long_delta > 0 else 0
                    ),
                    suggested_action=SuggestedAction.CLOSE,
                    action_details=(
                        f"Diagonal spread delta flip: short leg delta ({short_delta:.2f}) "
                        f"> long leg delta ({long_delta:.2f}). "
                        f"Short leg now ITM - high assignment risk. Close or roll immediately."
                    ),
                    layer=2,
                    metadata={
                        "long_strike": long_leg.strike,
                        "short_strike": short_leg.strike,
                        "long_expiry": long_leg.expiry,
                        "short_expiry": short_leg.expiry,
                        "long_delta": long_delta,
                        "short_delta": short_delta,
                    },
                )
            )

        return signals

    def _check_credit_spread(
        self,
        strategy: DetectedStrategy,
        position_risk_map: Dict[str, PositionRisk],
    ) -> List[RiskSignal]:
        """
        Check credit spread for R-multiple stop.

        Alert if loss > R-multiple × premium collected.
        Example: Collected $100 premium, stop at -$150 loss (1.5x R-multiple).
        """
        signals: List[RiskSignal] = []

        # Calculate total PnL using pre-calculated values from RiskEngine
        total_pnl = 0.0
        for pos in strategy.positions:
            pr = position_risk_map.get(pos.symbol)
            if not pr or not pr.has_market_data:
                return signals  # Need all position risks

            # Use pre-calculated unrealized_pnl (single source of truth)
            total_pnl += pr.unrealized_pnl

        # For credit spreads, premium is typically negative (credit received)
        # If current PnL is negative (loss), check R-multiple
        if total_pnl < 0:
            # Estimate premium collected (very simplified - should track actual premium)
            # For now, use max risk approximation
            max_risk = self._estimate_spread_max_risk(strategy)

            if max_risk and abs(total_pnl) > max_risk * self.credit_spread_r_multiple:
                breach_pct = (abs(total_pnl) / (max_risk * self.credit_spread_r_multiple) - 1) * 100

                signals.append(
                    RiskSignal(
                        signal_id=f"STRATEGY:{strategy.underlying}:Credit_Spread_Stop",
                        timestamp=datetime.now(),
                        level=SignalLevel.STRATEGY,
                        severity=SignalSeverity.CRITICAL,
                        symbol=strategy.underlying,
                        strategy_type=strategy.strategy_type,
                        trigger_rule="Credit_Spread_R_Multiple_Stop",
                        current_value=abs(total_pnl),
                        threshold=max_risk * self.credit_spread_r_multiple,
                        breach_pct=breach_pct,
                        suggested_action=SuggestedAction.CLOSE,
                        action_details=(
                            f"Credit spread loss ${abs(total_pnl):.0f} exceeds "
                            f"{self.credit_spread_r_multiple}x R-multiple stop "
                            f"(${max_risk * self.credit_spread_r_multiple:.0f}). "
                            f"Close or roll to limit losses."
                        ),
                        layer=2,
                        metadata={
                            "current_pnl": total_pnl,
                            "max_risk": max_risk,
                            "r_multiple": self.credit_spread_r_multiple,
                        },
                    )
                )

        return signals

    def _check_calendar_spread(
        self,
        strategy: DetectedStrategy,
        position_risk_map: Dict[str, PositionRisk],
    ) -> List[RiskSignal]:
        """
        Check calendar spread for IV crush on long leg.

        Calendar spreads profit from vega - if IV drops significantly on
        the long leg, the strategy may no longer be profitable.
        """
        signals: List[RiskSignal] = []

        # Get long leg (typically longer expiry)
        long_leg = next((p for p in strategy.positions if p.quantity > 0), None)
        if not long_leg:
            return signals

        pr = position_risk_map.get(long_leg.symbol)
        if not pr or not pr.iv:
            return signals

        # For IV crush detection, we'd need historical IV
        # For now, check if vega is negative (losing value from IV drop)
        if pr.vega and pr.vega < -10:  # Significant negative vega
            signals.append(
                RiskSignal(
                    signal_id=f"STRATEGY:{strategy.underlying}:Calendar_IV_Crush",
                    timestamp=datetime.now(),
                    level=SignalLevel.STRATEGY,
                    severity=SignalSeverity.WARNING,
                    symbol=strategy.underlying,
                    strategy_type=strategy.strategy_type,
                    trigger_rule="Calendar_IV_Crush",
                    current_value=pr.vega,
                    threshold=-10.0,
                    breach_pct=0.0,
                    suggested_action=SuggestedAction.MONITOR,
                    action_details=(
                        f"Calendar spread showing negative vega ({pr.vega:.2f}), "
                        f"indicating potential IV crush. Monitor closely."
                    ),
                    layer=2,
                    metadata={
                        "long_strike": long_leg.strike,
                        "long_expiry": long_leg.expiry,
                        "implied_volatility": pr.iv,
                        "vega": pr.vega,
                    },
                )
            )

        return signals

    def _check_iron_condor(
        self,
        strategy: DetectedStrategy,
        position_risk_map: Dict[str, PositionRisk],
    ) -> List[RiskSignal]:
        """
        Check iron condor for early profit take or stop loss.

        Iron condors are credit spreads - check if we should close early
        to lock in profits (e.g., at 50% max profit).
        """
        signals: List[RiskSignal] = []

        # Calculate total PnL using pre-calculated values from RiskEngine
        total_pnl = 0.0
        for pos in strategy.positions:
            pr = position_risk_map.get(pos.symbol)
            if not pr or not pr.has_market_data:
                return signals

            # Use pre-calculated unrealized_pnl (single source of truth)
            total_pnl += pr.unrealized_pnl

        # For iron condors, we collected premium (negative cost basis)
        # If current PnL is positive and > 50% of max profit, consider closing
        max_risk = self._estimate_spread_max_risk(strategy)

        if max_risk and total_pnl > max_risk * 0.5:  # 50% of max profit
            signals.append(
                RiskSignal(
                    signal_id=f"STRATEGY:{strategy.underlying}:Iron_Condor_Early_Exit",
                    timestamp=datetime.now(),
                    level=SignalLevel.STRATEGY,
                    severity=SignalSeverity.INFO,
                    symbol=strategy.underlying,
                    strategy_type=strategy.strategy_type,
                    trigger_rule="Iron_Condor_Early_Profit",
                    current_value=total_pnl,
                    threshold=max_risk * 0.5,
                    breach_pct=0.0,
                    suggested_action=SuggestedAction.REDUCE,
                    action_details=(
                        f"Iron condor at ${total_pnl:.0f} profit "
                        f"(~{total_pnl/max_risk*100:.0f}% of max profit). "
                        f"Consider closing early to lock in gains."
                    ),
                    layer=2,
                    metadata={
                        "current_pnl": total_pnl,
                        "max_profit": max_risk,
                    },
                )
            )

        return signals

    def _estimate_spread_max_risk(self, strategy: DetectedStrategy) -> Optional[float]:
        """
        Estimate max risk for a spread strategy.

        For vertical spreads: width × quantity × multiplier
        """
        if len(strategy.positions) < 2:
            return None

        # Get strike width
        strikes = [p.strike for p in strategy.positions if p.strike]
        if len(strikes) < 2:
            return None

        width = abs(max(strikes) - min(strikes))

        # Get quantity and multiplier
        quantity = abs(strategy.positions[0].quantity)
        multiplier = strategy.positions[0].multiplier

        return width * quantity * multiplier

    def __repr__(self) -> str:
        """Debug representation."""
        return (
            f"StrategyRiskAnalyzer(r_multiple={self.credit_spread_r_multiple}, "
            f"iv_crush={self.calendar_iv_crush_threshold:.0%})"
        )
