"""
Strategy Risk Analyzer - Strategy-specific risk rules.

Implements risk checks for multi-leg strategies:
- Diagonal delta flip (critical risk)
- Credit spread R-multiple stops
- Calendar spread IV crush detection
"""

from __future__ import annotations
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

from .strategy_detector import DetectedStrategy
from ...models.risk_signal import (
    RiskSignal,
    SignalLevel,
    SignalSeverity,
    SuggestedAction,
)
from ...models.market_data import MarketData


logger = logging.getLogger(__name__)


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
        market_data_map: Dict[str, MarketData],
    ) -> List[RiskSignal]:
        """
        Check strategy for risk conditions.

        Args:
            strategy: Detected strategy to analyze
            market_data_map: Map of symbol -> MarketData

        Returns:
            List of risk signals
        """
        signals = []

        # Route to specific strategy checker
        if strategy.strategy_type == "DIAGONAL_SPREAD":
            signals.extend(self._check_diagonal_spread(strategy, market_data_map))

        elif "CREDIT" in strategy.strategy_type:
            signals.extend(self._check_credit_spread(strategy, market_data_map))

        elif strategy.strategy_type == "CALENDAR_SPREAD":
            signals.extend(self._check_calendar_spread(strategy, market_data_map))

        elif strategy.strategy_type == "IRON_CONDOR":
            signals.extend(self._check_iron_condor(strategy, market_data_map))

        return signals

    def _check_diagonal_spread(
        self,
        strategy: DetectedStrategy,
        market_data_map: Dict[str, MarketData],
    ) -> List[RiskSignal]:
        """
        Check diagonal spread for delta flip.

        Critical risk: Short leg delta > long leg delta
        This indicates the short leg is now ITM and exposed to assignment risk.
        """
        signals = []

        if not self.diagonal_delta_flip_warning:
            return signals

        # Get long and short legs
        long_leg = next((p for p in strategy.positions if p.quantity > 0), None)
        short_leg = next((p for p in strategy.positions if p.quantity < 0), None)

        if not long_leg or not short_leg:
            return signals

        # Get market data
        long_md = market_data_map.get(long_leg.symbol)
        short_md = market_data_map.get(short_leg.symbol)

        if not long_md or not short_md:
            return signals

        # Check delta flip
        long_delta = abs(long_md.delta or 0.0)
        short_delta = abs(short_md.delta or 0.0)

        if short_delta > long_delta:
            # Delta flip detected - critical risk
            signals.append(RiskSignal(
                signal_id=f"STRATEGY:{strategy.underlying}:Diagonal_Delta_Flip",
                timestamp=datetime.now(),
                level=SignalLevel.STRATEGY,
                severity=SignalSeverity.CRITICAL,
                symbol=strategy.underlying,
                strategy_type=strategy.strategy_type,
                trigger_rule="Diagonal_Delta_Flip",
                current_value=short_delta,
                threshold=long_delta,
                breach_pct=((short_delta - long_delta) / long_delta * 100) if long_delta > 0 else 0,
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
            ))

        return signals

    def _check_credit_spread(
        self,
        strategy: DetectedStrategy,
        market_data_map: Dict[str, MarketData],
    ) -> List[RiskSignal]:
        """
        Check credit spread for R-multiple stop.

        Alert if loss > R-multiple × premium collected.
        Example: Collected $100 premium, stop at -$150 loss (1.5x R-multiple).
        """
        signals = []

        # Calculate current PnL
        total_pnl = 0.0
        for pos in strategy.positions:
            md = market_data_map.get(pos.symbol)
            if not md:
                return signals  # Need all market data

            current_price = md.effective_mid()
            if not current_price or not pos.avg_price:
                return signals

            # PnL for this leg
            leg_pnl = (current_price - pos.avg_price) * pos.quantity * pos.multiplier
            total_pnl += leg_pnl

        # For credit spreads, premium is typically negative (credit received)
        # If current PnL is negative (loss), check R-multiple
        if total_pnl < 0:
            # Estimate premium collected (very simplified - should track actual premium)
            # For now, use max risk approximation
            max_risk = self._estimate_spread_max_risk(strategy)

            if max_risk and abs(total_pnl) > max_risk * self.credit_spread_r_multiple:
                breach_pct = (abs(total_pnl) / (max_risk * self.credit_spread_r_multiple) - 1) * 100

                signals.append(RiskSignal(
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
                ))

        return signals

    def _check_calendar_spread(
        self,
        strategy: DetectedStrategy,
        market_data_map: Dict[str, MarketData],
    ) -> List[RiskSignal]:
        """
        Check calendar spread for IV crush on long leg.

        Calendar spreads profit from vega - if IV drops significantly on
        the long leg, the strategy may no longer be profitable.
        """
        signals = []

        # Get long leg (typically longer expiry)
        long_leg = next((p for p in strategy.positions if p.quantity > 0), None)
        if not long_leg:
            return signals

        md = market_data_map.get(long_leg.symbol)
        if not md or not md.implied_volatility:
            return signals

        # For IV crush detection, we'd need historical IV
        # For now, check if vega is negative (losing value from IV drop)
        if md.vega and md.vega < -10:  # Significant negative vega
            signals.append(RiskSignal(
                signal_id=f"STRATEGY:{strategy.underlying}:Calendar_IV_Crush",
                timestamp=datetime.now(),
                level=SignalLevel.STRATEGY,
                severity=SignalSeverity.WARNING,
                symbol=strategy.underlying,
                strategy_type=strategy.strategy_type,
                trigger_rule="Calendar_IV_Crush",
                current_value=md.vega,
                threshold=-10.0,
                breach_pct=0.0,
                suggested_action=SuggestedAction.MONITOR,
                action_details=(
                    f"Calendar spread showing negative vega ({md.vega:.2f}), "
                    f"indicating potential IV crush. Monitor closely."
                ),
                layer=2,
                metadata={
                    "long_strike": long_leg.strike,
                    "long_expiry": long_leg.expiry,
                    "implied_volatility": md.implied_volatility,
                    "vega": md.vega,
                },
            ))

        return signals

    def _check_iron_condor(
        self,
        strategy: DetectedStrategy,
        market_data_map: Dict[str, MarketData],
    ) -> List[RiskSignal]:
        """
        Check iron condor for early profit take or stop loss.

        Iron condors are credit spreads - check if we should close early
        to lock in profits (e.g., at 50% max profit).
        """
        signals = []

        # Calculate current PnL
        total_pnl = 0.0
        for pos in strategy.positions:
            md = market_data_map.get(pos.symbol)
            if not md or not pos.avg_price:
                return signals

            current_price = md.effective_mid()
            if not current_price:
                return signals

            leg_pnl = (current_price - pos.avg_price) * pos.quantity * pos.multiplier
            total_pnl += leg_pnl

        # For iron condors, we collected premium (negative cost basis)
        # If current PnL is positive and > 50% of max profit, consider closing
        max_risk = self._estimate_spread_max_risk(strategy)

        if max_risk and total_pnl > max_risk * 0.5:  # 50% of max profit
            signals.append(RiskSignal(
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
            ))

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
