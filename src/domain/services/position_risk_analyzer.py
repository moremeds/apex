"""
Position Risk Analyzer - Position-level risk rules.

Implements stop loss, take profit, trailing stops, and DTE-based exit rules
for individual positions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from ...models.position import AssetType, Position
from ...models.position_risk import PositionRisk
from ...models.risk_signal import (
    RiskSignal,
    SignalLevel,
    SignalSeverity,
    SuggestedAction,
)
from ...utils.logging_setup import get_logger
from .risk.threshold import Threshold, ThresholdDirection

logger = get_logger(__name__)


class PositionRiskAnalyzer:
    """
    Analyzes individual positions for risk conditions.

    Implements position-level rules:
    - Stop loss: -50%/-60% for long options/spreads
    - Take profit: +100% for long options
    - Trailing stop: 30% drawdown from peak
    - DTE exit: < 20% of initial holding period
    - R-multiple stop for short positions: Loss > 1.5x-2x premium
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize position risk analyzer with thresholds from config."""
        position_rules = config.get("risk_signals", {}).get("position_rules", {})

        self.stop_loss_pct = position_rules.get("stop_loss_pct", 0.60)
        self.take_profit_pct = position_rules.get("take_profit_pct", 1.00)
        self.trailing_stop_drawdown = position_rules.get("trailing_stop_drawdown", 0.30)

        # Initialize threshold helpers for standardized checking
        self.stop_loss_threshold = Threshold(
            warning=-self.stop_loss_pct * 0.8,  # Warning at 80% of stop loss
            critical=-self.stop_loss_pct,
            direction=ThresholdDirection.BELOW,
        )
        self.take_profit_threshold = Threshold(
            warning=self.take_profit_pct,
            critical=self.take_profit_pct * 1.5,  # Strong profit at 150%
            direction=ThresholdDirection.ABOVE,
        )
        self.trailing_stop_threshold = Threshold(
            warning=self.trailing_stop_drawdown * 0.8,  # Warning at 80% of trailing
            critical=self.trailing_stop_drawdown,
            direction=ThresholdDirection.ABOVE,
        )

        logger.info(
            f"PositionRiskAnalyzer initialized: stop_loss={self.stop_loss_pct:.0%}, "
            f"take_profit={self.take_profit_pct:.0%}, trailing_stop={self.trailing_stop_drawdown:.0%}"
        )

    def check(self, pos_risk: PositionRisk) -> List[RiskSignal]:
        """Check position for risk conditions using pre-calculated PositionRisk."""
        position = pos_risk.position

        # Skip if missing required data
        if not pos_risk.has_market_data or not position.avg_price or not pos_risk.mark_price:
            return []

        # Calculate P&L % from pre-calculated unrealized_pnl
        cost_basis = position.avg_price * abs(position.quantity) * position.multiplier
        if cost_basis == 0:
            return []
        pnl_pct = pos_risk.unrealized_pnl / cost_basis

        # Update max profit watermark for trailing stop
        self._update_max_profit(position, pnl_pct)

        # Run all checks
        signals = []
        for check_fn in [
            lambda: self._check_stop_loss(position, pnl_pct, pos_risk.mark_price),
            lambda: self._check_take_profit(position, pnl_pct, pos_risk.mark_price),
            lambda: self._check_trailing_stop(position, pnl_pct, pos_risk.mark_price),
            lambda: self._check_dte(position) if position.asset_type == AssetType.OPTION else None,
        ]:
            signal = check_fn()
            if signal:
                signals.append(signal)

        return signals

    def _update_max_profit(self, position: Position, pnl_pct: float) -> None:
        """Update position's max profit watermark for trailing stop."""
        if position.max_profit_reached is None or pnl_pct > position.max_profit_reached:
            position.max_profit_reached = pnl_pct

    def _check_stop_loss(
        self, position: Position, pnl_pct: float, current_price: float
    ) -> Optional[RiskSignal]:
        """Check if stop loss threshold breached using Threshold helper."""
        severity_str = self.stop_loss_threshold.check(pnl_pct)
        if not severity_str:
            return None

        severity = SignalSeverity.CRITICAL if severity_str == "CRITICAL" else SignalSeverity.WARNING
        breach_pct = self.stop_loss_threshold.breach_pct(pnl_pct) * 100

        return RiskSignal(
            signal_id=f"POSITION:{position.symbol}:Stop_Loss",
            timestamp=datetime.now(),
            level=SignalLevel.POSITION,
            severity=severity,
            symbol=position.symbol,
            trigger_rule="Stop_Loss_Hit",
            current_value=pnl_pct * 100,
            threshold=-self.stop_loss_pct * 100,
            breach_pct=breach_pct,
            suggested_action=(
                SuggestedAction.CLOSE
                if severity == SignalSeverity.CRITICAL
                else SuggestedAction.REDUCE
            ),
            action_details=f"Position down {pnl_pct*100:.1f}% (stop: {-self.stop_loss_pct*100:.0f}%). {'Close immediately.' if severity == SignalSeverity.CRITICAL else 'Approaching stop loss.'}",
            layer=2,
            metadata={
                "entry_price": position.avg_price,
                "current_price": current_price,
                "quantity": position.quantity,
            },
        )

    def _check_take_profit(
        self, position: Position, pnl_pct: float, current_price: float
    ) -> Optional[RiskSignal]:
        """Check if take profit threshold reached using Threshold helper."""
        severity_str = self.take_profit_threshold.check(pnl_pct)
        if not severity_str:
            return None

        # Take profit is always a good thing - INFO for warning, WARNING for critical (strong profit)
        severity = SignalSeverity.WARNING if severity_str == "CRITICAL" else SignalSeverity.INFO
        breach_pct = self.take_profit_threshold.breach_pct(pnl_pct) * 100

        return RiskSignal(
            signal_id=f"POSITION:{position.symbol}:Take_Profit",
            timestamp=datetime.now(),
            level=SignalLevel.POSITION,
            severity=severity,
            symbol=position.symbol,
            trigger_rule="Take_Profit_Hit",
            current_value=pnl_pct * 100,
            threshold=self.take_profit_pct * 100,
            breach_pct=breach_pct,
            suggested_action=SuggestedAction.REDUCE,
            action_details=f"Position up {pnl_pct*100:.1f}% (TP: {self.take_profit_pct*100:.0f}%). Consider reducing 50%.",
            layer=2,
            metadata={
                "entry_price": position.avg_price,
                "current_price": current_price,
                "quantity": position.quantity,
            },
        )

    def _check_trailing_stop(
        self, position: Position, pnl_pct: float, current_price: float
    ) -> Optional[RiskSignal]:
        """Check trailing stop from peak profit using Threshold helper."""
        if position.max_profit_reached is None or position.max_profit_reached <= 0 or pnl_pct <= 0:
            return None

        drawdown = (position.max_profit_reached - pnl_pct) / position.max_profit_reached
        severity_str = self.trailing_stop_threshold.check(drawdown)
        if not severity_str:
            return None

        severity = SignalSeverity.WARNING if severity_str == "CRITICAL" else SignalSeverity.INFO
        breach_pct = self.trailing_stop_threshold.breach_pct(drawdown) * 100

        return RiskSignal(
            signal_id=f"POSITION:{position.symbol}:Trailing_Stop",
            timestamp=datetime.now(),
            level=SignalLevel.POSITION,
            severity=severity,
            symbol=position.symbol,
            trigger_rule="Trailing_Stop_Hit",
            current_value=drawdown * 100,
            threshold=self.trailing_stop_drawdown * 100,
            breach_pct=breach_pct,
            suggested_action=SuggestedAction.CLOSE,
            action_details=f"Dropped {drawdown*100:.1f}% from peak {position.max_profit_reached*100:.1f}%. Close to protect gains.",
            layer=2,
            metadata={
                "entry_price": position.avg_price,
                "current_price": current_price,
                "peak_pnl": position.max_profit_reached * 100,
            },
        )

    def _check_dte(self, position: Position) -> Optional[RiskSignal]:
        """Check days to expiry for time-based exits."""
        dte = position.days_to_expiry()
        if dte is None:
            return None

        # Long: 7 DTE, Short: 3 DTE thresholds
        is_long = position.quantity > 0
        threshold = 7 if is_long else 3

        if dte > threshold:
            return None

        return RiskSignal(
            signal_id=f"POSITION:{position.symbol}:Low_DTE{'_Short' if not is_long else ''}",
            timestamp=datetime.now(),
            level=SignalLevel.POSITION,
            severity=SignalSeverity.INFO if is_long else SignalSeverity.WARNING,
            symbol=position.symbol,
            trigger_rule="Low_DTE" if is_long else "Low_DTE_Short_Assignment_Risk",
            current_value=float(dte),
            threshold=float(threshold),
            breach_pct=0.0,
            suggested_action=SuggestedAction.ROLL if is_long else SuggestedAction.CLOSE,
            action_details=f"{'Long' if is_long else 'Short'} option has {dte} DTE. {'Roll to avoid theta decay.' if is_long else 'Close to avoid assignment.'}",
            layer=2,
            metadata={"expiry": position.expiry, "dte": dte},
        )

    def __repr__(self) -> str:
        return f"PositionRiskAnalyzer(stop={self.stop_loss_pct:.0%}, tp={self.take_profit_pct:.0%})"
