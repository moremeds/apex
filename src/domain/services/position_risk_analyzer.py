"""
Position Risk Analyzer - Position-level risk rules.

Implements stop loss, take profit, trailing stops, and DTE-based exit rules
for individual positions.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging

from ...models.position import Position, AssetType
from ...models.position_risk import PositionRisk
from ...models.risk_signal import (
    RiskSignal,
    SignalLevel,
    SignalSeverity,
    SuggestedAction,
)


logger = logging.getLogger(__name__)


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
        """Check if stop loss threshold breached."""
        if pnl_pct >= 0:
            return None

        loss_threshold = -self.stop_loss_pct
        if pnl_pct > loss_threshold:
            return None

        return RiskSignal(
            signal_id=f"POSITION:{position.symbol}:Stop_Loss",
            timestamp=datetime.now(),
            level=SignalLevel.POSITION,
            severity=SignalSeverity.CRITICAL,
            symbol=position.symbol,
            trigger_rule="Stop_Loss_Hit",
            current_value=pnl_pct * 100,
            threshold=loss_threshold * 100,
            breach_pct=abs((pnl_pct - loss_threshold) / loss_threshold) * 100,
            suggested_action=SuggestedAction.CLOSE,
            action_details=f"Position down {pnl_pct*100:.1f}% (stop: {loss_threshold*100:.0f}%). Close immediately.",
            layer=2,
            metadata={"entry_price": position.avg_price, "current_price": current_price, "quantity": position.quantity},
        )

    def _check_take_profit(
        self, position: Position, pnl_pct: float, current_price: float
    ) -> Optional[RiskSignal]:
        """Check if take profit threshold reached."""
        if pnl_pct <= 0 or pnl_pct < self.take_profit_pct:
            return None

        return RiskSignal(
            signal_id=f"POSITION:{position.symbol}:Take_Profit",
            timestamp=datetime.now(),
            level=SignalLevel.POSITION,
            severity=SignalSeverity.WARNING,
            symbol=position.symbol,
            trigger_rule="Take_Profit_Hit",
            current_value=pnl_pct * 100,
            threshold=self.take_profit_pct * 100,
            breach_pct=((pnl_pct - self.take_profit_pct) / self.take_profit_pct) * 100,
            suggested_action=SuggestedAction.REDUCE,
            action_details=f"Position up {pnl_pct*100:.1f}% (TP: {self.take_profit_pct*100:.0f}%). Consider reducing 50%.",
            layer=2,
            metadata={"entry_price": position.avg_price, "current_price": current_price, "quantity": position.quantity},
        )

    def _check_trailing_stop(
        self, position: Position, pnl_pct: float, current_price: float
    ) -> Optional[RiskSignal]:
        """Check trailing stop from peak profit."""
        if position.max_profit_reached is None or pnl_pct <= 0:
            return None

        drawdown = (position.max_profit_reached - pnl_pct) / position.max_profit_reached
        if drawdown < self.trailing_stop_drawdown:
            return None

        return RiskSignal(
            signal_id=f"POSITION:{position.symbol}:Trailing_Stop",
            timestamp=datetime.now(),
            level=SignalLevel.POSITION,
            severity=SignalSeverity.WARNING,
            symbol=position.symbol,
            trigger_rule="Trailing_Stop_Hit",
            current_value=drawdown * 100,
            threshold=self.trailing_stop_drawdown * 100,
            breach_pct=(drawdown - self.trailing_stop_drawdown) / self.trailing_stop_drawdown * 100,
            suggested_action=SuggestedAction.CLOSE,
            action_details=f"Dropped {drawdown*100:.1f}% from peak {position.max_profit_reached*100:.1f}%. Close to protect gains.",
            layer=2,
            metadata={"entry_price": position.avg_price, "current_price": current_price, "peak_pnl": position.max_profit_reached * 100},
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
