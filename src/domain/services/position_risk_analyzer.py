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
from ...models.market_data import MarketData
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
        """
        Initialize position risk analyzer.

        Args:
            config: Configuration dictionary with position_rules section
        """
        self.config = config
        position_rules = config.get("risk_signals", {}).get("position_rules", {})

        # Thresholds from config (with defaults)
        self.stop_loss_pct = position_rules.get("stop_loss_pct", 0.60)  # -60%
        self.take_profit_pct = position_rules.get("take_profit_pct", 1.00)  # +100%
        self.trailing_stop_drawdown = position_rules.get("trailing_stop_drawdown", 0.30)  # 30%
        self.dte_exit_ratio = position_rules.get("dte_exit_ratio", 0.20)  # 20%
        self.short_r_multiple = position_rules.get("short_r_multiple", 1.5)  # 1.5x

        logger.info(
            f"PositionRiskAnalyzer initialized: stop_loss={self.stop_loss_pct:.0%}, "
            f"take_profit={self.take_profit_pct:.0%}, "
            f"trailing_stop={self.trailing_stop_drawdown:.0%}"
        )

    def check(
        self,
        position: Position,
        market_data: Optional[MarketData],
    ) -> List[RiskSignal]:
        """
        Check position for risk conditions.

        Args:
            position: Position to analyze
            market_data: Current market data for the position

        Returns:
            List of risk signals (empty if no issues)
        """
        signals = []

        # Skip if no market data
        if market_data is None:
            return signals

        # Skip if no avg_price (can't calculate PnL)
        if position.avg_price is None or position.avg_price == 0:
            return signals

        # Calculate current PnL %
        current_price = market_data.effective_mid()
        if current_price is None or current_price == 0:
            return signals

        pnl_pct = self._calculate_pnl_pct(position, current_price)

        # Check stop loss
        stop_loss_signal = self._check_stop_loss(position, pnl_pct, current_price)
        if stop_loss_signal:
            signals.append(stop_loss_signal)

        # Check take profit
        take_profit_signal = self._check_take_profit(position, pnl_pct, current_price)
        if take_profit_signal:
            signals.append(take_profit_signal)

        # Check trailing stop
        trailing_signal = self._check_trailing_stop(position, pnl_pct, current_price)
        if trailing_signal:
            signals.append(trailing_signal)

        # Check DTE for options
        if position.asset_type == AssetType.OPTION:
            dte_signal = self._check_dte(position)
            if dte_signal:
                signals.append(dte_signal)

        return signals

    def _calculate_pnl_pct(self, position: Position, current_price: float) -> float:
        """
        Calculate PnL percentage.

        For long positions: (current - entry) / entry
        For short positions: (entry - current) / entry
        """
        entry_price = position.avg_price

        if position.quantity > 0:
            # Long position
            return (current_price - entry_price) / entry_price
        else:
            # Short position
            return (entry_price - current_price) / entry_price

    def _check_stop_loss(
        self,
        position: Position,
        pnl_pct: float,
        current_price: float,
    ) -> Optional[RiskSignal]:
        """
        Check if stop loss threshold breached.

        Returns CRITICAL signal if position loss exceeds threshold.
        """
        # Stop loss only applies to losing positions
        if pnl_pct >= 0:
            return None

        # Check if loss exceeds threshold
        loss_threshold = -self.stop_loss_pct
        if pnl_pct <= loss_threshold:
            breach_pct = abs((pnl_pct - loss_threshold) / loss_threshold) * 100

            return RiskSignal(
                signal_id=f"POSITION:{position.symbol}:Stop_Loss",
                timestamp=datetime.now(),
                level=SignalLevel.POSITION,
                severity=SignalSeverity.CRITICAL,
                symbol=position.symbol,
                trigger_rule="Stop_Loss_Hit",
                current_value=pnl_pct * 100,  # Convert to percentage
                threshold=loss_threshold * 100,
                breach_pct=breach_pct,
                suggested_action=SuggestedAction.CLOSE,
                action_details=(
                    f"Position down {pnl_pct*100:.1f}% "
                    f"(stop loss: {loss_threshold*100:.0f}%). Close immediately."
                ),
                layer=2,
                metadata={
                    "entry_price": position.avg_price,
                    "current_price": current_price,
                    "quantity": position.quantity,
                },
            )

        return None

    def _check_take_profit(
        self,
        position: Position,
        pnl_pct: float,
        current_price: float,
    ) -> Optional[RiskSignal]:
        """
        Check if take profit threshold reached.

        Returns WARNING signal suggesting partial exit at +100%.
        """
        # Take profit only applies to winning positions
        if pnl_pct <= 0:
            return None

        # Check if profit exceeds threshold
        if pnl_pct >= self.take_profit_pct:
            breach_pct = ((pnl_pct - self.take_profit_pct) / self.take_profit_pct) * 100

            return RiskSignal(
                signal_id=f"POSITION:{position.symbol}:Take_Profit",
                timestamp=datetime.now(),
                level=SignalLevel.POSITION,
                severity=SignalSeverity.WARNING,
                symbol=position.symbol,
                trigger_rule="Take_Profit_Hit",
                current_value=pnl_pct * 100,
                threshold=self.take_profit_pct * 100,
                breach_pct=breach_pct,
                suggested_action=SuggestedAction.REDUCE,
                action_details=(
                    f"Position up {pnl_pct*100:.1f}% "
                    f"(take profit: {self.take_profit_pct*100:.0f}%). "
                    f"Consider reducing 50% to lock gains."
                ),
                layer=2,
                metadata={
                    "entry_price": position.avg_price,
                    "current_price": current_price,
                    "quantity": position.quantity,
                },
            )

        return None

    def _check_trailing_stop(
        self,
        position: Position,
        pnl_pct: float,
        current_price: float,
    ) -> Optional[RiskSignal]:
        """
        Check trailing stop from peak profit.

        If position hit peak profit, alert if current profit drops by more than
        trailing_stop_drawdown from peak.
        """
        # Need max_profit_reached to calculate trailing stop
        if position.max_profit_reached is None:
            return None

        # Only alert if we're still profitable
        if pnl_pct <= 0:
            return None

        # Calculate drawdown from peak
        drawdown_from_peak = (position.max_profit_reached - pnl_pct) / position.max_profit_reached

        if drawdown_from_peak >= self.trailing_stop_drawdown:
            breach_pct = (
                (drawdown_from_peak - self.trailing_stop_drawdown) /
                self.trailing_stop_drawdown * 100
            )

            return RiskSignal(
                signal_id=f"POSITION:{position.symbol}:Trailing_Stop",
                timestamp=datetime.now(),
                level=SignalLevel.POSITION,
                severity=SignalSeverity.WARNING,
                symbol=position.symbol,
                trigger_rule="Trailing_Stop_Hit",
                current_value=drawdown_from_peak * 100,
                threshold=self.trailing_stop_drawdown * 100,
                breach_pct=breach_pct,
                suggested_action=SuggestedAction.CLOSE,
                action_details=(
                    f"Position dropped {drawdown_from_peak*100:.1f}% from peak "
                    f"(peak: {position.max_profit_reached*100:.1f}%, "
                    f"current: {pnl_pct*100:.1f}%). Consider closing to protect gains."
                ),
                layer=2,
                metadata={
                    "entry_price": position.avg_price,
                    "current_price": current_price,
                    "peak_profit_pct": position.max_profit_reached * 100,
                    "current_profit_pct": pnl_pct * 100,
                },
            )

        return None

    def _check_dte(self, position: Position) -> Optional[RiskSignal]:
        """
        Check days to expiry for time-based exits.

        Alert if DTE < 20% of initial holding period (indicating time decay acceleration).
        """
        dte = position.days_to_expiry()
        if dte is None:
            return None

        # Alert for options with low DTE
        # Different thresholds based on option type
        if position.quantity > 0:
            # Long options - warn at 7 DTE for potential roll
            dte_threshold = 7
            if dte <= dte_threshold:
                return RiskSignal(
                    signal_id=f"POSITION:{position.symbol}:Low_DTE",
                    timestamp=datetime.now(),
                    level=SignalLevel.POSITION,
                    severity=SignalSeverity.INFO,
                    symbol=position.symbol,
                    trigger_rule="Low_DTE",
                    current_value=float(dte),
                    threshold=float(dte_threshold),
                    breach_pct=0.0,
                    suggested_action=SuggestedAction.ROLL,
                    action_details=(
                        f"Long option has {dte} days to expiry. "
                        f"Consider rolling or closing to avoid theta decay."
                    ),
                    layer=2,
                    metadata={
                        "expiry": position.expiry,
                        "dte": dte,
                    },
                )
        else:
            # Short options - warn at 3 DTE to avoid assignment risk
            dte_threshold = 3
            if dte <= dte_threshold:
                return RiskSignal(
                    signal_id=f"POSITION:{position.symbol}:Low_DTE_Short",
                    timestamp=datetime.now(),
                    level=SignalLevel.POSITION,
                    severity=SignalSeverity.WARNING,
                    symbol=position.symbol,
                    trigger_rule="Low_DTE_Short_Assignment_Risk",
                    current_value=float(dte),
                    threshold=float(dte_threshold),
                    breach_pct=0.0,
                    suggested_action=SuggestedAction.CLOSE,
                    action_details=(
                        f"Short option has {dte} days to expiry. "
                        f"Consider closing to avoid assignment risk."
                    ),
                    layer=2,
                    metadata={
                        "expiry": position.expiry,
                        "dte": dte,
                    },
                )

        return None

    def update_max_profit(self, position: Position, current_pnl_pct: float) -> bool:
        """
        Update position's max profit watermark.

        Args:
            position: Position to update
            current_pnl_pct: Current PnL percentage

        Returns:
            True if max profit was updated
        """
        if position.max_profit_reached is None:
            position.max_profit_reached = current_pnl_pct
            return True

        if current_pnl_pct > position.max_profit_reached:
            position.max_profit_reached = current_pnl_pct
            return True

        return False

    def __repr__(self) -> str:
        """Debug representation."""
        return (
            f"PositionRiskAnalyzer(stop_loss={self.stop_loss_pct:.0%}, "
            f"take_profit={self.take_profit_pct:.0%}, "
            f"trailing_stop={self.trailing_stop_drawdown:.0%})"
        )
