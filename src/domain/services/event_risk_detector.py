"""
Event Risk Detector - Earnings and scheduled event risk detection.

Detects:
- Earnings dates (T-3, T-1 warnings)
- FOMC announcements
- CPI/Economic data releases
- Ex-dividend dates

MVP: Manual earnings calendar from config
Future: API integration with Earnings Whispers / Yahoo Finance
"""

from __future__ import annotations
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional

from ...models.risk_snapshot import RiskSnapshot
from ...models.position import AssetType
from ...models.risk_signal import (
    RiskSignal,
    SignalLevel,
    SignalSeverity,
    SuggestedAction,
)
from .risk.threshold import Threshold, ThresholdDirection
from ...utils.logging_setup import get_logger


logger = get_logger(__name__)


class EventRiskDetector:
    """
    Detects event risk for positions.

    Features:
    - Earnings calendar warnings (T-3, T-1 days)
    - Short option assignment risk before earnings
    - High gamma positions before events
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize event risk detector.

        Args:
            config: Configuration dictionary with event_risk section
        """
        self.config = config
        event_config = config.get("risk_signals", {}).get("event_risk", {})

        # Configuration
        self.enabled = event_config.get("enabled", True)
        self.earnings_warning_days = event_config.get("earnings_warning_days", 3)
        self.earnings_critical_days = event_config.get("earnings_critical_days", 1)

        # Threshold for days to earnings (BELOW = warning when value is low)
        # Days <= critical_days → CRITICAL, Days <= warning_days → WARNING
        self.earnings_threshold = Threshold(
            warning=float(self.earnings_warning_days),
            critical=float(self.earnings_critical_days),
            direction=ThresholdDirection.BELOW,
        )

        # Load earnings calendar
        self.earnings_calendar = self._load_earnings_calendar(
            event_config.get("upcoming_earnings", {})
        )

        logger.info(
            f"EventRiskDetector initialized: "
            f"enabled={self.enabled}, "
            f"earnings_calendar={len(self.earnings_calendar)} symbols, "
            f"warning_days={self.earnings_warning_days}, "
            f"critical_days={self.earnings_critical_days}"
        )

    def _load_earnings_calendar(
        self,
        earnings_config: Dict[str, str],
    ) -> Dict[str, date]:
        """
        Load earnings calendar from config.

        Args:
            earnings_config: Dict of symbol -> date string (YYYY-MM-DD)

        Returns:
            Dict of symbol -> date object
        """
        calendar = {}
        for symbol, date_str in earnings_config.items():
            try:
                earnings_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                calendar[symbol] = earnings_date
            except ValueError:
                logger.warning(f"Invalid earnings date for {symbol}: {date_str}")
        return calendar

    def check(self, snapshot: RiskSnapshot) -> List[RiskSignal]:
        """
        Check positions for event risk.

        Args:
            snapshot: Risk snapshot with position risks

        Returns:
            List of risk signals
        """
        if not self.enabled:
            return []

        signals = []
        today = date.today()

        # Check each position for earnings risk
        for pos_risk in snapshot.position_risks:
            symbol = pos_risk.underlying or pos_risk.symbol

            # Check if symbol has upcoming earnings
            if symbol in self.earnings_calendar:
                earnings_date = self.earnings_calendar[symbol]
                days_to_earnings = (earnings_date - today).days

                # Check if within warning window
                if 0 <= days_to_earnings <= self.earnings_warning_days:
                    signal = self._create_earnings_signal(
                        pos_risk,
                        symbol,
                        earnings_date,
                        days_to_earnings,
                    )
                    if signal:
                        signals.append(signal)

        return signals

    def _create_earnings_signal(
        self,
        pos_risk,
        symbol: str,
        earnings_date: date,
        days_to_earnings: int,
    ) -> Optional[RiskSignal]:
        """
        Create earnings risk signal based on position type and days to earnings.

        Uses Threshold helper for standardized severity checking.

        Args:
            pos_risk: Position risk data
            symbol: Symbol
            earnings_date: Earnings date
            days_to_earnings: Days until earnings

        Returns:
            RiskSignal or None
        """
        # Determine severity using Threshold helper
        severity_str = self.earnings_threshold.check(float(days_to_earnings))
        if not severity_str:
            return None  # Outside warning window
        severity = SignalSeverity.CRITICAL if severity_str == "CRITICAL" else SignalSeverity.WARNING

        # Determine suggested action based on position type
        # Short options have higher risk (assignment, IV crush)
        is_short = pos_risk.quantity < 0 if hasattr(pos_risk, 'quantity') else False
        has_gamma = abs(pos_risk.gamma or 0.0) > 100  # Significant gamma exposure

        if is_short:
            suggested_action = SuggestedAction.CLOSE
            action_details = (
                f"{symbol} reports earnings in {days_to_earnings} day(s) on "
                f"{earnings_date.strftime('%Y-%m-%d')}. "
                f"Short option position exposed to assignment risk and IV crush. "
                f"Close or roll before earnings."
            )
        elif has_gamma:
            suggested_action = SuggestedAction.HEDGE
            action_details = (
                f"{symbol} reports earnings in {days_to_earnings} day(s) on "
                f"{earnings_date.strftime('%Y-%m-%d')}. "
                f"High gamma position exposed to large price swings. "
                f"Consider hedging or reducing size."
            )
        else:
            suggested_action = SuggestedAction.MONITOR
            action_details = (
                f"{symbol} reports earnings in {days_to_earnings} day(s) on "
                f"{earnings_date.strftime('%Y-%m-%d')}. "
                f"Monitor position closely for volatility."
            )

        return RiskSignal(
            signal_id=f"POSITION:{symbol}:Earnings_Risk",
            timestamp=datetime.now(),
            level=SignalLevel.POSITION,
            severity=severity,
            symbol=symbol,
            trigger_rule="Earnings_Event_Risk",
            current_value=float(days_to_earnings),
            threshold=float(self.earnings_warning_days),
            breach_pct=0.0,
            suggested_action=suggested_action,
            action_details=action_details,
            layer=4,  # Layer 4: Event risk
            metadata={
                "earnings_date": earnings_date.isoformat(),
                "days_to_earnings": days_to_earnings,
                "is_short": is_short,
                "gamma": pos_risk.gamma,
            },
        )

    def update_earnings_date(self, symbol: str, earnings_date: date):
        """
        Update earnings date for a symbol.

        Args:
            symbol: Symbol
            earnings_date: New earnings date
        """
        self.earnings_calendar[symbol] = earnings_date
        logger.info(f"Updated earnings date for {symbol}: {earnings_date}")

    def remove_earnings_date(self, symbol: str):
        """
        Remove earnings date (e.g., after earnings passed).

        Args:
            symbol: Symbol to remove
        """
        if symbol in self.earnings_calendar:
            del self.earnings_calendar[symbol]
            logger.info(f"Removed earnings date for {symbol}")

    def get_upcoming_earnings(self, days: int = 7) -> Dict[str, date]:
        """
        Get earnings in next N days.

        Args:
            days: Number of days to look ahead

        Returns:
            Dict of symbol -> earnings_date for upcoming earnings
        """
        today = date.today()
        upcoming = {}

        for symbol, earnings_date in self.earnings_calendar.items():
            days_to_earnings = (earnings_date - today).days
            if 0 <= days_to_earnings <= days:
                upcoming[symbol] = earnings_date

        return upcoming

    def cleanup_past_earnings(self):
        """
        Remove past earnings dates (housekeeping).

        Should be called periodically to keep calendar clean.
        """
        today = date.today()
        to_remove = []

        for symbol, earnings_date in self.earnings_calendar.items():
            if earnings_date < today:
                to_remove.append(symbol)

        for symbol in to_remove:
            del self.earnings_calendar[symbol]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} past earnings dates")

    def __repr__(self) -> str:
        """Debug representation."""
        return (
            f"EventRiskDetector(enabled={self.enabled}, "
            f"calendar_size={len(self.earnings_calendar)}, "
            f"warning_days={self.earnings_warning_days})"
        )
