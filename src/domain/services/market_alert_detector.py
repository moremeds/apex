"""
Market Alert Detector - Detects market-wide events and alerts.

This service monitors market-wide conditions and generates alerts for:
- VIX spikes
- Market drops
- High volatility
- Volume surges
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any
from datetime import datetime

from .risk.threshold import Threshold, ThresholdDirection
from ...utils.logging_setup import get_logger


logger = get_logger(__name__)


class MarketAlertDetector:
    """
    Market alert detector for system-wide market events.

    Monitors market conditions and generates alerts based on configurable thresholds.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize market alert detector.

        Args:
            config: Configuration dict with alert thresholds:
                - vix_warning_threshold: VIX level for WARNING (default: 25)
                - vix_critical_threshold: VIX level for CRITICAL (default: 35)
                - market_drop_warning: Daily drop % for WARNING (default: -2.0)
                - market_drop_critical: Daily drop % for CRITICAL (default: -3.0)
                - vix_spike_pct: VIX % change for spike alert (default: 15.0)
        """
        self.config = config or {}

        # VIX thresholds (using Threshold helper)
        self.vix_warning = self.config.get("vix_warning_threshold", 25.0)
        self.vix_critical = self.config.get("vix_critical_threshold", 35.0)
        self.vix_spike_pct = self.config.get("vix_spike_pct", 15.0)
        self.vix_threshold = Threshold(
            warning=self.vix_warning,
            critical=self.vix_critical,
            direction=ThresholdDirection.ABOVE,
        )

        # Market drop thresholds (using Threshold helper - direction BELOW)
        self.drop_warning = self.config.get("market_drop_warning", -2.0)
        self.drop_critical = self.config.get("market_drop_critical", -3.0)
        self.drop_threshold = Threshold(
            warning=self.drop_warning,
            critical=self.drop_critical,
            direction=ThresholdDirection.BELOW,
        )

        # Volatility thresholds (using Threshold helper)
        self.vol_threshold = Threshold(
            warning=30.0,
            critical=40.0,
            direction=ThresholdDirection.ABOVE,
        )

        # Track previous VIX for spike detection
        self._prev_vix: Optional[float] = None

    def detect_alerts(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect market alerts based on current market data.

        Args:
            market_data: Dictionary with market indicators:
                - vix: Current VIX value
                - vix_prev_close: Yesterday's VIX close (optional)
                - spy_change_pct: SPY daily % change
                - qqq_change_pct: QQQ daily % change
                - spy_realized_vol: SPY realized volatility
                - timestamp: Data timestamp

        Returns:
            List of alert dicts with keys: type, message, severity
        """
        alerts = []

        # Check VIX levels
        vix = market_data.get("vix")
        vix_prev_close = market_data.get("vix_prev_close")
        if vix is not None:
            alerts.extend(self._check_vix_alerts(vix, vix_prev_close))

        # Check market drops
        spy_change = market_data.get("spy_change_pct")
        if spy_change is not None:
            alerts.extend(self._check_market_drop_alerts(spy_change, "SPY"))

        qqq_change = market_data.get("qqq_change_pct")
        if qqq_change is not None:
            alerts.extend(self._check_market_drop_alerts(qqq_change, "QQQ"))

        # Check realized volatility
        spy_vol = market_data.get("spy_realized_vol")
        if spy_vol is not None:
            alerts.extend(self._check_volatility_alerts(spy_vol))

        logger.info(f"Market alert detection complete: {len(alerts)} alerts")
        return alerts

    def _check_vix_alerts(self, vix: float, prev_close: Optional[float] = None) -> List[Dict[str, Any]]:
        """Check VIX-related alerts using Threshold helper."""
        alerts = []

        # Check absolute VIX level using Threshold
        severity_str = self.vix_threshold.check(vix)
        if severity_str:
            alert_type = "VIX_CRITICAL" if severity_str == "CRITICAL" else "VIX_ELEVATED"
            threshold = self.vix_critical if severity_str == "CRITICAL" else self.vix_warning
            alerts.append({
                "type": alert_type,
                "message": f"VIX at {vix:.1f} ({severity_str.lower()} threshold: {threshold})",
                "severity": severity_str
            })

        # Check VIX spike (percentage change)
        baseline = self._prev_vix or prev_close
        if baseline and baseline != 0:
            vix_change_pct = ((vix - baseline) / baseline) * 100
            if vix_change_pct >= self.vix_spike_pct:
                severity = "CRITICAL" if severity_str == "CRITICAL" else "WARNING"
                alerts.append({
                    "type": "VIX_SPIKE",
                    "message": f"VIX jumped {vix_change_pct:.1f}% to {vix:.1f}",
                    "severity": severity
                })

        # Update previous VIX
        self._prev_vix = vix

        return alerts

    def _check_market_drop_alerts(self, change_pct: float, symbol: str) -> List[Dict[str, Any]]:
        """Check for market drop alerts using Threshold helper."""
        alerts = []

        severity_str = self.drop_threshold.check(change_pct)
        if severity_str:
            msg_suffix = "(critical threshold)" if severity_str == "CRITICAL" else "intraday"
            alerts.append({
                "type": "MARKET_DROP",
                "message": f"{symbol} down {abs(change_pct):.1f}% {msg_suffix}",
                "severity": severity_str
            })

        return alerts

    def _check_volatility_alerts(self, realized_vol: float) -> List[Dict[str, Any]]:
        """Check for high volatility alerts using Threshold helper."""
        alerts = []

        severity_str = self.vol_threshold.check(realized_vol)
        if severity_str == "CRITICAL":
            alerts.append({
                "type": "HIGH_VOLATILITY",
                "message": f"SPY realized vol at {realized_vol:.1f}% (elevated)",
                "severity": "WARNING"  # Downgrade to WARNING (vol is less urgent)
            })
        elif severity_str == "WARNING":
            alerts.append({
                "type": "VOLATILITY",
                "message": f"SPY realized vol at {realized_vol:.1f}% (above average)",
                "severity": "INFO"
            })

        return alerts

    def reset_state(self):
        """Reset internal state (useful for testing or daily resets)."""
        self._prev_vix = None
        logger.info("Market alert detector state reset")
