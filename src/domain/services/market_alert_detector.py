"""
Market Alert Detector - Detect market-wide alerts like VIX spikes.

Monitors market indicators and generates alerts based on configurable thresholds.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging


logger = logging.getLogger(__name__)


class MarketAlertDetector:
    """
    Detects market-wide alerts based on indicator thresholds.

    Monitors:
    - VIX level warnings/critical alerts
    - VIX spike detection (% change)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        vix_warning_threshold: float = 25.0,
        vix_critical_threshold: float = 35.0,
        vix_spike_pct: float = 15.0,
    ):
        """
        Initialize market alert detector.

        Args:
            config: Optional config dict with threshold values.
            vix_warning_threshold: VIX level for warning alert (default if not in config).
            vix_critical_threshold: VIX level for critical alert (default if not in config).
            vix_spike_pct: Percentage change threshold for VIX spike alert.
        """
        # Support both config dict and individual parameters
        if config and isinstance(config, dict):
            self.vix_warning_threshold = config.get("vix_warning_threshold", vix_warning_threshold)
            self.vix_critical_threshold = config.get("vix_critical_threshold", vix_critical_threshold)
            self.vix_spike_pct = config.get("vix_spike_pct", vix_spike_pct)
        else:
            self.vix_warning_threshold = vix_warning_threshold
            self.vix_critical_threshold = vix_critical_threshold
            self.vix_spike_pct = vix_spike_pct

    def detect_alerts(self, indicators: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect market alerts based on current indicators.

        Args:
            indicators: Dict with market data, e.g.:
                - "vix": Current VIX level
                - "vix_prev_close": Previous VIX close
                - "timestamp": Data timestamp

        Returns:
            List of alert dicts with "type", "severity", "message", "value".
        """
        alerts: List[Dict[str, Any]] = []

        vix = indicators.get("vix")
        vix_prev_close = indicators.get("vix_prev_close")
        timestamp = indicators.get("timestamp", datetime.now())

        if vix is None:
            return alerts

        # VIX level alerts
        if vix >= self.vix_critical_threshold:
            alerts.append({
                "type": "VIX_CRITICAL",
                "severity": "CRITICAL",
                "message": f"VIX at critical level: {vix:.2f} (threshold: {self.vix_critical_threshold})",
                "value": vix,
                "threshold": self.vix_critical_threshold,
                "timestamp": timestamp,
            })
        elif vix >= self.vix_warning_threshold:
            alerts.append({
                "type": "VIX_WARNING",
                "severity": "WARNING",
                "message": f"VIX elevated: {vix:.2f} (threshold: {self.vix_warning_threshold})",
                "value": vix,
                "threshold": self.vix_warning_threshold,
                "timestamp": timestamp,
            })

        # VIX spike detection
        if vix_prev_close and vix_prev_close > 0:
            pct_change = ((vix - vix_prev_close) / vix_prev_close) * 100
            if pct_change >= self.vix_spike_pct:
                alerts.append({
                    "type": "VIX_SPIKE",
                    "severity": "WARNING",
                    "message": f"VIX spiked {pct_change:.1f}% (from {vix_prev_close:.2f} to {vix:.2f})",
                    "value": pct_change,
                    "threshold": self.vix_spike_pct,
                    "timestamp": timestamp,
                })

        return alerts
