"""
Risk Alert Logger - Comprehensive logging for all risk alerts and signals.

Provides detailed audit trail for risk alerts including:
- Market alerts (VIX spikes, market drops)
- Risk signals (portfolio/position/strategy level)
- Full context: Greeks, IV, prices, timestamps, and reasons

Logs are written in JSON format for easy analysis and compliance.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional

from src.models.market_data import MarketData
from src.models.position_risk import PositionRisk
from src.models.risk_signal import RiskSignal
from src.models.risk_snapshot import RiskSnapshot
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class AlertContext:
    """
    Full context captured when an alert is triggered.

    Contains all relevant market data, Greeks, and position info
    for audit and analysis purposes.
    """

    # Timestamp
    timestamp: str

    # Alert identification
    alert_type: str  # "MARKET_ALERT" or "RISK_SIGNAL"
    alert_id: str
    severity: str

    # Alert details
    trigger_rule: str
    message: str
    reason: str

    # Values
    current_value: Optional[float] = None
    threshold: Optional[float] = None
    breach_pct: Optional[float] = None

    # Market context
    vix_level: Optional[float] = None
    spy_price: Optional[float] = None
    qqq_price: Optional[float] = None

    # Position context (for position/strategy alerts)
    symbol: Optional[str] = None
    underlying: Optional[str] = None
    underlying_price: Optional[float] = None
    position_quantity: Optional[float] = None
    avg_price: Optional[float] = None
    mark_price: Optional[float] = None

    # Option-specific context
    option_type: Optional[str] = None  # "CALL" or "PUT"
    strike: Optional[float] = None
    expiry: Optional[str] = None
    days_to_expiry: Optional[int] = None

    # Greeks
    iv: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None

    # P&L context
    unrealized_pnl: Optional[float] = None
    daily_pnl: Optional[float] = None
    pnl_pct: Optional[float] = None

    # Portfolio context (for portfolio-level alerts)
    portfolio_delta: Optional[float] = None
    portfolio_gamma: Optional[float] = None
    portfolio_vega: Optional[float] = None
    portfolio_theta: Optional[float] = None
    total_notional: Optional[float] = None
    margin_utilization: Optional[float] = None

    # Suggested action
    suggested_action: Optional[str] = None
    action_details: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, filtering out None values."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                result[key] = value
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class RiskAlertLogger:
    """
    Comprehensive risk alert logger with full context capture.

    Features:
    - Separate log file for risk alerts (risk_alerts_{env}_{date}.log)
    - JSON format for easy parsing and analysis
    - Captures full market/position/Greeks context
    - Daily rotation with retention policy
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        env: str = "dev",
        retention_days: int = 30,
    ):
        """
        Initialize risk alert logger.

        Args:
            log_dir: Directory for log files
            env: Environment name (dev/prod)
            retention_days: Number of days to retain logs
        """
        self.log_dir = Path(log_dir)
        self.env = env
        self.retention_days = retention_days

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Track current date for log file naming
        self._current_date = datetime.now().strftime("%Y-%m-%d")

        # Async logging: QueueListener handles writing in background thread
        # M7: Bounded queue to prevent memory leak under heavy alert load
        self._log_queue: Queue = Queue(maxsize=1000)
        self._queue_listener: Optional[logging.handlers.QueueListener] = None

        # Set up dedicated logger with async queue
        self._logger = self._setup_logger()

        # Cache for market indicators (updated each cycle)
        self._market_cache: Dict[str, Any] = {}

        log_filename = f"risk_alerts_{env}_{self._current_date}.log"
        logger.info(f"RiskAlertLogger initialized: {self.log_dir}/{log_filename}")

    def _setup_logger(self) -> logging.Logger:
        """
        Set up dedicated file logger for risk alerts with async queue.

        Uses QueueHandler + QueueListener pattern to prevent file I/O
        from blocking the event loop. Log records are queued immediately
        and written to disk in a background thread.
        """
        alert_logger = logging.getLogger(f"apex.risk_alerts.{self.env}")
        alert_logger.setLevel(logging.INFO)
        alert_logger.handlers.clear()
        alert_logger.propagate = False

        # File handler with daily rotation and custom namer
        # Base file uses current date in name: risk_alerts_dev_2025-11-28.log
        log_file = self.log_dir / f"risk_alerts_{self.env}_{self._current_date}.log"
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=str(log_file),
            when="midnight",
            interval=1,
            backupCount=self.retention_days,
            encoding="utf-8",
        )

        # Custom namer for rotated files: risk_alerts_dev_2025-11-27.log
        file_handler.namer = self._log_namer
        file_handler.rotator = self._log_rotator

        # Simple formatter - we'll write JSON directly
        file_handler.setFormatter(logging.Formatter("%(message)s"))

        # Async logging setup: QueueHandler -> queue -> QueueListener -> file_handler
        # This ensures log calls return immediately without blocking on file I/O
        queue_handler = logging.handlers.QueueHandler(self._log_queue)
        alert_logger.addHandler(queue_handler)

        # Start QueueListener in background thread to process queue
        self._queue_listener = logging.handlers.QueueListener(
            self._log_queue, file_handler, respect_handler_level=True
        )
        self._queue_listener.start()

        return alert_logger

    def stop(self) -> None:
        """
        Stop the async logging queue listener.

        Should be called during shutdown to ensure all logs are flushed.
        """
        if self._queue_listener:
            self._queue_listener.stop()
            self._queue_listener = None
            logger.debug("RiskAlertLogger queue listener stopped")

    def _log_namer(self, default_name: str) -> str:
        """
        Custom namer for rotated log files.

        Converts: risk_alerts_dev_2025-11-28.log.2025-11-27
        To:       risk_alerts_dev_2025-11-27.log
        """
        import re

        # Extract date from suffix (e.g., .2025-11-27)
        match = re.search(r"\.(\d{4}-\d{2}-\d{2})$", default_name)
        if match:
            date_str = match.group(1)
            # Build new filename with date
            return str(self.log_dir / f"risk_alerts_{self.env}_{date_str}.log")
        return default_name

    def _log_rotator(self, source: str, dest: str):
        """
        Custom rotator that renames the source file to the dated destination.

        After rotation, updates the current log file to use today's date.
        """
        import os
        import shutil

        # Rename old log to dated backup
        if os.path.exists(source):
            shutil.move(source, dest)

        # Update current date and recreate logger for new day
        new_date = datetime.now().strftime("%Y-%m-%d")
        if new_date != self._current_date:
            self._current_date = new_date
            # The handler will create the new file automatically

    def update_market_cache(
        self,
        vix: Optional[float] = None,
        spy_price: Optional[float] = None,
        qqq_price: Optional[float] = None,
        market_data_store=None,
    ):
        """
        Update cached market indicators for context enrichment.

        Call this at the start of each risk evaluation cycle.

        Args:
            vix: Current VIX level
            spy_price: Current SPY price
            qqq_price: Current QQQ price
            market_data_store: Optional market data store for lookups
        """
        if vix is not None:
            self._market_cache["vix"] = vix
        if spy_price is not None:
            self._market_cache["spy_price"] = spy_price
        if qqq_price is not None:
            self._market_cache["qqq_price"] = qqq_price

        # Try to get prices from market data store
        if market_data_store:
            for symbol in ["VIX", "SPY", "QQQ"]:
                md = market_data_store.get(symbol)
                if md:
                    price = md.effective_mid() or md.last
                    if price:
                        self._market_cache[
                            f"{symbol.lower()}_price" if symbol != "VIX" else "vix"
                        ] = price

    def log_market_alert(
        self,
        alert: Dict[str, Any],
        snapshot: Optional[RiskSnapshot] = None,
    ):
        """
        Log a market alert (VIX spike, market drop, etc.) with full context.

        Args:
            alert: Market alert dict with type, message, severity
            snapshot: Optional risk snapshot for portfolio context
        """
        now = datetime.now()

        context = AlertContext(
            timestamp=now.isoformat(),
            alert_type="MARKET_ALERT",
            alert_id=f"MKT_{alert.get('type', 'UNKNOWN')}_{now.strftime('%Y%m%d_%H%M%S')}",
            severity=alert.get("severity", "INFO"),
            trigger_rule=alert.get("type", "UNKNOWN"),
            message=alert.get("message", ""),
            reason=self._extract_reason_from_market_alert(alert),
            current_value=alert.get("value"),
            threshold=alert.get("threshold"),
            vix_level=self._market_cache.get("vix"),
            spy_price=self._market_cache.get("spy_price"),
            qqq_price=self._market_cache.get("qqq_price"),
        )

        # Add portfolio context if available
        if snapshot:
            context.portfolio_delta = snapshot.portfolio_delta
            context.portfolio_gamma = snapshot.portfolio_gamma
            context.portfolio_vega = snapshot.portfolio_vega
            context.portfolio_theta = snapshot.portfolio_theta
            context.total_notional = snapshot.total_gross_notional
            context.margin_utilization = snapshot.margin_utilization

        # Add any extra metadata from alert
        if "timestamp" in alert:
            context.metadata["alert_timestamp"] = str(alert["timestamp"])

        self._write_log(context)

    def log_risk_signal(
        self,
        signal: RiskSignal,
        snapshot: Optional[RiskSnapshot] = None,
        position_risk: Optional[PositionRisk] = None,
        market_data: Optional[MarketData] = None,
    ):
        """
        Log a risk signal with full context.

        Args:
            signal: RiskSignal object
            snapshot: Optional risk snapshot for portfolio context
            position_risk: Optional position risk for position-level signals
            market_data: Optional market data for the symbol
        """
        now = datetime.now()

        context = AlertContext(
            timestamp=now.isoformat(),
            alert_type="RISK_SIGNAL",
            alert_id=signal.signal_id,
            severity=signal.severity.value,
            trigger_rule=signal.trigger_rule,
            message=signal.action_details or str(signal),
            reason=self._extract_reason_from_signal(signal),
            current_value=signal.current_value,
            threshold=signal.threshold,
            breach_pct=signal.breach_pct,
            symbol=signal.symbol,
            suggested_action=signal.suggested_action.value if signal.suggested_action else None,
            action_details=signal.action_details,
            vix_level=self._market_cache.get("vix"),
            spy_price=self._market_cache.get("spy_price"),
            qqq_price=self._market_cache.get("qqq_price"),
        )

        # Add portfolio context if available
        if snapshot:
            context.portfolio_delta = snapshot.portfolio_delta
            context.portfolio_gamma = snapshot.portfolio_gamma
            context.portfolio_vega = snapshot.portfolio_vega
            context.portfolio_theta = snapshot.portfolio_theta
            context.total_notional = snapshot.total_gross_notional
            context.margin_utilization = snapshot.margin_utilization

        # Add position context if available
        if position_risk:
            context.underlying = position_risk.underlying
            context.position_quantity = position_risk.quantity
            context.mark_price = position_risk.mark_price
            context.unrealized_pnl = position_risk.unrealized_pnl
            context.daily_pnl = position_risk.daily_pnl
            context.delta = position_risk.delta
            context.gamma = position_risk.gamma
            context.vega = position_risk.vega
            context.theta = position_risk.theta
            context.iv = position_risk.iv

            # Calculate P&L percentage
            if position_risk.position.avg_price and position_risk.mark_price:
                context.avg_price = position_risk.position.avg_price
                if position_risk.position.avg_price != 0:
                    pnl_pct = (
                        (position_risk.mark_price - position_risk.position.avg_price)
                        / position_risk.position.avg_price
                        * 100
                    )
                    context.pnl_pct = round(pnl_pct, 2)

            # Option-specific fields
            if position_risk.expiry:
                context.expiry = position_risk.expiry
                context.days_to_expiry = position_risk.days_to_expiry()
            if position_risk.strike:
                context.strike = position_risk.strike
            if position_risk.right:
                context.option_type = "CALL" if position_risk.right == "C" else "PUT"

        # Add market data context if available
        if market_data:
            if not context.mark_price:
                context.mark_price = market_data.effective_mid() or market_data.last
            if not context.iv and market_data.iv:
                context.iv = market_data.iv
            if not context.delta and market_data.delta:
                context.delta = market_data.delta
            if not context.gamma and market_data.gamma:
                context.gamma = market_data.gamma
            if not context.vega and market_data.vega:
                context.vega = market_data.vega
            if not context.theta and market_data.theta:
                context.theta = market_data.theta

            # Underlying price
            context.underlying_price = market_data.underlying_price

        # Add signal metadata
        if signal.metadata:
            context.metadata.update(signal.metadata)

        context.metadata["layer"] = signal.layer
        context.metadata["signal_level"] = signal.level.value
        if signal.strategy_type:
            context.metadata["strategy_type"] = signal.strategy_type

        self._write_log(context)

    def log_batch(
        self,
        market_alerts: List[Dict[str, Any]],
        risk_signals: List[RiskSignal],
        snapshot: Optional[RiskSnapshot] = None,
        position_risks: Optional[List[PositionRisk]] = None,
        market_data_store=None,
    ):
        """
        Log a batch of alerts and signals from a single evaluation cycle.

        This is the main entry point for logging during each risk evaluation.

        Args:
            market_alerts: List of market alerts
            risk_signals: List of risk signals
            snapshot: Risk snapshot for portfolio context
            position_risks: List of position risks for position context
            market_data_store: Market data store for enrichment
        """
        # Build position risk lookup
        position_risk_map = {}
        if position_risks:
            for pr in position_risks:
                position_risk_map[pr.symbol] = pr

        # Log market alerts
        for alert in market_alerts:
            self.log_market_alert(alert, snapshot)

        # Log risk signals
        for signal in risk_signals:
            # Find matching position risk
            position_risk = position_risk_map.get(signal.symbol) if signal.symbol else None

            # Find market data
            market_data = None
            if market_data_store and signal.symbol:
                market_data = market_data_store.get(signal.symbol)

            self.log_risk_signal(signal, snapshot, position_risk, market_data)

    def _extract_reason_from_market_alert(self, alert: Dict[str, Any]) -> str:
        """Extract human-readable reason from market alert."""
        alert_type = alert.get("type", "")

        if "VIX" in alert_type:
            vix = alert.get("value") or self._market_cache.get("vix")
            alert.get("threshold")
            if "SPIKE" in alert_type:
                return f"VIX spiked significantly, indicating increased market fear/volatility"
            elif "CRITICAL" in alert_type:
                return f"VIX at extreme level ({vix}), market stress conditions"
            else:
                return f"VIX elevated above normal levels ({vix})"

        elif "MARKET_DROP" in alert_type:
            return "Significant market decline detected, potential risk-off environment"

        elif "VOLATILITY" in alert_type:
            return "Realized volatility elevated, increased price movements expected"

        return alert.get("message", "Market condition alert triggered")

    def _extract_reason_from_signal(self, signal: RiskSignal) -> str:
        """Extract human-readable reason from risk signal."""
        rule = signal.trigger_rule.lower()

        if "delta" in rule:
            return f"Portfolio delta exposure exceeds risk limits, directional risk elevated"
        elif "gamma" in rule:
            return f"Gamma exposure elevated, position sensitive to underlying moves"
        elif "vega" in rule:
            return f"Vega exposure elevated, position sensitive to volatility changes"
        elif "theta" in rule:
            return f"Theta decay significant, time decay impact on portfolio"
        elif "concentration" in rule:
            return f"Position concentration too high in single underlying"
        elif "margin" in rule:
            return f"Margin utilization approaching limits, capital at risk"
        elif "stop" in rule or "loss" in rule:
            return f"Position has breached stop loss threshold"
        elif "profit" in rule:
            return f"Position has reached profit target or drawdown from peak"
        elif "expir" in rule:
            return f"Option approaching expiration, gamma/pin risk elevated"
        elif "notional" in rule:
            return f"Notional exposure exceeds position limits"

        return signal.action_details or f"Risk rule '{signal.trigger_rule}' triggered"

    def _write_log(self, context: AlertContext):
        """Write log entry to file."""
        try:
            self._logger.info(context.to_json())
        except Exception as e:
            logger.exception(f"Failed to write risk alert log: {e}")

    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Read recent alerts from log files.

        Scans log files from recent days based on the hours parameter.

        Args:
            hours: Number of hours to look back

        Returns:
            List of alert dictionaries
        """
        import glob

        alerts = []
        cutoff = datetime.now().timestamp() - (hours * 3600)

        # Calculate how many days of logs we need to check
        days_to_check = (hours // 24) + 2  # +2 to ensure we cover edge cases

        # Find all matching log files
        pattern = str(self.log_dir / f"risk_alerts_{self.env}_*.log")
        log_files = sorted(glob.glob(pattern), reverse=True)[:days_to_check]

        for log_file in log_files:
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        try:
                            alert = json.loads(line.strip())
                            # Parse timestamp and check if within window
                            ts = datetime.fromisoformat(alert.get("timestamp", ""))
                            if ts.timestamp() >= cutoff:
                                alerts.append(alert)
                        except (json.JSONDecodeError, ValueError):
                            continue
            except Exception as e:
                logger.error(f"Failed to read risk alert log {log_file}: {e}")

        # Sort by timestamp descending (most recent first)
        alerts.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return alerts
