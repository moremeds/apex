"""Repository for risk alerts."""

from __future__ import annotations
from datetime import datetime
from typing import List, Optional, Dict, Any
import json
import logging

from ..duckdb_adapter import DuckDBAdapter
from src.models.risk_signal import RiskSignal

logger = logging.getLogger(__name__)


class AlertRepository:
    """Repository for risk alert persistence operations."""

    def __init__(self, db: DuckDBAdapter):
        self.db = db

    def save_alert(self, signal: RiskSignal) -> None:
        """Save a risk alert."""
        context_json = json.dumps(signal.metadata) if signal.metadata else None

        self.db.execute("""
            INSERT INTO risk_alerts (
                id, alert_time, alert_type, severity, trigger_rule,
                symbol, current_value, threshold, breach_pct,
                message, suggested_action, context_json
            ) VALUES (
                nextval('risk_alerts_id_seq'),
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            signal.timestamp,
            signal.level.value if signal.level else "UNKNOWN",
            signal.severity.value if signal.severity else "INFO",
            signal.trigger_rule,
            signal.symbol,
            signal.current_value,
            signal.threshold,
            signal.breach_pct,
            signal.action_details,
            signal.suggested_action.value if signal.suggested_action else None,
            context_json,
        ))

    def save_alerts_batch(self, signals: List[RiskSignal]) -> int:
        """Save multiple alerts in batch."""
        if not signals:
            return 0

        records = []
        for signal in signals:
            context_json = json.dumps(signal.metadata) if signal.metadata else None
            records.append((
                signal.timestamp,
                signal.level.value if signal.level else "UNKNOWN",
                signal.severity.value if signal.severity else "INFO",
                signal.trigger_rule,
                signal.symbol,
                signal.current_value,
                signal.threshold,
                signal.breach_pct,
                signal.action_details,
                signal.suggested_action.value if signal.suggested_action else None,
                context_json,
            ))

        self.db.executemany("""
            INSERT INTO risk_alerts (
                id, alert_time, alert_type, severity, trigger_rule,
                symbol, current_value, threshold, breach_pct,
                message, suggested_action, context_json
            ) VALUES (
                nextval('risk_alerts_id_seq'),
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, records)

        return len(records)

    def acknowledge_alert(self, alert_id: int, acknowledged_by: str = "user") -> bool:
        """Mark an alert as acknowledged."""
        self.db.execute("""
            UPDATE risk_alerts
            SET acknowledged = TRUE,
                acknowledged_at = ?,
                acknowledged_by = ?
            WHERE id = ?
        """, (datetime.now(), acknowledged_by, alert_id))
        return True

    def get_unacknowledged_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get unacknowledged alerts."""
        return self.db.fetch_all("""
            SELECT * FROM risk_alerts
            WHERE acknowledged = FALSE
            ORDER BY alert_time DESC
            LIMIT ?
        """, (limit,))

    def get_alerts_by_severity(
        self,
        severity: str,
        start_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get alerts by severity level."""
        if start_time:
            return self.db.fetch_all("""
                SELECT * FROM risk_alerts
                WHERE severity = ? AND alert_time >= ?
                ORDER BY alert_time DESC
                LIMIT ?
            """, (severity, start_time, limit))
        else:
            return self.db.fetch_all("""
                SELECT * FROM risk_alerts
                WHERE severity = ?
                ORDER BY alert_time DESC
                LIMIT ?
            """, (severity, limit))

    def get_alerts_by_symbol(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get alerts for a specific symbol."""
        if start_time:
            return self.db.fetch_all("""
                SELECT * FROM risk_alerts
                WHERE symbol = ? AND alert_time >= ?
                ORDER BY alert_time DESC
                LIMIT ?
            """, (symbol, start_time, limit))
        else:
            return self.db.fetch_all("""
                SELECT * FROM risk_alerts
                WHERE symbol = ?
                ORDER BY alert_time DESC
                LIMIT ?
            """, (symbol, limit))

    def get_alerts_today(self) -> List[Dict[str, Any]]:
        """Get all alerts for today."""
        return self.db.fetch_all("""
            SELECT * FROM risk_alerts
            WHERE alert_time >= CURRENT_DATE
            ORDER BY alert_time DESC
        """)

    def get_alert_summary(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get alert statistics for a time range."""
        result = self.db.fetch_one("""
            SELECT
                COUNT(*) as total_alerts,
                SUM(CASE WHEN severity = 'CRITICAL' THEN 1 ELSE 0 END) as critical_count,
                SUM(CASE WHEN severity = 'WARNING' THEN 1 ELSE 0 END) as warning_count,
                SUM(CASE WHEN severity = 'INFO' THEN 1 ELSE 0 END) as info_count,
                SUM(CASE WHEN acknowledged THEN 1 ELSE 0 END) as acknowledged_count
            FROM risk_alerts
            WHERE alert_time BETWEEN ? AND ?
        """, (start_time, end_time))

        return result if result else {}

    def get_most_triggered_rules(
        self,
        start_time: datetime,
        end_time: datetime,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get the most frequently triggered rules."""
        rows = self.db.fetch_all("""
            SELECT
                trigger_rule,
                COUNT(*) as trigger_count,
                COUNT(DISTINCT symbol) as symbols_affected
            FROM risk_alerts
            WHERE alert_time BETWEEN ? AND ?
            GROUP BY trigger_rule
            ORDER BY trigger_count DESC
            LIMIT ?
        """, (start_time, end_time, limit))

        return [{"rule": r.get("trigger_rule"), "count": r.get("trigger_count"), "symbols": r.get("symbols_affected")} for r in rows] if rows else []

    def cleanup_old_alerts(self, days_to_keep: int = 365) -> int:
        """Delete alerts older than specified days."""
        result = self.db.execute("""
            DELETE FROM risk_alerts
            WHERE alert_time < CURRENT_DATE - INTERVAL ? DAY
        """, (days_to_keep,))
        deleted = result.fetchone()[0] if result else 0
        logger.info(f"Deleted {deleted} old alerts (>{days_to_keep} days)")
        return deleted
