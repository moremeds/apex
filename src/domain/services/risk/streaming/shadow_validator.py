"""
Shadow Validator - Compares streaming vs batch risk calculations.

Used during Phase 2 to validate that the new streaming system produces
the same results as the existing batch-based RiskEngine.

Subscribes to SNAPSHOT_READY events and compares the snapshot's P&L
values with the RiskFacade's PortfolioState.

Logs discrepancies for investigation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.domain.events.event_types import EventType
from src.utils.logging_setup import get_logger

if TYPE_CHECKING:
    from src.domain.events.priority_event_bus import PriorityEventBus
    from src.domain.events.domain_events import SnapshotReadyEvent
    from src.models.risk_snapshot import RiskSnapshot
    from ..risk_facade import RiskFacade

logger = get_logger(__name__)


# Tolerance for P&L comparison (allow small floating point differences)
PNL_TOLERANCE = 0.01  # $0.01
PNL_TOLERANCE_PCT = 0.001  # 0.1%


class ShadowValidator:
    """
    Validates streaming risk calculations against batch calculations.

    Compares:
    - Total unrealized P&L
    - Total daily P&L
    - Portfolio delta

    Logs discrepancies at WARNING level for investigation.
    """

    def __init__(
        self,
        risk_facade: RiskFacade,
        event_bus: PriorityEventBus,
    ) -> None:
        """
        Initialize ShadowValidator.

        Args:
            risk_facade: Risk facade with streaming calculations.
            event_bus: Event bus for subscribing to snapshots.
        """
        self._facade = risk_facade
        self._bus = event_bus
        self._started = False

        # Statistics
        self._comparisons = 0
        self._matches = 0
        self._mismatches = 0

    def start(self) -> None:
        """Start listening for snapshot events."""
        if self._started:
            return

        self._bus.subscribe(EventType.SNAPSHOT_READY, self._on_snapshot_event)
        self._started = True
        logger.info("ShadowValidator started")

    def stop(self) -> None:
        """Stop listening for events."""
        if not self._started:
            return

        self._bus.unsubscribe(EventType.SNAPSHOT_READY, self._on_snapshot_event)
        self._started = False

        logger.info(
            "ShadowValidator stopped: comparisons=%d, matches=%d, mismatches=%d",
            self._comparisons,
            self._matches,
            self._mismatches,
        )

    def _on_snapshot_event(self, event: "SnapshotReadyEvent") -> None:
        """
        Handle SnapshotReadyEvent and compare with streaming state.

        Args:
            event: Snapshot ready event with summary metrics.
        """
        self._comparisons += 1

        # Get streaming state
        streaming_snapshot = self._facade.get_snapshot()

        # Compare key metrics available in SnapshotReadyEvent
        discrepancies = []

        # Skip comparison if coverage is too low (may cause false mismatches)
        if event.coverage_pct < 0.5:
            logger.debug(
                "Shadow validation skipped: coverage_pct=%.1f%% < 50%%",
                event.coverage_pct * 100,
            )
            return

        # Total unrealized P&L
        batch_pnl = event.unrealized_pnl
        stream_pnl = streaming_snapshot.total_unrealized_pnl
        if not self._values_match(batch_pnl, stream_pnl):
            discrepancies.append(
                f"unrealized_pnl: batch={batch_pnl:.2f}, stream={stream_pnl:.2f}, "
                f"diff={batch_pnl - stream_pnl:.2f}"
            )

        # Total daily P&L
        batch_daily = event.daily_pnl
        stream_daily = streaming_snapshot.total_daily_pnl
        if not self._values_match(batch_daily, stream_daily):
            discrepancies.append(
                f"daily_pnl: batch={batch_daily:.2f}, stream={stream_daily:.2f}, "
                f"diff={batch_daily - stream_daily:.2f}"
            )

        # Portfolio delta
        batch_delta = event.portfolio_delta
        stream_delta = streaming_snapshot.portfolio_delta
        if not self._values_match(batch_delta, stream_delta, tolerance_pct=0.01):
            discrepancies.append(
                f"portfolio_delta: batch={batch_delta:.2f}, stream={stream_delta:.2f}, "
                f"diff={batch_delta - stream_delta:.2f}"
            )

        # Position count
        batch_count = event.position_count
        stream_count = streaming_snapshot.total_positions
        if batch_count != stream_count:
            discrepancies.append(
                f"position_count: batch={batch_count}, stream={stream_count}"
            )

        # Log results
        if discrepancies:
            self._mismatches += 1
            logger.warning(
                "Shadow validation MISMATCH: %s",
                "; ".join(discrepancies),
            )
        else:
            self._matches += 1
            # Only log matches periodically to avoid spam
            if self._comparisons % 100 == 0:
                logger.debug(
                    "Shadow validation OK: %d comparisons, %d matches",
                    self._comparisons,
                    self._matches,
                )

    def _on_snapshot(self, snapshot: RiskSnapshot) -> None:
        """
        Compare full RiskSnapshot with streaming state.

        Used for testing. In production, use _on_snapshot_event with SnapshotReadyEvent.

        Args:
            snapshot: Full risk snapshot.
        """
        self._comparisons += 1
        streaming_snapshot = self._facade.get_snapshot()
        discrepancies = []

        if not self._values_match(snapshot.total_unrealized_pnl, streaming_snapshot.total_unrealized_pnl):
            discrepancies.append(
                f"unrealized_pnl: batch={snapshot.total_unrealized_pnl:.2f}, "
                f"stream={streaming_snapshot.total_unrealized_pnl:.2f}"
            )

        if not self._values_match(snapshot.total_daily_pnl, streaming_snapshot.total_daily_pnl):
            discrepancies.append(
                f"daily_pnl: batch={snapshot.total_daily_pnl:.2f}, "
                f"stream={streaming_snapshot.total_daily_pnl:.2f}"
            )

        if not self._values_match(snapshot.portfolio_delta, streaming_snapshot.portfolio_delta, tolerance_pct=0.01):
            discrepancies.append(
                f"portfolio_delta: batch={snapshot.portfolio_delta:.2f}, "
                f"stream={streaming_snapshot.portfolio_delta:.2f}"
            )

        if snapshot.total_positions != streaming_snapshot.total_positions:
            discrepancies.append(
                f"position_count: batch={snapshot.total_positions}, stream={streaming_snapshot.total_positions}"
            )

        if discrepancies:
            self._mismatches += 1
            logger.warning("Shadow validation MISMATCH: %s", "; ".join(discrepancies))
        else:
            self._matches += 1

    def _values_match(
        self,
        batch: float,
        stream: float,
        tolerance: float = PNL_TOLERANCE,
        tolerance_pct: float = PNL_TOLERANCE_PCT,
    ) -> bool:
        """
        Check if two values match within tolerance.

        Uses both absolute and percentage tolerance.

        Args:
            batch: Value from batch calculation.
            stream: Value from streaming calculation.
            tolerance: Absolute tolerance.
            tolerance_pct: Percentage tolerance.

        Returns:
            True if values match within tolerance.
        """
        diff = abs(batch - stream)

        # Absolute tolerance
        if diff <= tolerance:
            return True

        # Percentage tolerance (relative to larger value)
        max_val = max(abs(batch), abs(stream))
        if max_val > 0 and (diff / max_val) <= tolerance_pct:
            return True

        return False

    @property
    def stats(self) -> dict:
        """Get validation statistics."""
        return {
            "comparisons": self._comparisons,
            "matches": self._matches,
            "mismatches": self._mismatches,
            "match_rate": (
                self._matches / self._comparisons
                if self._comparisons > 0
                else 0.0
            ),
        }
