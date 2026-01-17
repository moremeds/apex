"""
TUI Event Bus - Thread-safe event routing for TUI updates.

Encapsulates the multi-queue approach for different event types,
each with appropriate conflation strategies:
- Signals: No conflation (each signal matters), max 20 per poll
- Confluence: Conflate to latest (only most recent matters)
- Alignment: Conflate to latest
- Snapshots: Conflate to latest

This keeps the performance of separate queues while providing
a cleaner API for event producers and consumers.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..domain.events.domain_events import PositionDeltaEvent


@dataclass
class DashboardUpdate:
    """Bundled update for the TUI dashboard."""

    snapshot: Optional[Any] = None
    signals: Optional[List[Any]] = None
    health: Optional[List[Any]] = None
    alerts: Optional[List[Dict[str, Any]]] = None


@dataclass
class PollResult:
    """Result of polling all event queues."""

    signals: List[Any]
    confluence: Optional[Any]
    alignment: Optional[Any]
    snapshot: Optional[DashboardUpdate]
    deltas: Dict[str, Any]  # symbol -> latest PositionDeltaEvent

    @property
    def has_data(self) -> bool:
        """Check if any data was received."""
        return (
            bool(self.signals)
            or self.confluence
            or self.alignment
            or self.snapshot
            or bool(self.deltas)
        )


class TUIEventBus:
    """
    Thread-safe event bus for TUI updates.

    Maintains separate queues per event type for optimal performance:
    - Different conflation strategies per type
    - No lock contention between producers
    - Type-safe event routing

    Usage:
        bus = TUIEventBus()

        # From event bus callbacks (producer thread):
        bus.push_signal(signal_event)
        bus.push_confluence(confluence_score)

        # From TUI poll timer (consumer thread):
        result = bus.poll()
        for signal in result.signals:
            dispatch_signal(signal)
        if result.confluence:
            dispatch_confluence(result.confluence)
    """

    __slots__ = (
        "_signal_queue",
        "_confluence_queue",
        "_alignment_queue",
        "_snapshot_queue",
        "_delta_buffer",
        "_delta_lock",
        "_max_signals_per_poll",
    )

    def __init__(
        self,
        signal_queue_size: int = 100,
        confluence_queue_size: int = 100,
        alignment_queue_size: int = 100,
        snapshot_queue_size: int = 10,
        max_signals_per_poll: int = 20,
    ) -> None:
        """
        Initialize event queues.

        Args:
            signal_queue_size: Max queued signals before dropping oldest.
            confluence_queue_size: Max queued confluence scores.
            alignment_queue_size: Max queued alignment updates.
            snapshot_queue_size: Max queued dashboard snapshots.
            max_signals_per_poll: Max signals to process per poll cycle.
        """
        self._signal_queue: queue.Queue = queue.Queue(maxsize=signal_queue_size)
        self._confluence_queue: queue.Queue = queue.Queue(maxsize=confluence_queue_size)
        self._alignment_queue: queue.Queue = queue.Queue(maxsize=alignment_queue_size)
        self._snapshot_queue: queue.Queue = queue.Queue(maxsize=snapshot_queue_size)
        # Delta buffer: coalesces by symbol (latest delta per symbol wins)
        self._delta_buffer: Dict[str, "PositionDeltaEvent"] = {}
        self._delta_lock = threading.Lock()
        self._max_signals_per_poll = max_signals_per_poll

    # ─────────────────────────────────────────────────────────────────────────
    # Producer API (called from event bus thread)
    # ─────────────────────────────────────────────────────────────────────────

    def push_signal(self, signal: Any) -> None:
        """Push trading signal (thread-safe)."""
        q = self._signal_queue
        try:
            q.put_nowait(signal)
        except queue.Full:
            try:
                q.get_nowait()
                q.put_nowait(signal)
            except Exception:
                pass

    def push_confluence(self, score: Any) -> None:
        """Push confluence score (thread-safe)."""
        q = self._confluence_queue
        try:
            q.put_nowait(score)
        except queue.Full:
            try:
                q.get_nowait()
                q.put_nowait(score)
            except Exception:
                pass

    def push_alignment(self, alignment: Any) -> None:
        """Push MTF alignment (thread-safe)."""
        q = self._alignment_queue
        try:
            q.put_nowait(alignment)
        except queue.Full:
            try:
                q.get_nowait()
                q.put_nowait(alignment)
            except Exception:
                pass

    def push_delta(self, delta: "PositionDeltaEvent") -> None:
        """
        Push position delta (thread-safe, coalesced by symbol).

        Only the latest delta per symbol is kept between polls.
        This prevents queue buildup during high tick rates.
        """
        with self._delta_lock:
            self._delta_buffer[delta.symbol] = delta

    def push_snapshot(
        self,
        snapshot: Any,
        signals: Optional[List[Any]] = None,
        health: Optional[List[Any]] = None,
        alerts: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Push dashboard snapshot update (thread-safe)."""
        update = DashboardUpdate(
            snapshot=snapshot,
            signals=signals,
            health=health,
            alerts=alerts,
        )
        q = self._snapshot_queue
        try:
            q.put_nowait(update)
        except queue.Full:
            try:
                q.get_nowait()
                q.put_nowait(update)
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # Consumer API (called from TUI thread)
    # ─────────────────────────────────────────────────────────────────────────

    def poll(self) -> PollResult:
        """
        Poll all queues and return accumulated events.

        Applies type-specific conflation:
        - Signals: Return up to max_signals_per_poll, no conflation
        - Deltas: Coalesced by symbol (latest per symbol)
        - Confluence/Alignment/Snapshot: Return only latest (conflate)

        Returns:
            PollResult with all pending events.
        """
        return PollResult(
            signals=self._poll_signals(),
            confluence=self._poll_latest(self._confluence_queue),
            alignment=self._poll_latest(self._alignment_queue),
            snapshot=self._poll_latest(self._snapshot_queue),
            deltas=self._poll_deltas(),
        )

    def _poll_deltas(self) -> Dict[str, Any]:
        """Poll and clear delta buffer (returns symbol -> delta mapping)."""
        with self._delta_lock:
            deltas = self._delta_buffer.copy()
            self._delta_buffer.clear()
        return deltas

    def _poll_signals(self) -> List[Any]:
        """Poll signals with batch limit (no conflation)."""
        signals = []
        try:
            for _ in range(self._max_signals_per_poll):
                signals.append(self._signal_queue.get_nowait())
        except queue.Empty:
            pass
        return signals

    @staticmethod
    def _poll_latest(q: queue.Queue) -> Optional[Any]:
        """Poll queue and return only latest (conflation)."""
        latest = None
        try:
            while True:
                latest = q.get_nowait()
        except queue.Empty:
            pass
        return latest

    # ─────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────────────────────

    def queue_sizes(self) -> Dict[str, int]:
        """Get current queue sizes (approximate, for diagnostics)."""
        with self._delta_lock:
            delta_count = len(self._delta_buffer)
        return {
            "signals": self._signal_queue.qsize(),
            "confluence": self._confluence_queue.qsize(),
            "alignment": self._alignment_queue.qsize(),
            "snapshots": self._snapshot_queue.qsize(),
            "deltas": delta_count,
        }
