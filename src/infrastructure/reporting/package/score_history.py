"""
Score History Manager - Tracks composite scores across report runs.

Maintains a JSON file with the last N snapshots of per-symbol composite scores,
enabling trend display (sparklines, arrows) in the ETF dashboard.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

MAX_SNAPSHOTS = 15


@dataclass
class ScoreSnapshot:
    """A single point-in-time snapshot of all symbol scores."""

    timestamp: str  # ISO 8601
    scores: Dict[str, float] = field(default_factory=dict)  # symbol -> composite_score_avg


@dataclass
class ScoreHistory:
    """Full score history with up to MAX_SNAPSHOTS entries."""

    snapshots: List[ScoreSnapshot] = field(default_factory=list)


class ScoreHistoryManager:
    """Manages loading, appending, trimming, and saving score history."""

    def __init__(self, max_entries: int = MAX_SNAPSHOTS) -> None:
        self.max_entries = max_entries
        self.history = ScoreHistory()

    def load(self, path: Path) -> None:
        """Load existing history from JSON file."""
        if not path.exists():
            logger.info("No existing score history found at %s", path)
            return

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            snapshots = []
            for snap in raw.get("snapshots", []):
                snapshots.append(
                    ScoreSnapshot(
                        timestamp=snap.get("timestamp", ""),
                        scores=snap.get("scores", {}),
                    )
                )
            self.history = ScoreHistory(snapshots=snapshots)
            logger.info("Loaded %d score history snapshots", len(snapshots))
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to parse score history: %s", e)
            self.history = ScoreHistory()

    def append_from_summary(
        self, summary: Dict[str, Any], timestamp: Optional[datetime] = None
    ) -> None:
        """
        Extract per-symbol composite_score_avg from summary and append a snapshot.

        Args:
            summary: The summary dict (with "tickers" list containing "composite_score_avg")
            timestamp: Override timestamp (defaults to now)
        """
        ts = (timestamp or datetime.now()).isoformat()
        scores: Dict[str, float] = {}

        for ticker in summary.get("tickers", []):
            symbol = ticker.get("symbol")
            score = ticker.get("composite_score_avg")
            if symbol and score is not None:
                scores[symbol] = round(float(score), 1)

        if not scores:
            logger.warning("No composite scores found in summary, skipping snapshot")
            return

        self.history.snapshots.append(ScoreSnapshot(timestamp=ts, scores=scores))
        self._trim()
        logger.info(
            "Appended score snapshot with %d symbols (total: %d)",
            len(scores),
            len(self.history.snapshots),
        )

    def _trim(self) -> None:
        """Keep only the last max_entries snapshots."""
        if len(self.history.snapshots) > self.max_entries:
            self.history.snapshots = self.history.snapshots[-self.max_entries :]

    def save(self, path: Path) -> None:
        """Write history to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"snapshots": [asdict(s) for s in self.history.snapshots]}
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Saved score history (%d snapshots) to %s", len(self.history.snapshots), path)

    def get_trend(self, symbol: str, window: int = 3) -> str:
        """
        Determine score trend for a symbol based on last N snapshots.

        Returns "up", "down", or "stable".
        """
        recent_scores: List[float] = []
        for snap in self.history.snapshots[-window:]:
            score = snap.scores.get(symbol)
            if score is not None:
                recent_scores.append(score)

        if len(recent_scores) < 2:
            return "stable"

        delta = recent_scores[-1] - recent_scores[0]
        if delta > 3:
            return "up"
        elif delta < -3:
            return "down"
        return "stable"

    def get_all_trends(self) -> Dict[str, str]:
        """Get trends for all symbols in the most recent snapshot."""
        if not self.history.snapshots:
            return {}
        latest = self.history.snapshots[-1]
        return {symbol: self.get_trend(symbol) for symbol in latest.scores}

    def get_sparkline_points(self, symbol: str) -> List[float]:
        """
        Get score history for a symbol as a list of values (oldest first).

        Returns up to max_entries floats for sparkline rendering.
        """
        points: List[float] = []
        for snap in self.history.snapshots:
            score = snap.scores.get(symbol)
            if score is not None:
                points.append(score)
        return points

    def get_all_sparklines(self) -> Dict[str, List[float]]:
        """Get sparkline data for all symbols in the most recent snapshot."""
        if not self.history.snapshots:
            return {}
        latest = self.history.snapshots[-1]
        return {symbol: self.get_sparkline_points(symbol) for symbol in latest.scores}

    def to_json_data(self) -> Dict[str, Any]:
        """Return serializable data for embedding in JS."""
        return {"snapshots": [asdict(s) for s in self.history.snapshots]}
