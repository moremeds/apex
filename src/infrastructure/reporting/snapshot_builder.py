"""
Snapshot Builder - Machine-readable payload snapshot for content diffing.

PR-02 Deliverable: Creates payload_snapshot.json for regression testing.

CRITICAL: Content diff ONLY via this JSON snapshot, NEVER via HTML parsing.
The snapshot provides a stable, machine-readable representation of the
report's data content that can be reliably diffed between runs.

Usage:
    builder = SnapshotBuilder()
    snapshot = builder.build(
        data=data,
        regime_outputs=regime_outputs,
        symbols=symbols,
        timeframes=timeframes,
    )

    # Write to file
    Path("payload_snapshot.json").write_text(json.dumps(snapshot, indent=2))

    # Diff between runs
    diff = builder.diff(old_snapshot, new_snapshot)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import pandas as pd

from src.utils.logging_setup import get_logger

if TYPE_CHECKING:
    from src.domain.signals.indicators.regime import RegimeOutput

logger = get_logger(__name__)

# Snapshot schema version
SNAPSHOT_VERSION = "1.0"


@dataclass(frozen=True)
class SnapshotDiff:
    """Result of comparing two snapshots."""

    symbols_added: Tuple[str, ...]
    symbols_removed: Tuple[str, ...]
    regime_changes: Dict[str, Dict[str, Any]]  # symbol -> {old_regime, new_regime}
    metric_changes: Dict[str, Dict[str, Any]]  # symbol -> {metric_name: {old, new}}
    bar_count_changes: Dict[str, Dict[str, int]]  # symbol_tf -> {old, new}

    @property
    def has_changes(self) -> bool:
        """Check if there are any differences."""
        return bool(
            self.symbols_added
            or self.symbols_removed
            or self.regime_changes
            or self.metric_changes
            or self.bar_count_changes
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "has_changes": self.has_changes,
            "symbols_added": list(self.symbols_added),
            "symbols_removed": list(self.symbols_removed),
            "regime_changes": self.regime_changes,
            "metric_changes": self.metric_changes,
            "bar_count_changes": self.bar_count_changes,
        }

    def summary(self) -> str:
        """Human-readable summary of changes."""
        lines = []

        if self.symbols_added:
            lines.append(f"Symbols added: {', '.join(self.symbols_added)}")
        if self.symbols_removed:
            lines.append(f"Symbols removed: {', '.join(self.symbols_removed)}")
        if self.regime_changes:
            for symbol, change in self.regime_changes.items():
                lines.append(f"Regime change: {symbol} {change['old']} -> {change['new']}")
        if self.metric_changes:
            for symbol, metrics in self.metric_changes.items():
                for metric, change in metrics.items():
                    lines.append(
                        f"Metric change: {symbol}.{metric} {change['old']} -> {change['new']}"
                    )
        if self.bar_count_changes:
            for key, change in self.bar_count_changes.items():
                lines.append(f"Bar count change: {key} {change['old']} -> {change['new']}")

        return "\n".join(lines) if lines else "No changes"


class SnapshotBuilder:
    """
    Builds machine-readable payload snapshots for content diffing.

    The snapshot captures:
    - Symbol/timeframe inventory
    - Regime states for each symbol
    - Key metrics (condensed)
    - Bar counts per symbol/timeframe
    - Content hash for quick comparison

    NO HTML PARSING - this is the ONLY source of truth for content comparison.
    """

    def build(
        self,
        data: Dict[Tuple[str, str], pd.DataFrame],
        regime_outputs: Dict[str, "RegimeOutput"],
        symbols: List[str],
        timeframes: List[str],
    ) -> Dict[str, Any]:
        """
        Build a snapshot of the report payload.

        Args:
            data: Dict mapping (symbol, timeframe) to DataFrame
            regime_outputs: Dict mapping symbol to RegimeOutput
            symbols: List of symbols
            timeframes: List of timeframes

        Returns:
            Snapshot dictionary suitable for JSON serialization
        """
        snapshot: Dict[str, Any] = {
            "version": SNAPSHOT_VERSION,
            "created_at": datetime.now().isoformat(),
            "inventory": {
                "symbols": sorted(symbols),
                "timeframes": sorted(timeframes, key=lambda x: self._tf_seconds(x)),
                "symbol_count": len(symbols),
                "timeframe_count": len(timeframes),
            },
        }

        # Regime states (condensed)
        regimes: Dict[str, Dict[str, Any]] = {}
        for symbol in symbols:
            regime = regime_outputs.get(symbol)
            if regime:
                regimes[symbol] = {
                    "regime": regime.final_regime.value,
                    "confidence": regime.confidence,
                    "regime_changed": regime.regime_changed,
                    "trend_state": regime.component_states.trend_state.value,
                    "vol_state": regime.component_states.vol_state.value,
                    "chop_state": regime.component_states.chop_state.value,
                    "ext_state": regime.component_states.ext_state.value,
                }
        snapshot["regimes"] = regimes

        # Key metrics (condensed)
        metrics: Dict[str, Dict[str, float]] = {}
        for symbol in symbols:
            regime = regime_outputs.get(symbol)
            if regime:
                metrics[symbol] = {
                    "close": round(regime.component_values.close, 2),
                    "atr_pct_63": round(regime.component_values.atr_pct_63, 1),
                    "chop_pct": round(regime.component_values.chop_pct_252, 1),
                    "ext": round(regime.component_values.ext, 2),
                }
        snapshot["metrics"] = metrics

        # Bar counts per symbol/timeframe
        bar_counts: Dict[str, int] = {}
        for (symbol, timeframe), df in data.items():
            key = f"{symbol}_{timeframe}"
            bar_counts[key] = len(df)
        snapshot["bar_counts"] = bar_counts

        # Turning point predictions if available
        from src.domain.signals.indicators.regime.turning_point.model import TurnState

        turning_points: Dict[str, Dict[str, Any]] = {}
        for symbol in symbols:
            regime = regime_outputs.get(symbol)
            if regime and regime.turning_point:
                tp = regime.turning_point
                top_prob = tp.turn_confidence if tp.turn_state == TurnState.TOP_RISK else 0.0
                bottom_prob = tp.turn_confidence if tp.turn_state == TurnState.BOTTOM_RISK else 0.0
                turning_points[symbol] = {
                    "top_prob": round(top_prob, 3),
                    "bottom_prob": round(bottom_prob, 3),
                    "state": tp.turn_state.value,
                    "confidence": round(tp.turn_confidence, 3),
                }
        snapshot["turning_points"] = turning_points

        # Content hash for quick comparison
        content_str = json.dumps(
            {
                "regimes": snapshot["regimes"],
                "metrics": snapshot["metrics"],
                "bar_counts": snapshot["bar_counts"],
            },
            sort_keys=True,
        )
        snapshot["content_hash"] = hashlib.sha256(content_str.encode()).hexdigest()[:16]

        return snapshot

    def diff(
        self,
        old_snapshot: Dict[str, Any],
        new_snapshot: Dict[str, Any],
    ) -> SnapshotDiff:
        """
        Compare two snapshots and return differences.

        Args:
            old_snapshot: Previous snapshot
            new_snapshot: Current snapshot

        Returns:
            SnapshotDiff with all differences
        """
        old_symbols = set(old_snapshot.get("inventory", {}).get("symbols", []))
        new_symbols = set(new_snapshot.get("inventory", {}).get("symbols", []))

        symbols_added = tuple(sorted(new_symbols - old_symbols))
        symbols_removed = tuple(sorted(old_symbols - new_symbols))

        # Regime changes
        regime_changes: Dict[str, Dict[str, Any]] = {}
        old_regimes = old_snapshot.get("regimes", {})
        new_regimes = new_snapshot.get("regimes", {})

        for symbol in old_symbols & new_symbols:
            old_regime = old_regimes.get(symbol, {}).get("regime")
            new_regime = new_regimes.get(symbol, {}).get("regime")
            if old_regime != new_regime:
                regime_changes[symbol] = {"old": old_regime, "new": new_regime}

        # Metric changes (only for significant differences)
        metric_changes: Dict[str, Dict[str, Any]] = {}
        old_metrics = old_snapshot.get("metrics", {})
        new_metrics = new_snapshot.get("metrics", {})

        for symbol in old_symbols & new_symbols:
            old_m = old_metrics.get(symbol, {})
            new_m = new_metrics.get(symbol, {})

            symbol_changes: Dict[str, Any] = {}
            for metric_name in set(old_m.keys()) | set(new_m.keys()):
                old_val = old_m.get(metric_name)
                new_val = new_m.get(metric_name)

                # Check for significant difference (>1% for percentages, >0.5% for prices)
                if old_val is not None and new_val is not None:
                    if metric_name in ("atr_pct_63", "chop_pct"):
                        # Percentage metrics: 1 point difference threshold
                        if abs(new_val - old_val) >= 1.0:
                            symbol_changes[metric_name] = {"old": old_val, "new": new_val}
                    elif metric_name == "close":
                        # Price: 0.5% difference threshold
                        if old_val > 0 and abs(new_val - old_val) / old_val >= 0.005:
                            symbol_changes[metric_name] = {"old": old_val, "new": new_val}
                    else:
                        # Default: any difference
                        if old_val != new_val:
                            symbol_changes[metric_name] = {"old": old_val, "new": new_val}

            if symbol_changes:
                metric_changes[symbol] = symbol_changes

        # Bar count changes
        bar_count_changes: Dict[str, Dict[str, int]] = {}
        old_counts = old_snapshot.get("bar_counts", {})
        new_counts = new_snapshot.get("bar_counts", {})

        for key in set(old_counts.keys()) | set(new_counts.keys()):
            old_count = old_counts.get(key, 0)
            new_count = new_counts.get(key, 0)
            if old_count != new_count:
                bar_count_changes[key] = {"old": old_count, "new": new_count}

        return SnapshotDiff(
            symbols_added=symbols_added,
            symbols_removed=symbols_removed,
            regime_changes=regime_changes,
            metric_changes=metric_changes,
            bar_count_changes=bar_count_changes,
        )

    def quick_compare(
        self,
        old_snapshot: Dict[str, Any],
        new_snapshot: Dict[str, Any],
    ) -> bool:
        """
        Quick comparison using content hash.

        Returns True if snapshots are identical (same hash).
        """
        return old_snapshot.get("content_hash") == new_snapshot.get("content_hash")

    @staticmethod
    def _tf_seconds(tf: str) -> int:
        """Convert timeframe to seconds for sorting."""
        mapping = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400,
            "1w": 604800,
        }
        return mapping.get(tf, 0)
