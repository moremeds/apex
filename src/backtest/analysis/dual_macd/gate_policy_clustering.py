"""
Auto-clustering pipeline for DualMACD gate policy assignment.

Clusters symbols into BLOCK / SIZE_DOWN / BYPASS / MACRO_PROXY buckets
based on behavioral metrics feature vectors. Output is a candidate YAML
that requires human review before activation.

INVARIANT: Clustering output changes gate POLICY only.
It must NEVER change DualMACD indicator parameters.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Market ETFs forced to MACRO_PROXY regardless of clustering
MACRO_PROXY_ETFS = frozenset({"SPY", "QQQ", "TLT", "IWM", "DIA", "GLD", "SLV", "UVXY"})


def _extract_features(result: Any) -> np.ndarray:
    """Extract 7-dim feature vector from a SymbolResult.

    Features:
        0: blocked_ratio = blocked_count / baseline_count
        1: blocked_loss_ratio
        2: blocked_avg_pnl
        3: size_down_ratio = size_down_count / baseline_count
        4: bypass_ratio = bypass_count / baseline_count
        5: allowed_trade_ratio
        6: fail_flag (0/1) — fails hard constraints
    """
    m = result.metrics
    base = max(m.baseline_trade_count, 1)

    fail = 0
    if m.blocked_trade_count > 0:
        if m.blocked_trade_loss_ratio < 0.6 or m.allowed_trade_ratio < 0.7:
            fail = 1

    return np.array(
        [
            m.blocked_trade_count / base,
            m.blocked_trade_loss_ratio,
            m.blocked_trade_avg_pnl,
            m.size_down_count / base,
            m.bypass_count / base,
            m.allowed_trade_ratio,
            float(fail),
        ]
    )


def _assign_labels(
    features: np.ndarray,
    symbols: List[str],
    n_clusters: int = 4,
) -> Dict[str, str]:
    """Cluster symbols and map clusters to policy labels via centroid rules."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler

    scaled = StandardScaler().fit_transform(features)

    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = clustering.fit_predict(scaled)

    # Compute centroids per cluster
    centroids: Dict[int, np.ndarray] = {}
    for c in range(n_clusters):
        mask = labels == c
        if mask.any():
            centroids[c] = features[mask].mean(axis=0)

    # Rule-based label mapping from centroids
    # Feature indices: 0=blocked_ratio, 1=blocked_loss_ratio, 2=blocked_avg_pnl
    cluster_labels: Dict[int, str] = {}
    remaining = set(range(n_clusters))

    # BLOCK: highest blocked_loss_ratio + most negative blocked_avg_pnl
    if remaining:
        best_block = max(
            remaining,
            key=lambda c: centroids[c][1] + abs(min(centroids[c][2], 0)),
        )
        cluster_labels[best_block] = "BLOCK"
        remaining.discard(best_block)

    # BYPASS: near-zero blocks (lowest blocked_ratio)
    if remaining:
        best_bypass = min(remaining, key=lambda c: centroids[c][0])
        cluster_labels[best_bypass] = "BYPASS"
        remaining.discard(best_bypass)

    # SIZE_DOWN: everything else
    for c in remaining:
        cluster_labels[c] = "SIZE_DOWN"

    # Map symbols
    symbol_labels: Dict[str, str] = {}
    for i, sym in enumerate(symbols):
        symbol_labels[sym] = cluster_labels[labels[i]]

    return symbol_labels


def generate_cluster_policies(
    results: List[Any],
    source_params: Dict[str, Any],
    output_path: Path,
) -> Path:
    """Run clustering and write candidate gate_policy_clusters.yaml.

    This is always dry-run: prints diff vs current policy, never auto-applies.

    Args:
        results: List of SymbolResult from behavioral run
        source_params: DualMACD params used (for metadata)
        output_path: Where to write the candidate YAML

    Returns:
        Path to the generated YAML
    """
    symbols = [r.symbol for r in results]
    features = np.array([_extract_features(r) for r in results])

    # Need at least 4 symbols for 4 clusters
    n_clusters = min(4, len(symbols))
    if n_clusters < 2:
        logger.warning("Too few symbols for clustering, skipping")
        return output_path

    symbol_labels = _assign_labels(features, symbols, n_clusters=n_clusters)

    # Forced overrides for macro ETFs
    for sym in symbols:
        if sym.upper() in MACRO_PROXY_ETFS:
            symbol_labels[sym] = "MACRO_PROXY"

    # Build cluster buckets
    buckets: Dict[str, List[str]] = {}
    for sym, label in symbol_labels.items():
        buckets.setdefault(label, []).append(sym)
    for bucket in buckets.values():
        bucket.sort()

    # Build YAML structure
    clusters_yaml: Dict[str, Any] = {}
    cluster_configs: Dict[str, Dict[str, Any]] = {
        "BLOCK": {"action_on_block": "BLOCK"},
        "SIZE_DOWN": {"action_on_block": "SIZE_DOWN", "size_factor": 0.5},
        "BYPASS": {"action_on_block": "BYPASS"},
        "MACRO_PROXY": {"action_on_block": "SIZE_DOWN", "size_factor": 0.7},
    }

    for label in ["BLOCK", "SIZE_DOWN", "BYPASS", "MACRO_PROXY"]:
        syms = buckets.get(label, [])
        if syms:
            cfg = dict(cluster_configs[label])
            cfg["symbols"] = syms
            clusters_yaml[label] = cfg

    output = {
        "# Auto-generated — requires human review before activation": None,
        "generated_at": datetime.now().strftime("%Y-%m-%d"),
        "source_params": source_params,
        "status": "candidate",
        "clusters": clusters_yaml,
    }

    # Write YAML (strip the None comment key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_content = yaml.dump(
        {k: v for k, v in output.items() if v is not None},
        default_flow_style=False,
        sort_keys=False,
    )
    # Add comment at top
    yaml_content = "# Auto-generated — requires human review before activation\n" + yaml_content
    output_path.write_text(yaml_content, encoding="utf-8")

    # Print summary (dry-run output)
    print(f"\n{'='*60}")
    print("Gate Policy Clustering Results (CANDIDATE — not auto-applied)")
    print(f"{'='*60}")
    for label, syms in buckets.items():
        cfg = cluster_configs.get(label, {})
        action = cfg.get("action_on_block", label)
        sf = cfg.get("size_factor", "")
        sf_str = f" @ {sf}x" if sf else ""
        print(f"  {label} ({action}{sf_str}): {len(syms)} symbols")
        for sym in syms:
            print(f"    - {sym}")
    print(f"\nCandidate written to: {output_path}")
    print("To activate: change 'status: candidate' → 'status: active' after review")

    return output_path
