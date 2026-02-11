"""
Experiment tracking page builder.

Reads Optuna optimization result JSONs from out/optimization/ and builds
a self-contained interactive HTML page showing per-strategy experiment
cards with param evolution, Sharpe trends, and trial details.

Usage:
    from src.infrastructure.reporting.experiment_tracker.builder import build_experiment_page

    path = build_experiment_page()
    print(f"Report at {path}")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .templates import render_experiment_html

logger = logging.getLogger(__name__)


def build_experiment_page(
    optimization_dir: str = "out/optimization",
    output_path: str = "out/experiments/index.html",
) -> str:
    """
    Build HTML experiment tracking page from Optuna result JSONs.

    Scans optimization_dir for JSON files matching the format produced by
    optimize_runner.py, groups them by strategy, and generates an interactive
    HTML page.

    Args:
        optimization_dir: Directory containing optimization result JSONs.
        output_path: Where to write the output HTML file.

    Returns:
        Path to the generated HTML file.
    """
    opt_dir = Path(optimization_dir)
    out_path = Path(output_path)

    # Collect all JSON files
    json_files = sorted(opt_dir.glob("*.json")) if opt_dir.exists() else []
    logger.info(f"Found {len(json_files)} optimization result files in {opt_dir}")

    # Parse and group by strategy
    experiments: Dict[str, List[Dict[str, Any]]] = {}
    parse_errors = 0

    for json_file in json_files:
        try:
            raw = json.loads(json_file.read_text(encoding="utf-8"))
            experiment = _normalize_experiment(raw, json_file.name)
            strategy = experiment["strategy"]

            if strategy not in experiments:
                experiments[strategy] = []
            experiments[strategy].append(experiment)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Skipping {json_file.name}: {e}")
            parse_errors += 1

    # Sort each strategy's experiments by date
    for strategy in experiments:
        experiments[strategy].sort(key=lambda e: e["date"])

    strategy_names = sorted(experiments.keys())

    total_experiments = sum(len(exps) for exps in experiments.values())
    logger.info(
        f"Parsed {total_experiments} experiments across {len(strategy_names)} strategies"
        f" ({parse_errors} parse errors)"
    )

    # Check for comparison dashboard
    comparison_url = _find_comparison_url(out_path)

    # Build template data
    data: Dict[str, Any] = {
        "experiments": experiments,
        "strategy_names": strategy_names,
        "strategy_count": len(strategy_names),
        "total_experiments": total_experiments,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "comparison_url": comparison_url,
    }

    # Render and write
    html = render_experiment_html(data)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    logger.info(f"Experiment tracking page written to {out_path}")
    return str(out_path)


def _normalize_experiment(raw: Dict[str, Any], filename: str) -> Dict[str, Any]:
    """
    Normalize a raw JSON experiment result into the expected format.

    Handles slight variations in the JSON schema gracefully.

    Args:
        raw: Parsed JSON dict from the optimization result file.
        filename: Source filename (for date fallback extraction).

    Returns:
        Normalized experiment dict.
    """
    strategy = raw["strategy"]

    # Extract date: prefer explicit field, fallback to filename parsing
    date = raw.get("date", "")
    if not date:
        # Try to extract date from filename like "trend_pulse_2026-02-08.json"
        parts = filename.rsplit("_", 1)
        if len(parts) == 2:
            date = parts[1].replace(".json", "")
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

    # Normalize best_trial
    best_trial = raw.get("best_trial", {})
    best_trial.setdefault("number", 0)
    best_trial.setdefault("score", 0.0)
    best_trial.setdefault("params", {})
    best_trial.setdefault("user_attrs", {})

    # Normalize all_trials
    all_trials = raw.get("all_trials", [])
    for trial in all_trials:
        trial.setdefault("number", 0)
        trial.setdefault("score", 0.0)
        trial.setdefault("params", {})
        trial.setdefault("state", "UNKNOWN")
        trial.setdefault("user_attrs", {})

    return {
        "strategy": strategy,
        "date": date,
        "n_trials": raw.get("n_trials", len(all_trials)),
        "best_trial": best_trial,
        "all_trials": all_trials,
    }


def _find_comparison_url(output_path: Path) -> str:
    """
    Look for the strategy comparison dashboard relative to the output path.

    Checks common locations where the comparison dashboard might exist.

    Args:
        output_path: The experiment page output path (for relative path calculation).

    Returns:
        Relative URL to the comparison dashboard, or empty string if not found.
    """
    # Check in out/signals/strategies.html (typical location)
    candidates = [
        output_path.parent.parent / "signals" / "strategies.html",
        output_path.parent / "strategies.html",
    ]

    for candidate in candidates:
        if candidate.exists():
            try:
                rel = candidate.relative_to(output_path.parent)
                return str(rel)
            except ValueError:
                return str(candidate)

    return ""
