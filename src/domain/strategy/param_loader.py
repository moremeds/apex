"""Strategy parameter loader — single source of truth for all strategy params.

Reads strategy YAML configs from config/strategy/*.yaml and exposes them
via simple accessor functions. All runners, signal generators, and playbooks
should pull default parameters from here rather than hardcoding values.

Each YAML file contains:
    name: strategy name (must match filename stem)
    tier: "TIER 1", "TIER 2", "BASELINE", etc.
    module_path: importable module path for the signal generator
    class_name: signal generator class name
    params: current best parameter dict
    history: list of past parameter versions with results
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

logger = logging.getLogger(__name__)

# Resolve config directory relative to project root
_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config" / "strategy"

# Cache loaded configs to avoid repeated disk I/O
_cache: Dict[str, Dict[str, Any]] = {}


def _load_all() -> Dict[str, Dict[str, Any]]:
    """Load all strategy YAML configs into cache."""
    if _cache:
        return _cache

    if not _CONFIG_DIR.is_dir():
        logger.warning(f"Strategy config directory not found: {_CONFIG_DIR}")
        return _cache

    for path in sorted(_CONFIG_DIR.glob("*.yaml")):
        with open(path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict) or "params" not in data:
            continue  # Skip non-strategy files (e.g. regime_policy.yaml)

        name = data.get("name", path.stem)
        _cache[name] = data

    return _cache


def load_strategy_config(name: str) -> Dict[str, Any]:
    """Load full strategy config (name, tier, module_path, class_name, params, history).

    Args:
        name: Strategy name (e.g. "trend_pulse").

    Returns:
        Full config dict from the YAML file.

    Raises:
        KeyError: If no config file exists for the strategy name.
    """
    configs = _load_all()
    if name not in configs:
        raise KeyError(
            f"No strategy config for '{name}'. "
            f"Available: {sorted(configs.keys())}. "
            f"Config dir: {_CONFIG_DIR}"
        )
    return configs[name]


def get_strategy_params(name: str) -> Dict[str, Any]:
    """Get current best params dict for a strategy.

    Args:
        name: Strategy name (e.g. "trend_pulse").

    Returns:
        The params dict from the YAML file.
    """
    config = load_strategy_config(name)
    return dict(config.get("params", {}))


def get_strategy_metadata(name: str) -> Tuple[str, str, Dict[str, Any], str]:
    """Get (module_path, class_name, params, tier) for a strategy.

    Matches the STRATEGY_REGISTRY tuple format used by strategy_compare_runner.

    Args:
        name: Strategy name (e.g. "trend_pulse").

    Returns:
        (module_path, class_name, params_dict, tier)
    """
    config = load_strategy_config(name)
    return (
        config["module_path"],
        config["class_name"],
        dict(config.get("params", {})),
        config.get("tier", "UNKNOWN"),
    )


def list_strategies() -> List[str]:
    """List all strategy names from YAML files in config/strategy/.

    Returns:
        Sorted list of strategy names.
    """
    configs = _load_all()
    return sorted(configs.keys())


def update_strategy_params(name: str, new_params: Dict[str, Any]) -> None:
    """Update strategy params in YAML, archiving current params to history.

    Loads the existing YAML file, moves the current ``params`` block into
    ``history`` with a timestamp, writes the new params, and clears the
    in-memory cache so subsequent reads see the update.

    Args:
        name: Strategy name (must match an existing YAML file).
        new_params: New parameter dict to write as the active ``params``.

    Raises:
        KeyError: If no config file exists for the strategy name.
        FileNotFoundError: If the YAML file is missing on disk.
    """
    # Validate strategy exists in cache
    _load_all()
    if name not in _cache:
        raise KeyError(
            f"No strategy config for '{name}'. "
            f"Available: {sorted(_cache.keys())}. "
            f"Config dir: {_CONFIG_DIR}"
        )

    yaml_path = _CONFIG_DIR / f"{name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Strategy YAML not found: {yaml_path}")

    # Load raw YAML to preserve structure
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Archive current params into history
    old_params = data.get("params", {})
    if old_params:
        history_entry: Dict[str, Any] = {
            "version": f"pre-optuna-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "source": "Superseded by Optuna optimization",
            "params": dict(old_params),
        }
        if "history" not in data or data["history"] is None:
            data["history"] = []
        data["history"].insert(0, history_entry)

    # Write new params
    data["params"] = dict(new_params)

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info(f"Updated strategy params for '{name}' in {yaml_path}")

    # Clear cache so next read picks up changes
    reload()


def reload() -> None:
    """Clear the config cache, forcing a reload on next access."""
    _cache.clear()
