"""
Deterministic hashing utilities for reproducible experiment IDs.

This module provides content-addressed hashing that ensures:
1. Same inputs always produce same hash (deterministic)
2. Float quantization prevents precision-related differences
3. Dictionary keys are sorted for consistent ordering
4. Special values (NaN, inf) are handled correctly
5. Type consistency (floats remain floats in JSON)

Key Functions:
- canonical_json(): Serialize objects to deterministic JSON
- content_hash(): Generate SHA-256 hash of content
- generate_experiment_id(): Create exp_<name>_dv<version>_<hash> identifiers
- generate_trial_id(): Create trial_XXXXXX identifiers
- generate_run_id(): Create run_XXXXXX identifiers
"""

import hashlib
import json
import math
import re
import subprocess
from datetime import date, datetime
from decimal import Decimal
from functools import lru_cache
from typing import Any, Optional, Union

# Sentinel values for special floats (JSON doesn't support NaN/Inf)
_NAN_SENTINEL = "__NaN__"
_POS_INF_SENTINEL = "__Inf__"
_NEG_INF_SENTINEL = "__-Inf__"


@lru_cache(maxsize=1)
def get_git_sha(short: bool = True) -> Optional[str]:
    """
    Get the current git commit SHA.

    This is cached for the lifetime of the process since the commit
    doesn't change during execution.

    Args:
        short: If True, return short SHA (first 8 chars). Default True.

    Returns:
        Git SHA string, or None if not in a git repo or git unavailable.

    Example:
        >>> sha = get_git_sha()
        >>> sha  # '57eabc7f' or None
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=8" if short else "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,  # Don't hang if git is slow
            cwd=None,  # Use current working directory
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        # git not available, not a git repo, or other error
        pass
    return None


def quantize_float(value: float, precision: int = 8) -> Union[float, str]:
    """
    Quantize a float to fixed decimal precision for deterministic hashing.

    This is critical for reproducibility - floating point representation
    differences across platforms can cause hash mismatches.

    Uses Decimal arithmetic internally to avoid float precision errors,
    then converts back to float for JSON type consistency.

    Args:
        value: Float value to quantize
        precision: Number of decimal places (default: 8)

    Returns:
        Quantized float value, or string sentinel for NaN/Inf

    Examples:
        >>> quantize_float(0.1 + 0.2)  # 0.30000000000000004 → 0.3
        0.3
        >>> quantize_float(1.23456789012345)
        1.23456789
        >>> quantize_float(float('nan'))
        '__NaN__'
    """
    if math.isnan(value):
        return _NAN_SENTINEL
    if math.isinf(value):
        return _POS_INF_SENTINEL if value > 0 else _NEG_INF_SENTINEL

    # Use Decimal for precise quantization, then convert back to float
    d = Decimal(str(value))
    quantized = round(d, precision)
    return float(quantized)


def canonical_json(obj: Any, sort_keys: bool = True, precision: int = 8) -> str:
    """
    Serialize object to canonical JSON for deterministic hashing.

    Handles:
    - Float quantization to prevent precision differences (preserves float type)
    - Decimal conversion to quantized float
    - datetime/date conversion to ISO format strings
    - set conversion to sorted list
    - Sorted dictionary keys for consistent ordering
    - Special float values (NaN, inf, -inf) as string sentinels
    - Nested structures recursively

    Args:
        obj: Object to serialize
        sort_keys: Whether to sort dictionary keys (default: True)
        precision: Float quantization precision (default: 8)

    Returns:
        Deterministic JSON string

    Example:
        >>> canonical_json({"b": 1, "a": 0.1 + 0.2})
        '{"a":0.3,"b":1}'
        >>> canonical_json({"d": date(2024, 1, 15)})
        '{"d":"2024-01-15"}'
    """

    def normalize(value: Any) -> Any:
        """Recursively normalize values for deterministic JSON."""
        if isinstance(value, float):
            return quantize_float(value, precision)
        elif isinstance(value, Decimal):
            # Convert Decimal to quantized float
            return quantize_float(float(value), precision)
        elif isinstance(value, datetime):
            # ISO format with microseconds for full precision
            return value.isoformat()
        elif isinstance(value, date):
            # ISO format for dates
            return value.isoformat()
        elif isinstance(value, set):
            # Convert set to sorted list for deterministic ordering
            return sorted(normalize(v) for v in value)
        elif isinstance(value, dict):
            return {str(k): normalize(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [normalize(v) for v in value]
        elif hasattr(value, "__dict__") and not isinstance(value, type):
            # Handle dataclasses and objects (but not class types)
            return normalize(vars(value))
        else:
            return value

    normalized = normalize(obj)
    return json.dumps(normalized, sort_keys=sort_keys, separators=(",", ":"))


def _slugify(value: str, max_length: int = 40) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower())
    slug = slug.strip("_")
    if not slug:
        slug = "experiment"
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("_")
    return slug


def content_hash(content: str, length: int = 12) -> str:
    """
    Generate SHA-256 hash of content, truncated to specified length.

    Args:
        content: String content to hash
        length: Number of hex characters to return (default: 12)

    Returns:
        Lowercase hex hash string
    """
    full_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return full_hash[:length]


def generate_experiment_id(
    name: str,
    strategy: str,
    parameters: dict,
    universe: dict,
    temporal: dict,
    data_version: str,
    profile_version: Optional[str] = None,
    code_version: Optional[str] = None,
) -> str:
    """
    Generate deterministic experiment ID from configuration.

    The ID incorporates all factors that affect experiment identity:
    - Strategy and parameter space
    - Universe configuration
    - Temporal splitting rules
    - Data version
    - Profile version (optional)
    - Code version (optional, typically git SHA)

    Including code_version ensures that running the same experiment
    with different code produces a different ID, enabling proper
    tracking of which code version produced which results.

    Args:
        name: Experiment name
        strategy: Strategy name
        parameters: Parameter definitions
        universe: Universe configuration
        temporal: Temporal configuration
        data_version: Data version string
        profile_version: Optional profile version
        code_version: Optional code/strategy version (e.g., git SHA).
            If None, auto-detects from git. Pass empty string "" to
            explicitly exclude code version from the hash.

    Returns:
        Experiment ID in format "exp_<name>_dv<version>_<hash>"
    """
    content = {
        "name": name,
        "strategy": strategy,
        "parameters": parameters,
        "universe": universe,
        "temporal": temporal,
        "data_version": data_version,
    }
    if profile_version:
        content["profile_version"] = profile_version

    # Include code version in hash if provided or auto-detected
    # Pass code_version="" to explicitly exclude
    if code_version is None:
        code_version = get_git_sha()
    if code_version:  # Only include if non-empty
        content["code_version"] = code_version

    json_str = canonical_json(content)
    hash_str = content_hash(json_str)
    name_slug = _slugify(name, max_length=48)
    data_slug = _slugify(data_version or "data", max_length=24)
    return f"exp_{name_slug}_dv{data_slug}_{hash_str}"


def get_next_version(db_manager: Any, base_experiment_id: str) -> int:
    """
    Determine the next run version for a base experiment ID.

    Queries the database for existing experiments with the same base_experiment_id
    and returns the next version number (starting at 1).

    Args:
        db_manager: DatabaseManager instance with fetchone method
        base_experiment_id: The versionless experiment ID (content hash)

    Returns:
        Next integer version (1 for first run, N+1 for subsequent runs)
    """
    row = db_manager.fetchone(
        """
        SELECT MAX(COALESCE(run_version, 1))
        FROM experiments
        WHERE base_experiment_id = ?
        """,
        (base_experiment_id,),
    )
    max_version = row[0] if row and row[0] is not None else 0
    return int(max_version) + 1


def generate_trial_id(
    experiment_id: str,
    params: dict,
    trial_index: Optional[int] = None,
) -> str:
    """
    Generate deterministic trial ID from experiment, parameters, and index.

    Args:
        experiment_id: Parent experiment ID
        params: Trial-specific parameter values
        trial_index: Optional trial index (required for Bayesian optimization
                     which may suggest duplicate params)

    Returns:
        Trial ID in format "trial_XXXXXXXXXXXX"
    """
    content = {
        "experiment_id": experiment_id,
        "params": params,
    }
    # Include trial_index if provided (Bayesian can suggest duplicates)
    if trial_index is not None:
        content["trial_index"] = trial_index

    json_str = canonical_json(content)
    hash_str = content_hash(json_str)
    return f"trial_{hash_str}"


def generate_run_id(
    trial_id: str,
    symbol: str,
    window_id: str,
    profile_version: str,
    data_version: str,
    is_train: Optional[bool] = None,
) -> str:
    """
    Generate deterministic run ID from all run-specific factors.

    Args:
        trial_id: Parent trial ID
        symbol: Symbol being tested
        window_id: Time window identifier
        profile_version: Execution profile version
        data_version: Data version string
        is_train: Optional IS/OOS flag to disambiguate runs on same window

    Returns:
        Run ID in format "run_XXXXXXXXXXXX"
    """
    content = {
        "trial_id": trial_id,
        "symbol": symbol,
        "window_id": window_id,
        "profile_version": profile_version,
        "data_version": data_version,
    }
    # Include is_train if provided (disambiguates IS vs OOS runs)
    if is_train is not None:
        content["is_train"] = is_train

    json_str = canonical_json(content)
    hash_str = content_hash(json_str)
    return f"run_{hash_str}"
