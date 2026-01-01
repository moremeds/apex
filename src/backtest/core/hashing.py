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
- generate_experiment_id(): Create exp_XXXXXX identifiers
- generate_trial_id(): Create trial_XXXXXX identifiers
- generate_run_id(): Create run_XXXXXX identifiers
"""

import hashlib
import json
import math
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
        Experiment ID in format "exp_XXXXXXXXXXXX"
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
    return f"exp_{hash_str}"


def generate_trial_id(experiment_id: str, params: dict) -> str:
    """
    Generate deterministic trial ID from experiment and parameters.

    Args:
        experiment_id: Parent experiment ID
        params: Trial-specific parameter values

    Returns:
        Trial ID in format "trial_XXXXXXXXXXXX"
    """
    content = {
        "experiment_id": experiment_id,
        "params": params,
    }
    json_str = canonical_json(content)
    hash_str = content_hash(json_str)
    return f"trial_{hash_str}"


def generate_run_id(
    trial_id: str,
    symbol: str,
    window_id: str,
    profile_version: str,
    data_version: str,
) -> str:
    """
    Generate deterministic run ID from all run-specific factors.

    Args:
        trial_id: Parent trial ID
        symbol: Symbol being tested
        window_id: Time window identifier
        profile_version: Execution profile version
        data_version: Data version string

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
    json_str = canonical_json(content)
    hash_str = content_hash(json_str)
    return f"run_{hash_str}"
