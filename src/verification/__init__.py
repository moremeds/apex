"""
Verification Framework for APEX.

Provides machine-readable verification of:
- Regime detection outputs
- Signal service contracts
- Indicator invariants

Includes:
- JSON Schema validation
- Semantic invariants (bounds, identities, no-NaN)
- Causality verification (no lookahead)
- Golden fixture regression

Usage:
    python -m src.verification.regime_verifier --phase P1 --profile dev
    python -m src.verification.signal_verifier --all --profile signal_dev
"""

from .base_verifier import BaseVerifier, PhaseResult, VerificationResult
from .invariants import (
    BoundsInvariant,
    CausalityInvariant,
    IdentityInvariant,
    InvariantType,
    NoNaNInvariant,
    create_invariant,
    evaluate_expression,
)

# Legacy import for backward compatibility
from .regime_verifier import ManifestVerifier

__all__ = [
    # Base classes
    "BaseVerifier",
    "VerificationResult",
    "PhaseResult",
    # Invariants
    "InvariantType",
    "BoundsInvariant",
    "IdentityInvariant",
    "NoNaNInvariant",
    "CausalityInvariant",
    "create_invariant",
    "evaluate_expression",
    # Legacy
    "ManifestVerifier",
]
