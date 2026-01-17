"""
Verification Framework for APEX RegimeDetector.

Provides machine-readable verification of regime detection outputs,
including schema validation, assertions, and performance checks.

Usage:
    python -m src.verification.regime_verifier --phase P1 --profile dev
"""

from .regime_verifier import ManifestVerifier, VerificationResult

__all__ = ["ManifestVerifier", "VerificationResult"]
