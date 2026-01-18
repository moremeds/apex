"""
Futu adapter exception hierarchy.

Provides typed exceptions for proper error classification instead of
string matching on generic Exception messages.
"""

from __future__ import annotations


class FutuError(Exception):
    """Base exception for all Futu adapter errors."""


class FutuConnectionError(FutuError):
    """Connection to Futu OpenD lost or unavailable."""


class FutuRateLimitError(FutuError):
    """Futu API rate limit exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", cooldown_seconds: int = 30):
        super().__init__(message)
        self.cooldown_seconds = cooldown_seconds


class FutuDataError(FutuError):
    """Invalid or missing data from Futu API."""


class FutuAuthError(FutuError):
    """Authentication or authorization error with Futu."""


def classify_futu_exception(e: Exception) -> FutuError:
    """
    Classify a generic exception into a typed FutuError.

    This function examines exception messages to determine the appropriate
    typed exception. Use this when catching exceptions from the Futu SDK
    which doesn't provide typed exceptions.

    Args:
        e: The original exception from Futu SDK

    Returns:
        A typed FutuError subclass with the original exception as __cause__
    """
    error_str = str(e).lower()

    exc: FutuError
    if "disconnect" in error_str or "connection" in error_str or "socket" in error_str:
        exc = FutuConnectionError(f"Connection error: {e}")
    elif "frequent" in error_str or "rate" in error_str or "too many" in error_str:
        exc = FutuRateLimitError(f"Rate limited: {e}")
    elif "auth" in error_str or "permission" in error_str or "unlock" in error_str:
        exc = FutuAuthError(f"Authentication error: {e}")
    else:
        exc = FutuDataError(f"Futu error: {e}")

    # Chain the original exception for full stack trace
    exc.__cause__ = e
    return exc
