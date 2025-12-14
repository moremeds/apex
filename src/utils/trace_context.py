"""
Trace context for correlating logs across a single refresh cycle.

Provides:
- Unique cycle IDs (6-char hex) for each refresh cycle
- Context propagation via contextvars (async-safe)
- Easy access to current cycle ID from any module

Usage:
    # In orchestrator (start of cycle)
    with new_cycle():
        await fetch_positions()
        build_snapshot()

    # In any module
    from src.utils.trace_context import get_cycle_id
    logger.info(f"[{get_cycle_id()}] Processing...")
"""

from __future__ import annotations

import secrets
from contextvars import ContextVar
from contextlib import contextmanager
from typing import Optional, Generator

# Context variable for the current cycle ID (async-safe)
_cycle_id: ContextVar[Optional[str]] = ContextVar("cycle_id", default=None)

# Counter for cycles within a session (for debugging)
_cycle_counter: int = 0


def generate_cycle_id() -> str:
    """
    Generate a new unique cycle ID.

    Returns:
        6-character hex string (e.g., "a7f3b2").
    """
    return secrets.token_hex(3)


def get_cycle_id() -> str:
    """
    Get the current cycle ID.

    Returns:
        Current cycle ID, or "------" if no cycle is active.
    """
    cycle_id = _cycle_id.get()
    return cycle_id if cycle_id else "------"


def set_cycle_id(cycle_id: str) -> None:
    """
    Set the current cycle ID.

    Args:
        cycle_id: The cycle ID to set.
    """
    _cycle_id.set(cycle_id)


def clear_cycle_id() -> None:
    """Clear the current cycle ID."""
    _cycle_id.set(None)


@contextmanager
def new_cycle() -> Generator[str, None, None]:
    """
    Context manager to create a new cycle with a unique ID.

    Automatically generates a new cycle ID, sets it as the current
    context, and clears it when the context exits.

    Yields:
        The new cycle ID.

    Example:
        with new_cycle() as cycle_id:
            logger.info(f"Starting cycle {cycle_id}")
            await process_data()
    """
    global _cycle_counter
    _cycle_counter += 1

    cycle_id = generate_cycle_id()
    token = _cycle_id.set(cycle_id)

    try:
        yield cycle_id
    finally:
        _cycle_id.reset(token)


def get_cycle_counter() -> int:
    """
    Get the total number of cycles created in this session.

    Useful for debugging and statistics.

    Returns:
        Total cycle count.
    """
    return _cycle_counter


def reset_cycle_counter() -> None:
    """Reset the cycle counter (for testing)."""
    global _cycle_counter
    _cycle_counter = 0
