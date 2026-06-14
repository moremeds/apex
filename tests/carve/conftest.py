"""Fixtures for Phase 0 carve isolation tests."""

from __future__ import annotations

import sys
from typing import Iterator

import pytest

from tests.carve.stubs import install, uninstall

# Top-level src packages this harness mutates (clears + stubs).
_TOUCHED_PREFIXES = (
    "src.domain",
    "src.infrastructure",
    "src.services",
    "src.application",
)


@pytest.fixture
def install_carve_stubs() -> Iterator[None]:
    """Import the keep-set under stub modules, then fully restore sys.modules.

    The harness deletes real modules and injects stubs so the cores import
    against fakes. Without restoring afterwards, later tests in the same pytest
    session would receive stub-bound cached modules (test pollution). So we
    snapshot every touched src.* module up front and restore the exact snapshot
    on teardown, deleting anything imported during the test.
    """
    snapshot = {
        name: mod for name, mod in sys.modules.items() if name.startswith(_TOUCHED_PREFIXES)
    }
    # Drop already-imported keep-set/infra modules so stubs take effect.
    for name in list(sys.modules):
        if name.startswith(_TOUCHED_PREFIXES):
            del sys.modules[name]

    installed = install()
    try:
        yield
    finally:
        uninstall(installed)
        # Delete everything imported during the test, then restore the snapshot
        # so subsequent tests re-bind to the real modules.
        for name in list(sys.modules):
            if name.startswith(_TOUCHED_PREFIXES):
                del sys.modules[name]
        sys.modules.update(snapshot)
