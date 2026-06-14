"""Fixtures for Phase 0 carve isolation tests."""

from __future__ import annotations

import sys
from typing import Iterator

import pytest

from tests.carve.stubs import install, uninstall


@pytest.fixture
def install_carve_stubs() -> Iterator[None]:
    # Drop any already-imported keep-set/infra modules so stubs take effect.
    for name in list(sys.modules):
        if name.startswith(
            ("src.domain", "src.infrastructure", "src.services", "src.application")
        ):
            del sys.modules[name]
    installed = install()
    try:
        yield
    finally:
        uninstall(installed)
        for name in list(sys.modules):
            if name.startswith("src.domain"):
                del sys.modules[name]
