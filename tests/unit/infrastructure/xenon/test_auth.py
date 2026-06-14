from __future__ import annotations

import pytest

from src.infrastructure.adapters.xenon.auth import AuthProvider, NoAuthProvider


@pytest.mark.asyncio
async def test_no_auth_provider_returns_none() -> None:
    assert await NoAuthProvider().ticket() is None


def test_no_auth_provider_satisfies_protocol() -> None:
    assert isinstance(NoAuthProvider(), AuthProvider)
