from __future__ import annotations

import pytest

from src.infrastructure.adapters.xenon.auth import (
    ApiKeyAuthProvider,
    AuthProvider,
    NoAuthProvider,
)


@pytest.mark.asyncio
async def test_no_auth_provider_returns_none() -> None:
    assert await NoAuthProvider().ticket() is None


def test_no_auth_provider_satisfies_protocol() -> None:
    assert isinstance(NoAuthProvider(), AuthProvider)


def test_no_auth_provider_sends_no_headers() -> None:
    assert NoAuthProvider().headers() == {}


@pytest.mark.asyncio
async def test_api_key_provider_sends_header_and_no_ticket() -> None:
    provider = ApiKeyAuthProvider("k-123")
    assert provider.headers() == {"X-API-Key": "k-123"}
    assert await provider.ticket() is None
    assert isinstance(provider, AuthProvider)


def test_api_key_provider_repr_redacts_key() -> None:
    assert "k-123" not in repr(ApiKeyAuthProvider("k-123"))
