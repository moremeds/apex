"""Every failure mode carries a machine-readable code, never a bare 500."""

from __future__ import annotations

import json

import pytest

from src.api.errors import STATUS_BY_CODE, ApiError, ApiErrorCode, api_error_response


def test_every_code_has_a_status() -> None:
    assert set(STATUS_BY_CODE) == set(ApiErrorCode)


@pytest.mark.parametrize(
    ("code", "status"),
    [
        (ApiErrorCode.UNSUPPORTED_TIMEFRAME, 400),
        (ApiErrorCode.UNSUPPORTED_ASSET_CLASS, 400),
        (ApiErrorCode.ADJUSTED_NOT_SUPPORTED, 400),
        (ApiErrorCode.UNKNOWN_SYMBOL, 404),
        (ApiErrorCode.AMBIGUOUS_SYMBOL, 409),
        (ApiErrorCode.NOT_YET_AVAILABLE, 501),
        (ApiErrorCode.PROVIDER_NOT_CONFIGURED, 503),
        (ApiErrorCode.ADJUSTED_UNAVAILABLE, 503),
    ],
)
def test_status_mapping(code: ApiErrorCode, status: int) -> None:
    assert ApiError(code, "boom").status_code == status


def test_adjusted_unavailable_is_503_not_500() -> None:
    """503 is deliberate: Silver quarantine is an upstream condition livewire may
    repair, so 'retry later' is the honest semantic."""
    exc = ApiError(
        ApiErrorCode.ADJUSTED_UNAVAILABLE,
        "no Silver artifact; symbol is quarantined upstream",
        symbol="HON",
        asset_class="equity",
    )
    assert exc.status_code == 503


def test_envelope_shape() -> None:
    exc = ApiError(
        ApiErrorCode.ADJUSTED_UNAVAILABLE,
        "quarantined",
        symbol="HON",
        asset_class="equity",
    )
    body = json.loads(api_error_response(exc).body)
    assert body == {
        "error": {
            "code": "adjusted_unavailable",
            "message": "quarantined",
            "symbol": "HON",
            "asset_class": "equity",
        }
    }


def test_envelope_omits_absent_context() -> None:
    body = json.loads(api_error_response(ApiError(ApiErrorCode.PROVIDER_NOT_CONFIGURED, "x")).body)
    assert body["error"] == {"code": "provider_not_configured", "message": "x"}


def test_code_values_are_stable_contract() -> None:
    """These strings are published to argon; renaming one is a breaking change."""
    assert {c.value for c in ApiErrorCode} == {
        "invalid_parameter",
        "query_timeout",
        "unauthorized",
        "forbidden",
        "internal_error",
        "unsupported_timeframe",
        "unsupported_asset_class",
        "adjusted_not_supported",
        "unknown_symbol",
        "ambiguous_symbol",
        "not_yet_available",
        "provider_not_configured",
        "adjusted_unavailable",
        "unknown_index",
        "ambiguous_security",
        "membership_unavailable",
        "unknown_revision",
        "pit_unavailable",
        "revision_not_supported",
    }


def test_every_lake_error_code_is_a_rest_code() -> None:
    """The application raises LakeError with these codes; REST must map each one."""
    from typing import get_args

    from src.application.lake.errors import LakeError, LakeErrorCode

    for code in get_args(LakeErrorCode):
        api = ApiError.from_lake(LakeError(code, "m"))
        assert api.code.value == code and api.status_code >= 400


def test_revision_statuses() -> None:
    assert ApiError(ApiErrorCode.UNKNOWN_REVISION, "x").status_code == 404
    assert ApiError(ApiErrorCode.PIT_UNAVAILABLE, "x").status_code == 503
    assert ApiError(ApiErrorCode.REVISION_NOT_SUPPORTED, "x").status_code == 400


def test_envelope_carries_details_when_present() -> None:
    from src.application.lake.errors import LakeError

    exc = LakeError("ambiguous_symbol", "two", symbol="FSLR", details={"scopes": [1, 2]})
    body = json.loads(api_error_response(ApiError.from_lake(exc)).body)
    assert body["error"]["details"] == {"scopes": [1, 2]}
    assert json.loads(api_error_response(ApiError(ApiErrorCode.FORBIDDEN, "x")).body)["error"] == {
        "code": "forbidden",
        "message": "x",
    }


def test_response_status_matches_the_code() -> None:
    resp = api_error_response(ApiError(ApiErrorCode.AMBIGUOUS_SYMBOL, "two of them", symbol="AAC"))
    assert resp.status_code == 409


def test_unauthorized_carries_the_bearer_challenge() -> None:
    resp = api_error_response(ApiError(ApiErrorCode.UNAUTHORIZED, "nope"))
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"
    other = api_error_response(ApiError(ApiErrorCode.INVALID_PARAMETER, "nope"))
    assert "www-authenticate" not in other.headers


def test_handler_is_installed_and_renders_the_envelope() -> None:
    """Wiring test: a route that raises ApiError must produce the envelope and the
    mapped status, not a bare 500 from Starlette's default handler."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api.errors import install_error_handlers

    app = FastAPI()

    @app.get("/boom")
    async def boom() -> dict[str, str]:
        raise ApiError(ApiErrorCode.ADJUSTED_UNAVAILABLE, "quarantined upstream", symbol="HON")

    install_error_handlers(app)

    response = TestClient(app, raise_server_exceptions=False).get("/boom")
    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "adjusted_unavailable",
            "message": "quarantined upstream",
            "symbol": "HON",
        }
    }


def test_real_app_registers_the_handler() -> None:
    """create_app() must install it; otherwise every raise becomes a 500 in production."""
    from src.api.server import create_app

    assert ApiError in create_app().exception_handlers
