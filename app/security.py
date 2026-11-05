"""API-key authentication for the private service."""

from __future__ import annotations

import hmac

from fastapi import Request, Security
from fastapi.security import APIKeyHeader

from app.errors import AuthenticationError

_api_key_header = APIKeyHeader(name="x-api-key", auto_error=False, scheme_name="ApiKeyAuth")


async def require_api_key(
    request: Request,
    provided_key: str | None = Security(_api_key_header),
) -> None:
    """Validate ``x-api-key`` in constant time.

    Missing, empty or wrong keys all produce 401. The configured key is read
    from the app state so tests can inject settings without monkeypatching.
    """

    expected = request.app.state.settings.ocr_service_api_key
    if not provided_key or not expected:
        raise AuthenticationError()

    if not hmac.compare_digest(provided_key.encode("utf-8"), expected.encode("utf-8")):
        raise AuthenticationError()
