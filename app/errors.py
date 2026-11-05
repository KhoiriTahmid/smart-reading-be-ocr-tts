"""Small, safe error types mapped to the HTTP contract.

Messages in this module are deliberately generic: they are returned to the
NestJS caller and must never contain user text, object keys, stack traces or
credentials.
"""

from __future__ import annotations


class ServiceError(Exception):
    """Base class for errors that map to a controlled HTTP response."""

    status_code: int = 500
    default_detail: str = "Internal server error"
    retry_after: int | None = None

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or self.default_detail
        super().__init__(self.detail)


class AuthenticationError(ServiceError):
    status_code = 401
    default_detail = "Invalid or missing API key"


class InvalidRequestError(ServiceError):
    status_code = 422
    default_detail = "Invalid request"


class ObjectNotFoundError(ServiceError):
    status_code = 404
    default_detail = "Object not found"


class UnsupportedImageError(ServiceError):
    status_code = 415
    default_detail = "Unsupported or corrupt image"


class PayloadTooLargeError(ServiceError):
    status_code = 413
    default_detail = "Payload too large"


class ServiceBusyError(ServiceError):
    status_code = 429
    default_detail = "Service busy"
    retry_after = 5


class ServiceUnavailableError(ServiceError):
    status_code = 503
    default_detail = "Service temporarily unavailable"


class TTSUnavailableError(ServiceUnavailableError):
    default_detail = "TTS synthesis failed"


class StorageUnavailableError(ServiceUnavailableError):
    default_detail = "Storage unavailable"
