"""Async wrapper around a blocking boto3 S3 client (Backblaze B2)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.config import Settings
from app.errors import (
    ObjectNotFoundError,
    PayloadTooLargeError,
    StorageUnavailableError,
)

logger = logging.getLogger(__name__)


def _client_config() -> Config:
    """B2 rejects some of botocore's newer default checksum headers.

    The two checksum options were added in recent botocore releases; fall back
    cleanly when running against an older SDK.
    """

    common: dict[str, Any] = {
        "signature_version": "s3v4",
        "s3": {"addressing_style": "path"},
        "retries": {"max_attempts": 3, "mode": "standard"},
    }
    try:
        return Config(
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            **common,
        )
    except (TypeError, ValueError):
        return Config(**common)


class S3Storage:
    def __init__(self, settings: Settings) -> None:
        missing = [
            name
            for name, value in (
                ("S3_ENDPOINT", settings.s3_endpoint),
                ("S3_REGION", settings.s3_region),
                ("S3_ACCESS_KEY", settings.s3_access_key),
                ("S3_SECRET_KEY", settings.s3_secret_key),
                ("S3_BUCKET", settings.s3_bucket),
            )
            if not value
        ]
        if missing:
            # Never print values, only missing variable names.
            raise RuntimeError(
                "Object storage configuration is incomplete: " + ", ".join(missing)
            )

        self._bucket = settings.s3_bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name=settings.s3_region,
            config=_client_config(),
        )

    async def get_bytes(self, key: str, max_bytes: int) -> bytes:
        return await asyncio.to_thread(self._get_bytes_sync, key, max_bytes)

    async def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        await asyncio.to_thread(self._put_bytes_sync, key, data, content_type)

    def _get_bytes_sync(self, key: str, max_bytes: int) -> bytes:
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            raise _map_client_error(exc) from exc
        except BotoCoreError as exc:
            raise StorageUnavailableError() from exc

        body = response.get("Body")
        try:
            content_length = response.get("ContentLength")
            if content_length is not None and int(content_length) > max_bytes:
                raise PayloadTooLargeError("Image too large")

            if body is None:
                raise StorageUnavailableError()

            try:
                data = body.read(max_bytes + 1)
            except (BotoCoreError, OSError) as exc:
                raise StorageUnavailableError() from exc
        finally:
            if body is not None:
                try:
                    body.close()
                except Exception:
                    pass

        if len(data) > max_bytes:
            raise PayloadTooLargeError("Image too large")
        return data

    def _put_bytes_sync(self, key: str, data: bytes, content_type: str) -> None:
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
                ContentLength=len(data),
            )
        except ClientError as exc:
            raise _map_client_error(exc) from exc
        except BotoCoreError as exc:
            raise StorageUnavailableError() from exc

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


def _map_client_error(exc: ClientError) -> Exception:
    error = exc.response.get("Error", {})
    code = str(error.get("Code", ""))
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")

    if code in {"NoSuchKey", "NotFound", "404"} or status == 404:
        return ObjectNotFoundError()

    # Everything else (permissions, throttling, 5xx, timeouts, bad endpoint)
    # is retryable from the NestJS side. Do not leak bucket details.
    return StorageUnavailableError()
