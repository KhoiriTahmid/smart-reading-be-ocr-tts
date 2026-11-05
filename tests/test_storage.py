"""S3 storage wrapper tests with a mocked boto3 client."""

from __future__ import annotations

import pytest
from botocore.exceptions import BotoCoreError, ClientError

from app.errors import ObjectNotFoundError, PayloadTooLargeError, StorageUnavailableError
from app.services import storage as storage_module
from app.services.storage import S3Storage
from tests.conftest import make_settings


def _settings(**overrides):
    values = {
        "s3_endpoint": "https://s3.example.invalid",
        "s3_region": "us-west-004",
        "s3_access_key": "key",
        "s3_secret_key": "secret",
        "s3_bucket": "bucket",
    }
    values.update(overrides)
    return make_settings(**values)


class FakeBody:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.read_limits: list[int] = []
        self.closed = False

    def read(self, amount: int) -> bytes:
        self.read_limits.append(amount)
        return self.data[:amount]

    def close(self) -> None:
        self.closed = True


class FakeBotoClient:
    def __init__(self) -> None:
        self.get_kwargs = None
        self.put_kwargs = None
        self.get_response = None
        self.get_error = None
        self.put_error = None

    def get_object(self, **kwargs):
        self.get_kwargs = kwargs
        if self.get_error is not None:
            raise self.get_error
        return self.get_response

    def put_object(self, **kwargs):
        self.put_kwargs = kwargs
        if self.put_error is not None:
            raise self.put_error
        return {}

    def close(self) -> None:
        pass


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeBotoClient:
    client = FakeBotoClient()
    monkeypatch.setattr(storage_module.boto3, "client", lambda *a, **k: client)
    return client


def test_storage_requires_all_configuration() -> None:
    with pytest.raises(RuntimeError) as exc_info:
        S3Storage(_settings(s3_bucket=""))
    assert "S3_BUCKET" in str(exc_info.value)
    assert "secret" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_bytes_success(fake_client: FakeBotoClient) -> None:
    body = FakeBody(b"hello")
    fake_client.get_response = {"Body": body, "ContentLength": 5}
    storage = S3Storage(_settings())

    result = await storage.get_bytes("uploads/a/source.png", max_bytes=10)

    assert result == b"hello"
    assert fake_client.get_kwargs == {
        "Bucket": "bucket",
        "Key": "uploads/a/source.png",
    }
    assert body.read_limits == [11]
    assert body.closed is True


@pytest.mark.asyncio
async def test_get_bytes_content_length_too_large(fake_client: FakeBotoClient) -> None:
    body = FakeBody(b"0123456789")
    fake_client.get_response = {"Body": body, "ContentLength": 20}
    storage = S3Storage(_settings())

    with pytest.raises(PayloadTooLargeError):
        await storage.get_bytes("uploads/a/source.png", max_bytes=10)
    assert body.closed is True


@pytest.mark.asyncio
async def test_get_bytes_stream_too_large(fake_client: FakeBotoClient) -> None:
    body = FakeBody(b"0123456789XYZ")
    fake_client.get_response = {"Body": body, "ContentLength": 5}
    storage = S3Storage(_settings())

    with pytest.raises(PayloadTooLargeError):
        await storage.get_bytes("uploads/a/source.png", max_bytes=10)


@pytest.mark.asyncio
async def test_get_bytes_missing_key_returns_404(fake_client: FakeBotoClient) -> None:
    fake_client.get_error = ClientError(
        {
            "Error": {"Code": "NoSuchKey"},
            "ResponseMetadata": {"HTTPStatusCode": 404},
        },
        "GetObject",
    )
    storage = S3Storage(_settings())

    with pytest.raises(ObjectNotFoundError):
        await storage.get_bytes("uploads/a/source.png", max_bytes=10)


@pytest.mark.asyncio
async def test_get_bytes_access_denied_is_retryable(fake_client: FakeBotoClient) -> None:
    fake_client.get_error = ClientError(
        {
            "Error": {"Code": "AccessDenied"},
            "ResponseMetadata": {"HTTPStatusCode": 403},
        },
        "GetObject",
    )
    storage = S3Storage(_settings())

    with pytest.raises(StorageUnavailableError):
        await storage.get_bytes("uploads/a/source.png", max_bytes=10)


@pytest.mark.asyncio
async def test_get_bytes_network_error_is_retryable(fake_client: FakeBotoClient) -> None:
    fake_client.get_error = BotoCoreError()
    storage = S3Storage(_settings())

    with pytest.raises(StorageUnavailableError):
        await storage.get_bytes("uploads/a/source.png", max_bytes=10)


@pytest.mark.asyncio
async def test_put_bytes_sets_content_type_and_length(fake_client: FakeBotoClient) -> None:
    storage = S3Storage(_settings())

    await storage.put_bytes("audio/abc.mp3", b"mp3data", "audio/mpeg")

    assert fake_client.put_kwargs == {
        "Bucket": "bucket",
        "Key": "audio/abc.mp3",
        "Body": b"mp3data",
        "ContentType": "audio/mpeg",
        "ContentLength": 7,
    }


@pytest.mark.asyncio
async def test_put_bytes_failure_is_retryable(fake_client: FakeBotoClient) -> None:
    fake_client.put_error = BotoCoreError()
    storage = S3Storage(_settings())

    with pytest.raises(StorageUnavailableError):
        await storage.put_bytes("audio/abc.mp3", b"mp3data", "audio/mpeg")


def test_client_config_falls_back_for_old_botocore(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    class OldConfig:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            if "request_checksum_calculation" in kwargs:
                raise TypeError("unknown option")

    monkeypatch.setattr(storage_module, "Config", OldConfig)
    result = storage_module._client_config()

    assert isinstance(result, OldConfig)
    assert "request_checksum_calculation" not in calls[-1]
    assert calls[-1]["signature_version"] == "s3v4"
