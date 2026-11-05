"""Shared fixtures and fakes. Unit tests never touch S3 or edge-tts."""

from __future__ import annotations

import asyncio
from io import BytesIO
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.errors import ObjectNotFoundError, PayloadTooLargeError
from app.main import AppServices, create_app
from app.services.limiter import JobLimiter
from app.services.ocr_engine import OcrResult

API_KEY = "test-key"
AUTH_HEADERS = {"x-api-key": API_KEY}


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "app_env": "test",
        "ocr_service_api_key": API_KEY,
        "ocr_min_confidence": 0.5,
        "ocr_max_side": 1600,
        "max_image_bytes": 10 * 1024 * 1024,
        "tts_max_chars": 20_000,
        "tts_chunk_chars": 900,
        "tts_default_voice": "female",
        "max_waiting_requests": 2,
        "ocr_timeout_seconds": 10.0,
        "tts_timeout_seconds": 10.0,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def make_png(size: tuple[int, int] = (160, 80)) -> bytes:
    image = Image.new("RGB", size, (255, 255, 255))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_jpeg(size: tuple[int, int] = (160, 80)) -> bytes:
    image = Image.new("RGB", size, (255, 255, 255))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


class FakeStorage:
    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects = dict(objects or {})
        self.get_calls: list[tuple[str, int]] = []
        self.put_calls: list[tuple[str, bytes, str]] = []
        self.get_error: Exception | None = None
        self.put_error: Exception | None = None

    async def get_bytes(self, key: str, max_bytes: int) -> bytes:
        self.get_calls.append((key, max_bytes))
        if self.get_error is not None:
            raise self.get_error
        if key not in self.objects:
            raise ObjectNotFoundError()
        data = self.objects[key]
        if len(data) > max_bytes:
            raise PayloadTooLargeError("Image too large")
        return data

    async def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        if self.put_error is not None:
            raise self.put_error
        self.put_calls.append((key, data, content_type))
        self.objects[key] = data


class FakeOcr:
    def __init__(
        self,
        result: OcrResult | None = None,
        delay: float = 0.0,
        on_extract=None,
    ) -> None:
        self.result = result or OcrResult(
            text="Halo, ini teks contoh.", lines=2, avg_confidence=0.93
        )
        self.delay = delay
        self.on_extract = on_extract
        self.calls: list[Any] = []

    def extract(self, image: Any) -> OcrResult:
        self.calls.append(image)
        if self.on_extract is not None:
            return self.on_extract(image)
        if self.delay:
            import time

            time.sleep(self.delay)
        return self.result


class FakeTts:
    def __init__(
        self,
        audio: bytes = b"ID3\x04\x00\x00\x00\x00\x00\x00fake-mp3",
        duration: float = 1.5,
        delay: float = 0.0,
        error: Exception | None = None,
        duration_error: Exception | None = None,
    ) -> None:
        self.audio = audio
        self.duration = duration
        self.delay = delay
        self.error = error
        self.duration_error = duration_error
        self.calls: list[tuple[str, str, str]] = []
        self.langs: list[str] = []

    async def synthesize(
        self, text: str, voice: str, rate: str, lang: str = "id"
    ) -> bytes:
        self.calls.append((text, voice, rate))
        self.langs.append(lang)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.audio

    async def duration_seconds(self, audio: bytes) -> float:
        if self.duration_error is not None:
            raise self.duration_error
        return self.duration


def make_services(
    settings: Settings | None = None,
    ocr: FakeOcr | None = None,
    tts: FakeTts | None = None,
    storage: FakeStorage | None = None,
    **service_kwargs: Any,
) -> AppServices:
    settings = settings or make_settings()
    return AppServices(
        ocr=ocr or FakeOcr(),
        tts=tts or FakeTts(),
        storage=storage or FakeStorage(),
        limiter=JobLimiter(settings.max_waiting_requests),
        **service_kwargs,
    )


def make_client(
    settings: Settings | None = None,
    services: AppServices | None = None,
    raise_server_exceptions: bool = False,
) -> TestClient:
    settings = settings or make_settings()
    services = services or make_services(settings)
    app = create_app(settings=settings, services=services)
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)
