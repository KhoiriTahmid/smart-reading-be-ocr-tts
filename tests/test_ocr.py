"""POST /ocr contract tests using a fake storage and fake OCR engine."""

from __future__ import annotations

import asyncio
import threading

import httpx
import pytest

from app.services.limiter import JobLimiter
from app.services.ocr_engine import OcrEngine, OcrResult
from tests.conftest import (
    AUTH_HEADERS,
    FakeOcr,
    FakeStorage,
    make_client,
    make_png,
    make_services,
    make_settings,
)


def test_ocr_success() -> None:
    png = make_png()
    storage = FakeStorage({"uploads/abc/source.png": png})
    ocr = FakeOcr(OcrResult("Baris satu\nBaris dua", 2, 0.93))
    settings = make_settings()
    services = make_services(settings, ocr=ocr, storage=storage)

    with make_client(settings=settings, services=services) as client:
        response = client.post(
            "/ocr",
            json={"file_key": "uploads/abc/source.png"},
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 200
    assert response.json() == {
        "text": "Baris satu\nBaris dua",
        "lines": 2,
        "avg_confidence": 0.93,
    }
    assert storage.get_calls == [
        ("uploads/abc/source.png", settings.max_image_bytes)
    ]
    assert len(ocr.calls) == 1
    assert ocr.calls[0].shape[2] == 3


def test_ocr_empty_result_is_200_with_empty_text() -> None:
    storage = FakeStorage({"uploads/abc/source.jpg": make_png()})
    ocr = FakeOcr(OcrResult("", 0, 0.0))
    services = make_services(ocr=ocr, storage=storage)

    with make_client(services=services) as client:
        response = client.post(
            "/ocr",
            json={"file_key": "uploads/abc/source.jpg"},
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 200
    assert response.json() == {"text": "", "lines": 0, "avg_confidence": 0.0}


def test_ocr_object_not_found_404() -> None:
    with make_client() as client:
        response = client.post(
            "/ocr",
            json={"file_key": "uploads/missing/source.png"},
            headers=AUTH_HEADERS,
        )
    assert response.status_code == 404
    assert response.json()["detail"] == "Object not found"


def test_ocr_non_image_returns_415() -> None:
    storage = FakeStorage({"uploads/abc/source.jpg": b"this is not an image"})
    services = make_services(storage=storage)

    with make_client(services=services) as client:
        response = client.post(
            "/ocr",
            json={"file_key": "uploads/abc/source.jpg"},
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 415
    assert response.json()["detail"] == "Unsupported or corrupt image"


def test_ocr_corrupt_png_returns_415() -> None:
    corrupt = b"\x89PNG\r\n\x1a\nnot really a png"
    storage = FakeStorage({"uploads/abc/source.png": corrupt})
    services = make_services(storage=storage)

    with make_client(services=services) as client:
        response = client.post(
            "/ocr",
            json={"file_key": "uploads/abc/source.png"},
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 415


def test_ocr_image_too_large_returns_413() -> None:
    settings = make_settings(max_image_bytes=4)
    storage = FakeStorage({"uploads/abc/source.png": make_png()})
    services = make_services(settings, storage=storage)

    with make_client(settings=settings, services=services) as client:
        response = client.post(
            "/ocr",
            json={"file_key": "uploads/abc/source.png"},
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 413
    assert response.json()["detail"] == "Image too large"


@pytest.mark.parametrize(
    "file_key",
    [
        "audio/abc.mp3",
        "uploads/../secret.png",
        "uploads/",
        "uploads/abc/../../etc/passwd",
        "",
    ],
)
def test_ocr_invalid_file_key_returns_422(file_key: str) -> None:
    with make_client() as client:
        response = client.post(
            "/ocr", json={"file_key": file_key}, headers=AUTH_HEADERS
        )
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid request"}


def test_ocr_missing_body_returns_422() -> None:
    with make_client() as client:
        response = client.post("/ocr", json={}, headers=AUTH_HEADERS)
    assert response.status_code == 422


def test_ocr_timeout_returns_503() -> None:
    settings = make_settings(ocr_timeout_seconds=0.05)
    storage = FakeStorage({"uploads/abc/source.png": make_png()})
    ocr = FakeOcr(delay=0.3)
    services = make_services(settings, ocr=ocr, storage=storage)

    with make_client(settings=settings, services=services) as client:
        response = client.post(
            "/ocr",
            json={"file_key": "uploads/abc/source.png"},
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "OCR request timed out"


@pytest.mark.asyncio
async def test_waiting_room_full_returns_429_with_retry_after() -> None:
    settings = make_settings(max_waiting_requests=2, ocr_timeout_seconds=10)
    started = threading.Event()
    release = threading.Event()

    def blocking_extract(image):
        started.set()
        release.wait(timeout=10)
        return OcrResult("ok", 1, 0.9)

    storage = FakeStorage({"uploads/abc/source.png": make_png()})
    ocr = FakeOcr(on_extract=blocking_extract)
    services = make_services(settings, ocr=ocr, storage=storage)
    from app.main import create_app

    app = create_app(settings=settings, services=services)
    payload = {"file_key": "uploads/abc/source.png"}

    try:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test", timeout=10
            ) as client:
                first = asyncio.create_task(
                    client.post("/ocr", json=payload, headers=AUTH_HEADERS)
                )
                assert await asyncio.to_thread(started.wait, 5)

                second = asyncio.create_task(
                    client.post("/ocr", json=payload, headers=AUTH_HEADERS)
                )
                third = asyncio.create_task(
                    client.post("/ocr", json=payload, headers=AUTH_HEADERS)
                )

                for _ in range(200):
                    if services.limiter.waiting == 2:
                        break
                    await asyncio.sleep(0.01)
                assert services.limiter.waiting == 2

                rejected = await client.post(
                    "/ocr", json=payload, headers=AUTH_HEADERS
                )
                assert rejected.status_code == 429
                assert rejected.headers["Retry-After"] == "5"
                assert rejected.json() == {"detail": "Service busy"}

                release.set()
                responses = await asyncio.gather(first, second, third)
                assert all(response.status_code == 200 for response in responses)
    finally:
        release.set()


@pytest.mark.asyncio
async def test_limiter_releases_slot_after_request() -> None:
    limiter = JobLimiter(max_waiting=0)
    async with limiter.slot():
        assert limiter.active is True
    assert limiter.active is False

    async with limiter.slot():
        assert limiter.active is True


class _StubRapidResult:
    def __init__(self, txts, scores) -> None:
        self.txts = txts
        self.scores = scores


def test_ocr_engine_drops_low_confidence_and_empty_lines() -> None:
    import numpy as np

    engine = OcrEngine(make_settings(ocr_min_confidence=0.5))

    class StubEngine:
        def __call__(self, image):
            return _StubRapidResult(
                txts=("Tinggi", " Rendah ", None, "Sedang"),
                scores=(0.91, 0.2, 0.99, 0.75),
            )

    engine._engine = StubEngine()
    result = engine.extract(np.zeros((8, 8, 3), dtype=np.uint8))

    assert result.text == "Tinggi\nSedang"
    assert result.lines == 2
    assert result.avg_confidence == 0.83


def test_ocr_engine_handles_empty_result() -> None:
    import numpy as np

    engine = OcrEngine(make_settings())

    class EmptyEngine:
        def __call__(self, image):
            return _StubRapidResult(txts=None, scores=None)

    engine._engine = EmptyEngine()
    result = engine.extract(np.zeros((8, 8, 3), dtype=np.uint8))

    assert result == OcrResult("", 0, 0.0)


def test_ocr_pdf_success_returns_pages() -> None:
    from tests.test_pdf import make_pdf

    pdf_bytes = make_pdf([
        "Halaman Pertama Buku Panduan Smart Reading",
        "Halaman Kedua Mengenai Penggunaan Suara TTS",
    ])
    storage = FakeStorage({"uploads/docs/document.pdf": pdf_bytes})
    settings = make_settings()
    services = make_services(settings, storage=storage)

    with make_client(settings=settings, services=services) as client:
        response = client.post(
            "/ocr",
            json={"file_key": "uploads/docs/document.pdf"},
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["total_pages"] == 2
    assert "Halaman Pertama" in data["text"]
    assert "Halaman Kedua" in data["text"]
    assert len(data["pages"]) == 2
    assert data["pages"][0]["page_number"] == 1
    assert "Halaman Pertama" in data["pages"][0]["text"]
    assert data["pages"][1]["page_number"] == 2
    assert "Halaman Kedua" in data["pages"][1]["text"]
