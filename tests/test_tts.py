"""POST /tts contract tests with a fake edge-tts engine and fake storage."""

from __future__ import annotations

import uuid

import pytest

from app.errors import StorageUnavailableError, TTSUnavailableError
from app.services.limiter import JobLimiter
from tests.conftest import (
    AUTH_HEADERS,
    FakeStorage,
    FakeTts,
    make_client,
    make_services,
    make_settings,
)

HISTORY_ID = uuid.UUID("3f2b6c1e-9d4a-4f5b-8c7d-1234567890ab")


def _tts_payload(**overrides):
    payload = {
        "history_id": str(HISTORY_ID),
        "text": "Teks yang akan dibacakan.",
        "voice": "female",
        "rate": "+0%",
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize("voice", ["female", "male"])
def test_tts_success_both_voices(voice: str) -> None:
    tts = FakeTts(audio=b"ID3fake-mp3", duration=12.4)
    storage = FakeStorage()
    services = make_services(tts=tts, storage=storage)

    with make_client(services=services) as client:
        response = client.post(
            "/tts", json=_tts_payload(voice=voice), headers=AUTH_HEADERS
        )

    assert response.status_code == 200
    assert response.json() == {
        "audio_key": f"audio/{HISTORY_ID}.mp3",
        "duration_seconds": 12.4,
        "chars": 25,
    }
    assert tts.calls == [("Teks yang akan dibacakan.", voice, "+0%")]
    assert storage.put_calls == [
        (f"audio/{HISTORY_ID}.mp3", b"ID3fake-mp3", "audio/mpeg")
    ]


def test_tts_default_voice_comes_from_settings() -> None:
    settings = make_settings(tts_default_voice="male")
    tts = FakeTts()
    services = make_services(settings, tts=tts)

    with make_client(settings=settings, services=services) as client:
        response = client.post(
            "/tts", json=_tts_payload(voice=None), headers=AUTH_HEADERS
        )

    assert response.status_code == 200
    assert tts.calls[0][1] == "male"


def test_tts_default_rate_is_plus_zero() -> None:
    tts = FakeTts()
    services = make_services(tts=tts)

    with make_client(services=services) as client:
        response = client.post(
            "/tts", json=_tts_payload(rate=None), headers=AUTH_HEADERS
        )

    assert response.status_code == 200
    assert tts.calls[0][2] == "+0%"


def test_tts_uppercase_uuid_is_canonicalised() -> None:
    tts = FakeTts()
    services = make_services(tts=tts)

    with make_client(services=services) as client:
        response = client.post(
            "/tts",
            json=_tts_payload(history_id=str(HISTORY_ID).upper()),
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 200
    assert response.json()["audio_key"] == f"audio/{HISTORY_ID}.mp3"


def test_tts_same_history_id_overwrites_deterministic_key() -> None:
    tts = FakeTts()
    storage = FakeStorage()
    services = make_services(tts=tts, storage=storage)

    with make_client(services=services) as client:
        first = client.post("/tts", json=_tts_payload(), headers=AUTH_HEADERS)
        second = client.post("/tts", json=_tts_payload(), headers=AUTH_HEADERS)

    assert first.status_code == second.status_code == 200
    key = f"audio/{HISTORY_ID}.mp3"
    assert first.json()["audio_key"] == second.json()["audio_key"] == key
    assert [call[0] for call in storage.put_calls] == [key, key]
    assert list(storage.objects) == [key]


@pytest.mark.parametrize("rate", ["-10%", "+20%", "+0%"])
def test_tts_accepts_valid_rates(rate: str) -> None:
    tts = FakeTts()
    services = make_services(tts=tts)

    with make_client(services=services) as client:
        response = client.post(
            "/tts", json=_tts_payload(rate=rate), headers=AUTH_HEADERS
        )

    assert response.status_code == 200
    assert tts.calls[0][2] == rate


@pytest.mark.parametrize("rate", ["10%", "+100%", "fast", "+1.5%", ""])
def test_tts_rejects_invalid_rates(rate: str) -> None:
    with make_client() as client:
        response = client.post(
            "/tts", json=_tts_payload(rate=rate), headers=AUTH_HEADERS
        )
    assert response.status_code == 422


def test_tts_rejects_invalid_voice() -> None:
    with make_client() as client:
        response = client.post(
            "/tts", json=_tts_payload(voice="robot"), headers=AUTH_HEADERS
        )
    assert response.status_code == 422


def test_tts_rejects_bad_uuid() -> None:
    with make_client() as client:
        response = client.post(
            "/tts",
            json=_tts_payload(history_id="not-a-uuid"),
            headers=AUTH_HEADERS,
        )
    assert response.status_code == 422


def test_tts_text_too_large_returns_413_without_synthesizing() -> None:
    settings = make_settings(tts_max_chars=100, tts_chunk_chars=100)
    tts = FakeTts()
    services = make_services(settings, tts=tts)

    with make_client(settings=settings, services=services) as client:
        response = client.post(
            "/tts", json=_tts_payload(text="x" * 101), headers=AUTH_HEADERS
        )

    assert response.status_code == 413
    assert response.json()["detail"] == "Text too large"
    assert tts.calls == []


def test_tts_whitespace_only_text_returns_422() -> None:
    tts = FakeTts()
    services = make_services(tts=tts)

    with make_client(services=services) as client:
        response = client.post(
            "/tts", json=_tts_payload(text="  \n\t  "), headers=AUTH_HEADERS
        )

    assert response.status_code == 422
    assert tts.calls == []


def test_tts_chars_uses_normalised_text() -> None:
    tts = FakeTts()
    services = make_services(tts=tts)

    with make_client(services=services) as client:
        response = client.post(
            "/tts",
            json=_tts_payload(text="Halo   dunia\n\nini  uji."),
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 200
    assert response.json()["chars"] == len("Halo dunia\nini uji.")
    assert tts.calls[0][0] == "Halo dunia\nini uji."


def test_tts_edge_failure_returns_503_not_500() -> None:
    tts = FakeTts(error=TTSUnavailableError())
    services = make_services(tts=tts)

    with make_client(services=services) as client:
        response = client.post("/tts", json=_tts_payload(), headers=AUTH_HEADERS)

    assert response.status_code == 503
    assert response.json() == {"detail": "TTS synthesis failed"}


def test_tts_storage_failure_returns_503() -> None:
    storage = FakeStorage()
    storage.put_error = StorageUnavailableError()
    services = make_services(storage=storage)

    with make_client(services=services) as client:
        response = client.post("/tts", json=_tts_payload(), headers=AUTH_HEADERS)

    assert response.status_code == 503
    assert response.json() == {"detail": "Storage unavailable"}


def test_tts_timeout_returns_503() -> None:
    settings = make_settings(tts_timeout_seconds=0.05)
    tts = FakeTts(delay=0.3)
    services = make_services(settings, tts=tts)

    with make_client(settings=settings, services=services) as client:
        response = client.post("/tts", json=_tts_payload(), headers=AUTH_HEADERS)

    assert response.status_code == 503
    assert response.json()["detail"] == "TTS request timed out"


def test_unexpected_error_returns_500_without_leaking_details() -> None:
    class ExplodingTts(FakeTts):
        async def synthesize(
            self, text: str, voice: str, rate: str, lang: str = "id"
        ) -> bytes:
            raise RuntimeError("secret user text and stack trace")

    services = make_services(tts=ExplodingTts())

    with make_client(services=services) as client:
        response = client.post("/tts", json=_tts_payload(), headers=AUTH_HEADERS)

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "secret" not in response.text
    assert "RuntimeError" not in response.text


def test_tts_limiter_is_used() -> None:
    services = make_services()
    assert isinstance(services.limiter, JobLimiter)
