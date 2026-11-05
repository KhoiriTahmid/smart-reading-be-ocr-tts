"""Language selection tests: detect_language() and lang pass-through on POST /tts."""

from __future__ import annotations

import uuid

import pytest

from app.services.tts_engine import VOICE_MAP_BY_LANG, detect_language
from tests.conftest import AUTH_HEADERS, FakeTts, make_client, make_services

HISTORY_ID = uuid.UUID("3f2b6c1e-9d4a-4f5b-8c7d-1234567890ab")


def _payload(**overrides):
    payload = {
        "history_id": str(HISTORY_ID),
        "text": "Teks yang akan dibacakan.",
        "voice": "female",
    }
    payload.update(overrides)
    return payload


# --- detect_language -------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "The quick brown fox jumps over the lazy dog and it was fun.",
        "This is a book that we can read with all of your friends.",
    ],
)
def test_detect_language_english(text: str) -> None:
    assert detect_language(text) == "en"


@pytest.mark.parametrize(
    "text",
    [
        "Saya akan pergi ke pasar dengan ibu untuk membeli sayur.",
        "Buku ini adalah buku yang sangat bagus dan dapat dibaca oleh kita.",
    ],
)
def test_detect_language_indonesian(text: str) -> None:
    assert detect_language(text) == "id"


@pytest.mark.parametrize("text", ["", "   ", "12345 !!! ???", "Xyzzy plugh"])
def test_detect_language_defaults_to_indonesian(text: str) -> None:
    assert detect_language(text) == "id"


def test_detect_language_is_case_insensitive() -> None:
    assert detect_language("THE BOOK IS ON THE TABLE AND IT IS RED") == "en"


def test_detect_language_tie_prefers_indonesian() -> None:
    # one English hit ("the"), one Indonesian hit ("yang")
    assert detect_language("the yang") == "id"


def test_voice_map_by_lang_covers_both_voices() -> None:
    for lang in ("id", "en"):
        assert set(VOICE_MAP_BY_LANG[lang]) == {"female", "male"}
    assert VOICE_MAP_BY_LANG["en"]["female"].startswith("en-")
    assert VOICE_MAP_BY_LANG["id"]["male"].startswith("id-")


# --- POST /tts lang pass-through -------------------------------------------


@pytest.mark.parametrize("lang", ["id", "en", "auto"])
def test_tts_passes_lang_to_engine(lang: str) -> None:
    tts = FakeTts()
    services = make_services(tts=tts)

    with make_client(services=services) as client:
        response = client.post("/tts", json=_payload(lang=lang), headers=AUTH_HEADERS)

    assert response.status_code == 200
    assert tts.langs == [lang]


def test_tts_lang_defaults_to_indonesian() -> None:
    tts = FakeTts()
    services = make_services(tts=tts)

    with make_client(services=services) as client:
        response = client.post("/tts", json=_payload(), headers=AUTH_HEADERS)

    assert response.status_code == 200
    assert tts.langs == ["id"]


@pytest.mark.parametrize("lang", ["fr", "ID", "english", ""])
def test_tts_rejects_unknown_lang(lang: str) -> None:
    tts = FakeTts()
    services = make_services(tts=tts)

    with make_client(services=services) as client:
        response = client.post("/tts", json=_payload(lang=lang), headers=AUTH_HEADERS)

    assert response.status_code == 422
    assert tts.calls == []
