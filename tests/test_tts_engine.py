"""edge-tts wrapper tests (edge_tts itself is mocked)."""

from __future__ import annotations

import pytest

from app.errors import TTSUnavailableError
from app.services import tts_engine as tts_module
from app.services.tts_engine import VOICE_MAP, EdgeTTSEngine
from app.utils.text import chunk_text
from tests.conftest import make_settings


class FakeCommunicate:
    instances: list["FakeCommunicate"] = []
    audio_chunks: list[bytes] = [b"mp3-chunk"]
    error: Exception | None = None

    def __init__(self, text, voice, **kwargs):
        self.text = text
        self.voice = voice
        self.rate = kwargs.get("rate")
        self.kwargs = kwargs
        FakeCommunicate.instances.append(self)

    async def stream(self):
        if FakeCommunicate.error is not None:
            raise FakeCommunicate.error
        for chunk in FakeCommunicate.audio_chunks:
            yield {"type": "audio", "data": chunk}
        yield {"type": "SentenceBoundary", "offset": 0, "duration": 1}


@pytest.fixture(autouse=True)
def reset_fake():
    FakeCommunicate.instances = []
    FakeCommunicate.audio_chunks = [b"mp3-chunk"]
    FakeCommunicate.error = None
    yield


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> EdgeTTSEngine:
    monkeypatch.setattr(
        tts_module.shutil, "which", lambda binary: f"/usr/bin/{binary}"
    )
    monkeypatch.setattr(tts_module.edge_tts, "Communicate", FakeCommunicate)
    return EdgeTTSEngine(make_settings(tts_chunk_chars=100))


def test_voice_map_is_fixed() -> None:
    assert VOICE_MAP == {
        "female": "id-ID-GadisNeural",
        "male": "id-ID-ArdiNeural",
    }


@pytest.mark.asyncio
async def test_single_chunk_synthesis_passes_voice_and_rate(engine: EdgeTTSEngine) -> None:
    audio = await engine.synthesize("Halo dunia.", "female", "+10%")

    assert audio == b"mp3-chunk"
    assert len(FakeCommunicate.instances) == 1
    instance = FakeCommunicate.instances[0]
    assert instance.text == "Halo dunia."
    assert instance.voice == "id-ID-GadisNeural"
    assert instance.rate == "+10%"


@pytest.mark.asyncio
async def test_male_voice_mapping(engine: EdgeTTSEngine) -> None:
    await engine.synthesize("Halo.", "male", "+0%")
    voice = FakeCommunicate.instances[0].voice
    assert voice == "id-ID-ArdiNeural"


@pytest.mark.asyncio
async def test_english_and_auto_voice_mapping(engine: EdgeTTSEngine) -> None:
    await engine.synthesize("Hello world.", "female", "+0%", lang="en")
    assert FakeCommunicate.instances[-1].voice == "en-US-AriaNeural"

    await engine.synthesize("Hello world.", "male", "+0%", lang="en")
    assert FakeCommunicate.instances[-1].voice == "en-US-GuyNeural"

    await engine.synthesize(
        "It is a truth that the man was in the house with all of his books.",
        "female",
        "+0%",
        lang="auto",
    )
    assert FakeCommunicate.instances[-1].voice == "en-US-AriaNeural"


@pytest.mark.asyncio
async def test_long_text_is_chunked_and_concatenated(
    engine: EdgeTTSEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = (
        "Ini kalimat pertama yang sangat panjang sekali untuk pengujian. "
        "Ini kalimat kedua yang juga sangat panjang sekali untuk pengujian. "
        "Ini kalimat ketiga yang tetap panjang untuk pengujian."
    )
    chunks = chunk_text(text, 100)
    assert len(chunks) > 1

    async def fake_concat(parts: list[bytes]) -> bytes:
        assert len(parts) == len(chunks)
        return b"concatenated"

    monkeypatch.setattr(engine, "_concat_mp3", fake_concat)
    audio = await engine.synthesize(text, "female", "+0%")

    assert audio == b"concatenated"
    assert [item.text for item in FakeCommunicate.instances] == chunks
    assert all(len(item.text) <= 100 for item in FakeCommunicate.instances)


@pytest.mark.asyncio
async def test_edge_failure_becomes_safe_tts_error(engine: EdgeTTSEngine) -> None:
    FakeCommunicate.error = RuntimeError("remote service exploded with text")
    with pytest.raises(TTSUnavailableError) as exc_info:
        await engine.synthesize("Halo.", "female", "+0%")
    assert "remote service" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_empty_audio_becomes_safe_tts_error(engine: EdgeTTSEngine) -> None:
    FakeCommunicate.audio_chunks = []
    with pytest.raises(TTSUnavailableError):
        await engine.synthesize("Halo.", "female", "+0%")
