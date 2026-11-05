"""Optional live edge-tts test. Requires outbound internet.

    RUN_TTS_INTEGRATION=1 pytest -m integration tests/test_tts_live.py
"""

from __future__ import annotations

import os

import edge_tts
import pytest

from app.services.tts_engine import EdgeTTSEngine
from tests.conftest import make_settings

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def engine() -> EdgeTTSEngine:
    if os.getenv("RUN_TTS_INTEGRATION") != "1":
        pytest.skip("set RUN_TTS_INTEGRATION=1 to call the live edge-tts service")
    return EdgeTTSEngine(make_settings())


@pytest.mark.asyncio
async def test_indonesian_voices_exist() -> None:
    if os.getenv("RUN_TTS_INTEGRATION") != "1":
        pytest.skip("set RUN_TTS_INTEGRATION=1 to call the live edge-tts service")
    voices = await edge_tts.list_voices()
    short_names = {voice["ShortName"] for voice in voices}
    assert "id-ID-GadisNeural" in short_names
    assert "id-ID-ArdiNeural" in short_names


@pytest.mark.asyncio
@pytest.mark.parametrize("voice", ["female", "male"])
async def test_live_synthesis(engine: EdgeTTSEngine, voice: str) -> None:
    audio = await engine.synthesize("Halo, ini uji suara bahasa Indonesia.", voice, "+0%")
    duration = await engine.duration_seconds(audio)

    assert len(audio) > 1_000
    assert duration > 0.5
