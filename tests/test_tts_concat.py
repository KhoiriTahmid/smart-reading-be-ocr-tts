"""Real MP3 concatenation test (edge-tts is mocked, ffmpeg is real)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from app.services import tts_engine as tts_module
from app.services.tts_engine import EdgeTTSEngine
from app.utils.text import chunk_text
from tests.conftest import make_settings

FFMPEG = shutil.which("ffmpeg")


class FakeCommunicate:
    instances: list["FakeCommunicate"] = []
    audio = b""

    def __init__(self, text, voice, **kwargs):
        self.text = text
        self.voice = voice
        self.rate = kwargs.get("rate")
        FakeCommunicate.instances.append(self)

    async def stream(self):
        yield {"type": "audio", "data": FakeCommunicate.audio}


@pytest.fixture(scope="module")
def silent_mp3() -> bytes:
    if not FFMPEG:
        pytest.skip("ffmpeg not installed")
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "silent.mp3"
        subprocess.run(
            [
                FFMPEG,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=24000:cl=mono",
                "-t",
                "0.25",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "48k",
                str(path),
            ],
            check=True,
            capture_output=True,
        )
        return path.read_bytes()


@pytest.mark.asyncio
async def test_5000_chars_produces_one_continuous_mp3(
    silent_mp3: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = ("Ini adalah kalimat uji yang cukup panjang untuk dibacakan. " * 200)[:5000]
    settings = make_settings(tts_chunk_chars=900)
    chunks = chunk_text(text, settings.tts_chunk_chars)
    assert len(chunks) > 1

    FakeCommunicate.instances = []
    FakeCommunicate.audio = silent_mp3
    monkeypatch.setattr(tts_module.edge_tts, "Communicate", FakeCommunicate)

    engine = EdgeTTSEngine(settings)
    audio = await engine.synthesize(text, "female", "+0%")
    duration = await engine.duration_seconds(audio)

    assert [item.text for item in FakeCommunicate.instances] == chunks
    assert all(len(item.text) <= 900 for item in FakeCommunicate.instances)
    assert len(audio) > len(silent_mp3)
    assert duration > 1.0

    # A full decode with ffmpeg proves the concatenated file is playable even
    # when ffprobe is not installed on the test machine.
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "output.mp3"
        path.write_bytes(audio)
        subprocess.run(
            [
                FFMPEG,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-f",
                "null",
                "-",
            ],
            check=True,
            capture_output=True,
        )
