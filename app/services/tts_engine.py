"""edge-tts wrapper, text chunking and MP3 concatenation."""

from __future__ import annotations

import asyncio
import logging
import math
import re
import shutil
import tempfile
from pathlib import Path

import edge_tts

from app.config import Settings
from app.errors import TTSUnavailableError
from app.utils.text import chunk_text, normalize_text

logger = logging.getLogger(__name__)

VOICE_MAP: dict[str, str] = {
    "female": "id-ID-GadisNeural",
    "male": "id-ID-ArdiNeural",
}

VOICE_MAP_BY_LANG: dict[str, dict[str, str]] = {
    "id": {
        "female": "id-ID-GadisNeural",
        "male": "id-ID-ArdiNeural",
    },
    "en": {
        "female": "en-US-AriaNeural",
        "male": "en-US-GuyNeural",
    },
}

_EN_STOPWORDS = {
    "the", "and", "of", "to", "in", "is", "it", "that", "was", "for",
    "on", "are", "with", "as", "his", "they", "be", "at", "one", "have",
    "this", "from", "or", "had", "by", "not", "word", "but", "what", "some",
    "we", "can", "out", "other", "were", "all", "there", "when", "up", "use",
    "your", "how", "said", "an", "each", "she",
}

_ID_STOPWORDS = {
    "yang", "dan", "di", "ini", "itu", "dengan", "untuk", "tidak", "dari",
    "dalam", "akan", "pada", "juga", "saya", "ke", "karena", "tersebut", "bisa",
    "ada", "mereka", "lebih", "sudah", "atau", "saat", "oleh", "sebagai", "ia",
    "telah", "dapat", "para", "harus", "namun", "kita", "seorang", "adalah",
}


def detect_language(text: str) -> str:
    """Lightweight heuristic for 'auto' language mode ('id' vs 'en')."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if not words:
        return "id"
    en_hits = sum(1 for w in words if w in _EN_STOPWORDS)
    id_hits = sum(1 for w in words if w in _ID_STOPWORDS)
    return "en" if en_hits > id_hits else "id"


class EdgeTTSEngine:
    """Small interface so another engine (for example Piper) can be added."""

    def __init__(self, settings: Settings) -> None:
        ffmpeg_path = shutil.which(settings.ffmpeg_binary)
        if ffmpeg_path is None:
            raise RuntimeError("ffmpeg was not found on PATH")

        self._settings = settings
        self._ffmpeg = ffmpeg_path
        self._ffprobe = shutil.which(settings.ffprobe_binary)
        if self._ffprobe is None:
            # ffprobe ships with normal ffmpeg packages; if it is missing we
            # still synthesize audio and estimate duration from the bitrate.
            logger.warning("ffprobe not found; duration will be estimated")

    async def synthesize(
        self,
        text: str,
        voice: str,
        rate: str = "+0%",
        lang: str = "id",
    ) -> bytes:
        """Synthesize text to a single MP3 byte string."""

        resolved_lang = detect_language(text) if lang == "auto" else (lang or "id")
        lang_voices = VOICE_MAP_BY_LANG.get(resolved_lang, VOICE_MAP)
        voice_name = lang_voices.get(voice)
        if voice_name is None:
            raise TTSUnavailableError("Unsupported voice")

        normalized = normalize_text(text)
        chunks = chunk_text(normalized, self._settings.tts_chunk_chars)
        if not chunks:
            raise TTSUnavailableError("No text to synthesize")

        logger.info(
            "Synthesizing TTS: %d chunk(s), voice=%s, lang=%s (%s)",
            len(chunks),
            voice,
            resolved_lang,
            voice_name,
        )
        parts = [
            await self._synthesize_chunk(chunk, voice_name, rate) for chunk in chunks
        ]

        if len(parts) == 1:
            return parts[0]
        return await self._concat_mp3(parts)

    async def _synthesize_chunk(self, text: str, voice_name: str, rate: str) -> bytes:
        audio = bytearray()
        try:
            communicate = edge_tts.Communicate(
                text,
                voice_name,
                rate=rate,
                connect_timeout=10,
                receive_timeout=60,
            )
            async for message in communicate.stream():
                if message.get("type") == "audio" and message.get("data"):
                    audio.extend(message["data"])
        except Exception as exc:
            # Includes edge-tts websocket/HTTP errors, NoAudioReceived and
            # aiohttp timeouts. Never include text or exception details in
            # the response; a single safe 503 lets NestJS retry.
            raise TTSUnavailableError() from exc

        if not audio:
            raise TTSUnavailableError()

        return bytes(audio)

    async def _concat_mp3(self, parts: list[bytes]) -> bytes:
        with tempfile.TemporaryDirectory(prefix="ocr-service-tts-") as temp_dir:
            temp_path = Path(temp_dir)
            concat_file = temp_path / "concat.txt"
            output_file = temp_path / "output.mp3"

            lines: list[str] = []
            for index, part in enumerate(parts):
                name = f"chunk_{index:03d}.mp3"
                (temp_path / name).write_bytes(part)
                lines.append(f"file '{name}'")
            concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

            # Fast path: copy MP3 frames without re-encoding.
            copied = await self._run_ffmpeg(
                [
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    "concat.txt",
                    "-c",
                    "copy",
                    "-vn",
                    "output.mp3",
                ],
                cwd=temp_dir,
            )
            if copied and output_file.exists() and output_file.stat().st_size > 0:
                return output_file.read_bytes()

            # Some MP3 streams cannot be safely concatenated with -c copy.
            logger.info("MP3 stream copy failed, re-encoding concatenated TTS")
            reencoded = await self._run_ffmpeg(
                [
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    "concat.txt",
                    "-c:a",
                    "libmp3lame",
                    "-b:a",
                    "64k",
                    "-ac",
                    "1",
                    "-ar",
                    "24000",
                    "-vn",
                    "output.mp3",
                ],
                cwd=temp_dir,
            )
            if not reencoded or not output_file.exists() or not output_file.stat().st_size:
                raise TTSUnavailableError("Audio concatenation failed")
            return output_file.read_bytes()

    async def _run_ffmpeg(self, args: list[str], cwd: str) -> bool:
        command = [self._ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *args]
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            async with asyncio.timeout(30):
                await process.communicate()
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise
        except TimeoutError:
            process.kill()
            await process.wait()
            logger.warning("ffmpeg timed out")
            return False

        if process.returncode != 0:
            logger.warning("ffmpeg failed with exit code %s", process.returncode)
            return False
        return True

    async def duration_seconds(self, audio: bytes) -> float:
        """Measure an MP3 with ffprobe; fall back to a size estimate."""

        if not self._ffprobe:
            return _estimate_duration(audio)

        with tempfile.TemporaryDirectory(prefix="ocr-service-tts-") as temp_dir:
            path = Path(temp_dir) / "audio.mp3"
            path.write_bytes(audio)
            try:
                process = await asyncio.create_subprocess_exec(
                    self._ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
            except OSError:
                return _estimate_duration(audio)
            try:
                async with asyncio.timeout(15):
                    stdout, _ = await process.communicate()
            except (TimeoutError, asyncio.CancelledError):
                process.kill()
                await process.wait()
                raise

        if process.returncode != 0:
            return _estimate_duration(audio)

        try:
            duration = float(stdout.decode("ascii", errors="ignore").strip())
        except ValueError:
            return _estimate_duration(audio)

        if not math.isfinite(duration) or duration <= 0:
            return _estimate_duration(audio)
        return round(duration, 2)


def _estimate_duration(audio: bytes) -> float:
    # edge-tts currently produces ~48 kbps mono MP3.
    if not audio:
        return 0.0
    return round(len(audio) * 8 / 48_000, 2)
