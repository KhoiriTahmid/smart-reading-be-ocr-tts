"""POST /tts - synthesize Indonesian speech and upload the MP3."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Request

from app.config import Settings
from app.errors import (
    InvalidRequestError,
    PayloadTooLargeError,
    ServiceUnavailableError,
    TTSUnavailableError,
)
from app.schemas import TtsRequest, TtsResponse
from app.security import require_api_key
from app.services.limiter import JobLimiter
from app.services.storage import S3Storage
from app.services.tts_engine import EdgeTTSEngine
from app.utils.text import normalize_text

router = APIRouter()


@router.post(
    "/tts",
    response_model=TtsResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(require_api_key)],
)
async def tts(payload: TtsRequest, request: Request) -> TtsResponse:
    settings: Settings = request.app.state.settings
    engine: EdgeTTSEngine = request.app.state.tts
    storage: S3Storage = request.app.state.storage
    limiter: JobLimiter = getattr(request.app.state, "tts_limiter", request.app.state.limiter)

    if len(payload.text) > settings.tts_max_chars:
        raise PayloadTooLargeError("Text too large")

    normalized = normalize_text(payload.text)
    if not normalized:
        raise InvalidRequestError("Text must not be empty")
    if len(normalized) > settings.tts_max_chars:
        raise PayloadTooLargeError("Text too large")

    voice = payload.voice.value if payload.voice is not None else settings.tts_default_voice
    lang = payload.lang.value if payload.lang is not None else "id"
    rate = payload.rate or "+0%"
    if payload.page_number is not None:
        audio_key = f"audio/{payload.history_id}_p{payload.page_number}.mp3"
    else:
        audio_key = f"audio/{payload.history_id}.mp3"

    try:
        async with asyncio.timeout(settings.tts_timeout_seconds):
            async with limiter.slot():
                audio = await engine.synthesize(normalized, voice, rate, lang=lang)
                if not audio:
                    raise TTSUnavailableError()
                duration = await engine.duration_seconds(audio)
                await storage.put_bytes(audio_key, audio, "audio/mpeg")
    except TimeoutError as exc:
        raise ServiceUnavailableError("TTS request timed out") from exc

    return TtsResponse(
        audio_key=audio_key,
        duration_seconds=duration,
        chars=len(normalized),
        page_number=payload.page_number,
    )
