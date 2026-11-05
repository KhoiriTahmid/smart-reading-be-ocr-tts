"""POST /ocr - download an image from S3 and return extracted text."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.errors import ServiceUnavailableError
from app.schemas import OcrRequest, OcrResponse
from app.security import require_api_key
from app.services.limiter import JobLimiter
from app.services.ocr_engine import OcrEngine
from app.services.storage import S3Storage
from app.utils.images import prepare_image

router = APIRouter()


@router.post(
    "/ocr",
    response_model=OcrResponse,
    dependencies=[Depends(require_api_key)],
)
async def ocr(payload: OcrRequest, request: Request) -> OcrResponse:
    settings: Settings = request.app.state.settings
    engine: OcrEngine = request.app.state.ocr
    storage: S3Storage = request.app.state.storage
    limiter: JobLimiter = request.app.state.limiter

    try:
        async with asyncio.timeout(settings.ocr_timeout_seconds):
            async with limiter.slot():
                image_bytes = await storage.get_bytes(
                    payload.file_key, settings.max_image_bytes
                )
                image_rgb = await run_in_threadpool(
                    prepare_image, image_bytes, settings.ocr_max_side
                )
                result = await run_in_threadpool(engine.extract, image_rgb)
    except TimeoutError as exc:
        raise ServiceUnavailableError("OCR request timed out") from exc

    return OcrResponse(
        text=result.text,
        lines=result.lines,
        avg_confidence=result.avg_confidence,
    )
