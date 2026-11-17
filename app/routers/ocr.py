"""POST /ocr - download an image from S3 and return extracted text."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.errors import ServiceUnavailableError
from app.schemas import OcrRequest, OcrResponse, PageResult
from app.security import require_api_key
from app.services.limiter import JobLimiter
from app.services.ocr_engine import OcrEngine
from app.services.storage import S3Storage
from app.utils.images import prepare_image
from app.utils.pdf import extract_pdf_pages, is_pdf

router = APIRouter()


@router.post(
    "/ocr",
    response_model=OcrResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(require_api_key)],
)
async def ocr(payload: OcrRequest, request: Request) -> OcrResponse:
    settings: Settings = request.app.state.settings
    engine: OcrEngine = request.app.state.ocr
    storage: S3Storage = request.app.state.storage
    limiter: JobLimiter = request.app.state.limiter

    is_pdf_key = payload.file_key.lower().endswith(".pdf")
    max_bytes = settings.max_pdf_bytes if is_pdf_key else settings.max_image_bytes

    try:
        async with asyncio.timeout(settings.ocr_timeout_seconds):
            async with limiter.slot():
                file_bytes = await storage.get_bytes(payload.file_key, max_bytes)

                if is_pdf_key or is_pdf(file_bytes):
                    pages = await run_in_threadpool(
                        extract_pdf_pages,
                        file_bytes,
                        settings.pdf_max_pages,
                        settings.pdf_render_dpi,
                        settings.ocr_max_side,
                        engine,
                    )
                    full_text = "\n\n".join(p.text for p in pages if p.text).strip()
                    total_lines = sum(p.lines for p in pages)
                    scored = [p for p in pages if p.lines > 0]
                    avg_conf = (
                        round(sum(p.avg_confidence for p in scored) / len(scored), 4)
                        if scored
                        else 0.0
                    )
                    return OcrResponse(
                        text=full_text,
                        lines=total_lines,
                        avg_confidence=avg_conf,
                        total_pages=len(pages),
                        pages=pages,
                    )

                image_rgb = await run_in_threadpool(
                    prepare_image, file_bytes, settings.ocr_max_side
                )
                result = await run_in_threadpool(engine.extract, image_rgb)
    except TimeoutError as exc:
        raise ServiceUnavailableError("OCR request timed out") from exc

    return OcrResponse(
        text=result.text,
        lines=result.lines,
        avg_confidence=result.avg_confidence,
    )
