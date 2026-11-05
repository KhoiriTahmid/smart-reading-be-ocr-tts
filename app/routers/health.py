"""Basic liveness/readiness endpoint. No authentication."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health", response_model=None)
async def health(request: Request) -> JSONResponse:
    ocr_loaded = bool(getattr(request.app.state, "ocr_loaded", False))
    tts_ready = bool(getattr(request.app.state, "tts_ready", False))
    ready = ocr_loaded and tts_ready

    body = {
        "status": "ok" if ready else "starting",
        "ocr_loaded": ocr_loaded,
        "tts_ready": tts_ready,
    }
    return JSONResponse(body, status_code=200 if ready else 503)
