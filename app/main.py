"""FastAPI application factory and lifespan management."""

from __future__ import annotations

import os

# Must be set before onnxruntime / numpy initialise their thread pools.
os.environ.setdefault("OMP_NUM_THREADS", "2")

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Settings
from app.errors import ServiceError
from app.routers import health, ocr, tts
from app.services.limiter import JobLimiter
from app.services.ocr_engine import OcrEngine
from app.services.storage import S3Storage
from app.services.tts_engine import EdgeTTSEngine

logger = logging.getLogger(__name__)


@dataclass
class AppServices:
    """Container used by tests (and potentially alternate deployments)."""

    ocr: Any
    tts: Any
    storage: Any
    limiter: JobLimiter
    ocr_loaded: bool = True
    tts_ready: bool = True


def _configure_logging(settings: Settings) -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def create_app(
    settings: Settings | None = None,
    services: AppServices | None = None,
) -> FastAPI:
    settings = settings or Settings()
    os.environ["OMP_NUM_THREADS"] = str(settings.omp_num_threads)
    _configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if (
            settings.app_env == "production"
            and settings.ocr_service_api_key in {"", "change-me"}
        ):
            raise RuntimeError(
                "OCR_SERVICE_API_KEY must be set to a non-default value in production"
            )

        created_internally = services is None
        storage: S3Storage | None = None

        try:
            if services is not None:
                app.state.ocr = services.ocr
                app.state.tts = services.tts
                app.state.storage = services.storage
                app.state.limiter = services.limiter
                app.state.ocr_loaded = services.ocr_loaded
                app.state.tts_ready = services.tts_ready
            else:
                app.state.ocr_loaded = False
                app.state.tts_ready = False

                app.state.limiter = JobLimiter(settings.max_waiting_requests)
                storage = S3Storage(settings)
                app.state.storage = storage
                app.state.tts = EdgeTTSEngine(settings)
                app.state.tts_ready = True

                engine = OcrEngine(settings)
                # warmup() loads both ONNX sessions now, not on the first
                # request. Model files are already in the image.
                await run_in_threadpool(engine.warmup)
                app.state.ocr = engine
                app.state.ocr_loaded = True

            logger.info("Service ready (env=%s)", settings.app_env)

            yield

        finally:
            if created_internally and storage is not None:
                storage.close()

    app = FastAPI(
        title="Smart Reading Device - OCR + TTS",
        version="1.0.0",
        # This is a private service; do not publish interactive docs.
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.settings = settings

    @app.exception_handler(ServiceError)
    async def handle_service_error(
        request: Request, exc: ServiceError
    ) -> JSONResponse:
        headers = (
            {"Retry-After": str(exc.retry_after)}
            if exc.retry_after is not None
            else None
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Do not echo the offending input; it may contain user text.
        return JSONResponse(status_code=422, content={"detail": "Invalid request"})

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, str) else "Request failed"
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": detail},
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request, exc: Exception
    ) -> JSONResponse:
        # Never log stack traces, request bodies or user text.
        logger.error(
            "Unhandled %s while processing %s %s",
            type(exc).__name__,
            request.method,
            request.url.path,
        )
        return JSONResponse(
            status_code=500, content={"detail": "Internal server error"}
        )

    app.include_router(health.router)
    app.include_router(ocr.router)
    app.include_router(tts.router)
    return app


app = create_app()
