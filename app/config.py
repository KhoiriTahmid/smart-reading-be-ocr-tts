"""Application configuration loaded from environment variables / .env."""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings.

    Environment variable names are case-insensitive (e.g. ``S3_ENDPOINT``).
    Unknown variables are ignored so the same environment can be shared with
    other services during local development.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: Literal["production", "development", "test"] = "production"

    # Shared secret used by the NestJS service. Must be changed in production.
    ocr_service_api_key: str = "change-me"

    # Object storage (Backblaze B2 S3-compatible endpoint).
    s3_endpoint: str = ""
    s3_region: str = ""
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_bucket: str = ""

    # OCR
    ocr_min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    ocr_max_side: int = Field(default=1600, ge=320, le=8000)
    max_image_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    max_pdf_bytes: int = Field(default=15 * 1024 * 1024, ge=1)
    pdf_max_pages: int = Field(default=10, ge=1, le=50)
    pdf_render_dpi: int = Field(default=150, ge=72, le=300)
    # Optional directory for downloaded ONNX models. When unset RapidOCR uses
    # its package directory (fine for local dev; Docker sets /app/models).
    ocr_model_dir: Path | None = None

    # TTS
    tts_max_chars: int = Field(default=20_000, ge=1)
    tts_chunk_chars: int = Field(default=2500, ge=100, le=4_000)
    tts_default_voice: Literal["female", "male"] = "female"
    tts_max_concurrency: int = Field(default=3, ge=1, le=10)
    tts_max_waiting_requests: int = Field(default=10, ge=0)

    # Concurrency / timeouts
    max_waiting_requests: int = Field(default=2, ge=0)
    ocr_timeout_seconds: float = Field(default=60.0, gt=0)
    tts_timeout_seconds: float = Field(default=150.0, gt=0)

    # Tooling paths
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"

    # Logging
    log_level: str = "INFO"

    # Threading (also exported as OMP_NUM_THREADS before OCR initialises).
    omp_num_threads: int = Field(default=2, ge=1, le=16)

    @field_validator(
        "s3_endpoint",
        "s3_region",
        "s3_access_key",
        "s3_secret_key",
        "s3_bucket",
        mode="before",
    )
    @classmethod
    def _strip_strings(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("ocr_model_dir", mode="before")
    @classmethod
    def _empty_model_dir_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("tts_chunk_chars")
    @classmethod
    def _chunk_must_fit_limit(cls, value: int, info) -> int:
        max_chars = info.data.get("tts_max_chars")
        if max_chars is not None and value > max_chars:
            raise ValueError("TTS_CHUNK_CHARS must be <= TTS_MAX_CHARS")
        return value
