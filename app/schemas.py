"""Pydantic request/response models (Pydantic v2)."""

from __future__ import annotations

from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class Voice(str, Enum):
    FEMALE = "female"
    MALE = "male"


class Lang(str, Enum):
    ID = "id"
    EN = "en"
    AUTO = "auto"


class OcrRequest(BaseModel):
    file_key: str = Field(min_length=1, max_length=1024)
    lang: Lang | None = None

    @field_validator("file_key")
    @classmethod
    def validate_file_key(cls, value: str) -> str:
        if not value.startswith("uploads/"):
            raise ValueError("file_key must start with uploads/")
        if len(value) <= len("uploads/"):
            raise ValueError("file_key must point to an object")
        if ".." in value or "\\" in value:
            raise ValueError("file_key contains forbidden path segments")
        if any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("file_key contains control characters")
        return value


class PageResult(BaseModel):
    page_number: int = Field(ge=1)
    text: str
    lines: int
    avg_confidence: float


class OcrResponse(BaseModel):
    text: str
    lines: int
    avg_confidence: float
    total_pages: int | None = None
    pages: list[PageResult] | None = None


class TtsRequest(BaseModel):
    history_id: UUID
    text: str
    voice: Voice | None = None
    lang: Lang | None = None
    rate: str | None = Field(default=None, pattern=r"^[+-]\d{1,2}%$")
    page_number: int | None = Field(default=None, ge=1)


class TtsResponse(BaseModel):
    audio_key: str
    duration_seconds: float
    chars: int
    page_number: int | None = None
