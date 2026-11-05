"""Integration tests for the running Python OCR/TTS FastAPI Service.

These tests hit the live local service running at http://127.0.0.1:8000
and verify endpoints, authentication, input validation, and service contracts.
"""

from __future__ import annotations

import os
import uuid
import pytest
import requests

BASE_URL = os.getenv("PYTHON_SERVICE_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("OCR_SERVICE_API_KEY", "my-local-secret-123")
HEADERS = {"x-api-key": API_KEY, "Content-Type": "application/json"}


def test_integration_health():
    """Verify health endpoint reports ready models."""
    res = requests.get(f"{BASE_URL}/health", timeout=5)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["ocr_loaded"] is True
    assert data["tts_ready"] is True


def test_integration_auth_rejection():
    """Verify endpoints reject requests with missing or invalid API keys."""
    res = requests.post(f"{BASE_URL}/ocr", json={"file_key": "uploads/test.png"}, timeout=5)
    assert res.status_code == 401

    res = requests.post(
        f"{BASE_URL}/ocr",
        headers={"x-api-key": "invalid-key"},
        json={"file_key": "uploads/test.png"},
        timeout=5,
    )
    assert res.status_code == 401


def test_integration_ocr_validation():
    """Verify OCR path traversal and invalid key validation."""
    res = requests.post(
        f"{BASE_URL}/ocr",
        headers=HEADERS,
        json={"file_key": "invalid/path.png"},
        timeout=5,
    )
    assert res.status_code == 422


def test_integration_ocr_s3_object_not_found():
    """Verify OCR requests against non-existent S3 objects return 404."""
    random_key = f"uploads/non-existent-{uuid.uuid4()}.png"
    res = requests.post(
        f"{BASE_URL}/ocr",
        headers=HEADERS,
        json={"file_key": random_key},
        timeout=10,
    )
    assert res.status_code == 404


def test_integration_tts_validation():
    """Verify TTS request schema validations (uuid, voice, rate, lang)."""
    # Invalid UUID
    res = requests.post(
        f"{BASE_URL}/tts",
        headers=HEADERS,
        json={"history_id": "not-a-uuid", "text": "Testing"},
        timeout=5,
    )
    assert res.status_code == 422

    # Unsupported lang
    res = requests.post(
        f"{BASE_URL}/tts",
        headers=HEADERS,
        json={"history_id": str(uuid.uuid4()), "text": "Testing", "lang": "fr"},
        timeout=5,
    )
    assert res.status_code == 422

    # Empty text
    res = requests.post(
        f"{BASE_URL}/tts",
        headers=HEADERS,
        json={"history_id": str(uuid.uuid4()), "text": "   "},
        timeout=5,
    )
    assert res.status_code == 422
