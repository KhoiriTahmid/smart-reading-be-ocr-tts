"""Real RapidOCR test. Disabled by default because it needs model files.

Run with:

    RUN_OCR_INTEGRATION=1 pytest -m integration tests/test_ocr_integration.py
"""

from __future__ import annotations

import os
from difflib import SequenceMatcher
from pathlib import Path

import pytest

from app.services.ocr_engine import OcrEngine
from app.utils.images import prepare_image
from app.utils.text import normalize_text
from tests.conftest import make_settings

pytestmark = pytest.mark.integration

SAMPLES = Path(__file__).parent / "samples"
EXPECTED = (
    "Selamat pagi. Ini adalah teks bahasa Indonesia.\n"
    "Harga buku ini Rp 25.000.\n"
    "Email: budi@example.com\n"
    "Telepon: 0812-3456-7890"
)


@pytest.fixture(scope="module")
def engine() -> OcrEngine:
    if os.getenv("RUN_OCR_INTEGRATION") != "1":
        pytest.skip("set RUN_OCR_INTEGRATION=1 to run real OCR")
    engine = OcrEngine(make_settings())
    engine.warmup()
    return engine


@pytest.mark.parametrize("filename", ["indonesian_text.png", "indonesian_text.jpg"])
def test_real_indonesian_sample(engine: OcrEngine, filename: str) -> None:
    data = (SAMPLES / filename).read_bytes()
    image = prepare_image(data, max_side=1600)
    result = engine.extract(image)

    expected = normalize_text(EXPECTED).replace("\n", " ")
    actual = normalize_text(result.text).replace("\n", " ")
    similarity = SequenceMatcher(None, expected, actual).ratio()

    assert result.text != ""
    assert result.lines >= 3
    assert result.avg_confidence >= 0.5
    assert similarity >= 0.80, f"OCR text too different: {actual!r}"
