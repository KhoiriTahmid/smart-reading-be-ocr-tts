"""Unit tests for PDF detection, validation, and multi-page text extraction."""

from __future__ import annotations

import numpy as np
import pymupdf
import pytest

from app.errors import (
    InvalidRequestError,
    PayloadTooLargeError,
    UnsupportedImageError,
)
from app.services.ocr_engine import OcrResult
from app.utils.pdf import extract_pdf_pages, is_pdf
from tests.conftest import FakeOcr, make_png


def make_pdf(pages_text: list[str]) -> bytes:
    """Create an in-memory PDF with digital text."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((50, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def make_scanned_pdf(num_pages: int = 1) -> bytes:
    """Create an in-memory PDF containing an image instead of selectable text."""
    png_bytes = make_png((200, 100))
    doc = pymupdf.open()
    for _ in range(num_pages):
        page = doc.new_page(width=300, height=200)
        page.insert_image(pymupdf.Rect(10, 10, 250, 150), stream=png_bytes)
    data = doc.tobytes()
    doc.close()
    return data


def test_is_pdf_detection() -> None:
    assert is_pdf(b"%PDF-1.4\n") is True
    assert is_pdf(b"%PDF-2.0") is True
    assert is_pdf(b"\x89PNG\r\n\x1a\n") is False
    assert is_pdf(b"\xff\xd8\xff") is False
    assert is_pdf(b"random bytes") is False


def test_extract_pdf_rejects_non_pdf() -> None:
    ocr = FakeOcr()
    with pytest.raises(UnsupportedImageError):
        extract_pdf_pages(b"not a pdf", max_pages=10, render_dpi=150, max_side=1600, ocr_engine=ocr)


def test_extract_pdf_rejects_corrupted_pdf() -> None:
    ocr = FakeOcr()
    with pytest.raises(UnsupportedImageError):
        extract_pdf_pages(b"%PDF-corrupted_data_that_fails_parsing", max_pages=10, render_dpi=150, max_side=1600, ocr_engine=ocr)


def test_extract_pdf_rejects_over_max_pages() -> None:
    pdf_bytes = make_pdf(["Page 1 digital text for testing", "Page 2 digital text for testing", "Page 3 digital text for testing"])
    ocr = FakeOcr()
    with pytest.raises(PayloadTooLargeError) as exc_info:
        extract_pdf_pages(pdf_bytes, max_pages=2, render_dpi=150, max_side=1600, ocr_engine=ocr)
    assert "exceeds the maximum of 2" in str(exc_info.value)


def test_extract_pdf_digital_text_fast_path() -> None:
    text_p1 = "Bab 1: Pengenalan Smart Reading System untuk Pembaca Buku."
    text_p2 = "Bab 2: Arsitektur Cloud OCR dan TTS menggunakan Edge-TTS."
    pdf_bytes = make_pdf([text_p1, text_p2])

    ocr = FakeOcr()
    pages = extract_pdf_pages(pdf_bytes, max_pages=10, render_dpi=150, max_side=1600, ocr_engine=ocr)

    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert "Bab 1: Pengenalan Smart Reading" in pages[0].text
    assert pages[0].avg_confidence == 1.0

    assert pages[1].page_number == 2
    assert "Bab 2: Arsitektur Cloud OCR" in pages[1].text
    assert pages[1].avg_confidence == 1.0

    # Digital text should not call the heavy ONNX OCR engine
    assert len(ocr.calls) == 0


def test_extract_pdf_scanned_image_fallback() -> None:
    pdf_bytes = make_scanned_pdf(num_pages=2)
    fake_result = OcrResult(text="Teks hasil scan halaman", lines=1, avg_confidence=0.91)
    ocr = FakeOcr(result=fake_result)

    pages = extract_pdf_pages(pdf_bytes, max_pages=10, render_dpi=150, max_side=1600, ocr_engine=ocr)

    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[0].text == "Teks hasil scan halaman"
    assert pages[0].avg_confidence == 0.91

    assert pages[1].page_number == 2
    assert pages[1].text == "Teks hasil scan halaman"

    # Both scanned pages should have invoked the OCR engine
    assert len(ocr.calls) == 2


def test_extract_pdf_blank_page() -> None:
    pdf_bytes = make_pdf([""])
    fake_result = OcrResult(text="", lines=0, avg_confidence=0.0)
    ocr = FakeOcr(result=fake_result)

    pages = extract_pdf_pages(pdf_bytes, max_pages=10, render_dpi=150, max_side=1600, ocr_engine=ocr)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].text == ""
    assert pages[0].lines == 0
