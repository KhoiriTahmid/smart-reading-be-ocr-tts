"""PDF validation and multi-page text extraction for OCR."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pymupdf

from app.errors import (
    InvalidRequestError,
    PayloadTooLargeError,
    UnsupportedImageError,
)
from app.schemas import PageResult

if TYPE_CHECKING:
    from app.services.ocr_engine import OcrEngine

logger = logging.getLogger(__name__)

PDF_SIGNATURE = b"%PDF-"


def is_pdf(data: bytes) -> bool:
    """Return True if the data contains the PDF signature in the first 1024 bytes (per ISO 32000-1)."""
    return PDF_SIGNATURE in data[:1024]


def extract_pdf_pages(
    pdf_bytes: bytes,
    max_pages: int,
    render_dpi: int,
    max_side: int,
    ocr_engine: OcrEngine,
) -> list[PageResult]:
    """Extract text from each page of a PDF.

    Checks:
    - Encrypted / password-protected -> InvalidRequestError
    - Page count > max_pages -> PayloadTooLargeError
    - Corrupt or invalid PDF -> UnsupportedImageError (415)

    For each page:
    - If digital text exists (>= 20 chars), extract directly with 1.0 confidence.
    - Otherwise rasterizes the page at `render_dpi` and runs the OCR engine.
    """
    if not is_pdf(pdf_bytes):
        raise UnsupportedImageError("Not a valid PDF file")

    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise UnsupportedImageError("Failed to open PDF") from exc

    try:
        if doc.is_encrypted:
            raise InvalidRequestError("Encrypted or password-protected PDF is not supported")

        total_pages = len(doc)
        if total_pages == 0:
            raise InvalidRequestError("PDF has no pages")

        if total_pages > max_pages:
            raise PayloadTooLargeError(
                f"PDF has {total_pages} pages, which exceeds the maximum of {max_pages}"
            )

        results: list[PageResult] = []

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            page = doc[page_idx]

            # Try digital text extraction first
            text = page.get_text().strip()
            if len(text) >= 20:
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                clean_text = "\n".join(lines)
                results.append(
                    PageResult(
                        page_number=page_num,
                        text=clean_text,
                        lines=len(lines),
                        avg_confidence=1.0,
                    )
                )
                continue

            # Rasterize page for OCR
            pix = page.get_pixmap(dpi=render_dpi)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

            if pix.n == 4:
                rgb = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            elif pix.n == 3:
                rgb = arr
            elif pix.n == 1:
                rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            else:
                raise UnsupportedImageError("Unsupported pixel format in PDF page")

            # Downscale if max side > max_side
            h, w = rgb.shape[:2]
            longest = max(h, w)
            if longest > max_side:
                scale = max_side / float(longest)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

            ocr_res = ocr_engine.extract(rgb)
            results.append(
                PageResult(
                    page_number=page_num,
                    text=ocr_res.text,
                    lines=ocr_res.lines,
                    avg_confidence=ocr_res.avg_confidence,
                )
            )

        return results
    finally:
        doc.close()
