#!/usr/bin/env python3
"""Download and initialise the PP-OCRv6 tiny models.

Run during the Docker image build so the first OCR request does not download
anything. It can also be run locally before starting the service.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Keep ONNX Runtime from grabbing every core during model initialisation.
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from app.config import Settings  # noqa: E402
from app.services.ocr_engine import OcrEngine  # noqa: E402


def main() -> int:
    settings = Settings()
    engine = OcrEngine(settings)
    engine.warmup()

    # A tiny inference also initialises the preprocessing and postprocessing
    # paths. Detection failing on this synthetic image is harmless.
    image = Image.new("RGB", (800, 240), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=56)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    draw.text((40, 80), "Warmup OCR 12345", font=font, fill="black")
    result = engine.extract(np.asarray(image))

    model_dir = settings.ocr_model_dir or "RapidOCR package model directory"
    print(f"OCR models ready ({model_dir}); warmup probe lines={result.lines}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
