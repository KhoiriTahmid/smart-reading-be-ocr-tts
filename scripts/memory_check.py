#!/usr/bin/env python3
"""Measure peak RSS for a mixed OCR/TTS workload.

Usage:
    python scripts/memory_check.py                 # OCR only
    python scripts/memory_check.py --with-tts      # OCR + live edge-tts

The number printed is the process's peak resident set size (RSS), measured by
the kernel via ``resource.getrusage``. Child ffmpeg memory is not included;
on the target box ffmpeg uses only a few MB.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import resource
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("OMP_NUM_THREADS", "2")

from app.config import Settings  # noqa: E402
from app.services.ocr_engine import OcrEngine  # noqa: E402
from app.services.tts_engine import EdgeTTSEngine  # noqa: E402
from app.utils.images import prepare_image  # noqa: E402


def peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-tts", action="store_true")
    parser.add_argument(
        "--image",
        default=str(
            Path(__file__).resolve().parents[1]
            / "tests"
            / "samples"
            / "indonesian_text.png"
        ),
    )
    args = parser.parse_args()

    settings = Settings()
    engine = OcrEngine(settings)
    engine.load()
    image = prepare_image(Path(args.image).read_bytes(), settings.ocr_max_side)
    result = engine.extract(image)
    print(f"OCR lines={result.lines}, peak RSS={peak_mb():.1f} MB")

    if args.with_tts:
        text = ("Ini adalah teks uji untuk pengukuran memori. " * 40)[:1_000]
        tts = EdgeTTSEngine(settings)
        audio = asyncio.run(tts.synthesize(text, "female", "+0%"))
        duration = asyncio.run(tts.duration_seconds(audio))
        print(
            f"TTS bytes={len(audio)}, duration={duration}s, "
            f"peak RSS={peak_mb():.1f} MB"
        )

    print(f"Peak RSS: {peak_mb():.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
