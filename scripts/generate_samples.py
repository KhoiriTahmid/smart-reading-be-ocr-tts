#!/usr/bin/env python3
"""Regenerate the sample images used by the OCR integration test."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

LINES = [
    "Selamat pagi. Ini adalah teks bahasa Indonesia.",
    "Harga buku ini Rp 25.000.",
    "Email: budi@example.com",
    "Telepon: 0812-3456-7890",
]


def main() -> int:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    font_path = next((path for path in candidates if Path(path).exists()), None)
    font = ImageFont.truetype(font_path, 44) if font_path else ImageFont.load_default(44)

    image = Image.new("RGB", (1200, 520), "white")
    draw = ImageDraw.Draw(image)
    y = 45
    for line in LINES:
        draw.text((50, y), line, font=font, fill="black")
        y += 105

    output = Path(__file__).resolve().parents[1] / "tests" / "samples"
    output.mkdir(parents=True, exist_ok=True)
    image.save(output / "indonesian_text.png")
    image.save(output / "indonesian_text.jpg", quality=95)
    print(f"Wrote samples to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
