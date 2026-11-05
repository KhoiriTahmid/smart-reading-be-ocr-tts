"""Image magic-byte validation and preprocessing tests."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from app.errors import PayloadTooLargeError, UnsupportedImageError
from app.utils import images
from app.utils.images import detect_image_format, prepare_image
from tests.conftest import make_jpeg, make_png


def test_detect_png_and_jpeg() -> None:
    assert detect_image_format(make_png()) == "PNG"
    assert detect_image_format(make_jpeg()) == "JPEG"


def test_detect_rejects_other_content() -> None:
    with pytest.raises(UnsupportedImageError):
        detect_image_format(b"GIF89a")


def test_prepare_png_returns_rgb_array() -> None:
    array = prepare_image(make_png((160, 80)), max_side=1600)
    assert array.shape == (80, 160, 3)
    assert array.dtype == np.uint8


def test_prepare_downscales_longest_side_only() -> None:
    array = prepare_image(make_png((3000, 1000)), max_side=1600)
    assert array.shape == (533, 1600, 3)


def test_prepare_never_upscales() -> None:
    array = prepare_image(make_png((100, 50)), max_side=1600)
    assert array.shape == (50, 100, 3)


def test_prepare_applies_exif_orientation() -> None:
    image = Image.new("RGB", (200, 100), (255, 255, 255))
    exif = Image.Exif()
    exif[274] = 6  # rotate 90 degrees CW
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif.tobytes())

    array = prepare_image(buffer.getvalue(), max_side=120)
    # Original 200x100 becomes 100x200, then the longest side is resized to 120.
    assert array.shape == (120, 60, 3)


def test_prepare_flattens_alpha_on_white() -> None:
    image = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    array = prepare_image(buffer.getvalue(), max_side=1600)
    assert tuple(array[0, 0]) == (255, 255, 255)


def test_prepare_corrupt_png_raises_415() -> None:
    with pytest.raises(UnsupportedImageError):
        prepare_image(b"\x89PNG\r\n\x1a\ngarbage", max_side=1600)


def test_prepare_rejects_gif_even_if_named_png() -> None:
    buffer = BytesIO()
    Image.new("RGB", (10, 10), (255, 255, 255)).save(buffer, format="GIF")
    with pytest.raises(UnsupportedImageError):
        prepare_image(buffer.getvalue(), max_side=1600)


def test_prepare_rejects_too_many_pixels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(images, "MAX_IMAGE_PIXELS", 1_000)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1_000)
    with pytest.raises(PayloadTooLargeError):
        prepare_image(make_png((100, 100)), max_side=1600)
