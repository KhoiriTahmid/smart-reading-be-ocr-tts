"""Image validation and preprocessing for OCR."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.errors import PayloadTooLargeError, UnsupportedImageError

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"

# Guard against small compressed files that expand to enormous bitmaps.
# 50 MP is far above a phone-camera photo while keeping peak RAM bounded.
MAX_IMAGE_PIXELS = 50_000_000

# Pillow's own bomb check uses this global; set it so an oversized file is
# rejected deterministically instead of relying on the default 89 MP warning.
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def detect_image_format(data: bytes) -> str:
    """Return ``"PNG"`` or ``"JPEG"`` based on magic bytes.

    Raises ``UnsupportedImageError`` for anything else, regardless of the
    object's extension.
    """

    if data.startswith(PNG_SIGNATURE):
        return "PNG"
    if data.startswith(JPEG_SIGNATURE):
        return "JPEG"
    raise UnsupportedImageError()


def _to_rgb(image: Image.Image) -> Image.Image:
    """Flatten alpha onto white, then return an RGB image."""

    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def prepare_image(image_bytes: bytes, max_side: int) -> np.ndarray:
    """Validate magic bytes / decodability, fix EXIF orientation, downscale.

    The returned array is RGB (H, W, 3) and its longest side is at most
    ``max_side``. Images are never upscaled.
    """

    detected = detect_image_format(image_bytes)

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.format != detected:
                raise UnsupportedImageError()

            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise PayloadTooLargeError("Image dimensions are too large")

            # Load now so truncated/corrupt files fail as 415, not later.
            image.load()

            try:
                oriented = ImageOps.exif_transpose(image)
            except Exception as exc:  # malformed EXIF; orientation is best-effort
                raise UnsupportedImageError() from exc

            if max(oriented.size) > max_side:
                oriented.thumbnail(
                    (max_side, max_side), resample=Image.Resampling.LANCZOS
                )

            rgb = _to_rgb(oriented)
            array = np.asarray(rgb, dtype=np.uint8)

    except Image.DecompressionBombError as exc:
        raise PayloadTooLargeError("Image dimensions are too large") from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise UnsupportedImageError() from exc

    if array.ndim != 3 or array.shape[2] != 3:
        raise UnsupportedImageError()

    return np.ascontiguousarray(array)
