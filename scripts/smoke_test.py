#!/usr/bin/env python3
"""End-to-end smoke test against a running service.

It uploads a sample image to S3, calls /ocr, then calls /tts for both
Indonesian voices and downloads the resulting MP3s. Requires the same S3
environment variables as the service.

Example:
    python scripts/smoke_test.py \
      --base-url http://127.0.0.1:8000 \
      --image tests/samples/indonesian_text.png
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from app.config import Settings  # noqa: E402
from app.services.storage import S3Storage  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default=os.getenv("OCR_SERVICE_URL", "http://127.0.0.1:8000"),
    )
    parser.add_argument("--api-key", default=os.getenv("OCR_SERVICE_API_KEY", ""))
    parser.add_argument(
        "--image",
        default=str(
            Path(__file__).resolve().parents[1]
            / "tests"
            / "samples"
            / "indonesian_text.png"
        ),
    )
    parser.add_argument(
        "--file-key",
        default=None,
        help="Use an existing uploads/ key instead of uploading --image",
    )
    parser.add_argument("--voice", choices=["female", "male"], default="female")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: set OCR_SERVICE_API_KEY or pass --api-key", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    storage = S3Storage(Settings())
    headers = {"x-api-key": args.api_key}
    sample_upload_key = None

    try:
        with httpx.Client(base_url=args.base_url, timeout=200) as client:
            health = client.get("/health")
            print(f"GET /health -> {health.status_code} {health.text}")
            health.raise_for_status()

            if args.file_key:
                file_key = args.file_key
            else:
                image_path = Path(args.image)
                data = image_path.read_bytes()
                suffix = image_path.suffix.lower()
                content_type = (
                    "image/png"
                    if data.startswith(b"\x89PNG\r\n\x1a\n")
                    else "image/jpeg"
                )
                if suffix not in {".png", ".jpg", ".jpeg"}:
                    print("ERROR: sample must be PNG/JPG", file=sys.stderr)
                    return 2
                sample_upload_key = f"uploads/smoke-{uuid.uuid4()}/source{suffix}"
                asyncio.run(storage.put_bytes(sample_upload_key, data, content_type))
                file_key = sample_upload_key
                print(f"Uploaded sample to s3://.../{file_key}")

            ocr_response = client.post(
                "/ocr", json={"file_key": file_key}, headers=headers
            )
            print(f"POST /ocr -> {ocr_response.status_code}")
            ocr_response.raise_for_status()
            ocr_body = ocr_response.json()
            print(ocr_body)
            text = ocr_body.get("text") or "Halo, ini uji coba teks bahasa Indonesia."

            voices = [args.voice]
            other = "male" if args.voice == "female" else "female"
            voices.append(other)

            for voice in voices:
                history_id = str(uuid.uuid4())
                payload = {
                    "history_id": history_id,
                    "text": text,
                    "voice": voice,
                    "rate": "+0%",
                }
                response = client.post("/tts", json=payload, headers=headers)
                print(f"POST /tts ({voice}) -> {response.status_code}")
                response.raise_for_status()
                body = response.json()
                print(body)

                if voice == args.voice:
                    # A retried job must overwrite the same deterministic key.
                    retry = client.post("/tts", json=payload, headers=headers)
                    retry.raise_for_status()
                    if retry.json()["audio_key"] != body["audio_key"]:
                        raise RuntimeError("deterministic audio key changed")

                audio = asyncio.run(
                    storage.get_bytes(body["audio_key"], max_bytes=20 * 1024 * 1024)
                )
                if not audio or (
                    not audio.startswith(b"ID3") and not audio.startswith(b"\xff")
                ):
                    raise RuntimeError("downloaded object is not an MP3")
                if body["duration_seconds"] <= 0:
                    raise RuntimeError("duration_seconds is not positive")
                target = output_dir / f"smoke-{history_id}-{voice}.mp3"
                target.write_bytes(audio)
                print(f"Saved {len(audio)} bytes to {target}")

    finally:
        if sample_upload_key:
            print(f"Note: uploaded test object {sample_upload_key} was left in S3")

    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
