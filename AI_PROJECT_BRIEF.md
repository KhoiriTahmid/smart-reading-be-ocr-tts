# Project Brief: OCR + TTS Service (FastAPI)

> **Instructions to the AI building this project:**
> Read this whole document first. Everything marked **DECIDED** is final, do not substitute libraries or change the API contract. Items under **VERIFY** must be checked against the installed library versions before relying on them. If something is ambiguous, state your assumption in the README and continue. Do not ask for clarification on things already specified here.

---

## 1. What this service is

A small, single-purpose Python service that does two things:

1. **OCR**: read an image from object storage, return the extracted text.
2. **TTS**: take text, synthesize Indonesian speech, upload the MP3 to object storage, return the storage key.

It is one of two backend services in a "smart reading device" system (a thesis project):

- **Service 1 (already built, NestJS)**: public API for the website and the physical device. Handles auth, uploads, the database (`reading_histories` table), and a BullMQ (Redis) queue. A BullMQ worker inside NestJS calls **this** service over REST.
- **Service 2 (this project, FastAPI)**: only does OCR and TTS. It is **private** (internal network only).

```
Website / Device -> NestJS API -> BullMQ (Redis) -> NestJS worker --HTTP--> THIS SERVICE
                                       |                                       |
                                  PostgreSQL                          Object storage (S3 API)
```

**This service must NOT:** connect to the database, connect to Redis/BullMQ, handle end-user auth, or expose anything publicly.

## 2. Constraints (DECIDED)

| Item | Value |
|---|---|
| Target hardware | **2 vCPU, 4 GB RAM, no GPU** |
| Language / framework | Python 3.12, FastAPI, Uvicorn |
| Process model | **1 Uvicorn worker** (models are loaded once; do not multiply memory) |
| Concurrency | **One heavy job (OCR or TTS) at a time** |
| Cost | Free/cheap only. No paid APIs. |
| Language of content | Primarily **Indonesian**, may contain English words, numbers, currency (Rp), emails, phone numbers |
| Input images | **PNG and JPG only** |

## 3. Technology choices (DECIDED)

### OCR: RapidOCR + PP-OCRv6 **tiny**

- Library: `rapidocr` with `onnxruntime` (CPU). No PaddlePaddle dependency.
- Model: PP-OCRv6 tiny for **both** detection and recognition (about 6 MB total).
- Benchmarked on 2 cores: roughly 2.3 s per image average, peak RAM about 440 MB.
- Config used in the benchmark (**VERIFY** parameter names against the installed `rapidocr` version; they changed across releases, `rapidocr>=3.10.0` routes models via `default_models.yaml`):

```python
from rapidocr import RapidOCR, OCRVersion, ModelType

engine = RapidOCR(params={
    "Det.ocr_version": OCRVersion.PPOCRV6,
    "Rec.ocr_version": OCRVersion.PPOCRV6,
    "Det.model_type": ModelType.TINY,
    "Rec.model_type": ModelType.TINY,
})
result = engine(image_path_or_ndarray)
texts = result.txts   # may be None when nothing is found
scores = result.scores
```

- Pre-processing (do this before OCR): apply EXIF orientation (`ImageOps.exif_transpose`), convert to RGB, downscale so the longest side is **at most 1600 px** (never upscale).
- Post-processing: drop lines with confidence below a configurable threshold (default `0.5`), join remaining lines with `\n`, strip trailing whitespace. Return empty string if nothing is found.
- Create the engine **once** at startup and reuse it.

### TTS: edge-tts (online) with two Indonesian voices

- Library: `edge-tts`. Output is MP3 directly.
- Voices (**DECIDED**):

| `voice` request value | edge-tts voice name |
|---|---|
| `"female"` (default) | `id-ID-GadisNeural` |
| `"male"` | `id-ID-ArdiNeural` |

- edge-tts is an **unofficial online service**: it can fail, rate-limit, or change without notice. Treat failures as transient (see error handling). **Do not** add other TTS engines in this version; keep the TTS code behind a small interface (`synthesize(text, voice, rate) -> bytes`) so a fallback engine (e.g. Piper) can be added later.
- Optional request field `rate` (edge-tts format, e.g. `"-10%"`, `"+20%"`). Validate with regex `^[+-]\d{1,2}%$`, default `"+0%"`.
- **Long text handling:** normalize text, split into chunks at sentence boundaries (about 800 to 1000 characters each, never mid-word), synthesize chunks sequentially, then concatenate into one MP3 (use `ffmpeg` concat demuxer with `-c copy`; fall back to re-encoding if concat fails). Hard limit: reject text over **20,000 characters** with HTTP 413.
- Text normalization before synthesis: Unicode NFC, collapse repeated whitespace/newlines, remove control characters, keep punctuation (it drives pauses). Do not over-process numbers or abbreviations in this version.

### Object storage: Backblaze B2 via the S3 API (DECIDED)

- Use `boto3`. The bucket is private. This service has its own credentials (read on `uploads/*`, write on `audio/*`).
- Required client config (B2 can reject the SDK's default checksum headers):

```python
import boto3
from botocore.config import Config

s3 = boto3.client(
    "s3",
    endpoint_url=settings.s3_endpoint,
    aws_access_key_id=settings.s3_access_key,
    aws_secret_access_key=settings.s3_secret_key,
    region_name=settings.s3_region,
    config=Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"},
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
        retries={"max_attempts": 3, "mode": "standard"},
    ),
)
```

(**VERIFY**: the two checksum options only exist in newer botocore; make the code tolerate older versions.)

- boto3 is blocking. Always call it through `run_in_threadpool` (or `asyncio.to_thread`) so the event loop stays free.
- Key layout (**DECIDED**, set by the NestJS side):
  - Input image: `uploads/{history_id}/source.png` or `source.jpg` (the key is given in the request).
  - Output audio: `audio/{history_id}.mp3` (**deterministic**: a retried call must overwrite the same object, never create duplicates).

## 4. API contract (DECIDED, the NestJS side is already written against this)

Base URL is internal, e.g. `http://ocr-service:8000`. Every endpoint except `/health` requires the header `x-api-key: <OCR_SERVICE_API_KEY>` (compare with `hmac.compare_digest`; return 401 if missing or wrong).

### `GET /health`

No auth. Returns 200 `{"status":"ok","ocr_loaded":true,"tts_ready":true}` once the OCR model is loaded. Return 503 while starting.

### `POST /ocr`

Request:
```json
{ "file_key": "uploads/3f2b.../source.jpg" }
```
Response 200:
```json
{ "text": "extracted text...", "lines": 12, "avg_confidence": 0.93 }
```
- `text` is the only field the NestJS client reads; the others are extra info.
- If no text is found: return **200** with `"text": ""` (NestJS handles this case itself).
- Validate that `file_key` starts with `uploads/` and contains no `..`.
- Validate the real file type by magic bytes (PNG: `89 50 4E 47 0D 0A 1A 0A`, JPEG: `FF D8 FF`), not by extension. Max downloaded size: 10 MB.

### `POST /tts`

Request:
```json
{ "history_id": "3f2b...uuid", "text": "Teks yang akan dibacakan...", "voice": "female", "rate": "+0%" }
```
`voice` and `rate` are optional.

Response 200:
```json
{ "audio_key": "audio/3f2b...uuid.mp3", "duration_seconds": 12.4, "chars": 340 }
```
- `history_id` must be a valid UUID; the output key is built **only** from it (`audio/{history_id}.mp3`). Never accept an arbitrary output key from the client.
- Upload with `ContentType: audio/mpeg`.

### Error handling (important, it controls NestJS retry behavior)

The NestJS client treats **4xx (except 408 and 429) as permanent**: the job fails immediately with no retry. It retries **429, 5xx, and network errors** with exponential backoff. Map errors accordingly:

| Situation | HTTP status | Retried by NestJS? |
|---|---|---|
| Missing or wrong `x-api-key` | 401 | No |
| Invalid body, bad `file_key`, bad UUID | 422 | No |
| Object not found in bucket | 404 | No |
| File is not a valid PNG/JPG, or corrupt | 415 | No |
| Image or text too large | 413 | No |
| Service busy (job queue full) | **429** + `Retry-After` header | Yes |
| edge-tts or storage unreachable, timeouts | **503** | Yes |
| Unexpected exception | 500 | Yes |

Error body shape: `{"detail": "short safe message"}`. Never include stack traces, credentials, or user text in responses or logs.

## 5. Concurrency and resource rules

- Models and clients are created in the FastAPI `lifespan` handler, once.
- Use one global `asyncio.Semaphore(1)` around heavy work (OCR inference; TTS chunk synthesis and ffmpeg). Allow a small waiting room (for example 2 waiting requests); beyond that return **429** immediately instead of piling up memory.
- Run CPU-bound OCR with `run_in_threadpool` so `/health` stays responsive.
- Set `OMP_NUM_THREADS=2` (and ONNX Runtime intra-op threads to 2).
- Per-request timeouts: OCR about 60 s, TTS about 150 s total. On timeout return 503.
- Stay well under 4 GB RAM (expected steady state under about 1 GB).

## 6. Suggested project structure

```
ocr-service/
  app/
    main.py              # app factory, lifespan (load models), routers, exception handlers
    config.py            # pydantic-settings, reads env
    security.py          # x-api-key dependency
    schemas.py           # request/response models (pydantic v2)
    routers/
      health.py
      ocr.py
      tts.py
    services/
      ocr_engine.py      # RapidOCR wrapper: load(), extract(image_bytes) -> OcrResult
      tts_engine.py      # edge-tts wrapper + chunking + ffmpeg concat
      storage.py         # boto3 wrapper: get_bytes(key), put_bytes(key, data, content_type)
      limiter.py         # semaphore + waiting-room logic
    utils/
      text.py            # normalize_text(), split_sentences()/chunk_text()
      images.py          # magic-byte check, EXIF transpose, resize
  tests/
    test_ocr.py  test_tts.py  test_security.py  test_text.py
    samples/             # a few PNG/JPG images with known text
  scripts/
    warmup_models.py     # downloads/initializes OCR model (used in Docker build)
    smoke_test.py        # end-to-end check against a running instance
  Dockerfile
  docker-compose.yml
  requirements.txt       # pinned versions
  .env.example
  README.md
```

## 7. Environment variables (`.env.example`)

```
APP_ENV=production
OCR_SERVICE_API_KEY=change-me

S3_ENDPOINT=https://s3.us-west-004.backblazeb2.com
S3_REGION=us-west-004
S3_ACCESS_KEY=
S3_SECRET_KEY=
S3_BUCKET=

OCR_MIN_CONFIDENCE=0.5
OCR_MAX_SIDE=1600
MAX_IMAGE_BYTES=10485760
TTS_MAX_CHARS=20000
TTS_CHUNK_CHARS=900
TTS_DEFAULT_VOICE=female
MAX_WAITING_REQUESTS=2
OMP_NUM_THREADS=2
```

## 8. Docker

- Base image `python:3.12-slim`; install `ffmpeg` via apt.
- Use **`opencv-python-headless`** (not `opencv-python`) to avoid missing `libGL` errors.
- Run `scripts/warmup_models.py` **during the image build** so the OCR model is already inside the image: no download on first request and no outbound dependency at start (edge-tts still needs outbound internet at runtime).
- Run as a non-root user. Single `uvicorn app.main:app --host 0.0.0.0 --port 8000` process (no `--workers`).
- Add a `HEALTHCHECK` that calls `/health`.
- `docker-compose.yml` for local dev: this service only (storage is remote B2). Do not expose the port publicly in the production example; show it on an internal network.
- Memory limit in compose: `mem_limit: 3g`, `cpus: 2`.

## 9. Testing and acceptance criteria

Write `pytest` tests with `S3` and `edge_tts` mocked (no network in unit tests), plus a `scripts/smoke_test.py` for a real end-to-end check.

The project is done when:

1. `POST /ocr` on a real Indonesian sample image returns the expected text (allow small OCR errors in tests, compare with a similarity threshold).
2. `POST /tts` with `voice=female` and `voice=male` both produce playable MP3 files at `audio/{history_id}.mp3`, and calling it twice with the same `history_id` overwrites the same key.
3. Text of about 5,000 characters is chunked and returned as one continuous MP3.
4. Wrong or missing `x-api-key` returns 401 on `/ocr` and `/tts`; `/health` works without a key.
5. A non-image file renamed to `.jpg` returns 415. A missing key returns 404.
6. When edge-tts is made to fail (mock), `/tts` returns 503, not 500, and not a hung request.
7. Sending more concurrent requests than the waiting room allows returns 429 with `Retry-After`.
8. Peak memory stays under 1.5 GB during a mixed OCR + TTS run (document how it was measured).
9. README explains: setup, env vars, running locally, running in Docker, the API contract with curl examples, and known limitations.

## 10. Out of scope (do not build)

- Database, Redis, BullMQ, user auth, rate limiting per user.
- PDF input, multi-page documents, handwriting, table extraction.
- Any GPU path, any paid or API-key-based AI service.
- Streaming audio, a web UI, voice cloning.
- Additional TTS voices or engines (design for it, but do not implement).

## 11. Items to VERIFY while building

1. Exact `RapidOCR` parameter and enum names for PP-OCRv6 tiny in the installed version.
2. Result object shape (`txts`, `scores`, possibly `None` when empty).
3. `edge-tts` current API (`Communicate(text, voice, rate=...)`, `.save()` or streaming) and that both voice names exist via `edge_tts.list_voices()`.
4. Whether `botocore` supports `request_checksum_calculation` in the pinned version.
5. That MP3 concat with `-c copy` yields a clean file; otherwise re-encode (`libmp3lame`, 64 kbps, mono).

## 12. Note for the NestJS side (not part of this project, tracked here for context)

The existing NestJS `OcrClientService` currently sends `{ history_id, text }` to `/tts`. After this service is ready, it should also pass `voice` (`"female"` or `"male"`), for example stored per reading history or chosen per request. Set the NestJS BullMQ worker `concurrency` to **1** to match this service's single-job design, with request timeouts of about 120 s (OCR) and 180 s (TTS).
