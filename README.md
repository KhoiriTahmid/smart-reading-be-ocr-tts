# OCR + TTS Service (FastAPI)

Private CPU-only service for the smart reading device thesis project. It does
exactly two things:

1. **OCR** — read a PNG/JPG from object storage and return text.
2. **TTS** — synthesize Indonesian speech with edge-tts, upload the MP3 to
   object storage, and return its deterministic key.

It deliberately has **no database, Redis/BullMQ, end-user auth, or public
exposure**. The NestJS API/worker is the only caller.

```
Website / Device -> NestJS API -> BullMQ (Redis) -> NestJS worker --HTTP--> this service
                                      |                                      |
                                 PostgreSQL                         Backblaze B2 (S3 API)
```

## Stack

| Area | Choice |
|---|---|
| Runtime | Python 3.12, FastAPI, Uvicorn (1 worker) |
| OCR | RapidOCR 3.10 + ONNX Runtime CPU, PP-OCRv6 **tiny** det + rec |
| TTS | edge-tts (`id-ID-GadisNeural`, `id-ID-ArdiNeural`), ffmpeg for MP3 concat |
| Storage | boto3 against Backblaze B2 S3 API |
| Concurrency | one heavy job, 2 waiting requests, then HTTP 429 |

The OCR models are downloaded during the Docker image build and reused at
runtime. edge-tts is an unofficial **online** service and still needs outbound
internet.

## Project layout

```
app/
  main.py                 # app factory, lifespan, exception handlers
  config.py               # pydantic-settings
  security.py             # x-api-key dependency
  schemas.py              # request/response models
  routers/                # health.py, ocr.py, tts.py
  services/               # ocr_engine.py, tts_engine.py, storage.py, limiter.py
  utils/                  # images.py, text.py
tests/                    # unit tests + optional integration tests + samples
scripts/
  warmup_models.py        # used in Docker build; downloads/initialises models
  smoke_test.py           # real end-to-end check against a running instance
  memory_check.py         # peak-RSS measurement helper
  generate_samples.py     # regenerate tests/samples
Dockerfile
docker-compose.yml        # local dev (loopback port)
docker-compose.prod.yml   # production-style (internal network, no port)
requirements.txt          # pinned runtime dependency closure
requirements-dev.txt      # test dependencies (includes requirements.txt)
.env.example
```

## Setup (local)

Requirements: Python **3.12**, `ffmpeg` (which includes `ffprobe`), and network
access for the model warmup.

```bash
sudo apt install ffmpeg            # Debian/Ubuntu
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env
# Edit .env: set OCR_SERVICE_API_KEY and the S3_* values.
# For local-only experimentation you may set APP_ENV=development and keep the
# placeholder key; production refuses to start with "change-me".

python -m scripts.warmup_models    # downloads PP-OCRv6 tiny once
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Do **not** add `--workers` or a Uvicorn worker count. The OCR engine is loaded
once per process, and the service is designed for one process on 2 vCPU / 4 GB.

### Environment variables

| Variable | Default | Notes |
|---|---|---|
| `APP_ENV` | `production` | `production`, `development`, or `test` |
| `OCR_SERVICE_API_KEY` | `change-me` | Shared secret from NestJS; production rejects the placeholder |
| `S3_ENDPOINT` | — | e.g. `https://s3.us-west-004.backblazeb2.com` |
| `S3_REGION` | — | e.g. `us-west-004` |
| `S3_ACCESS_KEY` | — | Service account key (read `uploads/*`, write `audio/*`) |
| `S3_SECRET_KEY` | — | Service account secret |
| `S3_BUCKET` | — | Private bucket name |
| `OCR_MIN_CONFIDENCE` | `0.5` | Drop OCR lines below this recognition score |
| `OCR_MAX_SIDE` | `1600` | Longest side after downscale; never upscales |
| `MAX_IMAGE_BYTES` | `10485760` | Hard download limit (10 MB) |
| `TTS_MAX_CHARS` | `20000` | Requests above this get HTTP 413 |
| `TTS_CHUNK_CHARS` | `900` | Target chunk size (sentence-aware; 100–3500) |
| `TTS_DEFAULT_VOICE` | `female` | Used when `voice` is omitted/null |
| `MAX_WAITING_REQUESTS` | `2` | Requests allowed to wait behind the active job |
| `OMP_NUM_THREADS` | `2` | Also sets ONNX Runtime intra-op threads to 2 |
| `OCR_TIMEOUT_SECONDS` | `60` | Whole-request timeout (including waiting) |
| `TTS_TIMEOUT_SECONDS` | `150` | Whole-request timeout (including waiting) |
| `OCR_MODEL_DIR` | unset | Optional model cache dir; Docker uses `/app/models` |
| `FFMPEG_BINARY` / `FFPROBE_BINARY` | `ffmpeg` / `ffprobe` | Tool paths |
| `LOG_LEVEL` | `INFO` | Python log level |

The service refuses to start if S3 configuration is incomplete or if `ffmpeg`
is missing. `ffprobe` (normally installed with ffmpeg) is used for accurate
durations; without it, durations are estimated from the MP3 size. Health
checks are unauthenticated.

## API contract

All endpoints except `/health` require `x-api-key: <OCR_SERVICE_API_KEY>`.
Missing or wrong keys return **401** with `{"detail": "..."}`.

The curl examples below use `$OCR_SERVICE_API_KEY`; export it in your shell
(or substitute the value from `.env`).

Interactive docs/OpenAPI are disabled because this is a private service.

### `GET /health`

```bash
curl -s http://127.0.0.1:8000/health
```

Ready response (`200`):

```json
{"status":"ok","ocr_loaded":true,"tts_ready":true}
```

While models/tools are not ready, **503** is returned with
`"status":"starting"`.

### `POST /ocr`

Request:

```json
{"file_key": "uploads/3f2b.../source.jpg"}
```

```bash
curl -s -X POST http://127.0.0.1:8000/ocr \
  -H 'content-type: application/json' \
  -H "x-api-key: $OCR_SERVICE_API_KEY" \
  -d '{"file_key":"uploads/3f2b-example/source.png"}'
```

`200`:

```json
{"text":"Baris pertama\nBaris kedua","lines":2,"avg_confidence":0.93}
```

- The only field NestJS needs is `text`.
- No detected text is a **200** with `"text": ""`, `"lines": 0`,
  `"avg_confidence": 0.0`.
- `file_key` must start with `uploads/` and must not contain `..`, backslashes,
  or control characters.
- The file type is determined by magic bytes, **not** the extension.
- Only PNG/JPEG are accepted. The image is EXIF-oriented, converted to RGB
  with transparency flattened onto white, and downscaled to at most 1600 px on
  the longest side before OCR.
- Lines below `OCR_MIN_CONFIDENCE` are dropped; remaining lines are joined
  with `\n`.

### `POST /tts`

Request:

```json
{
  "history_id": "3f2b6c1e-9d4a-4f5b-8c7d-1234567890ab",
  "text": "Teks yang akan dibacakan...",
  "voice": "female",
  "rate": "+0%"
}
```

`voice` and `rate` are optional. `voice` is `"female"` (default) or `"male"`.
`rate` must match `^[+-]\d{1,2}%$` and defaults to `"+0%"`.

```bash
curl -s -X POST http://127.0.0.1:8000/tts \
  -H 'content-type: application/json' \
  -H "x-api-key: $OCR_SERVICE_API_KEY" \
  -d '{"history_id":"3f2b6c1e-9d4a-4f5b-8c7d-1234567890ab","text":"Selamat pagi, ini teks bahasa Indonesia.","voice":"female","rate":"+0%"}'
```

`200`:

```json
{
  "audio_key": "audio/3f2b6c1e-9d4a-4f5b-8c7d-1234567890ab.mp3",
  "duration_seconds": 3.2,
  "chars": 40
}
```

- The output key is built **only** from the validated UUID:
  `audio/{history_id}.mp3`. A repeated call overwrites the same object.
- Upload uses `ContentType: audio/mpeg`.
- Text is NFC-normalized, whitespace/control characters are cleaned, then
  split at sentence boundaries into ~900-character chunks. Chunks are
  synthesized sequentially and concatenated into one MP3 with ffmpeg
  (`-c copy`, fallback re-encode at 64 kbps mono).
- `chars` is the length of the normalized text that was synthesized.
- `duration_seconds` comes from `ffprobe`; if probing fails, the service falls
  back to a bitrate-based estimate.
- A single token longer than the chunk size is hard-split (pathological case,
  e.g. a 2,000-character URL); normal text is never split mid-word.
- Hard limit: text over `TTS_MAX_CHARS` (20,000) returns **413**.

## Error contract (what NestJS retries)

`4xx` except 408/429 is permanent; 429/5xx/network errors are retried with
backoff.

| Situation | Status | Retried? |
|---|---|---|
| Missing/wrong `x-api-key` | 401 | No |
| Invalid body, bad `file_key`, bad UUID | 422 | No |
| Object not found | 404 | No |
| Not a valid PNG/JPG, or corrupt | 415 | No |
| Image or text too large | 413 | No |
| Service busy (waiting room full) | 429 + `Retry-After: 5` | Yes |
| edge-tts/storage unreachable, timeout | 503 | Yes |
| Unexpected exception | 500 | Yes |

The error body is always `{"detail": "short safe message"}`. Error responses
never contain stack traces, credentials, user text, or object keys. The
application logs likewise avoid request bodies and exception messages.

## Concurrency, timeouts, resources

- One global `asyncio.Semaphore(1)`; one OCR **or** TTS job at a time.
- At most `MAX_WAITING_REQUESTS` (default 2) requests may wait; the next one
  gets **429 immediately** with `Retry-After`.
- OCR inference and image preprocessing run in a thread pool so `/health`
  stays responsive.
- OCR whole-request timeout: 60 s. TTS whole-request timeout: 150 s. Timeouts
  return **503**.
- ONNX Runtime intra-op threads set to 2; `OMP_NUM_THREADS=2`.
- Image downloads are capped at 10 MB and decoded images at 50 MP; images are
  downscaled to 1600 px max side.
- Expected steady-state RAM is well below 1 GB. Docker compose limits the
  container to 3 GB and 2 CPUs.

### Memory measurement

`scripts/memory_check.py` reports the process peak RSS from
`resource.getrusage(RUSAGE_SELF).ru_maxrss`:

```bash
python scripts/memory_check.py             # OCR only
python scripts/memory_check.py --with-tts  # OCR + live edge-tts
```

Measured on the development machine (AMD Ryzen 5 5500U, 14 GB RAM, Linux,
Python 3.12.15, RapidOCR 3.10.0, ONNX Runtime 1.31.0, PP-OCRv6 tiny). The
mixed run was pinned to two CPUs with `taskset -c 0,1` to approximate the
2 vCPU target:

| Workload | Peak RSS |
|---|---|
| Model load + one 1200×520 OCR | 292 MB |
| OCR + live TTS synthesis (1,000 chars, 93 s MP3) | 453 MB worst observed (292 MB typical) |

The 3 GB compose limit is a hard guard; the measured peak is well under the
1.5 GB acceptance threshold and the 4 GB target hardware. RSS was read from
`resource.getrusage(RUSAGE_SELF).ru_maxrss` in the same process that loaded the
models; child `ffmpeg` processes are not included (they use only a few MB).

## Running with Docker

```bash
cp .env.example .env
# edit .env (API key + B2 credentials)
docker compose up --build
curl -s http://127.0.0.1:8000/health
```

- Base image `python:3.12-slim`, ffmpeg installed with apt.
- `scripts/warmup_models.py` runs during the build, so models are baked in.
- Container runs as non-root user `appuser`.
- One Uvicorn process, no `--workers`.
- Image `HEALTHCHECK` calls `/health`.
- Local compose publishes only `127.0.0.1:8000`, not a public interface.
- Production example (`docker-compose.prod.yml`) publishes **no port** and
  attaches to an existing internal Docker network:

```bash
# Build the image once (the prod file intentionally has no build: key)
docker build -t smart-reading-ocr-service:local .
docker network create backend
docker compose -f docker-compose.prod.yml up -d
# NestJS worker reaches http://ocr-service:8000 on the `backend` network.
```

`docker compose build` with the local compose file produces the same image tag
and is equivalent if you prefer that.

## Testing

Unit tests mock S3 and edge-tts and make no network calls. The MP3 concat test
uses the real `ffmpeg` binary and is skipped automatically when ffmpeg is not
installed:

```bash
pytest
```

On the verified environment (Python 3.12.15, ffmpeg 7.0.2) this runs **89
tests** (88 unit/contract tests plus the real-ffmpeg concat test); the five
network/model tests are skipped by default.

Real-model/live tests are marked `integration` and skipped by default:

```bash
# Real RapidOCR on tests/samples (downloads models if warmup was not run)
RUN_OCR_INTEGRATION=1 pytest -m integration tests/test_ocr_integration.py

# Live edge-tts voice list + female/male synthesis (outbound internet)
RUN_TTS_INTEGRATION=1 pytest -m integration tests/test_tts_live.py

# Only the real ffmpeg concat test
pytest tests/test_tts_concat.py

# Everything, including real models and live edge-tts
RUN_OCR_INTEGRATION=1 RUN_TTS_INTEGRATION=1 pytest
```

All five integration tests passed during verification (two OCR sample images
and three live edge-tts checks), and a separate live 5,000-character run
produced six chunks, one continuous 2.25 MB MP3, and decoded cleanly with
`ffmpeg`.

End-to-end smoke test against a running instance (uploads a sample to S3,
calls OCR/TTS for both voices, verifies deterministic overwrite, downloads the
MP3s):

```bash
python scripts/smoke_test.py --base-url http://127.0.0.1:8000
```

The smoke test needs the same `S3_*` env vars and `OCR_SERVICE_API_KEY`. By
default it uploads the sample image and downloads the resulting MP3s with those
credentials, so the key needs write access to `uploads/*` and read access to
`audio/*` for the smoke run (the service itself only needs read on `uploads/*`
and write on `audio/*`). Pass `--file-key uploads/...` to test an object that
was uploaded some other way and avoid granting the extra upload permission.

Docker image build/runtime was not executed in the author's environment
because the Docker socket was not accessible. The Dockerfile and both compose
files are provided and were reviewed, but the first `docker build` should be
run on a machine with Docker access before deployment.

## Verification notes (brief section 11)

Verified on 2026-10-26 with Python 3.12.15, `ffmpeg`/`ffprobe` 7.0.2, and the
pinned versions in `requirements.txt`: FastAPI 0.162.0, Uvicorn 0.54.0,
RapidOCR 3.10.0, ONNX Runtime 1.31.0, edge-tts 7.3.1, boto3/botocore 1.43.118,
NumPy 2.5.4, OpenCV headless 5.0.0.93, Pillow 13.0.0.

- **RapidOCR params/result shape**: verified against `rapidocr==3.10.0`.
  `OCRVersion.PPOCRV6`, `ModelType.TINY`, and the keys
  `Det.ocr_version`, `Rec.ocr_version`, `Det.model_type`, `Rec.model_type`,
  `Global.use_cls`, `Global.text_score`, `Global.max_side_len`,
  `Global.model_root_dir`, and
  `EngineConfig.onnxruntime.intra_op_num_threads` all exist. The result object
  exposes `txts` / `scores` and uses `None` when nothing is found. The engine
  is created once in lifespan.
- **OCR sample**: both PNG and JPEG samples (1200×520, Indonesian text with
  currency, email, and phone number) passed the integration test at the 0.80
  similarity threshold with 4 detected lines and confidence above 0.5.
- **edge-tts API/voices**: verified against `edge-tts==7.3.1`.
  `Communicate(text, voice, rate=..., connect_timeout=10, receive_timeout=60)`
  and `stream()` have the expected signatures. Both `id-ID-GadisNeural` and
  `id-ID-ArdiNeural` exist via `list_voices()` and synthesized live audio; see
  `tests/test_tts_live.py`.
- **botocore checksums**: botocore 1.43.118 accepts and applies
  `request_checksum_calculation="when_required"` and
  `response_checksum_validation="when_required"`; `_client_config()` falls back
  to a plain config on older versions, and the fallback is unit-tested.
- **MP3 concatenation**: `ffmpeg -f concat -safe 0 -i ... -c copy` produced a
  clean, decodable MP3 for both the real-ffmpeg test and a live 5,000-character
  edge-tts run (six chunks, 2.25 MB, 375.6 s). If `-c copy` fails, the code
  re-encodes with `libmp3lame` at 64 kbps mono.
- **HTTP edge cases**: the full unit suite covers 401/422/404/415/413/429/503/
  500 mapping, `Retry-After`, deterministic audio overwrite, and the bounded
  waiting room. A real Uvicorn process was also started with dummy S3 settings
  to verify `/health` 200, unauthenticated 401, invalid-body 422, and a
  storage-unreachable 503.
- **Not verified**: real Backblaze B2 upload/download (no B2 credentials
  were available in the build environment); the boto3 wrapper is unit-tested
  with a mocked client and the checksum configuration was checked directly.
  The Docker image build and compose runtime were also not executed because the
  Docker socket was unavailable in the build environment; see the note under
  "Testing".

## Assumptions and known limitations

- **OCR orientation classifier disabled** (`Global.use_cls = false`). The
  service already applies EXIF orientation, and the classifier is not part of
  the PP-OCRv6 tiny bundle; this saves a model and latency. Images with
  sideways text *without* EXIF orientation may OCR poorly.
- **edge-tts is online and unofficial.** It can fail, rate-limit, or change.
  Failures become 503 so NestJS retries. There is no fallback engine in this
  version, but `EdgeTTSEngine.synthesize(text, voice, rate) -> bytes` is a
  small interface that can be swapped for Piper later.
- **No runtime voice-list check.** Startup does not call `list_voices()`
  because that would add an outbound dependency; the two voices are fixed and
  verified by the live integration test.
- **TTS duration fallback** uses ~48 kbps when `ffprobe` cannot read a file;
  it is an estimate, not sample-accurate.
- **Pathological long token**: a single word longer than `TTS_CHUNK_CHARS` is
  split mid-word, because no alternative can satisfy the chunk limit.
- **Image limits**: JPEG/PNG only, 10 MB download, 50 MP decode, 1600 px long
  side. No PDF, multipage, handwriting or table extraction.
- **Requests are not streamed.** Audio is fully synthesized and uploaded
  before the response is returned.
- **OCR timeout and threads**: a CPU-bound OCR call runs in a worker thread.
  Python cannot force-kill that thread, so in the rare case it exceeds 60 s
  the HTTP request returns 503 and the limiter is released while the orphan
  thread finishes. Normal OCR calls take ~2–3 s on the target hardware.
- **S3 errors**: permission/credential/network problems map to 503 rather than
  404 to avoid hiding configuration mistakes as missing objects.
- **Production API key**: the app refuses `change-me` when
  `APP_ENV=production`. Use a real random secret.
- The project root is this directory (not a nested `ocr-service/` folder).
- `chars` reports normalized length, not the raw request length.

## NestJS integration note

The NestJS worker should pass `voice` on `/tts` (previously it sent only
`history_id` and `text`). Keep BullMQ worker `concurrency: 1` and request
timeouts around 120 s (OCR) / 180 s (TTS) to match this service.
