FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 \
    OCR_MODEL_DIR=/app/models

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts

# Bake the PP-OCRv6 tiny models into the image so the first request does not
# download anything. edge-tts still needs outbound internet at runtime.
RUN python -m scripts.warmup_models

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

ENV HOME=/home/appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

# One process on purpose: models are loaded once and must not be duplicated.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
