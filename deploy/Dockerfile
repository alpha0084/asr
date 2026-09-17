# Standalone ASR app image — FastAPI + whisperx/pyannote (GPU via torch cu12 wheels).
# Postgres and Ollama run as their own compose services, so this image is just the app.
# Build context is the repo root: `docker build -f deploy/Dockerfile .`
FROM python:3.12-slim

# ffmpeg for audio cleanup/resample; curl/ca-certs for model downloads; libpq for psycopg
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# non-root runtime user (fixed uid so bind-mounted data stays consistently owned)
RUN useradd -m -u 10001 -s /bin/bash asr
WORKDIR /app

# Deps first for layer caching (torch/whisperx/pyannote — the heavy layer).
COPY requirements-server.txt ./
RUN pip install --no-cache-dir -U pip setuptools wheel \
    && pip install --no-cache-dir -r requirements-server.txt 'psycopg[binary]'

# App code
COPY backend ./backend
COPY frontend ./frontend
COPY deploy/entrypoint.sh /usr/local/bin/entrypoint.sh
# /app/data is the app's DATA_DIR (downloads, tmp) AND the model cache — mount the
# `appdata` volume here so recordings + caches persist. Pre-create + own it so the
# named volume initialises as asr-owned on first mount.
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && mkdir -p /app/data/hf /app/data/cache /app/data/_downloads /app/data/tmp \
    && chown -R asr:asr /app

USER asr
ENV HF_HOME=/app/data/hf \
    XDG_CACHE_HOME=/app/data/cache \
    PYTHONUNBUFFERED=1
EXPOSE 5530
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
