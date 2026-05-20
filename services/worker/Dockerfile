# ── Base: Python 3.11 + system deps ───────────────────────────────────────────
# For GPU support swap base image to pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependencies ───────────────────────────────────────────────────────────────
FROM base AS deps

# Install PyTorch CPU-only first (smallest image; GPU users override in compose)
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
# Install remaining deps (whisperx installs from git, needs git above)
RUN pip install --no-cache-dir -r requirements.txt

# ── Runtime ────────────────────────────────────────────────────────────────────
FROM base AS runner

COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin
COPY . .

# Models are downloaded on first use and cached in CACHE_DIR.
# Pre-download during build by uncommenting:
# RUN python -c "import whisperx; whisperx.load_model('large-v3','cpu')"

RUN useradd -m -u 1001 worker
RUN mkdir -p /cache && chown worker:worker /cache
USER worker

ENV CACHE_DIR=/cache
ENV PORT=8000

EXPOSE 8000

CMD ["python", "main.py"]
