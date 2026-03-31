# ── Build stage ───────────────────────────────────────────────────────────────
# Install dependencies in a separate layer so they're cached between builds
# and don't re-download when only source code changes.
FROM python:3.12-slim AS builder

WORKDIR /app

# System libs required by OpenCV (used by albumentations) and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copy system libs installed above
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY quote.py .
COPY main.py .

# Download the model checkpoint from Hugging Face at build time.
# HF_TOKEN must be passed as a build argument:
#   docker build --build-arg HF_TOKEN=hf_... .
# On Render, set it under Settings → Environment as a build-time secret.
ARG HF_TOKEN
RUN mkdir -p checkpoints && \
    python -c "\
from huggingface_hub import hf_hub_download; \
import shutil; \
path = hf_hub_download( \
    repo_id='Plastuchino/auto-quote-maker', \
    filename='stage2_binary_best.pth', \
    token='${HF_TOKEN}', \
); \
shutil.copy(path, 'checkpoints/stage2_binary_best.pth')"

EXPOSE 8000

# GOOGLE_MAPS_API_KEY must be injected at runtime via --env or your host's
# environment variable settings (e.g. Render dashboard). Never bake it in here.
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
