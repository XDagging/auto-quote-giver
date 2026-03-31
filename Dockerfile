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

# Checkpoint is downloaded at runtime startup by main.py.
# Set HF_TOKEN as a Space secret in the HuggingFace dashboard.
RUN mkdir -p checkpoints

EXPOSE 8000

# GOOGLE_MAPS_API_KEY must be injected at runtime via --env or your host's
# environment variable settings (e.g. Render dashboard). Never bake it in here.
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
