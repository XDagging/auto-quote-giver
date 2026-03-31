"""
Auto Quote Maker — FastAPI Service
====================================
Wraps quote.py logic as a REST API.

Endpoints:
    POST /quote   — generate a quote from an address or lat/lng
    GET  /health  — liveness check

Environment variables required:
    GOOGLE_MAPS_API_KEY   — must have Geocoding API + Maps Static API enabled

Run locally:
    uvicorn main:app --reload --port 8000
"""

import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from huggingface_hub import hf_hub_download
from pydantic import BaseModel

from quote import (
    CONFIDENCE_THRESHOLD,
    calculate_quote,
    fetch_satellite_image,
    filter_to_center_property,
    geocode,
    get_api_key,
    load_model,
    segment_image,
)

# ── Model singleton ───────────────────────────────────────────────────────────
# Loaded once at startup, reused for every request.

_model = None
_device = None


HF_REPO_ID  = "Plastuchino/auto-quote-maker"
CKPT_LOCAL  = Path("checkpoints/stage2_binary_best.pth")


def ensure_checkpoint():
    """Download checkpoint from HF Hub if not already on disk."""
    if CKPT_LOCAL.exists():
        return
    print("Downloading model checkpoint from Hugging Face Hub...")
    hf_token = os.environ.get("HF_TOKEN")
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=CKPT_LOCAL.name,
        token=hf_token,
    )
    shutil.copy(path, CKPT_LOCAL)
    print("Checkpoint ready.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _device
    ensure_checkpoint()
    print("Loading model...")
    _model, _device = load_model()
    print("Model ready.")
    yield
    _model = None


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Auto Quote Maker",
    description="Generates hardscape cleaning quotes from satellite imagery.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class QuoteRequest(BaseModel):
    address: str | None = None
    lat: float | None = None
    lng: float | None = None


class QuoteResponse(BaseModel):
    address: str | None
    lat: float
    lng: float
    total_sqft: float
    quote_usd: float
    confidence_score: float
    pixel_counts: dict
    sqft: dict
    gsd_m_per_px: float
    area_sqft_per_px: float
    price_per_sqft: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/quote", response_model=QuoteResponse)
def quote(req: QuoteRequest):
    if not req.address and not (req.lat is not None and req.lng is not None):
        raise HTTPException(status_code=400, detail="Provide 'address' or both 'lat' and 'lng'.")

    try:
        api_key = get_api_key()
    except EnvironmentError as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        if req.address:
            lat, lng = geocode(req.address, api_key)
        else:
            lat, lng = req.lat, req.lng

        img = fetch_satellite_image(lat, lng, api_key)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch satellite image: {e}")

    pred_mask, confidence, class_probs = segment_image(_model, _device, img)
    pred_mask[class_probs[:, :, 1] < CONFIDENCE_THRESHOLD] = 0
    pred_mask = filter_to_center_property(pred_mask)

    result = calculate_quote(pred_mask, confidence, lat)

    return QuoteResponse(
        address=req.address,
        lat=lat,
        lng=lng,
        **result,
    )
