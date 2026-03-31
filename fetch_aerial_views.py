#!/usr/bin/env python3
"""
Fetch satellite aerial views of residential homes from Google Maps Static API.
All images are saved at identical zoom and resolution for consistent CVAT labeling.

Usage:
    python fetch_aerial_views.py --input addresses.csv

Input CSV format (one of):
    address                   (single column — geocoding resolves lat/lng)
    address,lat,lng           (pre-resolved coords skip geocoding)

Output:
    data/<sanitized_address>.png   640x640 px, zoom 20, satellite
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from io import BytesIO

from PIL import Image

from enhance import enhance

# ── Load API key from .env ─────────────────────────────────────────────────────

def load_env(path: str = ".env") -> dict:
    env = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                env[key.strip()] = val.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return env

_env = load_env()
API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY") or _env.get("GOOGLE_MAPS_API_KEY")

# ── Image settings (DO NOT change — keeps all images comparable in CVAT) ──────

ZOOM        = 20           # zoom 20 ≈ 50m across — ideal for driveways/patios/walkways
OUTPUT_SIZE = (640, 640)   # final saved resolution — identical across all images
FETCH_SCALE = 2            # fetch at 2× (1280×1280) then downsample — supersampling for sharper edges
MAP_TYPE    = "satellite"
IMG_FORMAT  = "png"

# ── API endpoints ──────────────────────────────────────────────────────────────

STATIC_MAP_URL = "https://maps.googleapis.com/maps/api/staticmap"
GEOCODE_URL    = "https://maps.googleapis.com/maps/api/geocode/json"

REQUEST_DELAY  = 0.1       # seconds between API calls


def geocode(address: str) -> tuple[float, float]:
    params = urllib.parse.urlencode({"address": address, "key": API_KEY})
    with urllib.request.urlopen(f"{GEOCODE_URL}?{params}") as resp:
        data = json.loads(resp.read())
    if data["status"] != "OK":
        raise ValueError(f"Geocoding failed: {data['status']}")
    loc = data["results"][0]["geometry"]["location"]
    return loc["lat"], loc["lng"]


def fetch_image(lat: float, lng: float) -> bytes:
    """
    Fetch a satellite image at 2× scale (1280×1280), downsample to 640×640 with
    LANCZOS, apply the enhancement pipeline, and return PNG bytes.
    """
    w, h    = OUTPUT_SIZE
    params  = urllib.parse.urlencode({
        "center":  f"{lat},{lng}",
        "zoom":    ZOOM,
        "size":    f"{w}x{h}",   # Google interprets size before scale multiplier
        "scale":   FETCH_SCALE,  # → actual fetch is 1280×1280
        "maptype": MAP_TYPE,
        "format":  IMG_FORMAT,
        "key":     API_KEY,
    })
    with urllib.request.urlopen(f"{STATIC_MAP_URL}?{params}") as resp:
        raw = resp.read()

    img = Image.open(BytesIO(raw)).convert("RGB")
    img = img.resize(OUTPUT_SIZE, Image.LANCZOS)   # downsample → supersampled sharpness
    img = enhance(img)                             # CLAHE + unsharp mask + saturation

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def safe_filename(address: str) -> str:
    name = re.sub(r"[^\w\s-]", "", address).strip()
    return re.sub(r"\s+", "_", name)[:120]


def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if rows and "address" not in rows[0]:
        sys.exit("CSV must have an 'address' column.")
    return rows


def main():
    if not API_KEY:
        sys.exit("GOOGLE_MAPS_API_KEY not found in .env or environment.")

    parser = argparse.ArgumentParser(description="Download 640x640 satellite images of residential homes.")
    parser.add_argument("--input",   default="addresses.csv", help="CSV of addresses (default: addresses.csv)")
    parser.add_argument("--out-dir", default="data",          help="Output directory (default: data/)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows  = load_csv(args.input)
    total = len(rows)

    w, h = OUTPUT_SIZE
    print(f"Fetching {total} image(s) — zoom={ZOOM}, output={w}×{h}px, fetch_scale={FETCH_SCALE}×, type={MAP_TYPE}\n")

    success = skipped = failed = 0

    for i, row in enumerate(rows, 1):
        address  = row["address"].strip()
        out_path = os.path.join(args.out_dir, f"{safe_filename(address)}.png")

        if os.path.exists(out_path):
            print(f"[{i}/{total}] SKIP  {address}")
            skipped += 1
            continue

        try:
            if row.get("lat") and row.get("lng"):
                lat, lng = float(row["lat"]), float(row["lng"])
            else:
                lat, lng = geocode(address)
                time.sleep(REQUEST_DELAY)

            image_bytes = fetch_image(lat, lng)
            time.sleep(REQUEST_DELAY)

            with open(out_path, "wb") as f:
                f.write(image_bytes)

            print(f"[{i}/{total}] OK    ({lat:.6f}, {lng:.6f})  →  {out_path}")
            success += 1

        except Exception as e:
            print(f"[{i}/{total}] FAIL  {address}  —  {e}")
            failed += 1

    w, h = OUTPUT_SIZE
    print(f"\n{success} saved, {skipped} skipped, {failed} failed.")
    print(f"All images: {w}×{h}px, zoom={ZOOM}, fetched at {FETCH_SCALE}×, {MAP_TYPE}, enhanced")


if __name__ == "__main__":
    main()
