"""
Driveway Cleaning Quote Generator
===================================
Loads the trained Stage 2 model, runs inference on a satellite image of a
residential address, and returns a quote broken down by surface type.

Usage:
    # By address (fetches from Google Maps Static API):
    python quote.py --address "5006 Elsmere Avenue, Bethesda, MD 20814"

    # By lat/lng (fetches image):
    python quote.py --lat 39.0105 --lng -77.0998

    # From local image file (must provide lat for GSD):
    python quote.py --image path/to/img.png --lat 39.0105

GSD formula (zoom always = 20):
    S = 40_075_016.686 × cos(lat_radians) / 2^28  [meters/pixel]
    area_per_pixel = S²                            [m²]

Pricing:
    PRICE_PER_SQFT = 0.20   ($0.20 per sq ft = 20 cents)
    Edit this constant to adjust your rate.

Confidence score:
    Mean of max-softmax probabilities across all detected foreground pixels.
    A value close to 1.0 means the model is very certain; near 0.5 means uncertain.
"""

import argparse
import json
import math
import os
import time
import urllib.parse
import urllib.request
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
import segmentation_models_pytorch as smp

# ── Pricing ──────────────────────────────────────────────────────────────────
PRICE_PER_SQFT = 0.20    # $0.20 per sq ft (20 cents) — edit to adjust rate

# ── GSD constants ────────────────────────────────────────────────────────────
EARTH_CIRCUMFERENCE_M = 40_075_016.686   # meters at equator
ZOOM                  = 20               # MUST match fetch_aerial_views.py
SQFT_PER_SQM          = 10.7639

# ── Model config ─────────────────────────────────────────────────────────────
STAGE2_CKPT   = "./checkpoints/stage2_binary_best.pth"
BACKBONE      = "resnet34"
NUM_CLASSES   = 2
IMG_SIZE      = 640

CLASS_NAMES = {0: "background", 1: "hardscape"}
FOREGROUND_CLASSES = [1]

# Minimum model confidence to count a pixel as hardscape.
# Raises the bar above the default argmax threshold (0.5), filtering out
# grass, shadows, and low-certainty boundary pixels.
CONFIDENCE_THRESHOLD = 0.8

# ── Google Maps API ───────────────────────────────────────────────────────────
STATIC_MAP_URL = "https://maps.googleapis.com/maps/api/staticmap"
GEOCODE_URL    = "https://maps.googleapis.com/maps/api/geocode/json"


def load_env(path: str = ".env") -> dict:
    env = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return env


def get_api_key() -> str:
    env = load_env()
    key = os.environ.get("GOOGLE_MAPS_API_KEY") or env.get("GOOGLE_MAPS_API_KEY")
    if not key:
        raise EnvironmentError("GOOGLE_MAPS_API_KEY not set in .env or environment.")
    return key


def geocode(address: str, api_key: str) -> tuple[float, float]:
    params = urllib.parse.urlencode({"address": address, "key": api_key})
    with urllib.request.urlopen(f"{GEOCODE_URL}?{params}") as resp:
        data = json.loads(resp.read())
    if data["status"] != "OK":
        raise ValueError(f"Geocoding failed: {data['status']} for '{address}'")
    loc = data["results"][0]["geometry"]["location"]
    return loc["lat"], loc["lng"]


def fetch_satellite_image(lat: float, lng: float, api_key: str) -> Image.Image:
    """Fetch 640×640 satellite image at zoom=20 (matches training data)."""
    params = urllib.parse.urlencode({
        "center":  f"{lat},{lng}",
        "zoom":    ZOOM,
        "size":    "640x640",
        "scale":   1,
        "maptype": "satellite",
        "format":  "png",
        "key":     api_key,
    })
    with urllib.request.urlopen(f"{STATIC_MAP_URL}?{params}") as resp:
        raw = resp.read()
    return Image.open(BytesIO(raw)).convert("RGB")


# ── GSD formula ───────────────────────────────────────────────────────────────

def gsd_meters_per_pixel(lat_degrees: float, zoom: int = ZOOM) -> float:
    """
    Ground Sample Distance formula:
        S = C × cos(lat) / 2^(zoom + 8)   [meters/pixel]
    """
    lat_rad = math.radians(lat_degrees)
    return EARTH_CIRCUMFERENCE_M * math.cos(lat_rad) / (2 ** (zoom + 8))


def area_sqft_per_pixel(lat_degrees: float) -> float:
    s = gsd_meters_per_pixel(lat_degrees)
    return (s ** 2) * SQFT_PER_SQM


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str = STAGE2_CKPT) -> torch.nn.Module:
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {ckpt_path}\n"
            "Run `python train_pipeline.py` first."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(
        encoder_name=BACKBONE,
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, device


def preprocess_image(img: Image.Image) -> torch.Tensor:
    """Resize to 640×640, normalize to ImageNet stats, return (1, 3, H, W)."""
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    arr  = (arr - mean) / std
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


@torch.no_grad()
def segment_image(model, device, img: Image.Image):
    """
    Run inference on a PIL image.

    Returns:
        pred_mask  : (H, W) int array — class predictions (0=background, 1=hardscape)
        confidence : (H, W) float array — max softmax prob per pixel
        class_probs: (H, W, 2) float array — full softmax distribution
    """
    tensor = preprocess_image(img).to(device)
    logits = model(tensor)                          # (1, 2, H, W)
    probs  = F.softmax(logits, dim=1)[0]            # (2, H, W)
    probs_np   = probs.permute(1, 2, 0).cpu().numpy()   # (H, W, 2)
    pred_mask  = np.argmax(probs_np, axis=-1).astype(np.uint8)
    confidence = np.max(probs_np, axis=-1)
    return pred_mask, confidence, probs_np


# ── Quote calculation ─────────────────────────────────────────────────────────

def calculate_quote(pred_mask: np.ndarray, confidence: np.ndarray,
                    lat: float) -> dict:
    """
    Convert pixel counts to square feet and compute the cleaning quote.

    Returns a dict with:
        - pixel_counts   : per-class pixel counts
        - sqft           : per-class area in sq ft
        - total_sqft     : total foreground sq ft
        - quote          : dollar amount
        - confidence_score: mean confidence over foreground pixels
        - gsd_m_per_px   : ground sample distance (informational)
        - area_sqft_per_px: sq ft per pixel (informational)
    """
    px_sqft = area_sqft_per_pixel(lat)
    gsd     = gsd_meters_per_pixel(lat)

    pixel_counts = {cls: int(np.sum(pred_mask == cls)) for cls in range(NUM_CLASSES)}
    sqft         = {cls: pixel_counts[cls] * px_sqft  for cls in range(NUM_CLASSES)}

    total_sqft = sum(sqft[c] for c in FOREGROUND_CLASSES)
    quote      = total_sqft * PRICE_PER_SQFT

    # Confidence: mean max-softmax over foreground pixels
    fg_mask = np.isin(pred_mask, FOREGROUND_CLASSES)
    if fg_mask.any():
        confidence_score = float(confidence[fg_mask].mean())
    else:
        confidence_score = 0.0

    return {
        "pixel_counts":      {CLASS_NAMES[c]: pixel_counts[c] for c in range(NUM_CLASSES)},
        "sqft":              {CLASS_NAMES[c]: round(sqft[c], 2) for c in FOREGROUND_CLASSES},
        "total_sqft":        round(total_sqft, 2),
        "quote_usd":         round(quote, 2),
        "confidence_score":  round(confidence_score, 4),
        "gsd_m_per_px":      round(gsd, 6),
        "area_sqft_per_px":  round(px_sqft, 6),
        "price_per_sqft":    PRICE_PER_SQFT,
        "zoom":              ZOOM,
    }


# ── Property isolation ───────────────────────────────────────────────────────

def filter_to_center_property(pred_mask: np.ndarray,
                               center_fraction: float = 0.2) -> np.ndarray:
    """
    Discard hardscape blobs that don't overlap the central region of the image.

    The Google Maps Static API always centers the image on the geocoded address,
    so the target property occupies the middle of the frame. Blobs that only
    appear near the edges belong to neighbors and are removed.

    center_fraction: fraction of image width/height to treat as 'center'.
                     0.5 means the inner 50% (160–480 px on a 640-px image).
    """
    labeled, num_features = ndimage.label(pred_mask == 1)
    if num_features == 0:
        return pred_mask

    h, w = pred_mask.shape
    margin_y = int(h * (1 - center_fraction) / 2)
    margin_x = int(w * (1 - center_fraction) / 2)

    center_labels = set(
        np.unique(labeled[margin_y:h - margin_y, margin_x:w - margin_x])
    ) - {0}

    if not center_labels:
        # Nothing in center — fall back to the single largest component
        counts = np.bincount(labeled.ravel())[1:]
        center_labels = {int(np.argmax(counts)) + 1}

    filtered = np.zeros_like(pred_mask)
    for lbl in center_labels:
        filtered[labeled == lbl] = 1
    return filtered


# ── Visualization ────────────────────────────────────────────────────────────

def save_mask_overlay(img: Image.Image, pred_mask: np.ndarray, out_path: str):
    """Save the satellite image with hardscape pixels highlighted in red."""
    base = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay_arr = np.array(overlay)
    overlay_arr[pred_mask == 1] = [255, 0, 0, 140]   # red, semi-transparent
    result_img = Image.alpha_composite(base, Image.fromarray(overlay_arr))
    result_img.convert("RGB").save(out_path)
    print(f"  Mask saved to: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def print_quote(result: dict, address: str = None):
    print("\n" + "=" * 50)
    print("  DRIVEWAY CLEANING QUOTE")
    print("=" * 50)
    if address:
        print(f"  Address : {address}")
    print(f"  GSD     : {result['gsd_m_per_px']:.4f} m/px  "
          f"({result['area_sqft_per_px']:.4f} sq ft/px)")
    print()
    print("  Surface breakdown:")
    for name, sqft in result["sqft"].items():
        print(f"    {name.capitalize():10s}: {sqft:8.1f} sq ft")
    print(f"    {'TOTAL':10s}: {result['total_sqft']:8.1f} sq ft")
    print()
    print(f"  Rate    : ${result['price_per_sqft']:.4f} / sq ft")
    print(f"  QUOTE   : ${result['quote_usd']:.2f}")
    print()
    print(f"  Model confidence: {result['confidence_score']*100:.1f}%")
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate a driveway cleaning quote.")
    parser.add_argument("--address", help="Street address to geocode and fetch")
    parser.add_argument("--lat",   type=float, help="Latitude (decimal degrees)")
    parser.add_argument("--lng",   type=float, help="Longitude (decimal degrees)")
    parser.add_argument("--image", help="Path to a local satellite image PNG")
    parser.add_argument("--ckpt",      default=STAGE2_CKPT, help="Model checkpoint path")
    parser.add_argument("--save-mask", metavar="PATH", help="Save prediction overlay image to this path")
    args = parser.parse_args()

    if not args.address and not (args.lat and args.lng) and not args.image:
        parser.error("Provide --address, --lat/--lng, or --image.")

    # ── Resolve image + lat ───────────────────────────────────────────────────
    lat = args.lat
    img = None

    if args.image:
        img = Image.open(args.image).convert("RGB")
        if lat is None:
            parser.error("--lat is required when using --image (needed for GSD).")

    else:
        api_key = get_api_key()
        if args.address:
            print(f"Geocoding: {args.address}")
            lat, lng = geocode(args.address, api_key)
            print(f"  -> ({lat:.6f}, {lng:.6f})")
        else:
            lat, lng = args.lat, args.lng

        print(f"Fetching satellite image at zoom={ZOOM}...")
        img = fetch_satellite_image(lat, lng, api_key)
        time.sleep(0.1)

    # ── Inference ─────────────────────────────────────────────────────────────
    print("Loading model...")
    model, device = load_model(args.ckpt)

    print("Running segmentation...")
    pred_mask, confidence, class_probs = segment_image(model, device, img)

    # Apply confidence threshold — zero out pixels where the model is below
    # CONFIDENCE_THRESHOLD certainty. This cuts grass, shadows, and the
    # fuzzy boundary pixels that bleed into neighboring properties.
    pred_mask[class_probs[:, :, 1] < CONFIDENCE_THRESHOLD] = 0

    pred_mask = filter_to_center_property(pred_mask)

    if args.save_mask:
        save_mask_overlay(img, pred_mask, args.save_mask)

    # ── Quote ─────────────────────────────────────────────────────────────────
    result = calculate_quote(pred_mask, confidence, lat)
    print_quote(result, address=args.address)

    # Also dump raw JSON for downstream use
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
