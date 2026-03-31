"""
Image enhancement pipeline for satellite aerial imagery.

Pipeline (applied to every image):
  1. CLAHE on luminance channel  — lifts shadows, boosts local contrast
  2. Unsharp mask                — sharpens edges (driveways, rooflines, curbs)
  3. Saturation boost            — helps visually separate grass / concrete / asphalt

This module is imported by fetch_aerial_views.py and can also be run standalone
to re-enhance images that were already downloaded:

    python3 enhance.py                        # re-process data/*.png → data/enhanced/
    python3 enhance.py --in-dir data --out-dir data/enhanced
"""

from __future__ import annotations

import argparse
import os
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# ── Tunable parameters ─────────────────────────────────────────────────────────

CLAHE_CLIP_LIMIT  = 2.5    # higher = more aggressive contrast boost
CLAHE_TILE_GRID   = 8      # NxN grid of tiles (8 → 80×80 px tiles on a 640px image)
UNSHARP_RADIUS    = 1.5    # neighbourhood radius for unsharp mask
UNSHARP_PERCENT   = 130    # sharpening strength (100 = no change, 200 = very aggressive)
UNSHARP_THRESHOLD = 3      # minimum edge delta to sharpen (avoids amplifying flat noise)
SATURATION_FACTOR = 1.25   # 1.0 = original, 1.25 = mild boost

# ──────────────────────────────────────────────────────────────────────────────


def _clahe(channel: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    """
    Vectorised CLAHE on a single uint8 channel (H×W).

    Algorithm:
      1. Divide into tile_grid×tile_grid tiles.
      2. Per tile: clip histogram at clip_limit × avg_count, redistribute excess.
      3. Build cumulative distribution function (CDF) mapping for each tile.
      4. Bilinear-interpolate between the four nearest tile CDFs for each pixel.
    """
    h, w = channel.shape

    # Pad so dimensions are exact multiples of tile_grid
    ph = ((h + tile_grid - 1) // tile_grid) * tile_grid
    pw = ((w + tile_grid - 1) // tile_grid) * tile_grid
    padded = np.pad(channel, ((0, ph - h), (0, pw - w)), mode="reflect")

    th = ph // tile_grid   # tile height in pixels
    tw = pw // tile_grid   # tile width  in pixels

    # Build per-tile CDFs  →  shape (tile_grid, tile_grid, 256)
    cdfs = np.zeros((tile_grid, tile_grid, 256), dtype=np.float32)
    clip  = max(1, int(clip_limit * th * tw / 256))

    for r in range(tile_grid):
        for c in range(tile_grid):
            tile = padded[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
            hist, _ = np.histogram(tile.ravel(), bins=256, range=(0, 256))
            # Clip and redistribute
            excess      = int(np.sum(np.maximum(hist - clip, 0)))
            hist        = np.minimum(hist, clip)
            hist       += excess // 256
            hist[excess % 256:] += 1  # spread remainder 1-by-1
            cdf         = np.cumsum(hist).astype(np.float32)
            lo, hi      = cdf.min(), cdf.max()
            cdfs[r, c]  = (cdf - lo) / max(hi - lo, 1) * 255.0

    # Vectorised bilinear interpolation across all pixels
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    # Map each pixel to a fractional tile coordinate (centre-aligned)
    tr = (yy + th / 2.0) / th - 0.5
    tc = (xx + tw / 2.0) / tw - 0.5

    r0 = np.clip(tr.astype(np.int32),     0, tile_grid - 1)
    c0 = np.clip(tc.astype(np.int32),     0, tile_grid - 1)
    r1 = np.clip(r0 + 1,                  0, tile_grid - 1)
    c1 = np.clip(c0 + 1,                  0, tile_grid - 1)
    dr = (tr - r0).astype(np.float32)
    dc = (tc - c0).astype(np.float32)

    v   = channel.ravel().astype(np.int32)
    rf  = r0.ravel(); r1f = r1.ravel()
    cf  = c0.ravel(); c1f = c1.ravel()
    drf = dr.ravel(); dcf = dc.ravel()

    q00 = cdfs[rf,  cf,  v]
    q10 = cdfs[r1f, cf,  v]
    q01 = cdfs[rf,  c1f, v]
    q11 = cdfs[r1f, c1f, v]

    result = (
        q00 * (1 - drf) * (1 - dcf) +
        q10 * drf       * (1 - dcf) +
        q01 * (1 - drf) * dcf       +
        q11 * drf       * dcf
    )
    return np.clip(result, 0, 255).reshape(h, w).astype(np.uint8)


def enhance(img: Image.Image) -> Image.Image:
    """
    Apply the full enhancement pipeline to a PIL Image.
    Input and output are both RGB.
    """
    # ── 1. CLAHE on luminance only (preserves natural colour) ─────────────────
    ycbcr = img.convert("YCbCr")
    y, cb, cr = ycbcr.split()

    y_arr       = np.array(y, dtype=np.uint8)
    y_enhanced  = _clahe(y_arr, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID)

    y_new  = Image.fromarray(y_enhanced, mode="L")
    result = Image.merge("YCbCr", (y_new, cb, cr)).convert("RGB")

    # ── 2. Unsharp mask — sharpens edges without ringing ──────────────────────
    result = result.filter(ImageFilter.UnsharpMask(
        radius=UNSHARP_RADIUS,
        percent=UNSHARP_PERCENT,
        threshold=UNSHARP_THRESHOLD,
    ))

    # ── 3. Saturation boost — differentiates grass / concrete / asphalt ───────
    result = ImageEnhance.Color(result).enhance(SATURATION_FACTOR)

    return result


# ── Standalone batch re-processing ────────────────────────────────────────────

def _process_dir(in_dir: str, out_dir: str) -> None:
    in_path  = Path(in_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pngs = sorted(in_path.glob("*.png"))
    if not pngs:
        print(f"No PNG files found in {in_dir}")
        return

    print(f"Enhancing {len(pngs)} image(s)  {in_dir} → {out_dir}\n")
    for i, src in enumerate(pngs, 1):
        dst = out_path / src.name
        img     = Image.open(src).convert("RGB")
        result  = enhance(img)
        result.save(dst, format="PNG")
        print(f"[{i}/{len(pngs)}] {src.name}")

    print(f"\nDone. Enhanced images saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance satellite images in batch.")
    parser.add_argument("--in-dir",  default="data",          help="Source directory (default: data/)")
    parser.add_argument("--out-dir", default="data/enhanced", help="Output directory (default: data/enhanced/)")
    args = parser.parse_args()
    _process_dir(args.in_dir, args.out_dir)
