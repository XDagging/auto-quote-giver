# Auto Quote Maker — Architecture & Usage

Automatically generates pressure-washing quotes for residential properties by
detecting hard surfaces (driveways, walkways, patios) in satellite imagery using
a trained segmentation model.

---

## How it works

### Overview

```
Address / lat-lng
      │
      ▼
Google Maps Static API  ──►  640×640 satellite image
      │
      ▼
U-Net segmentation model  ──►  per-pixel hardscape / background mask
      │
      ▼
Pixel count × GSD²  ──►  square footage
      │
      ▼
sq ft × $0.20  ──►  quote
```

### Model architecture

The model is a **U-Net with a ResNet34 encoder** trained in two stages:

**Stage 1 — Pavement pre-training**
Pre-trained on the Massachusetts Roads Dataset (binary: road vs. background).
This teaches the encoder general pavement features before it ever sees
driveway/walkway imagery. Output checkpoint: `checkpoints/stage1_best.pth`.

**Stage 2 — Hardscape fine-tuning (binary)**
Fine-tuned on 100 hand-labeled satellite images (COCO format, `result.json`).
All foreground categories — driveway, walkway, and patio — are collapsed into
a single **hardscape** class. The model outputs two classes:

| Class | Label |
|-------|-------|
| 0 | background |
| 1 | hardscape (driveway + walkway + patio) |

Loss: 50% CrossEntropy (weights `[0.3, 2.0]`) + 50% Dice loss.
Best validation AUC-ROC (foreground only, background excluded): **0.9517**
Output checkpoint: `checkpoints/stage2_binary_best.pth`

### Why binary?

Previously the model predicted 4 classes (background, driveway, walkway, patio)
separately. Two problems arose:

1. **Metric inflation** — AUC-ROC was macro-averaged including background, which
   dominates >90% of pixels. The model could score ~0.95 by doing nothing but
   predicting background well.
2. **Data starvation** — patio had only 2 annotations in the entire dataset,
   making it impossible to learn as a distinct class.

Collapsing all foreground into one hardscape class eliminates both problems. The
AUC-ROC now measures only foreground detection quality.

### Inference pipeline

1. **Geocode** the address via Google Geocoding API → lat/lng
2. **Fetch** a 640×640 satellite image at zoom 20 via Google Maps Static API
3. **Preprocess**: resize to 640×640, normalize to ImageNet stats
4. **Segment**: forward pass through U-Net → softmax → argmax → binary mask
5. **Measure**: count hardscape pixels, convert to sq ft using the GSD formula
6. **Quote**: multiply total sq ft by the rate ($0.20/sq ft)

### Ground Sample Distance (GSD)

Pixel area varies with latitude. The formula used (zoom = 20):

```
S = 40,075,016.686 × cos(lat_radians) / 2²⁸   [meters/pixel]
area_per_pixel = S² × 10.7639                   [sq ft/pixel]
```

### Confidence score

After segmentation, the mean max-softmax probability across all predicted
hardscape pixels is reported as a confidence score. Values above ~0.75 are
reliable; below ~0.60 indicates the model is uncertain (e.g. poor image
quality, unusual surface materials).

---

## Setup

### 1. Install dependencies

```bash
cd auto-quote-maker
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision segmentation-models-pytorch albumentations \
            scikit-learn tqdm pillow pandas
```

### 2. Set your Google Maps API key

Create a `.env` file in the project root:

```
GOOGLE_MAPS_API_KEY=your_key_here
```

The key needs **Geocoding API** and **Maps Static API** enabled.

---

## Training

### Train from scratch

```bash
source .venv/bin/activate
python3 train_pipeline.py
```

The pipeline:
1. Checks for `checkpoints/stage1_best.pth` — skips Stage 1 if it exists
2. Runs Stage 2 for up to 40 epochs per round
3. Repeats with a halved learning rate until foreground AUC-ROC >= 0.7

Stage 1 requires the Massachusetts Roads Dataset at:
```
/home/plastuchino/Downloads/archive(1) (3)/images/
/home/plastuchino/Downloads/archive(1) (3)/masks/
```

### Run Stage 2 only

```bash
python3 stage2_finetune.py
```

### Training outputs

| File | Contents |
|------|----------|
| `checkpoints/stage1_best.pth` | Best Stage 1 checkpoint (road pre-training) |
| `checkpoints/stage2_binary_best.pth` | Best Stage 2 checkpoint (binary hardscape) |
| `checkpoints/stage2_binary_history.json` | Per-epoch metrics (loss, AUC-ROC, LR) |

---

## Generating a quote

> **Note:** `quote.py` currently points to the old 4-class checkpoint. Update
> these two constants at the top of `quote.py` before running:
> ```python
> STAGE2_CKPT = "./checkpoints/stage2_binary_best.pth"
> NUM_CLASSES  = 2
> ```

### By address

```bash
python3 quote.py --address "5006 Elsmere Avenue, Bethesda, MD 20814"
```

### By lat/lng

```bash
python3 quote.py --lat 39.0105 --lng -77.0998
```

### From a local image file

```bash
python3 quote.py --image path/to/image.png --lat 39.0105
```

`--lat` is required when using a local image (needed for GSD calculation).

### Custom checkpoint

```bash
python3 quote.py --address "..." --ckpt ./checkpoints/stage2_binary_best.pth
```

### Example output

```
==================================================
  DRIVEWAY CLEANING QUOTE
==================================================
  Address : 5006 Elsmere Avenue, Bethesda, MD 20814
  GSD     : 0.0596 m/px  (0.0383 sq ft/px)

  Surface breakdown:
    Hardscape :    842.3 sq ft

  Rate    : $0.2000 / sq ft
  QUOTE   : $168.46

  Model confidence: 83.2%
==================================================
```

### Adjusting the rate

Edit `PRICE_PER_SQFT` at the top of `quote.py`:

```python
PRICE_PER_SQFT = 0.20    # $0.20 per sq ft — change this to update pricing
```

---

## Fetching satellite images in bulk

```bash
python3 fetch_aerial_views.py --input addresses.csv
```

Input CSV can be:
- Single column: `address` (geocoded automatically)
- Three columns: `address,lat,lng` (skips geocoding)

Images are saved to `data/<sanitized_address>.png` at 640×640, zoom 20,
with CLAHE contrast enhancement and unsharp masking applied automatically.

---

## File reference

| File | Purpose |
|------|---------|
| `stage1_pretrain.py` | Pre-train on Massachusetts Roads Dataset |
| `stage2_finetune.py` | Fine-tune binary hardscape model on `result.json` |
| `train_pipeline.py` | Orchestrates Stage 1 → Stage 2 with iterative fine-tuning |
| `quote.py` | Inference: address → satellite image → segmentation → quote |
| `fetch_aerial_views.py` | Bulk satellite image downloader with enhancement |
| `enhance.py` | CLAHE + unsharp mask + saturation boost for image quality |
| `populate_addresses.py` | Discovers residential addresses via OpenStreetMap |
| `result.json` | COCO-format annotations (100 images, 139 polygons) |
