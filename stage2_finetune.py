"""
Stage 2 + 3 — Fine-tune U-Net on custom driveway / walkway / patio dataset
============================================================================
Loads Stage 1 checkpoint (encoder weights), replaces the head with a 2-class
(binary) head, and fine-tunes on COCO-format annotations from result.json.

All foreground categories (driveway, walkway, patio) are collapsed into a
single "hardscape" class so the model learns foreground vs. background.
AUC-ROC is computed on the foreground class only (class 0 excluded).

Stage 3 augmentations (brightness, rotation, flipping) are integrated into
the training transforms to simulate different lighting and satellite angles.

Class mapping:
    0 – background
    1 – hardscape   (driveway, walkway, OR patio — all COCO foreground)

Outputs:
    checkpoints/stage2_binary_best.pth      — best model by foreground AUC-ROC
    checkpoints/stage2_binary_history.json  — epoch-level metrics (appended across rounds)
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

CFG = {
    "coco_path":    "./result.json",
    "ckpt_dir":     "./checkpoints",
    "stage1_ckpt":  "./checkpoints/stage1_best.pth",
    "stage2_ckpt":  "./checkpoints/stage2_binary_best.pth",
    "history_path": "./checkpoints/stage2_binary_history.json",

    "backbone":     "resnet34",
    "num_classes":  2,           # 0=background, 1=hardscape (driveway+walkway+patio)

    "img_size":     640,
    "batch_size":   4,           # custom dataset is small; keep batch small
    "num_epochs":   40,
    "lr":           3e-4,
    "weight_decay": 1e-4,
    "val_split":    0.2,
    "num_workers":  4,
    "seed":         42,
    "patience":     8,

    # Class weights for CrossEntropy (bg=0.3, hardscape=2.0)
    # Downweight dominant background, upweight foreground
    "class_weights": [0.3, 2.0],

    "resume": False,             # True = load stage2 checkpoint instead of stage1
}

CATEGORY_TO_CLASS = {0: 1, 1: 1, 2: 1}   # COCO cat_id → mask class (all foreground → 1)


# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------

class DriveWayDataset(Dataset):
    """
    COCO-format dataset for driveway / walkway / patio segmentation.
    Rasterizes polygon annotations into 4-class masks.
    """

    def __init__(self, coco_path: str, transform=None):
        with open(coco_path) as f:
            coco = json.load(f)

        self.img_info = {img["id"]: img for img in coco["images"]}
        self.anns_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            self.anns_by_image[ann["image_id"]].append(ann)

        self.image_ids = [img["id"] for img in coco["images"]]
        self.transform = transform

    @staticmethod
    def _resolve_path(file_name: str) -> str:
        return file_name.replace("../..", "")

    def _build_mask(self, image_id: int, height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        for ann in self.anns_by_image.get(image_id, []):
            class_id = CATEGORY_TO_CLASS[ann["category_id"]]
            for poly in ann["segmentation"]:
                pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
                pil_mask = Image.fromarray(mask)
                ImageDraw.Draw(pil_mask).polygon(pts, fill=class_id)
                mask = np.array(pil_mask)
        return mask

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        info = self.img_info[image_id]

        image = np.array(Image.open(self._resolve_path(info["file_name"])).convert("RGB"))
        mask  = self._build_mask(image_id, info["height"], info["width"])

        if self.transform:
            out   = self.transform(image=image, mask=mask)
            image = out["image"]
            mask  = out["mask"]

        return image, mask.long()


def build_transforms(img_size: int):
    """
    train_tf includes Stage-3 augmentations (brightness, rotation, flipping).
    val_tf is resize + normalize only.
    """
    normalise = [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    train_tf = A.Compose([
        A.Resize(img_size, img_size),
        # Stage 3: spatial augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=45, p=0.4),
        # Stage 3: photometric augmentations (simulate time of day / lighting)
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
        A.GaussNoise(p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.2),
    ] + normalise)

    val_tf = A.Compose([A.Resize(img_size, img_size)] + normalise)
    return train_tf, val_tf


# ---------------------------------------------------------------------------
# LOSS
# ---------------------------------------------------------------------------

class MulticlassDiceCELoss(nn.Module):
    """Combined CrossEntropy + multiclass Dice loss."""

    def __init__(self, num_classes: int, class_weights=None,
                 smooth: float = 1.0, ce_weight: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth      = smooth
        self.ce_weight   = ce_weight
        w = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.ce = nn.CrossEntropyLoss(weight=w)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Move class weights to same device
        if self.ce.weight is not None:
            self.ce.weight = self.ce.weight.to(logits.device)

        ce_loss = self.ce(logits, targets)

        probs = torch.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()

        dice_loss = 0.0
        for c in range(self.num_classes):
            p = probs[:, c]
            t = targets_oh[:, c]
            inter = (p * t).sum(dim=(1, 2))
            union = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
            dice  = (2.0 * inter + self.smooth) / (union + self.smooth)
            dice_loss += (1.0 - dice.mean())
        dice_loss /= self.num_classes

        return self.ce_weight * ce_loss + (1.0 - self.ce_weight) * dice_loss


# ---------------------------------------------------------------------------
# METRIC — Macro AUC-ROC (OvR)
# ---------------------------------------------------------------------------

def compute_auc_roc(all_probs: np.ndarray, all_labels: np.ndarray,
                    num_classes: int, max_pixels: int = 500_000) -> float:
    """
    Compute macro-averaged AUC-ROC (one-vs-rest) over a random subsample
    of pixels to keep memory and computation manageable.

    all_probs:  (N, num_classes) float32
    all_labels: (N,) int
    Returns scalar AUC-ROC, or 0.0 on failure.
    """
    n = len(all_labels)
    if n > max_pixels:
        idx = np.random.choice(n, max_pixels, replace=False)
        all_probs  = all_probs[idx]
        all_labels = all_labels[idx]

    # Compute OvR AUC for each class present in y_true and macro-average.
    # Skipping absent classes avoids sklearn returning nan for undefined AUC
    # (e.g. class 3 / patio has only 2 annotations and may not appear in val).
    present = set(np.unique(all_labels).astype(int))
    if len(present) < 2:
        return 0.0

    aucs = []
    for c in range(1, num_classes):   # skip class 0 (background) — foreground only
        y_bin = (all_labels == c).astype(np.int32)
        if y_bin.sum() == 0:          # class absent in this val batch — skip
            continue
        try:
            aucs.append(roc_auc_score(y_bin, all_probs[:, c]))
        except ValueError:
            pass

    return float(np.mean(aucs)) if aucs else 0.0


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------

def build_stage2_model(cfg: dict, stage1_ckpt_path: str = None) -> nn.Module:
    """
    Build a 4-class U-Net.  If stage1_ckpt_path is given and exists, transfer
    the encoder (and decoder where shapes match) weights from Stage 1.
    Otherwise falls back to ImageNet encoder weights.
    """
    model = smp.Unet(
        encoder_name=cfg["backbone"],
        encoder_weights="imagenet",   # warm-start encoder
        in_channels=3,
        classes=cfg["num_classes"],
        activation=None,
    )

    if stage1_ckpt_path and Path(stage1_ckpt_path).exists():
        print(f"Transferring encoder weights from {stage1_ckpt_path}")
        ckpt = torch.load(stage1_ckpt_path, map_location="cpu")
        s1_state = ckpt["model_state_dict"]

        # Only copy encoder weights (keys starting with "encoder.")
        model_state = model.state_dict()
        transferred = {
            k: v for k, v in s1_state.items()
            if k.startswith("encoder.") and k in model_state
               and model_state[k].shape == v.shape
        }
        model_state.update(transferred)
        model.load_state_dict(model_state)
        print(f"Transferred {len(transferred)} encoder parameter tensors from Stage 1.")

    return model


# ---------------------------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, masks in tqdm(loader, desc="train" if train else "val ", leave=False):
            images = images.to(device)
            masks  = masks.to(device)

            logits = model(images)          # (B, C, H, W)
            loss   = criterion(logits, masks)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            # Collect probabilities for AUC-ROC (only on val to save memory)
            if not train:
                probs = torch.softmax(logits, dim=1)   # (B, C, H, W)
                # Flatten spatial dims, move to CPU
                probs_np  = probs.permute(0, 2, 3, 1).reshape(-1, logits.shape[1]).cpu().numpy()
                labels_np = masks.reshape(-1).cpu().numpy()
                all_probs.append(probs_np)
                all_labels.append(labels_np)

    n = len(loader)
    avg_loss = total_loss / n

    if not train and all_probs:
        probs_cat  = np.concatenate(all_probs,  axis=0)
        labels_cat = np.concatenate(all_labels, axis=0)
        auc = compute_auc_roc(probs_cat, labels_cat, logits.shape[1])
    else:
        auc = None

    return avg_loss, auc


def train_stage2(cfg: dict) -> float:
    """
    Fine-tune Stage 2 model. Returns best val AUC-ROC achieved this run.
    If cfg['resume'] is True, loads the existing stage2 checkpoint instead of stage1.
    """
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    # ---- Data ----
    train_tf, val_tf = build_transforms(cfg["img_size"])
    full_ds = DriveWayDataset(cfg["coco_path"])
    n_val   = max(1, int(len(full_ds) * cfg["val_split"]))
    n_train = len(full_ds) - n_val
    train_ids, val_ids = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg["seed"])
    )

    # Assign transforms by wrapping indices
    train_ds = DriveWayDataset(cfg["coco_path"], transform=train_tf)
    val_ds_raw = DriveWayDataset(cfg["coco_path"], transform=val_tf)
    train_ds = torch.utils.data.Subset(train_ds, train_ids.indices)
    val_ds   = torch.utils.data.Subset(val_ds_raw, val_ids.indices)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=True)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ---- Model ----
    if cfg["resume"] and Path(cfg["stage2_ckpt"]).exists():
        print(f"Resuming from {cfg['stage2_ckpt']}")
        model = smp.Unet(encoder_name=cfg["backbone"], encoder_weights=None,
                         in_channels=3, classes=cfg["num_classes"], activation=None)
        ckpt = torch.load(cfg["stage2_ckpt"], map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model = build_stage2_model(cfg, stage1_ckpt_path=cfg["stage1_ckpt"])

    model = model.to(device)

    # ---- Loss / Optimizer / Scheduler ----
    criterion = MulticlassDiceCELoss(
        num_classes=cfg["num_classes"],
        class_weights=cfg["class_weights"],
        ce_weight=0.5,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=cfg["patience"]
    )

    # ---- Load existing history ----
    history_path = Path(cfg["history_path"])
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    best_existing = max(
        (h.get("val_auc_roc", 0) or 0.0 for h in history), default=0.0
    )
    best_val_auc = best_existing if not np.isnan(best_existing) else 0.0

    # ---- Train ----
    for epoch in range(1, cfg["num_epochs"] + 1):
        train_loss, _        = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss,   val_auc  = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)

        val_auc = val_auc if (val_auc is not None and not np.isnan(val_auc)) else 0.0
        scheduler.step(val_auc)

        log = {
            "epoch":        epoch,
            "train_loss":   round(train_loss, 4),
            "val_loss":     round(val_loss,   4),
            "val_auc_roc":  round(val_auc,    4),
            "lr":           round(optimizer.param_groups[0]["lr"], 8),
        }
        history.append(log)
        print(f"Epoch {epoch:03d}/{cfg['num_epochs']} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_auc_roc={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc_roc":      val_auc,
                "cfg":              cfg,
            }, cfg["stage2_ckpt"])
            print(f"  --> Saved best checkpoint (val_auc_roc={val_auc:.4f})")

        history_path.write_text(json.dumps(history, indent=2))

    print(f"\nStage 2 complete. Best val AUC-ROC: {best_val_auc:.4f}")
    return best_val_auc


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_stage2(CFG)
