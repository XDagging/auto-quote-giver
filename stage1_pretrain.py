"""
Stage 1 — Pre-train U-Net on Massachusetts Roads Dataset
=========================================================
Goal: Learn general pavement / hard-surface features before fine-tuning on
      our custom driveway/walkway/patio dataset in Stage 2.

Dataset layout expected on disk:
    <root>/
        images/   *.png   (1500×1500 RGB aerial tiles)
        masks/    *.png   (binary: 255 = road, 0 = background)

Download the dataset from:
    https://www.cs.toronto.edu/~vmnih/data/

Dependencies:
    pip install torch torchvision segmentation-models-pytorch albumentations tqdm
"""

import os
import json
from pathlib import Path

import pandas as pd

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import segmentation_models_pytorch as smp
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 0. CONFIG
# ---------------------------------------------------------------------------

CFG = {
    # --- paths ---
    "data_root":    "/home/plastuchino/Downloads/archive(1) (3)",
    "ckpt_dir":     "./checkpoints",
    "stage1_ckpt":  "./checkpoints/stage1_best.pth",

    # --- model ---
    "backbone":     "resnet34",   # swap to "efficientnet-b3" if preferred
    "in_channels":  3,
    "num_classes":  1,            # Stage 1 is binary: road vs. background

    # --- training ---
    "img_size":     640,          # resize tiles to match our inference resolution
    "batch_size":   8,
    "num_epochs":   30,
    "lr":           1e-4,
    "weight_decay": 1e-4,
    "val_split":    0.15,
    "num_workers":  4,
    "seed":         42,

    # --- scheduler ---
    "patience":     5,            # epochs without improvement before LR halved
}


# ---------------------------------------------------------------------------
# 1. DATASET
# ---------------------------------------------------------------------------

class MassRoadsDataset(Dataset):
    """
    Loads the Massachusetts Roads Dataset for binary road segmentation.

    Images : RGB PNG  (will be resized to CFG["img_size"])
    Masks  : Grayscale PNG, pixel value 255 = road → converted to 1 (float)
    """

    def __init__(self, root: str, split: str = "train", transform=None):
        root = Path(root)
        meta = pd.read_csv(root / "metadata.csv")
        meta = meta[meta["split"] == split].reset_index(drop=True)
        self.img_paths  = [root / row.tiff_image_path for _, row in meta.iterrows()]
        self.mask_paths = [root / row.tif_label_path  for _, row in meta.iterrows()]
        assert len(self.img_paths) > 0, f"No images found for split='{split}' in {root}"
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load as numpy for albumentations
        image = np.array(Image.open(self.img_paths[idx]).convert("RGB"))   # H×W×3 uint8
        mask  = np.array(Image.open(self.mask_paths[idx]).convert("L"))    # H×W uint8

        # Binarise: road pixels are 255 in the original dataset
        mask = (mask > 127).astype(np.float32)   # float32, values in {0, 1}

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]   # Tensor C×H×W, float32 [0,1]
            mask  = augmented["mask"]    # Tensor H×W,   float32 {0,1}

        # Model expects mask shape (1, H, W) for BCEWithLogitsLoss
        mask = mask.unsqueeze(0)
        return image, mask


def build_transforms(img_size: int):
    """
    Returns (train_transform, val_transform) using albumentations.
    Stage 1 augmentations are moderate — Stage 3 will apply heavier ones.
    """
    spatial = [
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
    color = [
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.GaussNoise(p=0.2),
    ]
    normalise = [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]

    train_tf = A.Compose(spatial + color + normalise)
    val_tf   = A.Compose([A.Resize(img_size, img_size)] + normalise)

    return train_tf, val_tf


# ---------------------------------------------------------------------------
# 2. LOSS — Dice Loss (handles road/background imbalance)
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """
    Binary Dice Loss.
        DL = 1 - (2 * |P ∩ G|) / (|P| + |G| + eps)
    Combined with BCEWithLogitsLoss for stable gradient flow.
    """

    def __init__(self, smooth: float = 1.0, bce_weight: float = 0.5):
        super().__init__()
        self.smooth     = smooth
        self.bce_weight = bce_weight
        self.bce        = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE component
        bce_loss = self.bce(logits, targets)

        # Dice component
        probs       = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union        = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_score   = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss    = 1.0 - dice_score.mean()

        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


# ---------------------------------------------------------------------------
# 3. METRIC — Mean IoU (binary)
# ---------------------------------------------------------------------------

def mean_iou_binary(logits: torch.Tensor, targets: torch.Tensor,
                    threshold: float = 0.5) -> float:
    """
    Binary Mean IoU computed over a batch.
    IoU = TP / (TP + FP + FN)
    Returns the average of IoU for class=0 and class=1.
    """
    preds = (torch.sigmoid(logits) > threshold).float()

    # Class 1 (road)
    inter_1 = (preds * targets).sum()
    union_1  = preds.sum() + targets.sum() - inter_1
    iou_1    = (inter_1 + 1e-6) / (union_1 + 1e-6)

    # Class 0 (background) — invert both
    bg_preds   = 1 - preds
    bg_targets = 1 - targets
    inter_0    = (bg_preds * bg_targets).sum()
    union_0    = bg_preds.sum() + bg_targets.sum() - inter_0
    iou_0      = (inter_0 + 1e-6) / (union_0 + 1e-6)

    return ((iou_0 + iou_1) / 2).item()


# ---------------------------------------------------------------------------
# 4. MODEL
# ---------------------------------------------------------------------------

def build_model(backbone: str, num_classes: int) -> nn.Module:
    """
    Instantiate a U-Net with an ImageNet-pretrained encoder via
    segmentation_models_pytorch.
    """
    model = smp.Unet(
        encoder_name=backbone,
        encoder_weights="imagenet",   # warm-start from ImageNet features
        in_channels=3,
        classes=num_classes,
        activation=None,              # raw logits — loss handles sigmoid
    )
    return model


# ---------------------------------------------------------------------------
# 5. TRAINING LOOP
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_iou  = 0.0
    context    = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, masks in tqdm(loader, desc="train" if train else "val ", leave=False):
            images = images.to(device)
            masks  = masks.to(device)

            logits = model(images)            # (B, 1, H, W)
            loss   = criterion(logits, masks)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_iou  += mean_iou_binary(logits, masks)

    n = len(loader)
    return total_loss / n, total_iou / n


def train_stage1(cfg: dict):
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    # ---- Data ----
    train_tf, val_tf = build_transforms(cfg["img_size"])
    train_ds = MassRoadsDataset(cfg["data_root"], split="train", transform=train_tf)
    val_ds   = MassRoadsDataset(cfg["data_root"], split="val",   transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True
    )

    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # ---- Model / Loss / Optimizer ----
    model     = build_model(cfg["backbone"], cfg["num_classes"]).to(device)
    criterion = DiceLoss(bce_weight=0.5)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=cfg["patience"]
    )

    # ---- Training ----
    best_val_iou = 0.0
    history = []

    for epoch in range(1, cfg["num_epochs"] + 1):
        train_loss, train_iou = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss,   val_iou   = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)

        scheduler.step(val_iou)

        log = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss,   4),
            "train_iou":  round(train_iou,  4),
            "val_iou":    round(val_iou,    4),
            "lr":         round(optimizer.param_groups[0]["lr"], 7),
        }
        history.append(log)
        print(f"Epoch {epoch:03d}/{cfg['num_epochs']} | "
              f"train_loss={train_loss:.4f} train_iou={train_iou:.4f} | "
              f"val_loss={val_loss:.4f} val_iou={val_iou:.4f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                "epoch":      epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou":    val_iou,
                "cfg":        cfg,
            }, cfg["stage1_ckpt"])
            print(f"  --> Saved best checkpoint (val_iou={val_iou:.4f})")

    # Save training history for review / plotting
    history_path = Path(cfg["ckpt_dir"]) / "stage1_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nStage 1 complete. Best val MeanIoU: {best_val_iou:.4f}")
    print(f"Checkpoint saved to: {cfg['stage1_ckpt']}")
    print(f"History saved to:    {history_path}")


# ---------------------------------------------------------------------------
# 6. ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_stage1(CFG)
