import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from dataset import CamVidDataset, CAMVID_COLORMAP, NUM_CLASSES, VOID_CLASS
from model import get_model

# -----------------------------
# Config
# -----------------------------
CHECKPOINT   = "models/deeplabv3_camvid_best.pth"   # swap to _final.pth if preferred
OUTPUT_DIR   = "predictions"
NUM_SAMPLES  = 8    # how many val images to visualize
IMG_SIZE     = (512,512)

CLASS_NAMES = [
    "Sky", "Building", "Pole", "Road Marking", "Road",
    "Pavement", "Tree", "Sign Symbol", "Fence", "Car",
    "Pedestrian", "Bicyclist", "Unlabelled"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Color lookup: class index → RGB (0-255)
# -----------------------------
def class_to_color(class_map):
    """Convert (H, W) class index map → (H, W, 3) RGB image."""
    h, w    = class_map.shape
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(CAMVID_COLORMAP):
        mask = class_map == cls_idx
        rgb_img[mask] = color
    return rgb_img

# -----------------------------
# Load model
# -----------------------------
checkpoint = torch.load(CHECKPOINT, map_location=device)



model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Loaded checkpoint: {CHECKPOINT}")

# -----------------------------
# Dataset
# -----------------------------
# Replace this:
val_dataset = CamVidDataset("dataset/CamVid/val", "dataset/CamVid/val_labels")

# With this:
from torch.utils.data import random_split
full_dataset = CamVidDataset("dataset/CamVid/train", "dataset/CamVid/train_labels")
train_size   = int(0.8 * len(full_dataset))
val_size     = len(full_dataset) - train_size
_, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # same seed = same split as train.py
)

# -----------------------------
# mIoU over full val set
# -----------------------------
def compute_miou_full(model, dataset, num_classes=NUM_CLASSES, ignore_index=VOID_CLASS):
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    ious_per_class = {c: {"inter": 0, "union": 0} for c in range(num_classes) if c != ignore_index}

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            preds  = model(images)['out'].argmax(dim=1).cpu()   # (B, H, W)
            for cls in ious_per_class:
                pred_c = (preds == cls)
                true_c = (masks == cls)
                ious_per_class[cls]["inter"] += (pred_c & true_c).sum().item()
                ious_per_class[cls]["union"] += (pred_c | true_c).sum().item()

    ious = []
    for cls, vals in ious_per_class.items():
        if vals["union"] > 0:
            iou = vals["inter"] / vals["union"]
            ious.append(iou)
            print(f"  Class {cls:>2} ({CLASS_NAMES[cls]:<14}): IoU = {iou:.4f}")
    miou = sum(ious) / len(ious) if ious else 0.0
    print(f"\n  → Mean IoU: {miou:.4f}")
    return miou

print("\n── Per-class IoU on val set ──")
miou = compute_miou_full(model, val_dataset)

# -----------------------------
# Visualize N samples
# -----------------------------
print(f"\n── Saving {NUM_SAMPLES} prediction visualizations to '{OUTPUT_DIR}/' ──")

indices = np.linspace(0, len(val_dataset) - 1, NUM_SAMPLES, dtype=int)

for i, idx in enumerate(indices):
    image_tensor, mask_tensor = val_dataset[idx]

    # Run inference
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0).to(device))['out']   # (1, C, H, W)
    pred = output.argmax(dim=1).squeeze(0).cpu().numpy()               # (H, W)

    # Convert to numpy for display
    image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    gt_np    = class_to_color(mask_tensor.numpy())
    pred_np  = class_to_color(pred)

    # ── Plot ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Sample {idx}  |  mIoU (full val): {miou:.4f}", fontsize=13)

    axes[0].imshow(image_np);  axes[0].set_title("Input Image");     axes[0].axis("off")
    axes[1].imshow(gt_np);     axes[1].set_title("Ground Truth");    axes[1].axis("off")
    axes[2].imshow(pred_np);   axes[2].set_title("Prediction");      axes[2].axis("off")

    # ── Color legend ──────────────────────────────────────
    patches = [
        mpatches.Patch(color=np.array(CAMVID_COLORMAP[c]) / 255.0, label=CLASS_NAMES[c])
        for c in range(len(CAMVID_COLORMAP))
    ]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=7,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08)
    )

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"pred_{i:03d}_idx{idx}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

print("\nDone.")