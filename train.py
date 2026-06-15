import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CamVidDataset, NUM_CLASSES, VOID_CLASS
from model import get_model

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Datasets
# -----------------------------
train_dataset = CamVidDataset(
    "dataset/CamVid/train",
    "dataset/CamVid/train_labels",
    size=(512, 512),
    training=True           # augmentation ON
)
val_dataset = CamVidDataset(
    "dataset/CamVid/val",
    "dataset/CamVid/val_labels",
    size=(512, 512),
    training=False          # augmentation OFF
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,  drop_last=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=4, shuffle=False, drop_last=False, num_workers=2)

print(f"Train samples: {len(train_dataset)}  |  Val samples: {len(val_dataset)}")


# -----------------------------
# Model (ResNet101 backbone)
# -----------------------------
model = get_model(backbone='resnet101').to(device)

# -----------------------------
# Loss & Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss(ignore_index=VOID_CLASS)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

# -----------------------------
# PolyLR Scheduler
# -----------------------------
epochs    = 100
scheduler = torch.optim.lr_scheduler.PolynomialLR(
    optimizer, total_iters=epochs, power=0.9
)

# -----------------------------
# mIoU helper
# -----------------------------
def compute_miou(outputs, masks, num_classes=NUM_CLASSES, ignore_index=VOID_CLASS):
    preds = outputs.argmax(dim=1)
    ious  = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_c = (preds == cls)
        true_c = (masks == cls)
        intersection = (pred_c & true_c).sum().item()
        union        = (pred_c | true_c).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return sum(ious) / len(ious) if ious else 0.0
start_epoch = 0
best_miou = 0.0

checkpoint_path = "models/latest_checkpoint.pth"

if os.path.exists(checkpoint_path):

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_miou = checkpoint['best_miou']

    print(f"Resuming from epoch {start_epoch}")
    print(f"Best mIoU so far: {best_miou:.4f}")
# -----------------------------
# Training loop
# -----------------------------

os.makedirs("models", exist_ok=True)

for epoch in range(start_epoch, epochs):

    # Train
    model.train()
    train_loss = 0.0

    for images, masks in train_loader:
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)['out']
        loss    = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

   
    avg_train_loss = train_loss / len(train_loader)

    # Validate
    model.eval()
    val_loss = 0.0
    val_miou = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks  = masks.to(device)

            outputs   = model(images)['out']
            val_loss += criterion(outputs, masks).item()
            val_miou += compute_miou(outputs, masks)

    avg_val_loss = val_loss / len(val_loader)
    avg_val_miou = val_miou / len(val_loader)
    current_lr   = scheduler.get_last_lr()[0]

    print(
        f"Epoch {epoch+1:>3}/{epochs}  |  "
        f"Train Loss: {avg_train_loss:.4f}  |  "
        f"Val Loss: {avg_val_loss:.4f}  |  "
        f"Val mIoU: {avg_val_miou:.4f}  |  "
        f"LR: {current_lr:.6f}"
    )
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_miou': best_miou
     }, "models/latest_checkpoint.pth")

    # Save best checkpoint
    if avg_val_miou > best_miou:
        best_miou = avg_val_miou
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_miou': best_miou
            }, "models/deeplabv3_camvid_best.pth")
        print(f"  ↑ New best mIoU: {best_miou:.4f}  — checkpoint saved.")
    scheduler.step()
# Save final weights
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_miou': best_miou
}, "models/deeplabv3_camvid_final.pth")
print("\nTraining complete.")
print(f"Best Val mIoU : {best_miou:.4f}")
print("Final model   : models/deeplabv3_camvid_final.pth")
print("Best model    : models/deeplabv3_camvid_best.pth")