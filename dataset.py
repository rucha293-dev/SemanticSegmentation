import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

CAMVID_COLORMAP = [
    (128, 128, 128),  # 0  Sky
    (128,   0,   0),  # 1  Building
    (192, 192, 128),  # 2  Pole
    (255,  69,   0),  # 3  Road Marking
    (128,  64, 128),  # 4  Road
    ( 60,  40, 222),  # 5  Pavement
    (128, 128,   0),  # 6  Tree
    (192, 128, 128),  # 7  Sign Symbol
    ( 64,  64, 128),  # 8  Fence
    ( 64,   0, 128),  # 9  Car
    ( 64,  64,   0),  # 10 Pedestrian
    (  0, 128, 192),  # 11 Bicyclist
    (  0,   0,   0),  # 12 Unlabelled (void)
]

NUM_CLASSES = len(CAMVID_COLORMAP)
VOID_CLASS  = 12

def encode_mask(mask_rgb):
    h, w      = mask_rgb.shape[:2]
    class_map = np.full((h, w), VOID_CLASS, dtype=np.int64)
    for class_idx, color in enumerate(CAMVID_COLORMAP):
        match = np.all(mask_rgb == color, axis=-1)
        class_map[match] = class_idx
    return class_map

def augment(image, mask_rgb):
    if random.random() > 0.5:
        image    = cv2.flip(image,    1)
        mask_rgb = cv2.flip(mask_rgb, 1)
    if random.random() > 0.5:
        h, w     = image.shape[:2]
        scale    = random.uniform(0.75, 1.0)
        new_h, new_w = int(h * scale), int(w * scale)
        top      = random.randint(0, h - new_h)
        left     = random.randint(0, w - new_w)
        image    = image   [top:top+new_h, left:left+new_w]
        mask_rgb = mask_rgb[top:top+new_h, left:left+new_w]
        image    = cv2.resize(image,    (w, h), interpolation=cv2.INTER_LINEAR)
        mask_rgb = cv2.resize(mask_rgb, (w, h), interpolation=cv2.INTER_NEAREST)
    if random.random() > 0.5:
        beta  = random.uniform(-30, 30)
        image = np.clip(image.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    if random.random() > 0.5:
        alpha = random.uniform(0.75, 1.25)
        image = np.clip(image.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    return image, mask_rgb

class CamVidDataset(Dataset):

    def __init__(self, image_dir, mask_dir, size=(512, 512), training=False):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.size      = size
        self.training  = training
        self.images    = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith('.png') or f.endswith('.jpg')
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.image_dir, image_name)
        mask_name  = image_name.replace(".png", "_L.png")
        mask_path  = os.path.join(self.mask_dir, mask_name)
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_dir, image_name)
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"No mask found for {image_name} in {self.mask_dir}")
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        mask_rgb = cv2.imread(mask_path)
        if mask_rgb is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.resize(mask_rgb, self.size, interpolation=cv2.INTER_NEAREST)
        if self.training:
            image, mask_rgb = augment(image, mask_rgb)
        mask  = encode_mask(mask_rgb)
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        mask  = torch.tensor(mask).long()
        return image, mask
