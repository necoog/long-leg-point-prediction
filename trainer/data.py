from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .constants import NUM_LANDMARKS, LANDMARK_NAMES


def read_txt_landmarks(label_path: str | os.PathLike[str]) -> List[float] | None:
    """
    Robust parsing (commas/tabs, empty lines):
      - returns [x1,y1,x2,y2,...] strictly in LANDMARK_NAMES order
      - missing names are filled with -1,-1 (masked in training)
      - unknown tags are ignored with a warning
      - returns None only when a line is malformed
    """
    pts: dict[str, Tuple[float, float]] = {}
    with open(label_path, "r", encoding="utf-8") as f:
        for ln_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = [
                p.strip()
                for p in line.replace("\t", ",").split(",")
                if p.strip() != ""
            ]
            if len(parts) != 3:
                print(f"[WARN] Bad line in {label_path}:{ln_no}: '{line}'")
                return None
            tag, sx, sy = parts
            try:
                x, y = float(sx), float(sy)
            except Exception:
                print(f"[WARN] Non-numeric coord in {label_path}:{ln_no}: '{line}'")
                return None
            pts[tag] = (x, y)

    extra = [n for n in pts.keys() if n not in LANDMARK_NAMES]
    if extra:
        print(f"[WARN] Ignoring unknown tags in {label_path}: {extra}")

    missing = [n for n in LANDMARK_NAMES if n not in pts]
    if missing:
        print(
            f"[WARN] Missing landmarks in {label_path} "
            f"({len(missing)}/{NUM_LANDMARKS}); using -1,-1 (masked): {missing}"
        )

    coords: List[float] = []
    for name in LANDMARK_NAMES:
        if name in pts:
            coords.extend(pts[name])
        else:
            coords.extend([-1.0, -1.0])
    return coords


def build_dataset_lists(
    image_folder: str | os.PathLike[str],
    label_folder: str | os.PathLike[str],
    *,
    image_ext: str = ".png",
) -> Tuple[List[str], List[List[float]]]:
    """
    Build paired image/label lists.
      - image: <base>.<ext>
      - label: <base>_points.txt
      - warns on missing labels / bad coords
      - skips invalid pairs
    """
    image_folder = str(image_folder)
    label_folder = str(label_folder)

    image_paths: List[str] = []
    keypoints: List[List[float]] = []

    for img_name in sorted(os.listdir(image_folder)):
        if not img_name.lower().endswith(image_ext.lower()):
            continue

        base = os.path.splitext(img_name)[0]
        label_path = os.path.join(label_folder, f"{base}_points.txt")
        if not os.path.exists(label_path):
            print(f"[WARN] Missing label for {img_name}")
            continue

        kp = read_txt_landmarks(label_path)
        if kp is None:
            continue
        if len(kp) != NUM_LANDMARKS * 2:
            print(f"[WARN] Wrong number of coords for {img_name}")
            continue

        image_paths.append(os.path.join(image_folder, img_name))
        keypoints.append(kp)

    return image_paths, keypoints


# Default canvas for Swin V2 (fixed H×W for positional embeddings)
TARGET_H = 1024
TARGET_W = 512


def _make_swin_transforms(
    target_h: int, target_w: int, in_chans: int
) -> Tuple[A.Compose, A.Compose]:
    """Resize + ShiftScaleRotate pipeline (mirrors swin.py)."""
    mean = (0.5,) * in_chans
    std = (0.5,) * in_chans
    train_transform = A.Compose(
        [
            A.Resize(target_h, target_w),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.18,
                rotate_limit=5,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.8,
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    val_transform = A.Compose(
        [
            A.Resize(target_h, target_w),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    return train_transform, val_transform


def make_transforms(
    *,
    transform_preset: str = "swin",
    target_h: int = TARGET_H,
    target_w: int = TARGET_W,
    in_chans: int = 3,
) -> Tuple[A.Compose, A.Compose]:
    """
    Build (train, val) Albumentations pipelines.

    Currently only ``"swin"`` is supported: Resize + ShiftScaleRotate
    (stretches to fixed H×W). Required because Swin V2 needs a fixed input
    size tied to its positional embeddings.

    Normalization mean/std is replicated to ``in_chans`` channels.
    """
    if transform_preset == "swin":
        return _make_swin_transforms(target_h, target_w, in_chans)
    raise ValueError(
        f"Unknown transform_preset: '{transform_preset}'. "
        "Expected: 'swin'."
    )


class XrayLandmarkDataset(Dataset):
    """
    Returns:
      - image:  (C, H, W) tensor where C == ``in_chans``
      - target: (L*2,) normalized to [0,1]
      - mask:   (L,) with 1 for valid, 0 for invalid/missing

    Source images are always read as grayscale from disk; when ``in_chans == 3``
    the single channel is stacked into 3 channels (mirrors ``swin.py``).
    """

    def __init__(
        self,
        image_paths: Sequence[str],
        keypoints: Sequence[Sequence[float]],
        *,
        transform: A.Compose | None = None,
        in_chans: int = 3,
    ):
        if in_chans not in (1, 3):
            raise ValueError(f"in_chans must be 1 or 3, got {in_chans}")
        self.image_paths = list(image_paths)
        self.keypoints = [list(k) for k in keypoints]
        self.transform = transform
        self.in_chans = in_chans

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {self.image_paths[idx]}")
        if self.in_chans == 3:
            image = np.stack([image, image, image], axis=-1)  # (H, W, 3)
        else:
            image = np.expand_dims(image, axis=-1)  # (H, W, 1)

        # Build keypoints as list of (x, y)
        kp_flat = self.keypoints[idx]
        if len(kp_flat) != NUM_LANDMARKS * 2:
            raise ValueError(
                f"Bad keypoints length for idx={idx}. "
                f"Expected {NUM_LANDMARKS*2}, got {len(kp_flat)}."
            )

        keypoints = [(kp_flat[i], kp_flat[i + 1]) for i in range(0, len(kp_flat), 2)]

        # Albumentations aug
        if self.transform:
            augmented = self.transform(image=image, keypoints=keypoints)
            image = augmented["image"]  # Tensor (C, H, W)
            keypoints = augmented["keypoints"]  # list[(x,y), ...]

        # Compute normalization and mask
        _, h, w = image.shape  # C,H,W
        targets: List[float] = []
        mask: List[float] = []

        for (x, y) in keypoints:
            valid = (
                (x is not None)
                and (y is not None)
                and (0 <= x < w)
                and (0 <= y < h)
            )
            if valid:
                tx = float(np.clip(x / w, 0.0, 1.0))
                ty = float(np.clip(y / h, 0.0, 1.0))
                mask.append(1.0)
            else:
                tx, ty = 0.0, 0.0
                mask.append(0.0)
            targets.extend([tx, ty])

        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        return image, targets_tensor, mask_tensor
