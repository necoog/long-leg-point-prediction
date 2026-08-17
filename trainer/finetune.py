from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .checkpointing import load_weights_into_model
from .constants import LANDMARK_NAMES, NUM_LANDMARKS
from .data import XrayLandmarkDataset, build_dataset_lists, make_transforms
from .losses import MultifacetedLoss
from .model_loader import load_model_from_file


@dataclass
class FineTuneConfig:
    # Data
    image_folder: str = ""
    label_folder: str = ""
    image_ext: str = ".png"
    val_split: float = 0.15

    # Model
    model_def: str = ""
    checkpoint: str | None = None
    strict_load: bool = True
    pretrained_backbone: bool = True

    # Architecture-dependent inputs (driven by trainer.arch_presets)
    in_chans: int = 3
    target_h: int = 1024
    target_w: int = 512
    transform_preset: str = "swin"
    # Extra kwargs forwarded to create_model(...) in the model-def file
    model_kwargs: Optional[Dict[str, Any]] = None
    # When True, finetune injects ``img_size=(target_h, target_w)`` into model_kwargs
    pass_img_size_to_model: bool = True

    # Train
    epochs: int = 20
    batch_size: int = 1
    lr: float = 5e-5
    num_workers: int = 2
    seed: int = 42
    pin_memory: bool = True
    resume_optimizer: bool = False
    freeze_backbone_epochs: int = 0
    weight_decay: float = 0.05
    max_grad_norm: float = 1.0

    # Output
    output_dir: str = "weights/finetune_run"
    best_model_name: str = "best_model.pth"

    # Optional: called after each epoch for UI/progress (epoch_1based, total_epochs, train_loss, val_loss, lr)
    progress_callback: Optional[Callable[[int, int, float, Optional[float], float], None]] = field(default=None, repr=False)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _set_backbone_trainable(model: torch.nn.Module, trainable: bool) -> None:
    """
    Freeze / unfreeze backbone:
      - if model has `base`, only its non-head params follow `trainable`
      - head is always trainable
    Falls back to toggling all parameters if no `base` attribute.
    """
    if hasattr(model, "base"):
        for n, p in model.base.named_parameters():
            if n.startswith("head"):
                p.requires_grad = True
            else:
                p.requires_grad = trainable
    else:
        for p in model.parameters():
            p.requires_grad = trainable


def run_finetune(cfg: FineTuneConfig) -> None:
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Data prep
    # -------------------------
    image_paths, keypoints = build_dataset_lists(
        cfg.image_folder, cfg.label_folder, image_ext=cfg.image_ext
    )
    if len(image_paths) == 0:
        raise RuntimeError(
            "No valid (image,label) pairs found. Expected images and "
            'labels named "<base>_points.txt" with exactly '
            f"{NUM_LANDMARKS} points."
        )

    if cfg.val_split > 0:
        train_image_paths, val_image_paths, train_keypoints, val_keypoints = (
            train_test_split(
                image_paths,
                keypoints,
                test_size=cfg.val_split,
                random_state=cfg.seed,
            )
        )
    else:
        train_image_paths, train_keypoints = image_paths, keypoints
        val_image_paths, val_keypoints = [], []

    train_transform, val_transform = make_transforms(
        transform_preset=cfg.transform_preset,
        target_h=cfg.target_h,
        target_w=cfg.target_w,
        in_chans=cfg.in_chans,
    )

    train_dataset = XrayLandmarkDataset(
        train_image_paths,
        train_keypoints,
        transform=train_transform,
        in_chans=cfg.in_chans,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and (device.type == "cuda"),
    )

    val_loader: Optional[DataLoader] = None
    if len(val_image_paths) > 0:
        val_dataset = XrayLandmarkDataset(
            val_image_paths,
            val_keypoints,
            transform=val_transform,
            in_chans=cfg.in_chans,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and (device.type == "cuda"),
        )

    # -------------------------
    # Model (loaded from a file path)
    # -------------------------
    num_outputs = NUM_LANDMARKS * 2
    model_extra_kwargs: Dict[str, Any] = dict(cfg.model_kwargs or {})
    if cfg.pass_img_size_to_model:
        # Swin needs img_size to match the actual tensor size after transforms.
        model_extra_kwargs.setdefault("img_size", (cfg.target_h, cfg.target_w))
    model = load_model_from_file(
        cfg.model_def,
        num_outputs=num_outputs,
        in_chans=cfg.in_chans,
        pretrained=cfg.pretrained_backbone,
        extra_kwargs=model_extra_kwargs,
    ).to(device)

    # -------------------------
    # Optimizer + scheduler
    # -------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        threshold=1e-4,
        min_lr=1e-8,
    )

    # -------------------------
    # Load checkpoint weights (fine-tune start)
    # -------------------------
    start_epoch = 0
    best_val = float("inf")
    if cfg.checkpoint:
        ckpt, missing, unexpected = load_weights_into_model(
            model=model,
            checkpoint_path=cfg.checkpoint,
            strict=cfg.strict_load,
        )
        if cfg.strict_load and (missing or unexpected):
            raise RuntimeError(
                f"Strict checkpoint load failed. Missing={missing}, Unexpected={unexpected}"
            )

        if isinstance(ckpt, dict):
            if "epoch" in ckpt and isinstance(ckpt["epoch"], int):
                start_epoch = ckpt["epoch"] + 1
            if "best_val_loss" in ckpt:
                try:
                    best_val = float(ckpt["best_val_loss"])
                except Exception:
                    pass

            if cfg.resume_optimizer:
                if "optimizer_state_dict" in ckpt and isinstance(
                    ckpt["optimizer_state_dict"], dict
                ):
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if "scheduler_state_dict" in ckpt and isinstance(
                    ckpt["scheduler_state_dict"], dict
                ):
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    # -------------------------
    # Loss
    # -------------------------
    criterion = MultifacetedLoss().to(device)

    # -------------------------
    # Output
    # -------------------------
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = out_dir / cfg.best_model_name

    # -------------------------
    # Fine-tuning loop
    # -------------------------
    num_epochs = cfg.epochs
    freeze_backbone_epochs = int(cfg.freeze_backbone_epochs)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Optional backbone freezing schedule
        if freeze_backbone_epochs > 0:
            if epoch < freeze_backbone_epochs:
                _set_backbone_trainable(model, False)
            elif epoch == freeze_backbone_epochs:
                _set_backbone_trainable(model, True)

        # ========================
        # TRAIN
        # ========================
        model.train()
        train_loss = 0.0

        for images, targets, masks in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)  # (B, L*2), expected in [0,1]
            loss = criterion(outputs, targets, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= max(1, len(train_loader.dataset))

        # ========================
        # VALIDATION
        # ========================
        val_loss: Any = None
        if val_loader is not None:
            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for images, targets, masks in val_loader:
                    images = images.to(device)
                    targets = targets.to(device)
                    masks = masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, targets, masks)
                    val_total += loss.item() * images.size(0)
            val_loss = val_total / max(1, len(val_loader.dataset))
            scheduler.step(val_loss)
        else:
            scheduler.step(train_loss)

        current_metric = float(val_loss) if val_loss is not None else float(train_loss)
        is_best = current_metric < best_val
        if is_best:
            best_val = current_metric

        # Config for checkpoint: exclude non-pickleable fields (e.g. progress_callback)
        config_dict = {
            k: v for k, v in cfg.__dict__.items()
            if k != "progress_callback"
        }
        ckpt_payload = {
            "epoch": epoch,
            "best_val_loss": best_val,
            "landmark_names": LANDMARK_NAMES,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss) if val_loss is not None else None,
            "config": config_dict,
        }

        # Save last checkpoint (full state)
        torch.save(ckpt_payload, out_dir / "last.pt")

        # Save best checkpoint + plain state_dict
        if is_best:
            torch.save(ckpt_payload, out_dir / "best.pt")
            torch.save(model.state_dict(), best_model_path)

        lr_now = optimizer.param_groups[0]["lr"]
        if cfg.progress_callback:
            cfg.progress_callback(
                epoch + 1,
                start_epoch + num_epochs,
                float(train_loss),
                float(val_loss) if val_loss is not None else None,
                float(lr_now),
            )
        if val_loss is not None:
            print(
                f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | LR: {lr_now:.6f}"
            )
        else:
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | LR: {lr_now:.6f}")

