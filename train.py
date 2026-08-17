#!/usr/bin/env python3
"""
Re-train (fine-tune) the Swin landmark model from the CLI.

Examples:
  python train.py --model weights/best_model.pth --data my_data.zip --output weights/run1
  python train.py --model weights/best_model.pth --data data --epochs 20
  python train.py --model weights/best_model.pth --data data/unzipped --output weights/run2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trainer.arch_presets import DEFAULT_ARCH_KEY, get_preset
from trainer.data_resolve import resolve_training_data
from trainer.finetune import FineTuneConfig, run_finetune
from trainer.paths import ProjectPaths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune Swin V2 landmark model. Pass --model and --data."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to checkpoint (.pth / .pt) to fine-tune from.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help=(
            "Training data: .zip (Images/+JsonVariables/), raw unzipped folder, "
            "or processed folder with images/ + labels/."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Directory for checkpoints (default: <project>/weights/train_run).",
    )
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parent),
        help="Project root (default: this repo).",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-ext",
        default=".png",
        help='Image extension filter for processed images (default: ".png").',
    )
    parser.add_argument(
        "--model-def",
        default=None,
        help="Optional path to create_model(...) .py (default: model_def_swin.py).",
    )
    parser.add_argument(
        "--non-strict-load",
        action="store_true",
        help="Load checkpoint with strict=False.",
    )
    parser.add_argument(
        "--resume-optimizer",
        action="store_true",
        help="Also restore optimizer/scheduler state from checkpoint if present.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    paths = ProjectPaths.from_root(project_root)
    paths.ensure_exists()

    model_path = Path(args.model).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    resolved = resolve_training_data(
        args.data,
        processed_image_dir=paths.images_dir,
        processed_label_dir=paths.labels_dir,
        extract_dir=paths.unzipped_dir,
    )

    output_dir = Path(args.output) if args.output else (paths.weights_dir / "train_run")
    output_dir = output_dir.resolve()

    preset = get_preset(DEFAULT_ARCH_KEY)
    model_def = (
        Path(args.model_def).resolve()
        if args.model_def
        else (project_root / preset.model_def)
    )

    cfg = FineTuneConfig(
        image_folder=str(resolved.image_folder),
        label_folder=str(resolved.label_folder),
        image_ext=args.image_ext,
        val_split=args.val_split,
        model_def=str(model_def),
        in_chans=preset.in_chans,
        target_h=preset.target_h,
        target_w=preset.target_w,
        transform_preset=preset.transform_preset,
        model_kwargs=dict(preset.model_kwargs),
        pass_img_size_to_model=preset.pass_img_size_to_model,
        checkpoint=str(model_path),
        strict_load=not args.non_strict_load,
        pretrained_backbone=True,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        seed=args.seed,
        resume_optimizer=args.resume_optimizer,
        output_dir=str(output_dir),
        best_model_name="best_model.pth",
    )

    print(f"[train] model      = {model_path}")
    print(f"[train] images     = {resolved.image_folder}")
    print(f"[train] labels     = {resolved.label_folder}")
    print(f"[train] output     = {output_dir}")
    print(f"[train] epochs={args.epochs} batch={args.batch_size} lr={args.lr}")

    run_finetune(cfg)
    print(f"[train] Done. Best weights: {output_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
