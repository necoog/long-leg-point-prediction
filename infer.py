#!/usr/bin/env python3
"""
Run landmark inference and write results to an output folder.

Examples:
  python infer.py --model weights/best_model.pth --input data/images/55461.png --output outputs/
  python infer.py --model weights/best_model.pth --input data/images --output outputs/batch
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trainer.inference import run_inference


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict landmarks with Swin V2 and save txt/json/overlay files."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to checkpoint (.pth / .pt).",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Single image path or a folder of images.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where prediction files are written.",
    )
    parser.add_argument(
        "--model-def",
        default=None,
        help="Optional path to create_model(...) .py (default: model_def_swin.py).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Force device, e.g. "cpu" or "cuda" (default: auto).',
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Do not write *_overlay.png.",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Do not write *_points.json.",
    )
    parser.add_argument(
        "--no-txt",
        action="store_true",
        help="Do not write *_points.txt.",
    )
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    results = run_inference(
        model_path=model_path,
        input_path=args.input,
        output_dir=args.output,
        model_def=args.model_def,
        device=args.device,
        save_overlay=not args.no_overlay,
        save_json=not args.no_json,
        save_txt=not args.no_txt,
    )
    print(f"[infer] Done. Predicted {len(results)} image(s) → {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
