"""
Swin landmark inference: load checkpoint, predict, save outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from .arch_presets import DEFAULT_ARCH_KEY, get_preset
from .checkpointing import load_weights_into_model
from .constants import LANDMARK_NAMES, NUM_LANDMARKS
from .model_loader import load_model_from_file


@dataclass(frozen=True)
class PredictionResult:
    image_path: Path
    orig_size: Tuple[int, int]  # (H, W)
    model_size: Tuple[int, int]  # (H, W)
    # name -> (x, y) in original image pixels
    points_orig: Dict[str, Tuple[float, float]]
    # name -> (x, y) on model canvas
    points_model: Dict[str, Tuple[float, float]]
    # name -> (x_norm, y_norm) in [0, 1]
    points_norm: Dict[str, Tuple[float, float]]


def _default_model_def() -> Path:
    return Path(__file__).resolve().parent.parent / "model_def_swin.py"


def build_inference_model(
    *,
    checkpoint_path: str | Path,
    model_def: str | Path | None = None,
    device: torch.device | None = None,
    strict: bool = True,
) -> Tuple[torch.nn.Module, torch.device]:
    preset = get_preset(DEFAULT_ARCH_KEY)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_def = Path(model_def) if model_def else _default_model_def()

    model = load_model_from_file(
        model_def,
        num_outputs=NUM_LANDMARKS * 2,
        in_chans=preset.in_chans,
        pretrained=False,
        extra_kwargs={
            **dict(preset.model_kwargs),
            "img_size": (preset.target_h, preset.target_w),
            "grad_checkpointing": False,
        },
    ).to(device)

    load_weights_into_model(
        model=model,
        checkpoint_path=str(checkpoint_path),
        strict=strict,
    )
    model.eval()
    return model, device


def _val_transform(target_h: int, target_w: int, in_chans: int) -> A.Compose:
    mean = (0.5,) * in_chans
    std = (0.5,) * in_chans
    return A.Compose(
        [
            A.Resize(target_h, target_w),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def predict_image(
    model: torch.nn.Module,
    image_path: str | Path,
    *,
    device: torch.device,
    target_h: int | None = None,
    target_w: int | None = None,
    in_chans: int | None = None,
) -> PredictionResult:
    preset = get_preset(DEFAULT_ARCH_KEY)
    target_h = target_h or preset.target_h
    target_w = target_w or preset.target_w
    in_chans = in_chans or preset.in_chans

    image_path = Path(image_path)
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    orig_h, orig_w = gray.shape[:2]

    if in_chans == 3:
        image = np.stack([gray, gray, gray], axis=-1)
    else:
        image = np.expand_dims(gray, axis=-1)

    tensor = _val_transform(target_h, target_w, in_chans)(image=image)["image"]
    batch = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(batch).cpu().numpy()[0]

    points_norm: Dict[str, Tuple[float, float]] = {}
    points_model: Dict[str, Tuple[float, float]] = {}
    points_orig: Dict[str, Tuple[float, float]] = {}
    for i, name in enumerate(LANDMARK_NAMES):
        xn, yn = float(preds[2 * i]), float(preds[2 * i + 1])
        points_norm[name] = (xn, yn)
        points_model[name] = (xn * target_w, yn * target_h)
        points_orig[name] = (xn * orig_w, yn * orig_h)

    return PredictionResult(
        image_path=image_path,
        orig_size=(orig_h, orig_w),
        model_size=(target_h, target_w),
        points_orig=points_orig,
        points_model=points_model,
        points_norm=points_norm,
    )


def _draw_overlay(
    gray: np.ndarray,
    points_xy: Dict[str, Tuple[float, float]],
    *,
    canvas_hw: Tuple[int, int] | None = None,
) -> np.ndarray:
    """Draw landmarks on grayscale (optionally resized) → BGR uint8."""
    if canvas_hw is not None:
        h, w = canvas_hw
        vis_gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
    else:
        vis_gray = gray
    vis = cv2.cvtColor(vis_gray, cv2.COLOR_GRAY2BGR)
    for name, (x, y) in points_xy.items():
        xi, yi = int(round(x)), int(round(y))
        cv2.circle(vis, (xi, yi), 6, (0, 255, 0), -1)
        cv2.putText(
            vis,
            name,
            (xi + 6, max(12, yi - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 80, 255),
            1,
            cv2.LINE_AA,
        )
    return vis


def save_prediction(
    result: PredictionResult,
    output_dir: str | Path,
    *,
    save_overlay: bool = True,
    save_json: bool = True,
    save_txt: bool = True,
) -> Dict[str, Path]:
    """
    Write prediction artifacts under ``output_dir``:

      <stem>_points.txt   — name,x,y in original pixels
      <stem>_points.json  — structured coords (orig / model / norm)
      <stem>_overlay.png  — landmarks drawn on resized model canvas
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = result.image_path.stem
    written: Dict[str, Path] = {}

    if save_txt:
        txt_path = output_dir / f"{stem}_points.txt"
        lines = [
            f"{name},{result.points_orig[name][0]:.2f},{result.points_orig[name][1]:.2f}"
            for name in LANDMARK_NAMES
        ]
        txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written["txt"] = txt_path

    if save_json:
        json_path = output_dir / f"{stem}_points.json"
        payload = {
            "image": str(result.image_path),
            "orig_size_hw": list(result.orig_size),
            "model_size_hw": list(result.model_size),
            "landmarks": {
                name: {
                    "orig_xy": list(result.points_orig[name]),
                    "model_xy": list(result.points_model[name]),
                    "norm_xy": list(result.points_norm[name]),
                }
                for name in LANDMARK_NAMES
            },
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written["json"] = json_path

    if save_overlay:
        gray = cv2.imread(str(result.image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Failed to read image for overlay: {result.image_path}")
        overlay = _draw_overlay(
            gray,
            result.points_model,
            canvas_hw=result.model_size,
        )
        overlay_path = output_dir / f"{stem}_overlay.png"
        cv2.imwrite(str(overlay_path), overlay)
        written["overlay"] = overlay_path

    return written


def collect_images(
    input_path: str | Path,
    *,
    image_exts: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
) -> List[Path]:
    input_path = Path(input_path).resolve()
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input not found: {input_path}")
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in image_exts}
    files = sorted(
        p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in exts
    )
    if not files:
        raise RuntimeError(f"No images with extensions {sorted(exts)} in {input_path}")
    return files


def run_inference(
    *,
    model_path: str | Path,
    input_path: str | Path,
    output_dir: str | Path,
    model_def: str | Path | None = None,
    device: str | None = None,
    save_overlay: bool = True,
    save_json: bool = True,
    save_txt: bool = True,
) -> List[PredictionResult]:
    """Batch-predict one image or a folder; save artifacts to ``output_dir``."""
    torch_device = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, torch_device = build_inference_model(
        checkpoint_path=model_path,
        model_def=model_def,
        device=torch_device,
    )
    images = collect_images(input_path)
    output_dir = Path(output_dir)
    results: List[PredictionResult] = []

    print(f"[infer] device={torch_device}  images={len(images)}  out={output_dir}")
    for img in images:
        print(f"[infer] {img.name}")
        result = predict_image(model, img, device=torch_device)
        written = save_prediction(
            result,
            output_dir,
            save_overlay=save_overlay,
            save_json=save_json,
            save_txt=save_txt,
        )
        for kind, path in written.items():
            print(f"  -> {kind}: {path}")
        results.append(result)
    return results
