from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Tuple

import cv2

from .constants import LANDMARK_NAMES, LANDMARK_NAME_MAPPING, BASE_TO_RIGHT, BASE_TO_LEFT


def _load_points_with_names(json_path: Path) -> Dict[str, Dict[str, float]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    pts: Dict[str, Dict[str, float]] = {}
    for name, v in data.get("drawingProps", {}).items():
        if "x" in v and "y" in v:
            pts[name] = {"x": float(v["x"]), "y": float(v["y"])}
    return pts


def _rescale_points(
    points_dict: Dict[str, Dict[str, float]],
    shape_1024: Tuple[int, int, int],
    shape_orig: Tuple[int, int, int],
) -> Dict[str, Dict[str, float]]:
    h1024, w1024 = shape_1024[:2]
    horig, worig = shape_orig[:2]

    sx = worig / w1024
    sy = horig / h1024

    new_pts: Dict[str, Dict[str, float]] = {}
    for name, p in points_dict.items():
        new_pts[name] = {
            "x": p["x"] * sx,
            "y": p["y"] * sy,
        }
    return new_pts


def process_unzipped_if_present(
    unzipped_dir: Path,
    output_image_dir: Path,
    output_label_dir: Path,
) -> None:
    """
    Mirrors the logic from nonprocessedtoprocessed.py:
      - expects unzipped_dir/Images and unzipped_dir/JsonVariables
      - for each case, finds original PNG + *-1024.jpg and associated object_*.json
      - rescales points back to original resolution
      - writes:
          <output_image_dir>/<case_id>.png
          <output_label_dir>/<case_id>_points.txt
        with lines: "<name>,x,y" in LANDMARK_NAMES order

    If unzipped_dir or its expected subfolders are missing/empty, this is a no-op.
    """
    unzipped_dir = unzipped_dir.resolve()
    if not unzipped_dir.exists():
        return

    images_root = unzipped_dir / "Images"
    json_root = unzipped_dir / "JsonVariables"
    if not images_root.is_dir() or not json_root.is_dir():
        # Nothing to do if structure is not present
        return

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    any_cases = False

    for case_id in os.listdir(images_root):
        img_case_dir = images_root / case_id
        json_case_dir = json_root / case_id

        if not img_case_dir.is_dir():
            continue

        any_cases = True
        print(f"[preprocess] Processing case: {case_id}")

        # ---- find images ----
        original_png: Path | None = None
        img_1024: Path | None = None

        for f in img_case_dir.glob("*"):
            if f.suffix.lower() == ".png":
                original_png = f
            elif f.name.endswith("-1024.jpg"):
                img_1024 = f

        if original_png is None or img_1024 is None:
            print(f"[preprocess][WARN] Missing images for {case_id}, skipping.")
            continue

        # ---- read shapes ----
        img1024 = cv2.imread(str(img_1024))
        orig = cv2.imread(str(original_png))
        if img1024 is None or orig is None:
            print(f"[preprocess][WARN] Failed to read images for {case_id}, skipping.")
            continue

        shape_1024 = img1024.shape
        shape_orig = orig.shape

        # ---- load label files ----
        if not json_case_dir.is_dir():
            print(f"[preprocess][WARN] No JsonVariables dir for {case_id}, skipping.")
            continue

        label_files = list(json_case_dir.glob("object_*.json"))
        if len(label_files) == 0:
            print(f"[preprocess][WARN] No labels for {case_id}, skipping.")
            continue

        # Sort by object ID: smaller ID = right leg (suffix 0), larger ID = left leg (suffix 1)
        def _object_id(p: Path) -> int:
            stem = p.stem  # e.g. "object_123"
            try:
                return int(stem.split("_")[-1])
            except (IndexError, ValueError):
                return 0

        label_files_sorted = sorted(label_files, key=_object_id)
        all_rescaled_points: Dict[str, Dict[str, float]] = {}
        for idx, lf in enumerate(label_files_sorted):
            pts = _load_points_with_names(lf)
            pts_rescaled = _rescale_points(pts, shape_1024, shape_orig)
            # First file (smaller ID) = right leg -> 0 suffix; second = left leg -> 1 suffix
            leg_map = BASE_TO_RIGHT if idx == 0 else BASE_TO_LEFT
            for json_key, coords in pts_rescaled.items():
                canonical = leg_map.get(json_key, LANDMARK_NAME_MAPPING.get(json_key, json_key))
                if canonical in LANDMARK_NAMES:
                    all_rescaled_points[canonical] = coords

        mapped_pts: Dict[str, Dict[str, float]] = all_rescaled_points

        # Require at least one valid landmark so we don't write empty cases
        if not mapped_pts:
            print(
                f"[preprocess][WARN] No landmarks mapped for {case_id} (JSON keys: {list(all_rescaled_points.keys())}). "
                "Check LANDMARK_NAME_MAPPING in trainer.constants. Skipping."
            )
            continue

        missing = [n for n in LANDMARK_NAMES if n not in mapped_pts]
        if missing:
            print(
                f"[preprocess] Case {case_id}: {len(mapped_pts)}/{len(LANDMARK_NAMES)} landmarks "
                f"(missing: {missing}); writing -1,-1 for missing."
            )

        # ================== SAVE ORIGINAL PNG ==================
        out_img_path = output_image_dir / f"{case_id}.png"
        shutil.copy(original_png, out_img_path)

        # ================== CREATE TXT ANNOTATION ==================
        txt_lines = []
        for name in LANDMARK_NAMES:
            if name in mapped_pts:
                p = mapped_pts[name]
                line = f"{name},{p['x']:.2f},{p['y']:.2f}"
            else:
                line = f"{name},-1,-1"
            txt_lines.append(line)

        out_txt_path = output_label_dir / f"{case_id}_points.txt"
        with out_txt_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines))

    if any_cases:
        print("[preprocess] DONE. Filled processed images/labels from unzipped data.")

