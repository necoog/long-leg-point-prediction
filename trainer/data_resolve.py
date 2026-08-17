"""
Resolve training data from a ZIP, an unzipped raw folder, or processed images/labels.
"""

from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .preprocess_unzipped import process_unzipped_if_present


@dataclass(frozen=True)
class ResolvedData:
    """Paths to processed PNG images and matching ``*_points.txt`` labels."""

    image_folder: Path
    label_folder: Path
    work_dir: Path | None = None  # temp/extracted dir if created


def _has_processed_layout(root: Path) -> bool:
    images = root / "images"
    labels = root / "labels"
    if images.is_dir() and labels.is_dir():
        return True
    # Also accept root that *is* the parent of images+labels already split
    return False


def _has_raw_layout(root: Path) -> bool:
    return (root / "Images").is_dir() and (root / "JsonVariables").is_dir()


def resolve_training_data(
    data_path: str | Path,
    *,
    processed_image_dir: Path,
    processed_label_dir: Path,
    extract_dir: Path,
) -> ResolvedData:
    """
    Turn ``data_path`` into processed ``images/`` + ``labels/``.

    Accepted inputs:
      1. ZIP with top-level ``Images/`` + ``JsonVariables/``
      2. Folder with ``Images/`` + ``JsonVariables/`` (raw / unzipped)
      3. Folder with ``images/`` + ``labels/`` (already processed)
      4. Folder that *is* ``images`` (then sibling ``labels`` is expected)
    """
    data_path = Path(data_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    extract_dir = extract_dir.resolve()
    processed_image_dir = processed_image_dir.resolve()
    processed_label_dir = processed_label_dir.resolve()
    work_dir: Path | None = None

    # --- ZIP ---
    if data_path.is_file() and data_path.suffix.lower() == ".zip":
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"[data] Extracting ZIP → {extract_dir}")
        with zipfile.ZipFile(data_path, "r") as zf:
            zf.extractall(extract_dir)
        work_dir = extract_dir
        if not _has_raw_layout(extract_dir):
            raise ValueError(
                f"ZIP must contain top-level folders Images/ and JsonVariables/. "
                f"Got: {sorted(p.name for p in extract_dir.iterdir())}"
            )
        process_unzipped_if_present(extract_dir, processed_image_dir, processed_label_dir)
        return ResolvedData(processed_image_dir, processed_label_dir, work_dir)

    if not data_path.is_dir():
        raise ValueError(f"Data path must be a .zip or a directory: {data_path}")

    # --- Already processed: .../images + .../labels ---
    if _has_processed_layout(data_path):
        print(f"[data] Using processed layout under {data_path}")
        return ResolvedData(data_path / "images", data_path / "labels", None)

    # Parent of images named "images"
    if data_path.name.lower() == "images":
        labels = data_path.parent / "labels"
        if not labels.is_dir():
            raise ValueError(f"Expected sibling labels/ next to {data_path}")
        print(f"[data] Using image folder {data_path} and labels {labels}")
        return ResolvedData(data_path, labels, None)

    # --- Raw unzipped ---
    if _has_raw_layout(data_path):
        print(f"[data] Preprocessing raw layout from {data_path}")
        process_unzipped_if_present(data_path, processed_image_dir, processed_label_dir)
        return ResolvedData(processed_image_dir, processed_label_dir, None)

    raise ValueError(
        "Unrecognized data layout. Expected one of:\n"
        "  - .zip with Images/ + JsonVariables/\n"
        "  - folder with Images/ + JsonVariables/\n"
        "  - folder with images/ + labels/\n"
        f"Got: {data_path}"
    )
