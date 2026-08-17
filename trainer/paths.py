from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """
    Standard folder layout under a project root.

    root/
      data/
        images/
        labels/
      weights/
    """

    root: Path
    data_dir: Path
    images_dir: Path
    labels_dir: Path
    weights_dir: Path
    unzipped_dir: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "ProjectPaths":
        root = Path(root).resolve()
        data_dir = root / "data"
        images_dir = data_dir / "images"
        labels_dir = data_dir / "labels"
        weights_dir = root / "weights"
        unzipped_dir = data_dir / "unzipped"
        return cls(
            root=root,
            data_dir=data_dir,
            images_dir=images_dir,
            labels_dir=labels_dir,
            weights_dir=weights_dir,
            unzipped_dir=unzipped_dir,
        )

    def ensure_exists(self) -> None:
        # Create the directories. Images/labels may be empty; training will error if no data.
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.unzipped_dir.mkdir(parents=True, exist_ok=True)

