"""
Model architecture presets.

Each preset bundles together the choices needed for an end-to-end fine-tuning
run: the model-definition file, the number of input channels, the target image
size, the transform/augmentation pipeline, and any extra kwargs passed to
`create_model(...)`.

Both CLIs (`train.py`, `infer.py`) consume this so defaults stay in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class ArchPreset:
    """A fine-tuning preset tied to a model architecture."""

    key: str
    display_name: str
    model_def: str  # relative path to a *.py file exposing create_model(...)
    in_chans: int
    target_h: int
    target_w: int
    transform_preset: str  # see trainer.data.make_transforms
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    # When True, finetune injects img_size=(target_h, target_w) into model_kwargs.
    # Required for Swin which builds a positional embedding tied to input size.
    pass_img_size_to_model: bool = False


ARCH_PRESETS: Dict[str, ArchPreset] = {
    "swin": ArchPreset(
        key="swin",
        display_name="Swin Transformer V2 (3ch, 1024x512)",
        model_def="model_def_swin.py",
        in_chans=3,
        target_h=1024,
        target_w=512,
        transform_preset="swin",
        model_kwargs={
            "model_name": "swinv2_base_window8_256.ms_in1k",
            "grad_checkpointing": True,
        },
        pass_img_size_to_model=True,
    ),
}


DEFAULT_ARCH_KEY = "swin"


def get_preset(key: str) -> ArchPreset:
    """Look up a preset by its key. Raises ValueError with the valid set."""
    if key not in ARCH_PRESETS:
        valid = ", ".join(sorted(ARCH_PRESETS.keys()))
        raise ValueError(
            f"Unknown model architecture: '{key}'. "
            f"Valid options: {valid}"
        )
    return ARCH_PRESETS[key]


def list_presets() -> Tuple[ArchPreset, ...]:
    """All known presets, in declaration order."""
    return tuple(ARCH_PRESETS.values())
