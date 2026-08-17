"""
Swin Transformer V2 model definition for fine-tuning.

Loaded dynamically by `train.py` / `infer.py` via:
  --model-def "c:/path/to/model_def_swin.py"

Architecture:
  - timm `swinv2_base_window8_256.ms_in1k` backbone
  - 3-channel input
  - `img_size` fixed to (target_h, target_w) for positional embeddings
  - gradient checkpointing enabled (large images at this size)
  - sigmoid head to constrain coordinates to [0, 1]
"""

from __future__ import annotations

from typing import Tuple

import timm
import torch
import torch.nn as nn


class CoordRegressor(nn.Module):
    """Wraps a base classifier and applies sigmoid for [0, 1] coordinate outputs."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        return torch.sigmoid(out)


def create_model(
    *,
    num_outputs: int,
    in_chans: int = 3,
    pretrained: bool = True,
    model_name: str = "swinv2_base_window8_256.ms_in1k",
    img_size: Tuple[int, int] = (1024, 512),
    grad_checkpointing: bool = True,
    **_unused_kwargs,
) -> nn.Module:
    """
    Build a Swin V2 backbone configured for coordinate regression.

    - **num_outputs**: number of regression outputs (e.g. num_landmarks * 2)
    - **in_chans**: input channels (Swin defaults to 3; grayscale is stacked
      to 3 channels upstream in the dataset)
    - **img_size**: (H, W) — must match the input tensor size after transforms
    - **grad_checkpointing**: enable activation checkpointing in the backbone
      to reduce VRAM at the cost of speed
    """
    base_model = timm.create_model(
        model_name,
        pretrained=pretrained,
        in_chans=in_chans,
        num_classes=num_outputs,
        img_size=tuple(img_size),
    )
    if grad_checkpointing and hasattr(base_model, "set_grad_checkpointing"):
        base_model.set_grad_checkpointing(True)
    return CoordRegressor(base_model)
